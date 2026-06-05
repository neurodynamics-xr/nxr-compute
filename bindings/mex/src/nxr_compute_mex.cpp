/**
 * nxr_compute_mex.cpp — MATLAB MEX dispatcher around the nxr-compute library.
 *
 * Single mexFunction with command dispatch (gpjs / libigl-matlab
 * style). One artifact (nxr_compute.mexw64) consumed by Brainstorm, SPM,
 * or any MATLAB analysis pipeline.
 *
 * Usage from MATLAB:
 *   ops    = nxr_compute('assembleManifoldOperators', V, F);
 *   eig    = nxr_compute('solve', ops.cotanLaplacian, ops.mass, k);
 *   eig.eigenvectors = nxr_compute('normalize', eig.eigenvectors, ops.mass);
 *   eig    = nxr_compute('removeDC', eig);
 *   result = nxr_compute('precompute', V, F, k);   % shorthand for the whole pipeline
 *
 * V is Vx3 double; F is Fx3 (double / int32 / uint32) with 1-based
 * indices (MATLAB convention). All sparse outputs are CSC double.
 *
 * Cancellation: long-running commands (solve, precompute)
 * honour Ctrl-C in MATLAB via libut's utIsInterruptPending — the
 * cancellation token polls it from inside nxr-compute's solver and throws
 * Error(Cancelled), which surfaces as MException 'nxr:cancelled'.
 */

#include "nxr/compute.h"
#include "marshal.h"
#include "mex.h"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

// libut is MATLAB's internal utilities lib. utIsInterruptPending()
// is the canonical Ctrl-C bridge used by libigl-matlab, gptoolbox,
// and others; it's been stable across MATLAB releases for 20+ years
// despite being undocumented. Linkage is automatic on Windows
// (libut.lib ships next to libmex.lib in the matlabroot/extern dir).
extern "C" bool utIsInterruptPending();

using namespace nxr::manifold::mex;

namespace {

// Build a CancellationToken that fires when MATLAB user hits Ctrl-C.
// Wrapped in std::function so the same nxr::core::CancellationToken type
// covers both this and SAB-backed JS flags.
nxr::core::CancellationToken makeCtrlCToken() {
    return nxr::core::CancellationToken([]() {
        return utIsInterruptPending();
    });
}

// Lowercase first letter of an error-code name for MATLAB identifier
// convention: 'nxr:nonManifold' rather than 'nxr:NON_MANIFOLD'.
// MATLAB's MException identifier rules require a lowercase first
// letter on each colon-segment.
std::string toMatlabIdentifier(nxr::core::ErrorCode code) {
    std::string name(nxr::core::errorCodeName(code));
    // Convert UPPER_SNAKE → camelCase: lowercase the first letter
    // and remove underscores, capitalizing the next char.
    std::string out;
    out.reserve(name.size());
    bool nextUpper = false;
    bool first = true;
    for (char c : name) {
        if (c == '_') { nextUpper = true; continue; }
        if (first) {
            out.push_back(static_cast<char>(std::tolower(c)));
            first = false;
        } else if (nextUpper) {
            out.push_back(c);  // already uppercase
            nextUpper = false;
        } else {
            out.push_back(static_cast<char>(std::tolower(c)));
        }
    }
    return out;
}

// ── Stateful context handle map ──────────────────────────────
//
// MEX analogue of the WASM ContextWrapper (bindings/wasm/src/
// nxr_compute_wasm.cpp). Each ContextHolder keeps the geometry-central
// mesh, the assembled operators, the CholeskyCache, the cached eigen
// result, and the stateful geometry-central solvers alive across calls,
// so repeated ops on one mesh skip the halfedge rebuild + refactorization.
// The uint64 handle returned to MATLAB is the proxy-pointer analogue.
//
// Single-threaded: MEX is invoked from MATLAB's one thread; no mutex.

struct ContextHolder {
    std::unique_ptr<nxr::manifold::Manifold>                      ctx;
    std::unique_ptr<nxr::manifold::ops::ManifoldOperators>        ops;       // lazy: ensureOps
    std::unique_ptr<nxr::manifold::ops::DECOperators>             dec;       // lazy: ensureDec
    std::unique_ptr<nxr::manifold::ops::CholeskyCache>            cache;     // built at create
    std::unique_ptr<nxr::manifold::solve::EigenResult>            eigCache;  // set by solve/precompute
    std::unique_ptr<nxr::manifold::transport::VectorHeatSolver>   vhm;       // lazy: ensureVHM
    std::unique_ptr<nxr::manifold::solve::SignedHeatSolver>       shs;       // lazy: ensureSHS
    std::unique_ptr<nxr::manifold::solve::HeatGeodesicSolver>     heatGeo;   // lazy: ensureHeatGeo

    std::map<std::pair<int, bool>, Eigen::MatrixXd>              smoothFaceFieldCache;
    std::map<std::pair<int, bool>,
             nxr::manifold::connection::SmoothVertexFieldResult> smoothVertexFieldCache;

    using CLKey = std::tuple<
        nxr::manifold::ops::laplacian::connection::ConnectionDomain,
        int, double,
        nxr::manifold::ops::laplacian::connection::ConnectionLaplacianFormat>;
    std::map<CLKey, std::shared_ptr<
        nxr::manifold::ops::laplacian::connection::ConnectionLaplacian>> clCache;
};

static uint64_t sNextHandle = 1;
static std::unordered_map<uint64_t, ContextHolder> sContexts;

// True iff prhs[1] is a scalar uint64 (drives additive handle dispatch).
bool isHandleArg(int nrhs, const mxArray** prhs) {
    return nrhs >= 2 && mxIsUint64(prhs[1]) && mxGetNumberOfElements(prhs[1]) == 1;
}

// Resolve the holder for a uint64 handle; throw nxr:invalidHandle on miss.
// References to unordered_map elements stay valid across rehash, so the
// returned reference is safe to hold for the duration of one command.
ContextHolder& getHolder(const mxArray* arr) {
    if (!mxIsUint64(arr) || mxGetNumberOfElements(arr) != 1) {
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidHandle,
            "expected a scalar uint64 context handle");
    }
    uint64_t h = *static_cast<const uint64_t*>(mxGetData(arr));
    auto it = sContexts.find(h);
    if (it == sContexts.end()) {
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidHandle,
            "invalid or destroyed context handle");
    }
    return it->second;
}

// Lazy initialisers — construct once, reuse (mirror WASM ensure*).
nxr::manifold::ops::ManifoldOperators& ensureOps(ContextHolder& h) {
    if (!h.ops) h.ops = std::make_unique<nxr::manifold::ops::ManifoldOperators>(
        nxr::manifold::ops::assembleManifoldOperators(*h.ctx));
    return *h.ops;
}
nxr::manifold::ops::DECOperators& ensureDec(ContextHolder& h) {
    if (!h.dec) h.dec = std::make_unique<nxr::manifold::ops::DECOperators>(
        nxr::manifold::ops::assembleDECOperators(*h.ctx));
    return *h.dec;
}
nxr::manifold::transport::VectorHeatSolver& ensureVHM(ContextHolder& h) {
    if (!h.vhm) h.vhm = std::make_unique<nxr::manifold::transport::VectorHeatSolver>(*h.ctx);
    return *h.vhm;
}
nxr::manifold::solve::SignedHeatSolver& ensureSHS(ContextHolder& h) {
    if (!h.shs) h.shs = std::make_unique<nxr::manifold::solve::SignedHeatSolver>(*h.ctx);
    return *h.shs;
}
nxr::manifold::solve::HeatGeodesicSolver& ensureHeatGeo(ContextHolder& h) {
    if (!h.heatGeo) h.heatGeo = std::make_unique<nxr::manifold::solve::HeatGeodesicSolver>(*h.ctx);
    return *h.heatGeo;
}

// ── create / destroy ─────────────────────────────────────────

void cmdCreate(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 3) {
        throw std::invalid_argument(
            "nxr_compute('create', V, F) takes exactly 2 arguments");
    }
    int nV = 0, nF = 0;
    auto verts = mxToVertexBuffer(prhs[1], nV);
    auto faces = mxToFaceBuffer(prhs[2], nF);

    ContextHolder holder;
    holder.ctx   = std::make_unique<nxr::manifold::Manifold>(
        verts.data(), nV, faces.data(), nF);
    holder.cache = std::make_unique<nxr::manifold::ops::CholeskyCache>();

    uint64_t h = sNextHandle++;
    sContexts.emplace(h, std::move(holder));

    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *static_cast<uint64_t*>(mxGetData(plhs[0])) = h;
}

void cmdDestroy(int /*nlhs*/, mxArray** /*plhs*/, int nrhs, const mxArray** prhs) {
    if (nrhs != 2) {
        throw std::invalid_argument(
            "nxr_compute('destroy', handle) takes exactly 1 argument");
    }
    if (mxIsUint64(prhs[1]) && mxGetNumberOfElements(prhs[1]) == 1) {
        sContexts.erase(*static_cast<const uint64_t*>(mxGetData(prhs[1])));
    }
}

// ── assembleManifoldOperators(V, F) → struct ─────────────────────

void cmdAssembleMeshOperators(int /*nlhs*/, mxArray** plhs,
                              int nrhs, const mxArray** prhs) {
    if (isHandleArg(nrhs, prhs)) {
        ContextHolder& h = getHolder(prhs[1]);
        auto& ops = ensureOps(h);
        plhs[0] = meshOperatorsToStruct(ops, h.ctx->nV(), h.ctx->nE(), h.ctx->nF());
        return;
    }
    if (nrhs != 3) {
        throw std::invalid_argument(
            "nxr_compute('assembleManifoldOperators', V, F) takes exactly 2 arguments");
    }
    int nV = 0, nF = 0;
    auto verts = mxToVertexBuffer(prhs[1], nV);
    auto faces = mxToFaceBuffer(prhs[2], nF);

    nxr::manifold::Manifold m(verts.data(), nV, faces.data(), nF);
    auto ops = nxr::manifold::ops::assembleManifoldOperators(m);
    plhs[0] = meshOperatorsToStruct(ops, m.nV(), m.nE(), m.nF());
}

// ── solve(K, M, k) → struct ────────────────────────

void cmdSolveEigenmodes(int /*nlhs*/, mxArray** plhs,
                        int nrhs, const mxArray** prhs) {
    if (isHandleArg(nrhs, prhs)) {
        if (nrhs < 3 || nrhs > 4) {
            throw std::invalid_argument(
                "nxr_compute('solve', handle, k, [sigma]) takes 2 or 3 arguments");
        }
        ContextHolder& h = getHolder(prhs[1]);
        auto& ops = ensureOps(h);
        int k = getIntArg(prhs[2]);
        double sigma = (nrhs >= 4) ? getDoubleArg(prhs[3]) : -1e-8;
        auto result = nxr::manifold::solve::eigen(
            ops.cotanLaplacian, ops.mass, k, sigma,
            /*normalize=*/false, /*removeDC=*/false, makeCtrlCToken());
        h.eigCache = std::make_unique<nxr::manifold::solve::EigenResult>(result);
        plhs[0] = eigenResultToStruct(result);
        return;
    }
    if (nrhs < 4 || nrhs > 5) {
        throw std::invalid_argument(
            "nxr_compute('solve', K, M, k, [sigma]) takes 3 or 4 arguments");
    }
    auto K = mxToEigenSparse(prhs[1]);
    auto M = mxToEigenSparse(prhs[2]);
    int k = getIntArg(prhs[3]);
    double sigma = (nrhs >= 5) ? getDoubleArg(prhs[4]) : -1e-8;

    // Ctrl-C polling lives entirely in the token; nxr-compute doesn't know about MATLAB.
    auto result = nxr::manifold::solve::eigen(K, M, k, sigma,
        /*normalize=*/false, /*removeDC=*/false, makeCtrlCToken());
    plhs[0] = eigenResultToStruct(result);
}

// ── normalize(U, M) → U ────────────────────────────

void cmdNormalizeEigenmodes(int /*nlhs*/, mxArray** plhs,
                            int nrhs, const mxArray** prhs) {
    if (nrhs != 3) {
        throw std::invalid_argument(
            "nxr_compute('normalize', U, M) takes exactly 2 arguments");
    }
    auto U = mxToEigenMatrix(prhs[1]);
    auto M = mxToEigenSparse(prhs[2]);

    auto Un = nxr::manifold::solve::normalize(U, M);
    plhs[0] = eigenMatrixToMx(Un);
}

// ── removeDC(eigStruct) → eigStruct ──────────────────────────

void cmdRemoveDC(int /*nlhs*/, mxArray** plhs,
                 int nrhs, const mxArray** prhs) {
    if (nrhs != 2) {
        throw std::invalid_argument(
            "nxr_compute('removeDC', eig) takes exactly 1 argument");
    }
    auto eig = mxToEigenResult(prhs[1]);
    auto trimmed = nxr::manifold::solve::removeDC(eig);
    plhs[0] = eigenResultToStruct(trimmed);
}

// ── precompute(V, F, k) → eigStruct (whole pipeline) ─────────
//
// One-shot wrapper that runs the canonical precompute pipeline:
// assemble → solve → normalize → removeDC. MATLAB users who don't
// need intermediate operators can just call this.

void cmdPrecompute(int /*nlhs*/, mxArray** plhs,
                   int nrhs, const mxArray** prhs) {
    if (isHandleArg(nrhs, prhs)) {
        if (nrhs != 3) {
            throw std::invalid_argument(
                "nxr_compute('precompute', handle, k) takes exactly 2 arguments");
        }
        ContextHolder& h = getHolder(prhs[1]);
        auto& ops = ensureOps(h);
        int k = getIntArg(prhs[2]);
        auto eig = nxr::manifold::solve::eigen(ops.cotanLaplacian, ops.mass, k, -1e-8,
            /*normalize=*/true, /*removeDC=*/true, makeCtrlCToken());
        h.eigCache = std::make_unique<nxr::manifold::solve::EigenResult>(eig);
        plhs[0] = eigenResultToStruct(eig);
        return;
    }
    if (nrhs != 4) {
        throw std::invalid_argument(
            "nxr_compute('precompute', V, F, k) takes exactly 3 arguments");
    }
    int nV = 0, nF = 0;
    auto verts = mxToVertexBuffer(prhs[1], nV);
    auto faces = mxToFaceBuffer(prhs[2], nF);
    int k = getIntArg(prhs[3]);

    nxr::manifold::Manifold m(verts.data(), nV, faces.data(), nF);
    auto ops = nxr::manifold::ops::assembleManifoldOperators(m);
    auto eig = nxr::manifold::solve::eigen(ops.cotanLaplacian, ops.mass, k, -1e-8,
        /*normalize=*/true, /*removeDC=*/true, makeCtrlCToken());

    plhs[0] = eigenResultToStruct(eig);
}

// ── Vector heat method (Sharp, Soliman, Crane 2019) ─────────
//
// In the stateless (V, F) calling convention each command builds a
// fresh Manifold + VectorHeatSolver per call, paying the factor cost
// every invocation. In the handle convention the VectorHeatSolver is
// cached on the ContextHolder via ensureVHM() — built once,
// back-substitution only thereafter. Prefer a handle
// (nxr_compute('create', V, F)) for repeated solves on one mesh.

void cmdVectorHeatTransport(int /*nlhs*/, mxArray** plhs,
                            int nrhs, const mxArray** prhs) {
    if (isHandleArg(nrhs, prhs)) {
        if (nrhs != 4) {
            throw std::invalid_argument(
                "nxr_compute('parallel', handle, sourceVerts, sourceVectors) "
                "takes exactly 3 arguments");
        }
        ContextHolder& h = getHolder(prhs[1]);
        auto srcIdx = mxToVertexIndices(prhs[2]);
        auto srcVecs = mxToEigenMatrix(prhs[3]);
        if (srcVecs.cols() != 3 ||
            static_cast<std::size_t>(srcVecs.rows()) != srcIdx.size()) {
            throw std::invalid_argument(
                "sourceVectors must be Nx3 with N = numel(sourceVerts)");
        }
        Eigen::MatrixXd out = nxr::manifold::transport::parallel(ensureVHM(h), srcIdx, srcVecs);
        plhs[0] = eigenMatrixToMx(out);
        return;
    }
    if (nrhs != 5) {
        throw std::invalid_argument(
            "nxr_compute('parallel', V, F, sourceVerts, sourceVectors) "
            "takes exactly 4 arguments");
    }
    int nV = 0, nF = 0;
    auto verts = mxToVertexBuffer(prhs[1], nV);
    auto faces = mxToFaceBuffer(prhs[2], nF);
    auto srcIdx = mxToVertexIndices(prhs[3]);   // 1-based → 0-based
    auto srcVecs = mxToEigenMatrix(prhs[4]);    // Nx3 column-major
    if (srcVecs.cols() != 3 ||
        static_cast<std::size_t>(srcVecs.rows()) != srcIdx.size()) {
        throw std::invalid_argument(
            "sourceVectors must be Nx3 with N = numel(sourceVerts)");
    }

    nxr::manifold::Manifold m(verts.data(), nV, faces.data(), nF);
    nxr::manifold::transport::VectorHeatSolver vhm(m);
    Eigen::MatrixXd out = nxr::manifold::transport::parallel(vhm, srcIdx, srcVecs);
    plhs[0] = eigenMatrixToMx(out);             // V×3 column-major (MATLAB native)
}

void cmdVectorHeatExtendScalar(int /*nlhs*/, mxArray** plhs,
                               int nrhs, const mxArray** prhs) {
    if (isHandleArg(nrhs, prhs)) {
        if (nrhs != 4) {
            throw std::invalid_argument(
                "nxr_compute('extendScalar', handle, sourceVerts, sourceValues) "
                "takes exactly 3 arguments");
        }
        ContextHolder& h = getHolder(prhs[1]);
        auto srcIdx = mxToVertexIndices(prhs[2]);
        auto srcVal = mxToEigenVector(prhs[3]);
        if (static_cast<std::size_t>(srcVal.size()) != srcIdx.size()) {
            throw std::invalid_argument(
                "sourceValues must have the same length as sourceVerts");
        }
        Eigen::VectorXd out = nxr::manifold::transport::extendScalar(ensureVHM(h), srcIdx, srcVal);
        plhs[0] = eigenVectorToMx(out);
        return;
    }
    if (nrhs != 5) {
        throw std::invalid_argument(
            "nxr_compute('extendScalar', V, F, sourceVerts, sourceValues) "
            "takes exactly 4 arguments");
    }
    int nV = 0, nF = 0;
    auto verts = mxToVertexBuffer(prhs[1], nV);
    auto faces = mxToFaceBuffer(prhs[2], nF);
    auto srcIdx = mxToVertexIndices(prhs[3]);
    auto srcVal = mxToEigenVector(prhs[4]);
    if (static_cast<std::size_t>(srcVal.size()) != srcIdx.size()) {
        throw std::invalid_argument(
            "sourceValues must have the same length as sourceVerts");
    }

    nxr::manifold::Manifold m(verts.data(), nV, faces.data(), nF);
    nxr::manifold::transport::VectorHeatSolver vhm(m);
    Eigen::VectorXd out = nxr::manifold::transport::extendScalar(vhm, srcIdx, srcVal);
    plhs[0] = eigenVectorToMx(out);
}

void cmdVectorHeatLogMap(int /*nlhs*/, mxArray** plhs,
                         int nrhs, const mxArray** prhs) {
    if (isHandleArg(nrhs, prhs)) {
        if (nrhs < 3 || nrhs > 4) {
            throw std::invalid_argument(
                "nxr_compute('logMap', handle, sourceVertex, [strategy]) "
                "takes 2 or 3 arguments");
        }
        ContextHolder& h = getHolder(prhs[1]);
        int sourceVertex = getIntArg(prhs[2]) - 1;
        int strategy = (nrhs >= 4)
            ? getIntArg(prhs[3])
            : static_cast<int>(nxr::manifold::transport::LogMapStrategy::AffineLocal);
        auto r = nxr::manifold::transport::logMap(
            ensureVHM(h), sourceVertex,
            static_cast<nxr::manifold::transport::LogMapStrategy>(strategy));
        const char* fields[] = {"logCoords", "sourceE1", "sourceE2"};
        mxArray* s = mxCreateStructMatrix(1, 1, 3, fields);
        mxSetField(s, 0, "logCoords", eigenMatrixToMx(r.logCoords));
        mxArray* e1 = mxCreateDoubleMatrix(1, 3, mxREAL);
        mxArray* e2 = mxCreateDoubleMatrix(1, 3, mxREAL);
        double* e1p = mxGetPr(e1);
        double* e2p = mxGetPr(e2);
        e1p[0] = r.sourceE1.x(); e1p[1] = r.sourceE1.y(); e1p[2] = r.sourceE1.z();
        e2p[0] = r.sourceE2.x(); e2p[1] = r.sourceE2.y(); e2p[2] = r.sourceE2.z();
        mxSetField(s, 0, "sourceE1", e1);
        mxSetField(s, 0, "sourceE2", e2);
        plhs[0] = s;
        return;
    }
    if (nrhs < 4 || nrhs > 5) {
        throw std::invalid_argument(
            "nxr_compute('logMap', V, F, sourceVertex, [strategy]) "
            "takes 3 or 4 arguments");
    }
    int nV = 0, nF = 0;
    auto verts = mxToVertexBuffer(prhs[1], nV);
    auto faces = mxToFaceBuffer(prhs[2], nF);
    int sourceVertex = getIntArg(prhs[3]) - 1;  // 1-based → 0-based
    int strategy = (nrhs >= 5)
        ? getIntArg(prhs[4])
        : static_cast<int>(nxr::manifold::transport::LogMapStrategy::AffineLocal);

    nxr::manifold::Manifold m(verts.data(), nV, faces.data(), nF);
    nxr::manifold::transport::VectorHeatSolver vhm(m);
    auto r = nxr::manifold::transport::logMap(
        vhm, sourceVertex,
        static_cast<nxr::manifold::transport::LogMapStrategy>(strategy));

    const char* fields[] = {"logCoords", "sourceE1", "sourceE2"};
    mxArray* s = mxCreateStructMatrix(1, 1, 3, fields);
    mxSetField(s, 0, "logCoords", eigenMatrixToMx(r.logCoords));
    mxArray* e1 = mxCreateDoubleMatrix(1, 3, mxREAL);
    mxArray* e2 = mxCreateDoubleMatrix(1, 3, mxREAL);
    double* e1p = mxGetPr(e1);
    double* e2p = mxGetPr(e2);
    e1p[0] = r.sourceE1.x(); e1p[1] = r.sourceE1.y(); e1p[2] = r.sourceE1.z();
    e2p[0] = r.sourceE2.x(); e2p[1] = r.sourceE2.y(); e2p[2] = r.sourceE2.z();
    mxSetField(s, 0, "sourceE1", e1);
    mxSetField(s, 0, "sourceE2", e2);
    plhs[0] = s;
}

void cmdVectorHeatFindCenter(int /*nlhs*/, mxArray** plhs,
                             int nrhs, const mxArray** prhs) {
    if (isHandleArg(nrhs, prhs)) {
        if (nrhs < 3 || nrhs > 4) {
            throw std::invalid_argument(
                "nxr_compute('findCenter', handle, sourceVerts, [p]) "
                "takes 2 or 3 arguments");
        }
        ContextHolder& h = getHolder(prhs[1]);
        auto srcIdx = mxToVertexIndices(prhs[2]);
        int p = (nrhs >= 4) ? getIntArg(prhs[3]) : 2;
        Eigen::Vector3d c = nxr::manifold::transport::findCenter(ensureVHM(h), srcIdx, p);
        mxArray* out = mxCreateDoubleMatrix(1, 3, mxREAL);
        double* op = mxGetPr(out);
        op[0] = c.x(); op[1] = c.y(); op[2] = c.z();
        plhs[0] = out;
        return;
    }
    if (nrhs < 4 || nrhs > 5) {
        throw std::invalid_argument(
            "nxr_compute('findCenter', V, F, sourceVerts, [p]) "
            "takes 3 or 4 arguments");
    }
    int nV = 0, nF = 0;
    auto verts = mxToVertexBuffer(prhs[1], nV);
    auto faces = mxToFaceBuffer(prhs[2], nF);
    auto srcIdx = mxToVertexIndices(prhs[3]);
    int p = (nrhs >= 5) ? getIntArg(prhs[4]) : 2;

    nxr::manifold::Manifold m(verts.data(), nV, faces.data(), nF);
    nxr::manifold::transport::VectorHeatSolver vhm(m);
    Eigen::Vector3d c = nxr::manifold::transport::findCenter(vhm, srcIdx, p);
    mxArray* out = mxCreateDoubleMatrix(1, 3, mxREAL);
    double* op = mxGetPr(out);
    op[0] = c.x(); op[1] = c.y(); op[2] = c.z();
    plhs[0] = out;
}

// ── Signed heat method (Feng & Crane 2024) ──────────────────

void cmdSignedHeatDistance(int /*nlhs*/, mxArray** plhs,
                           int nrhs, const mxArray** prhs) {
    if (isHandleArg(nrhs, prhs)) {
        if (nrhs < 4 || nrhs > 5) {
            throw std::invalid_argument(
                "nxr_compute('signedHeat', handle, curveVerts, isLoop, [levelSet]) "
                "takes 3 or 4 arguments");
        }
        ContextHolder& h = getHolder(prhs[1]);
        auto curveIdx = mxToVertexIndices(prhs[2]);
        bool isLoop = getIntArg(prhs[3]) != 0;
        int ls = (nrhs >= 5)
            ? getIntArg(prhs[4])
            : static_cast<int>(nxr::manifold::solve::SignedHeatLevelSet::ZeroSet);
        Eigen::VectorXd out = nxr::manifold::solve::signedHeat(
            ensureSHS(h), curveIdx, isLoop,
            static_cast<nxr::manifold::solve::SignedHeatLevelSet>(ls));
        plhs[0] = eigenVectorToMx(out);
        return;
    }
    if (nrhs < 5 || nrhs > 6) {
        throw std::invalid_argument(
            "nxr_compute('signedHeat', V, F, curveVerts, isLoop, [levelSet]) "
            "takes 4 or 5 arguments");
    }
    int nV = 0, nF = 0;
    auto verts    = mxToVertexBuffer(prhs[1], nV);
    auto faces    = mxToFaceBuffer(prhs[2], nF);
    auto curveIdx = mxToVertexIndices(prhs[3]);
    bool isLoop = getIntArg(prhs[4]) != 0;
    int  ls     = (nrhs >= 6)
        ? getIntArg(prhs[5])
        : static_cast<int>(nxr::manifold::solve::SignedHeatLevelSet::ZeroSet);

    nxr::manifold::Manifold m(verts.data(), nV, faces.data(), nF);
    nxr::manifold::solve::SignedHeatSolver shs(m);
    Eigen::VectorXd out = nxr::manifold::solve::signedHeat(
        shs, curveIdx, isLoop,
        static_cast<nxr::manifold::solve::SignedHeatLevelSet>(ls));
    plhs[0] = eigenVectorToMx(out);
}

// ── Smooth direction fields (Knöppel-Crane) ─────────────────

void cmdComputeSmoothFaceField(int /*nlhs*/, mxArray** plhs,
                               int nrhs, const mxArray** prhs) {
    if (isHandleArg(nrhs, prhs)) {
        if (nrhs < 2 || nrhs > 4) {
            throw std::invalid_argument(
                "nxr_compute('smoothFace', handle, [nSym], [alignToCurvature]) "
                "takes 1, 2, or 3 arguments");
        }
        ContextHolder& h = getHolder(prhs[1]);
        int  nSym  = (nrhs >= 3) ? getIntArg(prhs[2])      : 4;
        bool align = (nrhs >= 4) ? (getIntArg(prhs[3]) != 0) : false;
        auto key = std::make_pair(nSym, align);
        auto it = h.smoothFaceFieldCache.find(key);
        if (it == h.smoothFaceFieldCache.end()) {
            Eigen::MatrixXd v = nxr::manifold::connection::smoothFace(*h.ctx, nSym, align);
            it = h.smoothFaceFieldCache.emplace(key, std::move(v)).first;
        }
        plhs[0] = eigenMatrixToMx(it->second);
        return;
    }
    if (nrhs < 3 || nrhs > 5) {
        throw std::invalid_argument(
            "nxr_compute('smoothFace', V, F, [nSym], [alignToCurvature]) "
            "takes 2, 3, or 4 arguments");
    }
    int nV = 0, nF = 0;
    auto verts = mxToVertexBuffer(prhs[1], nV);
    auto faces = mxToFaceBuffer(prhs[2], nF);
    int  nSym  = (nrhs >= 4) ? getIntArg(prhs[3])    : 4;
    bool align = (nrhs >= 5) ? (getIntArg(prhs[4]) != 0) : false;

    nxr::manifold::Manifold m(verts.data(), nV, faces.data(), nF);
    Eigen::MatrixXd out = nxr::manifold::connection::smoothFace(m, nSym, align);
    plhs[0] = eigenMatrixToMx(out);   // F×3
}

void cmdComputeSmoothVertexField(int /*nlhs*/, mxArray** plhs,
                                 int nrhs, const mxArray** prhs) {
    if (isHandleArg(nrhs, prhs)) {
        if (nrhs < 2 || nrhs > 4) {
            throw std::invalid_argument(
                "nxr_compute('smoothVertex', handle, [nSym], [alignToCurvature]) "
                "takes 1, 2, or 3 arguments");
        }
        ContextHolder& h = getHolder(prhs[1]);
        int  nSym  = (nrhs >= 3) ? getIntArg(prhs[2])      : 2;
        bool align = (nrhs >= 4) ? (getIntArg(prhs[3]) != 0) : false;
        auto key = std::make_pair(nSym, align);
        auto it = h.smoothVertexFieldCache.find(key);
        if (it == h.smoothVertexFieldCache.end()) {
            auto r = nxr::manifold::connection::smoothVertex(*h.ctx, nSym, align);
            it = h.smoothVertexFieldCache.emplace(key, std::move(r)).first;
        }
        const auto& r = it->second;
        const char* fields[] = {"vertexVectors", "vertexFieldRaw", "nSym"};
        mxArray* s = mxCreateStructMatrix(1, 1, 3, fields);
        mxSetField(s, 0, "vertexVectors",  eigenMatrixToMx(r.vertexVectors));
        mxSetField(s, 0, "vertexFieldRaw", eigenVectorToMx(r.vertexFieldRaw));
        mxSetField(s, 0, "nSym",           mxCreateDoubleScalar(r.nSym));
        plhs[0] = s;
        return;
    }
    if (nrhs < 3 || nrhs > 5) {
        throw std::invalid_argument(
            "nxr_compute('smoothVertex', V, F, [nSym], [alignToCurvature]) "
            "takes 2, 3, or 4 arguments");
    }
    int nV = 0, nF = 0;
    auto verts = mxToVertexBuffer(prhs[1], nV);
    auto faces = mxToFaceBuffer(prhs[2], nF);
    int  nSym  = (nrhs >= 4) ? getIntArg(prhs[3])      : 2;
    bool align = (nrhs >= 5) ? (getIntArg(prhs[4]) != 0) : false;

    nxr::manifold::Manifold m(verts.data(), nV, faces.data(), nF);
    auto r = nxr::manifold::connection::smoothVertex(m, nSym, align);

    const char* fields[] = {"vertexVectors", "vertexFieldRaw", "nSym"};
    mxArray* s = mxCreateStructMatrix(1, 1, 3, fields);
    mxSetField(s, 0, "vertexVectors",  eigenMatrixToMx(r.vertexVectors));
    mxSetField(s, 0, "vertexFieldRaw", eigenVectorToMx(r.vertexFieldRaw));
    mxSetField(s, 0, "nSym",           mxCreateDoubleScalar(r.nSym));
    plhs[0] = s;
}

// ── Stripe patterns (Knöppel-Crane SIGGRAPH 2015) ───────────
//
// Returns positions as a (2*segCount)x3 matrix of endpoint pairs.
// MATLAB users render with line() / patch() over consecutive pairs.

namespace {
mxArray* stripePatternResultToStruct(const nxr::manifold::parametrization::stripes::StripePatternResult& r) {
    const char* fields[] = {"positions", "segmentCount"};
    mxArray* s = mxCreateStructMatrix(1, 1, 2, fields);
    mxSetField(s, 0, "positions",    eigenMatrixToMx(r.positions));
    mxSetField(s, 0, "segmentCount", mxCreateDoubleScalar(r.segmentCount));
    return s;
}
} // namespace

void cmdComputeStripePattern(int /*nlhs*/, mxArray** plhs,
                             int nrhs, const mxArray** prhs) {
    if (isHandleArg(nrhs, prhs)) {
        if (nrhs < 4 || nrhs > 5) {
            throw std::invalid_argument(
                "nxr_compute('compute', handle, vertexFieldRaw, frequency, [connect]) "
                "takes 3 or 4 arguments");
        }
        ContextHolder& h = getHolder(prhs[1]);
        Eigen::VectorXd raw = mxToEigenVector(prhs[2]);
        double freq = getDoubleArg(prhs[3]);
        bool connect = (nrhs >= 5) ? (getIntArg(prhs[4]) != 0) : true;
        auto r = nxr::manifold::parametrization::stripes::compute(*h.ctx, raw, freq, connect);
        plhs[0] = stripePatternResultToStruct(r);
        return;
    }
    if (nrhs < 5 || nrhs > 6) {
        throw std::invalid_argument(
            "nxr_compute('compute', V, F, vertexFieldRaw, frequency, [connect]) "
            "takes 4 or 5 arguments");
    }
    int nV = 0, nF = 0;
    auto verts = mxToVertexBuffer(prhs[1], nV);
    auto faces = mxToFaceBuffer(prhs[2], nF);
    Eigen::VectorXd raw = mxToEigenVector(prhs[3]);
    double freq = getDoubleArg(prhs[4]);
    bool connect = (nrhs >= 6) ? (getIntArg(prhs[5]) != 0) : true;

    nxr::manifold::Manifold m(verts.data(), nV, faces.data(), nF);
    auto r = nxr::manifold::parametrization::stripes::compute(m, raw, freq, connect);
    plhs[0] = stripePatternResultToStruct(r);
}

void cmdComputeStripePatternFreq(int /*nlhs*/, mxArray** plhs,
                                 int nrhs, const mxArray** prhs) {
    if (isHandleArg(nrhs, prhs)) {
        if (nrhs < 4 || nrhs > 5) {
            throw std::invalid_argument(
                "nxr_compute('computeFreq', handle, vertexFieldRaw, frequencies, [connect]) "
                "takes 3 or 4 arguments");
        }
        ContextHolder& h = getHolder(prhs[1]);
        Eigen::VectorXd raw   = mxToEigenVector(prhs[2]);
        Eigen::VectorXd freqs = mxToEigenVector(prhs[3]);
        bool connect = (nrhs >= 5) ? (getIntArg(prhs[4]) != 0) : true;
        auto r = nxr::manifold::parametrization::stripes::computeFreq(*h.ctx, raw, freqs, connect);
        plhs[0] = stripePatternResultToStruct(r);
        return;
    }
    if (nrhs < 5 || nrhs > 6) {
        throw std::invalid_argument(
            "nxr_compute('computeFreq', V, F, vertexFieldRaw, frequencies, [connect]) "
            "takes 4 or 5 arguments");
    }
    int nV = 0, nF = 0;
    auto verts = mxToVertexBuffer(prhs[1], nV);
    auto faces = mxToFaceBuffer(prhs[2], nF);
    Eigen::VectorXd raw   = mxToEigenVector(prhs[3]);
    Eigen::VectorXd freqs = mxToEigenVector(prhs[4]);
    bool connect = (nrhs >= 6) ? (getIntArg(prhs[5]) != 0) : true;

    nxr::manifold::Manifold m(verts.data(), nV, faces.data(), nF);
    auto r = nxr::manifold::parametrization::stripes::computeFreq(m, raw, freqs, connect);
    plhs[0] = stripePatternResultToStruct(r);
}

// ── Parity ops (handle-only) ─────────────────────────────────
//
// Group B — operators

void cmdAssembleDECOperators(int /*nlhs*/, mxArray** plhs,
                             int nrhs, const mxArray** prhs) {
    if (nrhs != 2) {
        throw std::invalid_argument(
            "nxr_compute('assembleDECOperators', handle) takes exactly 1 argument");
    }
    ContextHolder& h = getHolder(prhs[1]);
    plhs[0] = decOperatorsToStruct(ensureDec(h));
}

void cmdAssembleConnectionLaplacian(int /*nlhs*/, mxArray** plhs,
                                    int nrhs, const mxArray** prhs) {
    if (nrhs < 2 || nrhs > 3) {
        throw std::invalid_argument(
            "nxr_compute('assembleConnectionLaplacian', handle, [opts]) "
            "takes 1 or 2 arguments");
    }
    namespace cl = nxr::manifold::ops::laplacian::connection;
    ContextHolder& h = getHolder(prhs[1]);
    cl::ConnectionLaplacianOptions o;   // value-init → vertex / nSym=1 / 1e-8 / Real2N
    if (nrhs >= 3 && !mxIsEmpty(prhs[2])) {
        if (!mxIsStruct(prhs[2])) {
            throw std::invalid_argument("opts must be a struct");
        }
        const mxArray* f;
        if ((f = mxGetField(prhs[2], 0, "domain")))         o.domain = cl::parseConnectionDomain(getStringArg(f));
        if ((f = mxGetField(prhs[2], 0, "nSym")))           o.nSym = getIntArg(f);
        if ((f = mxGetField(prhs[2], 0, "regularization"))) o.regularization = getDoubleArg(f);
        if ((f = mxGetField(prhs[2], 0, "format")))         o.format = cl::parseConnectionLaplacianFormat(getStringArg(f));
    }
    ContextHolder::CLKey key{o.domain, o.nSym, o.regularization, o.format};
    auto it = h.clCache.find(key);
    if (it == h.clCache.end()) {
        auto clp = std::make_shared<cl::ConnectionLaplacian>(
            cl::assembleConnectionLaplacian(*h.ctx, o));
        it = h.clCache.emplace(key, std::move(clp)).first;
    }
    plhs[0] = connectionLaplacianToStruct(*it->second);
}

void cmdFrames(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 2) {
        throw std::invalid_argument("nxr_compute('frames', handle) takes exactly 1 argument");
    }
    ContextHolder& h = getHolder(prhs[1]);
    plhs[0] = faceFramesToStruct(nxr::manifold::geometry::frames(*h.ctx));
}

void cmdVertexFrames(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 2) {
        throw std::invalid_argument("nxr_compute('vertexFrames', handle) takes exactly 1 argument");
    }
    ContextHolder& h = getHolder(prhs[1]);
    plhs[0] = vertexFramesToStruct(nxr::manifold::geometry::vertexFrames(*h.ctx));
}

nxr::manifold::geometry::NormalType parseNormalType(const std::string& s) {
    using NT = nxr::manifold::geometry::NormalType;
    if (s == "angle")  return NT::AngleWeighted;
    if (s == "area")   return NT::AreaWeighted;
    if (s == "equal")  return NT::EqualWeighted;
    if (s == "sphere") return NT::SphereInscribed;
    if (s == "mean")   return NT::MeanCurvature;
    if (s == "gauss")  return NT::GaussCurvature;
    throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
        "unknown normal type \"" + s + "\" (use angle|area|equal|sphere|mean|gauss)");
}

void cmdNormals(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 2 || nrhs > 3) {
        throw std::invalid_argument(
            "nxr_compute('normals', handle, [type]) takes 1 or 2 arguments");
    }
    ContextHolder& h = getHolder(prhs[1]);
    auto type = nxr::manifold::geometry::NormalType::AngleWeighted;
    if (nrhs >= 3) type = parseNormalType(getStringArg(prhs[2]));
    plhs[0] = eigenMatrixToMx(nxr::manifold::geometry::normals(*h.ctx, type));
}

// Group C — solvers

void cmdPoisson(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 4) {
        throw std::invalid_argument(
            "nxr_compute('poisson', handle, sourceVerts, sourceValues) takes exactly 3 arguments");
    }
    ContextHolder& h = getHolder(prhs[1]);
    auto idx = mxToVertexIndices(prhs[2]);   // 1-based → 0-based
    auto val = mxToEigenVector(prhs[3]);
    if (static_cast<std::size_t>(val.size()) != idx.size()) {
        throw std::invalid_argument("sourceValues must match sourceVerts length");
    }
    std::map<int, double> density;
    for (std::size_t i = 0; i < idx.size(); ++i) {
        density[idx[i]] = val[static_cast<Eigen::Index>(i)];
    }
    Eigen::VectorXd phi = nxr::manifold::solve::poisson(ensureOps(h), *h.cache, density);
    plhs[0] = eigenVectorToMx(phi);
}

void cmdHeat(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 3) {
        throw std::invalid_argument(
            "nxr_compute('heat', handle, sourceVerts) takes exactly 2 arguments");
    }
    ContextHolder& h = getHolder(prhs[1]);
    auto idx = mxToVertexIndices(prhs[2]);
    Eigen::VectorXd d = nxr::manifold::solve::heat(ensureHeatGeo(h), idx);
    plhs[0] = eigenVectorToMx(d);
}

void cmdTracePath(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 4) {
        throw std::invalid_argument(
            "nxr_compute('tracePath', handle, vStart, vEnd) takes exactly 3 arguments");
    }
    ContextHolder& h = getHolder(prhs[1]);
    int a = getIntArg(prhs[2]) - 1;   // 1-based → 0-based
    int b = getIntArg(prhs[3]) - 1;
    Eigen::MatrixXd pts = nxr::manifold::query::tracePath(*h.ctx, a, b);
    plhs[0] = eigenMatrixToMx(pts);
}

void cmdHodge(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 3) {
        throw std::invalid_argument(
            "nxr_compute('hodge', handle, omega) takes exactly 2 arguments");
    }
    ContextHolder& h = getHolder(prhs[1]);
    Eigen::VectorXd omega = mxToEigenVector(prhs[2]);
    auto r = nxr::manifold::solve::hodge(*h.ctx, ensureDec(h), *h.cache, omega);
    plhs[0] = hodgeResultToStruct(r);
}

// Group D — geometric

void cmdCurvatures(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 2) {
        throw std::invalid_argument("nxr_compute('curvatures', handle) takes exactly 1 argument");
    }
    ContextHolder& h = getHolder(prhs[1]);
    plhs[0] = curvatureResultToStruct(nxr::manifold::geometry::curvatures(*h.ctx));
}

void cmdBff(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 2) {
        throw std::invalid_argument("nxr_compute('bff', handle) takes exactly 1 argument");
    }
    ContextHolder& h = getHolder(prhs[1]);
    plhs[0] = eigenMatrixToMx(nxr::manifold::parametrization::bff(*h.ctx));   // [V, 2]
}

void cmdIsoline(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 3 || nrhs > 6) {
        throw std::invalid_argument(
            "nxr_compute('isoline', handle, scalarField, [numLevels], [minVal], [maxVal]) "
            "takes 2 to 5 arguments");
    }
    ContextHolder& h = getHolder(prhs[1]);
    Eigen::VectorXd field = mxToEigenVector(prhs[2]);
    int    numLevels = (nrhs >= 4) ? getIntArg(prhs[3])    : 20;
    double minV      = (nrhs >= 5) ? getDoubleArg(prhs[4]) : 0.0;
    double maxV      = (nrhs >= 6) ? getDoubleArg(prhs[5]) : 0.0;
    auto r = nxr::field::extract::isoline(*h.ctx, field, numLevels, minV, maxV);
    plhs[0] = positionsSegmentsToStruct(r.positions, r.segmentCount);
}

void cmdDirectionField(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 4) {
        throw std::invalid_argument(
            "nxr_compute('directionField', handle, singVerts, singValues) takes exactly 3 arguments.\n"
            "Note: 'trivial' is a deprecated alias for 'directionField'.");
    }
    ContextHolder& h = getHolder(prhs[1]);
    auto idx = mxToVertexIndices(prhs[2]);   // 1-based → 0-based
    auto val = mxToEigenVector(prhs[3]);
    if (static_cast<std::size_t>(val.size()) != idx.size()) {
        throw std::invalid_argument("singValues must match singVerts length");
    }
    std::map<int, double> sing;
    for (std::size_t i = 0; i < idx.size(); ++i) {
        sing[idx[i]] = val[static_cast<Eigen::Index>(i)];
    }
    auto r = nxr::manifold::connection::trivial(*h.ctx, ensureDec(h), *h.cache, sing);
    plhs[0] = directionFieldResultToStruct(r);
}

void cmdTrivialConnectionLaplacian(int /*nlhs*/, mxArray** plhs,
                                    int nrhs, const mxArray** prhs) {
    if (nrhs < 4 || nrhs > 5) {
        throw std::invalid_argument(
            "nxr_compute('trivialConnectionLaplacian', handle, singVerts, singValues, [opts]) "
            "takes 3 or 4 arguments.\n"
            "  singVerts : 1-based vertex indices of singularities\n"
            "  singValues: corresponding singularity indices (sum must equal Euler characteristic)\n"
            "  opts      : optional struct with fields nSym (default 1),\n"
            "              regularization (default 1e-8), format ('complex'|'real2N')");
    }

    namespace cl = nxr::manifold::ops::laplacian::connection;
    ContextHolder& h = getHolder(prhs[1]);

    // Parse singularity map (1-based MATLAB indices → 0-based C++ indices)
    auto idx = mxToVertexIndices(prhs[2]);
    auto val = mxToEigenVector(prhs[3]);
    if (static_cast<std::size_t>(val.size()) != idx.size()) {
        throw std::invalid_argument(
            "trivialConnectionLaplacian: singValues must match singVerts length");
    }
    std::map<int, double> sing;
    for (std::size_t i = 0; i < idx.size(); ++i)
        sing[idx[i]] = val[static_cast<Eigen::Index>(i)];

    // Parse options — default to Complex format (primary MATLAB consumer calls eigs on K_complex)
    cl::ConnectionLaplacianOptions o;
    o.domain = cl::ConnectionDomain::Vertex;
    o.format = cl::ConnectionLaplacianFormat::Complex;
    if (nrhs >= 5 && !mxIsEmpty(prhs[4])) {
        if (!mxIsStruct(prhs[4]))
            throw std::invalid_argument("opts must be a struct");
        const mxArray* f;
        if ((f = mxGetField(prhs[4], 0, "nSym")))           o.nSym = getIntArg(f);
        if ((f = mxGetField(prhs[4], 0, "regularization"))) o.regularization = getDoubleArg(f);
        if ((f = mxGetField(prhs[4], 0, "format")))         o.format = cl::parseConnectionLaplacianFormat(getStringArg(f));
    }

    auto result = cl::assembleTrivialConnectionLaplacian(
        *h.ctx, sing, ensureDec(h), *h.cache, o);
    plhs[0] = connectionLaplacianToStruct(result);
}

void cmdStreamline(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 3 || nrhs > 6) {
        throw std::invalid_argument(
            "nxr_compute('streamline', handle, faceField, [numSeeds], [stepCoef], [maxSteps]) "
            "takes 2 to 5 arguments");
    }
    ContextHolder& h = getHolder(prhs[1]);
    Eigen::MatrixXd faceField = mxToEigenMatrix(prhs[2]);   // [nF, 3]
    int    numSeeds = (nrhs >= 4) ? getIntArg(prhs[3])    : 15;
    double stepCoef = (nrhs >= 5) ? getDoubleArg(prhs[4]) : 0.15;
    int    maxSteps = (nrhs >= 6) ? getIntArg(prhs[5])    : 1000;
    auto r = nxr::field::extract::streamline(*h.ctx, faceField, numSeeds, stepCoef, maxSteps);
    plhs[0] = positionsSegmentsToStruct(r.positions, r.segmentCount);
}

// Group E — vector field

void cmdWhitney(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 3) {
        throw std::invalid_argument("nxr_compute('whitney', handle, oneForm) takes exactly 2 arguments");
    }
    ContextHolder& h = getHolder(prhs[1]);
    Eigen::VectorXd oneForm = mxToEigenVector(prhs[2]);
    Eigen::MatrixXd v = nxr::field::interp::whitney(*h.ctx, ensureDec(h), oneForm);
    plhs[0] = eigenMatrixToMx(v);
}

void cmdGradient(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 3) {
        throw std::invalid_argument("nxr_compute('gradient', handle, scalarField) takes exactly 2 arguments");
    }
    ContextHolder& h = getHolder(prhs[1]);
    Eigen::VectorXd scalar = mxToEigenVector(prhs[2]);
    Eigen::MatrixXd g = nxr::field::op::gradient(*h.ctx, scalar);
    plhs[0] = eigenMatrixToMx(g);
}

// Group F — time-varying generators (need a prior solve/precompute)

void cmdHeatDiffusion(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 5 || nrhs > 6) {
        throw std::invalid_argument(
            "nxr_compute('heatDiffusion', handle, sourceVerts, sourceValues, timesteps, [alpha]) "
            "takes 4 or 5 arguments");
    }
    ContextHolder& h = getHolder(prhs[1]);
    if (!h.eigCache) {
        throw nxr::core::Error(nxr::core::ErrorCode::NotPrecomputed,
            "heatDiffusion requires a prior solve/precompute on this handle");
    }
    auto idx = mxToVertexIndices(prhs[2]);
    auto val = mxToEigenVector(prhs[3]);
    if (static_cast<std::size_t>(val.size()) != idx.size()) {
        throw std::invalid_argument("sourceValues must match sourceVerts length");
    }
    Eigen::VectorXd ts = mxToEigenVector(prhs[4]);
    double alpha = (nrhs >= 6) ? getDoubleArg(prhs[5]) : 1.0;
    std::map<int, double> sources;
    for (std::size_t i = 0; i < idx.size(); ++i) {
        sources[idx[i]] = val[static_cast<Eigen::Index>(i)];
    }
    Eigen::VectorXd u0 = nxr::field::generate::delta(h.ctx->nV(), sources);
    std::vector<double> timesteps(ts.data(), ts.data() + ts.size());
    Eigen::MatrixXf out = nxr::field::generate::heatDiffusion(
        ensureOps(h), *h.eigCache, u0, timesteps, alpha);
    plhs[0] = eigenMatrixXfToMx(out);
}

void cmdDampedWave(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 7) {
        throw std::invalid_argument(
            "nxr_compute('dampedWave', handle, modeIndices, amplitudes, dampings, phases, timesteps) "
            "takes exactly 6 arguments");
    }
    ContextHolder& h = getHolder(prhs[1]);
    if (!h.eigCache) {
        throw nxr::core::Error(nxr::core::ErrorCode::NotPrecomputed,
            "dampedWave requires a prior solve/precompute on this handle");
    }
    auto modeIdx = mxToVertexIndices(prhs[2]);   // 1-based → 0-based mode columns
    auto toVec = [](const mxArray* a) {
        auto v = mxToEigenVector(a);
        return std::vector<double>(v.data(), v.data() + v.size());
    };
    std::vector<double> amps   = toVec(prhs[3]);
    std::vector<double> damps  = toVec(prhs[4]);
    std::vector<double> phases = toVec(prhs[5]);
    std::vector<double> ts     = toVec(prhs[6]);
    Eigen::MatrixXf out = nxr::field::generate::dampedWave(
        *h.eigCache, modeIdx, amps, damps, phases, ts);
    plhs[0] = eigenMatrixXfToMx(out);
}

void cmdRandomDecomposed1Form(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 5 || nrhs > 6) {
        throw std::invalid_argument(
            "nxr_compute('randomDecomposed1Form', handle, alphaStr, betaStr, gammaStr, [seed]) "
            "takes 4 or 5 arguments");
    }
    ContextHolder& h = getHolder(prhs[1]);
    auto& dec = ensureDec(h);
    double aS = getDoubleArg(prhs[2]);
    double bS = getDoubleArg(prhs[3]);
    double gS = getDoubleArg(prhs[4]);
    unsigned int seed = (nrhs >= 6) ? static_cast<unsigned int>(getIntArg(prhs[5])) : 42u;
    Eigen::VectorXd omega = nxr::field::generate::randomDecomposed1Form(
        dec, h.ctx->nV(), h.ctx->nE(), h.ctx->nF(), aS, bS, gS, seed);
    plhs[0] = eigenVectorToMx(omega);
}

// ── version() → string ───────────────────────────────────────

void cmdVersion(int /*nlhs*/, mxArray** plhs,
                int /*nrhs*/, const mxArray** /*prhs*/) {
    plhs[0] = mxCreateString("nxr-compute 0.1.0");
}

} // namespace

// ── mexFunction entry point ──────────────────────────────────

void mexFunction(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
    // Register the context-map cleanup once, so `clear mex` / MATLAB exit
    // frees every live ContextHolder (geometry-central mesh + caches).
    static bool sAtExitRegistered = false;
    if (!sAtExitRegistered) {
        mexAtExit([]() { sContexts.clear(); });
        sAtExitRegistered = true;
    }

    if (nrhs < 1 || !mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("nxr:nrhs",
            "First argument must be a command string. Try nxr_compute('version').");
    }

    std::string cmd;
    try {
        cmd = getStringArg(prhs[0]);
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("nxr:badCommand", "%s", e.what());
        return;
    }

    try {
        if      (cmd == "create")        cmdCreate(nlhs, plhs, nrhs, prhs);
        else if (cmd == "destroy")       cmdDestroy(nlhs, plhs, nrhs, prhs);
        else if (cmd == "assembleManifoldOperators")   cmdAssembleMeshOperators(nlhs, plhs, nrhs, prhs);
        else if (cmd == "solve")         cmdSolveEigenmodes(nlhs, plhs, nrhs, prhs);
        else if (cmd == "normalize")     cmdNormalizeEigenmodes(nlhs, plhs, nrhs, prhs);
        else if (cmd == "removeDC")                cmdRemoveDC(nlhs, plhs, nrhs, prhs);
        else if (cmd == "precompute")              cmdPrecompute(nlhs, plhs, nrhs, prhs);
        else if (cmd == "parallel")     cmdVectorHeatTransport(nlhs, plhs, nrhs, prhs);
        else if (cmd == "extendScalar")  cmdVectorHeatExtendScalar(nlhs, plhs, nrhs, prhs);
        else if (cmd == "logMap")        cmdVectorHeatLogMap(nlhs, plhs, nrhs, prhs);
        else if (cmd == "findCenter")    cmdVectorHeatFindCenter(nlhs, plhs, nrhs, prhs);
        else if (cmd == "signedHeat")      cmdSignedHeatDistance(nlhs, plhs, nrhs, prhs);
        else if (cmd == "smoothFace")  cmdComputeSmoothFaceField(nlhs, plhs, nrhs, prhs);
        else if (cmd == "smoothVertex") cmdComputeSmoothVertexField(nlhs, plhs, nrhs, prhs);
        else if (cmd == "compute")    cmdComputeStripePattern(nlhs, plhs, nrhs, prhs);
        else if (cmd == "computeFreq") cmdComputeStripePatternFreq(nlhs, plhs, nrhs, prhs);
        else if (cmd == "assembleDECOperators")        cmdAssembleDECOperators(nlhs, plhs, nrhs, prhs);
        else if (cmd == "assembleConnectionLaplacian") cmdAssembleConnectionLaplacian(nlhs, plhs, nrhs, prhs);
        else if (cmd == "frames")                      cmdFrames(nlhs, plhs, nrhs, prhs);
        else if (cmd == "vertexFrames")                cmdVertexFrames(nlhs, plhs, nrhs, prhs);
        else if (cmd == "normals")                     cmdNormals(nlhs, plhs, nrhs, prhs);
        else if (cmd == "poisson")                     cmdPoisson(nlhs, plhs, nrhs, prhs);
        else if (cmd == "heat")                        cmdHeat(nlhs, plhs, nrhs, prhs);
        else if (cmd == "tracePath")                   cmdTracePath(nlhs, plhs, nrhs, prhs);
        else if (cmd == "hodge")                       cmdHodge(nlhs, plhs, nrhs, prhs);
        else if (cmd == "curvatures")                  cmdCurvatures(nlhs, plhs, nrhs, prhs);
        else if (cmd == "bff")                         cmdBff(nlhs, plhs, nrhs, prhs);
        else if (cmd == "isoline")                     cmdIsoline(nlhs, plhs, nrhs, prhs);
        else if (cmd == "directionField")              cmdDirectionField(nlhs, plhs, nrhs, prhs);
        else if (cmd == "trivial")                     cmdDirectionField(nlhs, plhs, nrhs, prhs); // deprecated alias
        else if (cmd == "trivialConnectionLaplacian")  cmdTrivialConnectionLaplacian(nlhs, plhs, nrhs, prhs);
        else if (cmd == "streamline")                  cmdStreamline(nlhs, plhs, nrhs, prhs);
        else if (cmd == "whitney")                     cmdWhitney(nlhs, plhs, nrhs, prhs);
        else if (cmd == "gradient")                    cmdGradient(nlhs, plhs, nrhs, prhs);
        else if (cmd == "heatDiffusion")               cmdHeatDiffusion(nlhs, plhs, nrhs, prhs);
        else if (cmd == "dampedWave")                  cmdDampedWave(nlhs, plhs, nrhs, prhs);
        else if (cmd == "randomDecomposed1Form")       cmdRandomDecomposed1Form(nlhs, plhs, nrhs, prhs);
        else if (cmd == "version")                 cmdVersion(nlhs, plhs, nrhs, prhs);
        else {
            mexErrMsgIdAndTxt("nxr:unknownCommand",
                "Unknown command: \"%s\". Available: create, destroy, assembleManifoldOperators, "
                "solve, normalize, removeDC, precompute, "
                "parallel, extendScalar, logMap, "
                "findCenter, signedHeat, smoothFace, "
                "smoothVertex, compute, "
                "computeFreq, version.",
                cmd.c_str());
        }
    } catch (const nxr::core::Error& e) {
        // Route structured nxr-compute errors to MException identifiers MATLAB
        // can pattern-match (e.g. ME.identifier == "nxr:cancelled").
        std::string id = "nxr:" + toMatlabIdentifier(e.code());
        if (e.hint().empty()) {
            mexErrMsgIdAndTxt(id.c_str(), "nxr_compute('%s'): %s", cmd.c_str(), e.what());
        } else {
            std::string hint(e.hint());
            mexErrMsgIdAndTxt(id.c_str(), "nxr_compute('%s'): %s\n  hint: %s",
                              cmd.c_str(), e.what(), hint.c_str());
        }
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("nxr:error", "nxr_compute('%s') failed: %s", cmd.c_str(), e.what());
    }
}
