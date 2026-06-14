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
#include "nxr/facets.h"
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

    std::unique_ptr<Eigen::SparseMatrix<double>> graphLap;       // lazy: graphLaplacian
    std::unique_ptr<Eigen::SparseMatrix<double>> massGalerkin;   // lazy: Galerkin mass

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

// Parse the 'operators' flag out of an optional opts struct.
// Returns true iff opts is a struct with a non-empty truthy 'operators' field.
static bool readOperatorsFlag(const mxArray* opts) {
    if (!opts || !mxIsStruct(opts)) return false;
    const mxArray* f = mxGetField(opts, 0, "operators");
    if (!f || mxIsEmpty(f)) return false;
    if (mxIsLogical(f)) return mxGetLogicals(f)[0];
    return mxIsNumeric(f) && mxGetScalar(f) != 0.0;
}

// Parse the 'coupling' option out of an optional opts struct.
// Reads opts.coupling string ('product'|'ambient'), default Ambient, throws on unknown.
static nxr::manifold::ops::laplacian::connection::CovariantCoupling
parseCoupling(const mxArray* opts) {
    namespace cl = nxr::manifold::ops::laplacian::connection;
    if (opts && mxIsStruct(opts)) {
        const mxArray* f = mxGetField(opts, 0, "coupling");
        if (f && mxIsChar(f)) {
            std::string s = getStringArg(f);
            if (s == "product") return cl::CovariantCoupling::Product;
            if (s == "ambient") return cl::CovariantCoupling::Ambient;
            throw std::invalid_argument("coupling must be 'product' or 'ambient'");
        }
    }
    return cl::CovariantCoupling::Ambient;   // default
}

// ── create / destroy ─────────────────────────────────────────

void cmdCreate(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 3 || nrhs > 4) {
        throw std::invalid_argument(
            "nxr_compute('create', V, F[, opts]) takes 2 or 3 arguments");
    }
    int nV = 0, nF = 0;
    auto verts = mxToVertexBuffer(prhs[1], nV);
    auto faces = mxToFaceBuffer(prhs[2], nF);

    bool intrinsicDelaunay = false;
    if (nrhs >= 4) {
        if (!mxIsStruct(prhs[3]))
            throw std::invalid_argument("nxr_compute('create', V, F, opts): opts must be a struct");
        const mxArray* f = mxGetField(prhs[3], 0, "intrinsicDelaunay");
        if (f && !mxIsEmpty(f))
            intrinsicDelaunay = mxIsLogical(f) ? mxGetLogicals(f)[0] : (mxGetScalar(f) != 0.0);
    }

    ContextHolder holder;
    holder.ctx   = std::make_unique<nxr::manifold::Manifold>(
        verts.data(), nV, faces.data(), nF, intrinsicDelaunay);
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

// ── fixDelaunay(V, F) → [V2, F2, nFlips]  (Delaunay edge-flip repair) ──
//
// Stateless command. Parses V (Vx3 double, 1-based-OK) and F (Fx3, 1-based),
// runs nxr::manifold::fixDelaunay, returns:
//   plhs[0] = V (passthrough via mxDuplicateArray — vertices unchanged)
//   plhs[1] = F2 (Fx3 double, 1-based)  — only when nlhs >= 2
//   plhs[2] = nFlips (scalar double)    — only when nlhs >= 3

void cmdFixDelaunay(int nlhs, mxArray** plhs,
                    int nrhs, const mxArray** prhs) {
    if (nrhs != 3) throw std::invalid_argument(
        "nxr_compute('fixDelaunay', V, F) takes V and F");
    int nV = 0, nF = 0;
    std::vector<double>        Vbuf = mxToVertexBuffer(prhs[1], nV);   // row-major xyz
    std::vector<std::int32_t>  Fbuf = mxToFaceBuffer(prhs[2], nF);     // 0-based

    auto r = nxr::manifold::fixDelaunay(Vbuf.data(), nV, Fbuf.data(), nF);

    plhs[0] = mxDuplicateArray(prhs[1]);                         // V2 == V (passthrough)
    if (nlhs >= 2) {
        Eigen::MatrixXd Fd = r.faces.cast<double>().array() + 1.0;  // 0-based → 1-based
        plhs[1] = eigenMatrixToMx(Fd);                               // nF×3, column-major
    }
    if (nlhs >= 3) {
        mxArray* nf = mxCreateDoubleMatrix(1, 1, mxREAL);
        *mxGetPr(nf) = static_cast<double>(r.flips);
        plhs[2] = nf;
    }
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

// ── Covariant differential: frameTransport / liftToWorld / liftToFrame ──
//
// Thin wrappers around nxr::manifold::differential::* that handle 1-based
// MATLAB index conversion and [nV×3] matrix marshalling via the existing
// mxToEigenMatrix / eigenMatrixToMx helpers from marshal.h.

// nxr_compute('frameTransport', h, i, j) -> [3×3]   (i,j are 1-based)
void cmdFrameTransport(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 4) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
        "frameTransport: expected nxr_compute('frameTransport', handle, i, j).");
    ContextHolder& h = getHolder(prhs[1]);
    int i = static_cast<int>(mxGetScalar(prhs[2])) - 1;   // 1-based -> 0-based
    int j = static_cast<int>(mxGetScalar(prhs[3])) - 1;
    plhs[0] = eigenMatrixToMx(nxr::manifold::differential::frameTransport(*h.ctx, i, j));
}

// nxr_compute('liftToWorld', h, Lloc[nV×3]) -> [nV×3]
void cmdLiftToWorld(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 3) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
        "liftToWorld: expected nxr_compute('liftToWorld', handle, field[nV x 3]).");
    ContextHolder& h = getHolder(prhs[1]);
    plhs[0] = eigenMatrixToMx(
        nxr::manifold::differential::liftToWorld(*h.ctx, mxToEigenMatrix(prhs[2])));
}

// nxr_compute('liftToFrame', h, Lworld[nV×3]) -> [nV×3]
void cmdLiftToFrame(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 3) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
        "liftToFrame: expected nxr_compute('liftToFrame', handle, field[nV x 3]).");
    ContextHolder& h = getHolder(prhs[1]);
    plhs[0] = eigenMatrixToMx(
        nxr::manifold::differential::liftToFrame(*h.ctx, mxToEigenMatrix(prhs[2])));
}

// ── operator sub-struct builders ─────────────────────────────
//
// Called when the caller passes opts.operators = true.
// Each builder pulls from the same lazy caches (ensureOps/ensureDec/clCache)
// and the new graphLap/massGalerkin slots on ContextHolder.

mxArray* buildTopologyOperators(ContextHolder& h) {
    if (!h.graphLap)
        h.graphLap = std::make_unique<Eigen::SparseMatrix<double>>(
            nxr::manifold::ops::graphLaplacian(*h.ctx));
    auto& dec = ensureDec(h);
    const char* f[] = {"laplacian","dec"};
    mxArray* s = mxCreateStructMatrix(1,1,2,f);
    mxSetField(s,0,"laplacian", eigenSparseToMx(*h.graphLap));
    const char* df[] = {"d0","d1"};
    mxArray* d = mxCreateStructMatrix(1,1,2,df);
    mxSetField(d,0,"d0", eigenSparseToMx(dec.d0));
    mxSetField(d,0,"d1", eigenSparseToMx(dec.d1));
    mxSetField(s,0,"dec", d);
    return s;
}

mxArray* buildGeometryOperators(ContextHolder& h) {
    auto& ops = ensureOps(h);
    auto& dec = ensureDec(h);
    if (!h.massGalerkin) {
        auto g = nxr::manifold::ops::assembleManifoldOperators(
                     *h.ctx, nxr::manifold::ops::MassMatrixVariant::Galerkin);
        h.massGalerkin = std::make_unique<Eigen::SparseMatrix<double>>(g.mass);
    }
    const char* f[] = {"laplacian","mass","hodge"};
    mxArray* s = mxCreateStructMatrix(1,1,3,f);
    mxSetField(s,0,"laplacian", eigenSparseToMx(ops.cotanLaplacian));
    const char* mf[] = {"lumped","galerkin"};
    mxArray* mm = mxCreateStructMatrix(1,1,2,mf);
    mxSetField(mm,0,"lumped",   eigenSparseToMx(dec.hodge0));
    mxSetField(mm,0,"galerkin", eigenSparseToMx(*h.massGalerkin));
    mxSetField(s,0,"mass", mm);
    const char* hf[] = {"h0","h1","h2","h1inv"};
    mxArray* hh = mxCreateStructMatrix(1,1,4,hf);
    mxSetField(hh,0,"h0",    eigenSparseToMx(dec.hodge0));
    mxSetField(hh,0,"h1",    eigenSparseToMx(dec.hodge1));
    mxSetField(hh,0,"h2",    eigenSparseToMx(dec.hodge2));
    mxSetField(hh,0,"h1inv", eigenSparseToMx(dec.hodge1Inverse));
    mxSetField(s,0,"hodge", hh);
    return s;
}

// type/opts mirror buildGaugeStruct; opts may carry singVerts/singValues for 'trivial'.
// coupling selects the covariant Laplacian variant (Product or Ambient, default Ambient).
mxArray* buildGaugeOperators(ContextHolder& h, const std::string& type, const mxArray* opts,
                              nxr::manifold::ops::laplacian::connection::CovariantCoupling coupling) {
    namespace cl = nxr::manifold::ops::laplacian::connection;
    namespace conn = nxr::manifold::connection;
    cl::ConnectionLaplacianOptions o;
    o.domain = cl::ConnectionDomain::Vertex;
    o.nSym   = 1;
    o.format = cl::ConnectionLaplacianFormat::Complex;
    Eigen::SparseMatrix<std::complex<double>> K;

    // Realized gauge grid: LC frame for euclidean/levi-civita; trivial rotates each row.
    Eigen::MatrixXcd grid = nxr::manifold::geometry::vertexGrid(*h.ctx);

    if (type == "trivial") {
        if (!opts || !mxIsStruct(opts))
            throw std::invalid_argument("gauge('trivial') operators require singVerts/singValues in opts");
        const mxArray* fv = mxGetField(opts, 0, "singVerts");
        const mxArray* fi = mxGetField(opts, 0, "singValues");
        if (!fv || !fi) throw std::invalid_argument("opts needs singVerts and singValues");
        std::vector<int> verts = mxToVertexIndices(fv);
        Eigen::VectorXd  vals  = mxToEigenVector(fi);
        if (static_cast<size_t>(vals.size()) != verts.size())
            throw std::invalid_argument("singVerts and singValues length mismatch");
        std::map<int,double> sing;
        for (size_t i = 0; i < verts.size(); ++i) sing[verts[i]] = vals[static_cast<Eigen::Index>(i)];
        // Compute trivial gauge rotations once; reuse both for K and for the realized grid.
        conn::GaugeRotations gr =
            conn::integrateTrivialGaugeRotations(*h.ctx, ensureDec(h), *h.cache, sing);
        auto clr = cl::assembleTrivialConnectionLaplacian(*h.ctx, sing, ensureDec(h), *h.cache, o);
        K = clr.K_complex;
        // Apply per-vertex rotation to realize the trivial gauge frame.
        for (int v = 0; v < static_cast<int>(grid.rows()); ++v)
            grid.row(v) *= gr.vertex(v);
    } else if (type == "euclidean" || type == "levi-civita") {  // Levi-Civita connection Laplacian (clCache)
        ContextHolder::CLKey key{o.domain, o.nSym, o.regularization, o.format};
        auto it = h.clCache.find(key);
        if (it == h.clCache.end()) {
            auto clp = std::make_shared<cl::ConnectionLaplacian>(
                cl::assembleConnectionLaplacian(*h.ctx, o));
            it = h.clCache.emplace(key, std::move(clp)).first;
        }
        K = it->second->K_complex;
        // grid stays as the raw LC frame (rotation == 1 for levi-civita).
    } else {
        throw std::invalid_argument("unknown gauge type '" + type + "'");
    }

    // Covariant 3-frame Laplacian.
    const auto& cotanL = ensureOps(h).cotanLaplacian;
    Eigen::SparseMatrix<double> covL =
        cl::assembleCovariantLaplacian(coupling, K, grid, cotanL);

    const char* f[] = {"laplacian", "covariantLaplacian"};
    mxArray* s = mxCreateStructMatrix(1,1,2,f);
    mxSetField(s,0,"laplacian",           eigenComplexSparseToMx(K));
    mxSetField(s,0,"covariantLaplacian",  eigenSparseToMx(covL));
    return s;
}

// ── buildGaugeStruct / buildGeometryStruct / buildTopologyStruct ──
//
// Reusable builders shared by the standalone cmdXxx commands and
// cmdBundle. Each takes a ContextHolder reference (already resolved)
// plus the command-specific parameters and returns a new mxArray*.

// ── gauge(handle, type[, opts]) → struct ─────────────────────
//
// Returns the Gauge struct for the requested connection type.
//   type == "euclidean" or "levi-civita":  vertex/face rotation = ones (identity)
//   type == "trivial":                     rotations = integrateTrivialGaugeRotations(...)
//                                          opts struct requires singVerts (1-based) + singValues
//
// Rotations are COMBING multipliers: rotation .* grid is the combed
// (trivially-parallel) frame, and real(rotation .* grid) is the trivial
// parallel direction field. See GaugeRotations in nxr/compute.h.
//
// Schema (schemaVersion == 1):
//   G.type                  string
//   G.vertex.rotation       [nV x 1]  complex double  (unit modulus)
//   G.face.rotation         [nF x 1]  complex double  (unit modulus)
//   G.singularity.vertices  [k x 1]   uint32  1-based
//   G.singularity.indices   [k x 1]   double
//   G.singularity.source    string

mxArray* buildGaugeStruct(ContextHolder& h, const std::string& type, const mxArray* opts,
                          bool withOps = false) {
    namespace conn = nxr::manifold::connection;
    nxr::manifold::Manifold& m = *h.ctx;

    const int nV = m.nV();
    const int nF = m.nF();
    Eigen::VectorXcd rot  = Eigen::VectorXcd::Ones(nV);  // identity by default
    Eigen::VectorXcd rotF = Eigen::VectorXcd::Ones(nF);  // identity by default
    std::vector<long> singVerts;
    Eigen::VectorXd   singIdx;
    std::string singSource = "none";

    if (type == "euclidean" || type == "levi-civita") {
        // rotation stays identity; nothing else to compute.
    } else if (type == "trivial") {
        if (opts == nullptr || !mxIsStruct(opts))
            throw std::invalid_argument(
                "gauge('trivial') requires opts struct with singVerts, singValues");
        const mxArray* fv = mxGetField(opts, 0, "singVerts");
        const mxArray* fi = mxGetField(opts, 0, "singValues");
        if (!fv || !fi) throw std::invalid_argument("opts needs singVerts and singValues");
        std::vector<int> verts = mxToVertexIndices(fv);     // 1-based -> 0-based
        Eigen::VectorXd  vals  = mxToEigenVector(fi);
        if (static_cast<size_t>(vals.size()) != verts.size())
            throw std::invalid_argument("singVerts and singValues length mismatch");
        std::map<int,double> sing;
        for (size_t i = 0; i < verts.size(); ++i) sing[verts[i]] = vals[static_cast<Eigen::Index>(i)];

        conn::GaugeRotations gr =
            conn::integrateTrivialGaugeRotations(m, ensureDec(h), *h.cache, sing);
        rot  = gr.vertex;
        rotF = gr.face;

        singVerts.assign(verts.begin(), verts.end());       // 0-based; marshal converts to 1-based
        singIdx = vals;
        if (const mxArray* fs = mxGetField(opts, 0, "source"))
            singSource = getStringArg(fs);
        else
            singSource = "manual";
    } else {
        throw std::invalid_argument("unknown gauge type '" + type + "'");
    }

    const char* topF[] = {"schemaVersion","type","vertex","face","singularity"};
    mxArray* s = mxCreateStructMatrix(1,1,5,topF);
    mxSetField(s,0,"schemaVersion",scalarToMx(1));
    mxSetField(s,0,"type",mxCreateString(type.c_str()));

    { const char* f[] = {"rotation"}; mxArray* g = mxCreateStructMatrix(1,1,1,f);
      mxSetField(g,0,"rotation",eigenComplexVectorToMx(rot));
      mxSetField(s,0,"vertex",g); }
    { const char* f[] = {"rotation"}; mxArray* g = mxCreateStructMatrix(1,1,1,f);
      mxSetField(g,0,"rotation",eigenComplexVectorToMx(rotF));
      mxSetField(s,0,"face",g); }
    { const char* f[] = {"vertices","indices","source"};
      mxArray* g = mxCreateStructMatrix(1,1,3,f);
      mxSetField(g,0,"vertices",indexVectorToMx1Based(singVerts));
      mxSetField(g,0,"indices",eigenVectorToMx(singIdx));
      mxSetField(g,0,"source",mxCreateString(singSource.c_str()));
      mxSetField(s,0,"singularity",g); }

    if (withOps) {
        int fn = mxAddField(s, "operators");
        mxSetFieldByNumber(s, 0, fn, buildGaugeOperators(h, type, opts, parseCoupling(opts)));
    }

    return s;
}

void cmdGauge(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 3) throw std::invalid_argument(
        "nxr_compute('gauge', handle, type[, opts]) — type in {euclidean,levi-civita,trivial}");
    ContextHolder& h = getHolder(prhs[1]);
    std::string type = getStringArg(prhs[2]);
    const mxArray* opts = (nrhs >= 4) ? prhs[3] : nullptr;
    bool withOps = readOperatorsFlag(opts);
    plhs[0] = buildGaugeStruct(h, type, opts, withOps);
}

// ── geometry(handle) → struct ─────────────────────────────────
//
// Returns a nested element-grouped struct of all light per-element
// geometry quantities (dual areas, angle sums, curvature, frames,
// edge lengths, face centroids, halfedge transport, corner angles).
// Heavy sparse operators are intentionally excluded.
//
// Schema (schemaVersion == 1):
//   G.totalArea                   double scalar
//   G.vertex.dualAreas            [nV x 1]  double
//   G.vertex.angleSums            [nV x 1]  double
//   G.vertex.curvature            [nV x 1]  complex double (2-RoSy deviatoric q)
//   G.vertex.meanCurvature        [nV x 1]  double
//   G.vertex.grid                 [nV x 3]  complex double (c = e1+i*e2)
//   G.edge.lengths                [nE x 1]  double
//   G.edge.cotanWeights           [nE x 1]  double
//   G.edge.dihedralAngles         [nE x 1]  double
//   G.face.areas                  [nF x 1]  double
//   G.face.centroids              [nF x 3]  double
//   G.face.grid                   [nF x 3]  complex double
//   G.halfedge.cotanWeights       [nH x 1]  double
//   G.halfedge.vectorsInVertex    [nH x 1]  complex double
//   G.halfedge.vectorsInFace      [nH x 1]  complex double
//   G.halfedge.transportAlong     [nH x 1]  complex double
//   G.halfedge.transportAcross    [nH x 1]  complex double
//   G.corner.angles               [nH x 1]  double (nH == nC on closed mesh)
//   G.corner.scaledAngles         [nH x 1]  double

mxArray* buildGeometryStruct(ContextHolder& h, bool withOps = false) {
    nxr::manifold::geometry::MeshGeometry g =
        nxr::manifold::geometry::meshGeometry(*h.ctx);

    const char* topF[] = {"schemaVersion","totalArea","vertex","edge","face","halfedge","corner"};
    mxArray* s = mxCreateStructMatrix(1, 1, 7, topF);
    mxSetField(s, 0, "schemaVersion", scalarToMx(1));
    mxSetField(s, 0, "totalArea",     scalarToMx(g.totalArea));

    { const char* f[] = {"dualAreas","angleSums","curvature","meanCurvature","grid"};
      mxArray* gg = mxCreateStructMatrix(1, 1, 5, f);
      mxSetField(gg, 0, "dualAreas",      eigenVectorToMx(g.vertexDualAreas));
      mxSetField(gg, 0, "angleSums",      eigenVectorToMx(g.vertexAngleSums));
      mxSetField(gg, 0, "curvature",      eigenComplexVectorToMx(g.vertexCurvatureDeviatoric));
      mxSetField(gg, 0, "meanCurvature",  eigenVectorToMx(g.vertexMeanCurvature));
      mxSetField(gg, 0, "grid",           eigenComplexMatrixToMx(g.vertexGrid));
      mxSetField(s,  0, "vertex", gg); }

    { const char* f[] = {"lengths","cotanWeights","dihedralAngles"};
      mxArray* gg = mxCreateStructMatrix(1, 1, 3, f);
      mxSetField(gg, 0, "lengths",        eigenVectorToMx(g.edgeLengths));
      mxSetField(gg, 0, "cotanWeights",   eigenVectorToMx(g.edgeCotanWeights));
      mxSetField(gg, 0, "dihedralAngles", eigenVectorToMx(g.edgeDihedralAngles));
      mxSetField(s,  0, "edge", gg); }

    { const char* f[] = {"areas","centroids","grid"};
      mxArray* gg = mxCreateStructMatrix(1, 1, 3, f);
      mxSetField(gg, 0, "areas",     eigenVectorToMx(g.faceAreas));
      mxSetField(gg, 0, "centroids", eigenMatrixToMx(g.faceCentroids));
      mxSetField(gg, 0, "grid",      eigenComplexMatrixToMx(g.faceGrid));
      mxSetField(s,  0, "face", gg); }

    { const char* f[] = {"cotanWeights","vectorsInVertex","vectorsInFace","transportAlong","transportAcross"};
      mxArray* gg = mxCreateStructMatrix(1, 1, 5, f);
      mxSetField(gg, 0, "cotanWeights",    eigenVectorToMx(g.halfedgeCotanWeights));
      mxSetField(gg, 0, "vectorsInVertex", eigenComplexVectorToMx(g.halfedgeVectorsInVertex));
      mxSetField(gg, 0, "vectorsInFace",   eigenComplexVectorToMx(g.halfedgeVectorsInFace));
      mxSetField(gg, 0, "transportAlong",  eigenComplexVectorToMx(g.halfedgeTransportAlong));
      mxSetField(gg, 0, "transportAcross", eigenComplexVectorToMx(g.halfedgeTransportAcross));
      mxSetField(s,  0, "halfedge", gg); }

    { const char* f[] = {"angles","scaledAngles"};
      mxArray* gg = mxCreateStructMatrix(1, 1, 2, f);
      mxSetField(gg, 0, "angles",       eigenVectorToMx(g.cornerAngles));
      mxSetField(gg, 0, "scaledAngles", eigenVectorToMx(g.cornerScaledAngles));
      mxSetField(s,  0, "corner", gg); }

    if (withOps) {
        int fn = mxAddField(s, "operators");
        mxSetFieldByNumber(s, 0, fn, buildGeometryOperators(h));
    }

    return s;
}

// ── facet builders (representation-grouped data; mirror buildGeometryStruct) ──
// NOTE: field names here are the SINGULAR names of the C++ facet view API
// (embedded()/intrinsic()/extrinsic()), e.g. dualArea/angleSum/cotanWeight/
// dihedralAngle. They intentionally differ from buildGeometryStruct's PLURAL
// names (dualAreas/angleSums/cotanWeights/dihedralAngles) — do NOT align them;
// callers depend on both shapes.

// ── embedded(handle) → struct ─────────────────────────────────
//
// Schema (schemaVersion == 1):
//   E.vertex.position   [nV x 3]  double
//   E.vertex.normal     [nV x 3]  double
//   E.vertex.grid       [nV x 3]  complex double (c = e1+i*e2)
//   E.face.normal       [nF x 3]  double
//   E.face.grid         [nF x 3]  complex double
//   E.face.centroid     [nF x 3]  double

mxArray* buildEmbeddedStruct(ContextHolder& h) {
    nxr::manifold::Manifold& m = *h.ctx;
    auto emb = m.embedded();
    auto vv = emb.vertex();
    auto fv = emb.face();

    const char* topF[] = {"schemaVersion","vertex","face"};
    mxArray* s = mxCreateStructMatrix(1,1,3,topF);
    mxSetField(s,0,"schemaVersion",scalarToMx(1));

    { const char* f[] = {"position","normal","grid"};
      mxArray* g = mxCreateStructMatrix(1,1,3,f);
      mxSetField(g,0,"position", eigenMatrixToMx(vv.position()));
      mxSetField(g,0,"normal",   eigenMatrixToMx(vv.normal()));
      mxSetField(g,0,"grid",     eigenComplexMatrixToMx(vv.grid()));
      mxSetField(s,0,"vertex",g); }
    { const char* f[] = {"normal","grid","centroid"};
      mxArray* g = mxCreateStructMatrix(1,1,3,f);
      mxSetField(g,0,"normal",   eigenMatrixToMx(fv.normal()));
      mxSetField(g,0,"grid",     eigenComplexMatrixToMx(fv.grid()));
      mxSetField(g,0,"centroid", eigenMatrixToMx(fv.centroid()));
      mxSetField(s,0,"face",g); }

    return s;
}

// ── intrinsic(handle) → struct ────────────────────────────────
//
// Schema (schemaVersion == 1):
//   I.vertex.dualArea         [nV x 1]  double
//   I.vertex.angleSum         [nV x 1]  double
//   I.edge.length             [nE x 1]  double
//   I.edge.cotanWeight        [nE x 1]  double
//   I.halfedge.cotanWeight    [nH x 1]  double (real)
//   I.halfedge.transportAlong [nH x 1]  complex double
//   I.halfedge.transportAcross[nH x 1]  complex double

mxArray* buildIntrinsicStruct(ContextHolder& h) {
    nxr::manifold::Manifold& m = *h.ctx;
    auto intr = m.intrinsic();
    auto vv = intr.vertex();
    auto ev = intr.edge();
    auto hv = intr.halfedge();

    const char* topF[] = {"schemaVersion","vertex","edge","halfedge"};
    mxArray* s = mxCreateStructMatrix(1,1,4,topF);
    mxSetField(s,0,"schemaVersion",scalarToMx(1));

    { const char* f[] = {"dualArea","angleSum"};
      mxArray* g = mxCreateStructMatrix(1,1,2,f);
      mxSetField(g,0,"dualArea", eigenVectorToMx(vv.dualArea()));
      mxSetField(g,0,"angleSum", eigenVectorToMx(vv.angleSum()));
      mxSetField(s,0,"vertex",g); }
    { const char* f[] = {"length","cotanWeight"};
      mxArray* g = mxCreateStructMatrix(1,1,2,f);
      mxSetField(g,0,"length",      eigenVectorToMx(ev.length()));
      mxSetField(g,0,"cotanWeight", eigenVectorToMx(ev.cotanWeight()));
      mxSetField(s,0,"edge",g); }
    { const char* f[] = {"cotanWeight","transportAlong","transportAcross"};
      mxArray* g = mxCreateStructMatrix(1,1,3,f);
      mxSetField(g,0,"cotanWeight",     eigenVectorToMx(hv.cotanWeight()));
      mxSetField(g,0,"transportAlong",  eigenComplexVectorToMx(hv.transportAlong()));
      mxSetField(g,0,"transportAcross", eigenComplexVectorToMx(hv.transportAcross()));
      mxSetField(s,0,"halfedge",g); }

    return s;
}

// ── extrinsic(handle) → struct ────────────────────────────────
//
// Schema (schemaVersion == 1):
//   X.vertex.curvature2RoSy  [nV x 1]  complex double (2-RoSy deviatoric)
//   X.vertex.meanCurvature   [nV x 1]  double
//   X.vertex.principalDir    [nV x 3]  double
//   X.edge.dihedralAngle     [nE x 1]  double

mxArray* buildExtrinsicStruct(ContextHolder& h) {
    nxr::manifold::Manifold& m = *h.ctx;
    auto ext = m.extrinsic();
    auto vv = ext.vertex();
    auto ev = ext.edge();

    const char* topF[] = {"schemaVersion","vertex","edge"};
    mxArray* s = mxCreateStructMatrix(1,1,3,topF);
    mxSetField(s,0,"schemaVersion",scalarToMx(1));

    { const char* f[] = {"curvature2RoSy","meanCurvature","principalDir"};
      mxArray* g = mxCreateStructMatrix(1,1,3,f);
      mxSetField(g,0,"curvature2RoSy", eigenComplexVectorToMx(vv.curvature2RoSy()));
      mxSetField(g,0,"meanCurvature",  eigenVectorToMx(vv.meanCurvature()));
      mxSetField(g,0,"principalDir",   eigenMatrixToMx(vv.principalDir()));
      mxSetField(s,0,"vertex",g); }
    { const char* f[] = {"dihedralAngle"};
      mxArray* g = mxCreateStructMatrix(1,1,1,f);
      mxSetField(g,0,"dihedralAngle", eigenVectorToMx(ev.dihedralAngle()));
      mxSetField(s,0,"edge",g); }

    return s;
}

void cmdGeometry(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 2 || nrhs > 3) {
        throw std::invalid_argument(
            "nxr_compute('geometry', handle[, opts]) takes 1 or 2 arguments");
    }
    ContextHolder& h = getHolder(prhs[1]);
    bool withOps = (nrhs >= 3) && readOperatorsFlag(prhs[2]);
    plhs[0] = buildGeometryStruct(h, withOps);
}

// ── topology(handle) → struct ─────────────────────────────────
//
// Returns a nested element-grouped struct of the halfedge mesh combinatorics.
// All indices are 1-based (MATLAB convention); 0 is the sentinel for
// boundary / invalid (geometry-central INVALID_IND → -1 → 0u).
//
// Schema (schemaVersion == 1):
//   T.vertex.count          uint32 scalar
//   T.vertex.halfedge       [nV × 1] uint32  — one canonical halfedge per vertex
//   T.edge.count / .halfedge
//   T.face.count / .halfedge
//   T.corner.count / .halfedge
//   T.halfedge.count        uint32 scalar
//   T.halfedge.twin         [nH × 1] uint32  (interior: 1-based; always valid on closed mesh)
//   T.halfedge.next         [nH × 1] uint32
//   T.halfedge.vertex       [nH × 1] uint32  — tail vertex
//   T.halfedge.edge         [nH × 1] uint32
//   T.halfedge.face         [nH × 1] uint32  (0 for exterior halfedges)
//   T.halfedge.corner       [nH × 1] uint32  (0 for exterior halfedges)
//   T.halfedge.orientation  [nH × 1] logical — true iff he == he.edge().halfedge()
//   T.halfedge.isInterior   [nH × 1] logical

mxArray* buildTopologyStruct(ContextHolder& h, bool withOps = false) {
    auto t = nxr::manifold::geometry::meshTopology(*h.ctx);

    const char* topFields[] = {"schemaVersion","vertex","edge","face","corner","halfedge"};
    mxArray* s = mxCreateStructMatrix(1, 1, 6, topFields);
    mxSetField(s, 0, "schemaVersion", scalarToMx(1));

    { const char* f[] = {"count","halfedge"}; mxArray* g = mxCreateStructMatrix(1,1,2,f);
      mxSetField(g,0,"count",scalarToMx(t.nV)); mxSetField(g,0,"halfedge",indexVectorToMx1Based(t.vertexHalfedge));
      mxSetField(s,0,"vertex",g); }
    { const char* f[] = {"count","halfedge"}; mxArray* g = mxCreateStructMatrix(1,1,2,f);
      mxSetField(g,0,"count",scalarToMx(t.nE)); mxSetField(g,0,"halfedge",indexVectorToMx1Based(t.edgeHalfedge));
      mxSetField(s,0,"edge",g); }
    { const char* f[] = {"count","halfedge"}; mxArray* g = mxCreateStructMatrix(1,1,2,f);
      mxSetField(g,0,"count",scalarToMx(t.nF)); mxSetField(g,0,"halfedge",indexVectorToMx1Based(t.faceHalfedge));
      mxSetField(s,0,"face",g); }
    { const char* f[] = {"count","halfedge"}; mxArray* g = mxCreateStructMatrix(1,1,2,f);
      mxSetField(g,0,"count",scalarToMx(t.nC)); mxSetField(g,0,"halfedge",indexVectorToMx1Based(t.cornerHalfedge));
      mxSetField(s,0,"corner",g); }
    { const char* f[] = {"count","twin","next","vertex","edge","face","corner","orientation","isInterior"};
      mxArray* g = mxCreateStructMatrix(1,1,9,f);
      mxSetField(g,0,"count",scalarToMx(t.nH));
      mxSetField(g,0,"twin",       indexVectorToMx1Based(t.heTwin));
      mxSetField(g,0,"next",       indexVectorToMx1Based(t.heNext));
      mxSetField(g,0,"vertex",     indexVectorToMx1Based(t.heVertex));
      mxSetField(g,0,"edge",       indexVectorToMx1Based(t.heEdge));
      mxSetField(g,0,"face",       indexVectorToMx1Based(t.heFace));
      mxSetField(g,0,"corner",     indexVectorToMx1Based(t.heCorner));
      mxSetField(g,0,"orientation",logicalVectorToMx(t.heOrientation));
      mxSetField(g,0,"isInterior", logicalVectorToMx(t.heIsInterior));
      mxSetField(s,0,"halfedge",g); }

    if (withOps) {
        int fn = mxAddField(s, "operators");
        mxSetFieldByNumber(s, 0, fn, buildTopologyOperators(h));
    }

    return s;
}

void cmdTopology(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 2 || nrhs > 3) {
        throw std::invalid_argument(
            "nxr_compute('topology', handle[, opts]) takes 1 or 2 arguments");
    }
    ContextHolder& h = getHolder(prhs[1]);
    bool withOps = (nrhs >= 3) && readOperatorsFlag(prhs[2]);
    plhs[0] = buildTopologyStruct(h, withOps);
}

// ── bundle(handle, gaugeType[, opts]) → struct ───────────────
//
// Returns a single struct with three sub-structs: Topology, Geometry, Gauge.
// Topology and Geometry are identical to the standalone 'topology' and
// 'geometry' commands; Gauge is identical to the standalone 'gauge' command
// with the given type + opts. All three are built from the same
// ContextHolder (one mesh, consistent indices).
//
//   B = nxr_compute('bundle', h, 'levi-civita')
//   B.Topology   % halfedge combinatorics
//   B.Geometry   % vertex/edge/face geometry quantities
//   B.Gauge      % connection gauge (rotation per vertex)

void cmdBundle(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 3) throw std::invalid_argument(
        "nxr_compute('bundle', handle, gaugeType[, opts]) — gaugeType in {euclidean,levi-civita,trivial}");
    ContextHolder& h = getHolder(prhs[1]);
    std::string type = getStringArg(prhs[2]);
    const mxArray* opts = (nrhs >= 4) ? prhs[3] : nullptr;
    bool withOps = readOperatorsFlag(opts);

    const char* f[] = {"Topology","Geometry","Gauge"};
    mxArray* s = mxCreateStructMatrix(1,1,3,f);
    mxSetField(s,0,"Topology", buildTopologyStruct(h, withOps));
    mxSetField(s,0,"Geometry", buildGeometryStruct(h, withOps));
    mxSetField(s,0,"Gauge",    buildGaugeStruct(h, type, opts, withOps));
    plhs[0] = s;
}

void cmdEmbedded(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 2) throw std::invalid_argument(
        "nxr_compute('embedded', handle) takes 1 argument");
    ContextHolder& h = getHolder(prhs[1]);
    plhs[0] = buildEmbeddedStruct(h);
}

void cmdIntrinsic(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 2) throw std::invalid_argument(
        "nxr_compute('intrinsic', handle) takes 1 argument");
    ContextHolder& h = getHolder(prhs[1]);
    plhs[0] = buildIntrinsicStruct(h);
}

void cmdExtrinsic(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 2) throw std::invalid_argument(
        "nxr_compute('extrinsic', handle) takes 1 argument");
    ContextHolder& h = getHolder(prhs[1]);
    plhs[0] = buildExtrinsicStruct(h);
}

// ── facets(handle, gaugeType[, opts]) → struct ───────────────
//
// Grouped command returning a single struct with five representation-
// grouped sub-structs (data only; no operators path):
//   Fc.Topology   — halfedge combinatorics (same as 'topology' command)
//   Fc.Embedded   — vertex/face positions, normals, grids (same as 'embedded')
//   Fc.Intrinsic  — dual areas, edge lengths, cotan weights, transport (same as 'intrinsic')
//   Fc.Extrinsic  — curvature, principal directions, dihedral angles (same as 'extrinsic')
//   Fc.Gauge      — connection gauge (rotation per vertex) (same as 'gauge' command)
//
// gaugeType in {euclidean, levi-civita, trivial}. For 'trivial', opts must
// carry singVerts (1-based) and singValues (sum == Euler characteristic).
//
// Usage:
//   Fc = nxr_compute('facets', h, 'levi-civita')
//   Fc = nxr_compute('facets', h, 'trivial', struct('singVerts',[1;2],'singValues',[1;1]))

void cmdFacets(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 3) throw std::invalid_argument(
        "nxr_compute('facets', handle, gaugeType[, opts]) — gaugeType in {euclidean,levi-civita,trivial}");
    ContextHolder& h = getHolder(prhs[1]);
    std::string type = getStringArg(prhs[2]);
    const mxArray* opts = (nrhs >= 4) ? prhs[3] : nullptr;

    const char* f[] = {"Topology","Embedded","Intrinsic","Extrinsic","Gauge"};
    mxArray* s = mxCreateStructMatrix(1,1,5,f);
    mxSetField(s,0,"Topology",  buildTopologyStruct(h, false));
    mxSetField(s,0,"Embedded",  buildEmbeddedStruct(h));
    mxSetField(s,0,"Intrinsic", buildIntrinsicStruct(h));
    mxSetField(s,0,"Extrinsic", buildExtrinsicStruct(h));
    mxSetField(s,0,"Gauge",     buildGaugeStruct(h, type, opts, false));
    plhs[0] = s;
}

// ── version() → string ───────────────────────────────────────

void cmdVersion(int /*nlhs*/, mxArray** plhs,
                int /*nrhs*/, const mxArray** /*prhs*/) {
    plhs[0] = mxCreateString("nxr-compute 0.1.0");
}

// ── operators(handle, family[, subtype]) → sparse / struct ───
//
// Exposes the OperatorsFacet operators to MATLAB by name so callers
// can pull a single named operator and run their own eig/eigs on it.
//
//   nxr_compute('operators', h, 'laplacian', 'cotan'|'graph'|'connection'|'covariant')
//   nxr_compute('operators', h, 'mass',      'lumped'|'galerkin')
//   nxr_compute('operators', h, 'hodge',     'h0'|'h1'|'h2'|'h1inv')
//   nxr_compute('operators', h, 'dec')       → struct {d0, d1}
//   nxr_compute('operators', h, 'gradient3D')         % [3E×3N] covariant gradient
//   nxr_compute('operators', h, 'dirac', tau)   % [4V×4V] relative-Dirac family,
//                                               % tau in [0,1] (0=cotan⊗I4, 1=D_N)
//   nxr_compute('operators', h, 'diracFace', tau)  % [4F×4F] FACE-domain (dual)
//                                                  % relative-Dirac, tau in [0,1]
//   nxr_compute('operators', h, 'diracD')       % [4F×4V] first-order Dirac D
//                                               % (DᵀW_F D == dirac(1)); cached
//   nxr_compute('operators', h, 'diracFaceD')   % [4V×4F] first-order face Dirac D̃
//                                               % (D̃ᵀW_V D̃ == diracFace(1)); cached
//   nxr_compute('operators', h, 'diracIntrinsicD') % [4F×4V] first-order INTRINSIC
//                                               % Dirac D_int (immersion/edge-based,
//                                               % spin-connection root); cached
//
// All sparse outputs are native MATLAB sparse (real or complex).
// 'covariant' uses the default Ambient coupling (same as the existing
//   .operators bundle returned by geometry/topology/gauge/bundle commands).

void cmdOperators(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 3)
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            "operators: expected nxr_compute('operators', handle, family[, subtype]).");
    ContextHolder& h = getHolder(prhs[1]);
    auto& m = *h.ctx;                             // Manifold&
    std::string family = getStringArg(prhs[2]);
    // Only parse prhs[3] as a string when it is actually a char array — the
    // 'dirac' and 'diracFace' families take a numeric tau there instead and read
    // prhs[3] directly (they do not use `sub`).
    std::string sub    = (nrhs >= 4 && mxIsChar(prhs[3])) ? getStringArg(prhs[3]) : "";
    auto ops = m.operators();
    namespace cl = nxr::manifold::ops::laplacian::connection;

    if (family == "laplacian") {
        if      (sub == "cotan")
            plhs[0] = eigenSparseToMx(ops.laplacian().cotan());
        else if (sub == "graph")
            plhs[0] = eigenSparseToMx(ops.laplacian().graph());
        else if (sub == "connection")
            plhs[0] = eigenComplexSparseToMx(ops.laplacian().connection());
        else if (sub == "covariant")
            plhs[0] = eigenSparseToMx(
                ops.laplacian().covariant(cl::CovariantCoupling::Ambient));
        else
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators laplacian: subtype must be cotan|graph|connection|covariant.");

    } else if (family == "mass") {
        if      (sub == "lumped")
            plhs[0] = eigenSparseToMx(ops.mass().lumped());
        else if (sub == "galerkin")
            plhs[0] = eigenSparseToMx(ops.mass().galerkin());
        else
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators mass: subtype must be lumped|galerkin.");

    } else if (family == "hodge") {
        if      (sub == "h0")    plhs[0] = eigenSparseToMx(ops.hodge().h0());
        else if (sub == "h1")    plhs[0] = eigenSparseToMx(ops.hodge().h1());
        else if (sub == "h2")    plhs[0] = eigenSparseToMx(ops.hodge().h2());
        else if (sub == "h1inv") plhs[0] = eigenSparseToMx(ops.hodge().h1inv());
        else
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators hodge: subtype must be h0|h1|h2|h1inv.");

    } else if (family == "dec") {
        const auto& dec = ops.dec();
        const char* f[] = {"d0", "d1"};
        mxArray* s = mxCreateStructMatrix(1, 1, 2, f);
        mxSetField(s, 0, "d0", eigenSparseToMx(dec.d0));
        mxSetField(s, 0, "d1", eigenSparseToMx(dec.d1));
        plhs[0] = s;

    } else if (family == "gradient3D") {
        plhs[0] = eigenSparseToMx(m.operators().gradient3D());   // cached on the handle

    } else if (family == "dirac") {
        if (nrhs < 4)
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators dirac: expected a scalar tau, "
                "nxr_compute('operators', h, 'dirac', tau).");
        double tau = getDoubleArg(prhs[3]);   // validates real numeric/logical scalar
        plhs[0] = eigenSparseToMx(m.operators().dirac(tau));   // [4V×4V], caches E on the handle

    } else if (family == "diracFace") {
        if (nrhs < 4)
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators diracFace: expected a scalar tau, "
                "nxr_compute('operators', h, 'diracFace', tau).");
        double tau = getDoubleArg(prhs[3]);
        plhs[0] = eigenSparseToMx(m.operators().diracFace(tau));   // [4F×4F], caches Ẽ

    } else if (family == "diracD") {
        plhs[0] = eigenSparseToMx(m.operators().diracD());   // [4F×4V], cached first-order D

    } else if (family == "diracFaceD") {
        plhs[0] = eigenSparseToMx(m.operators().diracFaceD());   // [4V×4F], cached first-order D̃

    } else if (family == "diracIntrinsicD") {
        plhs[0] = eigenSparseToMx(m.operators().diracIntrinsicD());   // [4F×4V], cached first-order INTRINSIC D_int

    } else {
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            "operators: family must be "
            "laplacian|mass|hodge|dec|gradient3D|dirac|diracFace|diracD|diracFaceD|diracIntrinsicD.");
    }
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
        else if (cmd == "fixDelaunay")   cmdFixDelaunay(nlhs, plhs, nrhs, prhs);
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
        else if (cmd == "topology")                    cmdTopology(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry")                    cmdGeometry(nlhs, plhs, nrhs, prhs);
        else if (cmd == "gauge")                       cmdGauge(nlhs, plhs, nrhs, prhs);
        else if (cmd == "bundle")                      cmdBundle(nlhs, plhs, nrhs, prhs);
        else if (cmd == "operators")                   cmdOperators(nlhs, plhs, nrhs, prhs);
        else if (cmd == "frameTransport")              cmdFrameTransport(nlhs, plhs, nrhs, prhs);
        else if (cmd == "liftToWorld")                 cmdLiftToWorld(nlhs, plhs, nrhs, prhs);
        else if (cmd == "liftToFrame")                 cmdLiftToFrame(nlhs, plhs, nrhs, prhs);
        else if (cmd == "embedded")                    cmdEmbedded(nlhs, plhs, nrhs, prhs);
        else if (cmd == "intrinsic")                   cmdIntrinsic(nlhs, plhs, nrhs, prhs);
        else if (cmd == "extrinsic")                   cmdExtrinsic(nlhs, plhs, nrhs, prhs);
        else if (cmd == "facets")                      cmdFacets(nlhs, plhs, nrhs, prhs);
        else if (cmd == "version")                 cmdVersion(nlhs, plhs, nrhs, prhs);
        else {
            mexErrMsgIdAndTxt("nxr:unknownCommand",
                "Unknown command: \"%s\". Available: create, destroy, assembleManifoldOperators, "
                "solve, normalize (eigenmodes: U,M), fixDelaunay (Delaunay edge-flip mesh repair: V,F), "
                "removeDC, precompute, "
                "parallel, extendScalar, logMap, "
                "findCenter, signedHeat, smoothFace, "
                "smoothVertex, compute, computeFreq, "
                "topology, geometry, gauge, bundle, operators, "
                "frameTransport, liftToWorld, liftToFrame, "
                "embedded, intrinsic, extrinsic, facets, version.",
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
