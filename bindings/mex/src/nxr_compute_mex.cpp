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
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>

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

// ── assembleManifoldOperators(V, F) → struct ─────────────────────

void cmdAssembleMeshOperators(int /*nlhs*/, mxArray** plhs,
                              int nrhs, const mxArray** prhs) {
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
    if (nrhs != 4) {
        throw std::invalid_argument(
            "nxr_compute('solve', K, M, k) takes exactly 3 arguments");
    }
    auto K = mxToEigenSparse(prhs[1]);
    auto M = mxToEigenSparse(prhs[2]);
    int k = getIntArg(prhs[3]);

    // Ctrl-C polling lives entirely in the token; nxr-compute doesn't know about MATLAB.
    auto result = nxr::manifold::solve::eigen(K, M, k, -1e-8,
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
// Each command builds a fresh Manifold + VectorHeatSolver per
// call. MEX is a stateless dispatcher (no holder pattern across
// calls), so the factor cost is paid every invocation — fine for
// MATLAB analysis pipelines where each cell typically does one solve.
// Callers needing repeated solves on the same mesh should batch the
// inputs through the multi-source overloads (transport, extendScalar)
// rather than calling once per source.

void cmdVectorHeatTransport(int /*nlhs*/, mxArray** plhs,
                            int nrhs, const mxArray** prhs) {
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

// ── version() → string ───────────────────────────────────────

void cmdVersion(int /*nlhs*/, mxArray** plhs,
                int /*nrhs*/, const mxArray** /*prhs*/) {
    plhs[0] = mxCreateString("nxr-compute 0.1.0");
}

} // namespace

// ── mexFunction entry point ──────────────────────────────────

void mexFunction(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
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
        if      (cmd == "assembleManifoldOperators")   cmdAssembleMeshOperators(nlhs, plhs, nrhs, prhs);
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
        else if (cmd == "version")                 cmdVersion(nlhs, plhs, nrhs, prhs);
        else {
            mexErrMsgIdAndTxt("nxr:unknownCommand",
                "Unknown command: \"%s\". Available: assembleManifoldOperators, "
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
