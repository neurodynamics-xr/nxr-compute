/**
 * cxf_mex.cpp — MATLAB MEX dispatcher around the cxf library.
 *
 * Single mexFunction with command dispatch (gpjs / libigl-matlab
 * style). One artifact (cxf.mexw64) consumed by Brainstorm, SPM,
 * or any MATLAB analysis pipeline.
 *
 * Usage from MATLAB:
 *   ops    = cxf('assembleMeshOperators', V, F);
 *   eig    = cxf('solveEigenmodes', ops.stiffness, ops.mass, k);
 *   eig.eigenvectors = cxf('normalizeEigenmodes', eig.eigenvectors, ops.mass);
 *   eig    = cxf('removeDC', eig);
 *   result = cxf('precompute', V, F, k);   % shorthand for the whole pipeline
 *
 * V is Vx3 double; F is Fx3 (double / int32 / uint32) with 1-based
 * indices (MATLAB convention). All sparse outputs are CSC double.
 */

#include "cxf/cxf.h"
#include "marshal.h"
#include "mex.h"

#include <cstring>
#include <stdexcept>
#include <string>

using namespace cxf::mex;

namespace {

// ── assembleMeshOperators(V, F) → struct ─────────────────────

void cmdAssembleMeshOperators(int /*nlhs*/, mxArray** plhs,
                              int nrhs, const mxArray** prhs) {
    if (nrhs != 3) {
        throw std::invalid_argument(
            "cxf('assembleMeshOperators', V, F) takes exactly 2 arguments");
    }
    int nV = 0, nF = 0;
    auto verts = mxToVertexBuffer(prhs[1], nV);
    auto faces = mxToFaceBuffer(prhs[2], nF);

    cxf::ComputeContext ctx(verts.data(), nV, faces.data(), nF);
    auto ops = cxf::assembleMeshOperators(ctx);
    plhs[0] = meshOperatorsToStruct(ops);
}

// ── solveEigenmodes(K, M, k) → struct ────────────────────────

void cmdSolveEigenmodes(int /*nlhs*/, mxArray** plhs,
                        int nrhs, const mxArray** prhs) {
    if (nrhs != 4) {
        throw std::invalid_argument(
            "cxf('solveEigenmodes', K, M, k) takes exactly 3 arguments");
    }
    auto K = mxToEigenSparse(prhs[1]);
    auto M = mxToEigenSparse(prhs[2]);
    int k = getIntArg(prhs[3]);

    auto result = cxf::solveEigenmodes(K, M, k);
    plhs[0] = eigenResultToStruct(result);
}

// ── normalizeEigenmodes(U, M) → U ────────────────────────────

void cmdNormalizeEigenmodes(int /*nlhs*/, mxArray** plhs,
                            int nrhs, const mxArray** prhs) {
    if (nrhs != 3) {
        throw std::invalid_argument(
            "cxf('normalizeEigenmodes', U, M) takes exactly 2 arguments");
    }
    auto U = mxToEigenMatrix(prhs[1]);
    auto M = mxToEigenSparse(prhs[2]);

    auto Un = cxf::normalizeEigenmodes(U, M);
    plhs[0] = eigenMatrixToMx(Un);
}

// ── removeDC(eigStruct) → eigStruct ──────────────────────────

void cmdRemoveDC(int /*nlhs*/, mxArray** plhs,
                 int nrhs, const mxArray** prhs) {
    if (nrhs != 2) {
        throw std::invalid_argument(
            "cxf('removeDC', eig) takes exactly 1 argument");
    }
    auto eig = mxToEigenResult(prhs[1]);
    auto trimmed = cxf::removeDC(eig);
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
            "cxf('precompute', V, F, k) takes exactly 3 arguments");
    }
    int nV = 0, nF = 0;
    auto verts = mxToVertexBuffer(prhs[1], nV);
    auto faces = mxToFaceBuffer(prhs[2], nF);
    int k = getIntArg(prhs[3]);

    cxf::ComputeContext ctx(verts.data(), nV, faces.data(), nF);
    auto ops = cxf::assembleMeshOperators(ctx);
    auto eig = cxf::solveEigenmodes(ops.stiffness, ops.mass, k);
    eig.eigenvectors = cxf::normalizeEigenmodes(eig.eigenvectors, ops.mass);
    eig = cxf::removeDC(eig);

    plhs[0] = eigenResultToStruct(eig);
}

// ── version() → string ───────────────────────────────────────

void cmdVersion(int /*nlhs*/, mxArray** plhs,
                int /*nrhs*/, const mxArray** /*prhs*/) {
    plhs[0] = mxCreateString("cxf 0.1.0");
}

} // namespace

// ── mexFunction entry point ──────────────────────────────────

void mexFunction(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 1 || !mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("cxf:nrhs",
            "First argument must be a command string. Try cxf('version').");
    }

    std::string cmd;
    try {
        cmd = getStringArg(prhs[0]);
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("cxf:badCommand", "%s", e.what());
        return;
    }

    try {
        if      (cmd == "assembleMeshOperators") cmdAssembleMeshOperators(nlhs, plhs, nrhs, prhs);
        else if (cmd == "solveEigenmodes")       cmdSolveEigenmodes(nlhs, plhs, nrhs, prhs);
        else if (cmd == "normalizeEigenmodes")   cmdNormalizeEigenmodes(nlhs, plhs, nrhs, prhs);
        else if (cmd == "removeDC")              cmdRemoveDC(nlhs, plhs, nrhs, prhs);
        else if (cmd == "precompute")            cmdPrecompute(nlhs, plhs, nrhs, prhs);
        else if (cmd == "version")               cmdVersion(nlhs, plhs, nrhs, prhs);
        else {
            mexErrMsgIdAndTxt("cxf:unknownCommand",
                "Unknown command: \"%s\". Available: assembleMeshOperators, "
                "solveEigenmodes, normalizeEigenmodes, removeDC, precompute, version.",
                cmd.c_str());
        }
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("cxf:error", "cxf('%s') failed: %s", cmd.c_str(), e.what());
    }
}
