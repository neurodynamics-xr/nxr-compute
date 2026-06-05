#pragma once

// ── nxr-compute/mex/marshal — mxArray ↔ Eigen converters ─────────────
//
// Stateless helpers used by the MEX dispatcher in nxr_compute_mex.cpp.
// MATLAB and Eigen both default to column-major dense storage,
// so most dense conversions are zero-shuffle copies.
//
// Two non-trivial cases:
//
//   1. **Vertex / face packed buffers.** MATLAB users pass V as
//      Vx3 double (column-major: x,x,…,y,y,…,z,z,…). nxr-compute's
//      Manifold takes a flat row-major xyz buffer. We
//      repack at the boundary.
//
//   2. **Face indexing.** MATLAB convention is 1-based; nxr-compute uses
//      0-based. mxToFaceBuffer subtracts 1 on conversion.
//
// Sparse matrices: MATLAB sparse is CSC, Eigen SparseMatrix
// defaults to CSC. The index types differ (`mwIndex` vs `int`)
// so we copy through rather than alias.

#include "nxr/compute.h"
#include "mex.h"

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nxr::manifold::mex {

// ── Scalar / string ─────────────────────────────────────────

inline std::string getStringArg(const mxArray* arr) {
    if (!mxIsChar(arr)) {
        throw std::invalid_argument("expected a string argument");
    }
    char buf[256];
    mxGetString(arr, buf, sizeof(buf));
    return std::string(buf);
}

// Accept logical as well as numeric: MATLAB leaves naturally pass flags as
// true/false (e.g. isLoop, alignToCurvature, connectOnSingularities), and a
// MATLAB logical is NOT mxIsNumeric. mxGetScalar yields 0/1 for logicals.
inline int getIntArg(const mxArray* arr) {
    if ((!mxIsNumeric(arr) && !mxIsLogical(arr)) || mxIsComplex(arr) ||
        mxGetNumberOfElements(arr) != 1) {
        throw std::invalid_argument("expected a real scalar");
    }
    return static_cast<int>(mxGetScalar(arr));
}

inline double getDoubleArg(const mxArray* arr) {
    if ((!mxIsNumeric(arr) && !mxIsLogical(arr)) || mxIsComplex(arr) ||
        mxGetNumberOfElements(arr) != 1) {
        throw std::invalid_argument("expected a real scalar");
    }
    return mxGetScalar(arr);
}

// ── Dense double matrix → Eigen ─────────────────────────────
// MATLAB and Eigen both column-major. Direct memcpy.

inline Eigen::MatrixXd mxToEigenMatrix(const mxArray* arr) {
    if (!mxIsDouble(arr) || mxIsComplex(arr) || mxIsSparse(arr)) {
        throw std::invalid_argument("expected real double dense matrix");
    }
    int rows = static_cast<int>(mxGetM(arr));
    int cols = static_cast<int>(mxGetN(arr));
    Eigen::MatrixXd out(rows, cols);
    std::memcpy(out.data(), mxGetPr(arr),
                static_cast<std::size_t>(rows) * cols * sizeof(double));
    return out;
}

inline Eigen::VectorXd mxToEigenVector(const mxArray* arr) {
    if (!mxIsDouble(arr) || mxIsComplex(arr) || mxIsSparse(arr)) {
        throw std::invalid_argument("expected real double dense vector");
    }
    std::size_t n = mxGetNumberOfElements(arr);
    Eigen::VectorXd out(n);
    std::memcpy(out.data(), mxGetPr(arr), n * sizeof(double));
    return out;
}

// ── Eigen → dense MATLAB ────────────────────────────────────

inline mxArray* eigenMatrixToMx(const Eigen::MatrixXd& m) {
    mxArray* arr = mxCreateDoubleMatrix(m.rows(), m.cols(), mxREAL);
    std::memcpy(mxGetPr(arr), m.data(),
                static_cast<std::size_t>(m.rows()) * m.cols() * sizeof(double));
    return arr;
}

inline mxArray* eigenVectorToMx(const Eigen::VectorXd& v) {
    // Return as column vector (Mx1).
    mxArray* arr = mxCreateDoubleMatrix(v.size(), 1, mxREAL);
    std::memcpy(mxGetPr(arr), v.data(),
                static_cast<std::size_t>(v.size()) * sizeof(double));
    return arr;
}

// ── Sparse double matrix conversion (CSC ↔ CSC) ─────────────

inline Eigen::SparseMatrix<double> mxToEigenSparse(const mxArray* arr) {
    if (!mxIsSparse(arr) || !mxIsDouble(arr) || mxIsComplex(arr)) {
        throw std::invalid_argument("expected real double sparse matrix");
    }
    int rows = static_cast<int>(mxGetM(arr));
    int cols = static_cast<int>(mxGetN(arr));
    const mwIndex* jc = mxGetJc(arr);   // column pointers, length cols+1
    const mwIndex* ir = mxGetIr(arr);   // row indices,  length nnz
    const double*  pr = mxGetPr(arr);   // values,       length nnz
    int nnz = static_cast<int>(jc[cols]);

    // Build via triplets — robust across Eigen versions and MATLAB
    // sparse layouts. The constant per-nnz overhead is fine for the
    // matrices we ship through MEX (cotan Laplacian on fsaverage6
    // is ~300k nnz; triplet build is ~10 ms).
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(nnz);
    for (int c = 0; c < cols; c++) {
        for (mwIndex k = jc[c]; k < jc[c + 1]; k++) {
            trips.emplace_back(static_cast<int>(ir[k]), c, pr[k]);
        }
    }
    Eigen::SparseMatrix<double> M(rows, cols);
    M.setFromTriplets(trips.begin(), trips.end());
    M.makeCompressed();
    return M;
}

inline mxArray* eigenSparseToMx(const Eigen::SparseMatrix<double>& src) {
    Eigen::SparseMatrix<double> m = src;  // ensure CSC + compressed
    m.makeCompressed();
    int rows = static_cast<int>(m.rows());
    int cols = static_cast<int>(m.cols());
    int nnz  = static_cast<int>(m.nonZeros());

    mxArray* arr = mxCreateSparse(rows, cols, nnz, mxREAL);
    mwIndex* jc = mxGetJc(arr);
    mwIndex* ir = mxGetIr(arr);
    double*  pr = mxGetPr(arr);

    // Eigen compressed CSC: outerIndexPtr() = col pointers (cols+1),
    // innerIndexPtr() = row indices (nnz), valuePtr() = values (nnz).
    for (int c = 0; c <= cols; c++) {
        jc[c] = static_cast<mwIndex>(m.outerIndexPtr()[c]);
    }
    for (int k = 0; k < nnz; k++) {
        ir[k] = static_cast<mwIndex>(m.innerIndexPtr()[k]);
        pr[k] = m.valuePtr()[k];
    }
    return arr;
}

// ── Vertex index lists (1-based MATLAB → 0-based C) ─────────
//
// Source-vertex arguments to vector heat / signed heat / find center
// are 1-based in MATLAB convention; we subtract 1 at the boundary to
// match the rest of nxr-compute. Accepts double, int32, or uint32 for
// caller flexibility.

inline std::vector<int> mxToVertexIndices(const mxArray* arr) {
    if (mxIsComplex(arr) || mxIsSparse(arr)) {
        throw std::invalid_argument("vertex indices must be a real dense vector");
    }
    std::size_t n = mxGetNumberOfElements(arr);
    std::vector<int> out(n);
    if (mxIsDouble(arr)) {
        const double* src = mxGetPr(arr);
        for (std::size_t i = 0; i < n; i++) out[i] = static_cast<int>(src[i]) - 1;
    } else if (mxIsInt32(arr)) {
        const std::int32_t* src = static_cast<const std::int32_t*>(mxGetData(arr));
        for (std::size_t i = 0; i < n; i++) out[i] = src[i] - 1;
    } else if (mxIsUint32(arr)) {
        const std::uint32_t* src = static_cast<const std::uint32_t*>(mxGetData(arr));
        for (std::size_t i = 0; i < n; i++) out[i] = static_cast<int>(src[i]) - 1;
    } else {
        throw std::invalid_argument("vertex indices must be double, int32, or uint32");
    }
    return out;
}

// ── Vertex / face buffer marshallers ────────────────────────
//
// MATLAB convention: vertices Vx3, faces Fx3 (1-based).
// nxr-compute convention: row-major flat xyz triples for vertices,
// row-major flat int32_t for faces (0-based).

inline std::vector<double> mxToVertexBuffer(const mxArray* arr, int& nVOut) {
    if (!mxIsDouble(arr) || mxIsComplex(arr) || mxIsSparse(arr)) {
        throw std::invalid_argument("vertices must be a real double matrix");
    }
    int rows = static_cast<int>(mxGetM(arr));
    int cols = static_cast<int>(mxGetN(arr));
    if (cols != 3) {
        throw std::invalid_argument("vertices must be Vx3 (got cols=" + std::to_string(cols) + ")");
    }
    const double* src = mxGetPr(arr);  // column-major: x col, y col, z col
    nVOut = rows;
    std::vector<double> out(static_cast<std::size_t>(rows) * 3);
    for (int r = 0; r < rows; r++) {
        out[3 * r + 0] = src[r + 0 * rows];
        out[3 * r + 1] = src[r + 1 * rows];
        out[3 * r + 2] = src[r + 2 * rows];
    }
    return out;
}

inline std::vector<std::int32_t> mxToFaceBuffer(const mxArray* arr, int& nFOut) {
    if (mxIsComplex(arr) || mxIsSparse(arr)) {
        throw std::invalid_argument("faces must be a real dense matrix");
    }
    int rows = static_cast<int>(mxGetM(arr));
    int cols = static_cast<int>(mxGetN(arr));
    if (cols != 3) {
        throw std::invalid_argument("faces must be Fx3 (got cols=" + std::to_string(cols) + ")");
    }
    nFOut = rows;
    std::vector<std::int32_t> out(static_cast<std::size_t>(rows) * 3);

    // Accept double, int32, or uint32 to be tolerant of how callers
    // store face indices in MATLAB. Convert 1-based → 0-based.
    if (mxIsDouble(arr)) {
        const double* src = mxGetPr(arr);
        for (int r = 0; r < rows; r++) {
            out[3 * r + 0] = static_cast<std::int32_t>(src[r + 0 * rows]) - 1;
            out[3 * r + 1] = static_cast<std::int32_t>(src[r + 1 * rows]) - 1;
            out[3 * r + 2] = static_cast<std::int32_t>(src[r + 2 * rows]) - 1;
        }
    } else if (mxIsInt32(arr)) {
        const std::int32_t* src = static_cast<const std::int32_t*>(mxGetData(arr));
        for (int r = 0; r < rows; r++) {
            out[3 * r + 0] = src[r + 0 * rows] - 1;
            out[3 * r + 1] = src[r + 1 * rows] - 1;
            out[3 * r + 2] = src[r + 2 * rows] - 1;
        }
    } else if (mxIsUint32(arr)) {
        const std::uint32_t* src = static_cast<const std::uint32_t*>(mxGetData(arr));
        for (int r = 0; r < rows; r++) {
            out[3 * r + 0] = static_cast<std::int32_t>(src[r + 0 * rows]) - 1;
            out[3 * r + 1] = static_cast<std::int32_t>(src[r + 1 * rows]) - 1;
            out[3 * r + 2] = static_cast<std::int32_t>(src[r + 2 * rows]) - 1;
        }
    } else {
        throw std::invalid_argument("faces must be double, int32, or uint32");
    }
    return out;
}

// ── Struct builders for compound return values ──────────────

inline mxArray* meshOperatorsToStruct(const nxr::manifold::ops::ManifoldOperators& ops,
                                      int nV, int nE, int nF) {
    const char* fields[] = {
        "cotanLaplacian", "mass", "vertexDualAreas", "vertexNormals",
        "totalArea", "nV", "nE", "nF",
    };
    mxArray* s = mxCreateStructMatrix(1, 1, 8, fields);

    // ops.vertexNormals is [nV, 3] row-major; MATLAB takes col-major so copy through.
    Eigen::MatrixXd normalsT = ops.vertexNormals;  // already (nV, 3)

    mxSetField(s, 0, "cotanLaplacian",  eigenSparseToMx(ops.cotanLaplacian));
    mxSetField(s, 0, "mass",            eigenSparseToMx(ops.mass));
    mxSetField(s, 0, "vertexDualAreas", eigenVectorToMx(ops.vertexDualAreas));
    mxSetField(s, 0, "vertexNormals",   eigenMatrixToMx(normalsT));
    mxSetField(s, 0, "totalArea",       mxCreateDoubleScalar(ops.totalArea));
    mxSetField(s, 0, "nV",              mxCreateDoubleScalar(nV));
    mxSetField(s, 0, "nE",              mxCreateDoubleScalar(nE));
    mxSetField(s, 0, "nF",              mxCreateDoubleScalar(nF));
    return s;
}

inline mxArray* eigenResultToStruct(const nxr::manifold::solve::EigenResult& r) {
    const char* fields[] = {"eigenvectors", "eigenvalues", "k", "nConverged"};
    mxArray* s = mxCreateStructMatrix(1, 1, 4, fields);
    mxSetField(s, 0, "eigenvectors", eigenMatrixToMx(r.eigenvectors));
    mxSetField(s, 0, "eigenvalues",  eigenVectorToMx(r.eigenvalues));
    mxSetField(s, 0, "k",            mxCreateDoubleScalar(r.k));
    mxSetField(s, 0, "nConverged",   mxCreateDoubleScalar(r.nConverged));
    return s;
}

/** Read back an EigenResult struct previously returned by
 *  eigenResultToStruct. Used by removeDC, which takes the result
 *  of solve and returns a trimmed copy. */
inline nxr::manifold::solve::EigenResult mxToEigenResult(const mxArray* s) {
    if (!mxIsStruct(s)) {
        throw std::invalid_argument("expected an EigenResult struct");
    }
    nxr::manifold::solve::EigenResult r;
    mxArray* uField = mxGetField(s, 0, "eigenvectors");
    mxArray* lField = mxGetField(s, 0, "eigenvalues");
    if (!uField || !lField) {
        throw std::invalid_argument("EigenResult struct missing eigenvectors / eigenvalues");
    }
    r.eigenvectors = mxToEigenMatrix(uField);
    r.eigenvalues  = mxToEigenVector(lField);
    r.k = static_cast<int>(r.eigenvalues.size());
    r.nConverged = r.k;
    return r;
}

// ── Parity return-struct builders (handle-mode ops) ─────────
//
// MEX is exempt from the §11 row-major flatten rule: dense → column-major
// mxArray, sparse → CSC, so these builders just forward to the existing
// eigen*ToMx helpers. Parity with WASM is numerical, not byte-layout.

inline mxArray* decOperatorsToStruct(const nxr::manifold::ops::DECOperators& dec) {
    const char* fields[] = {"d0", "d1", "hodge0", "hodge1", "hodge2", "hodge1Inverse"};
    mxArray* s = mxCreateStructMatrix(1, 1, 6, fields);
    mxSetField(s, 0, "d0",            eigenSparseToMx(dec.d0));
    mxSetField(s, 0, "d1",            eigenSparseToMx(dec.d1));
    mxSetField(s, 0, "hodge0",        eigenSparseToMx(dec.hodge0));
    mxSetField(s, 0, "hodge1",        eigenSparseToMx(dec.hodge1));
    mxSetField(s, 0, "hodge2",        eigenSparseToMx(dec.hodge2));
    mxSetField(s, 0, "hodge1Inverse", eigenSparseToMx(dec.hodge1Inverse));
    return s;
}

inline mxArray* connectionLaplacianToStruct(
    const nxr::manifold::ops::laplacian::connection::ConnectionLaplacian& cl) {
    namespace ns = nxr::manifold::ops::laplacian::connection;
    const char* fields[] = {"K_real", "K_imag", "frameE1", "frameE2",
                            "baseDim", "outputDim",
                            "nSym", "regularization", "domain", "format"};
    mxArray* s = mxCreateStructMatrix(1, 1, 10, fields);

    if (cl.format == ns::ConnectionLaplacianFormat::Real2N) {
        mxSetField(s, 0, "K_real", eigenSparseToMx(cl.K_real));
        Eigen::SparseMatrix<double> empty(cl.outputDim, cl.outputDim);
        mxSetField(s, 0, "K_imag", eigenSparseToMx(empty));
    } else {
        // Complex Hermitian → parallel real/imag real-sparse (mirrors WASM).
        Eigen::SparseMatrix<double> re(cl.K_complex.rows(), cl.K_complex.cols());
        Eigen::SparseMatrix<double> im(cl.K_complex.rows(), cl.K_complex.cols());
        std::vector<Eigen::Triplet<double>> tr, ti;
        for (int k = 0; k < cl.K_complex.outerSize(); ++k) {
            for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(cl.K_complex, k); it; ++it) {
                tr.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()), it.value().real());
                ti.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()), it.value().imag());
            }
        }
        re.setFromTriplets(tr.begin(), tr.end());
        im.setFromTriplets(ti.begin(), ti.end());
        mxSetField(s, 0, "K_real", eigenSparseToMx(re));
        mxSetField(s, 0, "K_imag", eigenSparseToMx(im));
    }
    mxSetField(s, 0, "frameE1",        eigenMatrixToMx(cl.frameE1));
    mxSetField(s, 0, "frameE2",        eigenMatrixToMx(cl.frameE2));
    mxSetField(s, 0, "baseDim",        mxCreateDoubleScalar(cl.baseDim));
    mxSetField(s, 0, "outputDim",      mxCreateDoubleScalar(cl.outputDim));
    mxSetField(s, 0, "nSym",           mxCreateDoubleScalar(cl.nSym));
    mxSetField(s, 0, "regularization", mxCreateDoubleScalar(cl.regularization));
    mxSetField(s, 0, "domain",         mxCreateDoubleScalar(static_cast<double>(cl.domain)));
    mxSetField(s, 0, "format",         mxCreateDoubleScalar(static_cast<double>(cl.format)));
    return s;
}

inline mxArray* faceFramesToStruct(const nxr::manifold::geometry::FaceFrames& f) {
    const char* fields[] = {"e1", "e2", "normals"};
    mxArray* s = mxCreateStructMatrix(1, 1, 3, fields);
    mxSetField(s, 0, "e1",      eigenMatrixToMx(f.e1));
    mxSetField(s, 0, "e2",      eigenMatrixToMx(f.e2));
    mxSetField(s, 0, "normals", eigenMatrixToMx(f.normals));
    return s;
}

inline mxArray* vertexFramesToStruct(const nxr::manifold::geometry::VertexFrames& f) {
    const char* fields[] = {"e1", "e2", "normals"};
    mxArray* s = mxCreateStructMatrix(1, 1, 3, fields);
    mxSetField(s, 0, "e1",      eigenMatrixToMx(f.e1));
    mxSetField(s, 0, "e2",      eigenMatrixToMx(f.e2));
    mxSetField(s, 0, "normals", eigenMatrixToMx(f.normals));
    return s;
}

inline mxArray* hodgeResultToStruct(const nxr::manifold::solve::HodgeResult& r) {
    const char* fields[] = {
        "exactPotential", "coExactPotentialF", "coExactPotentialV",
        "combinedPotential", "omega", "dAlpha", "deltaBeta", "gamma",
        "omegaVectors", "dAlphaVectors", "deltaBetaVectors", "gammaVectors"};
    mxArray* s = mxCreateStructMatrix(1, 1, 12, fields);
    mxSetField(s, 0, "exactPotential",    eigenVectorToMx(r.exactPotential));
    mxSetField(s, 0, "coExactPotentialF", eigenVectorToMx(r.coExactPotentialF));
    mxSetField(s, 0, "coExactPotentialV", eigenVectorToMx(r.coExactPotentialV));
    mxSetField(s, 0, "combinedPotential", eigenVectorToMx(r.combinedPotential));
    mxSetField(s, 0, "omega",             eigenVectorToMx(r.omega));
    mxSetField(s, 0, "dAlpha",            eigenVectorToMx(r.dAlpha));
    mxSetField(s, 0, "deltaBeta",         eigenVectorToMx(r.deltaBeta));
    mxSetField(s, 0, "gamma",             eigenVectorToMx(r.gamma));
    mxSetField(s, 0, "omegaVectors",      eigenMatrixToMx(r.omegaVectors));
    mxSetField(s, 0, "dAlphaVectors",     eigenMatrixToMx(r.dAlphaVectors));
    mxSetField(s, 0, "deltaBetaVectors",  eigenMatrixToMx(r.deltaBetaVectors));
    mxSetField(s, 0, "gammaVectors",      eigenMatrixToMx(r.gammaVectors));
    return s;
}

inline mxArray* curvatureResultToStruct(const nxr::manifold::geometry::CurvatureResult& r) {
    const char* fields[] = {"gaussian", "mean", "kMin", "kMax", "principalDirMax"};
    mxArray* s = mxCreateStructMatrix(1, 1, 5, fields);
    mxSetField(s, 0, "gaussian",        eigenVectorToMx(r.gaussian));
    mxSetField(s, 0, "mean",            eigenVectorToMx(r.mean));
    mxSetField(s, 0, "kMin",            eigenVectorToMx(r.kMin));
    mxSetField(s, 0, "kMax",            eigenVectorToMx(r.kMax));
    mxSetField(s, 0, "principalDirMax", eigenMatrixToMx(r.principalDirMax));
    return s;
}

// Shared by isoline / streamline (both are positions [2*segs, 3] + count).
inline mxArray* positionsSegmentsToStruct(const Eigen::MatrixXd& positions, int segmentCount) {
    const char* fields[] = {"positions", "segmentCount"};
    mxArray* s = mxCreateStructMatrix(1, 1, 2, fields);
    mxSetField(s, 0, "positions",    eigenMatrixToMx(positions));
    mxSetField(s, 0, "segmentCount", mxCreateDoubleScalar(segmentCount));
    return s;
}

inline mxArray* directionFieldResultToStruct(
    const nxr::manifold::connection::DirectionFieldResult& r) {
    const char* fields[] = {"connections", "directionVectors", "orthogonalVectors",
                            "vertexVectors", "vertexOrthogonalVectors",
                            "eulerCharacteristic", "gaussBonnetSatisfied"};
    mxArray* s = mxCreateStructMatrix(1, 1, 7, fields);
    mxSetField(s, 0, "connections",              eigenVectorToMx(r.connections));
    mxSetField(s, 0, "directionVectors",         eigenMatrixToMx(r.directionVectors));
    mxSetField(s, 0, "orthogonalVectors",        eigenMatrixToMx(r.orthogonalVectors));
    mxSetField(s, 0, "vertexVectors",            eigenMatrixToMx(r.vertexVectors));
    mxSetField(s, 0, "vertexOrthogonalVectors",  eigenMatrixToMx(r.vertexOrthogonalVectors));
    mxSetField(s, 0, "eulerCharacteristic",      mxCreateDoubleScalar(r.eulerCharacteristic));
    mxSetField(s, 0, "gaussBonnetSatisfied",     mxCreateLogicalScalar(r.gaussBonnetSatisfied));
    return s;
}

// MatrixXf [T, n] → double mxArray [T, n]. Both column-major; cast float→double.
inline mxArray* eigenMatrixXfToMx(const Eigen::MatrixXf& m) {
    mxArray* arr = mxCreateDoubleMatrix(m.rows(), m.cols(), mxREAL);
    double* dst = mxGetPr(arr);
    const float* src = m.data();
    std::size_t n = static_cast<std::size_t>(m.rows()) * m.cols();
    for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(src[i]);
    return arr;
}

} // namespace nxr::manifold::mex
