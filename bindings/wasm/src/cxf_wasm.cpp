/**
 * cxf_wasm.cpp — Embind bindings exposing cxf to JavaScript via WebAssembly.
 *
 * Mirrors the N-API addon's surface — same compute methods, same stateful
 * ComputeContext pattern, same data shapes — but built around Embind's
 * idioms instead of N-API. Designed for browser apps (three.js, plain JS,
 * frameworks) that consume cxf as a portable compute backend.
 *
 * Conventions:
 *   • Inputs:  JS typed arrays cross the boundary as `emscripten::val`,
 *              copied into Eigen / std::vector via convertJSArrayToNumberVector.
 *   • Outputs: returned as fresh JS typed arrays (.slice() of a heap view)
 *              so they're safe to keep across subsequent compute calls.
 *   • Sparse: returned as JS objects { row, col, data, rows, cols } in COO.
 *   • Structs (HodgeResult, EigenResult, …): returned as JS objects.
 *
 * The class `ContextWrapper` caches MeshOperators, DECOperators,
 * CholeskyCache, and EigenResult on the C++ side — same pattern as the
 * addon's ContextHolder. This means a Hodge solve after a Poisson solve
 * doesn't refactor the cotan Laplacian.
 */

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "cxf/cxf.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using emscripten::val;

namespace {

// ── Conversion helpers: Eigen → JS typed arrays ──────────────
//
// Pattern: build a heap view via typed_memory_view, then .slice() so
// JS owns the copy. The C++ source can safely go out of scope after.

template <typename T>
val toJsArrayCopy(const T* data, std::size_t n) {
    val view = val(emscripten::typed_memory_view(n, data));
    return view.call<val>("slice");
}

val eigenVectorToVal(const Eigen::VectorXd& v) {
    return toJsArrayCopy(v.data(), static_cast<std::size_t>(v.size()));
}

/** Eigen MatrixXd → row-major flat, suitable for V×3 / F×3 attributes
 *  where each row is one (x, y, z) triple — directly consumable as a
 *  three.js BufferAttribute. */
val eigenMatrixToVal(const Eigen::MatrixXd& m) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        rowMajor = m;
    return toJsArrayCopy(rowMajor.data(),
        static_cast<std::size_t>(rowMajor.rows()) * rowMajor.cols());
}

/** Eigen MatrixXd → column-major flat (Eigen's native bytes; no copy
 *  layout change). Used for eigenvectors specifically — for a V×K
 *  eigenmode matrix this puts column k (= mode k) at offset k*V in
 *  flat memory, so JS can extract one mode as a contiguous slice:
 *
 *      const mode_k = eigenvectors.subarray(k * nV, (k + 1) * nV)
 *
 *  Without column-major output, mode extraction would need a strided
 *  copy on the JS side. */
val eigenMatrixToValColumnMajor(const Eigen::MatrixXd& m) {
    return toJsArrayCopy(m.data(),
        static_cast<std::size_t>(m.rows()) * m.cols());
}

val eigenMatrixFloat32ToVal(const Eigen::MatrixXf& m) {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        rowMajor = m;
    return toJsArrayCopy(rowMajor.data(),
        static_cast<std::size_t>(rowMajor.rows()) * rowMajor.cols());
}

val sparseToVal(const Eigen::SparseMatrix<double>& M) {
    int nnz = static_cast<int>(M.nonZeros());
    std::vector<int32_t> rows(nnz), cols(nnz);
    std::vector<double>  vals(nnz);
    int k = 0;
    for (int outer = 0; outer < M.outerSize(); outer++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, outer); it; ++it) {
            rows[k] = static_cast<int32_t>(it.row());
            cols[k] = static_cast<int32_t>(it.col());
            vals[k] = it.value();
            k++;
        }
    }
    val obj = val::object();
    obj.set("row",  toJsArrayCopy(rows.data(), nnz));
    obj.set("col",  toJsArrayCopy(cols.data(), nnz));
    obj.set("data", toJsArrayCopy(vals.data(), nnz));
    obj.set("rows", static_cast<int>(M.rows()));
    obj.set("cols", static_cast<int>(M.cols()));
    obj.set("nnz",  nnz);
    return obj;
}

// ── Conversion helpers: JS typed arrays → Eigen ──────────────

Eigen::VectorXd valToEigenVector(const val& jsArr) {
    auto vec = emscripten::convertJSArrayToNumberVector<double>(jsArr);
    Eigen::VectorXd out(vec.size());
    std::memcpy(out.data(), vec.data(), vec.size() * sizeof(double));
    return out;
}

std::vector<int> valToInt32Vector(const val& jsArr) {
    auto vec = emscripten::convertJSArrayToNumberVector<int>(jsArr);
    return vec;
}

std::vector<double> valToDoubleVector(const val& jsArr) {
    return emscripten::convertJSArrayToNumberVector<double>(jsArr);
}

} // namespace

// ── ComputeContext wrapper class ─────────────────────────────
//
// Stateful: caches operators / DEC / Cholesky / eigenmodes after first
// compute, mirroring the addon's ContextHolder pattern. JS holds an
// Embind handle; calling .delete() on it frees the WASM memory.

class ContextWrapper {
public:
    ContextWrapper(val verticesArr, val facesArr) {
        verts_ = emscripten::convertJSArrayToNumberVector<double>(verticesArr);
        faces_ = emscripten::convertJSArrayToNumberVector<int>(facesArr);

        if (verts_.size() % 3 != 0) {
            throw std::invalid_argument("vertices length must be a multiple of 3");
        }
        if (faces_.size() % 3 != 0) {
            throw std::invalid_argument("faces length must be a multiple of 3");
        }

        int nV = static_cast<int>(verts_.size() / 3);
        int nF = static_cast<int>(faces_.size() / 3);

        // cxf::ComputeContext takes int32_t* for faces; copy into the
        // canonical type since convertJSArrayToNumberVector<int> may
        // produce a different underlying integer width on some platforms.
        faces32_.resize(faces_.size());
        for (std::size_t i = 0; i < faces_.size(); i++) {
            faces32_[i] = static_cast<int32_t>(faces_[i]);
        }

        ctx_ = std::make_unique<cxf::ComputeContext>(
            verts_.data(), nV, faces32_.data(), nF);
        cache_ = std::make_unique<cxf::CholeskyCache>();
    }

    int nV() const { return ctx_->nV(); }
    int nE() const { return ctx_->nE(); }
    int nF() const { return ctx_->nF(); }

    // ── Mesh operators ───────────────────────────────────────

    val assembleMeshOperators() {
        ensureOps();
        return meshOpsToVal();
    }

    val assembleDECOperators() {
        ensureDec();
        return decOpsToVal();
    }

    val computeFaceFrames() {
        auto frames = cxf::computeFaceFrames(*ctx_);
        val obj = val::object();
        obj.set("e1",      eigenMatrixToVal(frames.e1));
        obj.set("e2",      eigenMatrixToVal(frames.e2));
        obj.set("normals", eigenMatrixToVal(frames.normals));
        return obj;
    }

    val computeVertexNormals(int type) {
        cxf::NormalType nt = static_cast<cxf::NormalType>(type);
        Eigen::MatrixXd N = cxf::computeVertexNormals(*ctx_, nt);
        return eigenMatrixToVal(N);
    }

    // ── Spectral basis ───────────────────────────────────────

    val solveEigenmodes(int k, double sigma) {
        ensureOps();
        cxf::EigenResult r = cxf::solveEigenmodes(
            ops_->stiffness, ops_->mass, k, sigma);
        return eigenResultToVal(r);
    }

    val normalizeEigenmodes(val UJsArr, int rows, int cols) {
        ensureOps();
        auto vec = emscripten::convertJSArrayToNumberVector<double>(UJsArr);
        if (static_cast<int>(vec.size()) != rows * cols) {
            throw std::invalid_argument("eigenvectors size mismatch");
        }
        // JS hands us row-major; re-pack into Eigen column-major.
        Eigen::MatrixXd U(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                U(r, c) = vec[r * cols + c];
            }
        }
        Eigen::MatrixXd Un = cxf::normalizeEigenmodes(U, ops_->mass);
        return eigenMatrixToVal(Un);
    }

    val removeDC(val eigStruct) {
        cxf::EigenResult r = valToEigenResult(eigStruct);
        cxf::EigenResult t = cxf::removeDC(r);
        return eigenResultToVal(t);
    }

    /** One-shot precompute: assemble + DEC + eigensolve + normalize +
     *  removeDC + face frames, returned as a single struct. The
     *  visualization-defaults pack from docs/cxf/architecture.md. */
    val precompute(int k, double sigma) {
        ensureOps();
        ensureDec();

        cxf::EigenResult eig = cxf::solveEigenmodes(
            ops_->stiffness, ops_->mass, k, sigma);
        eig.eigenvectors = cxf::normalizeEigenmodes(eig.eigenvectors, ops_->mass);
        eig = cxf::removeDC(eig);
        eigCache_ = std::make_unique<cxf::EigenResult>(eig);

        auto frames = cxf::computeFaceFrames(*ctx_);

        val out = val::object();
        out.set("operators",  meshOpsToVal());
        out.set("dec",        decOpsToVal());
        out.set("eigenmodes", eigenResultToVal(eig));
        val framesVal = val::object();
        framesVal.set("e1",      eigenMatrixToVal(frames.e1));
        framesVal.set("e2",      eigenMatrixToVal(frames.e2));
        framesVal.set("normals", eigenMatrixToVal(frames.normals));
        out.set("faceFrames", framesVal);
        return out;
    }

    // ── Solvers ──────────────────────────────────────────────

    val solvePoisson(val sourceVertsArr, val sourceValuesArr) {
        ensureOps();
        auto verts = emscripten::convertJSArrayToNumberVector<int>(sourceVertsArr);
        auto vals  = emscripten::convertJSArrayToNumberVector<double>(sourceValuesArr);
        if (verts.size() != vals.size()) {
            throw std::invalid_argument("sourceVerts and sourceValues must have the same length");
        }
        std::map<int, double> srcMap;
        for (std::size_t i = 0; i < verts.size(); i++) srcMap[verts[i]] = vals[i];
        Eigen::VectorXd phi = cxf::solvePoisson(*ops_, *cache_, srcMap);
        return eigenVectorToVal(phi);
    }

    val computeGeodesicDistance(val sourceVertsArr) {
        auto sources = emscripten::convertJSArrayToNumberVector<int>(sourceVertsArr);
        Eigen::VectorXd d = cxf::computeGeodesicDistance(*ctx_, sources);
        return eigenVectorToVal(d);
    }

    val tracePath(int vStart, int vEnd) {
        Eigen::MatrixXd path = cxf::tracePath(*ctx_, vStart, vEnd);
        val out = val::object();
        out.set("positions", eigenMatrixToVal(path));
        out.set("nPoints",   static_cast<int>(path.rows()));
        return out;
    }

    val hodgeDecompose(val omegaArr) {
        ensureDec();
        Eigen::VectorXd omega = valToEigenVector(omegaArr);
        cxf::HodgeResult h = cxf::hodgeDecompose(*ctx_, *dec_, *cache_, omega);
        val o = val::object();
        o.set("exactPotential",     eigenVectorToVal(h.exactPotential));
        o.set("coExactPotentialV",  eigenVectorToVal(h.coExactPotentialV));
        o.set("combinedPotential",  eigenVectorToVal(h.combinedPotential));
        o.set("dAlpha",             eigenVectorToVal(h.dAlpha));
        o.set("deltaBeta",          eigenVectorToVal(h.deltaBeta));
        o.set("gamma",              eigenVectorToVal(h.gamma));
        o.set("omegaVectors",       eigenMatrixToVal(h.omegaVectors));
        o.set("dAlphaVectors",      eigenMatrixToVal(h.dAlphaVectors));
        o.set("deltaBetaVectors",   eigenMatrixToVal(h.deltaBetaVectors));
        o.set("gammaVectors",       eigenMatrixToVal(h.gammaVectors));
        return o;
    }

    val computeCurvatures() {
        cxf::CurvatureResult c = cxf::computeCurvatures(*ctx_);
        val o = val::object();
        o.set("gaussian",     eigenVectorToVal(c.gaussian));
        o.set("mean",         eigenVectorToVal(c.mean));
        o.set("kMin",         eigenVectorToVal(c.kMin));
        o.set("kMax",         eigenVectorToVal(c.kMax));
        o.set("principalDir", eigenMatrixToVal(c.principalDirMax));
        return o;
    }

    val computeUVCoordinates() {
        Eigen::MatrixXd uvs = cxf::computeUVCoordinates(*ctx_);
        return eigenMatrixToVal(uvs);
    }

    val computeIsolines(val scalarsArr, int numLevels, double minVal, double maxVal) {
        Eigen::VectorXd scalars = valToEigenVector(scalarsArr);
        cxf::IsolineResult r = cxf::computeIsolines(*ctx_, scalars, numLevels, minVal, maxVal);
        val out = val::object();
        out.set("positions",    eigenMatrixToVal(r.positions));
        out.set("segmentCount", r.segmentCount);
        return out;
    }

    val computeDirectionField(val singVertsArr, val singValuesArr) {
        ensureDec();
        auto verts = emscripten::convertJSArrayToNumberVector<int>(singVertsArr);
        auto vals  = emscripten::convertJSArrayToNumberVector<double>(singValuesArr);
        std::map<int, double> singMap;
        for (std::size_t i = 0; i < verts.size(); i++) singMap[verts[i]] = vals[i];
        cxf::DirectionFieldResult r = cxf::computeDirectionField(*ctx_, *dec_, *cache_, singMap);
        val o = val::object();
        o.set("connections",         eigenVectorToVal(r.connections));
        o.set("directionVectors",    eigenMatrixToVal(r.directionVectors));
        o.set("orthogonalVectors",   eigenMatrixToVal(r.orthogonalVectors));
        o.set("eulerCharacteristic", r.eulerCharacteristic);
        o.set("gaussBonnetSatisfied", r.gaussBonnetSatisfied);
        return o;
    }

    val traceStreamlines(val faceFieldArr, int numSeeds, double stepCoef, int maxSteps) {
        auto vec = emscripten::convertJSArrayToNumberVector<double>(faceFieldArr);
        int nFf = ctx_->nF();
        if (static_cast<int>(vec.size()) != nFf * 3) {
            throw std::invalid_argument("faceField must have length nF*3");
        }
        Eigen::MatrixXd faceField(nFf, 3);
        for (int i = 0; i < nFf; i++) {
            faceField(i, 0) = vec[i * 3 + 0];
            faceField(i, 1) = vec[i * 3 + 1];
            faceField(i, 2) = vec[i * 3 + 2];
        }
        cxf::StreamlineResult r = cxf::traceStreamlines(*ctx_, faceField, numSeeds, stepCoef, maxSteps);
        val o = val::object();
        o.set("positions",    eigenMatrixToVal(r.positions));
        o.set("segmentCount", r.segmentCount);
        return o;
    }

    // ── Vector field ops ────────────────────────────────────

    val whitneyInterpolate(val oneFormArr) {
        ensureDec();
        Eigen::VectorXd omega = valToEigenVector(oneFormArr);
        Eigen::MatrixXd faceVecs = cxf::whitneyInterpolate(*ctx_, *dec_, omega);
        return eigenMatrixToVal(faceVecs);
    }

    val scalarGradient(val scalarArr) {
        Eigen::VectorXd s = valToEigenVector(scalarArr);
        Eigen::MatrixXd grad = cxf::scalarGradient(*ctx_, s);
        return eigenMatrixToVal(grad);
    }

    // ── Time-varying field generators ───────────────────────

    val generateHeatDiffusion(val sourceVertsArr, val sourceValuesArr,
                              val timestepsArr, double alpha) {
        ensureOps();
        if (!eigCache_) {
            throw std::runtime_error(
                "generateHeatDiffusion: eigenmodes not yet computed; "
                "call solveEigenmodes() or precompute() first");
        }
        auto verts = emscripten::convertJSArrayToNumberVector<int>(sourceVertsArr);
        auto vals  = emscripten::convertJSArrayToNumberVector<double>(sourceValuesArr);
        std::map<int, double> sources;
        for (std::size_t i = 0; i < verts.size(); i++) sources[verts[i]] = vals[i];
        Eigen::VectorXd u0 = cxf::generateDelta(ctx_->nV(), sources);
        std::vector<double> ts = valToDoubleVector(timestepsArr);
        Eigen::MatrixXf field = cxf::generateHeatDiffusion(
            *ops_, *eigCache_, u0, ts, alpha);
        val out = val::object();
        out.set("data", eigenMatrixFloat32ToVal(field));
        out.set("T",  static_cast<int>(field.rows()));
        out.set("nV", static_cast<int>(field.cols()));
        return out;
    }

    val generateDampedWave(val modeIndicesArr, val amplitudesArr,
                           val dampingsArr, val phasesArr,
                           val timestepsArr) {
        if (!eigCache_) {
            throw std::runtime_error(
                "generateDampedWave: eigenmodes not yet computed; "
                "call solveEigenmodes() or precompute() first");
        }
        auto mi  = valToInt32Vector(modeIndicesArr);
        auto am  = valToDoubleVector(amplitudesArr);
        auto da  = valToDoubleVector(dampingsArr);
        auto ph  = valToDoubleVector(phasesArr);
        auto ts  = valToDoubleVector(timestepsArr);
        Eigen::MatrixXf field = cxf::generateDampedWave(
            *eigCache_, mi, am, da, ph, ts);
        val out = val::object();
        out.set("data", eigenMatrixFloat32ToVal(field));
        out.set("T",  static_cast<int>(field.rows()));
        out.set("nV", static_cast<int>(field.cols()));
        return out;
    }

    val generateRandomDecomposed1Form(double alphaStrength, double betaStrength,
                                      double gammaStrength, int seed) {
        ensureDec();
        Eigen::VectorXd omega = cxf::generateRandomDecomposed1Form(
            *dec_, ctx_->nV(), ctx_->nE(), ctx_->nF(),
            alphaStrength, betaStrength, gammaStrength,
            static_cast<unsigned int>(seed));
        return eigenVectorToVal(omega);
    }

private:
    // Lazy state (matches the addon's ContextHolder caching).
    void ensureOps() {
        if (!ops_) ops_ = std::make_unique<cxf::MeshOperators>(
            cxf::assembleMeshOperators(*ctx_));
    }
    void ensureDec() {
        if (!dec_) dec_ = std::make_unique<cxf::DECOperators>(
            cxf::assembleDECOperators(*ctx_));
    }

    val meshOpsToVal() {
        ensureOps();
        val o = val::object();
        o.set("stiffness",   sparseToVal(ops_->stiffness));
        o.set("mass",        sparseToVal(ops_->mass));
        o.set("vertexAreas", eigenVectorToVal(ops_->vertexAreas));
        o.set("normals",     eigenMatrixToVal(ops_->normals));
        o.set("totalArea",   ops_->totalArea);
        o.set("nV",          ops_->nV);
        o.set("nE",          ops_->nE);
        o.set("nF",          ops_->nF);
        return o;
    }

    val decOpsToVal() {
        ensureDec();
        val o = val::object();
        o.set("d0",            sparseToVal(dec_->d0));
        o.set("d1",            sparseToVal(dec_->d1));
        o.set("hodge0",        sparseToVal(dec_->hodge0));
        o.set("hodge1",        sparseToVal(dec_->hodge1));
        o.set("hodge2",        sparseToVal(dec_->hodge2));
        o.set("hodge1Inverse", sparseToVal(dec_->hodge1Inverse));
        return o;
    }

    val eigenResultToVal(const cxf::EigenResult& r) {
        val o = val::object();
        // Column-major: each mode (column) is contiguous in the flat
        // buffer at offset k*V — the natural layout for "switch
        // between modes" workflows in three.js consumers.
        o.set("eigenvectors", eigenMatrixToValColumnMajor(r.eigenvectors));
        o.set("eigenvalues",  eigenVectorToVal(r.eigenvalues));
        o.set("k",            r.k);
        o.set("nConverged",   r.nConverged);
        return o;
    }

    cxf::EigenResult valToEigenResult(const val& s) {
        cxf::EigenResult r;
        val uField = s["eigenvectors"];
        val lField = s["eigenvalues"];
        auto uVec = emscripten::convertJSArrayToNumberVector<double>(uField);
        auto lVec = emscripten::convertJSArrayToNumberVector<double>(lField);
        int K = static_cast<int>(lVec.size());
        int V = K > 0 ? static_cast<int>(uVec.size()) / K : 0;
        // Eigenvectors in WASM JS are column-major — same byte layout
        // as Eigen::MatrixXd, just memcpy the bytes back.
        r.eigenvectors.resize(V, K);
        std::memcpy(r.eigenvectors.data(), uVec.data(),
                    V * K * sizeof(double));
        r.eigenvalues.resize(K);
        std::memcpy(r.eigenvalues.data(), lVec.data(), K * sizeof(double));
        r.k = K;
        r.nConverged = K;
        return r;
    }

    std::unique_ptr<cxf::ComputeContext>   ctx_;
    std::unique_ptr<cxf::MeshOperators>    ops_;
    std::unique_ptr<cxf::DECOperators>     dec_;
    std::unique_ptr<cxf::CholeskyCache>    cache_;
    std::unique_ptr<cxf::EigenResult>      eigCache_;
    std::vector<double>  verts_;
    std::vector<int>     faces_;
    std::vector<int32_t> faces32_;
};

// ── Free functions ───────────────────────────────────────────

std::string getVersion() {
    return "cxf 0.1.0";
}

// ── Embind module bindings ───────────────────────────────────

EMSCRIPTEN_BINDINGS(cxf_wasm) {
    emscripten::class_<ContextWrapper>("ComputeContext")
        .constructor<val, val>()
        // Accessors
        .function("nV", &ContextWrapper::nV)
        .function("nE", &ContextWrapper::nE)
        .function("nF", &ContextWrapper::nF)
        // Operators
        .function("assembleMeshOperators",  &ContextWrapper::assembleMeshOperators)
        .function("assembleDECOperators",   &ContextWrapper::assembleDECOperators)
        .function("computeFaceFrames",      &ContextWrapper::computeFaceFrames)
        .function("computeVertexNormals",   &ContextWrapper::computeVertexNormals)
        // Spectral
        .function("solveEigenmodes",        &ContextWrapper::solveEigenmodes)
        .function("normalizeEigenmodes",    &ContextWrapper::normalizeEigenmodes)
        .function("removeDC",               &ContextWrapper::removeDC)
        .function("precompute",             &ContextWrapper::precompute)
        // Solvers
        .function("solvePoisson",           &ContextWrapper::solvePoisson)
        .function("computeGeodesicDistance",&ContextWrapper::computeGeodesicDistance)
        .function("tracePath",              &ContextWrapper::tracePath)
        .function("hodgeDecompose",         &ContextWrapper::hodgeDecompose)
        // Geometric
        .function("computeCurvatures",      &ContextWrapper::computeCurvatures)
        .function("computeUVCoordinates",   &ContextWrapper::computeUVCoordinates)
        .function("computeIsolines",        &ContextWrapper::computeIsolines)
        .function("computeDirectionField",  &ContextWrapper::computeDirectionField)
        .function("traceStreamlines",       &ContextWrapper::traceStreamlines)
        // Vector field
        .function("whitneyInterpolate",     &ContextWrapper::whitneyInterpolate)
        .function("scalarGradient",         &ContextWrapper::scalarGradient)
        // Time-varying generators
        .function("generateHeatDiffusion",  &ContextWrapper::generateHeatDiffusion)
        .function("generateDampedWave",     &ContextWrapper::generateDampedWave)
        .function("generateRandomDecomposed1Form",
                                            &ContextWrapper::generateRandomDecomposed1Form)
        ;

    emscripten::function("version", &getVersion);
}
