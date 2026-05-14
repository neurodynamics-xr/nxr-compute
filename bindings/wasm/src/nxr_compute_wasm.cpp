/**
 * nxr_compute_wasm.cpp — Embind bindings exposing nxr-compute to JavaScript via WebAssembly.
 *
 * Mirrors the N-API addon's surface — same compute methods, same stateful
 * ComputeContext pattern, same data shapes — but built around Embind's
 * idioms instead of N-API. Designed for browser apps (three.js, plain JS,
 * frameworks) that consume nxr-compute as a portable compute backend.
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

#include "nxr/compute.h"

#include <atomic>
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

// Per nxr-compute.h's hard layout rule (vMajor for eigenvectors), all
// V×K matrices flatten row-major into JS just like V×3 / F×3.
// Use eigenMatrixToVal above. The previous column-major helper
// was removed in Phase A — the Zarr schema and renderer both
// consume vMajor (U[v*K + k]).

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

// Complex sparse → COO with parallel real/imag arrays.
// Used by the connection-Laplacian "complex" output format.
val sparseComplexToVal(const Eigen::SparseMatrix<std::complex<double>>& M) {
    int nnz = static_cast<int>(M.nonZeros());
    std::vector<int32_t> rows(nnz), cols(nnz);
    std::vector<double>  re(nnz), im(nnz);
    int k = 0;
    for (int outer = 0; outer < M.outerSize(); outer++) {
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(M, outer); it; ++it) {
            rows[k] = static_cast<int32_t>(it.row());
            cols[k] = static_cast<int32_t>(it.col());
            re[k]   = it.value().real();
            im[k]   = it.value().imag();
            k++;
        }
    }
    val obj = val::object();
    obj.set("row",       toJsArrayCopy(rows.data(), nnz));
    obj.set("col",       toJsArrayCopy(cols.data(), nnz));
    obj.set("realData",  toJsArrayCopy(re.data(), nnz));
    obj.set("imagData",  toJsArrayCopy(im.data(), nnz));
    obj.set("rows",      static_cast<int>(M.rows()));
    obj.set("cols",      static_cast<int>(M.cols()));
    obj.set("nnz",       nnz);
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

        // nxr::compute::ComputeContext takes int32_t* for faces; copy into the
        // canonical type since convertJSArrayToNumberVector<int> may
        // produce a different underlying integer width on some platforms.
        faces32_.resize(faces_.size());
        for (std::size_t i = 0; i < faces_.size(); i++) {
            faces32_[i] = static_cast<int32_t>(faces_[i]);
        }

        ctx_ = std::make_unique<nxr::compute::ComputeContext>(
            verts_.data(), nV, faces32_.data(), nF);
        cache_ = std::make_unique<nxr::compute::CholeskyCache>();
    }

    int nV() const { return ctx_->nV(); }
    int nE() const { return ctx_->nE(); }
    int nF() const { return ctx_->nF(); }

    // ── Mesh operators ───────────────────────────────────────

    // Optional `variantName` accepts "voronoi" (default), "barycentric",
    // or "full". Empty string → default. Switching variants invalidates
    // the cached MeshOperators (and any cached factor that depended on
    // the previous mass).
    val assembleMeshOperators(const std::string& variantName) {
        nxr::compute::MassMatrixVariant variant = variantName.empty()
            ? nxr::compute::MassMatrixVariant::Voronoi
            : nxr::compute::parseMassMatrixVariant(variantName);
        if (!ops_ || ops_->massVariant != variant) {
            ops_ = std::make_unique<nxr::compute::MeshOperators>(
                nxr::compute::assembleMeshOperators(*ctx_, variant));
        }
        return meshOpsToVal();
    }

    val assembleDECOperators() {
        ensureDec();
        return decOpsToVal();
    }

    // Connection Laplacian on the chosen domain (vertex / face / edge).
    // Result-level cache keyed by (domain, nSym, regularization, format)
    // — same pattern as computeSmoothFaceField (CLAUDE.md rule 9).
    //
    // `opts` is a JS object: { domain?, nSym?, regularization?, format? }.
    // Defaults: { domain: 'vertex', nSym: 1, regularization: 1e-8, format: 'real2N' }.
    val assembleConnectionLaplacian(val opts) {
        nxr::compute::ConnectionLaplacianOptions o;
        if (!opts.isNull() && !opts.isUndefined()) {
            val domainV = opts["domain"];
            val nSymV   = opts["nSym"];
            val regV    = opts["regularization"];
            val fmtV    = opts["format"];
            if (!domainV.isUndefined())
                o.domain = nxr::compute::parseConnectionDomain(domainV.as<std::string>());
            if (!nSymV.isUndefined())
                o.nSym = nSymV.as<int>();
            if (!regV.isUndefined())
                o.regularization = regV.as<double>();
            if (!fmtV.isUndefined())
                o.format = nxr::compute::parseConnectionLaplacianFormat(fmtV.as<std::string>());
        }

        const CLKey key{o.domain, o.nSym, o.regularization, o.format};
        auto it = clCache_.find(key);
        if (it == clCache_.end()) {
            try {
                auto cl = std::make_shared<nxr::compute::ConnectionLaplacian>(
                    nxr::compute::assembleConnectionLaplacian(*ctx_, o));
                it = clCache_.emplace(key, std::move(cl)).first;
            } catch (const nxr::compute::Error& e) {
                std::string msg = "[";
                msg += nxr::compute::errorCodeName(e.code());
                msg += "] ";
                msg += e.what();
                if (!e.hint().empty()) {
                    msg += " | hint: ";
                    msg += std::string(e.hint());
                }
                throw std::runtime_error(msg);
            }
        }
        const nxr::compute::ConnectionLaplacian& cl = *it->second;

        const char* domainStr =
            cl.domain == nxr::compute::ConnectionDomain::Vertex              ? "vertex" :
            cl.domain == nxr::compute::ConnectionDomain::Face                ? "face"   :
            cl.domain == nxr::compute::ConnectionDomain::EdgeCrouzeixRaviart ? "edge"   : "?";
        const char* formatStr =
            cl.format == nxr::compute::ConnectionLaplacianFormat::Real2N ? "real2N" : "complex";

        val out = val::object();
        if (cl.format == nxr::compute::ConnectionLaplacianFormat::Real2N) {
            out.set("K", sparseToVal(cl.K_real));
        } else {
            out.set("K", sparseComplexToVal(cl.K_complex));
        }
        out.set("baseDim",        cl.baseDim);
        out.set("outputDim",      cl.outputDim);
        out.set("domain",         std::string(domainStr));
        out.set("nSym",           cl.nSym);
        out.set("regularization", cl.regularization);
        out.set("format",         std::string(formatStr));
        return out;
    }

    val computeFaceFrames() {
        auto frames = nxr::compute::computeFaceFrames(*ctx_);
        val obj = val::object();
        obj.set("e1",      eigenMatrixToVal(frames.e1));
        obj.set("e2",      eigenMatrixToVal(frames.e2));
        obj.set("normals", eigenMatrixToVal(frames.normals));
        return obj;
    }

    val computeVertexNormals(int type) {
        nxr::compute::NormalType nt = static_cast<nxr::compute::NormalType>(type);
        Eigen::MatrixXd N = nxr::compute::computeVertexNormals(*ctx_, nt);
        return eigenMatrixToVal(N);
    }

    // ── Spectral basis ───────────────────────────────────────

    // Cancellation/progress contract for WASM consumers:
    //   • cancelAddr — wasm heap pointer to a single int32. Pass 0 to
    //     opt out. Allocate via `Module._malloc(4)` and write via
    //     `Module.HEAP32[ptr >> 2] = 1` to cancel. With SHARED_MEMORY
    //     enabled, JS can use `Atomics.store(new Int32Array(Module
    //     .HEAPU8.buffer), ptr >> 2, 1)` from a different thread.
    //   • progressAddr — wasm heap pointer to a 3×int32 array. Pass 0
    //     to opt out. Layout: [iteration, totalIterations, residual×1e6].
    //
    // Errors land in JS as Error objects whose .message starts with
    // "[CODE] " (where CODE is the nxr::compute::ErrorCode name, e.g. CANCELLED).
    // A small parseError helper on the JS side recovers .code from the
    // message prefix; richer per-binding error mapping can land in a
    // future phase.
    val solveEigenmodes(int k, double sigma,
                        std::intptr_t cancelAddr,
                        std::intptr_t progressAddr,
                        int progressLen) {
        ensureOps();

        nxr::compute::CancellationToken cancel = cancelAddr
            ? nxr::compute::CancellationToken(reinterpret_cast<const std::atomic<int32_t>*>(cancelAddr))
            : nxr::compute::CancellationToken{};

        nxr::compute::ProgressObserver progress;
        if (progressAddr && progressLen >= 3) {
            auto* base = reinterpret_cast<std::atomic<int32_t>*>(progressAddr);
            progress.iteration       = &base[0];
            progress.totalIterations = &base[1];
            progress.residualMicro   = &base[2];
        }

        try {
            nxr::compute::EigenResult r = nxr::compute::solveEigenmodes(
                ops_->cotanLaplacian, ops_->mass, k, sigma, cancel, progress);
            return eigenResultToVal(r);
        } catch (const nxr::compute::Error& e) {
            // Prefix the message with the code so JS consumers can
            // pattern-match without losing the human-readable text.
            std::string msg = "[";
            msg += nxr::compute::errorCodeName(e.code());
            msg += "] ";
            msg += e.what();
            if (!e.hint().empty()) {
                msg += " | hint: ";
                msg += std::string(e.hint());
            }
            throw std::runtime_error(msg);
        }
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
        Eigen::MatrixXd Un = nxr::compute::normalizeEigenmodes(U, ops_->mass);
        return eigenMatrixToVal(Un);
    }

    val removeDC(val eigStruct) {
        nxr::compute::EigenResult r = valToEigenResult(eigStruct);
        nxr::compute::EigenResult t = nxr::compute::removeDC(r);
        return eigenResultToVal(t);
    }

    /** One-shot precompute: assemble + DEC + eigensolve + normalize +
     *  removeDC + face frames, returned as a single struct. The
     *  visualization-defaults pack from docs/nxr-compute/architecture.md. */
    val precompute(int k, double sigma) {
        ensureOps();
        ensureDec();

        nxr::compute::EigenResult eig = nxr::compute::solveEigenmodes(
            ops_->cotanLaplacian, ops_->mass, k, sigma);
        eig.eigenvectors = nxr::compute::normalizeEigenmodes(eig.eigenvectors, ops_->mass);
        eig = nxr::compute::removeDC(eig);
        eigCache_ = std::make_unique<nxr::compute::EigenResult>(eig);

        auto frames = nxr::compute::computeFaceFrames(*ctx_);

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
        Eigen::VectorXd phi = nxr::compute::solvePoisson(*ops_, *cache_, srcMap);
        return eigenVectorToVal(phi);
    }

    val computeGeodesicDistance(val sourceVertsArr) {
        ensureHeatGeo();
        auto sources = emscripten::convertJSArrayToNumberVector<int>(sourceVertsArr);
        Eigen::VectorXd d = nxr::compute::computeGeodesicDistance(*heatGeo_, sources);
        return eigenVectorToVal(d);
    }

    val tracePath(int vStart, int vEnd) {
        Eigen::MatrixXd path = nxr::compute::tracePath(*ctx_, vStart, vEnd);
        val out = val::object();
        out.set("positions", eigenMatrixToVal(path));
        out.set("nPoints",   static_cast<int>(path.rows()));
        return out;
    }

    val hodgeDecompose(val omegaArr) {
        ensureDec();
        Eigen::VectorXd omega = valToEigenVector(omegaArr);
        nxr::compute::HodgeResult h = nxr::compute::hodgeDecompose(*ctx_, *dec_, *cache_, omega);
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
        nxr::compute::CurvatureResult c = nxr::compute::computeCurvatures(*ctx_);
        val o = val::object();
        o.set("gaussian",     eigenVectorToVal(c.gaussian));
        o.set("mean",         eigenVectorToVal(c.mean));
        o.set("kMin",         eigenVectorToVal(c.kMin));
        o.set("kMax",         eigenVectorToVal(c.kMax));
        o.set("principalDir", eigenMatrixToVal(c.principalDirMax));
        return o;
    }

    val computeUVCoordinates() {
        Eigen::MatrixXd uvs = nxr::compute::computeUVCoordinates(*ctx_);
        return eigenMatrixToVal(uvs);
    }

    val computeIsolines(val scalarsArr, int numLevels, double minVal, double maxVal) {
        Eigen::VectorXd scalars = valToEigenVector(scalarsArr);
        nxr::compute::IsolineResult r = nxr::compute::computeIsolines(*ctx_, scalars, numLevels, minVal, maxVal);
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
        nxr::compute::DirectionFieldResult r = nxr::compute::computeDirectionField(*ctx_, *dec_, *cache_, singMap);
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
        nxr::compute::StreamlineResult r = nxr::compute::traceStreamlines(*ctx_, faceField, numSeeds, stepCoef, maxSteps);
        val o = val::object();
        o.set("positions",    eigenMatrixToVal(r.positions));
        o.set("segmentCount", r.segmentCount);
        return o;
    }

    // ── Vector field ops ────────────────────────────────────

    val whitneyInterpolate(val oneFormArr) {
        ensureDec();
        Eigen::VectorXd omega = valToEigenVector(oneFormArr);
        Eigen::MatrixXd faceVecs = nxr::compute::whitneyInterpolate(*ctx_, *dec_, omega);
        return eigenMatrixToVal(faceVecs);
    }

    val scalarGradient(val scalarArr) {
        Eigen::VectorXd s = valToEigenVector(scalarArr);
        Eigen::MatrixXd grad = nxr::compute::scalarGradient(*ctx_, s);
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
        Eigen::VectorXd u0 = nxr::compute::generateDelta(ctx_->nV(), sources);
        std::vector<double> ts = valToDoubleVector(timestepsArr);
        Eigen::MatrixXf field = nxr::compute::generateHeatDiffusion(
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
        Eigen::MatrixXf field = nxr::compute::generateDampedWave(
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
        Eigen::VectorXd omega = nxr::compute::generateRandomDecomposed1Form(
            *dec_, ctx_->nV(), ctx_->nE(), ctx_->nF(),
            alphaStrength, betaStrength, gammaStrength,
            static_cast<unsigned int>(seed));
        return eigenVectorToVal(omega);
    }

    // ── Vector heat method ──────────────────────────────────

    val vectorHeatTransport(val sourceVertsArr, val sourceVectorsArr) {
        ensureVHM();
        auto verts = emscripten::convertJSArrayToNumberVector<int>(sourceVertsArr);
        auto vecs  = emscripten::convertJSArrayToNumberVector<double>(sourceVectorsArr);
        if (vecs.size() != verts.size() * 3) {
            throw std::invalid_argument("sourceVectors must be Nx3 (length 3*sources)");
        }
        Eigen::MatrixXd S(static_cast<int>(verts.size()), 3);
        for (std::size_t i = 0; i < verts.size(); i++) {
            S(i, 0) = vecs[i * 3 + 0];
            S(i, 1) = vecs[i * 3 + 1];
            S(i, 2) = vecs[i * 3 + 2];
        }
        Eigen::MatrixXd out = nxr::compute::vectorHeatTransport(*vhm_, verts, S);
        return eigenMatrixToVal(out);
    }

    val vectorHeatExtendScalar(val sourceVertsArr, val sourceValuesArr) {
        ensureVHM();
        auto verts = emscripten::convertJSArrayToNumberVector<int>(sourceVertsArr);
        auto vals  = emscripten::convertJSArrayToNumberVector<double>(sourceValuesArr);
        if (verts.size() != vals.size()) {
            throw std::invalid_argument("sourceVerts and sourceValues must have the same length");
        }
        Eigen::VectorXd vv(vals.size());
        for (std::size_t i = 0; i < vals.size(); i++) vv[i] = vals[i];
        Eigen::VectorXd out = nxr::compute::vectorHeatExtendScalar(*vhm_, verts, vv);
        return eigenVectorToVal(out);
    }

    val vectorHeatLogMap(int sourceVertex, int strategy) {
        ensureVHM();
        auto s = static_cast<nxr::compute::LogMapStrategy>(strategy);
        nxr::compute::LogMapResult r = nxr::compute::vectorHeatLogMap(*vhm_, sourceVertex, s);
        val obj = val::object();
        obj.set("logCoords", eigenMatrixToVal(r.logCoords));
        double e1[3] = {r.sourceE1.x(), r.sourceE1.y(), r.sourceE1.z()};
        double e2[3] = {r.sourceE2.x(), r.sourceE2.y(), r.sourceE2.z()};
        obj.set("sourceE1", toJsArrayCopy(e1, 3));
        obj.set("sourceE2", toJsArrayCopy(e2, 3));
        return obj;
    }

    val vectorHeatFindCenter(val sourceVertsArr, int p) {
        ensureVHM();
        auto verts = emscripten::convertJSArrayToNumberVector<int>(sourceVertsArr);
        Eigen::Vector3d c = nxr::compute::vectorHeatFindCenter(*vhm_, verts, p);
        double xyz[3] = {c.x(), c.y(), c.z()};
        return toJsArrayCopy(xyz, 3);
    }

    // ── Signed heat method ──────────────────────────────────

    val signedHeatDistance(val curveVertsArr, bool isLoop, int levelSet) {
        ensureSHS();
        auto verts = emscripten::convertJSArrayToNumberVector<int>(curveVertsArr);
        auto ls    = static_cast<nxr::compute::SignedHeatLevelSet>(levelSet);
        Eigen::VectorXd out = nxr::compute::signedHeatDistance(*shs_, verts, isLoop, ls);
        return eigenVectorToVal(out);
    }

    // ── Smooth direction fields ─────────────────────────────
    //
    // geometry-central exposes computeSmoothest{,Boundary}AlignedFaceDirectionField
    // and computeCurvatureAlignedFaceDirectionField as stateless free
    // functions — each call rebuilds the connection-Laplacian and
    // factorizes it. We can't add a Solver class without reimplementing
    // the algorithm, so we cache at the result level keyed by
    // (nSym, alignToCurvature). The geometry doesn't change for the
    // lifetime of a ComputeContext, so the cache is correct by
    // construction. Repeated identical calls become near-zero-cost;
    // parameter changes still pay the full cold cost once per new key.

    val computeSmoothFaceField(int nSym, bool alignToCurvature) {
        auto key = std::make_pair(nSym, alignToCurvature);
        auto it = smoothFaceFieldCache_.find(key);
        if (it != smoothFaceFieldCache_.end()) {
            return eigenMatrixToVal(it->second);
        }
        Eigen::MatrixXd v = nxr::compute::computeSmoothFaceField(*ctx_, nSym, alignToCurvature);
        smoothFaceFieldCache_.emplace(key, v);
        return eigenMatrixToVal(v);
    }

    val computeSmoothVertexField(int nSym, bool alignToCurvature) {
        auto key = std::make_pair(nSym, alignToCurvature);
        auto it = smoothVertexFieldCache_.find(key);
        if (it != smoothVertexFieldCache_.end()) {
            const auto& r = it->second;
            val obj = val::object();
            obj.set("vertexVectors",  eigenMatrixToVal(r.vertexVectors));
            obj.set("vertexFieldRaw", eigenVectorToVal(r.vertexFieldRaw));
            obj.set("nSym",           r.nSym);
            return obj;
        }
        nxr::compute::SmoothVertexFieldResult r =
            nxr::compute::computeSmoothVertexField(*ctx_, nSym, alignToCurvature);
        val obj = val::object();
        obj.set("vertexVectors",  eigenMatrixToVal(r.vertexVectors));
        obj.set("vertexFieldRaw", eigenVectorToVal(r.vertexFieldRaw));
        obj.set("nSym",           r.nSym);
        smoothVertexFieldCache_.emplace(key, std::move(r));
        return obj;
    }

    // ── Stripe patterns ─────────────────────────────────────

    val computeStripePattern(val vertexFieldArr, double uniformFrequency,
                             bool connectOnSingularities) {
        Eigen::VectorXd raw = valToEigenVector(vertexFieldArr);
        nxr::compute::StripePatternResult r = nxr::compute::computeStripePattern(
            *ctx_, raw, uniformFrequency, connectOnSingularities);
        val obj = val::object();
        obj.set("positions",    eigenMatrixToVal(r.positions));
        obj.set("segmentCount", r.segmentCount);
        return obj;
    }

    val computeStripePatternFreq(val vertexFieldArr, val freqsArr,
                                 bool connectOnSingularities) {
        Eigen::VectorXd raw   = valToEigenVector(vertexFieldArr);
        Eigen::VectorXd freqs = valToEigenVector(freqsArr);
        nxr::compute::StripePatternResult r = nxr::compute::computeStripePatternFreq(
            *ctx_, raw, freqs, connectOnSingularities);
        val obj = val::object();
        obj.set("positions",    eigenMatrixToVal(r.positions));
        obj.set("segmentCount", r.segmentCount);
        return obj;
    }

private:
    // Lazy state (matches the addon's ContextHolder caching).
    void ensureOps() {
        if (!ops_) ops_ = std::make_unique<nxr::compute::MeshOperators>(
            nxr::compute::assembleMeshOperators(*ctx_));
    }
    void ensureDec() {
        if (!dec_) dec_ = std::make_unique<nxr::compute::DECOperators>(
            nxr::compute::assembleDECOperators(*ctx_));
    }
    void ensureVHM() {
        if (!vhm_) vhm_ = std::make_unique<nxr::compute::VectorHeatSolver>(*ctx_);
    }
    void ensureSHS() {
        if (!shs_) shs_ = std::make_unique<nxr::compute::SignedHeatSolver>(*ctx_);
    }
    void ensureHeatGeo() {
        if (!heatGeo_) heatGeo_ = std::make_unique<nxr::compute::HeatGeodesicSolver>(*ctx_);
    }

    val meshOpsToVal() {
        ensureOps();
        val o = val::object();
        o.set("cotanLaplacian",  sparseToVal(ops_->cotanLaplacian));
        o.set("mass",            sparseToVal(ops_->mass));
        o.set("massVariant", std::string(
            ops_->massVariant == nxr::compute::MassMatrixVariant::Voronoi      ? "voronoi" :
            ops_->massVariant == nxr::compute::MassMatrixVariant::Barycentric  ? "barycentric" :
                                                                                 "full"));
        o.set("vertexDualAreas", eigenVectorToVal(ops_->vertexDualAreas));
        o.set("vertexNormals",   eigenMatrixToVal(ops_->vertexNormals));
        o.set("totalArea",       ops_->totalArea);
        o.set("nV",              ctx_->nV());
        o.set("nE",              ctx_->nE());
        o.set("nF",              ctx_->nF());
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

    val eigenResultToVal(const nxr::compute::EigenResult& r) {
        val o = val::object();
        // vMajor row-major: U[v*K + k]. Matches the cortical-flow
        // Zarr schema (manifold/eigenmodes/eigenvectors stored as
        // [V, K]) and the GPU spectral-synthesis access pattern.
        o.set("eigenvectors", eigenMatrixToVal(r.eigenvectors));
        o.set("eigenvalues",  eigenVectorToVal(r.eigenvalues));
        o.set("k",            r.k);
        o.set("nConverged",   r.nConverged);
        return o;
    }

    nxr::compute::EigenResult valToEigenResult(const val& s) {
        nxr::compute::EigenResult r;
        val uField = s["eigenvectors"];
        val lField = s["eigenvalues"];
        auto uVec = emscripten::convertJSArrayToNumberVector<double>(uField);
        auto lVec = emscripten::convertJSArrayToNumberVector<double>(lField);
        int K = static_cast<int>(lVec.size());
        int V = K > 0 ? static_cast<int>(uVec.size()) / K : 0;
        // Round-trip layout: JS hands back vMajor (V×K row-major).
        // Map the buffer as a row-major matrix and let Eigen's
        // assignment operator transpose into its own column-major
        // storage — one copy, no manual indexing.
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            rowMajor(uVec.data(), V, K);
        r.eigenvectors = rowMajor;
        r.eigenvalues.resize(K);
        std::memcpy(r.eigenvalues.data(), lVec.data(), K * sizeof(double));
        r.k = K;
        r.nConverged = K;
        return r;
    }

    // ops_ and dec_ are view structs over ctx_'s geometry-central
    // cached matrices (see lifetime contract in compute.h). Declared
    // after ctx_ so destruction order tears them down first; the
    // references inside are never left dangling.
    std::unique_ptr<nxr::compute::ComputeContext>     ctx_;
    std::unique_ptr<nxr::compute::MeshOperators>      ops_;
    std::unique_ptr<nxr::compute::DECOperators>       dec_;
    std::unique_ptr<nxr::compute::CholeskyCache>      cache_;
    std::unique_ptr<nxr::compute::EigenResult>        eigCache_;
    std::unique_ptr<nxr::compute::VectorHeatSolver>   vhm_;
    std::unique_ptr<nxr::compute::SignedHeatSolver>   shs_;
    std::unique_ptr<nxr::compute::HeatGeodesicSolver> heatGeo_;
    // Smooth-field result caches — keyed by (nSym, alignToCurvature).
    // Stored by value so the cache owns the data; lookups copy back
    // out to JS via toJsArrayCopy on each return.
    std::map<std::pair<int, bool>, Eigen::MatrixXd>                       smoothFaceFieldCache_;
    std::map<std::pair<int, bool>, nxr::compute::SmoothVertexFieldResult> smoothVertexFieldCache_;
    // Connection-Laplacian result cache — keyed by all four assembly
    // options. Stored by shared_ptr so callers can hold references
    // without keeping the entire ContextWrapper alive longer than
    // intended.
    using CLKey = std::tuple<nxr::compute::ConnectionDomain,
                              int,
                              double,
                              nxr::compute::ConnectionLaplacianFormat>;
    std::map<CLKey, std::shared_ptr<nxr::compute::ConnectionLaplacian>>   clCache_;
    std::vector<double>  verts_;
    std::vector<int>     faces_;
    std::vector<int32_t> faces32_;
};

// ── Free functions ───────────────────────────────────────────

std::string getVersion() {
    return "nxr-compute 0.1.0";
}

/**
 * Generalized eigenproblem K φ = λ M φ from JS-supplied COO triplets.
 *
 * Mirrors `ContextWrapper::solveEigenmodes` — same Spectra IRAM /
 * shift-invert path and same cancel/progress contract — but takes K
 * and M as raw triplet arrays instead of building them through
 * `assembleMeshOperators`. Useful for callers that compute their own
 * sparse Laplacians (graph Laplacians, FEM assemblies on non-manifold
 * face soups, custom regularized matrices, etc.) without going through
 * the geometry-central halfedge stack.
 *
 * Args (all coordinate-format inputs are 0-based):
 *   K_rows, K_cols  Int32Array (or array of ints), length = K_nnz
 *   K_vals          Float64Array (or array of numbers), length = K_nnz
 *   M_rows, M_cols, M_vals  same shape as K
 *   n               common matrix size for K and M (square)
 *   k, sigma        eigensolve params (sigma defaults to -1e-8 if
 *                   the caller passes 0; explicit value preferred).
 *   cancelAddr, progressAddr, progressLen  same as ContextWrapper.solveEigenmodes
 *
 * Returns: { eigenvectors, eigenvalues, k, nConverged } in vMajor
 * row-major layout, identical to ContextWrapper.solveEigenmodes.
 *
 * Triplets with duplicate (row, col) pairs are summed when the sparse
 * matrix is built (Eigen's setFromTriplets default behavior). This is
 * the natural assembly order for FEM consistent-mass and cotangent
 * stiffness, which both accumulate per-triangle contributions.
 */
val solveEigenmodesFromTriplets(
    val K_rows, val K_cols, val K_vals,
    val M_rows, val M_cols, val M_vals,
    int n,
    int k, double sigma,
    std::intptr_t cancelAddr,
    std::intptr_t progressAddr,
    int progressLen)
{
    auto Kr = emscripten::convertJSArrayToNumberVector<int32_t>(K_rows);
    auto Kc = emscripten::convertJSArrayToNumberVector<int32_t>(K_cols);
    auto Kv = emscripten::convertJSArrayToNumberVector<double>(K_vals);
    auto Mr = emscripten::convertJSArrayToNumberVector<int32_t>(M_rows);
    auto Mc = emscripten::convertJSArrayToNumberVector<int32_t>(M_cols);
    auto Mv = emscripten::convertJSArrayToNumberVector<double>(M_vals);

    if (Kr.size() != Kc.size() || Kr.size() != Kv.size()) {
        throw std::invalid_argument(
            "solveEigenmodesFromTriplets: K row/col/val length mismatch");
    }
    if (Mr.size() != Mc.size() || Mr.size() != Mv.size()) {
        throw std::invalid_argument(
            "solveEigenmodesFromTriplets: M row/col/val length mismatch");
    }
    if (n <= 0) {
        throw std::invalid_argument(
            "solveEigenmodesFromTriplets: n must be > 0");
    }

    Eigen::SparseMatrix<double> K(n, n), M(n, n);
    {
        std::vector<Eigen::Triplet<double>> Kt;
        Kt.reserve(Kr.size());
        for (std::size_t i = 0; i < Kr.size(); ++i) {
            Kt.emplace_back(Kr[i], Kc[i], Kv[i]);
        }
        K.setFromTriplets(Kt.begin(), Kt.end());
        K.makeCompressed();
    }
    {
        std::vector<Eigen::Triplet<double>> Mt;
        Mt.reserve(Mr.size());
        for (std::size_t i = 0; i < Mr.size(); ++i) {
            Mt.emplace_back(Mr[i], Mc[i], Mv[i]);
        }
        M.setFromTriplets(Mt.begin(), Mt.end());
        M.makeCompressed();
    }

    nxr::compute::CancellationToken cancel = cancelAddr
        ? nxr::compute::CancellationToken(
            reinterpret_cast<const std::atomic<int32_t>*>(cancelAddr))
        : nxr::compute::CancellationToken{};
    nxr::compute::ProgressObserver progress;
    if (progressAddr && progressLen >= 3) {
        auto* base = reinterpret_cast<std::atomic<int32_t>*>(progressAddr);
        progress.iteration       = &base[0];
        progress.totalIterations = &base[1];
        progress.residualMicro   = &base[2];
    }

    try {
        nxr::compute::EigenResult r = nxr::compute::solveEigenmodes(
            K, M, k, sigma, cancel, progress);
        // Build the JS return object inline (mirrors ContextWrapper's
        // eigenResultToVal — that one is a class member, so we replicate
        // its three-line body here rather than re-host it).
        val o = val::object();
        o.set("eigenvectors", eigenMatrixToVal(r.eigenvectors));
        o.set("eigenvalues",  eigenVectorToVal(r.eigenvalues));
        o.set("k",            r.k);
        o.set("nConverged",   r.nConverged);
        return o;
    } catch (const nxr::compute::Error& e) {
        std::string msg = "[";
        msg += nxr::compute::errorCodeName(e.code());
        msg += "] ";
        msg += e.what();
        if (!e.hint().empty()) {
            msg += " | hint: ";
            msg += std::string(e.hint());
        }
        throw std::runtime_error(msg);
    }
}

// ── Embind module bindings ───────────────────────────────────

EMSCRIPTEN_BINDINGS(nxr_compute_wasm) {
    emscripten::class_<ContextWrapper>("ComputeContext")
        .constructor<val, val>()
        // Accessors
        .function("nV", &ContextWrapper::nV)
        .function("nE", &ContextWrapper::nE)
        .function("nF", &ContextWrapper::nF)
        // Operators
        .function("assembleMeshOperators",  &ContextWrapper::assembleMeshOperators)
        .function("assembleDECOperators",   &ContextWrapper::assembleDECOperators)
        .function("assembleConnectionLaplacian", &ContextWrapper::assembleConnectionLaplacian)
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
        // Vector heat method
        .function("vectorHeatTransport",    &ContextWrapper::vectorHeatTransport)
        .function("vectorHeatExtendScalar", &ContextWrapper::vectorHeatExtendScalar)
        .function("vectorHeatLogMap",       &ContextWrapper::vectorHeatLogMap)
        .function("vectorHeatFindCenter",   &ContextWrapper::vectorHeatFindCenter)
        // Signed heat method
        .function("signedHeatDistance",     &ContextWrapper::signedHeatDistance)
        // Smooth direction fields
        .function("computeSmoothFaceField",   &ContextWrapper::computeSmoothFaceField)
        .function("computeSmoothVertexField", &ContextWrapper::computeSmoothVertexField)
        // Stripe patterns
        .function("computeStripePattern",     &ContextWrapper::computeStripePattern)
        .function("computeStripePatternFreq", &ContextWrapper::computeStripePatternFreq)
        ;

    emscripten::function("version", &getVersion);
    emscripten::function("solveEigenmodesFromTriplets",
                         &solveEigenmodesFromTriplets);
}
