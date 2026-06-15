/**
 * nxr_compute_wasm.cpp — Embind bindings exposing nxr-compute to JavaScript via WebAssembly.
 *
 * Mirrors the N-API addon's surface — same compute methods, same stateful
 * Manifold pattern, same data shapes — but built around Embind's
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
 * The class `ContextWrapper` caches ManifoldOperators, DECOperators,
 * CholeskyCache, and EigenResult on the C++ side — same pattern as the
 * addon's ContextHolder. This means a Hodge solve after a Poisson solve
 * doesn't refactor the cotan Laplacian.
 */

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "nxr/compute.h"
#include "nxr/facets.h"   // OperatorsFacet definition (operators() command)

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

// Re-throw an nxr-compute Error as a std::runtime_error whose message carries
// the "[CODE] msg | hint: ..." shape JS consumers parse (same as solve() /
// assembleConnectionLaplacian above). Centralised so every binding method that
// surfaces nxr::core::Error produces an identical JS .message.
[[noreturn]] void rethrowAsJsError(const nxr::core::Error& e) {
    std::string msg = "[";
    msg += nxr::core::errorCodeName(e.code());
    msg += "] ";
    msg += e.what();
    if (!e.hint().empty()) {
        msg += " | hint: ";
        msg += std::string(e.hint());
    }
    throw std::runtime_error(msg);
}

} // namespace

// ── Manifold wrapper class ─────────────────────────────
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

        // nxr::manifold::Manifold takes int32_t* for faces; copy into the
        // canonical type since convertJSArrayToNumberVector<int> may
        // produce a different underlying integer width on some platforms.
        faces32_.resize(faces_.size());
        for (std::size_t i = 0; i < faces_.size(); i++) {
            faces32_[i] = static_cast<int32_t>(faces_[i]);
        }

        ctx_ = std::make_unique<nxr::manifold::Manifold>(
            verts_.data(), nV, faces32_.data(), nF);
        cache_ = std::make_unique<nxr::manifold::ops::CholeskyCache>();
    }

    int nV() const { return ctx_->nV(); }
    int nE() const { return ctx_->nE(); }
    int nF() const { return ctx_->nF(); }

    // ── Mesh operators ───────────────────────────────────────

    // Optional `variantName` accepts "lumped" (default) or "galerkin",
    // matching geometry-central's vertexLumpedMassMatrix and
    // vertexGalerkinMassMatrix. Empty string → default. Switching
    // variants invalidates the cached ManifoldOperators (and any cached
    // factor that depended on the previous mass).
    val assembleManifoldOperators(const std::string& variantName) {
        nxr::manifold::ops::MassMatrixVariant variant = variantName.empty()
            ? nxr::manifold::ops::MassMatrixVariant::Lumped
            : nxr::manifold::ops::parseMassMatrixVariant(variantName);
        if (!ops_ || ops_->massVariant != variant) {
            ops_ = std::make_unique<nxr::manifold::ops::ManifoldOperators>(
                nxr::manifold::ops::assembleManifoldOperators(*ctx_, variant));
        }
        return meshOpsToVal();
    }

    val assembleDECOperators() {
        ensureDec();
        return decOpsToVal();
    }

    // Connection Laplacian on the chosen domain (vertex / face / edge).
    // Result-level cache keyed by (domain, nSym, regularization, format)
    // — same pattern as smoothFace (CLAUDE.md rule 9).
    //
    // `opts` is a JS object: { domain?, nSym?, regularization?, format? }.
    // Defaults: { domain: 'vertex', nSym: 1, regularization: 1e-8, format: 'real2N' }.
    val assembleConnectionLaplacian(val opts) {
        nxr::manifold::ops::laplacian::connection::ConnectionLaplacianOptions o;
        if (!opts.isNull() && !opts.isUndefined()) {
            val domainV = opts["domain"];
            val nSymV   = opts["nSym"];
            val regV    = opts["regularization"];
            val fmtV    = opts["format"];
            if (!domainV.isUndefined())
                o.domain = nxr::manifold::ops::laplacian::connection::parseConnectionDomain(domainV.as<std::string>());
            if (!nSymV.isUndefined())
                o.nSym = nSymV.as<int>();
            if (!regV.isUndefined())
                o.regularization = regV.as<double>();
            if (!fmtV.isUndefined())
                o.format = nxr::manifold::ops::laplacian::connection::parseConnectionLaplacianFormat(fmtV.as<std::string>());
        }

        const CLKey key{o.domain, o.nSym, o.regularization, o.format};
        auto it = clCache_.find(key);
        if (it == clCache_.end()) {
            try {
                auto cl = std::make_shared<nxr::manifold::ops::laplacian::connection::ConnectionLaplacian>(
                    nxr::manifold::ops::laplacian::connection::assembleConnectionLaplacian(*ctx_, o));
                it = clCache_.emplace(key, std::move(cl)).first;
            } catch (const nxr::core::Error& e) {
                std::string msg = "[";
                msg += nxr::core::errorCodeName(e.code());
                msg += "] ";
                msg += e.what();
                if (!e.hint().empty()) {
                    msg += " | hint: ";
                    msg += std::string(e.hint());
                }
                throw std::runtime_error(msg);
            }
        }
        const nxr::manifold::ops::laplacian::connection::ConnectionLaplacian& cl = *it->second;

        const char* domainStr =
            cl.domain == nxr::manifold::ops::laplacian::connection::ConnectionDomain::Vertex              ? "vertex" :
            cl.domain == nxr::manifold::ops::laplacian::connection::ConnectionDomain::Face                ? "face"   :
            cl.domain == nxr::manifold::ops::laplacian::connection::ConnectionDomain::EdgeCrouzeixRaviart ? "edge"   : "?";
        const char* formatStr =
            cl.format == nxr::manifold::ops::laplacian::connection::ConnectionLaplacianFormat::Real2N ? "real2N" : "complex";

        val out = val::object();
        if (cl.format == nxr::manifold::ops::laplacian::connection::ConnectionLaplacianFormat::Real2N) {
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

    // Single named operator as COO sparse — mirrors the MEX 'operators' command
    // (bindings/mex/src/nxr_compute_mex.cpp::cmdOperators) for cross-binding
    // parity. The `arg` is polymorphic: a subtype STRING for laplacian/mass/hodge,
    // a numeric TAU for dirac/diracFace (∈ [0,1]), or null/undefined for the
    // no-argument families (dec/gradient3D/diracD/diracFaceD/diracIntrinsicD).
    // Returns a real COO object; 'connection' returns a complex COO
    // ({row,col,realData,imagData}); 'dec' returns { d0, d1 }. Same matrices the
    // internal solvers use (single source of truth — the Manifold operators facet).
    val operators(std::string family, val arg) {
        namespace cl = nxr::manifold::ops::laplacian::connection;
        nxr::manifold::Manifold& m = *ctx_;
        const bool hasNum = arg.isNumber();
        const std::string sub = arg.isString() ? arg.as<std::string>() : "";
        try {
            if (family == "laplacian") {
                if (sub == "cotan")      return sparseToVal(m.operators().laplacian().cotan());
                if (sub == "graph")      return sparseToVal(m.operators().laplacian().graph());
                if (sub == "connection") return sparseComplexToVal(m.operators().laplacian().connection());
                if (sub == "covariant")  return sparseToVal(m.operators().laplacian().covariant(cl::CovariantCoupling::Ambient));
                throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                    "operators laplacian: subtype must be cotan|graph|connection|covariant.");
            } else if (family == "mass") {
                if (sub == "lumped")   return sparseToVal(m.operators().mass().lumped());
                if (sub == "galerkin") return sparseToVal(m.operators().mass().galerkin());
                throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                    "operators mass: subtype must be lumped|galerkin.");
            } else if (family == "hodge") {
                if (sub == "h0")    return sparseToVal(m.operators().hodge().h0());
                if (sub == "h1")    return sparseToVal(m.operators().hodge().h1());
                if (sub == "h2")    return sparseToVal(m.operators().hodge().h2());
                if (sub == "h1inv") return sparseToVal(m.operators().hodge().h1inv());
                throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                    "operators hodge: subtype must be h0|h1|h2|h1inv.");
            } else if (family == "dec") {
                const auto& dec = m.operators().dec();
                val obj = val::object();
                obj.set("d0", sparseToVal(dec.d0));
                obj.set("d1", sparseToVal(dec.d1));
                return obj;
            } else if (family == "gradient3D") {
                return sparseToVal(m.operators().gradient3D());
            } else if (family == "dirac") {
                if (!hasNum)
                    throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                        "operators dirac: expected a numeric tau, operators('dirac', tau).");
                return sparseToVal(m.operators().dirac(arg.as<double>()));
            } else if (family == "diracFace") {
                if (!hasNum)
                    throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                        "operators diracFace: expected a numeric tau, operators('diracFace', tau).");
                return sparseToVal(m.operators().diracFace(arg.as<double>()));
            } else if (family == "diracD") {
                return sparseToVal(m.operators().diracD());
            } else if (family == "diracFaceD") {
                return sparseToVal(m.operators().diracFaceD());
            } else if (family == "diracIntrinsicD") {
                return sparseToVal(m.operators().diracIntrinsicD());
            }
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators: family must be "
                "laplacian|mass|hodge|dec|gradient3D|dirac|diracFace|diracD|diracFaceD|diracIntrinsicD.");
        } catch (const nxr::core::Error& e) {
            rethrowAsJsError(e);
        }
    }

    // Named-operator eigensolve — assembles the operator AND its natural
    // generalized mass C++-side (no JS-side ⊗I₄, one boundary crossing) and
    // eigensolves via the shared solve::eigenOperator (single source of truth).
    // opts: { operator: 'laplacian'|'cotan'|'graph'|'dirac'|'diracFace',
    //         subtype?: 'cotan'|'graph',           // when operator === 'laplacian'
    //         tau?: number,                        // dirac / diracFace blend ∈ [0,1]
    //         mass?: 'lumped'|'galerkin',          // vertex-domain mass (default galerkin)
    //         k: number, sigma?: number (= -1e-8),
    //         normalize?: boolean (= true),
    //         multiplets?: boolean (= false),      // Dirac: exact 4-fold ℍ-reconstruction
    //         dense?: boolean (= false) }          // exact small-mesh verification
    // Returns { eigenvectors (vMajor, n·k), eigenvalues, k, nConverged, blockSize }.
    val eigs(val opts) {
        if (opts.isNull() || opts.isUndefined() || opts["operator"].isUndefined())
            throw std::runtime_error("[INVALID_INPUT] eigs: expected { operator, k, ... }.");
        try {
            nxr::manifold::solve::EigenOperatorSpec spec;
            const std::string op = opts["operator"].as<std::string>();
            int blockSize = 1;
            if (op == "laplacian") {
                const std::string sub = opts["subtype"].isUndefined() ? "" : opts["subtype"].as<std::string>();
                if      (sub == "cotan") spec.op = nxr::manifold::solve::EigenOperator::LaplacianCotan;
                else if (sub == "graph") spec.op = nxr::manifold::solve::EigenOperator::LaplacianGraph;
                else throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                    "eigs laplacian: subtype must be cotan|graph.");
            } else if (op == "cotan") {
                spec.op = nxr::manifold::solve::EigenOperator::LaplacianCotan;
            } else if (op == "graph") {
                spec.op = nxr::manifold::solve::EigenOperator::LaplacianGraph;
            } else if (op == "dirac") {
                spec.op = nxr::manifold::solve::EigenOperator::Dirac; blockSize = 4;
            } else if (op == "diracFace") {
                spec.op = nxr::manifold::solve::EigenOperator::DiracFace; blockSize = 4;
            } else {
                throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                    "eigs: operator must be laplacian|cotan|graph|dirac|diracFace.");
            }

            if (!opts["tau"].isUndefined())  spec.tau  = opts["tau"].as<double>();
            if (!opts["mass"].isUndefined())
                spec.mass = nxr::manifold::ops::parseMassMatrixVariant(opts["mass"].as<std::string>());
            if (opts["k"].isUndefined())
                throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "eigs: k is required.");

            const int    k          = opts["k"].as<int>();
            const double sigma      = opts["sigma"].isUndefined()      ? -1e-8 : opts["sigma"].as<double>();
            const bool   normalize  = opts["normalize"].isUndefined()  ? true  : opts["normalize"].as<bool>();
            const bool   multiplets = opts["multiplets"].isUndefined() ? false : opts["multiplets"].as<bool>();
            const bool   dense      = opts["dense"].isUndefined()      ? false : opts["dense"].as<bool>();

            nxr::manifold::solve::EigenResult r = nxr::manifold::solve::eigenOperator(
                *ctx_, spec, k, sigma, normalize, multiplets, dense);
            val out = eigenResultToVal(r);
            out.set("blockSize", blockSize);
            return out;
        } catch (const nxr::core::Error& e) {
            rethrowAsJsError(e);
        }
    }

    val frames() {
        auto frames = nxr::manifold::geometry::frames(*ctx_);
        val obj = val::object();
        obj.set("e1",      eigenMatrixToVal(frames.e1));
        obj.set("e2",      eigenMatrixToVal(frames.e2));
        obj.set("normals", eigenMatrixToVal(frames.normals));
        return obj;
    }

    val normals(int type) {
        nxr::manifold::geometry::NormalType nt = static_cast<nxr::manifold::geometry::NormalType>(type);
        Eigen::MatrixXd N = nxr::manifold::geometry::normals(*ctx_, nt);
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
    // "[CODE] " (where CODE is the nxr::core::ErrorCode name, e.g. CANCELLED).
    // A small parseError helper on the JS side recovers .code from the
    // message prefix; richer per-binding error mapping can land in a
    // future phase.
    val solve(int k, double sigma,
                        std::intptr_t cancelAddr,
                        std::intptr_t progressAddr,
                        int progressLen) {
        ensureOps();

        nxr::core::CancellationToken cancel = cancelAddr
            ? nxr::core::CancellationToken(reinterpret_cast<const std::atomic<int32_t>*>(cancelAddr))
            : nxr::core::CancellationToken{};

        nxr::core::ProgressObserver progress;
        if (progressAddr && progressLen >= 3) {
            auto* base = reinterpret_cast<std::atomic<int32_t>*>(progressAddr);
            progress.iteration       = &base[0];
            progress.totalIterations = &base[1];
            progress.residualMicro   = &base[2];
        }

        try {
            nxr::manifold::solve::EigenResult r = nxr::manifold::solve::eigen(
                ops_->cotanLaplacian, ops_->mass, k, sigma,
                /*normalize=*/false, /*removeDC=*/false, cancel, progress);
            return eigenResultToVal(r);
        } catch (const nxr::core::Error& e) {
            // Prefix the message with the code so JS consumers can
            // pattern-match without losing the human-readable text.
            std::string msg = "[";
            msg += nxr::core::errorCodeName(e.code());
            msg += "] ";
            msg += e.what();
            if (!e.hint().empty()) {
                msg += " | hint: ";
                msg += std::string(e.hint());
            }
            throw std::runtime_error(msg);
        }
    }

    val normalize(val UJsArr, int rows, int cols) {
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
        Eigen::MatrixXd Un = nxr::manifold::solve::normalize(U, ops_->mass);
        return eigenMatrixToVal(Un);
    }

    val removeDC(val eigStruct) {
        nxr::manifold::solve::EigenResult r = valToEigenResult(eigStruct);
        nxr::manifold::solve::EigenResult t = nxr::manifold::solve::removeDC(r);
        return eigenResultToVal(t);
    }

    /** One-shot precompute: assemble + DEC + eigensolve + normalize +
     *  removeDC + face frames, returned as a single struct. The
     *  visualization-defaults pack from docs/nxr-compute/architecture.md. */
    val precompute(int k, double sigma) {
        ensureOps();
        ensureDec();

        nxr::manifold::solve::EigenResult eig = nxr::manifold::solve::eigen(
            ops_->cotanLaplacian, ops_->mass, k, sigma,
            /*normalize=*/true, /*removeDC=*/true);
        eigCache_ = std::make_unique<nxr::manifold::solve::EigenResult>(eig);

        auto frames = nxr::manifold::geometry::frames(*ctx_);

        val out = val::object();
        out.set("operators",  meshOpsToVal());
        out.set("dec",        decOpsToVal());
        out.set("eigenmodes", eigenResultToVal(eig));
        val framesVal = val::object();
        framesVal.set("e1",      eigenMatrixToVal(frames.e1));
        framesVal.set("e2",      eigenMatrixToVal(frames.e2));
        framesVal.set("normals", eigenMatrixToVal(frames.normals));
        out.set("frames", framesVal);
        return out;
    }

    // ── Solvers ──────────────────────────────────────────────

    val poisson(val sourceVertsArr, val sourceValuesArr) {
        ensureOps();
        auto verts = emscripten::convertJSArrayToNumberVector<int>(sourceVertsArr);
        auto vals  = emscripten::convertJSArrayToNumberVector<double>(sourceValuesArr);
        if (verts.size() != vals.size()) {
            throw std::invalid_argument("sourceVerts and sourceValues must have the same length");
        }
        std::map<int, double> srcMap;
        for (std::size_t i = 0; i < verts.size(); i++) srcMap[verts[i]] = vals[i];
        Eigen::VectorXd phi = nxr::manifold::solve::poisson(*ops_, *cache_, srcMap);
        return eigenVectorToVal(phi);
    }

    val heat(val sourceVertsArr) {
        ensureHeatGeo();
        auto sources = emscripten::convertJSArrayToNumberVector<int>(sourceVertsArr);
        Eigen::VectorXd d = nxr::manifold::solve::heat(*heatGeo_, sources);
        return eigenVectorToVal(d);
    }

    val tracePath(int vStart, int vEnd) {
        Eigen::MatrixXd path = nxr::manifold::query::tracePath(*ctx_, vStart, vEnd);
        val out = val::object();
        out.set("positions", eigenMatrixToVal(path));
        out.set("nPoints",   static_cast<int>(path.rows()));
        return out;
    }

    val hodge(val omegaArr) {
        ensureDec();
        Eigen::VectorXd omega = valToEigenVector(omegaArr);
        nxr::manifold::solve::HodgeResult h = nxr::manifold::solve::hodge(*ctx_, *dec_, *cache_, omega);
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

    val curvatures() {
        nxr::manifold::geometry::CurvatureResult c = nxr::manifold::geometry::curvatures(*ctx_);
        val o = val::object();
        o.set("gaussian",     eigenVectorToVal(c.gaussian));
        o.set("mean",         eigenVectorToVal(c.mean));
        o.set("kMin",         eigenVectorToVal(c.kMin));
        o.set("kMax",         eigenVectorToVal(c.kMax));
        o.set("principalDir", eigenMatrixToVal(c.principalDirMax));
        return o;
    }

    val bff() {
        Eigen::MatrixXd uvs = nxr::manifold::parametrization::bff(*ctx_);
        return eigenMatrixToVal(uvs);
    }

    val isoline(val scalarsArr, int numLevels, double minVal, double maxVal) {
        Eigen::VectorXd scalars = valToEigenVector(scalarsArr);
        nxr::field::extract::IsolineResult r = nxr::field::extract::isoline(*ctx_, scalars, numLevels, minVal, maxVal);
        val out = val::object();
        out.set("positions",    eigenMatrixToVal(r.positions));
        out.set("segmentCount", r.segmentCount);
        return out;
    }

    val trivial(val singVertsArr, val singValuesArr) {
        ensureDec();
        auto verts = emscripten::convertJSArrayToNumberVector<int>(singVertsArr);
        auto vals  = emscripten::convertJSArrayToNumberVector<double>(singValuesArr);
        std::map<int, double> singMap;
        for (std::size_t i = 0; i < verts.size(); i++) singMap[verts[i]] = vals[i];
        nxr::manifold::connection::DirectionFieldResult r = nxr::manifold::connection::trivial(*ctx_, *dec_, *cache_, singMap);
        val o = val::object();
        o.set("connections",         eigenVectorToVal(r.connections));
        o.set("directionVectors",    eigenMatrixToVal(r.directionVectors));
        o.set("orthogonalVectors",   eigenMatrixToVal(r.orthogonalVectors));
        o.set("eulerCharacteristic", r.eulerCharacteristic);
        o.set("gaussBonnetSatisfied", r.gaussBonnetSatisfied);
        return o;
    }

    val streamline(val faceFieldArr, int numSeeds, double stepCoef, int maxSteps) {
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
        nxr::field::extract::StreamlineResult r = nxr::field::extract::streamline(*ctx_, faceField, numSeeds, stepCoef, maxSteps);
        val o = val::object();
        o.set("positions",    eigenMatrixToVal(r.positions));
        o.set("segmentCount", r.segmentCount);
        return o;
    }

    // ── Vector field ops ────────────────────────────────────

    val whitney(val oneFormArr) {
        ensureDec();
        Eigen::VectorXd omega = valToEigenVector(oneFormArr);
        Eigen::MatrixXd faceVecs = nxr::field::interp::whitney(*ctx_, *dec_, omega);
        return eigenMatrixToVal(faceVecs);
    }

    val gradient(val scalarArr) {
        Eigen::VectorXd s = valToEigenVector(scalarArr);
        Eigen::MatrixXd grad = nxr::field::op::gradient(*ctx_, s);
        return eigenMatrixToVal(grad);
    }

    // ── Time-varying field generators ───────────────────────

    val heatDiffusion(val sourceVertsArr, val sourceValuesArr,
                              val timestepsArr, double alpha) {
        ensureOps();
        if (!eigCache_) {
            throw std::runtime_error(
                "heatDiffusion: eigenmodes not yet computed; "
                "call solve() or precompute() first");
        }
        auto verts = emscripten::convertJSArrayToNumberVector<int>(sourceVertsArr);
        auto vals  = emscripten::convertJSArrayToNumberVector<double>(sourceValuesArr);
        std::map<int, double> sources;
        for (std::size_t i = 0; i < verts.size(); i++) sources[verts[i]] = vals[i];
        Eigen::VectorXd u0 = nxr::field::generate::delta(ctx_->nV(), sources);
        std::vector<double> ts = valToDoubleVector(timestepsArr);
        Eigen::MatrixXf field = nxr::field::generate::heatDiffusion(
            *ops_, *eigCache_, u0, ts, alpha);
        val out = val::object();
        out.set("data", eigenMatrixFloat32ToVal(field));
        out.set("T",  static_cast<int>(field.rows()));
        out.set("nV", static_cast<int>(field.cols()));
        return out;
    }

    val dampedWave(val modeIndicesArr, val amplitudesArr,
                           val dampingsArr, val phasesArr,
                           val timestepsArr) {
        if (!eigCache_) {
            throw std::runtime_error(
                "dampedWave: eigenmodes not yet computed; "
                "call solve() or precompute() first");
        }
        auto mi  = valToInt32Vector(modeIndicesArr);
        auto am  = valToDoubleVector(amplitudesArr);
        auto da  = valToDoubleVector(dampingsArr);
        auto ph  = valToDoubleVector(phasesArr);
        auto ts  = valToDoubleVector(timestepsArr);
        Eigen::MatrixXf field = nxr::field::generate::dampedWave(
            *eigCache_, mi, am, da, ph, ts);
        val out = val::object();
        out.set("data", eigenMatrixFloat32ToVal(field));
        out.set("T",  static_cast<int>(field.rows()));
        out.set("nV", static_cast<int>(field.cols()));
        return out;
    }

    val randomDecomposed1Form(double alphaStrength, double betaStrength,
                                      double gammaStrength, int seed) {
        ensureDec();
        Eigen::VectorXd omega = nxr::field::generate::randomDecomposed1Form(
            *dec_, ctx_->nV(), ctx_->nE(), ctx_->nF(),
            alphaStrength, betaStrength, gammaStrength,
            static_cast<unsigned int>(seed));
        return eigenVectorToVal(omega);
    }

    // ── Vector heat method ──────────────────────────────────

    val parallel(val sourceVertsArr, val sourceVectorsArr) {
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
        Eigen::MatrixXd out = nxr::manifold::transport::parallel(*vhm_, verts, S);
        return eigenMatrixToVal(out);
    }

    val extendScalar(val sourceVertsArr, val sourceValuesArr) {
        ensureVHM();
        auto verts = emscripten::convertJSArrayToNumberVector<int>(sourceVertsArr);
        auto vals  = emscripten::convertJSArrayToNumberVector<double>(sourceValuesArr);
        if (verts.size() != vals.size()) {
            throw std::invalid_argument("sourceVerts and sourceValues must have the same length");
        }
        Eigen::VectorXd vv(vals.size());
        for (std::size_t i = 0; i < vals.size(); i++) vv[i] = vals[i];
        Eigen::VectorXd out = nxr::manifold::transport::extendScalar(*vhm_, verts, vv);
        return eigenVectorToVal(out);
    }

    val logMap(int sourceVertex, int strategy) {
        ensureVHM();
        auto s = static_cast<nxr::manifold::transport::LogMapStrategy>(strategy);
        nxr::manifold::transport::LogMapResult r = nxr::manifold::transport::logMap(*vhm_, sourceVertex, s);
        val obj = val::object();
        obj.set("logCoords", eigenMatrixToVal(r.logCoords));
        double e1[3] = {r.sourceE1.x(), r.sourceE1.y(), r.sourceE1.z()};
        double e2[3] = {r.sourceE2.x(), r.sourceE2.y(), r.sourceE2.z()};
        obj.set("sourceE1", toJsArrayCopy(e1, 3));
        obj.set("sourceE2", toJsArrayCopy(e2, 3));
        return obj;
    }

    val findCenter(val sourceVertsArr, int p) {
        ensureVHM();
        auto verts = emscripten::convertJSArrayToNumberVector<int>(sourceVertsArr);
        Eigen::Vector3d c = nxr::manifold::transport::findCenter(*vhm_, verts, p);
        double xyz[3] = {c.x(), c.y(), c.z()};
        return toJsArrayCopy(xyz, 3);
    }

    // ── Signed heat method ──────────────────────────────────

    val signedHeat(val curveVertsArr, bool isLoop, int levelSet) {
        ensureSHS();
        auto verts = emscripten::convertJSArrayToNumberVector<int>(curveVertsArr);
        auto ls    = static_cast<nxr::manifold::solve::SignedHeatLevelSet>(levelSet);
        Eigen::VectorXd out = nxr::manifold::solve::signedHeat(*shs_, verts, isLoop, ls);
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
    // lifetime of a Manifold, so the cache is correct by
    // construction. Repeated identical calls become near-zero-cost;
    // parameter changes still pay the full cold cost once per new key.

    val smoothFace(int nSym, bool alignToCurvature) {
        auto key = std::make_pair(nSym, alignToCurvature);
        auto it = smoothFaceFieldCache_.find(key);
        if (it != smoothFaceFieldCache_.end()) {
            return eigenMatrixToVal(it->second);
        }
        Eigen::MatrixXd v = nxr::manifold::connection::smoothFace(*ctx_, nSym, alignToCurvature);
        smoothFaceFieldCache_.emplace(key, v);
        return eigenMatrixToVal(v);
    }

    val smoothVertex(int nSym, bool alignToCurvature) {
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
        nxr::manifold::connection::SmoothVertexFieldResult r =
            nxr::manifold::connection::smoothVertex(*ctx_, nSym, alignToCurvature);
        val obj = val::object();
        obj.set("vertexVectors",  eigenMatrixToVal(r.vertexVectors));
        obj.set("vertexFieldRaw", eigenVectorToVal(r.vertexFieldRaw));
        obj.set("nSym",           r.nSym);
        smoothVertexFieldCache_.emplace(key, std::move(r));
        return obj;
    }

    // ── Stripe patterns ─────────────────────────────────────

    val compute(val vertexFieldArr, double uniformFrequency,
                             bool connectOnSingularities) {
        Eigen::VectorXd raw = valToEigenVector(vertexFieldArr);
        nxr::manifold::parametrization::stripes::StripePatternResult r = nxr::manifold::parametrization::stripes::compute(
            *ctx_, raw, uniformFrequency, connectOnSingularities);
        val obj = val::object();
        obj.set("positions",    eigenMatrixToVal(r.positions));
        obj.set("segmentCount", r.segmentCount);
        return obj;
    }

    val computeFreq(val vertexFieldArr, val freqsArr,
                                 bool connectOnSingularities) {
        Eigen::VectorXd raw   = valToEigenVector(vertexFieldArr);
        Eigen::VectorXd freqs = valToEigenVector(freqsArr);
        nxr::manifold::parametrization::stripes::StripePatternResult r = nxr::manifold::parametrization::stripes::computeFreq(
            *ctx_, raw, freqs, connectOnSingularities);
        val obj = val::object();
        obj.set("positions",    eigenMatrixToVal(r.positions));
        obj.set("segmentCount", r.segmentCount);
        return obj;
    }

private:
    // Lazy state (matches the addon's ContextHolder caching).
    void ensureOps() {
        if (!ops_) ops_ = std::make_unique<nxr::manifold::ops::ManifoldOperators>(
            nxr::manifold::ops::assembleManifoldOperators(*ctx_));
    }
    void ensureDec() {
        if (!dec_) dec_ = std::make_unique<nxr::manifold::ops::DECOperators>(
            nxr::manifold::ops::assembleDECOperators(*ctx_));
    }
    void ensureVHM() {
        if (!vhm_) vhm_ = std::make_unique<nxr::manifold::transport::VectorHeatSolver>(*ctx_);
    }
    void ensureSHS() {
        if (!shs_) shs_ = std::make_unique<nxr::manifold::solve::SignedHeatSolver>(*ctx_);
    }
    void ensureHeatGeo() {
        if (!heatGeo_) heatGeo_ = std::make_unique<nxr::manifold::solve::HeatGeodesicSolver>(*ctx_);
    }

    val meshOpsToVal() {
        ensureOps();
        val o = val::object();
        o.set("cotanLaplacian",  sparseToVal(ops_->cotanLaplacian));
        o.set("mass",            sparseToVal(ops_->mass));
        o.set("massVariant",     std::string(
            ops_->massVariant == nxr::manifold::ops::MassMatrixVariant::Lumped   ? "lumped"   :
            ops_->massVariant == nxr::manifold::ops::MassMatrixVariant::Galerkin ? "galerkin" : "?"));
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

    val eigenResultToVal(const nxr::manifold::solve::EigenResult& r) {
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

    nxr::manifold::solve::EigenResult valToEigenResult(const val& s) {
        nxr::manifold::solve::EigenResult r;
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

    std::unique_ptr<nxr::manifold::Manifold>     ctx_;
    std::unique_ptr<nxr::manifold::ops::ManifoldOperators>      ops_;
    std::unique_ptr<nxr::manifold::ops::DECOperators>       dec_;
    std::unique_ptr<nxr::manifold::ops::CholeskyCache>      cache_;
    std::unique_ptr<nxr::manifold::solve::EigenResult>        eigCache_;
    std::unique_ptr<nxr::manifold::transport::VectorHeatSolver>   vhm_;
    std::unique_ptr<nxr::manifold::solve::SignedHeatSolver>   shs_;
    std::unique_ptr<nxr::manifold::solve::HeatGeodesicSolver> heatGeo_;
    // Smooth-field result caches — keyed by (nSym, alignToCurvature).
    // Stored by value so the cache owns the data; lookups copy back
    // out to JS via toJsArrayCopy on each return.
    std::map<std::pair<int, bool>, Eigen::MatrixXd>                       smoothFaceFieldCache_;
    std::map<std::pair<int, bool>, nxr::manifold::connection::SmoothVertexFieldResult> smoothVertexFieldCache_;
    // Connection-Laplacian result cache — keyed by all four assembly
    // options. Stored by shared_ptr so callers can hold references
    // without keeping the entire ContextWrapper alive longer than
    // intended.
    using CLKey = std::tuple<nxr::manifold::ops::laplacian::connection::ConnectionDomain,
                              int,
                              double,
                              nxr::manifold::ops::laplacian::connection::ConnectionLaplacianFormat>;
    std::map<CLKey, std::shared_ptr<nxr::manifold::ops::laplacian::connection::ConnectionLaplacian>>   clCache_;
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
 * Mirrors `ContextWrapper::solve` — same Spectra IRAM /
 * shift-invert path and same cancel/progress contract — but takes K
 * and M as raw triplet arrays instead of building them through
 * `assembleManifoldOperators`. Useful for callers that compute their own
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
 *   cancelAddr, progressAddr, progressLen  same as ContextWrapper.solve
 *
 * Returns: { eigenvectors, eigenvalues, k, nConverged } in vMajor
 * row-major layout, identical to ContextWrapper.solve.
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

    nxr::core::CancellationToken cancel = cancelAddr
        ? nxr::core::CancellationToken(
            reinterpret_cast<const std::atomic<int32_t>*>(cancelAddr))
        : nxr::core::CancellationToken{};
    nxr::core::ProgressObserver progress;
    if (progressAddr && progressLen >= 3) {
        auto* base = reinterpret_cast<std::atomic<int32_t>*>(progressAddr);
        progress.iteration       = &base[0];
        progress.totalIterations = &base[1];
        progress.residualMicro   = &base[2];
    }

    try {
        nxr::manifold::solve::EigenResult r = nxr::manifold::solve::eigen(
            K, M, k, sigma,
            /*normalize=*/false, /*removeDC=*/false, cancel, progress);
        // Build the JS return object inline (mirrors ContextWrapper's
        // eigenResultToVal — that one is a class member, so we replicate
        // its three-line body here rather than re-host it).
        val o = val::object();
        o.set("eigenvectors", eigenMatrixToVal(r.eigenvectors));
        o.set("eigenvalues",  eigenVectorToVal(r.eigenvalues));
        o.set("k",            r.k);
        o.set("nConverged",   r.nConverged);
        return o;
    } catch (const nxr::core::Error& e) {
        std::string msg = "[";
        msg += nxr::core::errorCodeName(e.code());
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
    emscripten::class_<ContextWrapper>("Manifold")
        .constructor<val, val>()
        // Accessors
        .function("nV", &ContextWrapper::nV)
        .function("nE", &ContextWrapper::nE)
        .function("nF", &ContextWrapper::nF)
        // Operators
        .function("assembleManifoldOperators",  &ContextWrapper::assembleManifoldOperators)
        .function("assembleDECOperators",   &ContextWrapper::assembleDECOperators)
        .function("assembleConnectionLaplacian", &ContextWrapper::assembleConnectionLaplacian)
        .function("operators",   &ContextWrapper::operators)
        .function("eigs",        &ContextWrapper::eigs)
        .function("frames",      &ContextWrapper::frames)
        .function("normals",   &ContextWrapper::normals)
        // Spectral
        .function("solve",        &ContextWrapper::solve)
        .function("normalize",    &ContextWrapper::normalize)
        .function("removeDC",               &ContextWrapper::removeDC)
        .function("precompute",             &ContextWrapper::precompute)
        // Solvers
        .function("poisson",           &ContextWrapper::poisson)
        .function("heat",&ContextWrapper::heat)
        .function("tracePath",              &ContextWrapper::tracePath)
        .function("hodge",         &ContextWrapper::hodge)
        // Geometric
        .function("curvatures",      &ContextWrapper::curvatures)
        .function("bff",   &ContextWrapper::bff)
        .function("isoline",        &ContextWrapper::isoline)
        .function("trivial",  &ContextWrapper::trivial)
        .function("streamline",       &ContextWrapper::streamline)
        // Vector field
        .function("whitney",     &ContextWrapper::whitney)
        .function("gradient",         &ContextWrapper::gradient)
        // Time-varying generators
        .function("heatDiffusion",  &ContextWrapper::heatDiffusion)
        .function("dampedWave",     &ContextWrapper::dampedWave)
        .function("randomDecomposed1Form",
                                            &ContextWrapper::randomDecomposed1Form)
        // Vector heat method
        .function("parallel",    &ContextWrapper::parallel)
        .function("extendScalar", &ContextWrapper::extendScalar)
        .function("logMap",       &ContextWrapper::logMap)
        .function("findCenter",   &ContextWrapper::findCenter)
        // Signed heat method
        .function("signedHeat",     &ContextWrapper::signedHeat)
        // Smooth direction fields
        .function("smoothFace",   &ContextWrapper::smoothFace)
        .function("smoothVertex", &ContextWrapper::smoothVertex)
        // Stripe patterns
        .function("compute",     &ContextWrapper::compute)
        .function("computeFreq", &ContextWrapper::computeFreq)
        ;

    emscripten::function("version", &getVersion);
    emscripten::function("solveEigenmodesFromTriplets",
                         &solveEigenmodesFromTriplets);
}
