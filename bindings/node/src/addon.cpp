/**
 * addon.cpp — N-API bindings for the cortical compute library.
 *
 * Exposes the compute library to Node.js/Electron. All functions
 * accept TypedArrays (Float64Array, Int32Array) and return plain
 * JavaScript objects with TypedArray fields.
 *
 * The Manifold is held in a std::shared_ptr wrapped in a
 * Napi::External so JavaScript can create a context once and reuse
 * it across multiple compute calls.
 */

#include <napi.h>
#include "nxr/compute.h"
#include "nxr/facets.h"   // OperatorsFacet definition (operators() command)

#include <cstring>
#include <memory>
#include <map>
#include <string>
#include <vector>

using namespace nxr::manifold;
using namespace nxr::manifold::solve;
using namespace nxr::manifold::ops;
using namespace nxr::manifold::ops::laplacian::connection;
using namespace nxr::manifold::transport;
using namespace nxr::manifold::connection;
using namespace nxr::manifold::parametrization;
using namespace nxr::manifold::parametrization::stripes;
using namespace nxr::manifold::geometry;
using namespace nxr::manifold::query;
using namespace nxr::field::generate;
using namespace nxr::field::interp;
using namespace nxr::field::op;
using namespace nxr::field::extract;
namespace solve_ns = nxr::manifold::solve;

// ─── Synchronous error translation ──────────────────────────
//
// Every synchronous N-API entry point catches nxr::core::Error
// and re-raises as a Napi::Error decorated with the structured
// fields the AsyncWorker path already exposes:
//
//   • .code  — string-named ErrorCode enumerator (e.g. "INVALID_INPUT")
//   • .hint  — diagnostic hint, when the source set one
//   • message — "[CODE] message [| hint: ...]" so plain catch-all
//                handlers still see a useful string.
//
// Without this, the C++ exception escapes Embind's default handler
// and JS sees an Error with the raw message but no .code, which
// breaks `if (err.code === 'INVALID_INPUT')` consumers who expect
// the same contract the AsyncWorker eigensolve path already honours.
//
// Use as: return nxrSyncCall(env, [&] -> Napi::Value { ... });
template <typename Fn>
static Napi::Value nxrSyncCall(Napi::Env env, Fn&& fn) {
    try {
        return fn();
    } catch (const nxr::core::Error& e) {
        std::string codeName(nxr::core::errorCodeName(e.code()));
        std::string hint(e.hint());
        std::string msg = "[" + codeName + "] " + e.what();
        if (!hint.empty()) msg += " | hint: " + hint;
        Napi::Error err = Napi::Error::New(env, msg);
        err.Set("code", Napi::String::New(env, codeName));
        if (!hint.empty()) err.Set("hint", Napi::String::New(env, hint));
        err.ThrowAsJavaScriptException();
        return env.Null();
    } catch (const std::exception& e) {
        Napi::Error err = Napi::Error::New(env, e.what());
        err.Set("code", Napi::String::New(env, "INTERNAL_ERROR"));
        err.ThrowAsJavaScriptException();
        return env.Null();
    }
}

// ─── Helpers: TypedArray ↔ Eigen conversion ──────────────────

// Convert a Float64Array to Eigen::VectorXd (copy)
static Eigen::VectorXd toVectorXd(const Napi::Float64Array& arr) {
    Eigen::VectorXd v(arr.ElementLength());
    std::memcpy(v.data(), arr.Data(), arr.ByteLength());
    return v;
}

// Convert Eigen::VectorXd to Float64Array (copy)
static Napi::Float64Array toFloat64Array(Napi::Env env, const Eigen::VectorXd& v) {
    auto arr = Napi::Float64Array::New(env, v.size());
    std::memcpy(arr.Data(), v.data(), v.size() * sizeof(double));
    return arr;
}

// Convert Eigen::MatrixXd (column-major) to Float64Array (row-major flat [rows*cols])
// We flatten row-major so JavaScript can index as [v*k + k_idx]
static Napi::Float64Array matrixToFloat64Array(Napi::Env env, const Eigen::MatrixXd& m) {
    int rows = static_cast<int>(m.rows());
    int cols = static_cast<int>(m.cols());
    auto arr = Napi::Float64Array::New(env, rows * cols);
    double* data = arr.Data();
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            data[r * cols + c] = m(r, c);
        }
    }
    return arr;
}

// Convert Eigen::MatrixXf (column-major) to Float32Array (row-major flat [rows*cols]).
// Used by time-varying generators that return [T, V] activity-shaped arrays —
// row-major flattening matches the Zarr `recordings/.../activity` schema so
// the renderer can slot the result directly into the activity store.
static Napi::Float32Array matrixToFloat32Array(Napi::Env env, const Eigen::MatrixXf& m) {
    int rows = static_cast<int>(m.rows());
    int cols = static_cast<int>(m.cols());
    auto arr = Napi::Float32Array::New(env, rows * cols);
    float* data = arr.Data();
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            data[r * cols + c] = m(r, c);
        }
    }
    return arr;
}

// Convert sparse matrix to COO TypedArrays: { row, col, data, shape }
static Napi::Object sparseToCOO(Napi::Env env, const Eigen::SparseMatrix<double>& M) {
    int nnz = static_cast<int>(M.nonZeros());

    auto rowArr = Napi::Int32Array::New(env, nnz);
    auto colArr = Napi::Int32Array::New(env, nnz);
    auto dataArr = Napi::Float64Array::New(env, nnz);

    int32_t* rowData = rowArr.Data();
    int32_t* colData = colArr.Data();
    double* valData = dataArr.Data();

    int k = 0;
    for (int outer = 0; outer < M.outerSize(); outer++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, outer); it; ++it) {
            rowData[k] = static_cast<int32_t>(it.row());
            colData[k] = static_cast<int32_t>(it.col());
            valData[k] = it.value();
            k++;
        }
    }

    auto result = Napi::Object::New(env);
    result.Set("row", rowArr);
    result.Set("col", colArr);
    result.Set("data", dataArr);
    result.Set("rows", Napi::Number::New(env, M.rows()));
    result.Set("cols", Napi::Number::New(env, M.cols()));
    result.Set("nnz", Napi::Number::New(env, nnz));
    return result;
}

// Convert a complex sparse matrix to COO with parallel real/imag value arrays.
// Layout matches `sparseToCOO` for compatibility with the JS-side flatten
// convention (CLAUDE.md §11), but exposes complex values as two parallel
// Float64Array streams instead of one. Used for the connection-Laplacian
// "complex" output format.
static Napi::Object sparseComplexToCOO(Napi::Env env,
                                       const Eigen::SparseMatrix<std::complex<double>>& M) {
    int nnz = static_cast<int>(M.nonZeros());

    auto rowArr     = Napi::Int32Array::New(env, nnz);
    auto colArr     = Napi::Int32Array::New(env, nnz);
    auto realArr    = Napi::Float64Array::New(env, nnz);
    auto imagArr    = Napi::Float64Array::New(env, nnz);

    int32_t* rowData  = rowArr.Data();
    int32_t* colData  = colArr.Data();
    double*  realData = realArr.Data();
    double*  imagData = imagArr.Data();

    int k = 0;
    for (int outer = 0; outer < M.outerSize(); outer++) {
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(M, outer); it; ++it) {
            rowData[k]  = static_cast<int32_t>(it.row());
            colData[k]  = static_cast<int32_t>(it.col());
            realData[k] = it.value().real();
            imagData[k] = it.value().imag();
            k++;
        }
    }

    auto result = Napi::Object::New(env);
    result.Set("row",       rowArr);
    result.Set("col",       colArr);
    result.Set("realData",  realArr);
    result.Set("imagData",  imagArr);
    result.Set("rows",      Napi::Number::New(env, M.rows()));
    result.Set("cols",      Napi::Number::New(env, M.cols()));
    result.Set("nnz",       Napi::Number::New(env, nnz));
    return result;
}

// Convert diagonal of a sparse matrix to Float64Array (for mass matrix compactness)
static Napi::Float64Array sparseDiagonalToFloat64Array(Napi::Env env, const Eigen::SparseMatrix<double>& M) {
    int n = static_cast<int>(M.rows());
    auto arr = Napi::Float64Array::New(env, n);
    double* data = arr.Data();
    for (int i = 0; i < n; i++) data[i] = 0.0;
    for (int outer = 0; outer < M.outerSize(); outer++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, outer); it; ++it) {
            if (it.row() == it.col()) data[it.row()] = it.value();
        }
    }
    return arr;
}

// ─── Manifold holder ───────────────────────────────────
// Wraps a shared_ptr<Manifold> in a Napi::External so JS can
// pass the opaque handle around. The finalizer releases the C++ object.

// Connection-Laplacian cache key: (domain, nSym, regularization, format).
// Result-level cache per CLAUDE.md rule 9 — geometry-central exposes the
// connection-Laplacian as a free assembly with no Solver class, so we
// cache the assembled output keyed by all input parameters.
using CLCacheKey = std::tuple<ConnectionDomain, int, double, ConnectionLaplacianFormat>;

struct ContextHolder {
    std::shared_ptr<Manifold> manifold;
    std::shared_ptr<ManifoldOperators> ops;     // cached after first assemble
    std::shared_ptr<DECOperators> dec;      // cached after first DEC call
    std::shared_ptr<CholeskyCache> factors;   // pre-factored Cholesky/LU, lazy
    std::shared_ptr<EigenResult> eigenmodes; // populated by EigenSolveWorker
    std::shared_ptr<VectorHeatSolver> vhm;   // lazy
    std::shared_ptr<SignedHeatSolver> shs;   // lazy
    std::shared_ptr<HeatGeodesicSolver> heatGeo;  // lazy — Cholesky pre-factor at construction
    std::map<CLCacheKey, std::shared_ptr<ConnectionLaplacian>> connectionLaplacian;  // result-level cache
};

static void FinalizeContext(Napi::Env env, ContextHolder* holder) {
    delete holder;
}

// ─── createContext(vertices, faces) → external handle ────────

Napi::Value CreateContext(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsTypedArray() || !info[1].IsTypedArray()) {
        Napi::TypeError::New(env, "createContext(vertices: Float64Array, faces: Int32Array)").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto verticesArr = info[0].As<Napi::Float64Array>();
    auto facesArr = info[1].As<Napi::Int32Array>();

    int nV = static_cast<int>(verticesArr.ElementLength() / 3);
    int nF = static_cast<int>(facesArr.ElementLength() / 3);

    auto holder = new ContextHolder();
    holder->manifold = std::make_shared<Manifold>(
        verticesArr.Data(), nV,
        facesArr.Data(), nF
    );
    holder->factors = std::make_shared<CholeskyCache>();

    return Napi::External<ContextHolder>::New(env, holder, FinalizeContext);
}

static ContextHolder* getContext(const Napi::CallbackInfo& info, int argIdx = 0) {
    if (!info[argIdx].IsExternal()) {
        Napi::TypeError::New(info.Env(), "Expected Manifold handle as argument").ThrowAsJavaScriptException();
        return nullptr;
    }
    return info[argIdx].As<Napi::External<ContextHolder>>().Data();
}

// ─── assembleManifoldOperators(m) → { cotanLaplacian, mass, vertexDualAreas, vertexNormals, ... } ───

Napi::Value AssembleMeshOperators(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        holder->ops = std::make_shared<ManifoldOperators>(
            assembleManifoldOperators(*holder->manifold)
        );
        const auto& ops = *holder->ops;

        auto result = Napi::Object::New(env);
        result.Set("cotanLaplacian",  sparseToCOO(env, ops.cotanLaplacian));
        result.Set("mass",            sparseToCOO(env, ops.mass));
        result.Set("vertexDualAreas", toFloat64Array(env, ops.vertexDualAreas));
        result.Set("vertexNormals",   matrixToFloat64Array(env, ops.vertexNormals));
        result.Set("totalArea",       Napi::Number::New(env, ops.totalArea));
        result.Set("nV",              Napi::Number::New(env, holder->manifold->nV()));
        result.Set("nE",              Napi::Number::New(env, holder->manifold->nE()));
        result.Set("nF",              Napi::Number::New(env, holder->manifold->nF()));
        return result;
    });
}

// ─── assembleDECOperators(m) → { d0, d1, hodge1, hodge1Inv } ───

Napi::Value AssembleDECOperators(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        holder->dec = std::make_shared<DECOperators>(
            assembleDECOperators(*holder->manifold)
        );
        const auto& dec = *holder->dec;

        auto result = Napi::Object::New(env);
        result.Set("d0", sparseToCOO(env, dec.d0));
        result.Set("d1", sparseToCOO(env, dec.d1));
        result.Set("hodge1", sparseDiagonalToFloat64Array(env, dec.hodge1));
        return result;
    });
}

// ─── assembleConnectionLaplacian(m, options) → { K, baseDim, ... } ───
//
// Options object: { domain?, nSym?, regularization?, format? }
//   domain         : 'vertex' (default) | 'face' | 'edge'
//   nSym           : 1 (default) | 2 | 4 | …  (must be > 0)
//   regularization : number, default 1e-8
//   format         : 'real2N' (default) | 'complex'
//
// Returns:
//   {
//     K: { row, col, data,            rows, cols }   when format == 'real2N'
//        { row, col, realData, imagData, rows, cols } when format == 'complex'
//     baseDim, outputDim,
//     domain  : 'vertex' | 'face' | 'edge',
//     nSym    : number,
//     regularization : number,
//     format  : 'real2N' | 'complex'
//   }
//
// Result-level cached on ContextHolder per (domain, nSym, regularization,
// format) tuple — repeat calls with identical options are O(1).

Napi::Value AssembleConnectionLaplacian(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        ConnectionLaplacianOptions opts;

        if (info.Length() > 1 && info[1].IsObject()) {
            Napi::Object o = info[1].As<Napi::Object>();
            if (o.Has("domain"))
                opts.domain = parseConnectionDomain(o.Get("domain").As<Napi::String>().Utf8Value());
            if (o.Has("nSym"))
                opts.nSym = o.Get("nSym").As<Napi::Number>().Int32Value();
            if (o.Has("regularization"))
                opts.regularization = o.Get("regularization").As<Napi::Number>().DoubleValue();
            if (o.Has("format"))
                opts.format = parseConnectionLaplacianFormat(o.Get("format").As<Napi::String>().Utf8Value());
        }

        const CLCacheKey key{opts.domain, opts.nSym, opts.regularization, opts.format};
        auto it = holder->connectionLaplacian.find(key);
        if (it == holder->connectionLaplacian.end()) {
            it = holder->connectionLaplacian.emplace(
                key,
                std::make_shared<ConnectionLaplacian>(
                    assembleConnectionLaplacian(*holder->manifold, opts)
                )
            ).first;
        }
        const ConnectionLaplacian& cl = *it->second;

        const char* domainStr =
            cl.domain == ConnectionDomain::Vertex              ? "vertex" :
            cl.domain == ConnectionDomain::Face                ? "face"   :
            cl.domain == ConnectionDomain::EdgeCrouzeixRaviart ? "edge"   : "?";
        const char* formatStr =
            cl.format == ConnectionLaplacianFormat::Real2N ? "real2N" : "complex";

        auto result = Napi::Object::New(env);
        if (cl.format == ConnectionLaplacianFormat::Real2N) {
            result.Set("K", sparseToCOO(env, cl.K_real));
        } else {
            result.Set("K", sparseComplexToCOO(env, cl.K_complex));
        }
        result.Set("baseDim",        Napi::Number::New(env, cl.baseDim));
        result.Set("outputDim",      Napi::Number::New(env, cl.outputDim));
        result.Set("domain",         Napi::String::New(env, domainStr));
        result.Set("nSym",           Napi::Number::New(env, cl.nSym));
        result.Set("regularization", Napi::Number::New(env, cl.regularization));
        result.Set("format",         Napi::String::New(env, formatStr));
        return result;
    });
}

// ─── operators(handle, family, arg) → COO / complex-COO / {d0,d1} ───
// Mirrors ContextWrapper::operators (bindings/wasm/src/nxr_compute_wasm.cpp).
// Same family/subtype dispatch, same arg rules (string subtype | numeric
// tau/nSym | undefined), same error messages — single source of truth is the
// Manifold operators facet, so outputs are byte-identical to the WASM binding.
Napi::Value Operators(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);            // info[0] = External handle
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        if (info.Length() < 2 || !info[1].IsString())
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators(handle, family, arg): family must be a string.");
        const std::string family = info[1].As<Napi::String>().Utf8Value();
        const Napi::Value arg = info.Length() > 2 ? info[2] : env.Undefined();
        const bool        hasNum = arg.IsNumber();
        const double      num    = hasNum ? arg.As<Napi::Number>().DoubleValue() : 0.0;
        const std::string sub    = arg.IsString() ? arg.As<Napi::String>().Utf8Value() : "";
        Manifold& m = *holder->manifold;

        if (family == "laplacian") {
            if (sub == "cotan")      return sparseToCOO(env, m.operators().laplacian().cotan());
            if (sub == "graph")      return sparseToCOO(env, m.operators().laplacian().graph());
            if (sub == "connection") return sparseComplexToCOO(env, m.operators().laplacian().connection());
            if (sub == "covariant")  return sparseToCOO(env, m.operators().laplacian().covariant(CovariantCoupling::Ambient));
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators laplacian: subtype must be cotan|graph|connection|covariant.");
        } else if (family == "mass") {
            if (sub == "lumped")   return sparseToCOO(env, m.operators().mass().lumped());
            if (sub == "galerkin") return sparseToCOO(env, m.operators().mass().galerkin());
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators mass: subtype must be lumped|galerkin.");
        } else if (family == "hodge") {
            if (sub == "h0")    return sparseToCOO(env, m.operators().hodge().h0());
            if (sub == "h1")    return sparseToCOO(env, m.operators().hodge().h1());
            if (sub == "h2")    return sparseToCOO(env, m.operators().hodge().h2());
            if (sub == "h1inv") return sparseToCOO(env, m.operators().hodge().h1inv());
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators hodge: subtype must be h0|h1|h2|h1inv.");
        } else if (family == "dec") {
            const auto& dec = m.operators().dec();
            auto obj = Napi::Object::New(env);
            obj.Set("d0", sparseToCOO(env, dec.d0));
            obj.Set("d1", sparseToCOO(env, dec.d1));
            return obj;
        } else if (family == "gradient3D") {
            return sparseToCOO(env, m.operators().gradient3D());
        } else if (family == "dirac") {
            if (!hasNum) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators dirac: expected a numeric tau, operators('dirac', tau).");
            return sparseToCOO(env, m.operators().dirac(num));
        } else if (family == "diracFace") {
            if (!hasNum) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators diracFace: expected a numeric tau, operators('diracFace', tau).");
            return sparseToCOO(env, m.operators().diracFace(num));
        } else if (family == "diracD") {
            return sparseToCOO(env, m.operators().diracD());
        } else if (family == "diracFaceD") {
            return sparseToCOO(env, m.operators().diracFaceD());
        } else if (family == "diracIntrinsicD") {
            return sparseToCOO(env, m.operators().diracIntrinsicD());
        } else if (family == "diracFaceIntrinsicD") {
            return sparseToCOO(env, m.operators().diracFaceIntrinsicD());
        } else if (family == "gradFace") {
            return sparseToCOO(env, m.operators().gradFace());
        } else if (family == "lapFace") {
            return sparseToCOO(env, m.operators().lapFace());
        } else if (family == "connectionGradient") {
            int nSym = hasNum ? static_cast<int>(num) : 1;
            return sparseComplexToCOO(env, m.operators().connectionGradient(nSym));
        } else if (family == "extrinsicWeitzenbock") {
            return sparseToCOO(env, m.operators().extrinsicWeitzenbock());
        }
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            "operators: family must be "
            "laplacian|mass|hodge|dec|gradient3D|dirac|diracFace|diracD|diracFaceD|"
            "diracIntrinsicD|diracFaceIntrinsicD|gradFace|lapFace|connectionGradient|extrinsicWeitzenbock.");
    });
}

// ─── solve(m, k, cancelArr?, progressArr?) ───────
// AsyncWorker since this can take seconds.
//
// Optional cancel and progress are SharedArrayBuffer-backed
// Int32Arrays. The worker thread polls / writes them while the
// JS main thread reads / writes from anywhere via Atomics.*.
// Layout convention (matches WASM and MEX):
//   cancelArr[0]   — non-zero ⇒ cancel requested
//   progressArr[0] — current iteration counter
//   progressArr[1] — high-water bound for iteration
//   progressArr[2] — residual × 1e6
//
// Both arrays' lifetimes are guaranteed by ctxRef_'s persistent
// reference (the JS caller must keep the arrays reachable while
// the returned Promise is pending — same contract as cancellation
// in any other Node async API).

class EigenSolveWorker : public Napi::AsyncWorker {
public:
    EigenSolveWorker(Napi::Env env, Napi::Value ctxExternal, ContextHolder* holder, int k,
                     Napi::Value cancelVal, int32_t* cancelPtr,
                     Napi::Value progressVal, int32_t* progressPtr, std::size_t progressLen)
        : Napi::AsyncWorker(env), deferred(Napi::Promise::Deferred::New(env)),
          ctxRef_(Napi::Persistent(ctxExternal)), holder_(holder), k_(k),
          cancelPtr_(cancelPtr), progressPtr_(progressPtr), progressLen_(progressLen) {
        // ctxRef_ prevents V8's GC from finalizing the External<ContextHolder>
        // while the worker thread is running — an await on the returned Promise
        // allows a GC cycle that frees the ContextHolder mid-eigensolve →
        // segfault. Same hazard for the SAB-backed Int32Arrays the worker is
        // reading; pin them with persistent references for the worker's
        // lifetime so their backing memory stays live.
        if (!cancelVal.IsUndefined() && !cancelVal.IsNull()) {
            cancelRef_ = Napi::Persistent(cancelVal);
        }
        if (!progressVal.IsUndefined() && !progressVal.IsNull()) {
            progressRef_ = Napi::Persistent(progressVal);
        }
    }

    void Execute() override {
        try {
            if (!holder_->ops) {
                holder_->ops = std::make_shared<ManifoldOperators>(
                    assembleManifoldOperators(*holder_->manifold)
                );
            }

            // Build the cancellation token from the SAB-backed Int32Array.
            // std::atomic<int32_t> is layout/alignment-compatible with int32_t
            // on every platform we ship — this reinterpret is sound in
            // practice and the canonical way to bridge a SAB cell into
            // the C++ atomic-pointer API.
            CancellationToken cancel = cancelPtr_
                ? CancellationToken(reinterpret_cast<const std::atomic<int32_t>*>(cancelPtr_))
                : CancellationToken{};

            ProgressObserver progress;
            if (progressPtr_ && progressLen_ >= 3) {
                progress.iteration       = reinterpret_cast<std::atomic<int32_t>*>(progressPtr_ + 0);
                progress.totalIterations = reinterpret_cast<std::atomic<int32_t>*>(progressPtr_ + 1);
                progress.residualMicro   = reinterpret_cast<std::atomic<int32_t>*>(progressPtr_ + 2);
            }

            result_ = solve::eigen(
                holder_->ops->cotanLaplacian, holder_->ops->mass, k_,
                -1e-8, /*normalize=*/true, /*removeDC=*/true, cancel, progress);
            // Cache on the holder so time-varying generators can reuse
            // without round-tripping the eigenmode arrays through IPC.
            holder_->eigenmodes = std::make_shared<EigenResult>(result_);
        } catch (const nxr::core::Error& e) {
            // Preserve structured error info for OnError to attach to the JS Error.
            errorCode_   = std::string(nxr::core::errorCodeName(e.code()));
            errorHint_   = std::string(e.hint());
            SetError(e.what());
        } catch (const std::exception& e) {
            errorCode_ = "INTERNAL_ERROR";
            SetError(e.what());
        }
    }

    void OnOK() override {
        Napi::Env env = Env();
        auto obj = Napi::Object::New(env);
        // vMajor row-major: matches Zarr `[V, K]` schema and renderer expectations.
        obj.Set("eigenvectors", matrixToFloat64Array(env, result_.eigenvectors));
        obj.Set("eigenvalues", toFloat64Array(env, result_.eigenvalues));
        obj.Set("k", Napi::Number::New(env, result_.k));
        obj.Set("nV", Napi::Number::New(env, result_.eigenvectors.rows()));
        deferred.Resolve(obj);
    }

    void OnError(const Napi::Error& e) override {
        Napi::Env env = Env();
        // Attach structured fields to the Error so JS can switch on .code
        // and surface .hint without parsing the message.
        Napi::Error err = Napi::Error::New(env, e.Message());
        if (!errorCode_.empty()) err.Set("code", Napi::String::New(env, errorCode_));
        if (!errorHint_.empty()) err.Set("hint", Napi::String::New(env, errorHint_));
        deferred.Reject(err.Value());
    }

    Napi::Promise GetPromise() { return deferred.Promise(); }

private:
    Napi::Promise::Deferred deferred;
    Napi::Reference<Napi::Value> ctxRef_;       // prevent GC during async work
    Napi::Reference<Napi::Value> cancelRef_;    // pin SAB-backed Int32Array
    Napi::Reference<Napi::Value> progressRef_;  // pin SAB-backed Int32Array
    ContextHolder* holder_;
    int k_;
    int32_t*    cancelPtr_;        // nullable; points into a SAB cell
    int32_t*    progressPtr_;      // nullable; points into a SAB cell
    std::size_t progressLen_;      // element count, for safety-bound check
    EigenResult result_;
    std::string errorCode_;
    std::string errorHint_;
};

Napi::Value SolveEigenmodes(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    int k = info[1].As<Napi::Number>().Int32Value();

    // Optional cancel + progress arrays. Caller is expected to back
    // them with a SharedArrayBuffer so flips are visible cross-thread,
    // but a normal Int32Array also works (single-threaded JS still
    // sees writes between awaits).
    int32_t*    cancelPtr   = nullptr;
    int32_t*    progressPtr = nullptr;
    std::size_t progressLen = 0;
    Napi::Value cancelVal   = env.Undefined();
    Napi::Value progressVal = env.Undefined();
    if (info.Length() > 2 && info[2].IsTypedArray()) {
        auto arr = info[2].As<Napi::Int32Array>();
        if (arr.ElementLength() >= 1) {
            cancelPtr = arr.Data();
            cancelVal = info[2];
        }
    }
    if (info.Length() > 3 && info[3].IsTypedArray()) {
        auto arr = info[3].As<Napi::Int32Array>();
        progressPtr = arr.Data();
        progressLen = arr.ElementLength();
        progressVal = info[3];
    }

    auto* worker = new EigenSolveWorker(env, info[0], holder, k,
                                        cancelVal, cancelPtr,
                                        progressVal, progressPtr, progressLen);
    auto promise = worker->GetPromise();
    worker->Queue();
    return promise;
}

// ─── poisson(m, densityVertices, densityValues) ───────

Napi::Value SolvePoisson(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        auto vertsArr = info[1].As<Napi::Int32Array>();
        auto valuesArr = info[2].As<Napi::Float64Array>();

        if (!holder->ops) {
            holder->ops = std::make_shared<ManifoldOperators>(
                assembleManifoldOperators(*holder->manifold)
            );
        }

        std::map<int, double> densityMap;
        for (size_t i = 0; i < vertsArr.ElementLength(); i++) {
            densityMap[vertsArr[i]] = valuesArr[i];
        }

        Eigen::VectorXd phi = solve_ns::poisson(*holder->ops, *holder->factors, densityMap);
        return toFloat64Array(env, phi);
    });
}

// ─── heat(m, sourceVertices) ────────────

Napi::Value ComputeGeodesicDistance(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        auto sourcesArr = info[1].As<Napi::Int32Array>();
        std::vector<int> sources;
        sources.reserve(sourcesArr.ElementLength());
        for (size_t i = 0; i < sourcesArr.ElementLength(); i++) {
            sources.push_back(sourcesArr[i]);
        }

        // Lazy-construct the stateful HeatGeodesicSolver and reuse
        // across calls — its constructor pre-factors the Cholesky
        // systems (M+tA, A); subsequent calls do back-substitution
        // only. Same lifetime model as VectorHeatSolver / SignedHeatSolver.
        if (!holder->heatGeo) {
            holder->heatGeo = std::make_shared<HeatGeodesicSolver>(*holder->manifold);
        }
        Eigen::VectorXd dists = heat(*holder->heatGeo, sources);
        return toFloat64Array(env, dists);
    });
}

// ─── hodge(m, omega?) — returns all scalars + vectors ─

Napi::Value HodgeDecompose(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        if (!holder->dec) {
            holder->dec = std::make_shared<DECOperators>(
                assembleDECOperators(*holder->manifold)
            );
        }

        // Use provided omega or generate random
        Eigen::VectorXd omega;
        if (info.Length() > 1 && info[1].IsTypedArray()) {
            omega = toVectorXd(info[1].As<Napi::Float64Array>());
        } else {
            omega = randomOmega(holder->manifold->nE());
        }

        HodgeResult result = solve::hodge(*holder->manifold, *holder->dec, *holder->factors, omega);

        auto obj = Napi::Object::New(env);
        // Scalar potentials
        obj.Set("alpha", toFloat64Array(env, result.exactPotential));
        obj.Set("beta", toFloat64Array(env, result.coExactPotentialV));
        obj.Set("combined", toFloat64Array(env, result.combinedPotential));
        // Input 1-form and its components
        obj.Set("omega", toFloat64Array(env, result.omega));
        obj.Set("dAlpha", toFloat64Array(env, result.dAlpha));
        obj.Set("deltaBeta", toFloat64Array(env, result.deltaBeta));
        obj.Set("gamma", toFloat64Array(env, result.gamma));
        // Face-centered vector fields [nF * 3]
        obj.Set("omegaVectors", matrixToFloat64Array(env, result.omegaVectors));
        obj.Set("dAlphaVectors", matrixToFloat64Array(env, result.dAlphaVectors));
        obj.Set("deltaBetaVectors", matrixToFloat64Array(env, result.deltaBetaVectors));
        obj.Set("gammaVectors", matrixToFloat64Array(env, result.gammaVectors));
        return obj;
    });
}

// ─── curvatures(m) → { gaussian, mean, kMin, kMax, principalDir } ──

Napi::Value ComputeCurvatures(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        CurvatureResult result = curvatures(*holder->manifold);

        auto obj = Napi::Object::New(env);
        obj.Set("gaussian", toFloat64Array(env, result.gaussian));
        obj.Set("mean", toFloat64Array(env, result.mean));
        obj.Set("kMin", toFloat64Array(env, result.kMin));
        obj.Set("kMax", toFloat64Array(env, result.kMax));
        obj.Set("principalDir", matrixToFloat64Array(env, result.principalDirMax));
        return obj;
    });
}

// ─── normals(m, type) → Float64Array [nV*3] ──

Napi::Value ComputeVertexNormals(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        int typeInt = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : 0;
        NormalType type = static_cast<NormalType>(typeInt);

        Eigen::MatrixXd N = normals(*holder->manifold, type);
        return matrixToFloat64Array(env, N);
    });
}

// ─── whitney(m, oneForm) → Float64Array [nF*3] ──

Napi::Value WhitneyInterpolate(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        auto oneFormArr = info[1].As<Napi::Float64Array>();
        Eigen::VectorXd oneForm = toVectorXd(oneFormArr);

        if (!holder->dec) {
            holder->dec = std::make_shared<DECOperators>(
                assembleDECOperators(*holder->manifold)
            );
        }

        Eigen::MatrixXd vectors = whitney(*holder->manifold, *holder->dec, oneForm);
        return matrixToFloat64Array(env, vectors);
    });
}

// ─── gradient(m, vertexScalars) → Float64Array [nF*3] ──

Napi::Value ScalarGradient(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        auto scalarsArr = info[1].As<Napi::Float64Array>();
        Eigen::VectorXd scalars = toVectorXd(scalarsArr);

        Eigen::MatrixXd G = gradient(*holder->manifold, scalars);
        return matrixToFloat64Array(env, G);
    });
}

// ─── isoline(m, scalars, numLevels, min, max) → { positions, count } ──

Napi::Value ComputeIsolines(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        auto scalarsArr = info[1].As<Napi::Float64Array>();
        Eigen::VectorXd scalars = toVectorXd(scalarsArr);

        int numLevels = info.Length() > 2 ? info[2].As<Napi::Number>().Int32Value() : 20;
        double minVal = info.Length() > 3 ? info[3].As<Napi::Number>().DoubleValue() : 0.0;
        double maxVal = info.Length() > 4 ? info[4].As<Napi::Number>().DoubleValue() : 0.0;

        IsolineResult result = isoline(*holder->manifold, scalars, numLevels, minVal, maxVal);

        auto obj = Napi::Object::New(env);
        obj.Set("positions", matrixToFloat64Array(env, result.positions));
        obj.Set("segmentCount", Napi::Number::New(env, result.segmentCount));
        return obj;
    });
}

// ─── trivial(m, singVerts, singValues) → { ... } ──

Napi::Value ComputeDirectionField(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        auto vertsArr = info[1].As<Napi::Int32Array>();
        auto valsArr = info[2].As<Napi::Float64Array>();

        if (!holder->dec) {
            holder->dec = std::make_shared<DECOperators>(
                assembleDECOperators(*holder->manifold)
            );
        }

        std::map<int, double> singMap;
        for (size_t i = 0; i < vertsArr.ElementLength(); i++) {
            singMap[vertsArr[i]] = valsArr[i];
        }

        DirectionFieldResult result = trivial(*holder->manifold, *holder->dec, *holder->factors, singMap);

        auto obj = Napi::Object::New(env);
        obj.Set("connections", toFloat64Array(env, result.connections));
        obj.Set("directionVectors", matrixToFloat64Array(env, result.directionVectors));
        obj.Set("orthogonalVectors", matrixToFloat64Array(env, result.orthogonalVectors));
        obj.Set("eulerCharacteristic", Napi::Number::New(env, result.eulerCharacteristic));
        obj.Set("gaussBonnetSatisfied", Napi::Boolean::New(env, result.gaussBonnetSatisfied));
        return obj;
    });
}

// ─── streamline(m, faceField, numSeeds, stepCoef, maxSteps) → { positions, count } ──

Napi::Value TraceStreamlines(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        auto fieldArr = info[1].As<Napi::Float64Array>();
        int nF = holder->manifold->nF();
        Eigen::MatrixXd faceField(nF, 3);
        const double* data = fieldArr.Data();
        for (int i = 0; i < nF; i++) {
            faceField(i, 0) = data[i * 3 + 0];
            faceField(i, 1) = data[i * 3 + 1];
            faceField(i, 2) = data[i * 3 + 2];
        }

        int numSeeds = info.Length() > 2 ? info[2].As<Napi::Number>().Int32Value() : 15;
        double stepCoef = info.Length() > 3 ? info[3].As<Napi::Number>().DoubleValue() : 0.15;
        int maxSteps = info.Length() > 4 ? info[4].As<Napi::Number>().Int32Value() : 1000;

        StreamlineResult result = streamline(*holder->manifold, faceField, numSeeds, stepCoef, maxSteps);

        auto obj = Napi::Object::New(env);
        obj.Set("positions", matrixToFloat64Array(env, result.positions));
        obj.Set("segmentCount", Napi::Number::New(env, result.segmentCount));
        return obj;
    });
}

// ─── heatDiffusion(m, {sources, timesteps, alpha}) ────
//
// Returns a [T*nV] Float32Array (row-major frame-major) plus T and
// nV. Eigenmodes must already be computed via solve.

Napi::Value GenerateHeatDiffusion(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        if (!holder->eigenmodes) {
            throw nxr::core::Error(nxr::core::ErrorCode::NotPrecomputed,
                "heatDiffusion: eigenmodes not computed",
                "call solve first");
        }
        if (!holder->ops) {
            holder->ops = std::make_shared<ManifoldOperators>(
                assembleManifoldOperators(*holder->manifold)
            );
        }

        // args[1] = options object
        auto opts = info[1].As<Napi::Object>();
        auto sourceVertsArr = opts.Get("sourceVerts").As<Napi::Int32Array>();
        auto sourceValuesArr = opts.Get("sourceValues").As<Napi::Float64Array>();
        auto timestepsArr = opts.Get("timesteps").As<Napi::Float64Array>();
        double alpha = opts.Get("alpha").As<Napi::Number>().DoubleValue();

        int nV = holder->manifold->nV();
        std::map<int, double> sourcesMap;
        for (size_t i = 0; i < sourceVertsArr.ElementLength(); i++) {
            sourcesMap[sourceVertsArr[i]] = sourceValuesArr[i];
        }
        Eigen::VectorXd u0 = delta(nV, sourcesMap);

        std::vector<double> timesteps(
            timestepsArr.Data(),
            timestepsArr.Data() + timestepsArr.ElementLength()
        );

        Eigen::MatrixXf field = heatDiffusion(
            *holder->ops, *holder->eigenmodes, u0, timesteps, alpha);

        auto result = Napi::Object::New(env);
        result.Set("data", matrixToFloat32Array(env, field));
        result.Set("T", Napi::Number::New(env, static_cast<int>(field.rows())));
        result.Set("nV", Napi::Number::New(env, static_cast<int>(field.cols())));
        return result;
    });
}

// ─── dampedWave(m, {modeIndices, amplitudes, dampings, phases, timesteps}) ─

Napi::Value GenerateDampedWave(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    return nxrSyncCall(env, [&]() -> Napi::Value {
        if (!holder->eigenmodes) {
            throw nxr::core::Error(nxr::core::ErrorCode::NotPrecomputed,
                "dampedWave: eigenmodes not computed",
                "call solve first");
        }

        auto opts = info[1].As<Napi::Object>();
        auto modeIdxArr  = opts.Get("modeIndices").As<Napi::Int32Array>();
        auto ampsArr     = opts.Get("amplitudes").As<Napi::Float64Array>();
        auto dampsArr    = opts.Get("dampings").As<Napi::Float64Array>();
        auto phasesArr   = opts.Get("phases").As<Napi::Float64Array>();
        auto tsArr       = opts.Get("timesteps").As<Napi::Float64Array>();

        std::vector<int> modeIndices(
            modeIdxArr.Data(),
            modeIdxArr.Data() + modeIdxArr.ElementLength()
        );
        std::vector<double> amplitudes(
            ampsArr.Data(),
            ampsArr.Data() + ampsArr.ElementLength()
        );
        std::vector<double> dampings(
            dampsArr.Data(),
            dampsArr.Data() + dampsArr.ElementLength()
        );
        std::vector<double> phases(
            phasesArr.Data(),
            phasesArr.Data() + phasesArr.ElementLength()
        );
        std::vector<double> timesteps(
            tsArr.Data(),
            tsArr.Data() + tsArr.ElementLength()
        );

        Eigen::MatrixXf field = dampedWave(
            *holder->eigenmodes, modeIndices, amplitudes, dampings, phases, timesteps);

        auto result = Napi::Object::New(env);
        result.Set("data", matrixToFloat32Array(env, field));
        result.Set("T", Napi::Number::New(env, static_cast<int>(field.rows())));
        result.Set("nV", Napi::Number::New(env, static_cast<int>(field.cols())));
        return result;
    });
}

// ─── Vector heat method ──────────────────────────────────────

static void ensureVHM(ContextHolder* holder) {
    if (!holder->vhm) {
        holder->vhm = std::make_shared<VectorHeatSolver>(*holder->manifold);
    }
}
static void ensureSHS(ContextHolder* holder) {
    if (!holder->shs) {
        holder->shs = std::make_shared<SignedHeatSolver>(*holder->manifold);
    }
}

Napi::Value VectorHeatTransport(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        auto vertsArr = info[1].As<Napi::Int32Array>();
        auto vecsArr  = info[2].As<Napi::Float64Array>();
        int n = static_cast<int>(vertsArr.ElementLength());
        if (vecsArr.ElementLength() != static_cast<size_t>(n) * 3) {
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "sourceVectors must be Nx3 (length 3*sources)");
        }
        std::vector<int> verts(vertsArr.Data(), vertsArr.Data() + n);
        Eigen::MatrixXd S(n, 3);
        const double* vd = vecsArr.Data();
        for (int i = 0; i < n; i++) {
            S(i, 0) = vd[i * 3 + 0];
            S(i, 1) = vd[i * 3 + 1];
            S(i, 2) = vd[i * 3 + 2];
        }

        ensureVHM(holder);
        Eigen::MatrixXd out = parallel(*holder->vhm, verts, S);
        return matrixToFloat64Array(env, out);
    });
}

Napi::Value VectorHeatExtendScalar(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        auto vertsArr = info[1].As<Napi::Int32Array>();
        auto valsArr  = info[2].As<Napi::Float64Array>();
        int n = static_cast<int>(vertsArr.ElementLength());
        if (valsArr.ElementLength() != static_cast<size_t>(n)) {
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "sourceVerts and sourceValues must have the same length");
        }
        std::vector<int> verts(vertsArr.Data(), vertsArr.Data() + n);
        Eigen::VectorXd vals(n);
        std::memcpy(vals.data(), valsArr.Data(), n * sizeof(double));

        ensureVHM(holder);
        Eigen::VectorXd out = extendScalar(*holder->vhm, verts, vals);
        return toFloat64Array(env, out);
    });
}

Napi::Value VectorHeatLogMap(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        int sourceVertex = info[1].As<Napi::Number>().Int32Value();
        int strategyInt  = info.Length() > 2
            ? info[2].As<Napi::Number>().Int32Value()
            : static_cast<int>(LogMapStrategy::AffineLocal);
        auto strategy = static_cast<LogMapStrategy>(strategyInt);

        ensureVHM(holder);
        LogMapResult r = logMap(*holder->vhm, sourceVertex, strategy);

        auto obj = Napi::Object::New(env);
        obj.Set("logCoords", matrixToFloat64Array(env, r.logCoords));
        auto e1 = Napi::Float64Array::New(env, 3);
        auto e2 = Napi::Float64Array::New(env, 3);
        for (int i = 0; i < 3; i++) {
            e1[i] = r.sourceE1(i);
            e2[i] = r.sourceE2(i);
        }
        obj.Set("sourceE1", e1);
        obj.Set("sourceE2", e2);
        return obj;
    });
}

Napi::Value VectorHeatFindCenter(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        auto vertsArr = info[1].As<Napi::Int32Array>();
        int p = info.Length() > 2 ? info[2].As<Napi::Number>().Int32Value() : 2;
        std::vector<int> verts(vertsArr.Data(), vertsArr.Data() + vertsArr.ElementLength());

        ensureVHM(holder);
        Eigen::Vector3d c = findCenter(*holder->vhm, verts, p);
        auto out = Napi::Float64Array::New(env, 3);
        out[0] = c.x(); out[1] = c.y(); out[2] = c.z();
        return out;
    });
}

// ─── Signed heat method ──────────────────────────────────────

Napi::Value SignedHeatDistance(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        auto vertsArr = info[1].As<Napi::Int32Array>();
        bool isLoop = info.Length() > 2 ? info[2].As<Napi::Boolean>().Value() : true;
        int  lsInt  = info.Length() > 3
            ? info[3].As<Napi::Number>().Int32Value()
            : static_cast<int>(SignedHeatLevelSet::ZeroSet);
        auto ls = static_cast<SignedHeatLevelSet>(lsInt);

        std::vector<int> verts(vertsArr.Data(), vertsArr.Data() + vertsArr.ElementLength());

        ensureSHS(holder);
        Eigen::VectorXd out = signedHeat(*holder->shs, verts, isLoop, ls);
        return toFloat64Array(env, out);
    });
}

// ─── Smooth direction fields ────────────────────────────────

Napi::Value ComputeSmoothFaceField(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        int  nSym             = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : 4;
        bool alignToCurvature = info.Length() > 2 ? info[2].As<Napi::Boolean>().Value()     : false;

        Eigen::MatrixXd v = smoothFace(*holder->manifold, nSym, alignToCurvature);
        return matrixToFloat64Array(env, v);
    });
}

Napi::Value ComputeSmoothVertexField(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        int  nSym             = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : 2;
        bool alignToCurvature = info.Length() > 2 ? info[2].As<Napi::Boolean>().Value()     : false;

        SmoothVertexFieldResult r = smoothVertex(*holder->manifold, nSym, alignToCurvature);
        auto obj = Napi::Object::New(env);
        obj.Set("vertexVectors",  matrixToFloat64Array(env, r.vertexVectors));
        obj.Set("vertexFieldRaw", toFloat64Array(env, r.vertexFieldRaw));
        obj.Set("nSym", Napi::Number::New(env, r.nSym));
        return obj;
    });
}

// ─── Stripe patterns ────────────────────────────────────────

Napi::Value ComputeStripePattern(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        auto fieldArr = info[1].As<Napi::Float64Array>();
        Eigen::VectorXd raw = toVectorXd(fieldArr);
        double freq   = info[2].As<Napi::Number>().DoubleValue();
        bool   connect = info.Length() > 3 ? info[3].As<Napi::Boolean>().Value() : true;

        StripePatternResult r = compute(*holder->manifold, raw, freq, connect);
        auto obj = Napi::Object::New(env);
        obj.Set("positions",    matrixToFloat64Array(env, r.positions));
        obj.Set("segmentCount", Napi::Number::New(env, r.segmentCount));
        return obj;
    });
}

Napi::Value ComputeStripePatternFreq(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        auto fieldArr = info[1].As<Napi::Float64Array>();
        auto freqsArr = info[2].As<Napi::Float64Array>();
        Eigen::VectorXd raw   = toVectorXd(fieldArr);
        Eigen::VectorXd freqs = toVectorXd(freqsArr);
        bool connect = info.Length() > 3 ? info[3].As<Napi::Boolean>().Value() : true;

        StripePatternResult r = computeFreq(*holder->manifold, raw, freqs, connect);
        auto obj = Napi::Object::New(env);
        obj.Set("positions",    matrixToFloat64Array(env, r.positions));
        obj.Set("segmentCount", Napi::Number::New(env, r.segmentCount));
        return obj;
    });
}

// ─── Module Init ─────────────────────────────────────────────

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("createContext", Napi::Function::New(env, CreateContext));
    exports.Set("assembleManifoldOperators", Napi::Function::New(env, AssembleMeshOperators));
    exports.Set("assembleDECOperators", Napi::Function::New(env, AssembleDECOperators));
    exports.Set("assembleConnectionLaplacian", Napi::Function::New(env, AssembleConnectionLaplacian));
    exports.Set("operators", Napi::Function::New(env, Operators));
    exports.Set("solve", Napi::Function::New(env, SolveEigenmodes));
    exports.Set("poisson", Napi::Function::New(env, SolvePoisson));
    exports.Set("heat", Napi::Function::New(env, ComputeGeodesicDistance));
    exports.Set("hodge", Napi::Function::New(env, HodgeDecompose));
    exports.Set("curvatures", Napi::Function::New(env, ComputeCurvatures));
    exports.Set("normals", Napi::Function::New(env, ComputeVertexNormals));
    exports.Set("whitney", Napi::Function::New(env, WhitneyInterpolate));
    exports.Set("gradient", Napi::Function::New(env, ScalarGradient));
    exports.Set("isoline", Napi::Function::New(env, ComputeIsolines));
    exports.Set("trivial", Napi::Function::New(env, ComputeDirectionField));
    exports.Set("streamline", Napi::Function::New(env, TraceStreamlines));
    exports.Set("heatDiffusion", Napi::Function::New(env, GenerateHeatDiffusion));
    exports.Set("dampedWave", Napi::Function::New(env, GenerateDampedWave));
    // Vector heat method
    exports.Set("parallel",    Napi::Function::New(env, VectorHeatTransport));
    exports.Set("extendScalar", Napi::Function::New(env, VectorHeatExtendScalar));
    exports.Set("logMap",       Napi::Function::New(env, VectorHeatLogMap));
    exports.Set("findCenter",   Napi::Function::New(env, VectorHeatFindCenter));
    // Signed heat method
    exports.Set("signedHeat",     Napi::Function::New(env, SignedHeatDistance));
    // Smooth direction fields
    exports.Set("smoothFace",   Napi::Function::New(env, ComputeSmoothFaceField));
    exports.Set("smoothVertex", Napi::Function::New(env, ComputeSmoothVertexField));
    // Stripe patterns
    exports.Set("compute",     Napi::Function::New(env, ComputeStripePattern));
    exports.Set("computeFreq", Napi::Function::New(env, ComputeStripePatternFreq));
    return exports;
}

NODE_API_MODULE(nxr_compute_addon, Init)
