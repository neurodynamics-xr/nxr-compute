/**
 * addon.cpp — N-API bindings for the cortical compute library.
 *
 * Exposes the compute library to Node.js/Electron. All functions
 * accept TypedArrays (Float64Array, Int32Array) and return plain
 * JavaScript objects with TypedArray fields.
 *
 * The ComputeContext is held in a std::shared_ptr wrapped in a
 * Napi::External so JavaScript can create a context once and reuse
 * it across multiple compute calls.
 */

#include <napi.h>
#include "nxr/compute.h"

#include <cstring>
#include <memory>
#include <map>
#include <vector>

using namespace nxr::compute;

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

// ─── ComputeContext holder ───────────────────────────────────
// Wraps a shared_ptr<ComputeContext> in a Napi::External so JS can
// pass the opaque handle around. The finalizer releases the C++ object.

struct ContextHolder {
    std::shared_ptr<ComputeContext> ctx;
    std::shared_ptr<MeshOperators> ops;     // cached after first assemble
    std::shared_ptr<DECOperators> dec;      // cached after first DEC call
    std::shared_ptr<CholeskyCache> factors;   // pre-factored Cholesky/LU, lazy
    std::shared_ptr<EigenResult> eigenmodes; // populated by EigenSolveWorker
    std::shared_ptr<VectorHeatSolver> vhm;   // lazy
    std::shared_ptr<SignedHeatSolver> shs;   // lazy
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
    holder->ctx = std::make_shared<ComputeContext>(
        verticesArr.Data(), nV,
        facesArr.Data(), nF
    );
    holder->factors = std::make_shared<CholeskyCache>();

    return Napi::External<ContextHolder>::New(env, holder, FinalizeContext);
}

static ContextHolder* getContext(const Napi::CallbackInfo& info, int argIdx = 0) {
    if (!info[argIdx].IsExternal()) {
        Napi::TypeError::New(info.Env(), "Expected ComputeContext handle as argument").ThrowAsJavaScriptException();
        return nullptr;
    }
    return info[argIdx].As<Napi::External<ContextHolder>>().Data();
}

// ─── assembleMeshOperators(ctx) → { stiffness, mass, normals, ... } ───

Napi::Value AssembleMeshOperators(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    holder->ops = std::make_shared<MeshOperators>(
        assembleMeshOperators(*holder->ctx)
    );
    const auto& ops = *holder->ops;

    auto result = Napi::Object::New(env);
    result.Set("stiffness", sparseToCOO(env, ops.stiffness));
    result.Set("massDiag", toFloat64Array(env, ops.vertexAreas));
    result.Set("normals", matrixToFloat64Array(env, ops.normals));
    result.Set("totalArea", Napi::Number::New(env, ops.totalArea));
    result.Set("nV", Napi::Number::New(env, ops.nV));
    result.Set("nE", Napi::Number::New(env, ops.nE));
    result.Set("nF", Napi::Number::New(env, ops.nF));
    return result;
}

// ─── assembleDECOperators(ctx) → { d0, d1, hodge1, hodge1Inv } ───

Napi::Value AssembleDECOperators(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    holder->dec = std::make_shared<DECOperators>(
        assembleDECOperators(*holder->ctx)
    );
    const auto& dec = *holder->dec;

    auto result = Napi::Object::New(env);
    result.Set("d0", sparseToCOO(env, dec.d0));
    result.Set("d1", sparseToCOO(env, dec.d1));
    result.Set("hodge1", sparseDiagonalToFloat64Array(env, dec.hodge1));
    return result;
}

// ─── solveEigenmodes(ctx, k, cancelArr?, progressArr?) ───────
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
                holder_->ops = std::make_shared<MeshOperators>(
                    assembleMeshOperators(*holder_->ctx)
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

            result_ = solveEigenmodes(holder_->ops->stiffness, holder_->ops->mass, k_,
                                      -1e-8, cancel, progress);
            // Normalize eigenvectors
            result_.eigenvectors = normalizeEigenmodes(result_.eigenvectors, holder_->ops->mass);
            // Remove DC
            result_ = removeDC(result_);
            // Cache on the holder so time-varying generators can reuse
            // without round-tripping the eigenmode arrays through IPC.
            holder_->eigenmodes = std::make_shared<EigenResult>(result_);
        } catch (const nxr::compute::Error& e) {
            // Preserve structured error info for OnError to attach to the JS Error.
            errorCode_   = std::string(nxr::compute::errorCodeName(e.code()));
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

// ─── solvePoisson(ctx, densityVertices, densityValues) ───────

Napi::Value SolvePoisson(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    auto vertsArr = info[1].As<Napi::Int32Array>();
    auto valuesArr = info[2].As<Napi::Float64Array>();

    if (!holder->ops) {
        holder->ops = std::make_shared<MeshOperators>(
            assembleMeshOperators(*holder->ctx)
        );
    }

    std::map<int, double> densityMap;
    for (size_t i = 0; i < vertsArr.ElementLength(); i++) {
        densityMap[vertsArr[i]] = valuesArr[i];
    }

    Eigen::VectorXd phi = solvePoisson(*holder->ops, *holder->factors, densityMap);
    return toFloat64Array(env, phi);
}

// ─── computeGeodesicDistance(ctx, sourceVertices) ────────────

Napi::Value ComputeGeodesicDistance(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    auto sourcesArr = info[1].As<Napi::Int32Array>();
    std::vector<int> sources;
    sources.reserve(sourcesArr.ElementLength());
    for (size_t i = 0; i < sourcesArr.ElementLength(); i++) {
        sources.push_back(sourcesArr[i]);
    }

    Eigen::VectorXd dists = computeGeodesicDistance(*holder->ctx, sources);
    return toFloat64Array(env, dists);
}

// ─── hodgeDecompose(ctx, omega?) — returns all scalars + vectors ─

Napi::Value HodgeDecompose(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    if (!holder->dec) {
        holder->dec = std::make_shared<DECOperators>(
            assembleDECOperators(*holder->ctx)
        );
    }

    // Use provided omega or generate random
    Eigen::VectorXd omega;
    if (info.Length() > 1 && info[1].IsTypedArray()) {
        omega = toVectorXd(info[1].As<Napi::Float64Array>());
    } else {
        omega = generateRandomOmega(holder->ctx->nE());
    }

    HodgeResult result = hodgeDecompose(*holder->ctx, *holder->dec, *holder->factors, omega);

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
}

// ─── computeCurvatures(ctx) → { gaussian, mean, kMin, kMax, principalDir } ──

Napi::Value ComputeCurvatures(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    CurvatureResult result = computeCurvatures(*holder->ctx);

    auto obj = Napi::Object::New(env);
    obj.Set("gaussian", toFloat64Array(env, result.gaussian));
    obj.Set("mean", toFloat64Array(env, result.mean));
    obj.Set("kMin", toFloat64Array(env, result.kMin));
    obj.Set("kMax", toFloat64Array(env, result.kMax));
    obj.Set("principalDir", matrixToFloat64Array(env, result.principalDirMax));
    return obj;
}

// ─── computeVertexNormals(ctx, type) → Float64Array [nV*3] ──

Napi::Value ComputeVertexNormals(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    int typeInt = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : 0;
    NormalType type = static_cast<NormalType>(typeInt);

    Eigen::MatrixXd normals = computeVertexNormals(*holder->ctx, type);
    return matrixToFloat64Array(env, normals);
}

// ─── whitneyInterpolate(ctx, oneForm) → Float64Array [nF*3] ──

Napi::Value WhitneyInterpolate(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    auto oneFormArr = info[1].As<Napi::Float64Array>();
    Eigen::VectorXd oneForm = toVectorXd(oneFormArr);

    if (!holder->dec) {
        holder->dec = std::make_shared<DECOperators>(
            assembleDECOperators(*holder->ctx)
        );
    }

    Eigen::MatrixXd vectors = whitneyInterpolate(*holder->ctx, *holder->dec, oneForm);
    return matrixToFloat64Array(env, vectors);
}

// ─── scalarGradient(ctx, vertexScalars) → Float64Array [nF*3] ──

Napi::Value ScalarGradient(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    auto scalarsArr = info[1].As<Napi::Float64Array>();
    Eigen::VectorXd scalars = toVectorXd(scalarsArr);

    Eigen::MatrixXd gradient = scalarGradient(*holder->ctx, scalars);
    return matrixToFloat64Array(env, gradient);
}

// ─── computeIsolines(ctx, scalars, numLevels, min, max) → { positions, count } ──

Napi::Value ComputeIsolines(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    auto scalarsArr = info[1].As<Napi::Float64Array>();
    Eigen::VectorXd scalars = toVectorXd(scalarsArr);

    int numLevels = info.Length() > 2 ? info[2].As<Napi::Number>().Int32Value() : 20;
    double minVal = info.Length() > 3 ? info[3].As<Napi::Number>().DoubleValue() : 0.0;
    double maxVal = info.Length() > 4 ? info[4].As<Napi::Number>().DoubleValue() : 0.0;

    IsolineResult result = computeIsolines(*holder->ctx, scalars, numLevels, minVal, maxVal);

    auto obj = Napi::Object::New(env);
    obj.Set("positions", matrixToFloat64Array(env, result.positions));
    obj.Set("segmentCount", Napi::Number::New(env, result.segmentCount));
    return obj;
}

// ─── computeDirectionField(ctx, singVerts, singValues) → { ... } ──

Napi::Value ComputeDirectionField(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    auto vertsArr = info[1].As<Napi::Int32Array>();
    auto valsArr = info[2].As<Napi::Float64Array>();

    if (!holder->dec) {
        holder->dec = std::make_shared<DECOperators>(
            assembleDECOperators(*holder->ctx)
        );
    }

    std::map<int, double> singMap;
    for (size_t i = 0; i < vertsArr.ElementLength(); i++) {
        singMap[vertsArr[i]] = valsArr[i];
    }

    DirectionFieldResult result = computeDirectionField(*holder->ctx, *holder->dec, *holder->factors, singMap);

    auto obj = Napi::Object::New(env);
    obj.Set("connections", toFloat64Array(env, result.connections));
    obj.Set("directionVectors", matrixToFloat64Array(env, result.directionVectors));
    obj.Set("orthogonalVectors", matrixToFloat64Array(env, result.orthogonalVectors));
    obj.Set("eulerCharacteristic", Napi::Number::New(env, result.eulerCharacteristic));
    obj.Set("gaussBonnetSatisfied", Napi::Boolean::New(env, result.gaussBonnetSatisfied));
    return obj;
}

// ─── traceStreamlines(ctx, faceField, numSeeds, stepCoef, maxSteps) → { positions, count } ──

Napi::Value TraceStreamlines(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    auto fieldArr = info[1].As<Napi::Float64Array>();
    int nF = holder->ctx->nF();
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

    StreamlineResult result = traceStreamlines(*holder->ctx, faceField, numSeeds, stepCoef, maxSteps);

    auto obj = Napi::Object::New(env);
    obj.Set("positions", matrixToFloat64Array(env, result.positions));
    obj.Set("segmentCount", Napi::Number::New(env, result.segmentCount));
    return obj;
}

// ─── generateHeatDiffusion(ctx, {sources, timesteps, alpha}) ────
//
// Returns a [T*nV] Float32Array (row-major frame-major) plus T and
// nV. Eigenmodes must already be computed via solveEigenmodes.

Napi::Value GenerateHeatDiffusion(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    if (!holder->eigenmodes) {
        Napi::Error::New(env,
            "generateHeatDiffusion: eigenmodes not computed; "
            "call solveEigenmodes first").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (!holder->ops) {
        holder->ops = std::make_shared<MeshOperators>(
            assembleMeshOperators(*holder->ctx)
        );
    }

    // args[1] = options object
    auto opts = info[1].As<Napi::Object>();
    auto sourceVertsArr = opts.Get("sourceVerts").As<Napi::Int32Array>();
    auto sourceValuesArr = opts.Get("sourceValues").As<Napi::Float64Array>();
    auto timestepsArr = opts.Get("timesteps").As<Napi::Float64Array>();
    double alpha = opts.Get("alpha").As<Napi::Number>().DoubleValue();

    int nV = holder->ctx->nV();
    std::map<int, double> sourcesMap;
    for (size_t i = 0; i < sourceVertsArr.ElementLength(); i++) {
        sourcesMap[sourceVertsArr[i]] = sourceValuesArr[i];
    }
    Eigen::VectorXd u0 = generateDelta(nV, sourcesMap);

    std::vector<double> timesteps(
        timestepsArr.Data(),
        timestepsArr.Data() + timestepsArr.ElementLength()
    );

    Eigen::MatrixXf field = generateHeatDiffusion(
        *holder->ops, *holder->eigenmodes, u0, timesteps, alpha);

    auto result = Napi::Object::New(env);
    result.Set("data", matrixToFloat32Array(env, field));
    result.Set("T", Napi::Number::New(env, static_cast<int>(field.rows())));
    result.Set("nV", Napi::Number::New(env, static_cast<int>(field.cols())));
    return result;
}

// ─── generateDampedWave(ctx, {modeIndices, amplitudes, dampings, phases, timesteps}) ─

Napi::Value GenerateDampedWave(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    if (!holder->eigenmodes) {
        Napi::Error::New(env,
            "generateDampedWave: eigenmodes not computed; "
            "call solveEigenmodes first").ThrowAsJavaScriptException();
        return env.Null();
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

    Eigen::MatrixXf field = generateDampedWave(
        *holder->eigenmodes, modeIndices, amplitudes, dampings, phases, timesteps);

    auto result = Napi::Object::New(env);
    result.Set("data", matrixToFloat32Array(env, field));
    result.Set("T", Napi::Number::New(env, static_cast<int>(field.rows())));
    result.Set("nV", Napi::Number::New(env, static_cast<int>(field.cols())));
    return result;
}

// ─── Vector heat method ──────────────────────────────────────

static void ensureVHM(ContextHolder* holder) {
    if (!holder->vhm) {
        holder->vhm = std::make_shared<VectorHeatSolver>(*holder->ctx);
    }
}
static void ensureSHS(ContextHolder* holder) {
    if (!holder->shs) {
        holder->shs = std::make_shared<SignedHeatSolver>(*holder->ctx);
    }
}

Napi::Value VectorHeatTransport(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    auto vertsArr = info[1].As<Napi::Int32Array>();
    auto vecsArr  = info[2].As<Napi::Float64Array>();
    int n = static_cast<int>(vertsArr.ElementLength());
    if (vecsArr.ElementLength() != static_cast<size_t>(n) * 3) {
        Napi::TypeError::New(env, "sourceVectors must be Nx3 (length 3*sources)").ThrowAsJavaScriptException();
        return env.Null();
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
    Eigen::MatrixXd out = vectorHeatTransport(*holder->vhm, verts, S);
    return matrixToFloat64Array(env, out);
}

Napi::Value VectorHeatExtendScalar(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    auto vertsArr = info[1].As<Napi::Int32Array>();
    auto valsArr  = info[2].As<Napi::Float64Array>();
    int n = static_cast<int>(vertsArr.ElementLength());
    if (valsArr.ElementLength() != static_cast<size_t>(n)) {
        Napi::TypeError::New(env, "sourceVerts and sourceValues must have the same length").ThrowAsJavaScriptException();
        return env.Null();
    }
    std::vector<int> verts(vertsArr.Data(), vertsArr.Data() + n);
    Eigen::VectorXd vals(n);
    std::memcpy(vals.data(), valsArr.Data(), n * sizeof(double));

    ensureVHM(holder);
    Eigen::VectorXd out = vectorHeatExtendScalar(*holder->vhm, verts, vals);
    return toFloat64Array(env, out);
}

Napi::Value VectorHeatLogMap(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    int sourceVertex = info[1].As<Napi::Number>().Int32Value();
    int strategyInt  = info.Length() > 2
        ? info[2].As<Napi::Number>().Int32Value()
        : static_cast<int>(LogMapStrategy::AffineLocal);
    auto strategy = static_cast<LogMapStrategy>(strategyInt);

    ensureVHM(holder);
    LogMapResult r = vectorHeatLogMap(*holder->vhm, sourceVertex, strategy);

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
}

Napi::Value VectorHeatFindCenter(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    auto vertsArr = info[1].As<Napi::Int32Array>();
    int p = info.Length() > 2 ? info[2].As<Napi::Number>().Int32Value() : 2;
    std::vector<int> verts(vertsArr.Data(), vertsArr.Data() + vertsArr.ElementLength());

    ensureVHM(holder);
    Eigen::Vector3d c = vectorHeatFindCenter(*holder->vhm, verts, p);
    auto out = Napi::Float64Array::New(env, 3);
    out[0] = c.x(); out[1] = c.y(); out[2] = c.z();
    return out;
}

// ─── Signed heat method ──────────────────────────────────────

Napi::Value SignedHeatDistance(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    auto vertsArr = info[1].As<Napi::Int32Array>();
    bool isLoop = info.Length() > 2 ? info[2].As<Napi::Boolean>().Value() : true;
    int  lsInt  = info.Length() > 3
        ? info[3].As<Napi::Number>().Int32Value()
        : static_cast<int>(SignedHeatLevelSet::ZeroSet);
    auto ls = static_cast<SignedHeatLevelSet>(lsInt);

    std::vector<int> verts(vertsArr.Data(), vertsArr.Data() + vertsArr.ElementLength());

    ensureSHS(holder);
    Eigen::VectorXd out = signedHeatDistance(*holder->shs, verts, isLoop, ls);
    return toFloat64Array(env, out);
}

// ─── Smooth direction fields ────────────────────────────────

Napi::Value ComputeSmoothFaceField(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    int  nSym             = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : 4;
    bool alignToCurvature = info.Length() > 2 ? info[2].As<Napi::Boolean>().Value()     : false;

    Eigen::MatrixXd v = computeSmoothFaceField(*holder->ctx, nSym, alignToCurvature);
    return matrixToFloat64Array(env, v);
}

Napi::Value ComputeSmoothVertexField(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    int  nSym             = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : 2;
    bool alignToCurvature = info.Length() > 2 ? info[2].As<Napi::Boolean>().Value()     : false;

    SmoothFieldResult r = computeSmoothVertexField(*holder->ctx, nSym, alignToCurvature);
    auto obj = Napi::Object::New(env);
    obj.Set("vertexVectors",  matrixToFloat64Array(env, r.vertexVectors));
    obj.Set("vertexFieldRaw", toFloat64Array(env, r.vertexFieldRaw));
    obj.Set("nSym", Napi::Number::New(env, r.nSym));
    return obj;
}

// ─── Stripe patterns ────────────────────────────────────────

Napi::Value ComputeStripePattern(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    auto fieldArr = info[1].As<Napi::Float64Array>();
    Eigen::VectorXd raw = toVectorXd(fieldArr);
    double freq   = info[2].As<Napi::Number>().DoubleValue();
    bool   connect = info.Length() > 3 ? info[3].As<Napi::Boolean>().Value() : true;

    StripePatternResult r = computeStripePattern(*holder->ctx, raw, freq, connect);
    auto obj = Napi::Object::New(env);
    obj.Set("positions",    matrixToFloat64Array(env, r.positions));
    obj.Set("segmentCount", Napi::Number::New(env, r.segmentCount));
    return obj;
}

Napi::Value ComputeStripePatternFreq(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();

    auto fieldArr = info[1].As<Napi::Float64Array>();
    auto freqsArr = info[2].As<Napi::Float64Array>();
    Eigen::VectorXd raw   = toVectorXd(fieldArr);
    Eigen::VectorXd freqs = toVectorXd(freqsArr);
    bool connect = info.Length() > 3 ? info[3].As<Napi::Boolean>().Value() : true;

    StripePatternResult r = computeStripePatternFreq(*holder->ctx, raw, freqs, connect);
    auto obj = Napi::Object::New(env);
    obj.Set("positions",    matrixToFloat64Array(env, r.positions));
    obj.Set("segmentCount", Napi::Number::New(env, r.segmentCount));
    return obj;
}

// ─── Module Init ─────────────────────────────────────────────

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("createContext", Napi::Function::New(env, CreateContext));
    exports.Set("assembleMeshOperators", Napi::Function::New(env, AssembleMeshOperators));
    exports.Set("assembleDECOperators", Napi::Function::New(env, AssembleDECOperators));
    exports.Set("solveEigenmodes", Napi::Function::New(env, SolveEigenmodes));
    exports.Set("solvePoisson", Napi::Function::New(env, SolvePoisson));
    exports.Set("computeGeodesicDistance", Napi::Function::New(env, ComputeGeodesicDistance));
    exports.Set("hodgeDecompose", Napi::Function::New(env, HodgeDecompose));
    exports.Set("computeCurvatures", Napi::Function::New(env, ComputeCurvatures));
    exports.Set("computeVertexNormals", Napi::Function::New(env, ComputeVertexNormals));
    exports.Set("whitneyInterpolate", Napi::Function::New(env, WhitneyInterpolate));
    exports.Set("scalarGradient", Napi::Function::New(env, ScalarGradient));
    exports.Set("computeIsolines", Napi::Function::New(env, ComputeIsolines));
    exports.Set("computeDirectionField", Napi::Function::New(env, ComputeDirectionField));
    exports.Set("traceStreamlines", Napi::Function::New(env, TraceStreamlines));
    exports.Set("generateHeatDiffusion", Napi::Function::New(env, GenerateHeatDiffusion));
    exports.Set("generateDampedWave", Napi::Function::New(env, GenerateDampedWave));
    // Vector heat method
    exports.Set("vectorHeatTransport",    Napi::Function::New(env, VectorHeatTransport));
    exports.Set("vectorHeatExtendScalar", Napi::Function::New(env, VectorHeatExtendScalar));
    exports.Set("vectorHeatLogMap",       Napi::Function::New(env, VectorHeatLogMap));
    exports.Set("vectorHeatFindCenter",   Napi::Function::New(env, VectorHeatFindCenter));
    // Signed heat method
    exports.Set("signedHeatDistance",     Napi::Function::New(env, SignedHeatDistance));
    // Smooth direction fields
    exports.Set("computeSmoothFaceField",   Napi::Function::New(env, ComputeSmoothFaceField));
    exports.Set("computeSmoothVertexField", Napi::Function::New(env, ComputeSmoothVertexField));
    // Stripe patterns
    exports.Set("computeStripePattern",     Napi::Function::New(env, ComputeStripePattern));
    exports.Set("computeStripePatternFreq", Napi::Function::New(env, ComputeStripePatternFreq));
    return exports;
}

NODE_API_MODULE(nxr_compute_addon, Init)
