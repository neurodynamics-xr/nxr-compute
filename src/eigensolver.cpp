#include "nxr/compute.h"

#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/MatOp/SparseCholesky.h>

#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>

namespace nxr::manifold::solve {

namespace {

// ── CancelProgressOp — wraps a Spectra OpType to add cancel/progress ──
//
// Spectra's iteration calls `op.perform_op(x_in, y_out)` once per inner
// Lanczos step. Wrapping the OpType lets us intercept every step and:
//   1. Check the cancellation token. If set, throw Error(Cancelled);
//      Spectra's compute() unwinds cleanly because all its state is
//      Eigen members (RAII, no manual allocation).
//   2. Increment a progress counter and report it to the observer.
//
// The wrapper inherits the Inner type's interface (rows/cols/Scalar
// typedef, plus set_shift for shift-invert ops) and forwards
// perform_op to the inner. SymShiftInvert is the AOp here; SparseSymMatProd
// is the BOp and isn't wrapped (one wrap is enough — both ops fire each
// iteration so AOp gets the cancel/progress signal).
template <typename Inner>
class CancelProgressOp {
public:
    using Scalar = typename Inner::Scalar;

    CancelProgressOp(Inner& inner,
                     const CancellationToken& cancel,
                     const ProgressObserver& progress,
                     int totalEstimate)
        : inner_(inner), cancel_(cancel), progress_(progress),
          totalEstimate_(totalEstimate) {}

    Eigen::Index rows() const { return inner_.rows(); }
    Eigen::Index cols() const { return inner_.cols(); }

    // Required by SymShiftInvert (the shift-invert AOp specifically).
    void set_shift(const Scalar& sigma) { inner_.set_shift(sigma); }

    void perform_op(const Scalar* x_in, Scalar* y_out) const {
        if (cancel_.requested()) {
            throw Error(ErrorCode::Cancelled,
                "Eigensolve cancelled by request");
        }
        inner_.perform_op(x_in, y_out);
        ++count_;
        progress_.update(count_, totalEstimate_, 0.0);
    }

private:
    Inner&                     inner_;
    const CancellationToken&   cancel_;
    const ProgressObserver&    progress_;
    int                        totalEstimate_;
    mutable int                count_ = 0;
};

} // namespace

EigenResult eigen(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::SparseMatrix<double>& M,
    int k,
    double sigma,
    bool normalize,
    bool removeDC,
    const CancellationToken& cancel,
    const ProgressObserver& progress
) {
    int n = static_cast<int>(K.rows());

    // Validate inputs — caught before any expensive factorization.
    if (K.rows() != K.cols() || M.rows() != M.cols() || K.rows() != M.rows()) {
        throw Error(ErrorCode::InvalidInput,
            "eigen: K and M must be square and same size");
    }
    if (k < 1) {
        throw Error(ErrorCode::EigensolveInvalidK,
            "eigen: k must be >= 1");
    }
    if (k > n - 1) {
        throw Error(ErrorCode::EigensolveInvalidK,
            "eigen: k must be < N (matrix size)");
    }
    // K ceiling — Spectra's Krylov basis is `ncv = 2k+1` doubles per
    // matrix row (capped at n). At k=1000 on a 10k-vertex cortical
    // mesh, basis ≈ 160 MB; at k=5000 it approaches an n×n dense
    // matrix and the WASM linear-memory cap (2 GB) is hit before
    // convergence. Cap k at 1000 for browser/WASM consumers; native
    // builds (addon, MEX) are unaffected since they have headroom.
    constexpr int kMaxK = 1000;
    if (k > kMaxK) {
        throw Error(ErrorCode::EigensolveInvalidK,
            "solve: k must be <= 1000",
            "Browser/WASM Krylov basis ceiling — higher k requires "
            "server-side or memory64 builds. See "
            "docs/eigensolve-cap.md.");
    }

    // Convergence parameter: ncv must be > k and <= n.
    // Larger ncv costs more memory and per-iteration work but
    // converges in fewer iterations. 2*k+1 is Spectra's usual
    // recommendation; we clamp to n.
    int ncv = std::min(std::max(2 * k + 1, 20), n);

    // High-water bound on perform_op invocations for the progress
    // counter denominator. Spectra runs up to `maxit` outer
    // iterations, each performing roughly `ncv` SpMVs. The bound
    // is loose by design — most solves converge in a small fraction
    // of this — and consumers display a normalized fraction.
    constexpr int kMaxIter = 1000;
    int totalEstimate = kMaxIter * ncv;

    auto t0 = std::chrono::steady_clock::now();

    // Cancellation point before the (potentially several-second)
    // factorization. Catches "user clicked cancel before the solve
    // started" without paying for the LLT.
    if (cancel.requested()) {
        throw Error(ErrorCode::Cancelled,
            "Eigensolve cancelled before factorization");
    }

    // ─── Shift-invert mode ──────────────────────────────────
    //
    // Generalized problem: K φ = λ M φ
    //
    // Shift-invert transforms to:
    //   (K - σM)^{-1} M φ = ν φ   where ν = 1/(λ - σ)
    //
    // Eigenvalues near σ become the LARGEST of ν — which is
    // exactly what IRAM converges to fastest. For our use case
    // (smallest λ of the Laplacian, near 0), σ ≈ 0 gives
    // dramatically faster convergence than standard mode.
    //
    // The inner factorization of (K - σM) is done once by
    // SymShiftInvert's constructor; each IRAM iteration then
    // does one triangular forward/back-solve (O(V)) instead
    // of a full SpMV (O(nnz)).

    // SymShiftInvert factors (K - σM) using Eigen's SimplicialLLT.
    // Per CLAUDE.md §10, nxr-compute is constrained to Eigen + Spectra +
    // geometry-central; CHOLMOD's supernodal factorization would
    // be 3-10× faster on x86 but breaks the WASM and MEX targets,
    // so we don't pull it in.
    using ShiftInvertOp = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    ShiftInvertOp innerOp(K, M);
    CancelProgressOp<ShiftInvertOp> opShift(innerOp, cancel, progress, totalEstimate);

    // B operator for inner products: just M * x (matrix-vector product).
    // This is different from Cholesky mode where B needs to be decomposed.
    Spectra::SparseSymMatProd<double> opM(M);

    auto tFactor = std::chrono::steady_clock::now();
    double factorMs = std::chrono::duration<double, std::milli>(tFactor - t0).count();
    std::cout << "[eigensolver] Pre-factored (K - " << sigma << " M) in "
              << factorMs << " ms" << std::endl;

    Spectra::SymGEigsShiftSolver<
        CancelProgressOp<ShiftInvertOp>,
        Spectra::SparseSymMatProd<double>,
        Spectra::GEigsMode::ShiftInvert
    > solver(opShift, opM, k, ncv, sigma);

    solver.init();

    // compute() runs the IRAM loop. If the wrapped op throws
    // Error(Cancelled) mid-iteration, the exception propagates
    // out and Spectra's internal Eigen-typed members destruct
    // cleanly (no manual allocations to leak).
    int nconv = solver.compute(Spectra::SortRule::LargestMagn, kMaxIter, 1e-10);

    auto tSolve = std::chrono::steady_clock::now();
    double solveMs = std::chrono::duration<double, std::milli>(tSolve - tFactor).count();
    double totalMs = std::chrono::duration<double, std::milli>(tSolve - t0).count();

    EigenResult result;

    if (solver.info() == Spectra::CompInfo::Successful ||
        solver.info() == Spectra::CompInfo::NotConverging) {
        // Extract results (may be partial if not all converged)
        Eigen::VectorXd evals = solver.eigenvalues();
        Eigen::MatrixXd evecs = solver.eigenvectors();

        // Sort ascending by eigenvalue
        std::vector<int> idx(evals.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return evals(a) < evals(b);
        });

        result.eigenvalues.resize(evals.size());
        result.eigenvectors.resize(evecs.rows(), evecs.cols());
        for (int i = 0; i < static_cast<int>(evals.size()); i++) {
            result.eigenvalues(i) = evals(idx[i]);
            result.eigenvectors.col(i) = evecs.col(idx[i]);
        }

        result.k = static_cast<int>(evals.size());
        result.nConverged = nconv;

        std::cout << "[eigensolver] Converged: " << nconv << "/" << k
                  << " eigenvalues in " << totalMs << " ms"
                  << " (factor: " << factorMs << " ms, IRAM: " << solveMs << " ms)"
                  << ". Range: [" << result.eigenvalues(0)
                  << ", " << result.eigenvalues(result.k - 1) << "]" << std::endl;
    } else {
        std::cerr << "[eigensolver] FAILED: Spectra returned error code "
                  << static_cast<int>(solver.info())
                  << " after " << totalMs << " ms" << std::endl;
        throw Error(ErrorCode::EigensolveNotConverged,
            "Spectra eigensolver returned a numerical-issue error code");
    }

    if (removeDC) {
        result = ::nxr::manifold::solve::removeDC(result);
    }
    if (normalize) {
        result.eigenvectors = ::nxr::manifold::solve::normalize(result.eigenvectors, M);
    }

    return result;
}

} // namespace nxr::manifold::solve