#include "nxr/compute.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace nxr::field::generate {

using core::Error;
using core::ErrorCode;

// ── Static field generators ──────────────────────────────────

Eigen::VectorXd delta(int nV, const std::map<int, double>& sources) {
    Eigen::VectorXd f = Eigen::VectorXd::Zero(nV);
    for (const auto& [idx, val] : sources) {
        if (idx >= 0 && idx < nV) f(idx) = val;
    }
    return f;
}

Eigen::VectorXd randomVertexScalar(int nV, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Eigen::VectorXd f(nV);
    for (int i = 0; i < nV; i++) f(i) = dist(gen);
    return f;
}

Eigen::VectorXd randomFaceScalar(int nF, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Eigen::VectorXd f(nF);
    for (int i = 0; i < nF; i++) f(i) = dist(gen);
    return f;
}

Eigen::VectorXd eigenmodeField(const EigenResult& eig, int k_index) {
    if (k_index < 0 || k_index >= eig.k) {
        throw Error(ErrorCode::InvalidInput,
            "eigenmodeField: eigenmode index out of range");
    }
    return eig.eigenvectors.col(k_index);
}

Eigen::VectorXd randomDecomposed1Form(
    const DECOperators& dec,
    int nV, int nE, int nF,
    double alphaStrength,
    double betaStrength,
    double /*gammaStrength*/,    // reserved; harmonic basis deferred
    unsigned int seed
) {
    // α (random vertex scalar) → exact 1-form  dα = d0 · α
    Eigen::VectorXd alpha = randomVertexScalar(nV, seed);
    Eigen::VectorXd dAlpha = dec.d0 * alpha;

    // β (random face scalar) → co-exact 1-form  δβ = ★₁⁻¹ · d1ᵀ · β
    Eigen::VectorXd beta = randomFaceScalar(nF, seed + 1);
    Eigen::VectorXd deltaBeta = dec.hodge1Inverse * (dec.d1.transpose() * beta);

    // γ (harmonic) — not synthesized in v1. For genus-0 surfaces the
    // harmonic space is empty; for higher-genus surfaces we'd need to
    // build a basis (e.g. by orthogonalizing random 1-forms against
    // the exact + co-exact subspaces or via tree-cotree decomposition).
    // Deferred — see header for rationale.

    Eigen::VectorXd omega(nE);
    omega = alphaStrength * dAlpha + betaStrength * deltaBeta;
    return omega;
}

// ── Time-varying field generators ────────────────────────────

Eigen::MatrixXf heatDiffusion(
    const Eigen::SparseMatrix<double>& M,
    const EigenResult& eig,
    const Eigen::VectorXd& u0,
    const std::vector<double>& timesteps,
    double alpha
) {
    int n = static_cast<int>(M.rows());
    int T = static_cast<int>(timesteps.size());

    if (u0.size() != n) {
        throw Error(ErrorCode::InvalidInput,
            "heatDiffusion: u0 size must equal M.rows()");
    }
    if (eig.eigenvectors.rows() != n) {
        throw Error(ErrorCode::InvalidInput,
            "heatDiffusion: eigenvectors row count must equal M.rows()");
    }

    // M-weighted spatial mean is preserved exactly by the heat
    // equation. Separate it from the spectral evolution so the
    // generator works regardless of whether the DC mode is present
    // in `eig` (i.e. before/after removeDC). totalMass = 1ᵀ M 1 plays
    // the role of "totalArea" in the mesh case (where M is the lumped
    // Voronoi mass diagonal) and "total node weight" in the graph case.
    double totalMass = (M * Eigen::VectorXd::Ones(n)).sum();
    double u_mean = (M * u0).sum() / totalMass;
    Eigen::VectorXd u0_zm = u0.array() - u_mean;

    // Spectral coefficients of the zero-mean part: c₀ = Uᵀ M (u₀ - mean).
    // Assumes U is M-orthonormal (caller's responsibility).
    Eigen::VectorXd c0 = eig.eigenvectors.transpose() * (M * u0_zm);

    Eigen::MatrixXf output(T, n);
    for (int ti = 0; ti < T; ti++) {
        double t = timesteps[ti];
        Eigen::VectorXd attenuation = (-alpha * t * eig.eigenvalues.array()).exp();
        Eigen::VectorXd ct = c0.cwiseProduct(attenuation.matrix());
        Eigen::VectorXd u_t = eig.eigenvectors * ct;
        u_t.array() += u_mean;
        output.row(ti) = u_t.cast<float>().transpose();
    }
    return output;
}

Eigen::MatrixXf dampedWave(
    const EigenResult& eig,
    const std::vector<int>& modeIndices,
    const std::vector<double>& amplitudes,
    const std::vector<double>& dampings,
    const std::vector<double>& phases,
    const std::vector<double>& timesteps
) {
    int M = static_cast<int>(modeIndices.size());
    if (static_cast<int>(amplitudes.size()) != M ||
        static_cast<int>(dampings.size())   != M ||
        static_cast<int>(phases.size())     != M) {
        throw Error(ErrorCode::InvalidInput,
            "dampedWave: modeIndices/amplitudes/dampings/phases "
            "must have the same length");
    }

    int nV = static_cast<int>(eig.eigenvectors.rows());
    int T = static_cast<int>(timesteps.size());

    Eigen::MatrixXf output = Eigen::MatrixXf::Zero(T, nV);

    for (int m = 0; m < M; m++) {
        int k = modeIndices[m];
        if (k < 0 || k >= eig.k) {
            throw Error(ErrorCode::InvalidInput,
                "dampedWave: mode index out of range");
        }

        // Numerical noise can push the DC eigenvalue slightly negative;
        // clamp before sqrt to keep the angular frequency real.
        double lambda = std::max(eig.eigenvalues(k), 0.0);
        double omega = std::sqrt(lambda);
        Eigen::VectorXf phi_k = eig.eigenvectors.col(k).cast<float>();

        // Per-mode temporal envelope.
        Eigen::VectorXf temporal(T);
        for (int ti = 0; ti < T; ti++) {
            double t = timesteps[ti];
            double s = amplitudes[m]
                     * std::exp(-dampings[m] * t)
                     * std::cos(omega * t + phases[m]);
            temporal(ti) = static_cast<float>(s);
        }

        // Outer product: each frame gets `temporal(ti) * phi_k`.
        // Eigen handles the (T,1) × (1,V) accumulation.
        output.noalias() += temporal * phi_k.transpose();
    }

    return output;
}

} // namespace nxr::field::generate