// Parity with the nxr-cortical-flow eigenmode exports (the reference where implementations differ): matrices and the lowest
// spectra on bunny-closed (default) and on the glued sub-0002 cortex (NXR_CF_CORTEX=1). Skips (exit 0) when the fixtures are
// absent: NXR_CF_FIXTURES = nxr-cortical-flow/editor/fixtures.
#include "nxr/compute.h"
#include "nxr/facets.h"
#include "mesh_helpers.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
using namespace nxr::manifold;
using SM = Eigen::SparseMatrix<double>;

static std::string FX;
template <class T> static std::vector<T> readBin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<T> v(f.tellg() / sizeof(T)); f.seekg(0); f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(T));
    return v;
}
static std::vector<double> readNpy(const std::string& path) {   // 1-D float64 .npy
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    char magic[8]; f.read(magic, 8); unsigned short hl; f.read(reinterpret_cast<char*>(&hl), 2); f.seekg(10 + hl);
    std::vector<double> v; double x; while (f.read(reinterpret_cast<char*>(&x), 8)) v.push_back(x);
    return v;
}
static SM csr(const std::string& stem, int rows, int cols) {
    auto ip = readBin<int32_t>(stem + ".indptr.i32"), ix = readBin<int32_t>(stem + ".indices.i32"); auto d = readBin<double>(stem + ".data.f64");
    std::vector<Eigen::Triplet<double>> T; T.reserve(d.size());
    for (int r = 0; r < rows; ++r) for (int p = ip[r]; p < ip[r + 1]; ++p) T.emplace_back(r, ix[p], d[p]);
    SM S(rows, cols); S.setFromTriplets(T.begin(), T.end()); return S;
}
static SM coo(const std::string& stem, int rows, int cols) {
    auto r = readBin<int32_t>(stem + ".row.i32"), c = readBin<int32_t>(stem + ".col.i32"); auto d = readBin<double>(stem + ".data.f64");
    std::vector<Eigen::Triplet<double>> T; for (size_t i = 0; i < d.size(); ++i) T.emplace_back(r[i], c[i], d[i]);
    SM S(rows, cols); S.setFromTriplets(T.begin(), T.end()); return S;
}
static double rel(const SM& a, const SM& b) { return (a - b).norm() / std::max(1e-300, b.norm()); }
static std::string num(double x) { std::ostringstream o; o << x; return o.str(); }

// positive entries of an ascending signed μ list, the first n
static std::vector<double> positives(std::vector<double> mu, size_t n) {
    std::vector<double> p; for (double x : mu) if (x > 1e-6) p.push_back(x);
    std::sort(p.begin(), p.end()); p.resize(std::min(n, p.size())); return p;
}

int main() {
    const char* e = std::getenv("NXR_CF_FIXTURES");
    FX = e ? e : std::string(std::getenv("HOME")) + "/workspace/research/code/nxr-cortical-flow/editor/fixtures";
    if (!std::ifstream(FX + "/bunny-closed.json")) { std::cout << "SKIP: fixtures not found at " << FX << "\n"; return 0; }
    try {
        const std::string eb = FX + "/eigenmodes/bunny-closed";
        auto pf = readBin<float>(FX + "/bunny-closed.positions.f32"); auto fu = readBin<uint32_t>(FX + "/bunny-closed.faces.u32");
        std::vector<double> P(pf.begin(), pf.end()); std::vector<int32_t> Fi(fu.begin(), fu.end());
        const int nV = (int)P.size() / 3, nF = (int)Fi.size() / 3;
        std::cout << "\n=== bunny-closed: " << nV << " V, " << nF << " F ===\n";
        Manifold m(P.data(), nV, Fi.data(), nF);

        // Crane's Dirac D (Gauss map) and the edge-vector Dirac: matrices
        EXPECT(rel(ops::dirac::matrix(m), coo(eb + "/exports/export/diracD", 4 * nF, 4 * nV)) < 1e-9, "diracD matrix == cortical-flow export (rel Frobenius " + num(rel(ops::dirac::matrix(m), coo(eb + "/exports/export/diracD", 4 * nF, 4 * nV))) + ")");
        EXPECT(rel(ops::dirac::matrixIntrinsic(m), coo(eb + "/exports/export/diracIntrinsicD", 4 * nF, 4 * nV)) < 1e-9, "diracIntrinsicD matrix == export");

        // Crane spectrum: sqrt of the eigenvalues of E u = mu^2 M_V u, M_V = barycentric (lumped) x I4
        {
            SM E = ops::dirac::extrinsicBlock(m);
            (void)E;
            solve::EigenOperatorSpec crane{solve::EigenOperator::Dirac, 1.0, ops::MassMatrixVariant::Lumped};   // tau = 1: E, mass barycentric x I4
            auto r = solve::eigenOperator(m, crane, 12, solve::kShiftLadder, true, /*multiplets=*/true);
            auto ref = readNpy(eb + "/diracD_mu.npy");
            double d = 0; int cnt = 0;
            for (int i = 0; i < r.k && cnt < 8; ++i) if (r.eigenvalues(i) > 1.0) { d = std::max(d, std::abs(std::sqrt(r.eigenvalues(i)) - ref[4 + cnt]) / ref[4 + cnt]); ++cnt; }
            EXPECT(cnt == 8 && d < 1e-5, "diracD (Crane) lowest nonzero mu x8 == export diracD_mu.npy, max rel " + num(d));
        }

        // spin Diracs
        auto ex = ops::spin::assemble(m, ops::spin::Kind::Extrinsic), hy = ops::spin::assemble(m, ops::spin::Kind::IntrinsicHY), ye = ops::spin::assemble(m, ops::spin::Kind::IntrinsicYe);
        const std::string es = eb + "/exports/export-spin/bunny";
        EXPECT(rel(ex.D, csr(es + ".extrinsic", 4 * nF, 4 * nF)) < 1e-12, "spin extrinsic == export (rel " + num(rel(ex.D, csr(es + ".extrinsic", 4 * nF, 4 * nF))) + ")");
        EXPECT(rel(hy.D, csr(es + ".intrinsicHY", 4 * nF, 4 * nF)) < 1e-12, "spin intrinsicHY == export (rel " + num(rel(hy.D, csr(es + ".intrinsicHY", 4 * nF, 4 * nF))) + ")");
        EXPECT(rel(ye.D, csr(es + ".intrinsicYe", 4 * nF, 4 * nF)) < 1e-12, "spin intrinsicYe == export (rel " + num(rel(ye.D, csr(es + ".intrinsicYe", 4 * nF, 4 * nF))) + ")");
        EXPECT(rel(ops::spin::vertexToFace(m, true), csr(es + ".Acov", 4 * nF, 4 * nV)) < 1e-9, "vertexToFace covariant == export Acov");
        EXPECT(rel(ops::spin::vertexToFace(m, false), csr(es + ".Aplain", 4 * nF, 4 * nV)) < 1e-12, "vertexToFace plain == export Aplain");
        auto H = readBin<double>(es + ".H.f64"); double dH = 0; for (int f = 0; f < nF; ++f) dH = std::max(dH, std::abs(H[f] - ex.H(f)));
        EXPECT(dH < 1e-12, "H per face == export (max diff " + num(dH) + ")");
        for (auto [name, kind] : {std::pair<const char*, ops::spin::Kind>{"intrinsicHY", ops::spin::Kind::IntrinsicHY}, {"extrinsic", ops::spin::Kind::Extrinsic}}) {
            auto r = solve::eigenSpin(m, kind, 24);
            auto ref = readNpy(eb + "/spin_" + std::string(name) + "_mu.npy");
            std::vector<double> a(r.eigenvalues.data(), r.eigenvalues.data() + r.k), b(ref);
            std::sort(b.begin(), b.end(), [](double x, double y) { return std::abs(x) < std::abs(y); });
            // the python stores 128 nearest 0 including the constants' ~0 for the extrinsic; compare |mu| of the nonzero ones
            std::vector<double> A2, B2; for (double x : a) if (std::abs(x) > 1e-3) A2.push_back(std::abs(x)); for (double x : b) if (std::abs(x) > 1e-3) B2.push_back(std::abs(x));
            std::sort(A2.begin(), A2.end()); std::sort(B2.begin(), B2.end());
            double d = 0; size_t n = std::min<size_t>(A2.size(), 16); for (size_t i = 0; i < n; ++i) d = std::max(d, std::abs(A2[i] - B2[i]) / B2[i]);
            EXPECT(n >= 16 && d < 1e-5, std::string("spin ") + name + " lowest 16 |mu| == export (max rel " + num(d) + ")");
        }

        // Hodge-Dirac
        {
            auto h = ops::hodgedirac::assemble(m);
            const std::string eh = eb + "/exports/export";
            auto mass = readBin<double>(eh + "/hodge.mass.f64");
            EXPECT(h.nV == nV && h.nF == nF && (int)mass.size() == h.K.rows(), "Hodge N = V+E+F = " + std::to_string(h.K.rows()));
            double d0 = 0; for (int v = 0; v < nV; ++v) d0 = std::max(d0, std::abs(h.star0(v) - mass[v]) / mass[v]);
            EXPECT(d0 < 1e-9, "star0 (mixed-Voronoi lumped) per vertex == export, max rel " + num(d0));
            std::vector<double> s1(mass.begin() + h.nV, mass.begin() + h.nV + h.nE), s2(mass.begin() + h.nV + h.nE, mass.end());
            std::vector<double> m1(h.star1.data(), h.star1.data() + h.nE), m2(h.star2.data(), h.star2.data() + h.nF);
            std::sort(s1.begin(), s1.end()); std::sort(m1.begin(), m1.end()); std::sort(s2.begin(), s2.end()); std::sort(m2.begin(), m2.end());
            double d1 = 0, d2 = 0; for (size_t i = 0; i < s1.size(); ++i) d1 = std::max(d1, std::abs(s1[i] - m1[i]) / s1[i]); for (size_t i = 0; i < s2.size(); ++i) d2 = std::max(d2, std::abs(s2[i] - m2[i]) / s2[i]);
            EXPECT(d1 < 1e-8 && d2 < 1e-8, "star1, star2 (sorted) == export, max rel " + num(d1) + ", " + num(d2));
            auto ref = positives(readNpy(eb + "/hodge_mu.npy"), 60);
            auto r = solve::hodgeDiracModes(h, 60);
            double d = 0; for (int i = 0; i < 60; ++i) d = std::max(d, std::abs(r.eigenvalues(i) - ref[i]) / ref[i]);
            EXPECT(d < 1e-6, "lowest 60 +mu == export hodge_mu.npy, max rel " + num(d));
        }

        if (std::getenv("NXR_CF_CORTEX") && std::string(std::getenv("NXR_CF_CORTEX")) == "1") {
            const std::string ec = FX + "/eigenmodes/cortex_pial_low.glued-3mm", xc = ec + "/exports/export-cortex/glued";
            auto Pc = readBin<double>(xc + ".P.f64"); auto Fc = readBin<int32_t>(xc + ".F.i32");
            const int cV = (int)Pc.size() / 3, cF = (int)Fc.size() / 3;
            std::cout << "\n=== glued sub-0002 cortex: " << cV << " V, " << cF << " F ===\n";
            Manifold g(Pc.data(), cV, Fc.data(), cF);
            const double rc = rel(ops::dirac::matrixIntrinsic(g), coo(xc + ".crane", 4 * cF, 4 * cV));
            EXPECT(rc < 1e-9, "glued: diracIntrinsicD matrix (export crane) == export, rel " + num(rc));
            auto hy = ops::spin::assemble(g, ops::spin::Kind::IntrinsicHY);
            const double rh = rel(hy.D, csr(xc + ".intrinsicHY", 4 * cF, 4 * cF));
            // 8 of 1.9 M entries differ by up to 6e-7 (rel 1e-8): fold-back faces, where tan(theta/2) = sin/(1+cos) amplifies the rounding of cos
            EXPECT(rh < 1e-7, "glued: spin intrinsicHY == export (fold-back rounding aside), rel " + num(rh));
            auto h = ops::hodgedirac::assemble(g);
            auto mass = readBin<double>(xc + ".hodge.mass.f64");
            std::cout << "  Hodge N " << h.K.rows() << " vs export " << mass.size() << "; star1 clamped " << h.star1Clamped << "\n";
            EXPECT((int)mass.size() == h.K.rows(), "glued: Hodge N == export");
            double d0 = 0; for (int v = 0; v < cV; ++v) d0 = std::max(d0, std::abs(h.star0(v) - mass[v]) / mass[v]);
            EXPECT(d0 < 1e-8, "glued: star0 == export, max rel " + num(d0));
            auto ref = positives(readNpy(ec + "/glued_hodge_mu.npy"), 50);
            auto r = solve::hodgeDiracModes(h, 50);
            double d = 0; for (int i = 0; i < 50; ++i) d = std::max(d, std::abs(r.eigenvalues(i) - ref[i]) / ref[i]);
            EXPECT(d < 1e-6, "glued: lowest 50 +mu == export glued_hodge_mu.npy, max rel " + num(d));
        }
    } catch (const std::exception& ex) { std::cout << "  [FAIL] exception: " << ex.what() << "\n"; ++g_failures; }
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
