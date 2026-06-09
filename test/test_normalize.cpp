#include "nxr/compute.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>

using nxr::manifold::fixDelaunay;

static int g_failures = 0;
#define EXPECT(cond, msg) do { if (!(cond)) { \
    std::cerr << "  [FAIL] " << msg << "\n"; ++g_failures; } \
    else { std::cout << "  [PASS] " << msg << "\n"; } } while (0)

// True iff some face contains both a and b.
static bool edgePresent(const Eigen::MatrixXi& F, int a, int b) {
    for (int r = 0; r < F.rows(); ++r) {
        std::set<int> v{F(r,0), F(r,1), F(r,2)};
        if (v.count(a) && v.count(b)) return true;
    }
    return false;
}

static void testThinQuad() {
    std::cout << "\n=== fixDelaunay: thin quad (non-Delaunay) ===\n";
    // 0=(0,0) 1=(2,0) 2=(1,0.2) 3=(1,-0.2); split along the LONG diagonal 0-1.
    std::vector<double>  V = {0,0,0,  2,0,0,  1,0.2,0,  1,-0.2,0};
    std::vector<int32_t> F = {0,2,1,  0,1,3};   // 2 triangles, shared edge 0-1
    auto r = fixDelaunay(V.data(), 4, F.data(), 2);

    EXPECT(r.flips == 1, "exactly one flip");
    EXPECT(r.faces.rows() == 2 && r.faces.cols() == 3, "still 2 triangles");
    EXPECT(!edgePresent(r.faces, 0, 1), "long diagonal 0-1 removed");
    EXPECT(edgePresent(r.faces, 2, 3),  "short diagonal 2-3 present");
    // vertex indices stay in range
    EXPECT(r.faces.minCoeff() >= 0 && r.faces.maxCoeff() <= 3, "indices in [0,3]");

    // ── exact vertex set ──────────────────────────────────────────────────────
    // All 4 vertices {0,1,2,3} must appear in the normalized faces.
    {
        std::set<int> used;
        for (int i = 0; i < r.faces.rows(); ++i)
            for (int k = 0; k < 3; ++k) used.insert(r.faces(i, k));
        EXPECT((used == std::set<int>{0, 1, 2, 3}), "all 4 vertices used");
    }

    // ── cotan-weight (PSD) check ──────────────────────────────────────────────
    // Cotan Laplacian off-diagonals = −weight. A negative cotan weight
    // means a POSITIVE off-diagonal entry. The input (non-Delaunay) mesh
    // must have at least one positive off-diagonal; the normalized
    // (Delaunay) mesh must have all off-diagonals ≤ 1e-9.
    auto maxOffDiag = [](const Eigen::SparseMatrix<double>& L) {
        double m = -1e300;
        for (int col = 0; col < L.outerSize(); ++col)
            for (Eigen::SparseMatrix<double>::InnerIterator it(L, col); it; ++it)
                if (it.row() != it.col()) m = std::max(m, it.value());
        return m;
    };

    // Input: non-Delaunay mesh → some positive off-diagonal (negative cotan weight).
    {
        nxr::manifold::Manifold mIn(V.data(), 4, F.data(), 2);
        auto opsIn = nxr::manifold::ops::assembleManifoldOperators(mIn);
        EXPECT(maxOffDiag(opsIn.cotanLaplacian) > 1e-9,
               "input has a negative cotan weight");
    }

    // Output: Delaunay mesh → all off-diagonals ≤ 0 (weights ≥ 0).
    {
        std::vector<int32_t> Fout(static_cast<size_t>(r.faces.rows() * 3));
        for (int i = 0; i < r.faces.rows(); ++i)
            for (int k = 0; k < 3; ++k) Fout[3 * i + k] = r.faces(i, k);
        nxr::manifold::Manifold mOut(V.data(), 4, Fout.data(),
                                      static_cast<int>(r.faces.rows()));
        auto opsOut = nxr::manifold::ops::assembleManifoldOperators(mOut);
        EXPECT(maxOffDiag(opsOut.cotanLaplacian) < 1e-9,
               "normalized mesh has all cotan weights >= 0");
    }
}

static void testAlreadyDelaunay() {
    std::cout << "\n=== fixDelaunay: already-Delaunay (icosahedron) ===\n";
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    std::vector<double> V = {-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t,
                              0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1};
    for (int i=0;i<12;++i){ double n=std::sqrt(V[3*i]*V[3*i]+V[3*i+1]*V[3*i+1]+V[3*i+2]*V[3*i+2]);
        V[3*i]/=n; V[3*i+1]/=n; V[3*i+2]/=n; }
    std::vector<int32_t> F = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4,
        11,10,2, 10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5, 2,4,11,
        6,2,10, 8,6,7, 9,8,1};
    auto r = fixDelaunay(V.data(), 12, F.data(), 20);
    EXPECT(r.flips == 0, "icosphere already Delaunay (0 flips)");
    EXPECT(r.faces.rows() == 20, "20 faces preserved");

    // ── faces-as-set check ────────────────────────────────────────────────────
    // flips == 0 means the topology is unchanged; verify the output face set
    // equals the input face set (as unordered sets of sorted triples).
    {
        auto faceKeys = [](auto getRow, int n) {
            std::set<std::array<int, 3>> s;
            for (int i = 0; i < n; ++i) {
                std::array<int, 3> t = getRow(i);
                std::sort(t.begin(), t.end());
                s.insert(t);
            }
            return s;
        };

        auto inKeys = faceKeys(
            [&](int i) -> std::array<int, 3> {
                return {static_cast<int>(F[3 * i]),
                        static_cast<int>(F[3 * i + 1]),
                        static_cast<int>(F[3 * i + 2])};
            },
            20);

        auto outKeys = faceKeys(
            [&](int i) -> std::array<int, 3> {
                return {r.faces(i, 0), r.faces(i, 1), r.faces(i, 2)};
            },
            static_cast<int>(r.faces.rows()));

        EXPECT(inKeys == outKeys, "icosphere output face set equals input face set");
    }
}

int main() {
    testThinQuad();
    testAlreadyDelaunay();
    if (g_failures) { std::cerr << "\n" << g_failures << " failure(s)\n"; return 1; }
    std::cout << "\nALL PASSED\n"; return 0;
}
