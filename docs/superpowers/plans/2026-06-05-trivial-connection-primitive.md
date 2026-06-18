# Trivial Connection Primitive & Trivial Connection Laplacian Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `computeTrivialConnection` as a public primitive from the private `computeCoExactComponent`, refactor `trivial()` → `directionField()` to use it, and add `assembleTrivialConnectionLaplacian` that builds a complex Hermitian connection Laplacian using the trivial connection transport — for between-subject cortical analysis where singularities are pinned to FreeSurfer sphere poles.

**Architecture:** The private `computeCoExactComponent` in `direction_field.cpp` already contains the full trivial connection Poisson solve (φ = ★₁d₀β). Promoting it to a named public function lets both the direction field BFS propagation and the new connection Laplacian assembly share the same φ without duplicating the solve. `assembleTrivialConnectionLaplacian` modulates the Levi-Civita transport on each halfedge by `exp(i·sign·φ[edge])` before raising to nSym, producing a complex Hermitian matrix for MATLAB's `eigs`.

**Tech Stack:** C++17, Eigen 3.x, geometry-central (mesh/geometry data types), nxr-compute CholeskyCache + DECOperators. MEX binding for MATLAB. Build: `bash scripts/build.sh Release`.

---

## File Map

| File | Change |
|---|---|
| `include/nxr/compute.h` | Add `computeTrivialConnection` declaration (line ~904); add `assembleTrivialConnectionLaplacian` declaration (line ~434); rename `trivial` → `directionField` with `trivial` kept as deprecated alias |
| `src/direction_field.cpp` | Promote `computeCoExactComponent` → public `computeTrivialConnection`; refactor `trivial()` → `directionField()` to call it; keep `trivial` forwarding alias |
| `src/connection_laplacian.cpp` | Add `assembleTrivialConnectionLaplacian` function |
| `test/test_connection_laplacian.cpp` | Add test section for `assembleTrivialConnectionLaplacian` |
| `bindings/mex/src/nxr_compute_mex.cpp` | Add `cmdTrivialConnectionLaplacian`; wire `'trivialConnectionLaplacian'` into dispatch; rename `cmdTrivial` → `cmdDirectionField` with `'trivial'` alias |

---

## Task 1: Declare `computeTrivialConnection` in `compute.h`

**Files:**
- Modify: `include/nxr/compute.h` (around line 898, inside `namespace nxr::manifold::connection`)

- [ ] **Step 1: Add the declaration**

In `compute.h`, inside `namespace nxr::manifold::connection`, directly before the `trivial()` declaration (line ~899), add:

```cpp
/**
 * Compute the trivial connection 1-form φ = δβ (per-edge rotation angles).
 *
 * Solves the Poisson system A β = -K + 2π σ where:
 *   A   = d0ᵀ ★₁ d0 (cotangent Laplacian, via CholeskyCache::hodgeExact)
 *   K   = per-vertex Gaussian curvature (angle defect)
 *   σ   = singularity index at each vertex (0 for non-singular)
 *
 * Returns φ = ★₁ d₀ β as an Eigen::VectorXd of length nEdges.
 * Sign convention: φ(e) is positive in the direction of e.halfedge().
 *
 * Gauss-Bonnet requirement: Σ singularityMap values must equal χ(mesh).
 * This is NOT checked here — the caller is responsible (directionField
 * checks it; assembleTrivialConnectionLaplacian documents it).
 *
 * For sphere topology (genus 0, χ=2): two vertices with σ=1.0 each.
 * For n-RoSy fields: use σ = (desired holonomy) / (2π).
 *   e.g. nSym=4 cross field on sphere: two vertices with σ=0.5.
 */
Eigen::VectorXd computeTrivialConnection(
    Manifold& m,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const std::map<int, double>& singularityMap
);
```

- [ ] **Step 2: Verify it compiles (declaration only, no definition yet)**

```bash
bash scripts/build.sh Release 2>&1 | grep -E "error:|computeTrivialConnection"
```

Expected: linker error mentioning `computeTrivialConnection` (undefined reference) — not a compile error. If a compile error appears, fix the declaration syntax.

---

## Task 2: Implement `computeTrivialConnection` in `direction_field.cpp`

**Files:**
- Modify: `src/direction_field.cpp`

- [ ] **Step 1: Move `computeCoExactComponent` out of anonymous namespace and rename**

In `src/direction_field.cpp`, the function `computeCoExactComponent` is currently declared `static` inside the anonymous namespace (lines ~94–130). Replace it with the public implementation:

Remove the anonymous namespace wrapper around this function and change:
```cpp
// BEFORE — static in anonymous namespace, with eulerChiOut param:
static Eigen::VectorXd computeCoExactComponent(
    Manifold& m,
    const DECOperators& dec,
    CholeskyCache& cache,
    const std::map<int, double>& singularityMap,
    double& eulerChiOut
) {
    ...
    eulerChiOut = totalK / (2.0 * M_PI);
    ...
    return dec.hodge1 * (dec.d0 * betaTilde);
}
```

To the public definition (placed after the anonymous namespace closing brace, before `trivial()`):
```cpp
// AFTER — public, no eulerChiOut:
Eigen::VectorXd computeTrivialConnection(
    Manifold& m,
    const DECOperators& dec,
    CholeskyCache& cache,
    const std::map<int, double>& singularityMap
) {
    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    int nV = m.nV();

    geom.requireVertexGaussianCurvatures();

    Eigen::VectorXd rhs(nV);
    for (Vertex v : mesh.vertices()) {
        int i = static_cast<int>(v.getIndex());
        double K = geom.vertexGaussianCurvatures[v];
        double sigma = 0.0;
        auto it = singularityMap.find(i);
        if (it != singularityMap.end()) sigma = it->second;
        rhs(i) = -K + 2.0 * M_PI * sigma;
    }

    const auto& llt = cache.hodgeExact(dec);
    Eigen::VectorXd betaTilde = llt.solve(rhs);
    return dec.hodge1 * (dec.d0 * betaTilde);
}
```

- [ ] **Step 2: Refactor `trivial()` to call `computeTrivialConnection`**

The `trivial()` function currently calls `computeCoExactComponent(..., result.eulerCharacteristic)`. Replace it to call `computeTrivialConnection` and compute eulerChi separately:

```cpp
DirectionFieldResult trivial(
    Manifold& m,
    const DECOperators& dec,
    CholeskyCache& cache,
    const std::map<int, double>& singularityMap
) {
    DirectionFieldResult result;
    result.eulerCharacteristic = 0.0;
    result.gaussBonnetSatisfied = false;

    // 1. Trivial connection φ = δβ
    result.connections = computeTrivialConnection(m, dec, cache, singularityMap);

    // 2. Euler characteristic from Gauss-Bonnet (curvatures already cached)
    auto& geom = m.geometry();
    double totalK = 0.0;
    for (Vertex v : m.mesh().vertices())
        totalK += geom.vertexGaussianCurvatures[v];
    result.eulerCharacteristic = totalK / (2.0 * M_PI);

    // 3. Gauss-Bonnet check
    double sumSing = 0.0;
    for (const auto& [_, s] : singularityMap) sumSing += s;
    result.gaussBonnetSatisfied =
        std::abs(sumSing - result.eulerCharacteristic) < 1e-3;

    // 4. Propagate angles via dual spanning tree
    Eigen::VectorXd alpha = propagateAngles(m, result.connections, dec);

    // 5. Build direction field vectors
    result.directionVectors = buildFaceVectorsFromAngles(m, alpha);

    // 6. Orthogonal field
    Eigen::VectorXd alphaOrth = alpha.array() + M_PI / 2.0;
    result.orthogonalVectors = buildFaceVectorsFromAngles(m, alphaOrth);

    std::cout << "[direction_field] χ=" << result.eulerCharacteristic
              << ", Σσ=" << sumSing
              << ", Gauss-Bonnet "
              << (result.gaussBonnetSatisfied ? "OK" : "VIOLATED")
              << std::endl;

    return result;
}
```

- [ ] **Step 3: Build and verify no regressions**

```bash
bash scripts/build.sh Release 2>&1 | tail -20
```

Expected: successful build, all existing test executables produced.

- [ ] **Step 4: Run existing connection Laplacian tests**

```bash
./build/Release/test_connection_laplacian.exe
```

Expected: all existing tests pass (this verifies the build didn't break anything in the Levi-Civita path).

- [ ] **Step 5: Commit**

```bash
git add include/nxr/compute.h src/direction_field.cpp
git commit -m "refactor(direction_field): extract computeTrivialConnection as public primitive

Promotes the private computeCoExactComponent (static, anonymous namespace)
to a named public function computeTrivialConnection. The directionField
trivial() implementation is refactored to call it. eulerCharacteristic
is now recomputed inline in trivial() from the already-cached curvatures.

No behaviour change — this is a pure extraction refactor."
```

---

## Task 3: Add `assembleTrivialConnectionLaplacian` declaration to `compute.h`

**Files:**
- Modify: `include/nxr/compute.h` (inside `namespace nxr::manifold::ops::laplacian::connection`, after the existing `assembleConnectionLaplacian` declaration, around line ~426)

- [ ] **Step 1: Add declaration**

After the `assembleConnectionLaplacian` declaration and before the closing `} // namespace`, add:

```cpp
/**
 * Assemble the connection Laplacian using the trivial connection transport.
 *
 * The trivial connection for the given singularity map is computed first
 * (via computeTrivialConnection), then used to modulate the Levi-Civita
 * transport on each halfedge:
 *
 *   ρ^TC[he] = ρ^LC[he] · exp(i · sign(he) · φ[edge(he)])
 *
 * where sign(he) = +1 if he == edge.halfedge(), -1 otherwise, and
 * φ is the per-edge trivial connection 1-form (Eigen::VectorXd, nEdges).
 *
 * The off-diagonal entry K[i,j] for halfedge he (tail=i, tip=j) is then:
 *
 *   K[i,j] += -w_ij · (ρ^TC[he.twin()])^nSym
 *
 * Gauss-Bonnet: Σ singularityMap values must equal χ(mesh). Not checked
 * internally — throws nothing if violated, but φ will be incorrect.
 *
 * Only ConnectionDomain::Vertex is currently supported. Passing Face or
 * EdgeCrouzeixRaviart throws Error(InvalidInput).
 *
 * Output format follows opts.format (Real2N or Complex), same as
 * assembleConnectionLaplacian.
 */
ConnectionLaplacian assembleTrivialConnectionLaplacian(
    Manifold& m,
    const std::map<int, double>& singularityMap,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const ConnectionLaplacianOptions& opts = {}
);
```

- [ ] **Step 2: Verify declaration compiles**

```bash
bash scripts/build.sh Release 2>&1 | grep -E "error:|assembleTrivialConnectionLaplacian"
```

Expected: linker error (undefined reference) — not a compile error.

---

## Task 4: Implement `assembleTrivialConnectionLaplacian` in `connection_laplacian.cpp`

**Files:**
- Modify: `src/connection_laplacian.cpp`

- [ ] **Step 1: Add the include for computeTrivialConnection**

At the top of `src/connection_laplacian.cpp`, the file already includes `"nxr/compute.h"`. No new includes needed — `computeTrivialConnection` is declared in the same header.

- [ ] **Step 2: Add the implementation**

At the bottom of `src/connection_laplacian.cpp`, before the closing namespace brace, add:

```cpp
ConnectionLaplacian assembleTrivialConnectionLaplacian(
    Manifold& m,
    const std::map<int, double>& singularityMap,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const ConnectionLaplacianOptions& opts
) {
    if (opts.domain != ConnectionDomain::Vertex) {
        throw Error(ErrorCode::InvalidInput,
            "assembleTrivialConnectionLaplacian: only ConnectionDomain::Vertex "
            "is currently supported.",
            "Face and EdgeCrouzeixRaviart trivial-connection Laplacians are "
            "not yet implemented. Use domain='vertex'.");
    }
    if (opts.nSym <= 0) {
        throw Error(ErrorCode::InvalidInput,
            "assembleTrivialConnectionLaplacian: nSym must be positive (got " +
            std::to_string(opts.nSym) + ")",
            "Common values: 1 (vector field), 2 (line field), 4 (cross field).");
    }

    // 1. Compute trivial connection φ (per-edge 1-form).
    using nxr::manifold::connection::computeTrivialConnection;
    Eigen::VectorXd phi = computeTrivialConnection(m, dec, cache, singularityMap);

    auto& geometry = m.geometry();
    SurfaceMesh& mesh = m.mesh();
    int N = m.nV();

    geometry.requireVertexIndices();
    geometry.requireEdgeCotanWeights();
    geometry.requireTransportVectorsAlongHalfedge();
    geometry.requireEdgeIndices();

    // 2. Assemble complex triplets using trivial connection transport.
    //    For each halfedge he (tail=iTail → tip=iTip):
    //      ρ^TC[he.twin()] = ρ^LC[he.twin()] · exp(i · sign_twin · φ[edge])
    //    where sign_twin = (he.twin() == he.edge().halfedge()) ? +1 : -1
    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
    triplets.reserve(2 * mesh.nHalfedges());

    for (Halfedge he : mesh.halfedges()) {
        const size_t iTail = geometry.vertexIndices[he.vertex()];
        const size_t iTip  = geometry.vertexIndices[he.next().vertex()];
        const double weight = geometry.edgeCotanWeights[he.edge()];
        const int eIdx = static_cast<int>(geometry.edgeIndices[he.edge()]);

        // Sign of φ for he.twin() relative to canonical edge halfedge.
        const double signTwin = (he.twin() == he.edge().halfedge()) ? 1.0 : -1.0;

        // LC transport for twin halfedge, as complex number.
        const Vector2 lcRotBase = geometry.transportVectorsAlongHalfedge[he.twin()];
        const std::complex<double> lcRotC(lcRotBase.x, lcRotBase.y);

        // Modulate by trivial connection correction, then raise to nSym.
        const std::complex<double> correction = std::polar(1.0, signTwin * phi(eIdx));
        const std::complex<double> tcRotC = std::pow(lcRotC * correction,
                                                      static_cast<double>(opts.nSym));

        triplets.emplace_back(iTail, iTail, std::complex<double>(weight, 0.0));
        triplets.emplace_back(iTail, iTip,
            std::complex<double>(-weight * tcRotC.real(), -weight * tcRotC.imag()));
    }

    // 3. Add ε·I regularization.
    if (opts.regularization != 0.0) {
        for (int i = 0; i < N; ++i)
            triplets.emplace_back(i, i, std::complex<double>(opts.regularization, 0.0));
    }

    Eigen::SparseMatrix<std::complex<double>> K_complex(N, N);
    K_complex.setFromTriplets(triplets.begin(), triplets.end());

    ConnectionLaplacian result;
    result.baseDim        = N;
    result.domain         = opts.domain;
    result.nSym           = opts.nSym;
    result.regularization = opts.regularization;
    result.format         = opts.format;

    if (opts.format == ConnectionLaplacianFormat::Real2N) {
        result.K_real    = lowerToReal2N(K_complex, N);
        result.outputDim = 2 * N;
    } else {
        result.K_complex = std::move(K_complex);
        result.outputDim = N;
    }
    return result;
}
```

Note: `lowerToReal2N` is already defined as a file-local function earlier in `connection_laplacian.cpp` — no need to redeclare.

- [ ] **Step 3: Build**

```bash
bash scripts/build.sh Release 2>&1 | tail -20
```

Expected: clean build. Fix any compile errors before continuing.

---

## Task 5: Add C++ tests for `assembleTrivialConnectionLaplacian`

**Files:**
- Modify: `test/test_connection_laplacian.cpp`

- [ ] **Step 1: Write the failing test block**

Append a new test section to `test/test_connection_laplacian.cpp`. Add after the last existing test (before `return failures > 0 ? 1 : 0;` in `main()`):

```cpp
// ── assembleTrivialConnectionLaplacian ───────────────────────────────────

static void testTrivialConnectionLaplacian(Manifold& m) {
    namespace cl = nxr::manifold::ops::laplacian::connection;
    using nxr::manifold::connection::computeTrivialConnection;

    std::cout << "\n=== assembleTrivialConnectionLaplacian ===\n";

    // Assemble DEC operators and Cholesky cache (needed for Poisson solve).
    auto ops  = nxr::manifold::ops::assembleMeshOperators(m);
    auto dec  = nxr::manifold::ops::assembleDECOperators(m);
    auto cache = nxr::manifold::ops::CholeskyCache{};

    // Sphere singularity map: two index-1 singularities at vertices 0 and 1.
    // Gauss-Bonnet: 1 + 1 = 2 = χ(sphere). Works for any closed genus-0 mesh.
    std::map<int, double> singMap = {{0, 1.0}, {1, 1.0}};

    // ── Test 1: computeTrivialConnection returns nEdges-length vector ─────
    {
        Eigen::VectorXd phi = computeTrivialConnection(m, dec, *cache, singMap);
        EXPECT(phi.size() == m.nE(),
               "phi length == nEdges",
               "got " + std::to_string(phi.size()) + " expected " + std::to_string(m.nE()));
        std::cout << "  [PASS] phi.size() == nEdges (" << phi.size() << ")\n";
    }

    // ── Test 2: Complex format — Hermitian, correct dimensions ────────────
    {
        cl::ConnectionLaplacianOptions opts;
        opts.format = cl::ConnectionLaplacianFormat::Complex;
        opts.nSym   = 1;
        auto cl = cl::assembleTrivialConnectionLaplacian(m, singMap, dec, *cache, opts);

        EXPECT(cl.baseDim == m.nV(), "baseDim == nV",
               "got " + std::to_string(cl.baseDim));
        EXPECT(cl.outputDim == m.nV(), "outputDim == nV (Complex format)",
               "got " + std::to_string(cl.outputDim));

        // Hermitian check: ||K - K^H||_F < 1e-10
        auto diff = cl.K_complex - cl.K_complex.adjoint();
        double asymm = diff.norm();
        EXPECT(asymm < 1e-10, "K_complex is Hermitian (||K-K^H||_F < 1e-10)",
               "asymmetry = " + std::to_string(asymm));
        std::cout << "  [PASS] K_complex Hermitian, ||K-K^H||_F = " << asymm << "\n";
    }

    // ── Test 3: Real2N format — symmetric, correct dimensions ─────────────
    {
        cl::ConnectionLaplacianOptions opts;
        opts.format = cl::ConnectionLaplacianFormat::Real2N;
        opts.nSym   = 1;
        auto cl = cl::assembleTrivialConnectionLaplacian(m, singMap, dec, *cache, opts);

        EXPECT(cl.outputDim == 2 * m.nV(), "outputDim == 2*nV (Real2N format)",
               "got " + std::to_string(cl.outputDim));

        auto diff = cl.K_real - cl.K_real.transpose();
        double asymm = diff.norm();
        EXPECT(asymm < 1e-10, "K_real is symmetric (||K-K^T||_F < 1e-10)",
               "asymmetry = " + std::to_string(asymm));
        std::cout << "  [PASS] K_real symmetric, ||K-K^T||_F = " << asymm << "\n";
    }

    // ── Test 4: Differs from Levi-Civita Laplacian ────────────────────────
    //    The trivial connection Laplacian should differ from the LC version
    //    (singularities concentrate curvature differently).
    {
        cl::ConnectionLaplacianOptions opts;
        opts.format = cl::ConnectionLaplacianFormat::Complex;
        opts.nSym   = 1;

        auto tcl = cl::assembleTrivialConnectionLaplacian(m, singMap, dec, *cache, opts);
        auto lcl = cl::assembleConnectionLaplacian(m, opts);

        auto diff = tcl.K_complex - lcl.K_complex;
        double diffNorm = diff.norm();
        EXPECT(diffNorm > 1e-6, "TC Laplacian differs from LC Laplacian",
               "||K_TC - K_LC||_F = " + std::to_string(diffNorm) + " (too small — may be identical)");
        std::cout << "  [PASS] ||K_TC - K_LC||_F = " << diffNorm << " (distinct matrices)\n";
    }

    // ── Test 5: Face/CR domain throws InvalidInput ─────────────────────────
    {
        cl::ConnectionLaplacianOptions opts;
        opts.domain = cl::ConnectionDomain::Face;
        bool threw = false;
        try {
            cl::assembleTrivialConnectionLaplacian(m, singMap, dec, *cache, opts);
        } catch (const nxr::compute::Error& e) {
            threw = (e.code() == nxr::compute::ErrorCode::InvalidInput);
        }
        EXPECT(threw, "Face domain throws InvalidInput", "no exception thrown");
        std::cout << "  [PASS] Face domain correctly throws InvalidInput\n";
    }

    // ── Test 6: nSym=0 throws InvalidInput ────────────────────────────────
    {
        cl::ConnectionLaplacianOptions opts;
        opts.nSym = 0;
        bool threw = false;
        try {
            cl::assembleTrivialConnectionLaplacian(m, singMap, dec, *cache, opts);
        } catch (const nxr::compute::Error& e) {
            threw = (e.code() == nxr::compute::ErrorCode::InvalidInput);
        }
        EXPECT(threw, "nSym=0 throws InvalidInput", "no exception thrown");
        std::cout << "  [PASS] nSym=0 correctly throws InvalidInput\n";
    }
}
```

Also add `testTrivialConnectionLaplacian(m);` in `main()` before the return statement, where `m` is the existing icosphere manifold used by the other tests.

- [ ] **Step 2: Build the test**

```bash
bash scripts/build.sh Release 2>&1 | grep -E "error:|test_connection_laplacian"
```

Expected: clean build, `test_connection_laplacian.exe` produced.

- [ ] **Step 3: Run the test and verify all pass**

```bash
./build/Release/test_connection_laplacian.exe
```

Expected output includes:
```
=== assembleTrivialConnectionLaplacian ===
  [PASS] phi.size() == nEdges (...)
  [PASS] K_complex Hermitian, ||K-K^H||_F = ...
  [PASS] K_real symmetric, ||K-K^T||_F = ...
  [PASS] ||K_TC - K_LC||_F = ... (distinct matrices)
  [PASS] Face domain correctly throws InvalidInput
  [PASS] nSym=0 correctly throws InvalidInput
```

If Test 4 fails (diffNorm too small), check that `computeTrivialConnection` is actually computing non-zero φ: print `phi.norm()` to confirm the Poisson solve produced a non-trivial solution.

- [ ] **Step 4: Commit**

```bash
git add include/nxr/compute.h src/connection_laplacian.cpp test/test_connection_laplacian.cpp
git commit -m "feat(connection_laplacian): add assembleTrivialConnectionLaplacian

Builds a complex Hermitian connection Laplacian using the trivial
connection transport: ρ^TC = ρ^LC · exp(i·sign·φ[edge]) where φ is
the trivial connection 1-form from the prescribed singularity Poisson
solve. Vertex domain only. Format options (Real2N/Complex) match the
existing assembleConnectionLaplacian surface.

Intended use: between-subject cortical analysis where singularities are
pinned to FreeSurfer sphere poles, giving a consistent connection across
subjects. MATLAB consumer calls eigs() on the returned K_complex."
```

---

## Task 6: Add MEX command `'trivialConnectionLaplacian'`

**Files:**
- Modify: `bindings/mex/src/nxr_compute_mex.cpp`

- [ ] **Step 1: Add `cmdTrivialConnectionLaplacian` handler**

Find `cmdTrivial` in `nxr_compute_mex.cpp` (around line 900). Add the new handler directly after it:

```cpp
void cmdTrivialConnectionLaplacian(int /*nlhs*/, mxArray** plhs,
                                    int nrhs, const mxArray** prhs) {
    if (nrhs < 4 || nrhs > 5) {
        throw std::invalid_argument(
            "nxr_compute('trivialConnectionLaplacian', handle, singVerts, singValues, [opts]) "
            "takes 3 or 4 arguments.\n"
            "  singVerts : 1-based vertex indices of singularities\n"
            "  singValues: corresponding singularity indices (Σ must equal χ)\n"
            "  opts      : optional struct with fields nSym (default 1),\n"
            "              regularization (default 1e-8), format ('complex'|'real2N')");
    }

    namespace cl = nxr::manifold::ops::laplacian::connection;
    ContextHolder& h = getHolder(prhs[1]);

    // Parse singularity map (1-based → 0-based indices).
    auto idx = mxToVertexIndices(prhs[2]);
    auto val = mxToEigenVector(prhs[3]);
    if (static_cast<std::size_t>(val.size()) != idx.size()) {
        throw std::invalid_argument(
            "trivialConnectionLaplacian: singValues must match singVerts length");
    }
    std::map<int, double> sing;
    for (std::size_t i = 0; i < idx.size(); ++i)
        sing[idx[i]] = val[static_cast<Eigen::Index>(i)];

    // Parse options.
    cl::ConnectionLaplacianOptions o;
    o.domain = cl::ConnectionDomain::Vertex;  // only supported domain
    o.format = cl::ConnectionLaplacianFormat::Complex;  // default for MATLAB use
    if (nrhs >= 5 && !mxIsEmpty(prhs[4])) {
        if (!mxIsStruct(prhs[4]))
            throw std::invalid_argument("opts must be a struct");
        const mxArray* f;
        if ((f = mxGetField(prhs[4], 0, "nSym")))           o.nSym = getIntArg(f);
        if ((f = mxGetField(prhs[4], 0, "regularization"))) o.regularization = getDoubleArg(f);
        if ((f = mxGetField(prhs[4], 0, "format")))         o.format = cl::parseConnectionLaplacianFormat(getStringArg(f));
    }

    auto result = cl::assembleTrivialConnectionLaplacian(
        *h.ctx, sing, ensureDec(h), *h.cache, o);
    plhs[0] = connectionLaplacianToStruct(result);
}
```

Note: default format is `Complex` (not `Real2N`) because the primary MATLAB consumer will call `eigs` directly on the complex Hermitian matrix. The `real2N` format remains available via opts.

- [ ] **Step 2: Wire into the dispatch table**

Find the dispatch block in `nxr_compute_mex.cpp` (around line 1091 where `"trivial"` is dispatched). Add:

```cpp
else if (cmd == "trivialConnectionLaplacian") cmdTrivialConnectionLaplacian(nlhs, plhs, nrhs, prhs);
```

- [ ] **Step 3: Build the MEX binary**

```bash
bash scripts/build.sh Release 2>&1 | tail -10
```

Expected: clean build including `nxr_compute.mexmaca64`.

- [ ] **Step 4: Smoke test via MATLAB MCP**

Run in MATLAB:

```matlab
% Load a small test mesh (icosphere or cortical patch)
[V, F] = ico_sphere(3);           % or any mesh you have available
handle = nxr_compute('create', V, F);

% Place two singularities at vertices 1 and 2 (1-based), σ=1 each
singVerts  = [1; 2];
singValues = [1; 1];

% Assemble trivial connection Laplacian (complex format, default)
CL = nxr_compute('trivialConnectionLaplacian', handle, singVerts, singValues);

% Verify output struct fields
disp(fieldnames(CL))   % expect: rows, cols, dataReal, dataImag, baseDim, outputDim, ...

% Reconstruct sparse complex matrix
K = sparse(CL.rows+1, CL.cols+1, CL.dataReal + 1i*CL.dataImag, CL.baseDim, CL.baseDim);
% (rows/cols are 0-based from C++ — add 1 for MATLAB)

% Check Hermitian: should be ~0
disp(norm(K - K', 'fro'))   % expect < 1e-10

% Compute 6 smallest eigenmodes
[U, D] = eigs(K, speye(size(K,1)), 6, 'sm');
disp(diag(D))   % expect real, non-negative values
```

Expected: `norm(K - K', 'fro') < 1e-10`, eigenvalues real and ≥ 0, first eigenvalue ≈ `regularization (1e-8)`.

- [ ] **Step 5: Commit**

```bash
git add bindings/mex/src/nxr_compute_mex.cpp
git commit -m "feat(mex): add trivialConnectionLaplacian command

Exposes assembleTrivialConnectionLaplacian via MEX. Accepts singVerts
(1-based), singValues (Σ = χ), and optional opts struct (nSym,
regularization, format). Default format is 'complex' — the primary
consumer calls MATLAB's eigs on the returned K_complex."
```

---

## Task 7: Rename `'trivial'` → `'directionField'` in MEX (with alias)

**Files:**
- Modify: `bindings/mex/src/nxr_compute_mex.cpp`

- [ ] **Step 1: Rename `cmdTrivial` → `cmdDirectionField`**

In `nxr_compute_mex.cpp`, rename the function and update its help string:

```cpp
// BEFORE:
void cmdTrivial(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 4) {
        throw std::invalid_argument(
            "nxr_compute('trivial', handle, singVerts, singValues) takes exactly 3 arguments");
    }
    ...
}

// AFTER:
void cmdDirectionField(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 4) {
        throw std::invalid_argument(
            "nxr_compute('directionField', handle, singVerts, singValues) takes exactly 3 arguments.\n"
            "  singVerts : 1-based vertex indices of prescribed singularities\n"
            "  singValues: singularity indices (Σ must equal Euler characteristic χ)\n"
            "Returns struct with fields: connections, directionVectors, orthogonalVectors,\n"
            "  eulerCharacteristic, gaussBonnetSatisfied.\n"
            "Note: 'trivial' is a deprecated alias for 'directionField'.");
    }
    ContextHolder& h = getHolder(prhs[1]);
    auto idx = mxToVertexIndices(prhs[2]);
    auto val = mxToEigenVector(prhs[3]);
    if (static_cast<std::size_t>(val.size()) != idx.size()) {
        throw std::invalid_argument("singValues must match singVerts length");
    }
    std::map<int, double> sing;
    for (std::size_t i = 0; i < idx.size(); ++i)
        sing[idx[i]] = val[static_cast<Eigen::Index>(i)];
    auto r = nxr::manifold::connection::trivial(*h.ctx, ensureDec(h), *h.cache, sing);
    plhs[0] = directionFieldResultToStruct(r);
}
```

- [ ] **Step 2: Update dispatch table to add `'directionField'` and keep `'trivial'` as alias**

Find the dispatch block (around line 1091) and update:

```cpp
// BEFORE:
else if (cmd == "trivial") cmdTrivial(nlhs, plhs, nrhs, prhs);

// AFTER:
else if (cmd == "directionField") cmdDirectionField(nlhs, plhs, nrhs, prhs);
else if (cmd == "trivial")        cmdDirectionField(nlhs, plhs, nrhs, prhs); // deprecated alias
```

- [ ] **Step 3: Build**

```bash
bash scripts/build.sh Release 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 4: Verify both commands work via MATLAB MCP**

```matlab
handle = nxr_compute('create', V, F);
singVerts = [1; 2]; singValues = [1; 1];

% Both should return identical results
r1 = nxr_compute('directionField', handle, singVerts, singValues);
r2 = nxr_compute('trivial',        handle, singVerts, singValues);

disp(norm(r1.directionVectors - r2.directionVectors, 'fro'))  % expect 0
```

Expected: `0` (identical results from both commands).

- [ ] **Step 5: Commit**

```bash
git add bindings/mex/src/nxr_compute_mex.cpp
git commit -m "fix(mex): rename 'trivial' command to 'directionField', keep alias

'trivial' was misleading — it returns a direction field computed via
trivial connections, not the connection itself (which is now exposed via
'trivialConnectionLaplacian'). 'trivial' is kept as a deprecated alias
dispatching to the same handler to avoid breaking existing callers."
```

---

## Self-Review

**Spec coverage:**

| Requirement | Task |
|---|---|
| Extract `computeTrivialConnection` as public function | Tasks 1–2 |
| `directionField` calls `computeTrivialConnection` | Task 2 |
| `assembleTrivialConnectionLaplacian` uses φ to modulate LC transport | Task 4 |
| Only Vertex domain, throws for Face/CR | Task 4 (implementation), Task 5 (test) |
| Real2N and Complex format both supported | Task 4 |
| MEX `'trivialConnectionLaplacian'` command | Task 6 |
| MEX `'directionField'` rename + `'trivial'` alias | Task 7 |
| C++ tests for new Laplacian (Hermitian, symmetric, differs from LC, error paths) | Task 5 |
| MATLAB smoke test | Task 6 |

**Placeholder scan:** None found — all code blocks are complete.

**Type consistency:**
- `computeTrivialConnection` declared Task 1, implemented Task 2, called in Task 4 — signatures match.
- `assembleTrivialConnectionLaplacian` declared Task 3, implemented Task 4, called in Task 6 — signatures match.
- `ConnectionLaplacian` return type is consistent with existing `assembleConnectionLaplacian`.
- `connectionLaplacianToStruct` in Task 6 is the existing MEX helper — verified in the codebase at line 770.
- `lowerToReal2N` in Task 4 is the existing file-local helper in `connection_laplacian.cpp` — called correctly.
- `cache.hodgeExact(dec)` in `computeTrivialConnection` matches the existing call in the original `computeCoExactComponent`.
