# Face-domain (dual) Dirac Operator — Design

**Date:** 2026-06-10
**Status:** Draft (design); pending review
**Builds on:** `2026-06-10-extrinsic-dirac-operator-design.md` (the vertex-domain
`operators().dirac(τ)`). This is its **face-domain dual**.
**Reference:** Liu/Jacobson/Crane, *A Dirac Operator for Extrinsic Shape
Analysis* (SGP 2017); Crane–Pinkall–Schröder (SIGGRAPH 2011).

---

## Goal

A **face-domain** relative-Dirac family `operators().diracFace(τ)`: a curvature-
aware spectral operator whose quaternionic function lives on **faces** (one
quaternion per triangle / centroid), built so the Gauss map is sampled as the
**exact per-face normal** (a face normal is well-defined and constant on a flat
triangle — no angle/area-weighted averaging, unlike a vertex normal). Geometry-
only: it produces a face-based curvature-aware eigenbasis from the cortex; any
data (leadfield) expansion onto it is downstream and out of scope.

It is the `V ↔ F` dual of the vertex-domain `dirac(τ)`:

| | vertex-domain `dirac(τ)` | face-domain `diracFace(τ)` |
|---|---|---|
| function `ψ` lives on | vertices (`4V`) | faces (`4F`) |
| Gauss map sampled at | vertex normals (PL) | **face normals (exact)** |
| output cell aggregated over | each triangular **face** (3 vertices) | each **vertex star** = dual face (`d` faces) |
| measure `⋆` | face areas | vertex dual areas |
| τ=0 intrinsic anchor | `cotanL ⊗ I₄` | `K̃ ⊗ I₄` (DEC 2-form Laplacian) |
| eigenbasis on | vertices | faces |

---

## Why the dual structure is REQUIRED (not a stylistic choice)

The earlier exploration considered a cheaper "edge-only" assembly: one quaternion
per face, couple the two faces of each edge through the normal jump
`N_{f₂} − N_{f₁}`. **That construction is degenerate and must not be used.** With a
single jump per edge,

```
(D̃ψ)_e = w_e · L_(N_{f₂}−N_{f₁}) · (ψ_{f₂} − ψ_{f₁})
‖(D̃ψ)_e‖² = w_e² · |N_{f₂}−N_{f₁}|² · |ψ_{f₂}−ψ_{f₁}|²
```

because quaternion multiplication is norm-multiplicative (`|L_v u|² = |v|²|u|²`).
The quaternion structure cancels: the energy collapses to a **scalar** curvature²-
weighted face-graph Laplacian `⊗ I₄` — no inter-component coupling, trivially
4-fold degenerate. It is a weighted Laplacian, not a Dirac operator.

The vertex-domain operator avoids this because each output cell (a triangle)
aggregates **three** vertices with **three different** normal-differences, summed
*before* the `|·|²`; the cross terms `(N_k−N_j)ψ_i · \overline{(N_i−N_k)ψ_j}` are
products of *different* imaginary quaternions — non-commutative, generally
non-real — and populate the off-diagonal `4×4` blocks. That multi-term
aggregation is what makes it a Dirac.

So the genuine quaternionic coupling **requires an output cell that aggregates
≥ 2 faces with different normals**. On a triangle mesh that cell is exactly the
**vertex star (dual face)**. The edge cell (2 faces, one jump) is too small. Hence
the face-domain Dirac must aggregate over vertex stars — it is genuinely the dual
of Liu's per-face aggregation, not an edge assembly.

---

## Mathematical specification

Quaternion `q = a + b·i + c·j + d·k` ↔ ℝ⁴ in order `[w,x,y,z]`; left-multiplication
by a purely-imaginary `v=(x,y,z)` is the `4×4` antisymmetric `L_v` (same
`leftMulImag` as the vertex-domain operator). Storage is **face-interleaved**:
face `f`, component `c ∈ {0,1,2,3}` → index `4f + c`.

### Inputs (geometry-only)
- **Face normals** `N_f` — exact unit normals, one per triangle (from the embedding).
- **Vertex dual areas** `Ã_v` — the `⋆_V` measure (existing `vertexDualAreas`).
- **Face areas** `A_f` — the eigenproblem mass (`⋆₂`).
- **DEC operators** `d₁`, `⋆₁⁻¹` (= `hodge1Inverse`) — for the intrinsic anchor
  (existing `DECOperators`).
- **Cyclic one-ring face order** around each vertex (geometry-central
  `v.adjacentFaces()` / outgoing-halfedge traversal — closed cycle on interior
  vertices).

### Extrinsic block `Ẽ`
Rectangular operator `D̃ : ℍ^F → ℍ^V` (`[4V × 4F]`). For each **interior** vertex
`v` with cyclically ordered incident faces `f₀,…,f_{d−1}`:

```
(D̃ψ)_v = −(1 / 2Ã_v) · Σ_{k=0}^{d−1} (N_{f_{k+1}} − N_{f_{k−1}}) · ψ_{f_k}      (k mod d)
```

i.e. the block on `(v, f_k)` is `−L_(N_{f_{k+1}} − N_{f_{k−1}}) / (2Ã_v)`. The
coefficient on `f_k` is the difference of the normals of its two **neighbours in
the star cycle** (the dual of "the edge opposite vertex p" in the primal). The
sum telescopes around the closed star → the per-face constant `ψ_f ≡ c` is in
`ker D̃`.

Self-adjoint face operator via the Galerkin square with the vertex dual-area mass:

```
Ẽ = D̃ᵀ · ⋆_V · D̃ ,      ⋆_V = diag(Ã_v) ⊗ I₄        → [4F × 4F], symmetric PSD
```

### Intrinsic anchor `K̃` (the τ=0 block)
The DEC **2-form Hodge Laplacian** on faces, symmetric PSD `[F × F]`:

```
K̃ = d₁ · ⋆₁⁻¹ · d₁ᵀ
```

built from the existing `DECOperators.d1` / `hodge1Inverse`. It is a face-graph
Laplacian with dual weights `⋆₁⁻¹_e`; the per-face constant is in its kernel
(matching `Ẽ`). This is the face-domain analogue of `cotanL` — a scalar operator,
so it needs no quaternionic coupling and is correctly `d₁`-assembled.

### The family
```
L̃(τ) = (1 − τ)·(K̃ ⊗ I₄)  +  τ·Ẽ          (4F × 4F, real symmetric), τ ∈ [0,1]
```

`τ=0` ⇒ `K̃ ⊗ I₄` (intrinsic face Laplacian); `τ=1` ⇒ pure extrinsic face Dirac `Ẽ`.

### Eigenproblem
```
L̃(τ) φ = λ B̃ φ ,      B̃ = diag(A_f) ⊗ I₄         (face-area mass)
```

`L̃(τ)` commutes with right-ℍ-multiplication (the `D̃` blocks are left-multiplications,
`K̃⊗I₄` is block-scalar) ⇒ eigenvalues come in exact **4-fold quaternionic
multiplets**; the per-face constant is a 4-fold zero for all τ. Reuses the existing
`solve::eigen` / `solve::normalize` path unchanged.

---

## Architecture & surfaces

### C++ — operators facet
A sibling to `dirac(τ)` on `OperatorsFacet`:

```cpp
// L̃(τ) = (1−τ)(K̃ ⊗ I₄) + τ·Ẽ, [4F × 4F] real symmetric sparse, by value.
// τ=0 ⇒ DEC 2-form Laplacian ⊗ I₄; τ=1 ⇒ pure extrinsic face Dirac Ẽ. τ ∈ [0,1].
Eigen::SparseMatrix<double> diracFace(double tau) const;
```

Library backing: `ops::dirac::extrinsicBlockFace(Manifold&) → SparseMatrix<double>`
(the `Ẽ` assembly), in `src/dirac_operator.cpp` alongside the vertex `extrinsicBlock`.

### Caching
`OperatorId::DiracFace` caches **only `Ẽ`** (the expensive vertex-star assembly).
`K̃ ⊗ I₄` is derived on demand from the cached DEC `d1`/`hodge1Inverse` (cheap
products + kron), blended by value. `Ẽ` is geometry-only (face normals + dual
areas) ⇒ **not** gauge-dependent, **not** invalidated by `setGauge` (same as the
vertex `Dirac`). `τ=0` skips building `Ẽ`; `τ=1` skips `K̃`.

### MEX
`nxr_compute('operators', h, 'diracFace', τ)` → native real sparse `[4F × 4F]`
(numeric `τ` in `prhs[3]`, mirroring the `dirac` family). MATLAB eigenbasis:
`eigs(L, kron(speye(4), Af))`-style — note the **face-interleaved** `4f+c` layout
makes the mass `kron(Af, speye(4))` (face-major), consistent with the vertex
operator's `kron(Mg, speye(4))`.

### Storage convention
Face-interleaved `[w,x,y,z]` at index `4f + c` (the face-domain analogue of the
vertex operator's `4v+c`). `K̃ ⊗ I₄` = `kron(K̃, I₄)`.

---

## Tests (native `test/test_dirac_face_operator.cpp`, icosphere + flat fixtures)

1. **Intrinsic anchor (headline).** `diracFace(0)` equals `K̃ ⊗ I₄` byte-for-byte
   (independent `K̃ = d₁ ⋆₁⁻¹ d₁ᵀ` from the DEC operators).
2. **Shape & symmetry.** `[4F × 4F]`; `L̃(τ) == L̃(τ)ᵀ` for several τ.
3. **PSD** for `τ < 1`; `Ẽ` (τ=1) PSD with the per-face-constant in its kernel.
4. **Constants in kernel.** `Ẽ · (ψ_f ≡ c)` = 0 (telescoping), for a constant
   quaternion `c`.
5. **Flat fixture.** On a planar patch all face normals equal ⇒ `Ẽ ≡ 0` (pure kernel).
6. **Quaternionic structure.** Eigenvalues in 4-fold multiplets after
   `normalize`; `ΦᵀB̃Φ ≈ I`.
7. **Cache lifecycle.** `OperatorId::DiracFace` populates on `τ>0`, released by
   `releaseOperator`, re-blends correctly with cached `Ẽ`, `diracFace(0)` does not
   build `Ẽ`; `setGauge` does not invalidate it.

MEX test `bindings/mex/test/test_dirac_face_operator.m`: `diracFace(0)` matches
the DEC 2-form Laplacian kron; `eigs` returns real eigenvalues in 4-fold multiplets.

---

## Open question (one) — confirm before planning

**`Ẽ` vs `dirac(τ)` co-spectrality is NOT expected and not the goal.** Worth
confirming the intent: this operator genuinely *re-samples* the Gauss map at face
normals (exact, dihedral-curvature-based), so its spectrum differs from the
vertex operator's — it is a new operator, not a face-located view of the same one.
(The cheap "face-located, same spectrum" alternative — `D · M · Dᵀ` reusing the
vertex `D` — is explicitly NOT this; it keeps vertex normals. Flagging so the
choice is conscious.)

Everything else (the vertex-star aggregation, the `K̃` anchor, the face-area mass,
face-interleaved storage, caching model, MEX surface) follows the vertex-domain
operator's settled design.

---

## Scope boundaries (v1)

**In:** `operators().diracFace(τ)` (real `4F×4F`), its `Ẽ`/`OperatorId::DiracFace`
cache, the MEX `diracFace` surface, eigenbasis via the existing solver. Closed
cortex meshes.

**Deferred / out:** boundary vertices with **open stars** (the telescoping sum is
not closed — needs boundary terms; cortical hemispheres are closed, so v1 assumes
closed stars and should validate/raise on boundary); cross-surface canonical
representative; any leadfield/data expansion; Node/WASM exposure (MEX-only, like
the other bundle operators); a unified `dirac(τ, domain)` enum (kept as a separate
`diracFace` method for v1 — can be merged later if a third domain appears).
