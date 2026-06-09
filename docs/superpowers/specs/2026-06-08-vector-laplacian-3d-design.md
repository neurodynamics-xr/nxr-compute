# 3D Vector (Covariant) Laplacian — `Gauge.operators.laplacian3D`

**Date:** 2026-06-08
**Status:** Design — pending review
**Builds on:** the operators surface (`2026-06-08-operators-surface-design.md`) and the gauge/grid bundle.

---

## 1. Goal

Add the 3-component sibling to the existing Laplacian ladder:

| acts on | DOF/vertex | operator |
|---|---|---|
| scalars | 1 | cotan Laplace–Beltrami (`Geometry.operators.laplacian`) |
| tangent vectors | 2 (complex `z`) | connection Laplacian (`Gauge.operators.laplacian`) |
| **3D frame vectors** | **3 (`a,b` tangent + `c` normal)** | **`Gauge.operators.laplacian3D`** ← this spec |

It is the covariant Laplacian on the rank-3 frame bundle `[e1, e2, n]` defined by
the gauge grid, acting on a full 3D vector per vertex expressed in frame
coordinates `(a, b, c) = (Re z, Im z, normal)`.

## 2. The two couplings

The operator factors as **(tangent connection) × (coupling)**:

- **tangent connection** follows the gauge type (parallel to how `.laplacian`
  already does): Levi-Civita for `euclidean`/`levi-civita`, trivial for `trivial`.
- **coupling ∈ {'product', 'ambient'}**:
  - `product` — tangent and normal decoupled: `blkdiag(K_tangent, L_cotan)`. The
    Bochner Laplacian of `TM ⊕ NM` with the product connection.
  - `ambient` — tangent and normal coupled by the **discrete shape operator**
    (second fundamental form). The genuine extrinsic covariant Laplacian
    (Bochner Laplacian of the embedding-induced connection on `TℝΒ³|_M`).

This gives a 2×2 family (LC/trivial tangent × product/ambient coupling). Default
coupling is **`ambient`** (the true 3D covariant operator); `product` is the cheap
decoupled variant.

**Why the trivial frame is natural for `ambient` (design note).** The grid normal
`n = e1×e2` is gauge-invariant (a tangent rotation preserves the cross product),
so the trivial 3-frame is surface-adapted (true normal) with globally-consistent
tangent axes. The *full* frame-relative rotation `F_iᵀF_j` collapses the energy to
the frame-independent componentwise-Cartesian Laplacian (not covariant), so
covariance must come from the **normal-aligning** rotation (depends only on the
normals, gauge-independent) — whose leftover, after the trivial gauge trivializes
the tangent–tangent block, is exactly the tangent↔normal shape-operator coupling.

## 3. Representation

A single **3N×3N real symmetric sparse** matrix `L3` acting on stacked frame
coordinates `[a; b; c]` (each an N-block), where `a = Re z`, `b = Im z`,
`c = normal component`. This is exactly the `(z, c)` pair the leadfield
decomposition already yields (`z` from `Ltan = G·cᵀ`, `c` from `Ln`), so a consumer
applies it as `L3 * [real(z); imag(z); cn]`.

Block structure:
```
        a(N)        b(N)        c(N)
a   [  Kaa         Kab    |   Sac    ]   tangent 2N block =
b   [  Kba         Kbb    |   Sbc    ]     [[Re K, −Im K],[Im K, Re K]]
    [ ----------------------------- ]     (K = gauge connection Laplacian, V×V complex)
c   [  Sacᵀ        Sbcᵀ   |   Lcc    ]   normal block Lcc = cotan Laplacian
```
- `coupling='product'` → `Sac = Sbc = 0` (pure blkdiag).
- `coupling='ambient'` → `Sac, Sbc` are the discrete shape-operator coupling.

Symmetric PSD (Bochner Laplacian). Native MATLAB **real** sparse.

## 4. Construction

Unified per-edge transport. For edge `(i,j)` with cotan weight `w_ij`:
- `R_ij ∈ SO(3) = R^tan_ij ∘ R^nrm_ij`, where
  - `R^tan_ij` = the tangent rotation of the active gauge connection (Levi-Civita
    transport, or trivial = Levi-Civita · the trivial-connection edge rotation),
    acting in the tangent plane.
  - `R^nrm_ij` = the normal-aligning rotation taking `n_j` to `n_i` (Rodrigues
    about `n_i × n_j` by the angle between them). Included only for `ambient`;
    identity for `product`.
- Accumulate the Dirichlet energy `E = Σ w_ij |w_i − R_ij w_j|²` in frame coords
  → `L3 = Σ w_ij (selector)ᵀ(I − R_ij in frame coords)...` assembled as a
  symmetric sparse matrix (standard graph-Laplacian-with-rotations assembly, the
  same shape as the connection-Laplacian assembly but with 3×3 blocks).

Equivalent, and the recommended assembly: build the tangent `2N` block by the
complex→real expansion of `K` (the gauge connection Laplacian we already
assemble), the normal block from the cotan Laplacian, and — for `ambient` — the
two coupling blocks from the per-edge shape operator. The two assembly routes must
agree (cross-check in the test).

The discrete shape operator coupling per edge derives from the normal change
`n_i → n_j` resolved in the tangent frame; on a sphere of radius r it must reduce
to `S = (1/r)·I` (so `ambient` = `product` + `(1/r)`-coupling), and on a flat
patch `S = 0` (so `ambient ≡ product`). These are the validation anchors.

## 5. API / schema

```
Gauge.operators.laplacian3D    3N×3N real sparse   % present when operators=true
```
Selected by the gauge type (tangent connection) + a `coupling` option. Surfaced
on the `Gauge.operators` sub-struct alongside `.laplacian`. The `operators` opt-in
flag gains a companion for the coupling choice:

```matlab
% ambient (default) covariant 3D Laplacian, trivial tangent connection:
B = nxr_compute('bundle', h, 'trivial', ...
      struct('singVerts',sv,'singValues',si,'operators',true));
L3 = B.Gauge.operators.laplacian3D;          % 3N×3N real sparse, [a;b;c]

% product variant:
G = nxr_compute('gauge', h, 'levi-civita', ...
      struct('operators',true,'coupling','product'));
```
`coupling` is read from the same opts struct as `operators` (default `'ambient'`).

**Library:** a new `nxr::manifold::ops::laplacian::connection` entry, e.g.
`assembleVectorLaplacian3D(Manifold&, gaugeType, coupling, singMap?)`, returning a
real `Eigen::SparseMatrix<double>` of size `3N×3N`. Caching: keyed on
`(gaugeType, coupling)` on the ContextHolder (new slot or extend the operator
cache).

## 6. Test plan

- **Native** (`test/test_geometry_bundle.cpp`): on the icosphere,
  - `L3` is 3N×3N, symmetric (`||L3 − L3ᵀ|| < 1e-9`), PSD (min eig ≥ −1e-9).
  - `product` is exactly block-diagonal: the `(a,b)|c` off-diagonal blocks are
    zero; the `(a,b)` block equals the complex→real expansion of `K`; the `c`
    block equals the cotan Laplacian.
  - `ambient` differs from `product` only in the coupling blocks; on the unit
    icosphere the coupling magnitude ≈ the mean curvature (`κ ≈ 1`).
  - the two assembly routes (per-edge SO(3) vs block-assembly) agree.
- **MATLAB** (`bindings/mex/test/test_operators.m`):
  - `Gauge.operators.laplacian3D` is `3N×3N` real sparse, symmetric.
  - `coupling='product'` reproduces `blkdiag([[Re K,−Im K],[Im K,Re K]], L_cotan)`
    from the already-exposed `Gauge.operators.laplacian` (K) and
    `Geometry.operators.laplacian` (cotan) — a cross-surface identity.
  - `ambient` differs from `product` (non-zero coupling) and is symmetric PSD.
  - trivial vs levi-civita tangent gives different `L3` (tangent block differs).

## 7. Decisions log

| # | Decision | Resolution |
|---|---|---|
| 1 | acts-on level | 3D frame vectors (tangent 2 + normal 1) |
| 2 | representation | 3N×3N real symmetric sparse, frame coords, block `[a;b;c]=[Re z;Im z;normal]` |
| 3 | tangent connection | follows gauge type (LC / trivial), parallel to `.laplacian` |
| 4 | coupling | `product` (blkdiag) or `ambient` (shape-operator); default `ambient` |
| 5 | ambient transport | tangent-gauge rotation ∘ normal-aligning rotation; gauge-independent normal part |
| 6 | schema home | `Gauge.operators.laplacian3D`, opt-in via `operators=true`; `coupling` opt |
| 7 | validation | sphere `S=κI`, flat patch `S=0`; product == blkdiag identity |

## 8. Deferred

- World-coordinate form and the structured (complex-tangent ⊕ real-normal) return
  are not built; the single real frame-coord matrix is the v1 surface.
- Higher `nSym` 3D variants (line/cross 3D fields) — out of scope.
