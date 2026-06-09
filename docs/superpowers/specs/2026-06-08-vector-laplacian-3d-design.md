# 3D Vector (Covariant) Laplacian — `Gauge.operators.covariantLaplacian`

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
| **3D frame vectors** | **3 (`a,b` tangent + `c` normal)** | **`Gauge.operators.covariantLaplacian`** ← this spec |

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

Both variants assemble exactly from quantities we already have; neither needs a
discretization choice. Frame: `F_v = [e1_v, e2_v, n_v] ∈ SO(3)` (the gauge 3-frame
— grid tangent axes + cross-product normal). `L_cotan` = scalar cotan Laplacian;
`K` = the gauge connection Laplacian (V×V complex; Levi-Civita or trivial).

**ambient** — the covariant Laplacian of the flat ℝΒ³ (embedding) connection. Its
*world-coordinate* form is the scalar Laplacian on each Cartesian component,
`kron(I₃, L_cotan)`. Expressed in the moving gauge frame, the per-entry 3×3 block is
```
    L3_ambient[i,j]  =  L_cotan[i,j] · (F_iᵀ F_j)
```
so `L3_ambient = blockdiag(F)ᵀ · kron(I₃, L_cotan) · blockdiag(F)`. The
tangent↔normal coupling falls out of the relative frame rotation `F_iᵀF_j` exactly
— no shape-operator discretization. Symmetric PSD by construction (orthogonal
conjugation of a PSD matrix). The gauge only changes coordinates: the spectrum is
gauge-invariant. *Continuum note:* the tangent `(a,b)` sub-block is the
**extrinsic** frame-transport Laplacian `L_cotan[i,j]·(F_iᵀF_j)|_{tangent}`, which
converges to the intrinsic connection Laplacian `K` under refinement but is **not**
discretely equal to geometry-central's `K` (whose transport is angle-based, not the
3D frame dot-product) — on a coarse curved mesh they differ. So the ambient tangent
block is verified by self-consistency with the ambient formula, not by equality to `K`.

**product** — the Laplacian of the product connection `∇^{TM} ⊕ ∇^{NM}`, coupling
removed:
```
    L3_product  =  blkdiag( [[Re K, −Im K],[Im K, Re K]] , L_cotan )
```
Block-diagonal; tangent and normal decoupled. Gauge-dependent (`K` differs LC vs
trivial). `ambient − product` differs in the `(a,c)`/`(b,c)` coupling blocks, the
normal-block curvature reweighting (`L_ij·(n_iᵀn_j)` vs `L_ij`), and — discretely —
the tangent block itself (extrinsic frame transport in `ambient` vs intrinsic `K` in
`product`). All these vanish on a flat patch (where `F_iᵀF_j = I` and the two
transports coincide) and grow with curvature.

Both returned in frame coords, block layout `[a; b; c] = [Re z; Im z; normal]`.

## 5. API / schema

```
Gauge.operators.covariantLaplacian    3N×3N real sparse   % present when operators=true
```
Selected by the gauge type (tangent connection) + a `coupling` option. Surfaced
on the `Gauge.operators` sub-struct alongside `.laplacian`. The `operators` opt-in
flag gains a companion for the coupling choice:

```matlab
% ambient (default) covariant 3D Laplacian, trivial tangent connection:
B = nxr_compute('bundle', h, 'trivial', ...
      struct('singVerts',sv,'singValues',si,'operators',true));
L3 = B.Gauge.operators.covariantLaplacian;          % 3N×3N real sparse, [a;b;c]

% product variant:
G = nxr_compute('gauge', h, 'levi-civita', ...
      struct('operators',true,'coupling','product'));
```
`coupling` is read from the same opts struct as `operators` (default `'ambient'`).

**Library:** a new `nxr::manifold::ops::laplacian::connection` entry,
`assembleCovariantLaplacian(Manifold&, gaugeType, coupling, singMap?)`, returning a
real `Eigen::SparseMatrix<double>` of size `3N×3N`. Caching: keyed on
`(gaugeType, coupling)` on the ContextHolder (new slot or extend the operator
cache).

## 6. Test plan

- **Native** (`test/test_geometry_bundle.cpp`): on the icosphere,
  - `L3` is 3N×3N, symmetric (`||L3 − L3ᵀ|| < 1e-9`), PSD (min eig ≥ −1e-9), for
    both couplings and both tangent connections.
  - **ambient world-coords identity (exact):** `blockdiag(F) · L3_ambient ·
    blockdiag(F)ᵀ == kron(I₃, L_cotan)` to machine precision (the defining anchor).
  - **product blkdiag identity (exact):** `L3_product == blkdiag(real-expand(K),
    L_cotan)`.
  - **ambient tangent self-consistency (exact):** the `(a,b)` 2N sub-block of
    `L3_ambient` equals `L_cotan[i,j]·(F_iᵀF_j)|_{tangent}` (the ambient formula
    restricted to tangent) — *not* `real-expand(K)`, which differs discretely
    (extrinsic frame transport vs geometry-central's intrinsic angle-based `K`).
  - `ambient ≠ product` on the curved icosphere (non-zero coupling); they coincide
    in the flat limit.
- **MATLAB** (`bindings/mex/test/test_operators.m`):
  - `Gauge.operators.covariantLaplacian` is `3N×3N` real sparse, symmetric.
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
| 5 | ambient assembly | `L3[i,j] = L_cotan[i,j]·(F_iᵀF_j)` = frame-conjugate of `kron(I₃,L_cotan)`; coupling exact (no discretization) |
| 6 | schema home | `Gauge.operators.covariantLaplacian`, opt-in via `operators=true`; `coupling` opt |
| 7 | validation | ambient world-form `== kron(I₃,L_cotan)` (exact); product `== blkdiag(real-expand(K),L_cotan)` (exact); ambient tangent block self-consistent with the formula (extrinsic transport, ≠ intrinsic `K` discretely) |

## 8. Deferred

- World-coordinate form and the structured (complex-tangent ⊕ real-normal) return
  are not built; the single real frame-coord matrix is the v1 surface.
- Higher `nSym` 3D variants (line/cross 3D fields) — out of scope.
