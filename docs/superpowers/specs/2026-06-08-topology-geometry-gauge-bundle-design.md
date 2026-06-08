# Topology / Geometry / Gauge Bundle — Cortical Coordinate System for Brainstorm

**Date:** 2026-06-08
**Status:** Design — approved, pending spec review before implementation plan
**Consumer:** Brainstorm (MATLAB) via the MEX binding, for MEG source mapping

---

## 1. Goal

Give Brainstorm a halfedge-mesh coordinate system over the cortical surface,
supplied by nxr-compute as three nested MATLAB structs — `Topology`,
`Geometry`, `Gauge` — that mirror geometry-central's combinatorics +
geometry + frame-field concepts.

The concrete near-term driver is the **forward / leadfield** stage of MEG
source mapping: establish a clean, lossless, invertible correspondence
between the Cartesian leadfield (unconstrained, 3 components per source) and
its **intrinsic cortical readout** in a surface-adapted basis. The same
coordinate system is the foundation for later inverse work (where a typed
`MeshData` container, out of scope here, will be built on top in Brainstorm).

Non-goals (explicitly deferred):
- **Operators** (`cotanLaplacian`, mass, DEC, connection Laplacian): a future
  `Operators` struct with lazily-populated subfields by type. Not designed here.
- **MeshData** typed containers + encode/decode helpers: Brainstorm-side, later.
- Inverse modelling.

---

## 2. Architecture

### 2.1 Hybrid delivery, two tiers by materialization cost

```
                 ┌──────────────── nxr-compute (C++/MEX) ────────────────┐
  TessMat.V,F ─► │ create(V,F) ─► live handle (Manifold + session caches)│
                 │     ├─ 'topology'  ─► Topology struct  (light)         │
                 │     ├─ 'geometry'  ─► Geometry struct  (light)         │
                 │     ├─ 'gauge',t   ─► Gauge struct      (light)         │
                 │     └─ 'bundle',t  ─► {Topology,Geometry,Gauge}         │
                 │     (operators: future 'operators' surface, on-demand)  │
                 └───────────────────────────────────────────────────────┘
                        │ persist light structs into TessMat .mat
                        ▼
                  Topology / Geometry / Gauge   (survive save/load, no handle)
```

- **Persistent tier** (saved into the TessMat `.mat`): `Topology`, `Geometry`,
  `Gauge`. Pure arrays — survive save/load, readable with no live handle.
- **On-demand tier** (handle-only, session-cached, never serialized): all
  heavy sparse operators. Out of scope for this spec; reconstructed from
  `V,F` by re-minting a handle when needed. Operator *assembly* is the cheap
  part; the costly Cholesky factor was never serializable anyway.

### 2.2 Three-layer separation of concerns

| Layer | Holds | Nature |
|---|---|---|
| `Topology` | halfedge table + reverse lookups + counts | pure combinatorics |
| `Geometry` | one intrinsic complex frame (`grid`) + metric + curvature + transport | geometry of the embedding |
| `Gauge` | a (mostly parameter-free) **transform** of the Geometry frame | choice of frame field |

The only intrinsic frame data on the surface is the canonical Levi-Civita
`grid` in `Geometry`. Every gauge is a recipe for transforming it; only the
`trivial` gauge carries per-vertex data (the connection rotation + singularities).

---

## 3. Index-base contract (single enforced boundary)

- **C++ / geometry-central is 0-based everywhere** internally. No exceptions.
- **The MEX shell is the sole conversion layer.** Everything crossing into
  MATLAB is 1-based (index arrays) or carries no exposed index at all.
- **Index *arguments* from MATLAB are 1-based and decremented on entry**
  (the existing `mxToVertexIndices` already does this — it becomes the
  universal rule).
- **One helper pair** (`idxToMatlab` / `idxFromMatlab`) performs every
  conversion, with a debug-build range assertion (`1 ≤ i ≤ N`, or `i == 0`
  sentinel for "none"). No ad-hoc `+1` / `-1` anywhere else.
- **Future operators cross as native MATLAB sparse** (`mxCreateSparse`,
  complex sparse for the connection Laplacian) — no exposed `ir/jc` index
  base. The JS-style COO `{row,col,data}` convention is *not* used at the MEX
  boundary (MEX is already §11-exempt). [Applies when the Operators surface
  is built.]

Net effect: a MATLAB consumer never sees a 0-based index. The only 0-based
code lives behind the MEX wall.

---

## 4. Struct layouts

All index fields are **1-based**, with `0` as the "none / boundary" sentinel.
Each persistent struct carries a `schemaVersion` scalar so the layout can
evolve without silent breakage.

Field-naming rule: **field name = the geometry-central cache name with the
element-type prefix stripped** (the prefix now lives in the path). E.g.
`vertexDualAreas` → `vertex.dualAreas`, `edgeLengths` → `edge.lengths`. This
keeps each field traceable to exactly one geometry-central `require*` call.

### 4.1 `Topology` — `nxr_compute('topology', h)`

```
Topology.schemaVersion                            scalar
Topology.vertex.count                             scalar
Topology.vertex.halfedge          V×1 uint32      % one outgoing he per vertex
Topology.edge.count                               scalar
Topology.edge.halfedge            E×1 uint32      % canonical he per edge
Topology.face.count                               scalar
Topology.face.halfedge            F×1 uint32      % one he per face
Topology.corner.count                             scalar
Topology.corner.halfedge          nCorners×1 uint32
Topology.halfedge.count                           scalar
Topology.halfedge.twin            H×1 uint32      % he.twin()
Topology.halfedge.next            H×1 uint32      % he.next()
Topology.halfedge.vertex          H×1 uint32      % he.vertex()  (tail)
Topology.halfedge.edge            H×1 uint32      % he.edge()
Topology.halfedge.face            H×1 uint32      % he.face()    (0 = exterior)
Topology.halfedge.corner          H×1 uint32      % he.corner()  (0 = boundary he)
Topology.halfedge.orientation     H×1 logical     % he == he.edge().halfedge()
Topology.halfedge.isInterior      H×1 logical     % ~he.isExterior()
```

`nCorners == 3·nFaces` for a triangle mesh. All `*.halfedge` / `halfedge.*`
index fields are 1-based; `0` marks a missing twin/face/corner at a boundary.

### 4.2 `Geometry` — `nxr_compute('geometry', h)`

Light per-element fields only. Flat-within-element-group; element type in the path.

```
Geometry.schemaVersion                            scalar
Geometry.totalArea                                scalar          % Σ faceAreas (global)

% ── vertex ──────────────────────────────────────────────────────────────
Geometry.vertex.dualAreas         V×1 double      % geom.vertexDualAreas (lumped barycentric A/3)
Geometry.vertex.angleSums         V×1 double      % geom.vertexAngleSums; intrinsic Gaussian = 2π − angleSum
Geometry.vertex.curvature         V×1 complex     % deviatoric 2-RoSy shape operator (see §5.2)
Geometry.vertex.meanCurvature     V×1 double      % trace/2 of shape operator (see §5.2)
Geometry.vertex.grid              V×3 complex     % c = e1 + i·e2 (canonical Levi-Civita frame; see §5.1)

% ── edge ────────────────────────────────────────────────────────────────
Geometry.edge.lengths             E×1 double      % geom.edgeLengths
Geometry.edge.cotanWeights        E×1 double      % geom.edgeCotanWeights
Geometry.edge.dihedralAngles      E×1 double      % geom.edgeDihedralAngles

% ── face ────────────────────────────────────────────────────────────────
Geometry.face.areas               F×1 double      % geom.faceAreas
Geometry.face.centroids           F×3 double      % mean of face vertex positions
Geometry.face.grid                F×3 complex     % c = e1 + i·e2 (face tangent frame)

% ── halfedge ────────────────────────────────────────────────────────────
Geometry.halfedge.cotanWeights    H×1 double      % geom.halfedgeCotanWeights
Geometry.halfedge.vectorsInVertex H×1 complex     % geom.halfedgeVectorsInVertex
Geometry.halfedge.vectorsInFace   H×1 complex     % geom.halfedgeVectorsInFace
Geometry.halfedge.transportAlong  H×1 complex     % geom.transportVectorsAlongHalfedge
Geometry.halfedge.transportAcross H×1 complex     % geom.transportVectorsAcrossHalfedge

% ── corner ──────────────────────────────────────────────────────────────
Geometry.corner.angles            nCorners×1 double % geom.cornerAngles
Geometry.corner.scaledAngles      nCorners×1 double % geom.cornerScaledAngles
```

Deliberately **not** stored (all derivable — see §5):
`vertex.normals`, `face.normals` (derive from `grid`); `tangentBasis`
(replaced by `grid`); `min/maxPrincipalCurvatures`, `gaussianCurvatures`,
`principalDirections` (derive from `curvature` + `meanCurvature` + `angleSums`).
`coordinateSystem` is Brainstorm metadata (nxr only sees raw `V,F`), so it is
caller-owned, not in this struct.

### 4.3 `Gauge` — `nxr_compute('gauge', h, type, opts)`

```
Gauge.schemaVersion                               scalar
Gauge.type                        char            % 'euclidean' | 'levi-civita' | 'trivial'
Gauge.vertex.rotation             V×1 complex     % exp(iθ_v) relative to Levi-Civita grid; 'trivial' only
Gauge.face.rotation               F×1 complex     % 'trivial' only
Gauge.singularity.vertices        S×1 uint32      % 1-based; 'trivial' only
Gauge.singularity.indices         S×1 double      % Σ indices must equal χ (Gauss-Bonnet)
Gauge.singularity.source          char            % 'freesurfer-sphere' | 'optimal' | 'manual'
```

For `euclidean` and `levi-civita`, the per-vertex fields are empty — `type`
alone determines the frame (see §5.3). Only `trivial` carries data.

`opts` (for `type == 'trivial'`): `singVerts` (1-based), `singValues`
(Σ = χ; defaults to two sphere-pole singularities of index 1 for the
genus-0 cortical hemisphere). Brainstorm chooses placement, e.g. from
`TessMat.Reg` sphere coordinates.

---

## 5. The complex-frame framework

### 5.1 `grid`: the frame as a complex 3-vector

At each element define `c = e1 + i·e2 ∈ ℂ³`, where `(e1, e2)` is the
orthonormal tangent basis (geometry-central `vertexTangentBasis` /
`faceTangentBasis`). Then:

- `real(c) = e1`, `imag(c) = e2`.
- The unit normal is recovered (not stored): `n = real(c) × imag(c)`. Because
  geometry-central's tangent basis is right-handed with its own normal,
  this equals `vertexNormals` exactly — a free validation invariant.

The frame is therefore one V×3 complex array; nothing is lost by dropping a
separate `normals` field.

**Lossless vector transform** (Cartesian `J ∈ ℝ³` ↔ intrinsic):
```matlab
n   = cross(real(c), imag(c), 2);     % unit normal (derived, not stored)
z   = sum(c .* J, 2);                 % complex tangential coordinate = (e1·J) + i(e2·J)
j_n = sum(n .* J, 2);                 % real normal coordinate
% inverse (exact, because c is orthonormal):
J   = real(conj(z) .* c) + j_n .* n;  % = a·e1 + b·e2 + j_n·n
```
The inverse identity holds because
`real(conj(z)·c) = real((a−ib)(e1+i·e2)) = a·e1 + b·e2` (the `−ib·e1` and
`+ia·e2` terms are imaginary and drop). The map is a pure rotation — never a
projection — so the intrinsic readout carries the same information as the
Cartesian input.

### 5.2 `curvature`: 2-RoSy complex + mean

The shape operator (second fundamental form) is a symmetric 2-tensor with 3
real DOF. It is **not** representable as a complex 3-vector (that would be 6
DOF, over-parameterized). The complex-native, gauge-covariant encoding splits
trace from deviatoric:

- `Geometry.vertex.meanCurvature` (real) = `trace/2 = H`.
- `Geometry.vertex.curvature` (complex `q`) = the traceless symmetric part as a
  **2-RoSy** number: `arg(q)/2` is the principal direction, `|q| = (κmax−κmin)/2`.

Under a tangent rotation `exp(iθ)`, the grid transforms as `exp(iθ)` and the
curvature as `q·exp(2iθ)` — curvature is the square-symmetry partner of the
frame. This matches geometry-central's native principal-direction output.

Source: build from geometry-central `vertexMin/MaxPrincipalCurvatures` +
`vertexPrincipalCurvatureDirections` (`H = (κmax+κmin)/2`,
`|q| = (κmax−κmin)/2`, `arg(q) = ` principal-direction 2-RoSy angle), or
directly from the per-vertex shape operator if assembled.

**Derivations** (everything the dropped scalar fields used to hold):
```matlab
q = Geometry.vertex.curvature(v);  H = Geometry.vertex.meanCurvature(v);
c = Geometry.vertex.grid(v,:);
kappa  = H + [abs(q), -abs(q)];                     % κmax, κmin
phi    = angle(q)/2;                                % principal direction angle (tangent)
dir3D  = real( exp(-1i*phi) .* c );                 % max principal direction lifted to ℝ³ via the grid
K_ext  = H^2 - abs(q)^2;                            % extrinsic Gaussian (κmax·κmin)
K_int  = 2*pi - Geometry.vertex.angleSums(v);       % intrinsic Gaussian (exact Gauss-Bonnet)
```

### 5.3 Gauge realization (the "rotation logic")

The Gauge stores a transform, not a frame. The active frame is realized from
the canonical `grid`:

| `Gauge.type` | realized frame `c_gauge` | stored data |
|---|---|---|
| `euclidean`   | `[1, i, 0]` (world axes, broadcast) | none |
| `levi-civita` | `Geometry.vertex.grid`             | none |
| `trivial`     | `Gauge.vertex.rotation .* Geometry.vertex.grid` | `rotation` + singularities |

```matlab
function c = realizeGauge(grid, Gauge)
  switch Gauge.type
    case 'euclidean',   c = repmat([1, 1i, 0], size(grid,1), 1);
    case 'levi-civita', c = grid;
    case 'trivial',     c = Gauge.vertex.rotation .* grid;
  end
end
% coordinate change between surface gauges (LC ↔ trivial) is a pure phase:
z_trivial = Gauge.vertex.rotation .* z_levicivita;
```

`euclidean` is the one gauge that is *not* a tangent-plane rotation of
Levi-Civita (its tangent plane is the world xy-plane, not the surface tangent
plane); `type` distinguishes it. The Levi-Civita→trivial rotation is the
integrated trivial-connection 1-form (§6).

---

## 6. The `trivial` gauge: per-vertex connection integration

`euclidean` and `levi-civita` are parameter-free. `trivial` is the only gauge
that runs a solve:

1. Compute the trivial-connection 1-form `φ` (per edge) via the existing
   `nxr::manifold::connection::computeTrivialConnection(m, dec, cache, singMap)`
   — the prescribed-singularity Poisson solve already in the codebase.
2. **New work:** integrate `φ` to a per-vertex rotation angle `θ_v`. The
   existing `propagateAngles` is **face**-based (for direction fields); the
   gauge needs a **vertex**-side BFS over the vertex graph, accumulating
   `transportVectorsAlongHalfedge · exp(i·sign·φ[edge])` from a root vertex.
   Store `Gauge.vertex.rotation = exp(iθ_v)` (and the analogous
   `face.rotation`).
3. Singularities are caller-supplied (`opts.singVerts/singValues`), defaulting
   to two index-1 sphere-pole singularities for the genus-0 hemisphere
   (`Σ = χ = 2`).

This is the principal new algorithm introduced by this spec.

---

## 7. Leadfield correspondence (the near-term deliverable)

The unconstrained Cartesian leadfield gain block for source vertex `v` is
`G_v` (`nSensors × 3`). Its intrinsic readout is obtained by the same complex
transform applied to each gain row:

```matlab
c   = realizeGauge(Geometry.vertex.grid, Gauge);   % choose gauge
n   = cross(real(c), imag(c), 2);
Ltan = G_v * c(v,:).';     % nSensors×1 COMPLEX — intrinsic tangential leadfield
Ln   = G_v * n(v,:).';     % nSensors×1 real    — radial leadfield
% intrinsic forward model for an intrinsic dipole (z tangential, j_n normal):
%   measurement = real(conj(z) .* Ltan) + j_n .* Ln
```

`Ltan = G·cᵀ` is the intrinsic cortical readout of the leadfield: magnitude is
tangential gain strength, phase is the tangential dipole orientation that
produces it. The correspondence is exact and invertible (orthonormal frame),
never projective.

---

## 8. MEX command surface

```matlab
h   = nxr_compute('create', V, F);          % mint handle (existing)
T   = nxr_compute('topology', h);           % → persist into TessMat
Geo = nxr_compute('geometry', h);           % → persist
Ga  = nxr_compute('gauge', h, type, opts);  % type ∈ {'euclidean','levi-civita','trivial'} → persist
B   = nxr_compute('bundle', h, type);       % returns struct B.Topology/.Geometry/.Gauge in one call
nxr_compute('destroy', h);
```

Typical Brainstorm flow: `create` → `bundle` → store the three structs into
TessMat → `destroy`. Re-mint a handle later only when on-demand operators are
needed (future Operators surface).

Validation at the boundary: range-assert all emitted indices; assert each
`grid` row is a valid frame (`|real(c)| ≈ |imag(c)| ≈ 1`, `real(c)·imag(c) ≈ 0`)
in debug builds.

---

## 9. Decisions log

| # | Decision | Resolution |
|---|---|---|
| 1 | Delivery model | Hybrid — persistent light structs in TessMat + handle for heavy compute |
| 2 | Frame encoding | Complex 3-vector `grid` `c = e1+i·e2`; normal derived as `real×imag` |
| 3 | Heavy operators | On-demand only, session-cached, native MATLAB sparse, never serialized (deferred to Operators surface) |
| 4 | Scope | nxr supplies raw `Topology/Geometry/Gauge`; MeshData/encode-decode is Brainstorm-side, later |
| 5 | Frame normal | geometry-central `vertexNormals`; gauge is always a lossless rotation, never projection |
| 6 | Index base | C++ 0-based; MEX is sole conversion wall; MATLAB sees 1-based arrays / native sparse; single helper pair + debug range asserts |
| 7 | Operators in Geometry | Removed — split to the (future) on-demand surface |
| 8 | Topology indices | 1-based, `0` = none sentinel |
| 9 | Gauge types v1 | `euclidean` + `levi-civita` + `trivial` (new per-vertex angle integration) |
| 10 | Gauge representation | Transform, not frame — `type` + (`trivial` only) `rotation` + singularities |
| 11 | Geometry grouping | Element-type sub-structs (`vertex`/`edge`/`face`/`halfedge`/`corner`) |
| 12 | Curvature representation | 2-RoSy complex `curvature` (deviatoric) + real `meanCurvature` (option A); drop precomputed min/max/Gaussian/principalDir |
| 13 | Schema versioning | `schemaVersion` scalar on each persistent struct |

---

## 10. Deferred / future work

- **Operators struct** — lazily-populated subfields by type (Laplacians, mass,
  DEC, connection Laplacian), returned as native MATLAB sparse, session-cached.
  Designed when the solve stage begins.
- **MeshData** typed containers + dipole/leadfield encode-decode helpers
  (Brainstorm-side), for the inverse stage.
- **Boundary handling** — the cortical hemisphere is treated as closed genus-0;
  revisit sentinel/`0` semantics if cut/boundary surfaces are introduced.

---

## 11. Open items for spec review

1. Curvature **option A** (2-RoSy complex + mean real) is locked in; confirm
   no consumer needs a literal V×2×2 shape-operator matrix instead.
2. `grid` naming — confirm it does not collide with Brainstorm's source-grid
   (`GridLoc`/`GridOrient`) terminology in TessMat; `frame`/`basis` are
   alternatives.
3. `face.rotation` for the `trivial` gauge — keep (symmetry with `face.grid`)
   or drop until a face-based consumer needs it.
