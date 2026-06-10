# nxr Facet Data Commands (embedded / intrinsic / extrinsic / facets) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MATLAB MEX commands that expose nxr-compute's representation facets as data — standalone `embedded` / `intrinsic` / `extrinsic`, plus a grouped `facets` command returning `{Topology, Embedded, Intrinsic, Extrinsic, Gauge}` — so Brainstorm can consume the per-element frames and geometry/topology quantities.

**Architecture:** Three new struct builders (`buildEmbeddedStruct` / `buildIntrinsicStruct` / `buildExtrinsicStruct`) marshal the already-cached typed facet views (`Manifold::embedded()/intrinsic()/extrinsic()` from `include/nxr/facets.h`) into MATLAB structs using the existing converters. Four new dispatch commands wire them up. The existing flat `geometry`/`bundle` commands are untouched (back-compat). Data-only — no `.operators`.

**Tech Stack:** C++17 (geometry-central, Eigen, MATLAB MEX C API), MATLAB script-style tests in `bindings/mex/test/`, CMake build via `scripts/build.sh`.

**Repo:** `~/workspace/research/code/nxr-compute` (branch `main`).

**Reference patterns (read before starting):**
- `bindings/mex/src/nxr_compute_mex.cpp` — `buildGeometryStruct` (line ~1379), `buildGaugeStruct` (~1274), `buildTopologyStruct` (~1466), `cmdBundle` (~1529), the dispatch block (~1650), and the converters `scalarToMx` / `eigenVectorToMx` / `eigenComplexVectorToMx` / `eigenMatrixToMx` / `eigenComplexMatrixToMx`.
- `include/nxr/facets.h` + `src/facets.cpp` — the typed facet views and their return types.
- `bindings/mex/test/test_bundle.m` — test conventions (find built MEX via `dir(build/**)`, `clear nxr_compute`, `local_icosahedron`, `assert`, final `ALL TESTS PASSED` line).

**Build/run discipline (all tasks):**
- Build: from repo root, `bash scripts/build.sh Release` → emits `build/Release/nxr_compute.mexmaca64`.
- Run a test (MATLAB): `cd bindings/mex/test; clear nxr_compute; test_facets` (the test re-`addpath`s the freshly built MEX and `clear`s the function so the new binary loads). Use only `clear nxr_compute` — never a bare `clear`.

---

## File Structure

| File | Responsibility |
|---|---|
| `bindings/mex/src/nxr_compute_mex.cpp` | **Modify.** Add 3 builders + 4 `cmd*` functions; register in dispatch + help string. |
| `bindings/mex/test/test_facets.m` | **Create.** Property tests for the standalone + grouped facet commands. |

---

## Task 1: Facet struct builders + standalone commands

**Files:**
- Create: `bindings/mex/test/test_facets.m`
- Modify: `bindings/mex/src/nxr_compute_mex.cpp`

- [ ] **Step 1: Write the failing test**

Create `bindings/mex/test/test_facets.m`:

```matlab
function test_facets
fprintf('[test_facets] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found', mexext);
addpath(hits(1).folder); clear nxr_compute

[V, F] = local_icosahedron();
nV = size(V,1); nF = size(F,1);
h = nxr_compute('create', V, F);

% ── embedded ──
E = nxr_compute('embedded', h);
assert(isfield(E,'vertex') && isfield(E,'face'), 'embedded has vertex+face');
assert(isequal(size(E.vertex.position), [nV 3]), 'embedded vertex.position nV x3');
assert(max(abs(E.vertex.position(:) - V(:))) < 1e-9, 'embedded position == input V');
e1 = real(E.vertex.grid); e2 = imag(E.vertex.grid);
assert(max(abs(sqrt(sum(e1.^2,2))-1)) < 1e-6, 'e1 unit');
assert(max(abs(sqrt(sum(e2.^2,2))-1)) < 1e-6, 'e2 unit');
assert(max(abs(sum(e1.*e2,2))) < 1e-6, 'e1 perp e2');
assert(max(abs(E.vertex.normal - cross(e1,e2,2)),[],'all') < 1e-6, 'normal == e1 x e2');
assert(isequal(size(E.face.grid),[nF 3]), 'face grid nF x3');

% ── intrinsic ──
I = nxr_compute('intrinsic', h);
assert(isfield(I,'vertex')&&isfield(I,'edge')&&isfield(I,'halfedge'), 'intrinsic groups');
assert(isequal(size(I.vertex.dualArea),[nV 1]), 'dualArea nV x1');
assert(all(I.vertex.dualArea > 0), 'dual areas positive');
assert(~isreal(I.halfedge.transportAlong), 'transportAlong complex');

% ── extrinsic (carries what flat geometry drops) ──
X = nxr_compute('extrinsic', h);
assert(isfield(X.vertex,'principalDir'), 'extrinsic has principalDir');
assert(isequal(size(X.vertex.principalDir),[nV 3]), 'principalDir nV x3');
assert(~isreal(X.vertex.curvature2RoSy), 'curvature2RoSy complex');

nxr_compute('destroy', h);
fprintf('ALL TESTS PASSED: test_facets (standalone)\n');
end

function [V, F] = local_icosahedron()
t = (1 + sqrt(5)) / 2;
V = [-1  t  0;  1  t  0; -1 -t  0;  1 -t  0; ...
      0 -1  t;  0  1  t;  0 -1 -t;  0  1 -t; ...
      t  0 -1;  t  0  1; -t  0 -1; -t  0  1];
V = V ./ sqrt(sum(V.^2, 2));
F = [1 12 6; 1 6 2; 1 2 8; 1 8 11; 1 11 12; ...
     2 6 10; 6 12 5; 12 11 3; 11 8 7; 8 2 9; ...
     4 10 5; 4 5 3; 4 3 7; 4 7 9; 4 9 10; ...
     5 10 6; 3 5 12; 7 3 11; 9 7 8; 10 9 2];
end
```

- [ ] **Step 2: Run the test against the current build to verify it fails**

```matlab
cd bindings/mex/test; clear nxr_compute; test_facets
```
Expected: ERROR `Unknown command: "embedded"` (the command isn't registered yet).

- [ ] **Step 3: Add `buildEmbeddedStruct`**

In `bindings/mex/src/nxr_compute_mex.cpp`, immediately **after** `buildGeometryStruct` (ends ~line 1432), add:

```cpp
// ── facet builders (representation-grouped data; mirror buildGeometryStruct) ──

mxArray* buildEmbeddedStruct(ContextHolder& h) {
    nxr::manifold::Manifold& m = *h.ctx;
    auto emb = m.embedded();
    auto vv = emb.vertex();
    auto fv = emb.face();

    const char* topF[] = {"schemaVersion","vertex","face"};
    mxArray* s = mxCreateStructMatrix(1,1,3,topF);
    mxSetField(s,0,"schemaVersion",scalarToMx(1));

    { const char* f[] = {"position","normal","grid"};
      mxArray* g = mxCreateStructMatrix(1,1,3,f);
      mxSetField(g,0,"position", eigenMatrixToMx(vv.position()));
      mxSetField(g,0,"normal",   eigenMatrixToMx(vv.normal()));
      mxSetField(g,0,"grid",     eigenComplexMatrixToMx(vv.grid()));
      mxSetField(s,0,"vertex",g); }
    { const char* f[] = {"normal","grid","centroid"};
      mxArray* g = mxCreateStructMatrix(1,1,3,f);
      mxSetField(g,0,"normal",   eigenMatrixToMx(fv.normal()));
      mxSetField(g,0,"grid",     eigenComplexMatrixToMx(fv.grid()));
      mxSetField(g,0,"centroid", eigenMatrixToMx(fv.centroid()));
      mxSetField(s,0,"face",g); }

    return s;
}
```

- [ ] **Step 4: Add `buildIntrinsicStruct`** (immediately after `buildEmbeddedStruct`)

```cpp
mxArray* buildIntrinsicStruct(ContextHolder& h) {
    nxr::manifold::Manifold& m = *h.ctx;
    auto intr = m.intrinsic();
    auto vv = intr.vertex();
    auto ev = intr.edge();
    auto hv = intr.halfedge();

    const char* topF[] = {"schemaVersion","vertex","edge","halfedge"};
    mxArray* s = mxCreateStructMatrix(1,1,4,topF);
    mxSetField(s,0,"schemaVersion",scalarToMx(1));

    { const char* f[] = {"dualArea","angleSum"};
      mxArray* g = mxCreateStructMatrix(1,1,2,f);
      mxSetField(g,0,"dualArea", eigenVectorToMx(vv.dualArea()));
      mxSetField(g,0,"angleSum", eigenVectorToMx(vv.angleSum()));
      mxSetField(s,0,"vertex",g); }
    { const char* f[] = {"length","cotanWeight"};
      mxArray* g = mxCreateStructMatrix(1,1,2,f);
      mxSetField(g,0,"length",      eigenVectorToMx(ev.length()));
      mxSetField(g,0,"cotanWeight", eigenVectorToMx(ev.cotanWeight()));
      mxSetField(s,0,"edge",g); }
    { const char* f[] = {"cotanWeight","transportAlong","transportAcross"};
      mxArray* g = mxCreateStructMatrix(1,1,3,f);
      mxSetField(g,0,"cotanWeight",     eigenVectorToMx(hv.cotanWeight()));
      mxSetField(g,0,"transportAlong",  eigenComplexVectorToMx(hv.transportAlong()));
      mxSetField(g,0,"transportAcross", eigenComplexVectorToMx(hv.transportAcross()));
      mxSetField(s,0,"halfedge",g); }

    return s;
}
```

- [ ] **Step 5: Add `buildExtrinsicStruct`** (immediately after `buildIntrinsicStruct`)

```cpp
mxArray* buildExtrinsicStruct(ContextHolder& h) {
    nxr::manifold::Manifold& m = *h.ctx;
    auto ext = m.extrinsic();
    auto vv = ext.vertex();
    auto ev = ext.edge();

    const char* topF[] = {"schemaVersion","vertex","edge"};
    mxArray* s = mxCreateStructMatrix(1,1,3,topF);
    mxSetField(s,0,"schemaVersion",scalarToMx(1));

    { const char* f[] = {"curvature2RoSy","meanCurvature","principalDir"};
      mxArray* g = mxCreateStructMatrix(1,1,3,f);
      mxSetField(g,0,"curvature2RoSy", eigenComplexVectorToMx(vv.curvature2RoSy()));
      mxSetField(g,0,"meanCurvature",  eigenVectorToMx(vv.meanCurvature()));
      mxSetField(g,0,"principalDir",   eigenMatrixToMx(vv.principalDir()));
      mxSetField(s,0,"vertex",g); }
    { const char* f[] = {"dihedralAngle"};
      mxArray* g = mxCreateStructMatrix(1,1,1,f);
      mxSetField(g,0,"dihedralAngle", eigenVectorToMx(ev.dihedralAngle()));
      mxSetField(s,0,"edge",g); }

    return s;
}
```

- [ ] **Step 6: Add the three `cmd*` functions** (immediately after `cmdBundle`, ~line 1543)

```cpp
void cmdEmbedded(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 2) throw std::invalid_argument(
        "nxr_compute('embedded', handle) takes 1 argument");
    ContextHolder& h = getHolder(prhs[1]);
    plhs[0] = buildEmbeddedStruct(h);
}

void cmdIntrinsic(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 2) throw std::invalid_argument(
        "nxr_compute('intrinsic', handle) takes 1 argument");
    ContextHolder& h = getHolder(prhs[1]);
    plhs[0] = buildIntrinsicStruct(h);
}

void cmdExtrinsic(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 2) throw std::invalid_argument(
        "nxr_compute('extrinsic', handle) takes 1 argument");
    ContextHolder& h = getHolder(prhs[1]);
    plhs[0] = buildExtrinsicStruct(h);
}
```

- [ ] **Step 7: Register the three commands in the dispatch + help string**

In the dispatch chain (after the `else if (cmd == "operators") …` line, ~1692), add:

```cpp
        else if (cmd == "embedded")   cmdEmbedded(nlhs, plhs, nrhs, prhs);
        else if (cmd == "intrinsic")  cmdIntrinsic(nlhs, plhs, nrhs, prhs);
        else if (cmd == "extrinsic")  cmdExtrinsic(nlhs, plhs, nrhs, prhs);
```

In the "Unknown command … Available:" `mexErrMsgIdAndTxt` string (~1697), append before `version`:
`"topology, geometry, gauge, bundle, operators, embedded, intrinsic, extrinsic, version."`

- [ ] **Step 8: Build and run to verify the standalone tests pass**

```bash
bash scripts/build.sh Release
```
Then in MATLAB:
```matlab
cd bindings/mex/test; clear nxr_compute; test_facets
```
Expected: `ALL TESTS PASSED: test_facets (standalone)`.

- [ ] **Step 9: Commit**

```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_facets.m
git commit -m "feat(mex): embedded/intrinsic/extrinsic facet data commands"
```

---

## Task 2: Grouped `facets` command

**Files:**
- Modify: `bindings/mex/src/nxr_compute_mex.cpp`
- Test: `bindings/mex/test/test_facets.m` (extend)

- [ ] **Step 1: Add the failing grouped-command assertions**

In `bindings/mex/test/test_facets.m`, insert **before** `nxr_compute('destroy', h);`:

```matlab
% ── grouped facets command ──
opts = struct('singVerts', uint32([1;2]), 'singValues', [1;1], 'source', 'manual');
Fc = nxr_compute('facets', h, 'trivial', opts);
assert(isequal(sort(fieldnames(Fc)), ...
    sort({'Topology';'Embedded';'Intrinsic';'Extrinsic';'Gauge'})), 'five groups');
assert(strcmp(Fc.Gauge.type,'trivial'), 'gauge type trivial');
assert(abs(sum(Fc.Gauge.singularity.indices) - 2) < 1e-12, 'Gauss-Bonnet chi==2');
% grouped sub-structs == standalone
assert(isequal(Fc.Embedded.vertex.grid, E.vertex.grid), 'facets Embedded == embedded');
assert(isequal(Fc.Intrinsic.edge.length, I.edge.length), 'facets Intrinsic == intrinsic');
assert(isequal(Fc.Extrinsic.vertex.meanCurvature, X.vertex.meanCurvature), 'facets Extrinsic == extrinsic');
T = nxr_compute('topology', h);
assert(isequal(Fc.Topology.halfedge.twin, T.halfedge.twin), 'facets Topology == topology');
```

Also update the final success line to:
```matlab
fprintf('ALL TESTS PASSED: test_facets\n');
```

- [ ] **Step 2: Run against current build to verify it fails**

```matlab
cd bindings/mex/test; clear nxr_compute; test_facets
```
Expected: ERROR `Unknown command: "facets"`.

- [ ] **Step 3: Add `cmdFacets`** (immediately after `cmdEmbedded/Intrinsic/Extrinsic`)

```cpp
void cmdFacets(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 3) throw std::invalid_argument(
        "nxr_compute('facets', handle, gaugeType[, opts]) — gaugeType in {euclidean,levi-civita,trivial}");
    ContextHolder& h = getHolder(prhs[1]);
    std::string type = getStringArg(prhs[2]);
    const mxArray* opts = (nrhs >= 4) ? prhs[3] : nullptr;

    const char* f[] = {"Topology","Embedded","Intrinsic","Extrinsic","Gauge"};
    mxArray* s = mxCreateStructMatrix(1,1,5,f);
    mxSetField(s,0,"Topology",  buildTopologyStruct(h, false));
    mxSetField(s,0,"Embedded",  buildEmbeddedStruct(h));
    mxSetField(s,0,"Intrinsic", buildIntrinsicStruct(h));
    mxSetField(s,0,"Extrinsic", buildExtrinsicStruct(h));
    mxSetField(s,0,"Gauge",     buildGaugeStruct(h, type, opts, false));
    plhs[0] = s;
}
```

- [ ] **Step 4: Register `facets` in the dispatch + help string**

In the dispatch chain (after the `extrinsic` line from Task 1):
```cpp
        else if (cmd == "facets")     cmdFacets(nlhs, plhs, nrhs, prhs);
```
Append `facets` to the "Available:" help string (next to `embedded, intrinsic, extrinsic`).

- [ ] **Step 5: Build and run to verify all pass**

```bash
bash scripts/build.sh Release
```
```matlab
cd bindings/mex/test; clear nxr_compute; test_facets
```
Expected: `ALL TESTS PASSED: test_facets`.

- [ ] **Step 6: Commit**

```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_facets.m
git commit -m "feat(mex): grouped facets command {Topology,Embedded,Intrinsic,Extrinsic,Gauge}"
```

---

## Task 3: Back-compat regression + install into the Brainstorm plugin

**Files:** none (verification + install only).

- [ ] **Step 1: Run the existing MEX tests to confirm nothing broke**

```matlab
cd bindings/mex/test; clear nxr_compute
test_bundle; test_geometry_bundle; test_gauge; test_topology
```
Expected: each prints its `ALL TESTS PASSED` line (the flat `geometry`/`bundle`/`gauge`/`topology` commands are unchanged).

- [ ] **Step 2: Back up the installed plugin MEX, then install the new build**

```bash
PLUGDIR="$HOME/.brainstorm/plugins/nxr-compute/nxr-compute-mex-r2023b"
cp "$PLUGDIR/nxr_compute.mexmaca64" "$PLUGDIR/nxr_compute.mexmaca64.bak.prefacets-20260609"
cp build/Release/nxr_compute.mexmaca64 "$PLUGDIR/nxr_compute.mexmaca64"
ls -la "$PLUGDIR/"nxr_compute.mexmaca64*
```

- [ ] **Step 3: Smoke-test the installed MEX from a Brainstorm-context MATLAB session**

```matlab
clear nxr_compute   % drop any repo-build copy so the installed plugin binary loads
which nxr_compute    % should resolve under ~/.brainstorm/plugins/nxr-compute/...
V = [1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1];
F = [1 3 5; 3 2 5; 2 4 5; 4 1 5; 3 1 6; 2 3 6; 4 2 6; 1 4 6];
h = nxr_compute('create', V, F);
Fc = nxr_compute('facets', h, 'trivial', struct('singVerts',uint32([5;6]),'singValues',[1;1]));
disp(fieldnames(Fc));   % {Topology;Embedded;Intrinsic;Extrinsic;Gauge}
nxr_compute('destroy', h);
```
Expected: the five field names print; no error.

- [ ] **Step 4: Commit any plan/doc updates (no source changes in this task)**

If a CLAUDE.md note about the facet commands is desired, add it under the "MATLAB coordinate-system bundle" section and commit:
```bash
git add -A
git commit -m "docs(mex): note facet data commands (embedded/intrinsic/extrinsic/facets)"
```
(If no doc change is made, skip this step.)

---

## Final verification

- [ ] **All MEX tests green together.**

```matlab
cd bindings/mex/test; clear nxr_compute
test_facets; test_bundle; test_geometry_bundle; test_gauge; test_topology
```
Expected: every file prints its `ALL TESTS PASSED` line.

- [ ] **Then proceed to Phase 2** (`tess_frame`, brainstorm3) — a separate plan, now that the installed MEX exposes `facets`.

---

## Notes for the implementer

- **Facet view lifetime:** the views (`emb.vertex()`, etc.) hold `Manifold&`; calling `m.embedded()` then a view method is the documented pattern (`facets.h`). `m` must be the `*h.ctx` reference (non-const) because `lightGeometry()` lazily fills a cache.
- **`normal`/`principalDir` recompute:** `EmbeddedFacet::VertexView::normal()` calls `geometry::vertexFrames(m)` and `ExtrinsicFacet::VertexView::principalDir()` calls `geometry::curvatures(m)` — these are not in the light-geometry cache, so they recompute per call. That's acceptable here (built once per `tess_frame` hemisphere). Do not "optimize" by caching unless a later profile demands it (YAGNI).
- **Converters return column vectors / N×k matrices** already in MATLAB layout; complex arrays cross via the R2018a interleaved API (same as `geometry`/`gauge`). No manual transpose.
- **Data-only:** do NOT add a `withOps` path to the facet commands — operators stay on the `operators` command (spec, out of scope).
- **Icosahedron** is closed genus-0 (χ=2), so the trivial gauge with two +1 singularities satisfies Gauss-Bonnet — that's why `singValues=[1;1]`.
