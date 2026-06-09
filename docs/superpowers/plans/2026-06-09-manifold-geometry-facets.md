# Manifold Geometry Facets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `nxr::manifold::Manifold`'s C++ surface into six GC-faithful, embedding-rooted structures — `topology / embedded / intrinsic / extrinsic` (geometry data), `gauge` (transforms), `operators` (matrices) — with a gauge construction workflow and a lazy, independently-cached operators accessor (plus a thin MEX `operators` command).

**Architecture:** Additive C++ only. Facets are thin typed views that slice from existing lazily-cached bundles (`geometry::meshGeometry` / `geometry::meshTopology`) and delegate to existing free functions (`vertexGrid`, `assembleConnectionLaplacian`, `integrateTrivialGaugeRotations`, …). The existing `mesh()/geometry()/operatorGeometry()` accessors and all bindings keep working unchanged underneath. The three Parts (A/B/C) are independently mergeable; execute in order (B and C depend on A's facet header existing).

**Tech Stack:** C++17, Eigen, geometry-central, MATLAB MEX. Build `bash scripts/build.sh Release`. Native test binaries land in `build/`. MATLAB tests run via the MATLAB MCP `run_matlab_file`.

**Spec:** `docs/superpowers/specs/2026-06-09-manifold-geometry-facets-design.md`.

---

## File Structure

| File | Responsibility | Part |
|---|---|---|
| Create `include/nxr/facets.h` | Facet view classes (`TopologyFacet`/`EmbeddedFacet`/`IntrinsicFacet`/`ExtrinsicFacet`/`GaugeFacet`/`OperatorsFacet`) | A,B,C |
| Create `src/facets.cpp` | Facet method bodies + `Manifold` facet accessors + lazy caches | A,B,C |
| Modify `include/nxr/compute.h` (Manifold class ~80-115) | New facet accessors, raw-input aliases, gauge ctor/state/`setGauge`, lazy caches, `GaugeType`/`OperatorId` enums | A,B,C |
| Modify `src/mesh_operators.cpp` (Manifold ctor/impl) | Retain raw vertex positions; singularity ctor; init gauge state | A,B |
| Create `test/test_facets.cpp` | Part A native tests | A |
| Create `test/test_gauge_workflow.cpp` | Part B native tests | B |
| Create `test/test_operators_facet.cpp` | Part C native tests | C |
| Modify `bindings/mex/src/nxr_compute_mex.cpp` | `operators` string-dispatch command | C |
| Create `bindings/mex/test/test_operators_command.m` | Part C MATLAB test | C |
| Modify `CMakeLists.txt` (~188-190 region) | Register the three new native test executables | A,B,C |

**Backing-store insight:** the light per-element facet *data* already exists as
`geometry::MeshGeometry` (vertex/edge/face/halfedge grids, dual areas, angle sums, edge
lengths, cotan weights, dihedral angles, centroids, curvature) and `geometry::MeshTopology`.
Data facets slice from a **lazily-cached** copy of these on the `Manifold`. A few quantities
not in `MeshGeometry` (vertex/face normals, principal directions, raw positions) delegate to
`geometry::vertexFrames` / `geometry::frames` / `geometry::curvatures` / retained input.

**Gauss–Bonnet note:** the library already documents (compute.h:486, :1021) the contract
`Σ singularityMap values == χ(mesh)` but does **not** enforce it. The gauge workflow adds the
enforcement (`χ = nV − nE + nF`).

---

# PART A — Geometry facets (topology / embedded / intrinsic / extrinsic)

## Task A1: Facet header skeleton + lazy bundle caches + `TopologyFacet`

**Files:**
- Create: `include/nxr/facets.h`
- Create: `src/facets.cpp`
- Modify: `include/nxr/compute.h` (Manifold class, ~88-115)
- Modify: `src/mesh_operators.cpp` (Manifold impl — add cache members' definitions if needed)
- Create: `test/test_facets.cpp`
- Modify: `CMakeLists.txt` (after line 190)

- [ ] **Step 1: Add facet forward-declarations + accessors + lazy caches to `Manifold`** in `include/nxr/compute.h`. Insert into the `Manifold` public section (after `operatorGeometry()`, before `private:` at line ~110):

```cpp
    // ── Geometry facets (additive; thin views over the accessors above) ──
    // Each returns a lightweight view object; see include/nxr/facets.h.
    facet::TopologyFacet  topology();
    facet::EmbeddedFacet  embedded();
    facet::IntrinsicFacet intrinsic();
    facet::ExtrinsicFacet extrinsic();

    // Raw-input aliases (facet-agnostic literal input).
    const Eigen::MatrixXd& vertexPositions() const;  // [nV,3], retained from ctor
    Eigen::MatrixXi        faces() const;             // [nF,3], 0-based

    // Lazily-cached light per-element bundles backing the data facets.
    const geometry::MeshGeometry& lightGeometry();   // geometry::meshGeometry(*this), cached
    const geometry::MeshTopology& topologyData();    // geometry::meshTopology(*this), cached
```

Add the forward declarations near the top of the `nxr::manifold` namespace (before `class Manifold`, ~line 78):

```cpp
namespace facet {
    class TopologyFacet;  class EmbeddedFacet;  class IntrinsicFacet;
    class ExtrinsicFacet; class GaugeFacet;     class OperatorsFacet;
}
namespace geometry { struct MeshGeometry; struct MeshTopology; }
```

Add to the `Manifold` `private:` section (after line 114):

```cpp
    Eigen::MatrixXd                          vertexPositions_;  // [nV,3] retained input
    std::unique_ptr<geometry::MeshGeometry>  lightGeometryCache_;
    std::unique_ptr<geometry::MeshTopology>  topologyDataCache_;
```

- [ ] **Step 2: Create `include/nxr/facets.h`** with the topology facet (other facets added in later tasks):

```cpp
#pragma once
#include "nxr/compute.h"

namespace nxr::manifold::facet {

// Thin view over the halfedge combinatorics. Holds a Manifold& only.
class TopologyFacet {
public:
    explicit TopologyFacet(Manifold& m) : m_(m) {}
    int nV() const; int nE() const; int nF() const; int nH() const; int nC() const;
    int eulerCharacteristic() const;                  // nV - nE + nF
    const geometry::MeshTopology& all() const;        // cached SoA
private:
    Manifold& m_;
};

} // namespace nxr::manifold::facet
```

- [ ] **Step 3: Create `src/facets.cpp`** with the Manifold accessors + topology facet bodies:

```cpp
#include "nxr/facets.h"

namespace nxr::manifold {

const geometry::MeshGeometry& Manifold::lightGeometry() {
    if (!lightGeometryCache_)
        lightGeometryCache_ =
            std::make_unique<geometry::MeshGeometry>(geometry::meshGeometry(*this));
    return *lightGeometryCache_;
}
const geometry::MeshTopology& Manifold::topologyData() {
    if (!topologyDataCache_)
        topologyDataCache_ =
            std::make_unique<geometry::MeshTopology>(geometry::meshTopology(*this));
    return *topologyDataCache_;
}
const Eigen::MatrixXd& Manifold::vertexPositions() const { return vertexPositions_; }
Eigen::MatrixXi Manifold::faces() const {
    const auto& t = const_cast<Manifold*>(this)->topologyData();
    (void)t;
    // Faces are reconstructed from the mesh: 3 vertices per face, 0-based.
    Eigen::MatrixXi F(nF(), 3);
    auto& mesh = const_cast<Manifold*>(this)->mesh();
    int fi = 0;
    for (auto f : mesh.faces()) {
        int k = 0;
        for (auto v : f.adjacentVertices()) F(fi, k++) = static_cast<int>(v.getIndex());
        ++fi;
    }
    return F;
}

facet::TopologyFacet  Manifold::topology()  { return facet::TopologyFacet(*this); }

} // namespace nxr::manifold

namespace nxr::manifold::facet {
int TopologyFacet::nV() const { return m_.nV(); }
int TopologyFacet::nE() const { return m_.nE(); }
int TopologyFacet::nF() const { return m_.nF(); }
int TopologyFacet::nH() const { return m_.topologyData().nH; }
int TopologyFacet::nC() const { return m_.topologyData().nC; }
int TopologyFacet::eulerCharacteristic() const { return m_.nV() - m_.nE() + m_.nF(); }
const geometry::MeshTopology& TopologyFacet::all() const { return m_.topologyData(); }
} // namespace nxr::manifold::facet
```

- [ ] **Step 4: Retain raw positions in the Manifold constructor.** In `src/mesh_operators.cpp`, find the `Manifold::Manifold(const double* vertices, int nV, ...)` constructor body. After `geometry_` is built, add:

```cpp
    vertexPositions_.resize(nV, 3);
    for (int i = 0; i < nV; ++i) {
        vertexPositions_(i, 0) = vertices[3 * i + 0];
        vertexPositions_(i, 1) = vertices[3 * i + 1];
        vertexPositions_(i, 2) = vertices[3 * i + 2];
    }
```

(Confirm the ctor already iterates `vertices` in `xyz xyz` row-major to build polygons; mirror that indexing.)

- [ ] **Step 5: Write the failing test** `test/test_facets.cpp`:

```cpp
#include "nxr/facets.h"
#include <cmath>
#include <iostream>
using namespace nxr::manifold;

static int g_failures = 0;
#define EXPECT(cond, msg) do { \
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; } \
    else { std::cout << "  [FAIL] " << msg << "\n"; ++g_failures; } } while (0)

// Unit icosphere (12 verts), closed genus-0, chi = 2.
static void icosphere(std::vector<double>& V, std::vector<int32_t>& F) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    V = {-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t,
          0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1};
    F = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2,
         10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5,
         2,4,11, 6,2,10, 8,6,7, 9,8,1};
}

static void testTopologyFacet() {
    std::cout << "\n=== facets: topology ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    auto topo = m.topology();
    EXPECT(topo.nV() == 12, "topology nV");
    EXPECT(topo.nF() == 20, "topology nF");
    EXPECT(topo.nE() == 30, "topology nE");
    EXPECT(topo.eulerCharacteristic() == 2, "chi = V-E+F = 2 (closed genus-0)");
    EXPECT(topo.nH() == 60, "topology nH = 2E");
    EXPECT((int)topo.all().heTwin.size() == 60, "MeshTopology SoA wired");
}

int main() {
    testTopologyFacet();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
```

Register in `CMakeLists.txt` after line 190:

```cmake
    add_executable(test_facets test/test_facets.cpp)
    target_link_libraries(test_facets PRIVATE nxr_compute)
    add_test(NAME test_facets COMMAND test_facets)
```

Add `src/facets.cpp` to the `nxr_compute` library sources (find the `add_library(nxr_compute ...)` / sources list and append `src/facets.cpp`).

- [ ] **Step 6: Build to verify it fails** (link/compile error until facets compile):

Run: `bash scripts/build.sh Release 2>&1 | tail -8`
Expected: builds the facet code; `./build/test_facets` runs.

- [ ] **Step 7: Run the test**

Run: `./build/test_facets`
Expected: `ALL PASSED` (6 PASS lines).

- [ ] **Step 8: Commit**

```bash
git add include/nxr/facets.h src/facets.cpp include/nxr/compute.h src/mesh_operators.cpp test/test_facets.cpp CMakeLists.txt
git commit -m "feat(facets): TopologyFacet + lazy MeshGeometry/MeshTopology caches + raw-input aliases

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Task A2: `EmbeddedFacet` (vertex/face position, normal, grid, centroid)

**Files:** Modify `include/nxr/facets.h`, `src/facets.cpp`, `test/test_facets.cpp`

- [ ] **Step 1: Add `EmbeddedFacet` to `include/nxr/facets.h`** (after `TopologyFacet`):

```cpp
class EmbeddedFacet {
public:
    explicit EmbeddedFacet(Manifold& m) : m_(m) {}
    struct VertexView { Manifold& m;
        Eigen::MatrixXd  position() const;   // [nV,3] raw input
        Eigen::MatrixXd  normal()   const;   // [nV,3]
        Eigen::MatrixXcd grid()     const;   // [nV,3] c = e1 + i e2
    };
    struct FaceView { Manifold& m;
        Eigen::MatrixXd  normal()   const;   // [nF,3]
        Eigen::MatrixXcd grid()     const;   // [nF,3]
        Eigen::MatrixXd  centroid() const;   // [nF,3]
    };
    VertexView vertex() const { return VertexView{m_}; }
    FaceView   face()   const { return FaceView{m_}; }
private:
    Manifold& m_;
};
```

- [ ] **Step 2: Add `EmbeddedFacet` bodies to `src/facets.cpp`** (in the `nxr::manifold::facet` block, and the accessor in the `nxr::manifold` block):

```cpp
// in namespace nxr::manifold:
facet::EmbeddedFacet  Manifold::embedded()  { return facet::EmbeddedFacet(*this); }

// in namespace nxr::manifold::facet:
Eigen::MatrixXd  EmbeddedFacet::VertexView::position() const { return m.vertexPositions(); }
Eigen::MatrixXd  EmbeddedFacet::VertexView::normal()   const { return geometry::vertexFrames(m).normals; }
Eigen::MatrixXcd EmbeddedFacet::VertexView::grid()     const { return m.lightGeometry().vertexGrid; }
Eigen::MatrixXd  EmbeddedFacet::FaceView::normal()     const { return geometry::frames(m).normals; }
Eigen::MatrixXcd EmbeddedFacet::FaceView::grid()       const { return m.lightGeometry().faceGrid; }
Eigen::MatrixXd  EmbeddedFacet::FaceView::centroid()   const { return m.lightGeometry().faceCentroids; }
```

- [ ] **Step 3: Write the failing test** — add to `test/test_facets.cpp` and call from `main()`:

```cpp
static void testEmbeddedFacet() {
    std::cout << "\n=== facets: embedded ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    auto e = m.embedded();
    EXPECT(e.vertex().position().rows() == 12 && e.vertex().position().cols() == 3, "vertex.position [12,3]");
    // raw-input identity: position row 0 equals the input
    EXPECT(std::abs(e.vertex().position()(0,0) - (-1.0)) < 1e-12, "vertex.position is raw input");
    EXPECT(e.vertex().grid().rows() == 12 && e.vertex().grid().cols() == 3, "vertex.grid [12,3] complex");
    // facet-identity: embedded.vertex.grid == geometry::vertexGrid(m)
    EXPECT((e.vertex().grid() - geometry::vertexGrid(m)).cwiseAbs().maxCoeff() < 1e-12, "vertex.grid == vertexGrid(m)");
    EXPECT(e.face().grid().rows() == 20, "face.grid [20,3]");
    EXPECT(e.face().centroid().rows() == 20 && e.face().centroid().cols() == 3, "face.centroid [20,3]");
    EXPECT(e.vertex().normal().rows() == 12, "vertex.normal [12,3]");
    // grid encodes the normal: Re(c) x Im(c) is unit and aligns with vertex.normal
    Eigen::RowVector3d e1 = e.vertex().grid().row(0).real();
    Eigen::RowVector3d e2 = e.vertex().grid().row(0).imag();
    Eigen::RowVector3d nFromGrid = e1.cross(e2);
    EXPECT(std::abs(nFromGrid.norm() - 1.0) < 1e-9, "Re(c) x Im(c) is unit normal");
}
```

- [ ] **Step 4: Build + run**

Run: `bash scripts/build.sh Release 2>&1 | tail -4 && ./build/test_facets`
Expected: `ALL PASSED` (topology + embedded PASS lines).

- [ ] **Step 5: Commit**

```bash
git add include/nxr/facets.h src/facets.cpp test/test_facets.cpp
git commit -m "feat(facets): EmbeddedFacet (vertex/face position, normal, grid, centroid)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Task A3: `IntrinsicFacet` (vertex/edge/halfedge metric quantities)

**Files:** Modify `include/nxr/facets.h`, `src/facets.cpp`, `test/test_facets.cpp`

- [ ] **Step 1: Add `IntrinsicFacet` to `include/nxr/facets.h`:**

```cpp
class IntrinsicFacet {
public:
    explicit IntrinsicFacet(Manifold& m) : m_(m) {}
    struct VertexView { Manifold& m;
        Eigen::VectorXd dualArea() const;     // [nV]
        Eigen::VectorXd angleSum() const;     // [nV]
    };
    struct EdgeView { Manifold& m;
        Eigen::VectorXd length()      const;  // [nE]
        Eigen::VectorXd cotanWeight() const;  // [nE]
    };
    struct HalfedgeView { Manifold& m;
        Eigen::VectorXd  cotanWeight()     const;  // [nH]
        Eigen::VectorXcd transportAlong()  const;  // [nH]
        Eigen::VectorXcd transportAcross() const;  // [nH]
    };
    VertexView   vertex()   const { return VertexView{m_}; }
    EdgeView     edge()     const { return EdgeView{m_}; }
    HalfedgeView halfedge() const { return HalfedgeView{m_}; }
private:
    Manifold& m_;
};
```

- [ ] **Step 2: Add `IntrinsicFacet` bodies to `src/facets.cpp`:**

```cpp
// in namespace nxr::manifold:
facet::IntrinsicFacet Manifold::intrinsic() { return facet::IntrinsicFacet(*this); }

// in namespace nxr::manifold::facet:
Eigen::VectorXd  IntrinsicFacet::VertexView::dualArea() const { return m.lightGeometry().vertexDualAreas; }
Eigen::VectorXd  IntrinsicFacet::VertexView::angleSum() const { return m.lightGeometry().vertexAngleSums; }
Eigen::VectorXd  IntrinsicFacet::EdgeView::length()       const { return m.lightGeometry().edgeLengths; }
Eigen::VectorXd  IntrinsicFacet::EdgeView::cotanWeight()  const { return m.lightGeometry().edgeCotanWeights; }
Eigen::VectorXd  IntrinsicFacet::HalfedgeView::cotanWeight()     const { return m.lightGeometry().halfedgeCotanWeights; }
Eigen::VectorXcd IntrinsicFacet::HalfedgeView::transportAlong()  const { return m.lightGeometry().halfedgeTransportAlong; }
Eigen::VectorXcd IntrinsicFacet::HalfedgeView::transportAcross() const { return m.lightGeometry().halfedgeTransportAcross; }
```

- [ ] **Step 3: Write the failing test** — add to `test/test_facets.cpp`, call from `main()`:

```cpp
static void testIntrinsicFacet() {
    std::cout << "\n=== facets: intrinsic ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    auto in = m.intrinsic();
    EXPECT(in.vertex().dualArea().size() == 12, "vertex.dualArea [12]");
    EXPECT(in.vertex().angleSum().size() == 12, "vertex.angleSum [12]");
    EXPECT(in.edge().length().size() == 30, "edge.length [30]");
    EXPECT(in.edge().cotanWeight().size() == 30, "edge.cotanWeight [30]");
    EXPECT(in.halfedge().transportAlong().size() == 60, "halfedge.transportAlong [60]");
    // dual areas sum to total area (closed mesh)
    EXPECT(std::abs(in.vertex().dualArea().sum() - m.lightGeometry().totalArea) < 1e-9, "dualArea sums to totalArea");
    // facet-identity: intrinsic.edge.length matches a direct GC require
    auto& g = m.operatorGeometry(); g.requireEdgeLengths();
    Eigen::VectorXd direct(30);
    for (auto edge : m.mesh().edges()) direct(edge.getIndex()) = g.edgeLengths[edge];
    EXPECT((in.edge().length() - direct).cwiseAbs().maxCoeff() < 1e-12, "edge.length == GC edgeLengths");
}
```

- [ ] **Step 4: Build + run**

Run: `bash scripts/build.sh Release 2>&1 | tail -4 && ./build/test_facets`
Expected: `ALL PASSED`.

- [ ] **Step 5: Commit**

```bash
git add include/nxr/facets.h src/facets.cpp test/test_facets.cpp
git commit -m "feat(facets): IntrinsicFacet (dual area, angle sum, edge length/cotan, halfedge transport)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Task A4: `ExtrinsicFacet` (dihedral, principal direction, 2-RoSy curvature)

**Files:** Modify `include/nxr/facets.h`, `src/facets.cpp`, `test/test_facets.cpp`

- [ ] **Step 1: Add `ExtrinsicFacet` to `include/nxr/facets.h`:**

```cpp
class ExtrinsicFacet {
public:
    explicit ExtrinsicFacet(Manifold& m) : m_(m) {}
    struct VertexView { Manifold& m;
        Eigen::VectorXcd curvature2RoSy() const;  // [nV] deviatoric q
        Eigen::VectorXd  meanCurvature()  const;  // [nV] H
        Eigen::MatrixXd  principalDir()   const;  // [nV,3] max principal dir
    };
    struct EdgeView { Manifold& m;
        Eigen::VectorXd dihedralAngle() const;    // [nE]
    };
    VertexView vertex() const { return VertexView{m_}; }
    EdgeView   edge()   const { return EdgeView{m_}; }
private:
    Manifold& m_;
};
```

- [ ] **Step 2: Add `ExtrinsicFacet` bodies to `src/facets.cpp`:**

```cpp
// in namespace nxr::manifold:
facet::ExtrinsicFacet Manifold::extrinsic() { return facet::ExtrinsicFacet(*this); }

// in namespace nxr::manifold::facet:
Eigen::VectorXcd ExtrinsicFacet::VertexView::curvature2RoSy() const { return m.lightGeometry().vertexCurvatureDeviatoric; }
Eigen::VectorXd  ExtrinsicFacet::VertexView::meanCurvature()  const { return m.lightGeometry().vertexMeanCurvature; }
Eigen::MatrixXd  ExtrinsicFacet::VertexView::principalDir()   const { return geometry::curvatures(m).principalDirMax; }
Eigen::VectorXd  ExtrinsicFacet::EdgeView::dihedralAngle()    const { return m.lightGeometry().edgeDihedralAngles; }
```

- [ ] **Step 3: Write the failing test** — add to `test/test_facets.cpp`, call from `main()`:

```cpp
static void testExtrinsicFacet() {
    std::cout << "\n=== facets: extrinsic ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    auto ex = m.extrinsic();
    EXPECT(ex.vertex().curvature2RoSy().size() == 12, "vertex.curvature2RoSy [12]");
    EXPECT(ex.vertex().meanCurvature().size() == 12, "vertex.meanCurvature [12]");
    EXPECT(ex.vertex().principalDir().rows() == 12 && ex.vertex().principalDir().cols() == 3, "vertex.principalDir [12,3]");
    EXPECT(ex.edge().dihedralAngle().size() == 30, "edge.dihedralAngle [30]");
    // identity vs geometry::vertexCurvature
    auto vc = geometry::vertexCurvature(m);
    EXPECT((ex.vertex().meanCurvature() - vc.mean).cwiseAbs().maxCoeff() < 1e-12, "meanCurvature == vertexCurvature.mean");
}
```

- [ ] **Step 4: Build + run**

Run: `bash scripts/build.sh Release 2>&1 | tail -4 && ./build/test_facets`
Expected: `ALL PASSED` (topology + embedded + intrinsic + extrinsic).

- [ ] **Step 5: Regression**

Run: `./build/test_geometry_bundle && ./build/test_intrinsic_delaunay`
Expected: both `ALL PASSED` (facets are additive — existing behavior unchanged).

- [ ] **Step 6: Commit**

```bash
git add include/nxr/facets.h src/facets.cpp test/test_facets.cpp
git commit -m "feat(facets): ExtrinsicFacet (dihedral, principal direction, 2-RoSy curvature)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Part A — before you begin
Read `include/nxr/compute.h` lines 78-115 (Manifold) and 1182-1361 (geometry namespace), and
`src/mesh_operators.cpp`'s `Manifold` constructor (to mirror the `vertices` row-major indexing
when retaining `vertexPositions_`). The facet data all flows from `geometry::meshGeometry` /
`geometry::meshTopology` / the frame + curvature free functions — facets only slice and delegate;
no new geometry math. Confirm `src/facets.cpp` is added to the `nxr_compute` library sources in
`CMakeLists.txt`. clangd "file not found" diagnostics on Eigen/nxr headers are IDE noise — trust
`scripts/build.sh`.

---

# PART B — Gauge construction workflow

## Task B1: `GaugeType` enum + Gauss–Bonnet validation + active-gauge state

**Files:** Modify `include/nxr/compute.h` (Manifold + a `GaugeType` enum), `src/facets.cpp`, `src/mesh_operators.cpp`, Create `test/test_gauge_workflow.cpp`, Modify `CMakeLists.txt`

- [ ] **Step 1: Add `GaugeType` + gauge state/methods to `Manifold`** in `include/nxr/compute.h`. Add the enum just before `class Manifold` (~line 77):

```cpp
enum class GaugeType { Euclidean, LeviCivita, Trivial };
```

Add to the `Manifold` public section (after the facet accessors from Task A1):

```cpp
    // ── Gauge workflow ──
    // Singularity-aware constructor (pattern 2): default active gauge = Trivial.
    // Throws Error(InvalidInput) if Σ singularities != eulerCharacteristic().
    Manifold(const double* vertices, int nV, const int32_t* faces, int nF,
             const std::map<int,double>& singularities, bool intrinsicDelaunay = false);

    int eulerCharacteristic() const;                 // nV - nE + nF
    GaugeType activeGaugeType() const;
    const std::map<int,double>& activeSingularities() const;
    // Re-point the active/default gauge. Trivial validates Gauss-Bonnet.
    void setGauge(GaugeType type, const std::map<int,double>& singularities = {});

    facet::GaugeFacet gauge();                                   // the active gauge
    facet::GaugeFacet gauge(GaugeType type,
                            const std::map<int,double>& singularities = {});
```

Add to the `Manifold` `private:` section:

```cpp
    GaugeType            activeGaugeType_ = GaugeType::LeviCivita;
    std::map<int,double> activeSingularities_;
    // Validates Σ singularities == eulerCharacteristic(); throws on mismatch.
    void validateSingularities_(const std::map<int,double>& s) const;
```

Ensure `<map>` is included in compute.h (it is — `std::map` already used elsewhere).

- [ ] **Step 2: Implement validation + state accessors** in `src/facets.cpp` (namespace `nxr::manifold`):

```cpp
int Manifold::eulerCharacteristic() const { return nV() - nE() + nF(); }
GaugeType Manifold::activeGaugeType() const { return activeGaugeType_; }
const std::map<int,double>& Manifold::activeSingularities() const { return activeSingularities_; }

void Manifold::validateSingularities_(const std::map<int,double>& s) const {
    if (s.empty())
        throw Error(ErrorCode::InvalidInput,
            "Trivial gauge requires a singularity map; Levi-Civita needs none.",
            "Pass a {vertexIndex -> index} map whose values sum to the Euler "
            "characteristic chi = " + std::to_string(eulerCharacteristic()) + ".");
    double sum = 0.0;
    for (auto& kv : s) sum += kv.second;
    const int chi = eulerCharacteristic();
    if (std::abs(sum - static_cast<double>(chi)) > 1e-9)
        throw Error(ErrorCode::InvalidInput,
            "Gauss-Bonnet violated: singularity indices sum to " + std::to_string(sum) +
            " but must equal the Euler characteristic chi = " + std::to_string(chi) + ".",
            "A closed genus-0 surface has chi=2 (e.g. two +1 singularities); a disk chi=1.");
}

void Manifold::setGauge(GaugeType type, const std::map<int,double>& singularities) {
    if (type == GaugeType::Trivial) validateSingularities_(singularities);
    activeGaugeType_ = type;
    activeSingularities_ = (type == GaugeType::Trivial) ? singularities : std::map<int,double>{};
}
```

(Confirm `Error` / `ErrorCode::InvalidInput` are the correct symbols — they are, per
`src/connection_laplacian.cpp` usage. Include `<cmath>` for `std::abs` if not present.)

- [ ] **Step 3: Implement the singularity constructor** in `src/mesh_operators.cpp`. Add a delegating constructor after the existing ctor:

```cpp
Manifold::Manifold(const double* vertices, int nV, const int32_t* faces, int nF,
                   const std::map<int,double>& singularities, bool intrinsicDelaunay)
    : Manifold(vertices, nV, faces, nF, intrinsicDelaunay) {
    validateSingularities_(singularities);     // throws on Gauss-Bonnet mismatch
    activeGaugeType_     = GaugeType::Trivial;
    activeSingularities_ = singularities;
}
```

(If the primary ctor is not delegatable as written — e.g. it's defined out-of-line — replicate
its body then append the four lines. Confirm the member-init delegation compiles.)

- [ ] **Step 4: Write the failing test** `test/test_gauge_workflow.cpp`:

```cpp
#include "nxr/facets.h"
#include <cmath>
#include <iostream>
using namespace nxr::manifold;

static int g_failures = 0;
#define EXPECT(cond, msg) do { \
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; } \
    else { std::cout << "  [FAIL] " << msg << "\n"; ++g_failures; } } while (0)

static void icosphere(std::vector<double>& V, std::vector<int32_t>& F) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    V = {-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t,
          0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1};
    F = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2,
         10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5,
         2,4,11, 6,2,10, 8,6,7, 9,8,1};
}

static void testDefaultsAndValidation() {
    std::cout << "\n=== gauge: defaults + Gauss-Bonnet ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    EXPECT(m.activeGaugeType() == GaugeType::LeviCivita, "pattern 1 default = Levi-Civita");
    EXPECT(m.eulerCharacteristic() == 2, "chi = 2");

    // valid singularities (sum == chi == 2)
    bool ok = true;
    try { m.setGauge(GaugeType::Trivial, {{0,1.0},{3,1.0}}); }
    catch (...) { ok = false; }
    EXPECT(ok && m.activeGaugeType() == GaugeType::Trivial, "setGauge(trivial, sum=2) ok");

    // invalid sum throws
    bool threw = false;
    try { m.setGauge(GaugeType::Trivial, {{0,1.0}}); } catch (const Error&) { threw = true; }
    EXPECT(threw, "Gauss-Bonnet violation (sum=1 != chi=2) throws");

    // empty singularities throws
    threw = false;
    try { m.setGauge(GaugeType::Trivial, {}); } catch (const Error&) { threw = true; }
    EXPECT(threw, "trivial with empty singularities throws");

    // pattern 2: singularity ctor -> default trivial
    Manifold m2(V.data(), 12, F.data(), 20, std::map<int,double>{{0,1.0},{3,1.0}});
    EXPECT(m2.activeGaugeType() == GaugeType::Trivial, "pattern 2 ctor default = Trivial");
    // pattern 2 with bad sum throws at construction
    threw = false;
    try { Manifold mb(V.data(), 12, F.data(), 20, std::map<int,double>{{0,1.0}}); }
    catch (const Error&) { threw = true; }
    EXPECT(threw, "pattern 2 ctor validates Gauss-Bonnet");
}

int main() {
    testDefaultsAndValidation();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
```

Register in `CMakeLists.txt`:

```cmake
    add_executable(test_gauge_workflow test/test_gauge_workflow.cpp)
    target_link_libraries(test_gauge_workflow PRIVATE nxr_compute)
    add_test(NAME test_gauge_workflow COMMAND test_gauge_workflow)
```

- [ ] **Step 5: Build + run**

Run: `bash scripts/build.sh Release 2>&1 | tail -6 && ./build/test_gauge_workflow`
Expected: `ALL PASSED`.

- [ ] **Step 6: Commit**

```bash
git add include/nxr/compute.h src/facets.cpp src/mesh_operators.cpp test/test_gauge_workflow.cpp CMakeLists.txt
git commit -m "feat(gauge): GaugeType + Gauss-Bonnet validation + singularity ctor + setGauge

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Task B2: `GaugeFacet` (grid realization: Levi-Civita + trivial) + lazy DEC/cache

**Files:** Modify `include/nxr/facets.h`, `src/facets.cpp`, `include/nxr/compute.h` (lazy DEC/cache on Manifold), `test/test_gauge_workflow.cpp`

- [ ] **Step 1: Add lazy DEC + CholeskyCache accessors to `Manifold`** in `include/nxr/compute.h` public section (needed to realize the trivial grid):

```cpp
    // Lazy operator-assembly helpers shared by gauge + operators facets.
    ops::DECOperators& decOperators();     // assembleDECOperators(*this), cached
    ops::CholeskyCache& choleskyCache();   // owned cache, cached
```

Add to `private:`:

```cpp
    std::unique_ptr<ops::DECOperators>  decCache_;
    std::unique_ptr<ops::CholeskyCache> choleskyCache_;
```

(`ops::DECOperators` and `ops::CholeskyCache` are already declared in compute.h above the
Manifold class? They are declared later — forward-declare them before `class Manifold`:
`namespace ops { struct DECOperators; class CholeskyCache; }`. Confirm `CholeskyCache` is a
`class`; adjust the tag if it's a `struct`.)

- [ ] **Step 2: Add `GaugeFacet` to `include/nxr/facets.h`:**

```cpp
class GaugeFacet {
public:
    GaugeFacet(Manifold& m, GaugeType type, std::map<int,double> singularities)
        : m_(m), type_(type), sing_(std::move(singularities)) {}
    GaugeType type() const { return type_; }
    const std::map<int,double>& singularities() const { return sing_; }
    // Realized per-vertex frame in this gauge: [nV,3] complex c = e1 + i e2.
    // Levi-Civita/Euclidean: the raw vertex grid. Trivial: exp(i phi_v) .* grid.
    Eigen::MatrixXcd grid() const;
private:
    Manifold& m_;
    GaugeType type_;
    std::map<int,double> sing_;
};
```

- [ ] **Step 3: Implement DEC/cache accessors, the two `gauge()` overloads, and `GaugeFacet::grid()`** in `src/facets.cpp`:

```cpp
// in namespace nxr::manifold:
ops::DECOperators& Manifold::decOperators() {
    if (!decCache_)
        decCache_ = std::make_unique<ops::DECOperators>(ops::assembleDECOperators(*this));
    return *decCache_;
}
ops::CholeskyCache& Manifold::choleskyCache() {
    if (!choleskyCache_) choleskyCache_ = std::make_unique<ops::CholeskyCache>();
    return *choleskyCache_;
}
facet::GaugeFacet Manifold::gauge() {
    return facet::GaugeFacet(*this, activeGaugeType_, activeSingularities_);
}
facet::GaugeFacet Manifold::gauge(GaugeType type, const std::map<int,double>& singularities) {
    if (type == GaugeType::Trivial) validateSingularities_(singularities);
    return facet::GaugeFacet(*this, type,
                             type == GaugeType::Trivial ? singularities : std::map<int,double>{});
}

// in namespace nxr::manifold::facet:
Eigen::MatrixXcd GaugeFacet::grid() const {
    Eigen::MatrixXcd g = geometry::vertexGrid(m_);            // LC / euclidean base frame
    if (type_ != GaugeType::Trivial) return g;
    connection::GaugeRotations gr = connection::integrateTrivialGaugeRotations(
        m_, m_.decOperators(), m_.choleskyCache(), sing_);
    for (int v = 0; v < g.rows(); ++v) g.row(v) *= gr.vertex(v);  // exp(i phi_v) .* grid
    return g;
}
```

(Confirm the namespace of `GaugeRotations` / `integrateTrivialGaugeRotations` — they are in
`nxr::manifold::connection` per compute.h. Add `namespace conn = nxr::manifold::connection;` or
qualify fully. `CholeskyCache` default-constructible — confirm; if it needs no args, the above is
correct.)

- [ ] **Step 4: Write the failing test** — add to `test/test_gauge_workflow.cpp`, call from `main()`:

```cpp
static void testGridRealization() {
    std::cout << "\n=== gauge: grid realization ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    // pattern 1: gauge() is LC; grid == raw vertexGrid
    auto gLC = m.gauge();
    EXPECT(gLC.type() == GaugeType::LeviCivita, "active gauge LC");
    EXPECT((gLC.grid() - geometry::vertexGrid(m)).cwiseAbs().maxCoeff() < 1e-12, "LC grid == vertexGrid");

    // pattern 3: ad-hoc trivial request returns a value; default unchanged
    std::map<int,double> sing{{0,1.0},{3,1.0}};
    auto gTriReq = m.gauge(GaugeType::Trivial, sing);
    EXPECT(m.activeGaugeType() == GaugeType::LeviCivita, "ad-hoc gauge() does not mutate default");
    Eigen::MatrixXcd triGrid = gTriReq.grid();
    EXPECT(triGrid.rows() == 12 && triGrid.cols() == 3, "trivial grid [12,3]");
    // realized trivial frame stays unit-tangent (|c row| preserved vs LC up to rotation)
    EXPECT(std::abs(triGrid.row(0).norm() - geometry::vertexGrid(m).row(0).norm()) < 1e-9,
           "trivial rotation preserves frame magnitude");

    // pattern 2 default trivial equals the pattern-3 value for same singularities
    Manifold m2(V.data(), 12, F.data(), 20, sing);
    EXPECT((m2.gauge().grid() - triGrid).cwiseAbs().maxCoeff() < 1e-9,
           "pattern 2 default == pattern 3 request (same singularities)");

    // setGauge re-points default
    m.setGauge(GaugeType::Trivial, sing);
    EXPECT(m.gauge().type() == GaugeType::Trivial, "setGauge re-points active gauge");
}
```

- [ ] **Step 5: Build + run**

Run: `bash scripts/build.sh Release 2>&1 | tail -6 && ./build/test_gauge_workflow`
Expected: `ALL PASSED`.

- [ ] **Step 6: Commit**

```bash
git add include/nxr/facets.h src/facets.cpp include/nxr/compute.h test/test_gauge_workflow.cpp
git commit -m "feat(gauge): GaugeFacet grid realization (LC + trivial) + lazy DEC/Cholesky cache

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Part B — before you begin
Read compute.h:1005-1035 (`computeTrivialConnection`, `GaugeRotations`,
`integrateTrivialGaugeRotations`) and 495-501 (`assembleTrivialConnectionLaplacian`) for exact
signatures and namespaces (`nxr::manifold::connection`). Confirm `CholeskyCache`'s constructor
(default-constructible vs requires args) in compute.h's CholeskyCache section (~line 293+). The
trivial realization mirrors the MEX `buildGaugeOperators` rotation loop
(`grid.row(v) *= gr.vertex(v)`). Gauss-Bonnet for the icosphere: χ=2, so `{{0,1},{3,1}}` is valid.

---

# PART C — Operators access (`manifold/operators`)

## Task C1: `OperatorsFacet` core — `laplacian.cotan` / `laplacian.graph` + independent cache

**Files:** Modify `include/nxr/facets.h`, `src/facets.cpp`, `include/nxr/compute.h` (operator cache + `OperatorId`), Create `test/test_operators_facet.cpp`, Modify `CMakeLists.txt`

- [ ] **Step 1: Add `OperatorId` enum + per-operator cache to `Manifold`** in `include/nxr/compute.h`. Add the enum before `class Manifold`:

```cpp
enum class OperatorId {
    LaplacianCotan, LaplacianGraph, LaplacianConnection, LaplacianCovariant,
    Dec, MassLumped, MassGalerkin
};
```

Add to `Manifold` public:

```cpp
    facet::OperatorsFacet operators();
    void releaseOperator(OperatorId id);     // drop a cached operator (GC unrequire-style)
    bool isOperatorCached(OperatorId id) const;
```

Add to `private:` (independent slots — requesting one never builds another):

```cpp
    std::unique_ptr<Eigen::SparseMatrix<double>>                cacheLaplacianCotan_;
    std::unique_ptr<Eigen::SparseMatrix<double>>                cacheLaplacianGraph_;
    std::unique_ptr<Eigen::SparseMatrix<std::complex<double>>>  cacheLaplacianConnection_;
    std::unique_ptr<Eigen::SparseMatrix<double>>                cacheLaplacianCovariant_;
    std::unique_ptr<Eigen::SparseMatrix<double>>                cacheMassLumped_;
    std::unique_ptr<Eigen::SparseMatrix<double>>                cacheMassGalerkin_;
```

(The `Dec` slot reuses the existing `decCache_` from Task B1. Include `<complex>` — already used.)

- [ ] **Step 2: Add `OperatorsFacet` (cotan + graph only for now) to `include/nxr/facets.h`:**

```cpp
class OperatorsFacet {
public:
    explicit OperatorsFacet(Manifold& m) : m_(m) {}
    struct LaplacianView { Manifold& m;
        const Eigen::SparseMatrix<double>& cotan() const;   // real, intrinsic
        const Eigen::SparseMatrix<double>& graph() const;   // real, topology
        // connection()/covariant() added in Task C3
        const Eigen::SparseMatrix<std::complex<double>>& connection() const;
        const Eigen::SparseMatrix<double>& covariant(
            ops::laplacian::connection::CovariantCoupling coupling) const;
    };
    // mass()/dec()/hodge() added in Task C2
    LaplacianView laplacian() const { return LaplacianView{m_}; }
private:
    Manifold& m_;
};
```

- [ ] **Step 3: Implement cotan/graph + cache management** in `src/facets.cpp`:

```cpp
// in namespace nxr::manifold:
facet::OperatorsFacet Manifold::operators() { return facet::OperatorsFacet(*this); }

bool Manifold::isOperatorCached(OperatorId id) const {
    switch (id) {
        case OperatorId::LaplacianCotan:      return (bool)cacheLaplacianCotan_;
        case OperatorId::LaplacianGraph:      return (bool)cacheLaplacianGraph_;
        case OperatorId::LaplacianConnection: return (bool)cacheLaplacianConnection_;
        case OperatorId::LaplacianCovariant:  return (bool)cacheLaplacianCovariant_;
        case OperatorId::Dec:                 return (bool)decCache_;
        case OperatorId::MassLumped:          return (bool)cacheMassLumped_;
        case OperatorId::MassGalerkin:        return (bool)cacheMassGalerkin_;
    }
    return false;
}
void Manifold::releaseOperator(OperatorId id) {
    switch (id) {
        case OperatorId::LaplacianCotan:      cacheLaplacianCotan_.reset();      break;
        case OperatorId::LaplacianGraph:      cacheLaplacianGraph_.reset();      break;
        case OperatorId::LaplacianConnection: cacheLaplacianConnection_.reset(); break;
        case OperatorId::LaplacianCovariant:  cacheLaplacianCovariant_.reset();  break;
        case OperatorId::Dec:                 decCache_.reset();                 break;
        case OperatorId::MassLumped:          cacheMassLumped_.reset();          break;
        case OperatorId::MassGalerkin:        cacheMassGalerkin_.reset();        break;
    }
}

// in namespace nxr::manifold::facet:
const Eigen::SparseMatrix<double>& OperatorsFacet::LaplacianView::cotan() const {
    // Sources DIRECTLY from operatorGeometry()'s cotan Laplacian — does NOT build mass
    // (decoupled from assembleManifoldOperators, which fuses cotan+mass+normals).
    if (!m.isOperatorCached(OperatorId::LaplacianCotan)) {
        auto& geom = m.operatorGeometry();
        geom.requireCotanLaplacian();
        m.cacheLaplacianCotan_ =
            std::make_unique<Eigen::SparseMatrix<double>>(geom.cotanLaplacian);
    }
    return *m.cacheLaplacianCotan_;
}
const Eigen::SparseMatrix<double>& OperatorsFacet::LaplacianView::graph() const {
    if (!m.isOperatorCached(OperatorId::LaplacianGraph))
        m.cacheLaplacianGraph_ =
            std::make_unique<Eigen::SparseMatrix<double>>(ops::graphLaplacian(m));
    return *m.cacheLaplacianGraph_;
}
```

(Note: `LaplacianView` reads private cache members of `Manifold`. Make `facet::OperatorsFacet`
a `friend` of `Manifold`, OR add private setters. Simplest: add
`friend class facet::OperatorsFacet;` and `friend struct facet::OperatorsFacet::LaplacianView;`
to `Manifold`. Since nested-struct friendship is awkward, prefer giving `Manifold` thin private
mutators the views call — e.g. route through `m.cacheLaplacianCotan_` by making `OperatorsFacet`
a friend and accessing the unique_ptrs directly. Confirm the friend declaration compiles; if
nested-struct friendship is rejected, move the cache logic into `Manifold` methods
`buildCotanLaplacian_()` returning `const&` and have the view call those.)

**Implementation note for the agent:** to avoid nested-friend complexity, implement the cache
fill as private `Manifold` methods and have the views delegate:

```cpp
// compute.h private:  const Eigen::SparseMatrix<double>& cotanLaplacianCached_();
//                     const Eigen::SparseMatrix<double>& graphLaplacianCached_();
// facets.cpp: LaplacianView::cotan() { return m.cotanLaplacianCached_(); }
```

Pick whichever (friend or private-method) compiles cleanly; the private-method route is
recommended.

- [ ] **Step 4: Write the failing test** `test/test_operators_facet.cpp`:

```cpp
#include "nxr/facets.h"
#include <complex>
#include <cmath>
#include <iostream>
using namespace nxr::manifold;

static int g_failures = 0;
#define EXPECT(cond, msg) do { \
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; } \
    else { std::cout << "  [FAIL] " << msg << "\n"; ++g_failures; } } while (0)

static void icosphere(std::vector<double>& V, std::vector<int32_t>& F) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    V = {-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t,
          0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1};
    F = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2,
         10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5,
         2,4,11, 6,2,10, 8,6,7, 9,8,1};
}

static void testLaplacianCotanGraph() {
    std::cout << "\n=== operators: laplacian cotan/graph + independent cache ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    auto ops = m.operators();

    const auto& L = ops.laplacian().cotan();
    EXPECT(L.rows() == 12 && L.cols() == 12, "cotan Laplacian [12,12]");
    EXPECT(m.isOperatorCached(OperatorId::LaplacianCotan), "cotan cached after request");
    // INDEPENDENT cache: requesting cotan did NOT build mass
    EXPECT(!m.isOperatorCached(OperatorId::MassLumped), "cotan request did not build mass (decoupled)");

    const auto& G = ops.laplacian().graph();
    EXPECT(G.rows() == 12, "graph Laplacian [12,12]");
    // graph Laplacian diagonal = vertex degree (>0)
    EXPECT(G.coeff(0,0) > 0.5, "graph Laplacian diagonal = degree");

    // release drops the slot; re-request rebuilds
    m.releaseOperator(OperatorId::LaplacianCotan);
    EXPECT(!m.isOperatorCached(OperatorId::LaplacianCotan), "release(cotan) clears slot");
    (void)ops.laplacian().cotan();
    EXPECT(m.isOperatorCached(OperatorId::LaplacianCotan), "re-request rebuilds");
}

int main() {
    testLaplacianCotanGraph();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
```

Register in `CMakeLists.txt`:

```cmake
    add_executable(test_operators_facet test/test_operators_facet.cpp)
    target_link_libraries(test_operators_facet PRIVATE nxr_compute)
    add_test(NAME test_operators_facet COMMAND test_operators_facet)
```

- [ ] **Step 5: Build + run**

Run: `bash scripts/build.sh Release 2>&1 | tail -6 && ./build/test_operators_facet`
Expected: `ALL PASSED`.

- [ ] **Step 6: Commit**

```bash
git add include/nxr/facets.h src/facets.cpp include/nxr/compute.h test/test_operators_facet.cpp CMakeLists.txt
git commit -m "feat(operators): OperatorsFacet core (laplacian.cotan/graph) + independent lazy cache + release

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Task C2: `dec` bundle + `mass.{lumped,galerkin}` + `hodge.{h0,h1,h2,h1inv}`

**Files:** Modify `include/nxr/facets.h`, `src/facets.cpp`, `test/test_operators_facet.cpp`

- [ ] **Step 1: Extend `OperatorsFacet`** in `include/nxr/facets.h` (add inside the class):

```cpp
    struct MassView { Manifold& m;
        const Eigen::SparseMatrix<double>& lumped()   const;
        const Eigen::SparseMatrix<double>& galerkin() const;
    };
    struct HodgeView { Manifold& m;
        const Eigen::SparseMatrix<double>& h0()    const;
        const Eigen::SparseMatrix<double>& h1()    const;
        const Eigen::SparseMatrix<double>& h2()    const;
        const Eigen::SparseMatrix<double>& h1inv() const;
    };
    const ops::DECOperators& dec()  const;     // {d0,d1,hodge0..2,hodge1inv}
    MassView                 mass()  const { return MassView{m_}; }
    HodgeView                hodge() const { return HodgeView{m_}; }
```

- [ ] **Step 2: Implement** in `src/facets.cpp` (namespace `nxr::manifold::facet`). Use the
private-method pattern from C1 for the mass slots (add `massLumpedCached_()` /
`massGalerkinCached_()` private methods to `Manifold` mirroring cotan):

```cpp
// Manifold private methods (compute.h + facets.cpp):
const Eigen::SparseMatrix<double>& Manifold::massLumpedCached_() {
    if (!cacheMassLumped_) {
        auto& geom = operatorGeometry();
        geom.requireVertexLumpedMassMatrix();
        cacheMassLumped_ = std::make_unique<Eigen::SparseMatrix<double>>(geom.vertexLumpedMassMatrix);
    }
    return *cacheMassLumped_;
}
const Eigen::SparseMatrix<double>& Manifold::massGalerkinCached_() {
    if (!cacheMassGalerkin_) {
        auto& geom = operatorGeometry();
        geom.requireVertexGalerkinMassMatrix();
        cacheMassGalerkin_ = std::make_unique<Eigen::SparseMatrix<double>>(geom.vertexGalerkinMassMatrix);
    }
    return *cacheMassGalerkin_;
}

// facet bodies:
const ops::DECOperators& OperatorsFacet::dec() const { return m_.decOperators(); }
const Eigen::SparseMatrix<double>& OperatorsFacet::MassView::lumped()   const { return m.massLumpedCached_(); }
const Eigen::SparseMatrix<double>& OperatorsFacet::MassView::galerkin() const { return m.massGalerkinCached_(); }
const Eigen::SparseMatrix<double>& OperatorsFacet::HodgeView::h0()    const { return m.decOperators().hodge0; }
const Eigen::SparseMatrix<double>& OperatorsFacet::HodgeView::h1()    const { return m.decOperators().hodge1; }
const Eigen::SparseMatrix<double>& OperatorsFacet::HodgeView::h2()    const { return m.decOperators().hodge2; }
const Eigen::SparseMatrix<double>& OperatorsFacet::HodgeView::h1inv() const { return m.decOperators().hodge1Inverse; }
```

(Declare `massLumpedCached_()` / `massGalerkinCached_()` in the `Manifold` `private:` section of
compute.h. Confirm GC field names `vertexLumpedMassMatrix` / `vertexGalerkinMassMatrix` on the
`IntrinsicGeometryInterface` — they are, per compute.h:138-156.)

- [ ] **Step 3: Write the failing test** — add to `test/test_operators_facet.cpp`, call from `main()`:

```cpp
static void testMassDecHodge() {
    std::cout << "\n=== operators: dec/mass/hodge ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    auto ops = m.operators();

    EXPECT(ops.mass().lumped().rows() == 12, "mass.lumped [12,12]");
    EXPECT(ops.mass().lumped().nonZeros() == 12, "mass.lumped is diagonal (12 nnz)");
    EXPECT(ops.mass().galerkin().nonZeros() > 12, "mass.galerkin has off-diagonals");
    // lumped mass sums to total area
    double s = 0; auto Ml = ops.mass().lumped();
    for (int k = 0; k < Ml.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(Ml,k); it; ++it) s += it.value();
    EXPECT(std::abs(s - m.lightGeometry().totalArea) < 1e-9, "lumped mass sums to total area");

    const auto& dec = ops.dec();
    EXPECT(dec.d0.rows() == 30 && dec.d0.cols() == 12, "dec.d0 [E,V] = [30,12]");
    EXPECT(dec.d1.rows() == 20 && dec.d1.cols() == 30, "dec.d1 [F,E] = [20,30]");
    EXPECT(ops.hodge().h1().rows() == 30, "hodge.h1 [E,E] = [30,30]");
    // requesting mass.lumped only built the lumped slot, not galerkin (independent)
    Manifold m2(V.data(), 12, F.data(), 20);
    (void)m2.operators().mass().lumped();
    EXPECT(m2.isOperatorCached(OperatorId::MassLumped) && !m2.isOperatorCached(OperatorId::MassGalerkin),
           "mass.lumped did not build galerkin (independent)");
}
```

- [ ] **Step 4: Build + run**

Run: `bash scripts/build.sh Release 2>&1 | tail -6 && ./build/test_operators_facet`
Expected: `ALL PASSED`.

- [ ] **Step 5: Commit**

```bash
git add include/nxr/facets.h src/facets.cpp include/nxr/compute.h test/test_operators_facet.cpp
git commit -m "feat(operators): dec bundle + mass.{lumped,galerkin} + hodge.{h0,h1,h2,h1inv}

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Task C3: `laplacian.connection` (active gauge) + `laplacian.covariant`

**Files:** Modify `src/facets.cpp`, `test/test_operators_facet.cpp` (header already declares them in C1)

- [ ] **Step 1: Implement `connection()` + `covariant()`** in `src/facets.cpp`. They read the
**active gauge** (`m.activeGaugeType()` / `m.activeSingularities()`):

```cpp
// in namespace nxr::manifold::facet:
const Eigen::SparseMatrix<std::complex<double>>& OperatorsFacet::LaplacianView::connection() const {
    if (!m.isOperatorCached(OperatorId::LaplacianConnection)) {
        namespace cl = ops::laplacian::connection;
        cl::ConnectionLaplacianOptions o;
        o.domain = cl::ConnectionDomain::Vertex; o.nSym = 1;
        o.format = cl::ConnectionLaplacianFormat::Complex;
        cl::ConnectionLaplacian r =
            (m.activeGaugeType() == GaugeType::Trivial)
              ? cl::assembleTrivialConnectionLaplacian(m, m.activeSingularities(),
                                                       m.decOperators(), m.choleskyCache(), o)
              : cl::assembleConnectionLaplacian(m, o);
        m.setConnectionCache_(r.K_complex);   // private mutator (below)
    }
    return m.connectionCacheRef_();
}
const Eigen::SparseMatrix<double>& OperatorsFacet::LaplacianView::covariant(
        ops::laplacian::connection::CovariantCoupling coupling) const {
    if (!m.isOperatorCached(OperatorId::LaplacianCovariant)) {
        namespace cl = ops::laplacian::connection;
        const auto& K      = connection();                 // active-gauge K (builds/caches it)
        Eigen::MatrixXcd grid = m.gauge().grid();          // realized active-gauge frame
        const auto& cotanL = cotan();                      // builds/caches cotan
        m.setCovariantCache_(cl::assembleCovariantLaplacian(coupling, K, grid, cotanL));
    }
    return m.covariantCacheRef_();
}
```

Add the four private mutators/refs to `Manifold` (compute.h `private:` + facets.cpp):

```cpp
void Manifold::setConnectionCache_(Eigen::SparseMatrix<std::complex<double>> K) {
    cacheLaplacianConnection_ =
        std::make_unique<Eigen::SparseMatrix<std::complex<double>>>(std::move(K));
}
const Eigen::SparseMatrix<std::complex<double>>& Manifold::connectionCacheRef_() {
    return *cacheLaplacianConnection_;
}
void Manifold::setCovariantCache_(Eigen::SparseMatrix<double> C) {
    cacheLaplacianCovariant_ = std::make_unique<Eigen::SparseMatrix<double>>(std::move(C));
}
const Eigen::SparseMatrix<double>& Manifold::covariantCacheRef_() {
    return *cacheLaplacianCovariant_;
}
```

(Declare these four in `Manifold` `private:`. The covariant default coupling — pass
`CovariantCoupling::Ambient` from callers that want the default; the view requires an explicit
coupling argument, matching the spec's `coupling ∈ {product, ambient}`.)

- [ ] **Step 2: Write the failing test** — add to `test/test_operators_facet.cpp`, call from `main()`:

```cpp
#include <Eigen/Eigenvalues>
static double minEigC(const Eigen::SparseMatrix<std::complex<double>>& K) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Eigen::MatrixXcd(K));
    return es.eigenvalues().minCoeff();
}
static void testConnectionCovariant() {
    std::cout << "\n=== operators: connection + covariant (active gauge) ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);     // LC active
    auto ops = m.operators();
    namespace cl = ops::laplacian::connection;

    const auto& K = ops.laplacian().connection();
    EXPECT(K.rows() == 12 && K.cols() == 12, "connection L [12,12] complex");
    auto herm = (K - Eigen::SparseMatrix<std::complex<double>>(K.adjoint())).norm();
    EXPECT(herm < 1e-9, "LC connection L Hermitian");
    EXPECT(minEigC(K) > -1e-9, "LC connection L PSD on icosphere (Delaunay)");

    const auto& C = ops.laplacian().covariant(cl::CovariantCoupling::Ambient);
    EXPECT(C.rows() == 36 && C.cols() == 36, "ambient covariant [3V,3V] = [36,36]");
    EXPECT((C - Eigen::SparseMatrix<double>(C.transpose())).norm() < 1e-9, "covariant symmetric");

    // active-gauge sensitivity: trivial gauge yields a different connection L
    std::map<int,double> sing{{0,1.0},{3,1.0}};
    Manifold mt(V.data(), 12, F.data(), 20, sing);    // trivial active
    const auto& Kt = mt.operators().laplacian().connection();
    EXPECT((Kt - K).norm() > 1e-6, "trivial-gauge connection L differs from LC");
}
```

- [ ] **Step 3: Build + run**

Run: `bash scripts/build.sh Release 2>&1 | tail -6 && ./build/test_operators_facet`
Expected: `ALL PASSED`.

- [ ] **Step 4: Regression**

Run: `./build/test_connection_laplacian && ./build/test_geometry_bundle && ./build/test_intrinsic_delaunay`
Expected: all `ALL PASSED` (operators facet is additive).

- [ ] **Step 5: Commit**

```bash
git add include/nxr/facets.h src/facets.cpp include/nxr/compute.h test/test_operators_facet.cpp
git commit -m "feat(operators): laplacian.connection (active gauge) + laplacian.covariant

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Task C4: MEX `operators` string-dispatch command + MATLAB test

**Files:** Modify `bindings/mex/src/nxr_compute_mex.cpp`, Create `bindings/mex/test/test_operators_command.m`

- [ ] **Step 1: Add `cmdOperators`** in `bindings/mex/src/nxr_compute_mex.cpp` (near the other
`cmd*` handlers, before `mexFunction`). It dispatches on the string key(s) and returns a native
sparse. Reuse `getHolder`, `eigenSparseToMx`, `eigenComplexSparseToMx`:

```cpp
// nxr_compute('operators', h, family[, subtype])
//   ('operators',h,'laplacian','cotan'|'graph'|'connection'|'covariant')
//   ('operators',h,'mass','lumped'|'galerkin')
//   ('operators',h,'hodge','h0'|'h1'|'h2'|'h1inv')
//   ('operators',h,'dec')  -> struct {d0,d1}
void cmdOperators(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 3) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
        "operators: expected nxr_compute('operators', handle, family[, subtype]).");
    ContextHolder& h = getHolder(prhs[1]);
    auto& m = *h.ctx;                            // Manifold& (confirm holder field name)
    std::string family = getStringArg(prhs[2]);
    std::string sub = (nrhs >= 4) ? getStringArg(prhs[3]) : "";
    auto ops = m.operators();
    namespace cl = nxr::manifold::ops::laplacian::connection;

    if (family == "laplacian") {
        if      (sub == "cotan")      plhs[0] = eigenSparseToMx(ops.laplacian().cotan());
        else if (sub == "graph")      plhs[0] = eigenSparseToMx(ops.laplacian().graph());
        else if (sub == "connection") plhs[0] = eigenComplexSparseToMx(ops.laplacian().connection());
        else if (sub == "covariant")  plhs[0] = eigenSparseToMx(
                                          ops.laplacian().covariant(cl::CovariantCoupling::Ambient));
        else throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            "operators laplacian: subtype must be cotan|graph|connection|covariant.");
    } else if (family == "mass") {
        if      (sub == "lumped")   plhs[0] = eigenSparseToMx(ops.mass().lumped());
        else if (sub == "galerkin") plhs[0] = eigenSparseToMx(ops.mass().galerkin());
        else throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            "operators mass: subtype must be lumped|galerkin.");
    } else if (family == "hodge") {
        if      (sub == "h0")    plhs[0] = eigenSparseToMx(ops.hodge().h0());
        else if (sub == "h1")    plhs[0] = eigenSparseToMx(ops.hodge().h1());
        else if (sub == "h2")    plhs[0] = eigenSparseToMx(ops.hodge().h2());
        else if (sub == "h1inv") plhs[0] = eigenSparseToMx(ops.hodge().h1inv());
        else throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            "operators hodge: subtype must be h0|h1|h2|h1inv.");
    } else if (family == "dec") {
        const auto& dec = ops.dec();
        const char* f[] = {"d0","d1"};
        mxArray* s = mxCreateStructMatrix(1,1,2,f);
        mxSetField(s,0,"d0", eigenSparseToMx(dec.d0));
        mxSetField(s,0,"d1", eigenSparseToMx(dec.d1));
        plhs[0] = s;
    } else {
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            "operators: family must be laplacian|mass|hodge|dec.");
    }
}
```

(Confirm `h.ctx` is the `Manifold` (the holder field is named `ctx` per the holder struct in this
file). Confirm `nxr::core::Error` / `ErrorCode` is the type the file already throws — it catches
`nxr::core::Error` in `mexFunction`.)

- [ ] **Step 2: Register the command** in `mexFunction` dispatch (add alongside the other
`else if`s, ~line 1619):

```cpp
        else if (cmd == "operators")                   cmdOperators(nlhs, plhs, nrhs, prhs);
```

Add `operators` to the unknown-command help string list (~line 1629).

- [ ] **Step 3: Write the MATLAB test** `bindings/mex/test/test_operators_command.m`:

```matlab
function test_operators_command
fprintf('[test_operators_command] starting\n');
thisDir = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'mex not found'); addpath(hits(1).folder); clear nxr_compute

% icosphere, chi = 2
t = (1+sqrt(5))/2;
V = [-1 t 0; 1 t 0; -1 -t 0; 1 -t 0; 0 -1 t; 0 1 t; 0 -1 -t; 0 1 -t; t 0 -1; t 0 1; -t 0 -1; -t 0 1];
F = [1 12 6; 1 6 2; 1 2 8; 1 8 11; 1 11 12; 2 6 10; 6 12 5; 12 11 3; 11 8 7; 8 2 9; ...
     4 10 5; 4 5 3; 4 3 7; 4 7 9; 4 9 10; 5 10 6; 3 5 12; 7 3 11; 9 7 8; 10 9 2];
h = nxr_compute('create', V, F);

L = nxr_compute('operators', h, 'laplacian', 'cotan');
assert(issparse(L) && isequal(size(L),[12 12]), 'cotan [12,12] sparse');
M = nxr_compute('operators', h, 'mass', 'lumped');
assert(issparse(M) && nnz(M)==12, 'mass.lumped diagonal');

% generalized eigensolve in MATLAB on the exported (K, M) pair
kEig = 6;
ev = eigs(L, M, kEig, 'smallestabs');
assert(numel(ev)==kEig && all(abs(ev) < 1e6), 'eigs(L,M) runs on exported operators');
assert(min(real(ev)) > -1e-6, 'smallest eigenvalue ~ 0 (PSD on icosphere)');

K = nxr_compute('operators', h, 'laplacian', 'connection');
assert(~isreal(K) && isequal(size(K),[12 12]), 'connection L complex [12,12]');
assert(norm(K - K','fro') < 1e-9, 'connection L Hermitian');

dec = nxr_compute('operators', h, 'dec');
assert(isequal(size(dec.d0),[30 12]) && isequal(size(dec.d1),[20 30]), 'dec d0/d1 shapes');

% equals the existing opt-in bundle (single source of truth)
G = nxr_compute('geometry', h, struct('operators', true));
assert(max(max(abs(L - G.operators.laplacian))) < 1e-12, 'operators cotan == geometry.operators.laplacian');

nxr_compute('destroy', h);
fprintf('ALL TESTS PASSED: test_operators_command\n');
end
```

- [ ] **Step 4: Build + run via MATLAB MCP**

Run: `bash scripts/build.sh Release 2>&1 | tail -4`
Then run `bindings/mex/test/test_operators_command.m` via the MATLAB MCP `run_matlab_file`.
Expected: `ALL TESTS PASSED: test_operators_command`.
Also re-run `test_operators.m` and `test_bundle.m` (existing `.operators` opt-in unchanged) →
`ALL TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_operators_command.m
git commit -m "feat(mex): operators string-dispatch command (laplacian/mass/hodge/dec) for direct MATLAB export

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Part C — before you begin
Read `bindings/mex/src/nxr_compute_mex.cpp`: `getHolder` (~133), the holder struct (field `ctx`
is the `Manifold`), `eigenSparseToMx` / `eigenComplexSparseToMx` (marshal.h), and the
`buildGeometryOperators` builder (~1165) for the existing operator marshaling idiom. The new
command mirrors that marshaling but is keyed by string. Confirm the covariant default coupling
matches the existing `.operators` default (`Ambient`). The MATLAB `F` above is the icosphere with
1-based indices.

---

## Self-Review

**Spec coverage** (`2026-06-09-manifold-geometry-facets-design.md`):

| Spec section | Task |
|---|---|
| §2 six structures (topology/embedded/intrinsic/extrinsic/gauge/operators) | A1–A4, B1–B2, C1–C3 |
| §3 element-first facet contents | A1–A4 |
| §3 root aliases (vertexPositions, faces) | A1 |
| §4 C++ facet accessors on Manifold | A1–A4, B1–B2, C1 |
| §5 gauge patterns 1/2/3 + default-at-construction | B1 (state/ctor), B2 (realization) |
| §5 setGauge | B1 |
| §5 Gauss–Bonnet validation | B1 |
| §5b operators taxonomy (laplacian/dec/mass/hodge) | C1–C3 |
| §5b dual surface (typed C++ + MEX string) | C1–C3 (C++), C4 (MEX) |
| §5b independent lazy cache + release | C1 (cotan/mass cache + release test), C2 |
| §5b matched (K,M) eigensolve + PSD note | C4 (eigs(L,M) test) |
| §6 backward compatibility (additive) | A4/C3 regressions, C4 (existing bundle parity) |
| §7 intrinsic-Delaunay interaction | cotan/connection source from operatorGeometry (C1/C3) |

**Placeholder scan:** no TBD/TODO; every code step shows complete code. The two "pick whichever
compiles" notes (C1 friend-vs-private-method, B1 ctor delegation) give a concrete recommended
path plus the fallback — not placeholders.

**Type consistency:** `GaugeType`, `OperatorId`, `CovariantCoupling`, facet class names, and
method names (`grid()`, `cotan()`, `connection()`, `mass().lumped()`, `decOperators()`,
`isOperatorCached`, `releaseOperator`, `validateSingularities_`) are used identically across all
tasks. `eulerCharacteristic()` defined once (B1) and reused. The lazy `decOperators()` /
`choleskyCache()` introduced in B1 are reused by C2/C3.

**Known scope notes (documented, not gaps):**
- Part A `intrinsic` facet data slices from `meshGeometry` (embedded-sourced). The *operators*
  carry the Delaunay-swappable guarantee via `operatorGeometry()` (C1/C3); routing the light
  intrinsic *data* through `operatorGeometry()` under normalization is a Phase-3-adjacent
  follow-up and does not affect operator PSD.
- The three Parts are independently mergeable; if executed as separate branches, Part B/C depend
  on Part A's `facets.h` + the lazy caches existing.
