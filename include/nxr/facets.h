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

// NOTE (intrinsic-Delaunay scope): the intrinsic facet DATA below slices from
// the embedded-sourced light geometry bundle, so under intrinsicDelaunay=true it
// is NOT swapped to the Delaunay metric — it stays consistent with the input
// embedding. The intrinsic OPERATORS (operators().laplacian().cotan(), mass) DO
// source from operatorGeometry() and are certified-PSD under normalization. So
// edge().cotanWeight() here can differ from the weights backing the cotan
// operator when normalized. Routing this data through operatorGeometry() is
// deferred to the fully-intrinsic Phase 3.
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

class GaugeFacet {
public:
    GaugeFacet(Manifold& m, GaugeType type, std::map<int,double> singularities)
        : m_(m), type_(type), sing_(std::move(singularities)) {}
    GaugeType type() const { return type_; }
    const std::map<int,double>& singularities() const { return sing_; }
    // Realized per-vertex frame in this gauge: [nV,3] complex c = e1 + i e2.
    // Levi-Civita/Euclidean: the raw vertex grid. Trivial: rotation .* grid —
    // the combed frame (real part == the trivial parallel field). See
    // GaugeRotations in compute.h for the convention.
    Eigen::MatrixXcd grid() const;
    // Realized per-face frame in this gauge: [nF,3] complex. Same combing
    // convention as grid(), via the dual-graph face rotations.
    Eigen::MatrixXcd faceGrid() const;
private:
    Manifold& m_;
    GaugeType type_;
    std::map<int,double> sing_;
};

class OperatorsFacet {
public:
    explicit OperatorsFacet(Manifold& m) : m_(m) {}

    // Views hold Manifold& (not OperatorsFacet&) so a stored view
    // (auto v = m.operators().laplacian()) does not dangle when the
    // OperatorsFacet temporary expires. Manifold befriends OperatorsFacet,
    // and per CWG 45 its nested views inherit that access to the private
    // *Cached_() helpers.
    struct LaplacianView {
        Manifold& m;
        const Eigen::SparseMatrix<double>& cotan() const;   // real, intrinsic (directly from GC cotan cache)
        const Eigen::SparseMatrix<double>& graph() const;   // real, topological

        // Declared; implemented in Task C3.
        const Eigen::SparseMatrix<std::complex<double>>& connection() const;
        // Default coupling is Ambient — the full 3D-frame embedded covariant.
        // Product (intrinsic connection on the tangent block) is opt-in.
        const Eigen::SparseMatrix<double>& covariant(
            ops::laplacian::connection::CovariantCoupling coupling =
                ops::laplacian::connection::CovariantCoupling::Ambient) const;
    };

    // ── Task C2: dec / mass / hodge ──────────────────────────────────────────
    // Views hold Manifold& m (not OperatorsFacet&) — same C1 pattern;
    // a stored view (auto v = m.operators().mass()) does not dangle.

    struct MassView {
        Manifold& m;
        const Eigen::SparseMatrix<double>& lumped()   const;
        const Eigen::SparseMatrix<double>& galerkin() const;
    };

    struct HodgeView {
        Manifold& m;
        const Eigen::SparseMatrix<double>& h0()    const;
        const Eigen::SparseMatrix<double>& h1()    const;
        const Eigen::SparseMatrix<double>& h2()    const;
        const Eigen::SparseMatrix<double>& h1inv() const;
    };

    // dec(): returns the full DECOperators bundle {d0,d1,hodge0..2,hodge1Inverse}.
    const ops::DECOperators& dec()  const;
    MassView  mass()  const { return MassView{m_}; }
    HodgeView hodge() const { return HodgeView{m_}; }

    LaplacianView laplacian() const { return LaplacianView{m_}; }

    // gradient3D(): the covariant gradient operator G (3E x 3N), cached. Artifact-free
    // covariant difference of a 3-vector cortical field (differential::covariantGradient).
    const Eigen::SparseMatrix<double>& gradient3D() const;

    // dirac(tau): the relative-Dirac family L(τ) = (1−τ)(cotanL⊗I₄) + τ·E, a
    // [4V×4V] real symmetric sparse matrix, returned BY VALUE (τ-dependent blend).
    // τ=0 ⇒ block cotan-Laplacian; τ=1 ⇒ pure relative Dirac E. τ ∈ [0,1].
    Eigen::SparseMatrix<double> dirac(double tau) const;

    // diracFace(tau): the FACE-domain (dual) relative-Dirac family
    // L̃(τ) = (1−τ)(K̃⊗I₄) + τ·Ẽ, a [4F×4F] real symmetric sparse matrix, by value.
    // τ=0 ⇒ DEC 2-form Laplacian ⊗ I₄; τ=1 ⇒ pure extrinsic face Dirac Ẽ. τ ∈ [0,1].
    Eigen::SparseMatrix<double> diracFace(double tau) const;

    // diracD(): the first-order (rectangular) extrinsic Dirac operator D [4F×4V],
    // cached, returned by const-ref (τ-free, geometry-only). Its area-weighted
    // Galerkin square DᵀW_F D is the τ=1 block of dirac(τ).
    const Eigen::SparseMatrix<double>& diracD() const;

    // diracFaceD(): the first-order FACE-domain (dual) Dirac operator D̃ [4V×4F],
    // cached, by const-ref. Its Galerkin square D̃ᵀW_V D̃ is the τ=1 block of
    // diracFace(τ). Closed-mesh v1: throws on an open boundary (as diracFace).
    const Eigen::SparseMatrix<double>& diracFaceD() const;

    // diracIntrinsicD(): the first-order INTRINSIC Dirac operator D_int [4F×4V],
    // cached, by const-ref (τ-free, geometry-only). Built from the immersion (edge
    // vectors) rather than the Gauss map; its area-weighted Galerkin square is the
    // intrinsic Dirac² (the spin-connection root, scalar part = cotan Laplacian).
    const Eigen::SparseMatrix<double>& diracIntrinsicD() const;

    // diracFaceIntrinsicD(): the first-order INTRINSIC face Dirac D̃_int [4V×4F],
    // cached, by const-ref. The centroid (immersion) dual of diracFaceD (face centroids
    // instead of the Gauss map), mirroring diracIntrinsicD on the vertex side. Closed-mesh v1.
    const Eigen::SparseMatrix<double>& diracFaceIntrinsicD() const;

    // gradFace(): barycentric dual-mesh gradient of a per-face scalar [3F×F], cached.
    // Green–Gauss / DEC; annihilates constants, output tangent to each face. The dual
    // of the vertex FEM gradient. Closed-mesh v1.
    const Eigen::SparseMatrix<double>& gradFace() const;
    // lapFace(): face Laplacian K̃ = gradFaceᵀ ⋆_F gradFace [F×F], cached. Built from
    // the same gradFace (never drift); symmetric PSD; kernel = constants (genus-0).
    const Eigen::SparseMatrix<double>& lapFace() const;

private:
    Manifold& m_;
};

} // namespace nxr::manifold::facet
