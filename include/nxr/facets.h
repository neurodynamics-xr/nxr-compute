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
    // Levi-Civita/Euclidean: the raw vertex grid. Trivial: exp(i phi_v) .* grid.
    Eigen::MatrixXcd grid() const;
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
        const Eigen::SparseMatrix<double>& covariant(
            ops::laplacian::connection::CovariantCoupling coupling) const;
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

private:
    Manifold& m_;
};

} // namespace nxr::manifold::facet
