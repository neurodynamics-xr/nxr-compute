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

} // namespace nxr::manifold::facet
