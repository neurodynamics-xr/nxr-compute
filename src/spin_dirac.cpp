// Spin Diracs on quaternion spinors per face (Hoffmann & Ye 2018; Ye et al. 2018) and their eigensolve on vertex spinors.
// Port of nxr-cortical-flow spinDirac.ts (reference) = Rheome operators.spin_dirac / spin_vertex_to_face.
#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace nxr::manifold::ops::spin {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// 4×4 row-major matrix of LEFT multiplication by the quaternion (a,b,c,d), [w,x,y,z]
static std::array<double, 16> leftMul(double a, double b, double c, double d) {
    return {a, -b, -c, -d,  b, a, -d, c,  c, d, a, -b,  d, -c, b, a};
}

static void requireClosed(Manifold& m, const char* who) {
    if (m.mesh().hasBoundary())
        throw Error(ErrorCode::ClosedMeshRequired, std::string(who) + ": the mesh must be closed", "spin Diracs are for closed, oriented surfaces");
}

static Eigen::Vector3d faceNormal(const VertexPositionGeometry& g, Face f) {
    Halfedge h = f.halfedge();
    Vector3 a = g.vertexPositions[h.vertex()], b = g.vertexPositions[h.next().vertex()], c = g.vertexPositions[h.next().next().vertex()];
    Vector3 x = cross(b - a, c - a);
    double l = norm(x);
    if (!(l > 0)) throw Error(ErrorCode::GeometryDegenerate, "spin: zero-area face " + std::to_string(f.getIndex()));
    return Eigen::Vector3d(x.x / l, x.y / l, x.z / l);
}

SpinDirac assemble(Manifold& m, Kind kind) {
    requireClosed(m, "spin::assemble");
    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    const int nF = m.nF();
    std::vector<Eigen::Vector3d> n(nF);
    SpinDirac out;
    out.H = Eigen::VectorXd::Zero(nF);
    out.area.resize(nF);
    for (Face f : mesh.faces()) {
        n[f.getIndex()] = faceNormal(geom, f);
        Halfedge h = f.halfedge();
        Vector3 a = geom.vertexPositions[h.vertex()], b = geom.vertexPositions[h.next().vertex()], c = geom.vertexPositions[h.next().next().vertex()];
        out.area(f.getIndex()) = 0.5 * norm(cross(b - a, c - a));
    }
    std::vector<Eigen::Triplet<double>> T;
    T.reserve(static_cast<size_t>(nF) * 3 * 16 + 4 * nF);
    for (Face f : mesh.faces()) {
        const int i = static_cast<int>(f.getIndex());
        for (Halfedge h : f.adjacentHalfedges()) {
            const int j = static_cast<int>(h.twin().face().getIndex());
            Vector3 ev = geom.vertexPositions[h.next().vertex()] - geom.vertexPositions[h.vertex()];
            const double len = norm(ev);
            const Eigen::Vector3d &ni = n[i], &nj = n[j];
            const Eigen::Vector3d cx = ni.cross(nj);
            const double sn = (cx.x() * ev.x + cx.y() * ev.y + cx.z() * ev.z) / len, cs = ni.dot(nj);
            const double th = sn / (1 + cs);                       // tan(θ/2), stable away from a fold-back (θ = π)
            const double s = 0.5 * (kind == Kind::IntrinsicYe ? 1.0 / std::sqrt(1 + th * th) : 1.0);
            const auto B = leftMul(len * th, ev.x, ev.y, ev.z);
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    if (B[4 * r + c] != 0.0) T.emplace_back(4 * i + r, 4 * j + c, s * B[4 * r + c]);
            out.H(i) += 0.5 * len * th;
        }
    }
    if (kind == Kind::Extrinsic)
        for (int i = 0; i < nF; ++i)
            for (int c = 0; c < 4; ++c) T.emplace_back(4 * i + c, 4 * i + c, -out.H(i));
    out.D.resize(4 * nF, 4 * nF);
    out.D.setFromTriplets(T.begin(), T.end());
    out.D.makeCompressed();
    return out;
}

Eigen::SparseMatrix<double> vertexToFace(Manifold& m, bool covariant) {
    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    const int nF = m.nF(), nV = m.nV();
    std::vector<Eigen::Vector3d> vn(nV, Eigen::Vector3d::Zero());
    for (Face f : mesh.faces()) {
        Halfedge h = f.halfedge();
        Vector3 a = geom.vertexPositions[h.vertex()], b = geom.vertexPositions[h.next().vertex()], c = geom.vertexPositions[h.next().next().vertex()];
        Vector3 x = cross(b - a, c - a);                           // area-weighted: |x| = 2·area
        for (Vertex v : f.adjacentVertices()) vn[v.getIndex()] += Eigen::Vector3d(x.x, x.y, x.z) / 2;
    }
    for (auto& v : vn) { double l = v.norm(); if (l > 0) v /= l; }
    std::vector<Eigen::Triplet<double>> T;
    T.reserve(static_cast<size_t>(nF) * 3 * 16);
    for (Face f : mesh.faces()) {
        const int fi = static_cast<int>(f.getIndex());
        const Eigen::Vector3d nf = faceNormal(geom, f);
        for (Vertex v : f.adjacentVertices()) {
            double q[4] = {1, 0, 0, 0};
            if (covariant) {
                const Eigen::Vector3d& nv = vn[v.getIndex()];
                const Eigen::Vector3d cx = nf.cross(nv);
                q[0] = 1 + nf.dot(nv); q[1] = -cx.x(); q[2] = -cx.y(); q[3] = -cx.z();
                const double l = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
                for (double& x : q) x /= l;
            }
            const auto B = leftMul(q[0], q[1], q[2], q[3]);
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    if (B[4 * r + c] != 0.0) T.emplace_back(4 * fi + r, 4 * static_cast<int>(v.getIndex()) + c, B[4 * r + c]);
        }
    }
    Eigen::SparseMatrix<double> A(4 * nF, 4 * nV);
    A.setFromTriplets(T.begin(), T.end());
    A.makeCompressed();
    return A;
}

}  // namespace nxr::manifold::ops::spin

namespace nxr::manifold::solve {

EigenResult eigenSpin(Manifold& m, ops::spin::Kind kind, int k, const CancellationToken&, const ProgressObserver&) {
    using SM = Eigen::SparseMatrix<double>;
    const int nV = m.nV(), nF = m.nF();
    if (k < 1 || k > 4 * nV - 1) throw Error(ErrorCode::EigensolveInvalidK, "eigenSpin: k out of range");
    ops::spin::SpinDirac sd = ops::spin::assemble(m, kind);
    SM A = ops::spin::vertexToFace(m, kind != ops::spin::Kind::Extrinsic);
    Eigen::VectorXd a4(4 * nF);
    for (int f = 0; f < nF; ++f) a4.segment(4 * f, 4).setConstant(sd.area(f));
    SM M(4 * nF, 4 * nF), Mi(4 * nF, 4 * nF);
    M = a4.asDiagonal(); Mi = a4.cwiseInverse().asDiagonal();
    SM At = A.transpose();
    SM DA = sd.D * A;
    SM K2 = At * DA, K1 = DA.transpose() * Mi * DA, G = At * M * A;
    auto sym = [](const SM& X) { SM Y = (X + SM(X.transpose())) * 0.5; Y.makeCompressed(); return Y; };
    K2 = sym(K2); K1 = sym(K1); G = sym(G);
    double trK1 = 0, trG = 0;
    for (int i = 0; i < K1.rows(); ++i) { trK1 += K1.coeff(i, i); trG += G.coeff(i, i); }
    const double eps = 1e-9 * trK1 / trG;                       // makes K1 + εG definite (K1 is only PSD)
    SM B = K1 + eps * G;
    // K2 ψ = ν (K1+εG) ψ, the k largest |ν| = the k nearest μ = 0. Right-ℍ multiplication commutes with D and A, so every
    // eigenvalue is a multiple of 4 — Lanczos from one start vector under-counts them: ask a quartet more, close the clusters.
    const int kReq = std::min(((k + 3) / 4) * 4 + 4, 4 * nV - 1);
    EigenResult r0 = eigenLargestMagn(K2, B, kReq, false);
    for (int c = 0; c < r0.k; ++c) r0.eigenvalues(c) = 1.0 / r0.eigenvalues(c);   // μ
    {   // ascending μ for the cluster pass
        std::vector<int> o(r0.k); std::iota(o.begin(), o.end(), 0);
        std::sort(o.begin(), o.end(), [&](int x, int y) { return r0.eigenvalues(x) < r0.eigenvalues(y); });
        EigenResult t = r0;
        for (int c = 0; c < r0.k; ++c) { t.eigenvalues(c) = r0.eigenvalues(o[c]); t.eigenvectors.col(c) = r0.eigenvectors.col(o[c]); }
        r0 = t;
    }
    EigenResult rc = reconstructDiracMultiplets(r0, G);                           // μ, ascending, clusters closed
    std::vector<int> byAbs(rc.k); std::iota(byAbs.begin(), byAbs.end(), 0);
    std::stable_sort(byAbs.begin(), byAbs.end(), [&](int x, int y) { return std::abs(rc.eigenvalues(x)) < std::abs(rc.eigenvalues(y)); });
    byAbs.resize(std::min<int>(k, rc.k));                                          // the k nearest 0
    std::sort(byAbs.begin(), byAbs.end());                                         // back to ascending μ
    EigenResult r; r.k = r.nConverged = static_cast<int>(byAbs.size());
    r.eigenvalues.resize(r.k); r.eigenvectors.resize(rc.eigenvectors.rows(), r.k);
    for (int c = 0; c < r.k; ++c) { r.eigenvalues(c) = 1.0 / rc.eigenvalues(byAbs[c]); r.eigenvectors.col(c) = rc.eigenvectors.col(byAbs[c]); }   // ν for the lines below
    EigenResult out;
    out.k = r.k; out.nConverged = r.nConverged;
    std::vector<int> idx(r.k);
    Eigen::VectorXd mu(r.k);
    for (int c = 0; c < r.k; ++c) { idx[c] = c; mu(c) = 1.0 / r.eigenvalues(c); }
    std::sort(idx.begin(), idx.end(), [&](int x, int y) { return mu(x) < mu(y); });
    out.eigenvalues.resize(r.k); out.eigenvectors.resize(4 * nV, r.k);
    for (int c = 0; c < r.k; ++c) {
        Eigen::VectorXd psi = r.eigenvectors.col(idx[c]);
        psi /= std::sqrt(psi.dot(G * psi));                    // ψᵀ AᵀMA ψ = 1: the face spinor Aψ has unit L²
        out.eigenvalues(c) = mu(idx[c]);
        out.eigenvectors.col(c) = psi;
    }
    return out;
}

}  // namespace nxr::manifold::solve
