function test_operators
fprintf('[test_operators] starting\n');
thisDir = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'mex not found'); addpath(hits(1).folder); clear nxr_compute

[V,F] = local_icosahedron(); nV=size(V,1); nF=size(F,1); nE=nV+nF-2;
h = nxr_compute('create', V, F);

% light by default: no operators field
G0 = nxr_compute('geometry', h);
assert(~isfield(G0,'operators'), 'geometry light by default');
T0 = nxr_compute('topology', h);
assert(~isfield(T0,'operators'), 'topology light by default');
Gl0 = nxr_compute('gauge', h, 'levi-civita');
assert(~isfield(Gl0,'operators'), 'gauge light by default');

% topology operators
T = nxr_compute('topology', h, struct('operators',true));
assert(isfield(T,'operators'), 'topology.operators present');
L = T.operators.laplacian;
assert(issparse(L) && isequal(size(L),[nV nV]), 'graph L V×V sparse');
assert(max(abs(L*ones(nV,1))) < 1e-12, 'graph L zero row sums');
assert(nnz(L-L') == 0, 'graph L symmetric');
assert(isequal(size(T.operators.dec.d0),[nE nV]), 'd0 E×V');
assert(isequal(size(T.operators.dec.d1),[nF nE]), 'd1 F×E');
assert(nnz(T.operators.dec.d1 * T.operators.dec.d0) == 0, 'd1*d0 == 0');

% geometry operators
Gg = nxr_compute('geometry', h, struct('operators',true));
Lc = Gg.operators.laplacian;
assert(issparse(Lc) && isequal(size(Lc),[nV nV]), 'cotan V×V');
assert(max(abs(Lc*ones(nV,1))) < 1e-9, 'cotan zero row sums');
assert(isequal(size(Gg.operators.hodge.h1),[nE nE]), 'h1 E×E');
% cross-surface identity: cotan == d0' * h1 * d0
d0 = T.operators.dec.d0; h1 = Gg.operators.hodge.h1;
assert(max(max(abs(Lc - d0'*h1*d0))) < 1e-9, 'cotan == d0''*h1*d0');
assert(isequal(size(Gg.operators.mass.lumped),[nV nV]), 'mass.lumped V×V');
assert(isequal(size(Gg.operators.mass.galerkin),[nV nV]), 'mass.galerkin V×V');
assert(isequal(size(Gg.operators.hodge.h0),[nV nV]), 'h0 V×V');
assert(isequal(size(Gg.operators.hodge.h2),[nF nF]), 'h2 F×F');
assert(isequal(size(Gg.operators.hodge.h1inv),[nE nE]), 'h1inv E×E');

% gauge operators (levi-civita vs trivial differ; complex Hermitian)
Gl = nxr_compute('gauge', h, 'levi-civita', struct('operators',true));
Kl = Gl.operators.laplacian;
assert(~isreal(Kl) && isequal(size(Kl),[nV nV]), 'connection L complex V×V');
assert(norm(Kl - Kl','fro') < 1e-9, 'connection L Hermitian');
opts = struct('singVerts',uint32([1;2]),'singValues',[1;1],'operators',true);
Gt = nxr_compute('gauge', h, 'trivial', opts);
Kt = Gt.operators.laplacian;
assert(norm(Kt - Kt','fro') < 1e-9, 'trivial connection L Hermitian');
assert(norm(Kt - Kl,'fro') > 1e-6, 'trivial differs from levi-civita');

% bundle with operators == standalone with operators
B = nxr_compute('bundle', h, 'levi-civita', struct('operators',true));
assert(isequal(B.Topology.operators.laplacian, T.operators.laplacian), 'bundle topo ops match');
assert(isequal(B.Geometry.operators.laplacian, Gg.operators.laplacian), 'bundle geo ops match');
assert(isequal(B.Gauge.operators.laplacian, Gl.operators.laplacian), 'bundle gauge ops match');
Gg2 = nxr_compute('geometry', h, struct('operators',true));
assert(isequal(Gg2.operators.laplacian, Gg.operators.laplacian), 'operators deterministic/cached');

% covariant (3-frame) Laplacian
G = nxr_compute('geometry', h);                                % plain geometry — has vertex.grid
nVc = nV;
Lc3 = Gl.operators.covariantLaplacian;                        % default 'ambient', levi-civita
assert(issparse(Lc3) && isequal(size(Lc3),[3*nVc 3*nVc]), 'covariantLaplacian 3N×3N sparse');
assert(norm(Lc3 - Lc3','fro') < 1e-9, 'covariantLaplacian symmetric');

% product == blkdiag(real-expand(K), cotanL), using already-exposed operators
Gp = nxr_compute('gauge', h, 'levi-civita', struct('operators',true,'coupling','product'));
Lp3 = Gp.operators.covariantLaplacian;
Kc  = Gl.operators.laplacian;                                  % V×V complex (connection L)
Lcot= Gg.operators.laplacian;                                  % V×V cotan (from geometry ops)
ReK = real(Kc); ImK = imag(Kc);
D = [ ReK, -ImK, sparse(nVc,nVc);
      ImK,  ReK, sparse(nVc,nVc);
      sparse(nVc,nVc), sparse(nVc,nVc), Lcot ];
assert(norm(Lp3 - D, 'fro') < 1e-9, 'product == blkdiag(real-expand(K), cotanL)');

% ambient world-form == kron(I3, cotanL) via the realized frame
c = Gl.vertex.rotation .* G.vertex.grid;  % realized LC frame (rotation==1 for LC)
e1 = real(c); e2 = imag(c); nrm = cross(e1, e2, 2);
% block-diag frame Fbd (3N×3N), component-major [a;b;c]
Fbd = [ spdiags(e1(:,1),0,nVc,nVc), spdiags(e2(:,1),0,nVc,nVc), spdiags(nrm(:,1),0,nVc,nVc);
        spdiags(e1(:,2),0,nVc,nVc), spdiags(e2(:,2),0,nVc,nVc), spdiags(nrm(:,2),0,nVc,nVc);
        spdiags(e1(:,3),0,nVc,nVc), spdiags(e2(:,3),0,nVc,nVc), spdiags(nrm(:,3),0,nVc,nVc) ];
kronI3L = blkdiag(Lcot, Lcot, Lcot);
assert(norm(Fbd*Lc3*Fbd' - kronI3L, 'fro') < 1e-9, 'ambient world-form == kron(I3, cotanL)');
assert(norm(Lc3 - Lp3,'fro') > 1e-6, 'ambient differs from product');

nxr_compute('destroy', h);
fprintf('ALL TESTS PASSED: test_operators\n');
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
