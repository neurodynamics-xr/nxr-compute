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
