function test_intrinsic_delaunay
fprintf('[test_intrinsic_delaunay] starting\n');
thisDir = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'mex not found'); addpath(hits(1).folder); clear nxr_compute

% non-Delaunay rhombus (long-diagonal split, 1-based)
V = [0 0 0; 2 0 0; 1 0.2 0; 1 -0.2 0];
F = [1 3 2; 1 2 4];

hRaw = nxr_compute('create', V, F);
hN   = nxr_compute('create', V, F, struct('intrinsicDelaunay', true));

Graw = nxr_compute('geometry', hRaw, struct('operators',true));
GN   = nxr_compute('geometry', hN,   struct('operators',true));
Lraw = Graw.operators.laplacian;   % cotan
LN   = GN.operators.laplacian;
assert(isequal(size(Lraw), size(LN)) && size(LN,1)==4, 'both cotan 4x4');

offdiag = @(L) max(max(triu(L,1)));   % largest off-diagonal entry
assert(offdiag(Lraw) > 1e-9, 'raw cotan has a negative weight (non-Delaunay)');
assert(offdiag(LN)   < 1e-9, 'normalized cotan: all weights >= 0 (Delaunay)');
assert(min(eig(full(LN))) > -1e-9, 'normalized cotan PSD');

% ambient covariant under normalization is symmetric PSD
GaN = nxr_compute('gauge', hN, 'levi-civita', struct('operators',true,'coupling','ambient'));
C = GaN.operators.covariantLaplacian;
assert(norm(C - C','fro') < 1e-9, 'covariant symmetric');
assert(min(eig(full(C))) > -1e-9, 'covariant PSD under normalization');

nxr_compute('destroy', hRaw); nxr_compute('destroy', hN);
fprintf('ALL TESTS PASSED: test_intrinsic_delaunay\n');
end
