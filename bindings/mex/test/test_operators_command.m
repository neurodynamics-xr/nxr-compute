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
