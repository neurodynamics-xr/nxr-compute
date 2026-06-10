function test_dirac_operator
fprintf('[test_dirac_operator] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found', mexext);
addpath(hits(1).folder); clear nxr_compute

[V, F] = local_icosahedron();
nV = size(V,1);
h = nxr_compute('create', V, F);

% τ=0 anchor: dirac(0) == kron(cotanL, I4).
% The C++ builds the 4V×4V block via 4*row+c, 4*col+c — each scalar weight
% becomes a 4×4 identity block, i.e. kron(cotanL, speye(4)).
L0 = nxr_compute('operators', h, 'dirac', 0);
Lc = nxr_compute('operators', h, 'laplacian', 'cotan');
anchor = kron(Lc, speye(4));
assert(norm(L0 - anchor, 'fro') < 1e-10, 'dirac(0) != kron(cotanL, I4)');
assert(isequal(size(L0), [4*nV, 4*nV]), 'dirac size wrong');

% Generalized eigenproblem. B must match L's vertex-interleaved (4*v+c) layout,
% so the Galerkin mass is kron(Mg, I4) — the SAME ordering as the cotan anchor
% above, NOT kron(I4, Mg). This is the canonical way to set up the Dirac
% eigenbasis in MATLAB: eigs(L, kron(Mg, speye(4))).
L = nxr_compute('operators', h, 'dirac', 0.5);
Mg = nxr_compute('operators', h, 'mass', 'galerkin');
B = kron(Mg, speye(4));
assert(norm(L - L', 'fro') < 1e-9, 'dirac(0.5) not symmetric');
d = eigs(L, B, 8, 'smallestabs');
assert(all(abs(imag(d)) < 1e-8), 'eigenvalues not real');
d = sort(real(d));
% Quaternionic structure ⇒ eigenvalues come in 4-fold multiplets.
assert(d(4) - d(1) < 1e-4 * (1 + abs(d(4))), 'first multiplet not 4-fold');
assert(d(8) - d(5) < 1e-4 * (1 + abs(d(8))), 'second multiplet not 4-fold');

% τ out of range errors.
threw = false;
try, nxr_compute('operators', h, 'dirac', 2.0); catch, threw = true; end
assert(threw, 'dirac(2.0) did not error');

nxr_compute('destroy', h);
fprintf('test_dirac_operator: ALL PASSED\n');
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
