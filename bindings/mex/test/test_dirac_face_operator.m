function test_dirac_face_operator
fprintf('[test_dirac_face_operator] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found', mexext);
addpath(hits(1).folder); clear nxr_compute

[V, F] = local_icosahedron();
nF = size(F,1);
h = nxr_compute('create', V, F);

% tau=0 anchor: diracFace(0) == kron(K~, I4), K~ = d1 * h1inv * d1'.
decs  = nxr_compute('operators', h, 'dec');               % struct .d0 .d1
h1inv = nxr_compute('operators', h, 'hodge', 'h1inv');
Ktilde = decs.d1 * h1inv * decs.d1';
anchor = kron(Ktilde, speye(4));
L0 = nxr_compute('operators', h, 'diracFace', 0);
assert(isequal(size(L0), [4*nF, 4*nF]), 'diracFace size wrong');
assert(norm(L0 - anchor, 'fro') < 1e-9, 'diracFace(0) != kron(K~, I4)');

% tau=0.5 symmetric; real eigenvalues; 4-fold multiplets. Generalized problem against
% the face-area mass B~ = diag(A_f)xI4. L~(0.5) has an exact 4-fold zero, so we
% shift-invert at a small sigma (sigma=-1e-8, matching the native solver) rather
% than 'smallestabs' on the singular matrix — finds the same low modes, no warning.
L = nxr_compute('operators', h, 'diracFace', 0.5);
assert(norm(L - L', 'fro') < 1e-9, 'diracFace(0.5) not symmetric');
v1 = V(F(:,1),:); v2 = V(F(:,2),:); v3 = V(F(:,3),:);
Af = 0.5 * sqrt(sum(cross(v2-v1, v3-v1, 2).^2, 2));        % per-face areas from coords
B  = kron(spdiags(Af, 0, nF, nF), speye(4));
ev = eigs(L, B, 8, -1e-8);                                 % single solve, shifted off the zero
assert(all(abs(imag(ev)) < 1e-8), 'eigenvalues not real');
d  = sort(real(ev));
assert(d(4) - d(1) < 1e-4*(1+abs(d(4))), 'first multiplet not >=4-fold');
assert(d(8) - d(5) < 1e-4*(1+abs(d(8))), 'second multiplet not >=4-fold');

% tau out of range errors.
threw = false;
try, nxr_compute('operators', h, 'diracFace', 2.0); catch, threw = true; end
assert(threw, 'diracFace(2.0) did not error');

nxr_compute('destroy', h);
fprintf('test_dirac_face_operator: ALL PASSED\n');
end

function [V, F] = local_icosahedron()
t = (1 + sqrt(5)) / 2;
V = [-1 t 0; 1 t 0; -1 -t 0; 1 -t 0; 0 -1 t; 0 1 t; ...
      0 -1 -t; 0 1 -t; t 0 -1; t 0 1; -t 0 -1; -t 0 1];
V = V ./ sqrt(sum(V.^2, 2));
F = [1 12 6; 1 6 2; 1 2 8; 1 8 11; 1 11 12; ...
     2 6 10; 6 12 5; 12 11 3; 11 8 7; 8 2 9; ...
     4 10 5; 4 5 3; 4 3 7; 4 7 9; 4 9 10; ...
     5 10 6; 3 5 12; 7 3 11; 9 7 8; 10 9 2];
end
