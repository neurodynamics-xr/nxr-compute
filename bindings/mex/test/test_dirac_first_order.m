function test_dirac_first_order
% Validates the first-order Dirac families exposed by the MEX 'operators' command:
%   nxr_compute('operators', h, 'diracD')      -> [4F x 4V] first-order D
%   nxr_compute('operators', h, 'diracFaceD')  -> [4V x 4F] first-order D~
% Anchors are independent MATLAB-side oracles (face areas / barycentric vertex
% dual areas computed here, not read back from the library).
fprintf('[test_dirac_first_order] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found', mexext);
addpath(hits(1).folder); clear nxr_compute

[V, F] = local_icosahedron();
nV = size(V,1); nF = size(F,1);
h = nxr_compute('create', V, F);

% Independent measures (MATLAB-side oracle).
e1 = V(F(:,2),:) - V(F(:,1),:);
e2 = V(F(:,3),:) - V(F(:,1),:);
faceArea = 0.5 * sqrt(sum(cross(e1,e2,2).^2, 2));        % [F x 1]
vDual = zeros(nV,1);                                     % barycentric dual = (1/3) Σ incident areas
for f = 1:nF
    vDual(F(f,:)) = vDual(F(f,:)) + faceArea(f)/3;
end
WF = kron(spdiags(faceArea, 0, nF, nF), speye(4));       % ⋆_F ⊗ I4  [4F x 4F]
WV = kron(spdiags(vDual,    0, nV, nV), speye(4));       % ⋆_V ⊗ I4  [4V x 4V]

%% ---- vertex-domain first-order D : [4F x 4V] ----
D = nxr_compute('operators', h, 'diracD');
assert(isequal(size(D), [4*nF, 4*nV]), 'diracD size != [4F, 4V]');
assert(issparse(D) && isreal(D), 'diracD must be real sparse');

% Headline: DᵀW_F D == dirac(1) (the squared/Galerkin family at τ=1).
E = nxr_compute('operators', h, 'dirac', 1.0);
assert(norm(D.' * WF * D - E, 'fro') < 1e-9, 'DᵀW_F D != dirac(1)');

% First-order property: D annihilates constant quaternionic fields (4 columns,
% one per constant unit quaternion). Telescoping cyclic normal differences.
U = kron(ones(nV,1), eye(4));                            % [4V x 4] constant fields
assert(max(abs(D * U), [], 'all') < 1e-9, 'D does not kill constant quaternion fields');

% Cache consistency: a repeat call returns the identical matrix.
assert(norm(D - nxr_compute('operators', h, 'diracD'), 'fro') == 0, 'diracD not stable across calls');

%% ---- face-domain (dual) first-order D~ : [4V x 4F] ----
Dt = nxr_compute('operators', h, 'diracFaceD');
assert(isequal(size(Dt), [4*nV, 4*nF]), 'diracFaceD size != [4V, 4F]');
assert(issparse(Dt) && isreal(Dt), 'diracFaceD must be real sparse');

% Headline: D~ᵀW_V D~ == diracFace(1).
Et = nxr_compute('operators', h, 'diracFace', 1.0);
assert(norm(Dt.' * WV * Dt - Et, 'fro') < 1e-9, 'D~ᵀW_V D~ != diracFace(1)');

% First-order property on faces: D~ kills constant face-quaternionic fields.
Uf = kron(ones(nF,1), eye(4));                           % [4F x 4] constant face fields
assert(max(abs(Dt * Uf), [], 'all') < 1e-9, 'D~ does not kill constant face fields');

nxr_compute('destroy', h);
fprintf('test_dirac_first_order: ALL PASSED\n');
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
