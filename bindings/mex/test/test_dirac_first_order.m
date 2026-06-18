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

%% ---- vertex-domain first-order INTRINSIC D_int : [4F x 4V] ----
% The immersion/edge-based root (vertex positions instead of the Gauss map): the
% spin-connection Dirac. Its W_F-Galerkin square is the intrinsic Dirac², whose
% SCALAR (w-w) block is exactly the cotan Laplacian (the Crane property).
Dint = nxr_compute('operators', h, 'diracIntrinsicD');
assert(isequal(size(Dint), [4*nF, 4*nV]), 'diracIntrinsicD size != [4F, 4V]');
assert(issparse(Dint) && isreal(Dint), 'diracIntrinsicD must be real sparse');

Lint = Dint.' * WF * Dint;                               % intrinsic Dirac² [4V x 4V]
Lw   = Lint(1:4:end, 1:4:end);                           % scalar (w-w) block [V x V]
Kc   = nxr_compute('operators', h, 'laplacian', 'cotan');
assert(norm(Lw - Kc, 'fro') / norm(Kc, 'fro') < 1e-9, ...
    'intrinsic Dirac^2 scalar part != cotan Laplacian');

% First-order property: D_int kills constant quaternionic fields (same U as above).
assert(max(abs(Dint * U), [], 'all') < 1e-9, 'D_int does not kill constant quaternion fields');

% Cache consistency.
assert(norm(Dint - nxr_compute('operators', h, 'diracIntrinsicD'), 'fro') == 0, ...
    'diracIntrinsicD not stable across calls');

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

%% ---- face-domain (dual) INTRINSIC first-order D~_int : [4V x 4F] ----
% Centroid (immersion) dual of D~ (face centroids instead of the Gauss map),
% mirroring diracIntrinsicD on the vertex side.
DtI = nxr_compute('operators', h, 'diracFaceIntrinsicD');
assert(isequal(size(DtI), [4*nV, 4*nF]), 'diracFaceIntrinsicD size != [4V, 4F]');
assert(issparse(DtI) && isreal(DtI), 'diracFaceIntrinsicD must be real sparse');
assert(max(abs(DtI * Uf), [], 'all') < 1e-9, 'D~_int does not kill constant face fields');
assert(norm(DtI - nxr_compute('operators', h, 'diracFaceIntrinsicD'), 'fro') == 0, ...
    'diracFaceIntrinsicD not stable across calls');

%% ---- face-native scalar gradient gradFace [3F x F] + Laplacian lapFace [F x F] ----
% Barycentric dual-mesh gradient of a per-face scalar (Green-Gauss) and its Galerkin
% face Laplacian. The dual of the vertex FEM gradient + cotan Laplacian.
G  = nxr_compute('operators', h, 'gradFace');
Kf = nxr_compute('operators', h, 'lapFace');
assert(isequal(size(G),  [3*nF, nF]), 'gradFace size != [3F, F]');
assert(isequal(size(Kf), [nF,   nF]), 'lapFace size != [F, F]');
assert(issparse(G) && isreal(G) && issparse(Kf) && isreal(Kf), 'gradFace/lapFace must be real sparse');
% constant precision + symmetry
assert(norm(G * ones(nF,1)) < 1e-10, 'gradFace does not annihilate constants');
assert(norm(Kf - Kf.', 'fro') < 1e-10, 'lapFace not symmetric');
% single source of truth: lapFace == gradFace' W_F gradFace  (W_F = area on each of 3 comps)
WF3 = kron(spdiags(faceArea, 0, nF, nF), speye(3));      % [3F x 3F]
assert(norm(Kf - G.' * WF3 * G, 'fro') < 1e-9 * norm(Kf, 'fro'), 'lapFace != gradFace'' W_F gradFace');
% tangency: each per-face gradient vector is perpendicular to its face normal
faceN = cross(e1, e2, 2); faceN = faceN ./ sqrt(sum(faceN.^2,2));   % [F x 3]
psi   = randn(nF,1);  gv = reshape(G*psi, 3, [])';                  % [F x 3]
assert(max(abs(sum(gv .* faceN, 2))) < 1e-9, 'gradFace output not tangent');
% kernel is 1-dim (constants) on the closed icosahedron
assert(abs(eigs(Kf, 1, 'smallestabs')) < 1e-9, 'lapFace smallest eig != 0');
assert(norm(G - nxr_compute('operators', h, 'gradFace'), 'fro') == 0, 'gradFace not stable across calls');

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
