function test_vertex_frames
% test_vertex_frames — the per-vertex tangent-frame export ('vertexFrames').
% Raw-mex test on the icosahedron fixture (no Brainstorm, no +nxr): asserts the
% frame is [nV x 3], unit-length, orthonormal, and right-handed (e1 x e2 = n).
% NOTE: this checks geometric validity only. The gauge ROTATION (that e1 aligns
% with the connection Laplacian's angle-0 axis) is verified in the M3 phase-engine
% plan via a decoded-Fiedler vs smoothVertex cross-check.
fprintf('[test_vertex_frames] starting\n');

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found under %s/build', mexext, repoRoot);
addpath(hits(1).folder);
clear nxr_compute   % drop any previously-loaded copy so the fresh build resolves

[V, F] = local_icosahedron();
nV = size(V, 1);
h  = nxr_compute('create', V, F);

vf = nxr_compute('vertexFrames', h);

assert(isequal(size(vf.e1),      [nV 3]), 'e1 must be nV x 3');
assert(isequal(size(vf.e2),      [nV 3]), 'e2 must be nV x 3');
assert(isequal(size(vf.normals), [nV 3]), 'normals must be nV x 3');

n1 = sqrt(sum(vf.e1.^2, 2));
n2 = sqrt(sum(vf.e2.^2, 2));
nn = sqrt(sum(vf.normals.^2, 2));
assert(max(abs(n1 - 1)) < 1e-9, 'e1 must be unit length');
assert(max(abs(n2 - 1)) < 1e-9, 'e2 must be unit length');
assert(max(abs(nn - 1)) < 1e-9, 'normals must be unit length');

assert(max(abs(sum(vf.e1 .* vf.e2,      2))) < 1e-9, 'e1 . e2 must be 0');
assert(max(abs(sum(vf.e1 .* vf.normals, 2))) < 1e-9, 'e1 . n must be 0');
assert(max(abs(sum(vf.e2 .* vf.normals, 2))) < 1e-9, 'e2 . n must be 0');

cr = cross(vf.e1, vf.e2, 2);
assert(max(max(abs(cr - vf.normals))) < 1e-9, 'e1 x e2 must equal n (right-handed)');

% Determinism
vf2 = nxr_compute('vertexFrames', h);
assert(isequal(vf.e1, vf2.e1) && isequal(vf.e2, vf2.e2) && isequal(vf.normals, vf2.normals), ...
    'vertexFrames must be deterministic');

nxr_compute('destroy', h);
fprintf('PASSED: per-vertex frames [%d x 3], orthonormal, right-handed, deterministic.\n', nV);
fprintf('ALL TESTS PASSED: test_vertex_frames\n');
end


function [V, F] = local_icosahedron()
% Unit icosahedron (12 vertices, 20 faces).
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
