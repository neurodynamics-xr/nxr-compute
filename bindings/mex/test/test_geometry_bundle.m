function test_geometry_bundle
fprintf('[test_geometry_bundle] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found', mexext);
addpath(hits(1).folder); clear nxr_compute

[V, F] = local_icosahedron();
nV = size(V,1); nF = size(F,1); nE = nV + nF - 2;
h = nxr_compute('create', V, F);
G = nxr_compute('geometry', h);

assert(isequal(size(G.vertex.grid), [nV 3]), 'vertex.grid nV x 3');
assert(~isreal(G.vertex.grid), 'vertex.grid is complex');
assert(isequal(size(G.vertex.curvature), [nV 1]), 'vertex.curvature nV x 1');
assert(~isreal(G.vertex.curvature), 'curvature complex');
assert(isequal(size(G.vertex.meanCurvature), [nV 1]), 'meanCurvature nV x 1');
assert(numel(G.edge.lengths) == nE, 'edge.lengths nE');
assert(numel(G.face.areas) == nF, 'face.areas nF');
assert(isequal(size(G.face.grid), [nF 3]), 'face.grid nF x 3');

e1 = real(G.vertex.grid); e2 = imag(G.vertex.grid);
assert(max(abs(sqrt(sum(e1.^2,2)) - 1)) < 1e-9, 'real(grid) unit');
assert(max(abs(sqrt(sum(e2.^2,2)) - 1)) < 1e-9, 'imag(grid) unit');
assert(max(abs(sum(e1.*e2,2))) < 1e-9, 'real perp imag');
nrm = cross(e1, e2, 2);
assert(max(abs(sqrt(sum(nrm.^2,2)) - 1)) < 1e-9, 'cross is unit normal');

assert(abs(G.totalArea - sum(G.face.areas)) < 1e-9, 'totalArea == sum faceAreas');
assert(abs(sum(G.vertex.dualAreas) - G.totalArea) < 1e-9, 'sum dualAreas == totalArea');

Kint = 2*pi - G.vertex.angleSums;
assert(abs(sum(Kint) - 4*pi) < 1e-6, 'sum angle defect == 4pi');

assert(max(abs(G.vertex.curvature)) < 0.5 * min(G.vertex.meanCurvature), 'deviatoric small');
Kext = G.vertex.meanCurvature.^2 - abs(G.vertex.curvature).^2;
assert(all(Kext > 0), 'extrinsic Gaussian > 0');

nxr_compute('destroy', h);
fprintf('ALL TESTS PASSED: test_geometry_bundle\n');
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
