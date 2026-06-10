function test_facets
fprintf('[test_facets] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found', mexext);
addpath(hits(1).folder); clear nxr_compute

[V, F] = local_icosahedron();
nV = size(V,1); nF = size(F,1);
nE = nV + nF - 2;   % Euler: V - E + F = 2 (closed genus-0)
h = nxr_compute('create', V, F);

% embedded
E = nxr_compute('embedded', h);
assert(isfield(E,'vertex') && isfield(E,'face'), 'embedded has vertex+face');
assert(isequal(size(E.vertex.position), [nV 3]), 'embedded vertex.position nV x3');
assert(max(abs(E.vertex.position(:) - V(:))) < 1e-9, 'embedded position == input V');
e1 = real(E.vertex.grid); e2 = imag(E.vertex.grid);
assert(max(abs(sqrt(sum(e1.^2,2))-1)) < 1e-6, 'e1 unit');
assert(max(abs(sqrt(sum(e2.^2,2))-1)) < 1e-6, 'e2 unit');
assert(max(abs(sum(e1.*e2,2))) < 1e-6, 'e1 perp e2');
assert(max(abs(E.vertex.normal - cross(e1,e2,2)),[],'all') < 1e-6, 'normal == e1 x e2');
assert(isequal(size(E.vertex.normal),[nV 3]), 'embedded vertex.normal nV x3');
assert(isequal(size(E.face.grid),[nF 3]), 'face grid nF x3');
assert(~isreal(E.face.grid), 'face.grid complex');

% intrinsic
I = nxr_compute('intrinsic', h);
assert(isfield(I,'vertex')&&isfield(I,'edge')&&isfield(I,'halfedge'), 'intrinsic groups');
assert(isequal(size(I.vertex.dualArea),[nV 1]), 'dualArea nV x1');
assert(all(I.vertex.dualArea > 0), 'dual areas positive');
assert(isequal(size(I.edge.length),[nE 1]), 'edge.length nE x1');
assert(isequal(size(I.edge.cotanWeight),[nE 1]), 'edge.cotanWeight nE x1');
assert(~isreal(I.halfedge.transportAlong), 'transportAlong complex');
assert(~isreal(I.halfedge.transportAcross), 'transportAcross complex');
nH = 3*nF;
assert(isequal(size(I.halfedge.cotanWeight),[nH 1]), 'halfedge cotanWeight nH x1');

% extrinsic
X = nxr_compute('extrinsic', h);
assert(isfield(X.vertex,'principalDir'), 'extrinsic has principalDir');
assert(isequal(size(X.vertex.principalDir),[nV 3]), 'principalDir nV x3');
assert(~isreal(X.vertex.curvature2RoSy), 'curvature2RoSy complex');
assert(isequal(size(X.edge.dihedralAngle),[nE 1]), 'dihedralAngle nE x1');

% ── grouped facets command ──
opts = struct('singVerts', uint32([1;2]), 'singValues', [1;1], 'source', 'manual');
Fc = nxr_compute('facets', h, 'trivial', opts);
assert(isequal(sort(fieldnames(Fc)), ...
    sort({'Topology';'Embedded';'Intrinsic';'Extrinsic';'Gauge'})), 'five groups');
assert(strcmp(Fc.Gauge.type,'trivial'), 'gauge type trivial');
assert(abs(sum(Fc.Gauge.singularity.indices) - 2) < 1e-12, 'Gauss-Bonnet chi==2');
% grouped sub-structs == standalone
assert(isequal(Fc.Embedded.vertex.grid, E.vertex.grid), 'facets Embedded == embedded');
assert(isequal(Fc.Intrinsic.edge.length, I.edge.length), 'facets Intrinsic == intrinsic');
assert(isequal(Fc.Extrinsic.vertex.meanCurvature, X.vertex.meanCurvature), 'facets Extrinsic == extrinsic');
T = nxr_compute('topology', h);
assert(isequal(Fc.Topology.halfedge.twin, T.halfedge.twin), 'facets Topology == topology');
Ga = nxr_compute('gauge', h, 'trivial', opts);
assert(isequal(Fc.Gauge.vertex.rotation, Ga.vertex.rotation), 'facets Gauge == gauge');

nxr_compute('destroy', h);
fprintf('ALL TESTS PASSED: test_facets\n');
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
