function test_topology
fprintf('[test_topology] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found', mexext);
addpath(hits(1).folder); clear nxr_compute

[V, F] = local_icosahedron();
nV = size(V,1); nF = size(F,1); nE = nV + nF - 2;   % Euler: V - E + F = 2
h = nxr_compute('create', V, F);
T = nxr_compute('topology', h);

assert(T.schemaVersion == 1, 'schemaVersion');
assert(T.vertex.count == nV, 'vertex.count');
assert(T.face.count   == nF, 'face.count');
assert(T.edge.count   == nE, 'edge.count');
assert(T.corner.count == 3*nF, 'corner.count == 3F');
assert(T.halfedge.count == 2*nE, 'halfedge.count == 2E');

H = T.halfedge.count;
assert(all(T.halfedge.twin >= 1 & T.halfedge.twin <= H), 'twin in 1..H');
assert(all(T.halfedge.next >= 1 & T.halfedge.next <= H), 'next in 1..H');
assert(all(T.halfedge.vertex >= 1 & T.halfedge.vertex <= nV), 'vertex in 1..nV');
assert(all(T.halfedge.edge >= 1 & T.halfedge.edge <= nE), 'edge in 1..nE');
assert(all(T.halfedge.face >= 1 & T.halfedge.face <= nF), 'face in 1..nF (closed)');
assert(isa(T.halfedge.twin, 'uint32'), 'indices are uint32');
assert(islogical(T.halfedge.orientation), 'orientation logical');

tw = double(T.halfedge.twin);
assert(isequal(tw(tw), (1:H)'), 'twin is an involution');
nx = double(T.halfedge.next);
assert(isequal(nx(nx(nx)), (1:H)'), 'next^3 == identity (triangles)');

nxr_compute('destroy', h);
fprintf('ALL TESTS PASSED: test_topology\n');
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
