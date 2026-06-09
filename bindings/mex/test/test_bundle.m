function test_bundle
fprintf('[test_bundle] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found', mexext);
addpath(hits(1).folder); clear nxr_compute

[V, F] = local_icosahedron();
nV = size(V,1);
h = nxr_compute('create', V, F);

opts = struct('singVerts', uint32([1;2]), 'singValues', [1;1], 'source', 'manual');
B = nxr_compute('bundle', h, 'levi-civita');
assert(isfield(B,'Topology') && isfield(B,'Geometry') && isfield(B,'Gauge'), 'three sub-structs');
assert(strcmp(B.Gauge.type,'levi-civita'), 'bundle gauge type');

% bundle sub-structs match the standalone commands
T = nxr_compute('topology', h);
G = nxr_compute('geometry', h);
assert(isequal(B.Topology.halfedge.twin, T.halfedge.twin), 'bundle Topology == topology');
assert(isequal(B.Geometry.vertex.grid, G.vertex.grid), 'bundle Geometry == geometry');

Ge2 = nxr_compute('gauge', h, 'levi-civita');
assert(isequal(B.Gauge.vertex.rotation, Ge2.vertex.rotation), 'bundle Gauge == gauge');

% ── leadfield round-trip (the deliverable) ──
rng(0); nSensors = 7;
c = B.Gauge.vertex.rotation .* B.Geometry.vertex.grid;   % realized frame (LC: rotation==1)
n = cross(real(c), imag(c), 2);
maxErr = 0;
for v = 1:nV
    Gv = randn(nSensors, 3);                  % unconstrained Cartesian gain block
    cv = c(v,:); nv = n(v,:);
    Ltan = Gv * cv.';                         % nSensors×1 complex
    Ln   = Gv * nv.';                         % nSensors×1 real
    J  = randn(1,3);
    z  = sum(cv .* J, 2);  jn = sum(nv .* J, 2);
    direct    = Gv * J.';
    intrinsic = real(conj(z) .* Ltan) + jn .* Ln;
    maxErr = max(maxErr, max(abs(direct - intrinsic)));
end
assert(maxErr < 1e-9, sprintf('leadfield round-trip exact (err=%.2e)', maxErr));

% ── frame inverse is exact (lossless rotation) ──
J  = randn(nV,3);
z  = sum(c .* J, 2);  jn = sum(n .* J, 2);
Jr = real(conj(z) .* c) + jn .* n;
assert(max(abs(Jr(:) - J(:))) < 1e-9, 'Cartesian -> intrinsic -> Cartesian is identity');

% ── trivial bundle: Gauss-Bonnet input valid, realized frame orthonormal ──
Bt = nxr_compute('bundle', h, 'trivial', opts);
assert(abs(sum(Bt.Gauge.singularity.indices) - 2) < 1e-12, 'sum indices == chi == 2');
ct = Bt.Gauge.vertex.rotation .* Bt.Geometry.vertex.grid;
assert(max(abs(sqrt(sum(real(ct).^2,2)) - 1)) < 1e-9, 'trivial realized frame unit');

nxr_compute('destroy', h);
fprintf('PASSED leadfield round-trip, err=%.2e\n', maxErr);
fprintf('ALL TESTS PASSED: test_bundle\n');
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
