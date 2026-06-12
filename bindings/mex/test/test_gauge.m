function test_gauge
fprintf('[test_gauge] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found', mexext);
addpath(hits(1).folder); clear nxr_compute

[V, F] = local_icosahedron();
nV = size(V,1);
nF = size(F,1);
h = nxr_compute('create', V, F);

Ge = nxr_compute('gauge', h, 'euclidean');
assert(strcmp(Ge.type,'euclidean'), 'type euclidean');
assert(numel(Ge.vertex.rotation) == nV, 'rotation nV');
assert(max(abs(Ge.vertex.rotation - 1)) < 1e-12, 'euclidean rotation == 1');
assert(Ge.schemaVersion == 1, 'gauge schemaVersion == 1');
assert(numel(Ge.face.rotation) == nF, 'face.rotation nF');
assert(max(abs(Ge.face.rotation - 1)) < 1e-12, 'euclidean face rotation == 1');
assert(isempty(Ge.singularity.vertices), 'euclidean has no singularities');
assert(isempty(Ge.singularity.indices), 'euclidean has no singularity indices');

Gl = nxr_compute('gauge', h, 'levi-civita');
assert(max(abs(Gl.vertex.rotation - 1)) < 1e-12, 'levi-civita rotation == 1');

opts = struct('singVerts', uint32([1;2]), 'singValues', [1;1], 'source', 'manual');
Gt = nxr_compute('gauge', h, 'trivial', opts);
assert(strcmp(Gt.type,'trivial'), 'type trivial');
assert(max(abs(abs(Gt.vertex.rotation) - 1)) < 1e-9, 'trivial rotation unit modulus');
assert(numel(Gt.face.rotation) == nF, 'trivial face rotation nF');
assert(max(abs(abs(Gt.face.rotation) - 1)) < 1e-9, 'trivial face rotation unit modulus');
assert(numel(Gt.singularity.vertices) == 2, 'two singularities');
assert(all(Gt.singularity.vertices == uint32([1;2])), 'singularity verts 1-based round-trip');
assert(abs(sum(Gt.singularity.indices) - 2) < 1e-12, 'sum indices == chi == 2');
assert(max(abs(Gt.vertex.rotation - 1)) > 1e-6, 'trivial differs from identity');

G = nxr_compute('geometry', h);
c = Gt.vertex.rotation .* G.vertex.grid;        % broadcast V×1 over V×3
e1 = real(c); e2 = imag(c);
assert(max(abs(sqrt(sum(e1.^2,2)) - 1)) < 1e-9, 'realized real(c) unit');
assert(max(abs(sum(e1.*e2,2))) < 1e-9, 'realized real perp imag');

% ── combing anchor: rotation .* grid realizes the trivial parallel field ──
% real(rotation .* grid) must equal directionField's parallel field on both
% domains (vertexVectors / directionVectors) — the gauge and the direction
% field integrate the same connection 1-form phi.
D = nxr_compute('directionField', h, opts.singVerts, opts.singValues);
errV = max(vecnorm(real(c) - D.vertexVectors, 2, 2));
assert(errV < 1e-9, sprintf('vertex comb == parallel field (err=%.2e)', errV));
cf = Gt.face.rotation .* G.face.grid;
errF = max(vecnorm(real(cf) - D.directionVectors, 2, 2));
assert(errF < 1e-9, sprintf('face comb == parallel field (err=%.2e)', errF));

nxr_compute('destroy', h);
fprintf('ALL TESTS PASSED: test_gauge\n');
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
