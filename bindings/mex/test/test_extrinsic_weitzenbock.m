function test_extrinsic_weitzenbock
fprintf('[test_extrinsic_weitzenbock] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found', mexext);
addpath(hits(1).folder); clear nxr_compute

[V, F] = local_icosahedron();
nV = size(V, 1);
h  = nxr_compute('create', V, F);

% ── basic structure ────────────────────────────────────────────
W = nxr_compute('operators', h, 'extrinsicWeitzenbock');   % real sparse [3V×3V]
assert(issparse(W),                                   'extrinsicWeitzenbock must be sparse');
assert(isreal(W),                                     'extrinsicWeitzenbock must be real');
assert(size(W, 1) == 3*nV && size(W, 2) == 3*nV,    'shape must be [3V×3V]');
assert(nnz(W) > 0,                                    'extrinsicWeitzenbock must have nonzeros');

% ── symmetry ───────────────────────────────────────────────────
assert(max(abs(W - W'), [], 'all') < 1e-9,            'extrinsicWeitzenbock must be symmetric');

nxr_compute('destroy', h);
fprintf('[test_extrinsic_weitzenbock] PASSED\n');
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
