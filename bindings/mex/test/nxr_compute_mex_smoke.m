% nxr_compute_mex_smoke.m — verifies nxr_compute.mexw64 against the same icosahedron
% fixture used by the C++ test_eigen.exe. Mirrors the gpjs MATLAB
% bindings smoke pattern: build a known mesh, exercise each command,
% assert outputs are sensible.
%
% Run from MATLAB (or via the matlab MCP tool):
%   cd <repo>/native/mex/test
%   nxr_compute_mex_smoke

clear; clc;
fprintf('[nxr_compute_mex] starting smoke test\n');

% Locate the freshly-built MEX (relative to this script). The build
% script writes to <repo>/build/Release; this file lives three levels
% deep at bindings/mex/test/, so the artifact resolves at
% ../../../build/Release.
thisDir = fileparts(mfilename('fullpath'));
mexDir  = fullfile(thisDir, '..', '..', '..', 'build', 'Release');
addpath(mexDir);

assert(exist(fullfile(mexDir, 'nxr_compute.mexw64'), 'file') == 3, ...
    'nxr_compute.mexw64 not found in %s', mexDir);

% ── Sanity: version string ───────────────────────────────────
v = nxr_compute('version');
fprintf('  version: %s\n', v);
assert(startsWith(v, 'nxr-compute '), 'version() should return a version string');

% ── Build the same icosahedron fixture as test_eigen.cpp ─────
t = (1 + sqrt(5)) / 2;
raw = [-1  t  0;  1  t  0; -1 -t  0;  1 -t  0;
        0 -1  t;  0  1  t;  0 -1 -t;  0  1 -t;
        t  0 -1;  t  0  1; -t  0 -1; -t  0  1];
V = raw ./ vecnorm(raw, 2, 2);   % normalise each row to unit sphere
F = [ 1 12  6;  1  6  2;  1  2  8;  1  8 11;  1 11 12; ...
      2  6 10;  6 12  5; 12 11  3; 11  8  7;  8  2  9; ...
      4 10  5;  4  5  3;  4  3  7;  4  7  9;  4  9 10; ...
      5 10  6;  3  5 12;  7  3 11;  9  7  8; 10  9  2];   % 1-based
nV = size(V, 1);
nF = size(F, 1);
fprintf('  mesh: nV=%d nF=%d\n', nV, nF);

% ── 1. assembleManifoldOperators ─────────────────────────────────
ops = nxr_compute('assembleManifoldOperators', V, F);
fprintf('  [1] assembleManifoldOperators ✓\n');
assert(ops.nV == 12 && ops.nF == 20, 'unexpected vertex / face count');
assert(issparse(ops.cotanLaplacian), 'stiffness must be sparse');
assert(issparse(ops.mass),      'mass must be sparse');
assert(isequal(size(ops.cotanLaplacian), [nV nV]), 'stiffness shape');
assert(isequal(size(ops.mass),      [nV nV]), 'mass shape');
assert(abs(ops.totalArea - 9.5745) < 1e-3, ...
    'icosahedron unit-sphere total area should be ~9.5745, got %g', ops.totalArea);

% Stiffness should be (numerically) symmetric.
asym = norm(ops.cotanLaplacian - ops.cotanLaplacian', 'fro');
assert(asym < 1e-10, 'stiffness asymmetry too large: %g', asym);

% Mass diagonal entries should sum to totalArea.
massDiag = full(diag(ops.mass));
assert(abs(sum(massDiag) - ops.totalArea) < 1e-9, 'mass diagonal sum vs totalArea');

% ── 2. solve ───────────────────────────────────────
K = 6;
result = nxr_compute('solve', ops.cotanLaplacian, ops.mass, K);
fprintf('  [2] solve ✓ (k=%d, nConverged=%d)\n', ...
    result.k, result.nConverged);
assert(result.k == K, 'wrong k returned');
assert(isequal(size(result.eigenvectors), [nV K]), 'eigenvectors shape');
assert(numel(result.eigenvalues) == K, 'eigenvalues length');

% Eigenvalues sorted ascending. The icosahedron has known structure:
% λ_0 ≈ 0 (DC), then a triplet at λ ≈ 2, then a doublet at λ ≈ 4.34.
assert(issorted(result.eigenvalues), 'eigenvalues not sorted ascending');
fprintf('  eigenvalues: [%s]\n', strjoin(arrayfun( ...
    @(x) sprintf('%.4f', x), result.eigenvalues, 'UniformOutput', false), ', '));
assert(abs(result.eigenvalues(2) - 2.0) < 1e-6, ...
    'expected λ_1 ≈ 2 on icosahedron, got %g', result.eigenvalues(2));

% ── 3. normalize ───────────────────────────────────
Un = nxr_compute('normalize', result.eigenvectors, ops.mass);
fprintf('  [3] normalize ✓\n');
assert(isequal(size(Un), [nV K]), 'normalized U shape');

% Verify M-orthonormality: U' M U should be very close to identity.
gram = Un' * ops.mass * Un;
deviation = norm(gram - eye(K), 'fro');
fprintf('       ||U'' M U - I||_F = %.3e\n', deviation);
assert(deviation < 1e-12, 'M-orthonormality violated');

% ── 4. removeDC ──────────────────────────────────────────────
result.eigenvectors = Un;
trimmed = nxr_compute('removeDC', result);
fprintf('  [4] removeDC ✓ (k=%d → %d)\n', result.k, trimmed.k);
assert(trimmed.k == result.k - 1, 'removeDC should drop one mode');
assert(size(trimmed.eigenvectors, 2) == trimmed.k, 'trimmed eigenvectors width');
assert(numel(trimmed.eigenvalues) == trimmed.k, 'trimmed eigenvalues length');

% ── 5. precompute (one-shot pipeline) ────────────────────────
oneShot = nxr_compute('precompute', V, F, K);
fprintf('  [5] precompute (one-shot) ✓ (k=%d)\n', oneShot.k);
assert(oneShot.k == trimmed.k, ...
    'one-shot pipeline should match step-by-step (k mismatch)');

% Eigenvalues should match step-by-step (within solver tolerance).
deltaLambda = max(abs(oneShot.eigenvalues - trimmed.eigenvalues));
fprintf('       max|Δλ| step-by-step vs one-shot = %.3e\n', deltaLambda);
assert(deltaLambda < 1e-10, ...
    'one-shot eigenvalues differ from step-by-step by %g', deltaLambda);

% ── 6. parallel ───────────────────────────────────
% Transport one tangent vector from vertex 1 (MATLAB 1-based).
% Output is V×3 in MATLAB's column-major convention, with each
% column being x / y / z respectively.
srcVecs = [1 0 0];
transported = nxr_compute('parallel', V, F, 1, srcVecs);
fprintf('  [6] parallel ✓ (size %dx%d)\n', size(transported, 1), size(transported, 2));
assert(isequal(size(transported), [nV 3]), 'transported should be V×3');
assert(any(vecnorm(transported, 2, 2) > 1e-6), 'transported field should be non-trivial');

% ── 7. extendScalar ────────────────────────────────
extended = nxr_compute('extendScalar', V, F, [1; 7], [1.0; 0.0]);
fprintf('  [7] extendScalar ✓\n');
assert(numel(extended) == nV, 'extended field length must be nV');

% ── 8. logMap ──────────────────────────────────────
% Default strategy = AffineLocal (1).
lm = nxr_compute('logMap', V, F, 1);
fprintf('  [8] logMap ✓\n');
assert(isequal(size(lm.logCoords), [nV 2]), 'logCoords should be V×2');
assert(numel(lm.sourceE1) == 3 && numel(lm.sourceE2) == 3, ...
    'source frame should be two 3-vectors');
% Log map of the source vertex itself is (0, 0).
assert(norm(lm.logCoords(1, :)) < 1e-6, 'log map at source should be ~0');

% ── 9. findCenter ──────────────────────────────────
center = nxr_compute('findCenter', V, F, [1; 2; 6]);
fprintf('  [9] findCenter ✓ ([%g %g %g])\n', center(1), center(2), center(3));
assert(numel(center) == 3 && all(isfinite(center)), 'center should be a finite 3-vector');
% Unit-sphere icosahedron — center should sit roughly on the surface.
assert(abs(norm(center) - 1) < 0.4, 'center should be near unit-sphere surface');

% ── 10. signedHeat ───────────────────────────────────
% Loop around vertex 1 (north pole). Curve indices are 1-based.
sd = nxr_compute('signedHeat', V, F, [12; 6; 2; 8; 11], 1);
fprintf('  [10] signedHeat ✓ (range [%g, %g])\n', min(sd), max(sd));
assert(numel(sd) == nV, 'signed distance length nV');
assert(min(sd) < 0 && max(sd) > 0, 'signed distance should straddle zero');

% ── 11. smoothFace ───────────────────────────────
faceField = nxr_compute('smoothFace', V, F, 4);
fprintf('  [11] smoothFace ✓ (nSym=4)\n');
assert(isequal(size(faceField), [nF 3]), 'face field should be F×3');
assert(any(vecnorm(faceField, 2, 2) > 1e-6), 'face field should be non-trivial');

% ── 12. smoothVertex ─────────────────────────────
vfield = nxr_compute('smoothVertex', V, F, 2);
fprintf('  [12] smoothVertex ✓ (nSym=2)\n');
assert(isequal(size(vfield.vertexVectors), [nV 3]), 'vertexVectors should be V×3');
assert(numel(vfield.vertexFieldRaw) == nV * 2, 'vertexFieldRaw should be V*2');
assert(vfield.nSym == 2, 'nSym should round-trip');

% ── 13. compute ─────────────────────────────────
stripes = nxr_compute('compute', V, F, vfield.vertexFieldRaw, 8.0);
fprintf('  [13] compute ✓ (segs=%d)\n', stripes.segmentCount);
assert(stripes.segmentCount >= 0, 'segmentCount non-negative');
assert(isequal(size(stripes.positions), [stripes.segmentCount * 2, 3]), ...
    'positions should be (2*segs)×3');

fprintf('[nxr_compute_mex] all assertions passed ✓\n');
