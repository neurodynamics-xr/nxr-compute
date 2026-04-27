% cxf_mex_smoke.m — verifies cxf.mexw64 against the same icosahedron
% fixture used by the C++ test_eigen.exe. Mirrors the gpjs MATLAB
% bindings smoke pattern: build a known mesh, exercise each command,
% assert outputs are sensible.
%
% Run from MATLAB (or via the matlab MCP tool):
%   cd <repo>/native/mex/test
%   cxf_mex_smoke

clear; clc;
fprintf('[cxf_mex] starting smoke test\n');

% Locate the freshly-built MEX (relative to this script).
thisDir = fileparts(mfilename('fullpath'));
mexDir  = fullfile(thisDir, '..', '..', 'build_node', 'Release');
addpath(mexDir);

assert(exist(fullfile(mexDir, 'cxf.mexw64'), 'file') == 3, ...
    'cxf.mexw64 not found in %s', mexDir);

% ── Sanity: version string ───────────────────────────────────
v = cxf('version');
fprintf('  version: %s\n', v);
assert(startsWith(v, 'cxf '), 'version() should return a version string');

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

% ── 1. assembleMeshOperators ─────────────────────────────────
ops = cxf('assembleMeshOperators', V, F);
fprintf('  [1] assembleMeshOperators ✓\n');
assert(ops.nV == 12 && ops.nF == 20, 'unexpected vertex / face count');
assert(issparse(ops.stiffness), 'stiffness must be sparse');
assert(issparse(ops.mass),      'mass must be sparse');
assert(isequal(size(ops.stiffness), [nV nV]), 'stiffness shape');
assert(isequal(size(ops.mass),      [nV nV]), 'mass shape');
assert(abs(ops.totalArea - 9.5745) < 1e-3, ...
    'icosahedron unit-sphere total area should be ~9.5745, got %g', ops.totalArea);

% Stiffness should be (numerically) symmetric.
asym = norm(ops.stiffness - ops.stiffness', 'fro');
assert(asym < 1e-10, 'stiffness asymmetry too large: %g', asym);

% Mass diagonal entries should sum to totalArea.
massDiag = full(diag(ops.mass));
assert(abs(sum(massDiag) - ops.totalArea) < 1e-9, 'mass diagonal sum vs totalArea');

% ── 2. solveEigenmodes ───────────────────────────────────────
K = 6;
result = cxf('solveEigenmodes', ops.stiffness, ops.mass, K);
fprintf('  [2] solveEigenmodes ✓ (k=%d, nConverged=%d)\n', ...
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

% ── 3. normalizeEigenmodes ───────────────────────────────────
Un = cxf('normalizeEigenmodes', result.eigenvectors, ops.mass);
fprintf('  [3] normalizeEigenmodes ✓\n');
assert(isequal(size(Un), [nV K]), 'normalized U shape');

% Verify M-orthonormality: U' M U should be very close to identity.
gram = Un' * ops.mass * Un;
deviation = norm(gram - eye(K), 'fro');
fprintf('       ||U'' M U - I||_F = %.3e\n', deviation);
assert(deviation < 1e-12, 'M-orthonormality violated');

% ── 4. removeDC ──────────────────────────────────────────────
result.eigenvectors = Un;
trimmed = cxf('removeDC', result);
fprintf('  [4] removeDC ✓ (k=%d → %d)\n', result.k, trimmed.k);
assert(trimmed.k == result.k - 1, 'removeDC should drop one mode');
assert(size(trimmed.eigenvectors, 2) == trimmed.k, 'trimmed eigenvectors width');
assert(numel(trimmed.eigenvalues) == trimmed.k, 'trimmed eigenvalues length');

% ── 5. precompute (one-shot pipeline) ────────────────────────
oneShot = cxf('precompute', V, F, K);
fprintf('  [5] precompute (one-shot) ✓ (k=%d)\n', oneShot.k);
assert(oneShot.k == trimmed.k, ...
    'one-shot pipeline should match step-by-step (k mismatch)');

% Eigenvalues should match step-by-step (within solver tolerance).
deltaLambda = max(abs(oneShot.eigenvalues - trimmed.eigenvalues));
fprintf('       max|Δλ| step-by-step vs one-shot = %.3e\n', deltaLambda);
assert(deltaLambda < 1e-10, ...
    'one-shot eigenvalues differ from step-by-step by %g', deltaLambda);

fprintf('[cxf_mex] all assertions passed ✓\n');
