% test_mex_parity.m — full WASM-parity numerical harness for the stateful
% MEX, mirroring scripts/_smoke-wasm.mjs. Every handle-mode op is exercised
% on the shared icosahedron fixture with the same numerical invariants,
% asserting MATLAB-native shapes (column-major dense, CSC sparse). Run as
% a script (asserts + fprintf).

clear; clc;
fprintf('[test_mex_parity] starting\n');

thisDir = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found under %s/build', mexext, repoRoot);
addpath(hits(1).folder);

[V, F] = local_icosahedron();
nV = size(V, 1);
nF = size(F, 1);
h = nxr_compute('create', V, F);

% ── Group B: operators ───────────────────────────────────────
dec = nxr_compute('assembleDECOperators', h);
nE = size(dec.d0, 1);
assert(nE == 30, 'icosahedron nE=30 (got %d)', nE);
assert(isequal(size(dec.d0), [nE nV]), 'd0 nE×nV');
assert(isequal(size(dec.d1), [nF nE]), 'd1 nF×nE');
assert(issparse(dec.hodge1) && isequal(size(dec.hodge1), [nE nE]), 'hodge1 nE×nE');
fprintf('  assembleDECOperators ✓ (nE=%d)\n', nE);

cl = nxr_compute('assembleConnectionLaplacian', h);
assert(isequal(size(cl.K_real), [2 * nV, 2 * nV]), 'connection K_real 2V×2V (default Real2N)');
assert(norm(cl.K_real - cl.K_real', 'fro') < 1e-9, 'connection K_real symmetric');
assert(cl.outputDim == 2 * nV, 'connection outputDim = 2*nV');
fprintf('  assembleConnectionLaplacian ✓ (outputDim=%d)\n', cl.outputDim);

fr = nxr_compute('frames', h);
assert(isequal(size(fr.e1), [nF 3]), 'frames e1 F×3');
assert(max(abs(vecnorm(fr.e1, 2, 2) - 1)) < 1e-9, 'frames e1 unit-length');
assert(max(abs(sum(fr.e1 .* fr.e2, 2))) < 1e-9, 'frames e1 ⊥ e2');
fprintf('  frames ✓\n');

nrm = nxr_compute('normals', h);
assert(numel(nrm) == nV * 3 && all(isfinite(nrm(:))), 'normals nV*3 finite');
fprintf('  normals ✓\n');

% ── Group C: solvers ─────────────────────────────────────────
phi = nxr_compute('poisson', h, [1; 7], [1.0; -1.0]);
assert(numel(phi) == nV && all(isfinite(phi)), 'poisson → finite length nV');
fprintf('  poisson ✓\n');

d = nxr_compute('heat', h, 1);
assert(numel(d) == nV, 'heat length nV');
assert(abs(d(1)) < 1e-9, 'heat distance at source ≈ 0 (got %g)', d(1));
assert(all(d >= -1e-9), 'heat distances nonnegative');
fprintf('  heat ✓ (max d = %.4f)\n', max(d));

pth = nxr_compute('tracePath', h, 1, 4);   % antipodal pair on the icosahedron
assert(size(pth, 1) >= 2 && size(pth, 2) == 3, 'tracePath ≥2 points × 3');
plen = sum(vecnorm(diff(pth, 1, 1), 2, 2));
assert(plen > 2.0 && plen < pi, 'antipode path in (chord=2, arc=π), got %g', plen);
fprintf('  tracePath ✓ (length %.4f)\n', plen);

rng(42);
omega = randn(nE, 1);
hd = nxr_compute('hodge', h, omega);
recomp = hd.dAlpha + hd.deltaBeta + hd.gamma;
err = max(abs(recomp - omega));
assert(err < 1e-10, 'hodge recomposition dα+δβ+γ = ω (err %g)', err);
fprintf('  hodge ✓ (recomposition err %.2e)\n', err);

% ── Group D: geometric ───────────────────────────────────────
cv = nxr_compute('curvatures', h);
assert(numel(cv.gaussian) == nV, 'curvatures gaussian length nV');
assert(abs(sum(cv.gaussian) - 4 * pi) < 1e-6, ...
    'discrete Gauss-Bonnet: Σ gaussian = 2πχ = 4π (got %g)', sum(cv.gaussian));
assert(isequal(size(cv.principalDirMax), [nV 3]), 'principalDirMax V×3');
fprintf('  curvatures ✓ (Σκ_gauss = %.6f ≈ 4π)\n', sum(cv.gaussian));

bffThrew = false;   % closed icosahedron has no boundary
try
    nxr_compute('bff', h);
catch e
    bffThrew = startsWith(e.identifier, 'nxr:');
end
assert(bffThrew, 'bff on a closed mesh should raise a structured nxr:* error');
fprintf('  bff (closed-mesh throw) ✓\n');

iso = nxr_compute('isoline', h, V(:, 3), 10);   % contours of the z-coordinate
assert(iso.segmentCount >= 0, 'isoline segmentCount >= 0');
assert(isequal(size(iso.positions), [iso.segmentCount * 2, 3]), 'isoline positions (2*segs)×3');
fprintf('  isoline ✓ (segs=%d)\n', iso.segmentCount);

df = nxr_compute('trivial', h, [1; 4], [1.0; 1.0]);   % Σindices = 2 = χ(sphere)
assert(isequal(size(df.directionVectors), [nF 3]), 'trivial directionVectors F×3');
assert(df.gaussBonnetSatisfied, 'trivial: Σindices = χ should satisfy Gauss-Bonnet');
fprintf('  trivial ✓ (χ=%g, gaussBonnet=%d)\n', df.eulerCharacteristic, df.gaussBonnetSatisfied);

ff = nxr_compute('smoothFace', h, 4);
sl = nxr_compute('streamline', h, ff, 8);
assert(sl.segmentCount >= 0, 'streamline segmentCount >= 0');
assert(isequal(size(sl.positions), [sl.segmentCount * 2, 3]), 'streamline positions (2*segs)×3');
fprintf('  streamline ✓ (segs=%d)\n', sl.segmentCount);

% ── Group E: vector field ────────────────────────────────────
rng(7);
oneForm = randn(nE, 1);
wv = nxr_compute('whitney', h, oneForm);
assert(isequal(size(wv), [nF 3]) && all(isfinite(wv(:))), 'whitney → F×3 finite');
fprintf('  whitney ✓\n');

u = zeros(nV, 1); u(1) = 1.0;
gr = nxr_compute('gradient', h, u);
assert(isequal(size(gr), [nF 3]), 'gradient → F×3');
assert(max(vecnorm(gr, 2, 2)) > 0, 'gradient nontrivial');
fprintf('  gradient ✓\n');

% ── Group F: time-varying generators (need eigenmodes) ───────
nxr_compute('precompute', h, 6);

ts = (0:4) * 0.1;
hdif = nxr_compute('heatDiffusion', h, 1, 1.0, ts, 1.0);
assert(isequal(size(hdif), [numel(ts) nV]), 'heatDiffusion → [T, nV]');
assert(all(isfinite(hdif(:))), 'heatDiffusion finite');
fprintf('  heatDiffusion ✓ ([T nV]=[%d %d])\n', size(hdif, 1), size(hdif, 2));

dw = nxr_compute('dampedWave', h, [2; 3], [1.0; 1.0], [0.1; 0.1], [0.0; 0.0], (0:2) * 0.1);
assert(isequal(size(dw), [3 nV]), 'dampedWave → [T, nV]');
assert(all(isfinite(dw(:))), 'dampedWave finite');
fprintf('  dampedWave ✓ ([T nV]=[%d %d])\n', size(dw, 1), size(dw, 2));

om = nxr_compute('randomDecomposed1Form', h, 1.0, 1.0, 0.0, 42);
assert(numel(om) == nE && all(isfinite(om)), 'randomDecomposed1Form length nE finite');
hd2 = nxr_compute('hodge', h, om);
assert(max(abs(hd2.dAlpha + hd2.deltaBeta + hd2.gamma - om)) < 1e-10, ...
    'randomDecomposed1Form ∘ hodge round-trip');
fprintf('  randomDecomposed1Form ✓\n');

% generators before solve raise NotPrecomputed on a fresh handle
h3 = nxr_compute('create', V, F);
npThrew = false;
try
    nxr_compute('heatDiffusion', h3, 1, 1.0, ts, 1.0);
catch e
    npThrew = strcmp(e.identifier, 'nxr:notPrecomputed');
end
assert(npThrew, 'heatDiffusion before solve must raise nxr:notPrecomputed');
nxr_compute('destroy', h3);
fprintf('  notPrecomputed guard ✓\n');

nxr_compute('destroy', h);
fprintf('[test_mex_parity] all parity assertions passed ✓\n');

% ── local helpers ─────────────────────────────────────────────
function [V, F] = local_icosahedron()
    t = (1 + sqrt(5)) / 2;
    raw = [-1  t  0;  1  t  0; -1 -t  0;  1 -t  0;
            0 -1  t;  0  1  t;  0 -1 -t;  0  1 -t;
            t  0 -1;  t  0  1; -t  0 -1; -t  0  1];
    V = raw ./ vecnorm(raw, 2, 2);
    F = [ 1 12  6;  1  6  2;  1  2  8;  1  8 11;  1 11 12; ...
          2  6 10;  6 12  5; 12 11  3; 11  8  7;  8  2  9; ...
          4 10  5;  4  5  3;  4  3  7;  4  7  9;  4  9 10; ...
          5 10  6;  3  5 12;  7  3 11;  9  7  8; 10  9  2];
end
