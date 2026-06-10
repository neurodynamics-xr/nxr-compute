function test_covariant_differential
fprintf('[test_covariant_differential] starting\n');
thisDir = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'mex not found'); addpath(hits(1).folder); clear nxr_compute

t = (1+sqrt(5))/2;
V = [-1 t 0; 1 t 0; -1 -t 0; 1 -t 0; 0 -1 t; 0 1 t; 0 -1 -t; 0 1 -t; t 0 -1; t 0 1; -t 0 -1; -t 0 1];
F = [1 12 6; 1 6 2; 1 2 8; 1 8 11; 1 11 12; 2 6 10; 6 12 5; 12 11 3; 11 8 7; 8 2 9; ...
     4 10 5; 4 5 3; 4 3 7; 4 7 9; 4 9 10; 5 10 6; 3 5 12; 7 3 11; 9 7 8; 10 9 2];
h = nxr_compute('create', V, F);
N = 12; E = 30;

% frame transport is orthogonal
P = nxr_compute('frameTransport', h, 1, 4);
assert(isequal(size(P),[3 3]) && norm(P'*P - eye(3),'fro') < 1e-12, 'frameTransport orthogonal');

% lifts round-trip
Lloc = randn(N,3);
back = nxr_compute('liftToFrame', h, nxr_compute('liftToWorld', h, Lloc));
assert(max(abs(back(:) - Lloc(:))) < 1e-12, 'lift round-trip');

% ARTIFACT REMOVAL end-to-end: Cartesian-constant field -> zero covariant gradient
Lworld = repmat([0.3 -0.7 0.2], N, 1);
Lloc_c = nxr_compute('liftToFrame', h, Lworld);          % [N,3] local coords (differ per vertex)
G = nxr_compute('operators', h, 'gradient3D');           % sparse [3E, 3N]
x = Lloc_c(:);                                           % column-major [N,3] -> [a;b;c] = component-major 3N
assert(max(abs(G*x)) < 1e-9, 'G·(Cartesian-constant) = 0 (artifact removed)');
% naive local difference is NOT zero (frames rotate)
assert(max(abs(Lloc_c(1,:) - Lloc_c(3,:))) > 1e-3, 'naive local difference nonzero');

% two Cartesian-parallel leadfields lift to the SAME world vector (sulcal-wall artifact gone)
assert(max(abs(nxr_compute('liftToWorld', h, Lloc_c) - Lworld), [], 'all') < 1e-12, ...
       'lift recovers the constant world field');

nxr_compute('destroy', h);
fprintf('ALL TESTS PASSED: test_covariant_differential\n');
end
