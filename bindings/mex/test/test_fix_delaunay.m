function test_fix_delaunay
fprintf('[test_fix_delaunay] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'mex not found'); addpath(hits(1).folder); clear nxr_compute

% thin quad, split along the long diagonal (1-2 in 1-based) -> must flip
V = [0 0 0; 2 0 0; 1 0.2 0; 1 -0.2 0];
F = [1 3 2; 1 2 4];            % 1-based; shared edge 1-2 (the long diagonal)
[V2, F2, n] = nxr_compute('fixDelaunay', V, F);
assert(isequal(V2, V), 'vertices unchanged');
assert(n == 1, 'one flip');
assert(isequal(size(F2), size(F)), 'same #faces');
assert(all(F2(:) >= 1 & F2(:) <= 4), 'faces 1-based in range');
% long diagonal (1,2) gone; short diagonal (3,4) present
hasEdge = @(F,a,b) any(arrayfun(@(r) all(ismember([a b], F(r,:))), 1:size(F,1)));
assert(~hasEdge(F2,1,2), 'long diagonal removed');
assert(hasEdge(F2,3,4),  'short diagonal present');

% already-Delaunay icosahedron -> 0 flips
[Vi, Fi] = local_icosahedron();
[~, Fi2, ni] = nxr_compute('fixDelaunay', Vi, Fi);
assert(ni == 0, 'icosahedron already Delaunay');
assert(isequal(sortrows(sort(Fi2,2)), sortrows(sort(Fi,2))), 'faces unchanged as a set');

fprintf('ALL TESTS PASSED: test_fix_delaunay\n');
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
