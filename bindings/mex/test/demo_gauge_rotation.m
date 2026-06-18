function demo_gauge_rotation
% demo_gauge_rotation — illustrate what the gauge rotation does to a vector
% field's COORDINATES while leaving the geometric vector fixed.
%
% Two gauges over the SAME mesh:
%   LC      : Geometry.vertex.grid                              (arbitrary base frame)
%   trivial : Gauge.vertex.rotation .* Geometry.vertex.grid    (combed frame)
%
% A tangent vector field is one geometric object. Its complex coordinate z in a
% frame c = e1 + i*e2 is  z = <V,e1> + i<V,e2>;  the world vector is recovered
% (gauge-free) by  V = real(conj(z) .* c).  Changing gauge multiplies z by the
% rotation; |z| and V are invariant.

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found — build first', mexext);
addpath(hits(1).folder); clear nxr_compute

% ── 1. A reasonably dense sphere (chi = 2) via Fibonacci points + convex hull ──
N  = 1200;
gr = (1 + sqrt(5)) / 2;
i  = (0:N-1)';
phi   = 2*pi * i / gr;
ct    = 1 - 2*(i + 0.5)/N;       % cos(theta), uniform in [-1,1]
st    = sqrt(max(0, 1 - ct.^2));
V = [st.*cos(phi), st.*sin(phi), ct];
F = convhull(V);                  % outward-oriented triangulation of the sphere

h = nxr_compute('create', V, F);

% Singularities at the two poles (indices sum to chi = 2).
[~, iN] = max(V(:,3));  [~, iS] = min(V(:,3));
opts = struct('singVerts', uint32([iN; iS]), 'singValues', [1; 1], 'source', 'manual');

G  = nxr_compute('geometry', h);
Gt = nxr_compute('gauge', h, 'trivial', opts);

cL = G.vertex.grid;                       % [nV x 3] complex — LC frame
r  = Gt.vertex.rotation;                  % [nV x 1] complex — combing rotation
cT = r .* cL;                             % [nV x 3] complex — combed frame
nrm = cross(real(cL), imag(cL), 2);       % vertex normal (frame-independent)

% ── 2. An example world tangent field: a fixed Cartesian direction, projected
%       onto each tangent plane. Gauge-free by construction. ──
d  = [0 0 1];                             % world "north" direction
Vw = d - (Vd(d, nrm)) .* nrm;             % remove normal component
Vw = Vw ./ max(vecnorm(Vw, 2, 2), 1e-12); % unit tangent (skip near poles)

% ── 3. Read its coordinates in each gauge by projection ──
zL = proj(Vw, cL);                        % coordinates in LC frame
zT = proj(Vw, cT);                        % coordinates in combed frame

% ── 4. The gauge-transformation law (verify, don't assume the direction) ──
errSame = max(abs(zT - r        .* zL));  % candidate: z' = rotation .* z
errConj = max(abs(zT - conj(r)  .* zL));  % candidate: z' = conj(rotation) .* z
fprintf('coordinate law:  |zT - r.*zL|      = %.3e\n', errSame);
fprintf('coordinate law:  |zT - conj(r).*zL| = %.3e\n', errConj);

% Magnitude is gauge-invariant; the world vector is unchanged in both gauges.
fprintf('magnitude invariant:   max||zT|-|zL|| = %.3e\n', max(abs(abs(zT)-abs(zL))));
VwL = real(conj(zL) .* cL);               % reconstruct from LC coords
VwT = real(conj(zT) .* cT);               % reconstruct from combed coords
fprintf('world vector invariant: max|VwL - VwT| = %.3e\n', max(vecnorm(VwL-VwT,2,2)));
fprintf('reconstruction exact:   max|VwL - Vw|  = %.3e\n', max(vecnorm(VwL-Vw, 2,2)));

% Per-vertex phase difference IS the gauge rotation angle.
dphi = angle(zT) - angle(zL);
dphi = atan2(sin(dphi), cos(dphi));       % wrap to (-pi,pi]
fprintf('phase(zT)-phase(zL) == angle(rotation): max err = %.3e\n', ...
        max(abs(atan2(sin(dphi - angle(r)), cos(dphi - angle(r))))));

% ── 5. The combed field as the reference: it is z == 1 (constant) in the combed
%       gauge, and swirls (phase = -angle(rotation)... numerically conj(r)) in LC. ──
Vcomb = real(r .* cL);                     % the comb (== real(cT))
zc_T  = proj(Vcomb, cT);                   % combed-gauge coords of the comb
zc_L  = proj(Vcomb, cL);                   % LC-gauge coords of the comb
fprintf('comb in combed gauge: max|zc_T - 1| = %.3e  (constant real)\n', ...
        max(abs(zc_T - 1)));
fprintf('comb in LC gauge swirls: std(angle(zc_L)) = %.3f rad\n', std(angle(zc_L)));

% ── 6. Figure: surface coloured by coordinate phase in each gauge ──
figure('Color','w','Name','Gauge rotation acting on a vector field');
tiledlayout(1,2);
step = max(1, round(numel(zL)/300));      % subsample arrows
sel  = 1:step:size(V,1);

nexttile; hold on; axis equal off; title('LC gauge: arg(z) (swirls)');
trisurf(F, V(:,1),V(:,2),V(:,3), angle(zL), 'EdgeColor','none'); colormap(gca,hsv);
quiver3(V(sel,1),V(sel,2),V(sel,3), Vw(sel,1),Vw(sel,2),Vw(sel,3), 0.7,'k');
plot3(V([iN iS],1),V([iN iS],2),V([iN iS],3),'w.','MarkerSize',22);

nexttile; hold on; axis equal off; title('Trivial (combed) gauge: arg(z)');
trisurf(F, V(:,1),V(:,2),V(:,3), angle(zT), 'EdgeColor','none'); colormap(gca,hsv);
quiver3(V(sel,1),V(sel,2),V(sel,3), Vw(sel,1),Vw(sel,2),Vw(sel,3), 0.7,'k');
plot3(V([iN iS],1),V([iN iS],2),V([iN iS],3),'w.','MarkerSize',22);
% NOTE: the BLACK ARROWS (world vectors) are identical in both panels — only the
% colour (the coordinate phase) changes. That is the whole point: same vector,
% different gauge.

nxr_compute('destroy', h);
fprintf('demo_gauge_rotation: DONE\n');
end

function z = proj(Vw, c)
% Coordinates of world tangent field Vw [nV x 3] in frame c = e1 + i e2 [nV x 3].
z = sum(Vw .* real(c), 2) + 1i * sum(Vw .* imag(c), 2);
end

function s = Vd(d, n)
% Per-row dot of a single world vector d [1x3] with normals n [nV x 3] -> [nV x 1].
s = n * d(:);
end
