function test_operator_info
fprintf('[test_operator_info] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found', mexext);
addpath(hits(1).folder); clear nxr_compute

% ── laplaceBeltrami ──────────────────────────────────────────────────────────
info = nxr_compute('operatorInfo', 'laplaceBeltrami');
assert(strcmp(info.bundle,    'scalar'),    'laplaceBeltrami bundle');
assert(strcmp(info.order,     'second'),    'laplaceBeltrami order');
assert(strcmp(info.role,      'laplacian'), 'laplaceBeltrami role');
assert(strcmp(info.status,    'built'),     'laplaceBeltrami status');
assert(strcmp(info.field_type,'real'),      'laplaceBeltrami field_type');
assert(strcmp(info.domain,    'vertex'),    'laplaceBeltrami domain');

% ── relativeDirac ─────────────────────────────────────────────────────────────
rd = nxr_compute('operatorInfo', 'relativeDirac');
assert(strcmp(rd.bundle,     'immersion'),  'relativeDirac bundle');
assert(strcmp(rd.holonomy,   'graded'),     'relativeDirac holonomy');
assert(strcmp(rd.field_type, 'quaternion'), 'relativeDirac field_type');
assert(rd.graded == 1,                      'relativeDirac graded flag');

% ── extrinsicWeitzenbockLaplacian (planned) ────────────────────────────────────
ew = nxr_compute('operatorInfo', 'extrinsicWeitzenbockLaplacian');
assert(strcmp(ew.status, 'planned'), 'extrinsicWeitzenbockLaplacian status');

% ── unknown id must error ─────────────────────────────────────────────────────
try
    nxr_compute('operatorInfo', 'nonExistentOperatorXYZ');
    error('should have thrown');
catch ME
    assert(strcmp(ME.identifier, 'nxr:invalidInput'), ...
        'expected nxr:invalidInput, got %s', ME.identifier);
end

fprintf('test_operator_info PASSED\n');
end
