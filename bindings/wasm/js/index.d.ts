// TypeScript declarations for the nxr-compute WASM binding.
// See docs/nxr-compute/three-js-integration.md and docs/nxr-compute/visualization-recipes.md
// for usage patterns.

/** Sparse matrix in COO triplet form. Always real double. */
export interface SparseMatrixCOO {
  row:  Int32Array
  col:  Int32Array
  data: Float64Array
  rows: number
  cols: number
  nnz:  number
}

/** Sparse complex matrix in COO triplet form. Used by the
 *  connection-Laplacian "complex" output format — `realData` and
 *  `imagData` carry the real and imaginary parts of each entry. */
export interface SparseMatrixCOOComplex {
  row:       Int32Array
  col:       Int32Array
  realData:  Float64Array
  imagData:  Float64Array
  rows:      number
  cols:      number
  nnz:       number
}

/** Connection-Laplacian assembly options. All fields are optional;
 *  defaults are { domain: 'vertex', nSym: 1, regularization: 1e-8,
 *  format: 'real2N' }. */
export interface ConnectionLaplacianOptions {
  domain?:         'vertex' | 'face' | 'edge'
  /** Symmetry order. 1 = vector field, 2 = line field, 4 = cross field. Must be > 0. */
  nSym?:           number
  /** Additive ε·I shift. Default 1e-8. */
  regularization?: number
  format?:         'real2N' | 'complex'
}

/** Connection Laplacian on a chosen domain.
 *  - For `format: 'real2N'`, `K` is a 2N×2N symmetric real COO matrix
 *    (each complex entry a + bi lowered to the 2×2 block [[a,-b],[b,a]]).
 *    Drops directly into `solveEigenmodesFromTriplets` with a
 *    block-diagonal real mass matrix `blkdiag(M, M)`.
 *  - For `format: 'complex'`, `K` is an N×N complex Hermitian COO
 *    matrix with parallel `realData` / `imagData` value arrays. */
export type ConnectionLaplacianResult =
  | {
      K:              SparseMatrixCOO
      baseDim:        number
      outputDim:      number   // 2 * baseDim
      domain:         'vertex' | 'face' | 'edge'
      nSym:           number
      regularization: number
      format:         'real2N'
    }
  | {
      K:              SparseMatrixCOOComplex
      baseDim:        number
      outputDim:      number   // baseDim
      domain:         'vertex' | 'face' | 'edge'
      nSym:           number
      regularization: number
      format:         'complex'
    }

/** FEM mass-matrix variant string passed to assembleManifoldOperators().
 *  Names match geometry-central's vertexLumpedMassMatrix and
 *  vertexGalerkinMassMatrix exactly. */
export type MassMatrixVariant = "lumped" | "galerkin"

export interface ManifoldOperators {
  /** Cotangent Laplacian (PSD, symmetrized), V×V sparse */
  stiffness:   SparseMatrixCOO
  /** Mass matrix, V×V sparse. SPD; diagonal for "lumped",
   *  has off-diagonal couplings for "galerkin". */
  mass:        SparseMatrixCOO
  /** Which variant produced `mass`. */
  massVariant: MassMatrixVariant
  /** Per-vertex dual area (A/3-per-vertex from GC), V — always sourced
   *  from `vertexDualAreas` regardless of `massVariant`. */
  vertexAreas: Float64Array
  /** Per-vertex normals, V*3 row-major */
  normals:     Float64Array
  totalArea:   number
  nV:          number
  nE:          number
  nF:          number
}

export interface DECOperators {
  /** Vertex → edge gradient, E×V sparse */
  d0:            SparseMatrixCOO
  /** Edge → face curl,     F×E sparse */
  d1:            SparseMatrixCOO
  hodge0:        SparseMatrixCOO
  hodge1:        SparseMatrixCOO
  hodge2:        SparseMatrixCOO
  hodge1Inverse: SparseMatrixCOO
}

export interface EigenResult {
  /** Eigenvectors stored row-major as V*K floats. Column k starts at offset k*nV in this flat buffer. */
  eigenvectors: Float64Array
  /** Eigenvalues, length K, sorted ascending. */
  eigenvalues:  Float64Array
  k:            number
  nConverged:   number
}

export interface FaceFrames {
  /** First per-face tangent vector, F*3 row-major */
  e1:      Float64Array
  /** Second per-face tangent vector (n × e1), F*3 row-major */
  e2:      Float64Array
  /** Per-face normal, F*3 row-major */
  normals: Float64Array
}

export interface PrecomputeResult {
  operators:  ManifoldOperators
  dec:        DECOperators
  eigenmodes: EigenResult
  frames: FaceFrames
}

export interface HodgeResult {
  /** Per-vertex exact (gradient) potential α */
  exactPotential:    Float64Array
  /** Per-vertex co-exact (curl) potential β (averaged from face) */
  coExactPotentialV: Float64Array
  combinedPotential: Float64Array
  /** Per-edge dα 1-form */
  dAlpha:            Float64Array
  /** Per-edge δβ 1-form */
  deltaBeta:         Float64Array
  /** Per-edge harmonic γ 1-form */
  gamma:             Float64Array
  /** Face-centered 3D vector fields, each F*3 row-major */
  omegaVectors:      Float64Array
  dAlphaVectors:     Float64Array
  deltaBetaVectors:  Float64Array
  gammaVectors:      Float64Array
}

export interface CurvatureResult {
  gaussian:     Float64Array
  mean:         Float64Array
  kMin:         Float64Array
  kMax:         Float64Array
  /** Per-vertex max-curvature direction, V*3 row-major */
  principalDir: Float64Array
}

export interface PolylineResult {
  /** N*3 row-major points */
  positions: Float64Array
  nPoints:   number
}

export interface SegmentsResult {
  /** (2*segmentCount)*3 row-major endpoint pairs */
  positions:    Float64Array
  segmentCount: number
}

export interface DirectionFieldResult {
  /** Per-edge connection 1-form */
  connections:          Float64Array
  /** F*3 per-face direction vectors */
  directionVectors:     Float64Array
  /** F*3 per-face 90°-rotated direction vectors */
  orthogonalVectors:    Float64Array
  eulerCharacteristic:  number
  gaussBonnetSatisfied: boolean
}

export interface TimeSeriesField {
  /** T*nV row-major frames; frame ti starts at offset ti*nV */
  data: Float32Array
  T:    number
  nV:   number
}

export interface LogMapResult {
  /** V*2 row-major (logX, logY) coordinates in source's tangent frame.
   *  norm is geodesic distance from source; atan2 is bearing. */
  logCoords: Float64Array
  /** Source vertex's tangent basis e1 in world coords (3 floats). */
  sourceE1:  Float64Array
  /** Source vertex's tangent basis e2 in world coords (3 floats). */
  sourceE2:  Float64Array
}

export interface SmoothVertexFieldResult {
  /** V*3 row-major lifted world-space vectors (principal nSym-RoSy representative). */
  vertexVectors:  Float64Array
  /** V*2 row-major raw nSym-RoSy field in vertex tangent basis.
   *  Pass back into compute as-is. */
  vertexFieldRaw: Float64Array
  nSym: number
}

/** Strategy for VHM log-map. AffineLocal recommended (fast, accurate near source). */
export enum LogMapStrategy {
  VectorHeat     = 0,
  AffineLocal    = 1,
  AffineAdaptive = 2,
}

export enum SignedHeatLevelSet {
  None     = 0,
  ZeroSet  = 1,
  Multiple = 2,
}

/** Vertex-normal estimator selection. */
export enum NormalType {
  AngleWeighted   = 0,
  AreaWeighted    = 1,
  EqualWeighted   = 2,
  SphereInscribed = 3,
  MeanCurvature   = 4,
  GaussCurvature  = 5,
}

/**
 * Stateful compute context for one mesh. Holds the nxr::manifold::Manifold
 * in WASM linear memory plus cached operators / Cholesky factors /
 * eigenmodes. Call `delete()` when finished.
 */
export interface Manifold {
  nV(): number
  nE(): number
  nF(): number

  // Operators / geometry
  /** Assemble cotangent stiffness + mass matrix.
   *  @param variant — mass-matrix variant ("lumped" default, "galerkin"). */
  assembleManifoldOperators(variant?: MassMatrixVariant | ""): ManifoldOperators
  assembleDECOperators():  DECOperators

  /** Single named operator as COO sparse — parity with the MEX 'operators'
   *  command (same matrices the internal solvers use). `connection` returns a
   *  complex COO; `dec` returns `{ d0, d1 }`; `dirac`/`diracFace` take a numeric
   *  tau ∈ [0,1]; the remaining families take no second argument. */
  operators(family: "laplacian", subtype: "cotan" | "graph" | "covariant"): SparseMatrixCOO
  operators(family: "laplacian", subtype: "connection"): SparseMatrixCOOComplex
  operators(family: "mass", subtype: "lumped" | "galerkin"): SparseMatrixCOO
  operators(family: "hodge", subtype: "h0" | "h1" | "h2" | "h1inv"): SparseMatrixCOO
  operators(family: "dec"): { d0: SparseMatrixCOO; d1: SparseMatrixCOO }
  operators(family: "gradient3D" | "diracD" | "diracFaceD" | "diracIntrinsicD"): SparseMatrixCOO
  operators(family: "dirac" | "diracFace", tau: number): SparseMatrixCOO

  frames():     FaceFrames
  normals(type?: NormalType): Float64Array

  // Spectral
  solve(k: number, sigma?: number): EigenResult
  normalize(U: Float64Array, rows: number, cols: number): Float64Array
  removeDC(eig: EigenResult): EigenResult

  /**
   * One-shot precompute: returns the visualization-defaults pack
   * (operators + DEC + eigenmodes + face frames) in one call.
   * Cheaper than calling the four steps separately.
   */
  precompute(options?: { k?: number; sigma?: number }): PrecomputeResult

  // Solvers
  poisson(sourceVerts: Int32Array | number[], sourceValues: Float64Array | number[]): Float64Array
  heat(sourceVerts: Int32Array | number[]): Float64Array
  tracePath(vStart: number, vEnd: number): PolylineResult
  hodge(omega: Float64Array): HodgeResult

  // Geometric
  curvatures(): CurvatureResult
  /** Throws if the mesh has no boundary (BFF requires open mesh). */
  bff(): Float64Array
  isoline(scalars: Float64Array, numLevels: number, minVal?: number, maxVal?: number): SegmentsResult
  trivial(
    singVerts:  Int32Array  | number[],
    singValues: Float64Array | number[],
  ): DirectionFieldResult
  streamline(
    faceField: Float64Array,
    numSeeds?: number,
    stepCoef?: number,
    maxSteps?: number,
  ): SegmentsResult

  // Vector field
  whitney(oneForm: Float64Array): Float64Array
  gradient(scalar: Float64Array): Float64Array

  // Time-varying generators (require eigenmodes computed first)
  heatDiffusion(
    sources:      Int32Array  | number[],
    sourceValues: Float64Array | number[],
    timesteps:    Float64Array | number[],
    alpha?: number,
  ): TimeSeriesField
  dampedWave(
    modeIndices: Int32Array  | number[],
    amplitudes:  Float64Array | number[],
    dampings:    Float64Array | number[],
    phases:      Float64Array | number[],
    timesteps:   Float64Array | number[],
  ): TimeSeriesField
  randomDecomposed1Form(
    alphaStrength: number,
    betaStrength:  number,
    gammaStrength: number,
    seed?: number,
  ): Float64Array

  // Vector heat method (Sharp, Soliman, Crane 2019)
  /** Parallel-transport tangent vectors. sourceVectors is N*3 world-space. Returns V*3 row-major. */
  parallel(
    sourceVerts:   Int32Array  | number[],
    sourceVectors: Float64Array | number[],
  ): Float64Array
  /** Smoothly extend a sparse scalar from the given source vertices to all V. */
  extendScalar(
    sourceVerts:  Int32Array  | number[],
    sourceValues: Float64Array | number[],
  ): Float64Array
  /** Logarithmic map at one source vertex. */
  logMap(
    sourceVertex: number,
    strategy?:    LogMapStrategy,
  ): LogMapResult
  /** Karcher mean of source vertices. Returns 3 floats (xyz). */
  findCenter(
    sourceVerts: Int32Array | number[],
    p?:          number,
  ): Float64Array

  // Signed heat method (Feng & Crane 2024)
  /** Signed geodesic distance from a curve given as ordered vertex indices. */
  signedHeat(
    curveVerts: Int32Array | number[],
    isLoop:     boolean,
    levelSet?:  SignedHeatLevelSet,
  ): Float64Array

  // Smooth direction fields (Knöppel-Crane)
  /** Smoothest face-based direction field, lifted to F*3 world. nSym=4 → cross field. */
  smoothFace(
    nSym?:             number,
    alignToCurvature?: boolean,
  ): Float64Array
  /** Smoothest vertex-based direction field. Use the returned `vertexFieldRaw`
   *  as input to compute. nSym=2 typical for stripes. */
  smoothVertex(
    nSym?:             number,
    alignToCurvature?: boolean,
  ): SmoothVertexFieldResult

  // Stripe patterns (Knöppel-Crane SIGGRAPH 2015)
  /** Sinusoidal stripes aligned to a 2-RoSy field, returned as 3D line segments. */
  compute(
    vertexFieldRaw:           Float64Array,
    uniformFrequency:         number,
    connectOnSingularities?:  boolean,
  ): SegmentsResult
  /** Stripe pattern with a per-vertex frequency (length V). */
  computeFreq(
    vertexFieldRaw:           Float64Array,
    frequencies:              Float64Array,
    connectOnSingularities?:  boolean,
  ): SegmentsResult

  /** Release WASM heap memory; the context is invalid after delete(). */
  delete(): void
}

export interface NxrCompute {
  version(): string
  /** Flat surface (legacy). Returns a Manifold with the historical
   *  per-method names (solve, heat, …). */
  createContext(
    vertices: Float64Array | number[],
    faces:    Int32Array  | number[],
  ): Manifold
  /** Six-group nested namespace `nxr.manifold.*` over the same compute
   *  context. Internally constructs a single ContextWrapper. */
  createManifoldContext(
    vertices: Float64Array | number[],
    faces:    Int32Array  | number[],
  ): ManifoldContext
  /** Functional form of the six-group namespace. Each leaf takes a
   *  ManifoldContext as its first argument. Identical behaviour to the
   *  per-context methods. */
  nxr: {
    manifold: ManifoldFunctional
  }
}

/* ---------------------------------------------------------------------- */
/*           Six-group nested namespace `nxr.manifold.*` types            */
/* ---------------------------------------------------------------------- */

/** Solver entry points — Poisson, heat diffusion, eigenmodes, Hodge. */
export interface SolveGroup {
  poisson(
    sourceVerts:  Int32Array  | number[],
    sourceValues: Float64Array | number[],
  ): Float64Array
  /** Heat diffusion over the given timesteps; consolidates the legacy
   *  `heatDiffusion` entry point. */
  heat(
    sources:      Int32Array  | number[],
    sourceValues: Float64Array | number[],
    timesteps:    Float64Array | number[],
    alpha?: number,
  ): TimeSeriesField
  eigen(k: number, sigma?: number): EigenResult
  /** Provisional placement — Hodge–Helmholtz decomposition may be
   *  reclassified to `operator.hodgeDecomp` in a future round. */
  hodge(omega: Float64Array): HodgeResult
}

/** DEC and mesh operators as individual sparse matrices. The first call
 *  to any DEC piece triggers a single bulk assembly internally; the same
 *  applies to mass / stiffness / laplacian. */
export interface OperatorGroup {
  d0():           SparseMatrixCOO
  d1():           SparseMatrixCOO
  star0():        SparseMatrixCOO
  star1():        SparseMatrixCOO
  star2():        SparseMatrixCOO
  star1Inverse(): SparseMatrixCOO
  mass():         SparseMatrixCOO
  stiffness():    SparseMatrixCOO
  /** Cotangent Laplacian — same matrix as `stiffness()`. */
  laplacian():    SparseMatrixCOO
  /** Connection Laplacian on the chosen domain. Drives smoothest n-RoSy
   *  direction fields, parallel-transport energies, and tangent-bundle
   *  eigendecompositions. Result-level cached by all four options. */
  connectionLaplacian(options?: ConnectionLaplacianOptions): ConnectionLaplacianResult
  /** Drop the cached DEC and mesh operator results so the next access
   *  re-runs `assembleDECOperators` / `assembleManifoldOperators`. */
  invalidateCache(): void
}

/** Geometric selections. Most are round-1 placeholders. */
export interface QueryGroup {
  vertex(v: number): { vertexIndex: number }
  /** TODO (round-1 stub) — flip-out geodesic between two vertices. */
  line(v1: number, v2: number):    { method: 'todo'; name: string }
  /** TODO (round-1 stub) — isoline of geodesic distance at radius r. */
  circle(v: number, r: number):    { method: 'todo'; name: string }
  /** TODO (round-1 stub) — vertex set within geodesic radius r. */
  region(v: number, r: number):    { method: 'todo'; name: string }
  /** Single-level isoline of a per-vertex scalar field. */
  isoline(field: Float64Array | number[], level: number): SegmentsResult
  /** Karcher mean of source vertices (vector-heat findCenter). */
  center(sourceVerts: Int32Array | number[], p?: number): Float64Array
}

/** `measure.distance` is callable (heat-method geodesic) AND carries a
 *  `.signed` sub-leaf for the signed heat distance from a curve. */
export interface DistanceFn {
  (sourceVerts: Int32Array | number[]): Float64Array
  signed(
    curveVerts: Int32Array | number[],
    isLoop?:    boolean,
    levelSet?:  SignedHeatLevelSet,
  ): Float64Array
}

export interface MeasureGroup {
  distance: DistanceFn
  /** TODO (round-1 stub) — area of a region selection. */
  area(region: unknown):                    { method: 'todo'; name: string }
  /** TODO (round-1 stub) — average a scalar field over a region. */
  density(region: unknown, field: unknown): { method: 'todo'; name: string }
  curvature():                              CurvatureResult
  normal(type?: NormalType):                Float64Array
  frame():                                  FaceFrames
}

export interface UvGroup {
  /** Boundary First Flattening — throws on closed meshes. */
  bff(): Float64Array
  logMap(sourceVertex: number, strategy?: LogMapStrategy): LogMapResult
  stripe(
    vertexFieldRaw:           Float64Array,
    uniformFrequency:         number,
    connectOnSingularities?:  boolean,
  ): SegmentsResult
  stripeFreq(
    vertexFieldRaw:           Float64Array,
    frequencies:              Float64Array,
    connectOnSingularities?:  boolean,
  ): SegmentsResult
}

export interface InterpolateGroup {
  transport(
    sourceVerts:   Int32Array  | number[],
    sourceVectors: Float64Array | number[],
  ): Float64Array
  extend(
    sourceVerts:  Int32Array  | number[],
    sourceValues: Float64Array | number[],
  ): Float64Array
  trivial(
    singVerts:  Int32Array  | number[],
    singValues: Float64Array | number[],
  ): DirectionFieldResult
  smoothFace(nSym?: number, alignToCurvature?: boolean): Float64Array
  smoothVertex(nSym?: number, alignToCurvature?: boolean): SmoothVertexFieldResult
}

/** Stateful compute context exposed under the six-group nested
 *  namespace. Same underlying ContextWrapper as the flat surface; both
 *  may be used interchangeably during migration via `_flat`. */
export interface ManifoldContext {
  nV(): number
  nE(): number
  nF(): number

  solve:       SolveGroup
  operator:    OperatorGroup
  query:       QueryGroup
  measure:     MeasureGroup
  uv:          UvGroup
  interpolate: InterpolateGroup

  /** Release WASM heap memory; the context is invalid after dispose(). */
  dispose(): void
  /** Flat compatibility surface (same as `createContext()`'s return). */
  _flat: Manifold
}

/** Functional form of the six-group namespace. Each leaf takes the
 *  ManifoldContext as its first argument and forwards to the matching
 *  per-context method. */
export interface ManifoldFunctional {
  solve: {
    poisson(mctx: ManifoldContext, sourceVerts: Int32Array | number[], sourceValues: Float64Array | number[]): Float64Array
    heat(
      mctx: ManifoldContext,
      sources: Int32Array | number[],
      sourceValues: Float64Array | number[],
      timesteps: Float64Array | number[],
      alpha?: number,
    ): TimeSeriesField
    eigen(mctx: ManifoldContext, k: number, sigma?: number): EigenResult
    hodge(mctx: ManifoldContext, omega: Float64Array): HodgeResult
  }
  operator: {
    d0(mctx: ManifoldContext):           SparseMatrixCOO
    d1(mctx: ManifoldContext):           SparseMatrixCOO
    star0(mctx: ManifoldContext):        SparseMatrixCOO
    star1(mctx: ManifoldContext):        SparseMatrixCOO
    star2(mctx: ManifoldContext):        SparseMatrixCOO
    star1Inverse(mctx: ManifoldContext): SparseMatrixCOO
    mass(mctx: ManifoldContext):         SparseMatrixCOO
    stiffness(mctx: ManifoldContext):    SparseMatrixCOO
    laplacian(mctx: ManifoldContext):    SparseMatrixCOO
    connectionLaplacian(
      mctx:    ManifoldContext,
      options?: ConnectionLaplacianOptions,
    ): ConnectionLaplacianResult
  }
  query: {
    vertex(mctx: ManifoldContext, v: number):                    { vertexIndex: number }
    line(mctx: ManifoldContext, v1: number, v2: number):         { method: 'todo'; name: string }
    circle(mctx: ManifoldContext, v: number, r: number):         { method: 'todo'; name: string }
    region(mctx: ManifoldContext, v: number, r: number):         { method: 'todo'; name: string }
    isoline(mctx: ManifoldContext, field: Float64Array | number[], level: number): SegmentsResult
    center(mctx: ManifoldContext, sourceVerts: Int32Array | number[], p?: number): Float64Array
  }
  measure: {
    distance: {
      (mctx: ManifoldContext, sourceVerts: Int32Array | number[]): Float64Array
      signed(
        mctx: ManifoldContext,
        curveVerts: Int32Array | number[],
        isLoop?: boolean,
        levelSet?: SignedHeatLevelSet,
      ): Float64Array
    }
    area(mctx: ManifoldContext, region: unknown):                    { method: 'todo'; name: string }
    density(mctx: ManifoldContext, region: unknown, field: unknown): { method: 'todo'; name: string }
    curvature(mctx: ManifoldContext):                                CurvatureResult
    normal(mctx: ManifoldContext, type?: NormalType):                Float64Array
    frame(mctx: ManifoldContext):                                    FaceFrames
  }
  uv: {
    bff(mctx: ManifoldContext): Float64Array
    logMap(mctx: ManifoldContext, sourceVertex: number, strategy?: LogMapStrategy): LogMapResult
    stripe(
      mctx: ManifoldContext,
      vertexFieldRaw: Float64Array,
      uniformFrequency: number,
      connectOnSingularities?: boolean,
    ): SegmentsResult
    stripeFreq(
      mctx: ManifoldContext,
      vertexFieldRaw: Float64Array,
      frequencies: Float64Array,
      connectOnSingularities?: boolean,
    ): SegmentsResult
  }
  interpolate: {
    transport(
      mctx: ManifoldContext,
      sourceVerts: Int32Array | number[],
      sourceVectors: Float64Array | number[],
    ): Float64Array
    extend(
      mctx: ManifoldContext,
      sourceVerts: Int32Array | number[],
      sourceValues: Float64Array | number[],
    ): Float64Array
    trivial(
      mctx: ManifoldContext,
      singVerts: Int32Array | number[],
      singValues: Float64Array | number[],
    ): DirectionFieldResult
    smoothFace(mctx: ManifoldContext, nSym?: number, alignToCurvature?: boolean): Float64Array
    smoothVertex(mctx: ManifoldContext, nSym?: number, alignToCurvature?: boolean): SmoothVertexFieldResult
  }
}

export interface InitOptions {
  /** Custom locator for nxr_compute.wasm. */
  locateFile?: (filename: string) => string
}

/**
 * Load the nxr-compute WASM module. Idempotent. The promise resolves to a
 * NxrCompute object exposing the public API.
 */
export function initNxrCompute(options?: InitOptions): Promise<NxrCompute>

export default initNxrCompute
