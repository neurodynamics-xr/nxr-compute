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

export interface MeshOperators {
  /** Cotangent Laplacian (PSD, symmetrized), V×V sparse */
  stiffness:   SparseMatrixCOO
  /** Voronoi mass matrix (SPD, diagonal), V×V sparse */
  mass:        SparseMatrixCOO
  /** Per-vertex dual area, V */
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
  operators:  MeshOperators
  dec:        DECOperators
  eigenmodes: EigenResult
  faceFrames: FaceFrames
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
   *  Pass back into computeStripePattern as-is. */
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
 * Stateful compute context for one mesh. Holds the nxr::compute::ComputeContext
 * in WASM linear memory plus cached operators / Cholesky factors /
 * eigenmodes. Call `delete()` when finished.
 */
export interface ComputeContext {
  nV(): number
  nE(): number
  nF(): number

  // Operators / geometry
  assembleMeshOperators(): MeshOperators
  assembleDECOperators():  DECOperators
  computeFaceFrames():     FaceFrames
  computeVertexNormals(type?: NormalType): Float64Array

  // Spectral
  solveEigenmodes(k: number, sigma?: number): EigenResult
  normalizeEigenmodes(U: Float64Array, rows: number, cols: number): Float64Array
  removeDC(eig: EigenResult): EigenResult

  /**
   * One-shot precompute: returns the visualization-defaults pack
   * (operators + DEC + eigenmodes + face frames) in one call.
   * Cheaper than calling the four steps separately.
   */
  precompute(options?: { k?: number; sigma?: number }): PrecomputeResult

  // Solvers
  solvePoisson(sourceVerts: Int32Array | number[], sourceValues: Float64Array | number[]): Float64Array
  computeGeodesicDistance(sourceVerts: Int32Array | number[]): Float64Array
  tracePath(vStart: number, vEnd: number): PolylineResult
  hodgeDecompose(omega: Float64Array): HodgeResult

  // Geometric
  computeCurvatures(): CurvatureResult
  /** Throws if the mesh has no boundary (BFF requires open mesh). */
  computeUVCoordinates(): Float64Array
  computeIsolines(scalars: Float64Array, numLevels: number, minVal?: number, maxVal?: number): SegmentsResult
  computeDirectionField(
    singVerts:  Int32Array  | number[],
    singValues: Float64Array | number[],
  ): DirectionFieldResult
  traceStreamlines(
    faceField: Float64Array,
    numSeeds?: number,
    stepCoef?: number,
    maxSteps?: number,
  ): SegmentsResult

  // Vector field
  whitneyInterpolate(oneForm: Float64Array): Float64Array
  scalarGradient(scalar: Float64Array): Float64Array

  // Time-varying generators (require eigenmodes computed first)
  generateHeatDiffusion(
    sources:      Int32Array  | number[],
    sourceValues: Float64Array | number[],
    timesteps:    Float64Array | number[],
    alpha?: number,
  ): TimeSeriesField
  generateDampedWave(
    modeIndices: Int32Array  | number[],
    amplitudes:  Float64Array | number[],
    dampings:    Float64Array | number[],
    phases:      Float64Array | number[],
    timesteps:   Float64Array | number[],
  ): TimeSeriesField
  generateRandomDecomposed1Form(
    alphaStrength: number,
    betaStrength:  number,
    gammaStrength: number,
    seed?: number,
  ): Float64Array

  // Vector heat method (Sharp, Soliman, Crane 2019)
  /** Parallel-transport tangent vectors. sourceVectors is N*3 world-space. Returns V*3 row-major. */
  vectorHeatTransport(
    sourceVerts:   Int32Array  | number[],
    sourceVectors: Float64Array | number[],
  ): Float64Array
  /** Smoothly extend a sparse scalar from the given source vertices to all V. */
  vectorHeatExtendScalar(
    sourceVerts:  Int32Array  | number[],
    sourceValues: Float64Array | number[],
  ): Float64Array
  /** Logarithmic map at one source vertex. */
  vectorHeatLogMap(
    sourceVertex: number,
    strategy?:    LogMapStrategy,
  ): LogMapResult
  /** Karcher mean of source vertices. Returns 3 floats (xyz). */
  vectorHeatFindCenter(
    sourceVerts: Int32Array | number[],
    p?:          number,
  ): Float64Array

  // Signed heat method (Feng & Crane 2024)
  /** Signed geodesic distance from a curve given as ordered vertex indices. */
  signedHeatDistance(
    curveVerts: Int32Array | number[],
    isLoop:     boolean,
    levelSet?:  SignedHeatLevelSet,
  ): Float64Array

  // Smooth direction fields (Knöppel-Crane)
  /** Smoothest face-based direction field, lifted to F*3 world. nSym=4 → cross field. */
  computeSmoothFaceField(
    nSym?:             number,
    alignToCurvature?: boolean,
  ): Float64Array
  /** Smoothest vertex-based direction field. Use the returned `vertexFieldRaw`
   *  as input to computeStripePattern. nSym=2 typical for stripes. */
  computeSmoothVertexField(
    nSym?:             number,
    alignToCurvature?: boolean,
  ): SmoothVertexFieldResult

  // Stripe patterns (Knöppel-Crane SIGGRAPH 2015)
  /** Sinusoidal stripes aligned to a 2-RoSy field, returned as 3D line segments. */
  computeStripePattern(
    vertexFieldRaw:           Float64Array,
    uniformFrequency:         number,
    connectOnSingularities?:  boolean,
  ): SegmentsResult
  /** Stripe pattern with a per-vertex frequency (length V). */
  computeStripePatternFreq(
    vertexFieldRaw:           Float64Array,
    frequencies:              Float64Array,
    connectOnSingularities?:  boolean,
  ): SegmentsResult

  /** Release WASM heap memory; the context is invalid after delete(). */
  delete(): void
}

export interface NxrCompute {
  version(): string
  createContext(
    vertices: Float64Array | number[],
    faces:    Int32Array  | number[],
  ): ComputeContext
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
