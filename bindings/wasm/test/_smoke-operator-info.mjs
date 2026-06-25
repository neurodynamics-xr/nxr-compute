// Smoke test for the operatorInfo Embind binding.
// Mirrors the MEX operatorInfo command (bindings/mex/src/nxr_compute_mex.cpp).
//
// Usage (from repo root):
//   node bindings/wasm/test/_smoke-operator-info.mjs

import path from 'node:path'
import { pathToFileURL, fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot  = path.resolve(__dirname, '../../..')

const wasmJsPath = path.join(repoRoot, 'build_wasm', 'nxr_compute.js')
const moduleUrl  = pathToFileURL(wasmJsPath).href

console.log(`[smoke] loading ${wasmJsPath}`)
const { default: createNxrComputeModule } = await import(moduleUrl)
const Module = await createNxrComputeModule()
console.log(`[smoke] module ready — version: ${Module.version()}`)

function require(cond, msg) {
  if (!cond) {
    console.error('FAIL:', msg)
    process.exit(1)
  }
}

// ── Test 1: laplaceBeltrami metadata ─────────────────────────
const info = Module.operatorInfo('laplaceBeltrami')
require(info.bundle    === 'scalar', `bundle: expected 'scalar', got '${info.bundle}'`)
require(info.order     === 'second', `order: expected 'second', got '${info.order}'`)
require(info.role      === 'laplacian', `role: expected 'laplacian', got '${info.role}'`)
require(info.status    === 'built',  `status: expected 'built', got '${info.status}'`)
require(typeof info.id    === 'string' && info.id.length > 0, 'id is non-empty string')
require(typeof info.label === 'string' && info.label.length > 0, 'label is non-empty string')
require(typeof info.natural_mass === 'string', 'natural_mass is string')
require(typeof info.graded       === 'boolean', 'graded is boolean')
require(typeof info.tau_presets  === 'string', 'tau_presets is string')
require(typeof info.notes        === 'string', 'notes is string')
require(typeof info.squares_to   === 'string', 'squares_to is string')
require(typeof info.square_of    === 'string', 'square_of is string')
require(typeof info.relation     === 'string', 'relation is string')
console.log(`  laplaceBeltrami ✓ (bundle=${info.bundle}, order=${info.order}, status=${info.status})`)

// ── Test 2: extrinsicWeitzenbockLaplacian status = planned ────
const ew = Module.operatorInfo('extrinsicWeitzenbockLaplacian')
require(ew.status === 'planned', `Weitzenbock status: expected 'planned', got '${ew.status}'`)
console.log(`  extrinsicWeitzenbockLaplacian ✓ (status=${ew.status})`)

// ── Test 3: unknown id throws ─────────────────────────────────
let threw = false
try { Module.operatorInfo('doesNotExist') } catch { threw = true }
require(threw, 'operatorInfo with unknown id should throw')
console.log('  unknown id throws ✓')

console.log('operatorInfo WASM smoke PASSED')
