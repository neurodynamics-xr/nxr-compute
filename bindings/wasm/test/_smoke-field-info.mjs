// Smoke test for the fieldInfo Embind binding and operatorInfo field I/O fields.
// Mirrors the MEX cmdFieldInfo command (bindings/mex/src/nxr_compute_mex.cpp).
//
// Usage (from repo root):
//   node bindings/wasm/test/_smoke-field-info.mjs

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

// ── Test 1: fieldInfo for tangentVertex ──────────────────────
const fi = Module.fieldInfo('tangentVertex')
require(fi.bundle         === 'tangent',          `bundle: expected 'tangent', got '${fi.bundle}'`)
require(fi.field_type     === 'complex',          `field_type: expected 'complex', got '${fi.field_type}'`)
require(fi.representation === 'intrinsic_complex',`representation: expected 'intrinsic_complex', got '${fi.representation}'`)
require(typeof fi.id      === 'string' && fi.id.length > 0, 'id is non-empty string')
require(typeof fi.label   === 'string' && fi.label.length > 0, 'label is non-empty string')
require(typeof fi.domain  === 'string', 'domain is string')
require(typeof fi.n_form  === 'string', 'n_form is string')
require(typeof fi.gauge   === 'string', 'gauge is string')
require(typeof fi.nSym    === 'number', 'nSym is number')
require(typeof fi.notes   === 'string', 'notes is string')
console.log(`  tangentVertex ✓ (bundle=${fi.bundle}, field_type=${fi.field_type}, representation=${fi.representation})`)

// ── Test 2: operatorInfo now includes input_field / output_field ──
const oi = Module.operatorInfo('laplaceBeltrami')
require(oi.input_field  === 'scalarVertex', `operatorInfo input_field: expected 'scalarVertex', got '${oi.input_field}'`)
require(oi.output_field === 'scalarVertex', `operatorInfo output_field: expected 'scalarVertex', got '${oi.output_field}'`)
console.log(`  laplaceBeltrami field I/O ✓ (input=${oi.input_field}, output=${oi.output_field})`)

// ── Test 3: unknown id throws ─────────────────────────────────
let threw = false
try { Module.fieldInfo('doesNotExist') } catch { threw = true }
require(threw, 'fieldInfo with unknown id should throw')
console.log('  unknown id throws ✓')

console.log('fieldInfo WASM smoke PASSED')
