// Smoke test: addon operatorInfo()/fieldInfo() handle-free registry lookups.
// Usage (from repo root): node scripts/_smoke-addon-registry.mjs
import { initNxrCompute } from '../bindings/node/index.mjs'

const nxr = await initNxrCompute()
let failures = 0
const check = (cond, msg) => { if (cond) console.log(`  ✓ ${msg}`); else { console.error(`  ✗ ${msg}`); failures++ } }

const op = nxr.operatorInfo('laplaceBeltrami')
check(op.id === 'laplaceBeltrami', 'operatorInfo id round-trips')
check(typeof op.bundle === 'string' && op.bundle.length > 0, 'operator bundle is a non-empty string')
check(typeof op.graded === 'boolean', 'operator graded is a boolean')
check('input_field' in op && 'output_field' in op, 'operator carries input/output_field')

const fld = nxr.fieldInfo('scalarVertex')
check(fld.id === 'scalarVertex', 'fieldInfo id round-trips')
check(typeof fld.nSym === 'number', 'field nSym is a number')
check(typeof fld.representation === 'string', 'field representation is a string')

let threw = false
try { nxr.operatorInfo('nope_not_a_real_id') } catch (e) { threw = true; check(e.code === 'INVALID_INPUT', 'unknown operator id throws INVALID_INPUT') }
check(threw, 'unknown operator id throws')

console.log(failures === 0 ? '\nPASS' : `\nFAIL (${failures})`)
process.exit(failures === 0 ? 0 : 1)
