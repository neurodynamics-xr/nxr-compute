#pragma once

// ── nxr::compute::ProgressObserver — lock-free progress slots ─────────
//
// Long-running cxf operations accept an optional ProgressObserver.
// The struct holds three pointers to atomic<int32_t> slots; the
// solver writes them as it makes progress. Default-constructed
// (all pointers null) means "discard progress" — used as the
// default argument so simple call sites stay short.
//
// All slots are owned by the caller; the solver only writes.
// The caller's storage is typically a SharedArrayBuffer-backed
// Int32Array (so a JS UI thread can read the same atomics
// concurrently), but a stack-allocated atomic works equally well
// for native or MATLAB consumers.
//
// Why atomic-pointer instead of std::function callback:
//   • Allocation-free; safe to construct on hot paths.
//   • Cross-language: a SharedArrayBuffer slot in JS is the same
//     memory cell the C++ side stores into.
//   • No Asyncify under WASM (a JS callback would balloon the
//     bundle ~60% by requiring asynchronous unwinding).
//   • C++ users can poll the same atomics from any thread.
//
// Semantics of the slots:
//   • iteration       — opaque monotonically-increasing counter.
//                       The solver decides what counts as a unit
//                       of work; document per operation.
//   • totalIterations — high-water bound for `iteration`. May be
//                       updated mid-run if the solver discovers
//                       more work than originally estimated.
//   • residualMicro   — current residual norm × 1e6, fits int32.
//                       0 if the solver has no residual concept.

#include <atomic>
#include <cstdint>

namespace nxr::compute {

struct ProgressObserver {
    std::atomic<int32_t>* iteration       = nullptr;
    std::atomic<int32_t>* totalIterations = nullptr;
    std::atomic<int32_t>* residualMicro   = nullptr;

    // Write all three slots in one call. memory_order_relaxed is
    // sufficient — observers polling these counters don't need a
    // happens-before relationship with the solver's other state;
    // they just need to see eventually-consistent values.
    void update(int iter, int total, double residual) const noexcept {
        if (iteration) {
            iteration->store(iter, std::memory_order_relaxed);
        }
        if (totalIterations) {
            totalIterations->store(total, std::memory_order_relaxed);
        }
        if (residualMicro) {
            // Saturate to int32 range to avoid UB on very large residuals.
            const double scaled = residual * 1e6;
            int32_t v;
            if (scaled >= 2147483647.0) {
                v = 2147483647;
            } else if (scaled <= -2147483648.0) {
                v = -2147483648;
            } else {
                v = static_cast<int32_t>(scaled);
            }
            residualMicro->store(v, std::memory_order_relaxed);
        }
    }
};

} // namespace nxr::compute
