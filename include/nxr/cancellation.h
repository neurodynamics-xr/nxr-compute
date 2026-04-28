#pragma once

// ── nxr::compute::CancellationToken — binding-agnostic cancel signal ──
//
// Long-running cxf operations accept an optional CancellationToken.
// The solver polls token.requested() at well-defined points (e.g.
// once per Spectra perform_op call inside the eigensolver) and
// throws Error(ErrorCode::Cancelled, ...) when it sees true.
//
// The token has two construction paths so every binding can plug
// in its native cancellation mechanism:
//
//   1. Atomic-flag pointer — for SharedArrayBuffer-backed Int32Array
//      from JS (both the N-API addon and the Embind/WASM module use
//      this). The caller guarantees the storage outlives the token.
//      Non-zero means "stop requested".
//
//   2. std::function<bool()> poller — for any in-process polling
//      mechanism: MATLAB's utIsInterruptPending (so Ctrl-C inside
//      a long MEX call triggers the same cancel path), POSIX SIGINT
//      handlers, future native UI cancel buttons, etc.
//
// Default-constructed tokens are never cancelled (no-op).
//
// Why not std::stop_token: stop_token's only spec-compliant mechanism
// is std::stop_source ownership. Bridging an external atomic flag
// to a stop_token requires either a watcher thread (broken on
// single-threaded WASM) or solver-side awareness (defeats the
// abstraction). A custom token with explicit semantics is honest
// about how cancellation actually flows. C++ users who want
// stop_token interop can build the bridge themselves.

#include <atomic>
#include <cstdint>
#include <functional>
#include <utility>

namespace nxr::compute {

class CancellationToken {
public:
    // Default-constructed: never cancelled.
    CancellationToken() noexcept = default;

    // Atomic-flag bridge. The pointed-to atomic must outlive this
    // token (caller's responsibility — typically a SharedArrayBuffer
    // backing on the JS side, or a stack/static atomic on native).
    explicit CancellationToken(const std::atomic<int32_t>* flag) noexcept
        : flag_(flag) {}

    // Function-poll bridge. Invoked on every requested() call.
    // Used by MEX (utIsInterruptPending), CLI (signal handler),
    // or any custom in-process source.
    explicit CancellationToken(std::function<bool()> poll)
        : poll_(std::move(poll)) {}

    // Copy/move are fine — std::function is the only non-trivial
    // member, and it's standard library copyable.
    CancellationToken(const CancellationToken&)            = default;
    CancellationToken(CancellationToken&&) noexcept        = default;
    CancellationToken& operator=(const CancellationToken&) = default;
    CancellationToken& operator=(CancellationToken&&) noexcept = default;

    // O(1) when the token wraps an atomic (one relaxed load) or is
    // default-constructed (one branch). std::function-backed tokens
    // pay one virtual call per check; the caller is expected to
    // poll only at coarse intervals (e.g. once per Spectra
    // perform_op, not once per inner-loop SpMV element).
    //
    // Not noexcept because std::function::operator() may throw if
    // the held callable throws. Atomic-flag and default paths are
    // noexcept in practice; the compiler will inline them away.
    bool requested() const {
        if (flag_) {
            return flag_->load(std::memory_order_relaxed) != 0;
        }
        if (poll_) {
            return poll_();
        }
        return false;
    }

private:
    const std::atomic<int32_t>* flag_ = nullptr;
    std::function<bool()>       poll_;
};

} // namespace nxr::compute
