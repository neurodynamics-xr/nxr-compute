#pragma once

// ── nxr::compute::Error — structured exception type ────────────────
//
// Every failure raised by cxf is a Error carrying a stable
// machine-readable ErrorCode plus a human-readable message and
// optional hint. Bindings translate code() to:
//   • JS:    error.code === 'NON_MANIFOLD'  (string-named enumerator)
//   • MATLAB: MException identifier 'cxf:nonManifold'
//   • CLI:   exit code (failure path)
//
// The `what()` string is diagnostic-only — not a stable contract.
// Consumers must switch on code(), never pattern-match on what().
//
// Error derives from std::runtime_error so existing
// `catch (const std::exception&)` handlers keep working during
// the migration; new code should catch Error specifically.

#include <stdexcept>
#include <string>
#include <string_view>

namespace nxr::compute {

enum class ErrorCode {
    InvalidInput,
    NonManifold,
    OpenMeshRequired,
    ClosedMeshRequired,
    GeometryDegenerate,
    EigensolveInvalidK,
    EigensolveNotConverged,
    NotPrecomputed,
    Cancelled,
    OutOfMemory,
    InternalError,
};

// Stringify the enumerator name (e.g. ErrorCode::NonManifold → "NON_MANIFOLD").
// Bindings use this to populate `error.code` on the JS side.
constexpr std::string_view errorCodeName(ErrorCode c) noexcept {
    switch (c) {
        case ErrorCode::InvalidInput:           return "INVALID_INPUT";
        case ErrorCode::NonManifold:            return "NON_MANIFOLD";
        case ErrorCode::OpenMeshRequired:       return "OPEN_MESH_REQUIRED";
        case ErrorCode::ClosedMeshRequired:     return "CLOSED_MESH_REQUIRED";
        case ErrorCode::GeometryDegenerate:     return "GEOMETRY_DEGENERATE";
        case ErrorCode::EigensolveInvalidK:     return "EIGENSOLVE_INVALID_K";
        case ErrorCode::EigensolveNotConverged: return "EIGENSOLVE_NOT_CONVERGED";
        case ErrorCode::NotPrecomputed:         return "NOT_PRECOMPUTED";
        case ErrorCode::Cancelled:              return "CANCELLED";
        case ErrorCode::OutOfMemory:            return "OUT_OF_MEMORY";
        case ErrorCode::InternalError:          return "INTERNAL_ERROR";
    }
    return "UNKNOWN";
}

class Error : public std::runtime_error {
public:
    Error(ErrorCode code, std::string message, std::string hint = {})
        : std::runtime_error(std::move(message)),
          code_(code),
          hint_(std::move(hint)) {}

    ErrorCode        code() const noexcept { return code_; }
    std::string_view hint() const noexcept { return hint_; }

private:
    ErrorCode   code_;
    std::string hint_;
};

} // namespace nxr::compute
