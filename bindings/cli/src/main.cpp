/**
 * nxr-compute — command-line interface around the nxr-compute compute library.
 *
 * v0 closes the offline-precompute gap: take a cortical surface
 * mesh, assemble operators, solve eigenmodes, write a Zarr store
 * the renderer can load directly.
 *
 *   nxr-compute precompute --input <FS-surface.pial> --output <session.zarr> [--k 300]
 *
 * Single-hemisphere only in v0; bilateral support comes later.
 */

#include "nxr/compute.h"
#include "cxf-io/freesurfer.h"
#include "cxf-io/zarr.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

// ── SIGINT (Ctrl-C) → cancellation token ─────────────────────
//
// One process-wide atomic flag, flipped by the SIGINT handler.
// nxr-compute solvers wrap a CancellationToken around it via ctrl_c_token().
// The handler is async-signal-safe (only writes to the atomic).
std::atomic<int32_t> g_sigintFlag{0};

void onSigint(int /*sig*/) {
    g_sigintFlag.store(1, std::memory_order_relaxed);
}

nxr::compute::CancellationToken ctrl_c_token() {
    return nxr::compute::CancellationToken(&g_sigintFlag);
}

constexpr const char* kVersion = "nxr-compute 0.1.0";

void printUsage() {
    std::cout
        << "Usage: nxr-compute <command> [options]\n"
        << "\n"
        << "Commands:\n"
        << "  precompute   Assemble operators + solve eigenmodes for a surface,\n"
        << "               writing the result as a Zarr v2 store.\n"
        << "\n"
        << "Global flags:\n"
        << "  -h, --help      Print this help and exit.\n"
        << "  -v, --version   Print version and exit.\n"
        << "\n"
        << "Run `nxr-compute <command> --help` for command-specific options.\n";
}

void printPrecomputeUsage() {
    std::cout
        << "Usage: nxr-compute precompute --input <path> --output <path> [--k <N>]\n"
        << "\n"
        << "Required:\n"
        << "  --input  <path>   FreeSurfer surface file (e.g. lh.pial).\n"
        << "  --output <path>   Output Zarr store directory (will be created).\n"
        << "\n"
        << "Optional:\n"
        << "  --k <N>           Number of eigenmodes to solve (default: 100).\n"
        << "                    Eigensolve runs at sigma = -1e-8 (smallest k).\n";
}

double elapsedMs(std::chrono::steady_clock::time_point t0) {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - t0).count();
}

int cmdPrecompute(int argc, char** argv) {
    std::string inputPath;
    std::string outputPath;
    int k = 100;

    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            inputPath = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "--k" && i + 1 < argc) {
            k = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            printPrecomputeUsage();
            return 0;
        } else {
            std::cerr << "nxr-compute precompute: unknown argument \"" << arg << "\"\n";
            printPrecomputeUsage();
            return 2;
        }
    }

    if (inputPath.empty() || outputPath.empty()) {
        std::cerr << "nxr-compute precompute: --input and --output are required\n";
        printPrecomputeUsage();
        return 2;
    }
    if (k <= 0) {
        std::cerr << "nxr-compute precompute: --k must be positive (got " << k << ")\n";
        return 2;
    }

    // ── 1. Read FreeSurfer surface ────────────────────────
    std::cout << "[nxr-compute] reading " << inputPath << std::endl;
    auto t0 = std::chrono::steady_clock::now();
    cxf_io::freesurfer::Surface surf;
    try {
        surf = cxf_io::freesurfer::readSurface(inputPath);
    } catch (const std::exception& e) {
        std::cerr << "nxr-compute precompute: failed to read surface: " << e.what() << "\n";
        return 1;
    }
    std::cout << "[nxr-compute]   nV=" << surf.nV << "  nF=" << surf.nF
              << "  (" << elapsedMs(t0) << " ms)" << std::endl;

    // ── 2. ComputeContext (needs double-precision vertices) ──
    std::vector<double> verts64(surf.vertices.size());
    for (std::size_t i = 0; i < surf.vertices.size(); i++) {
        verts64[i] = static_cast<double>(surf.vertices[i]);
    }
    nxr::compute::ComputeContext ctx(verts64.data(), surf.nV, surf.faces.data(), surf.nF);

    // ── 3. Mesh operators (Laplacian, mass, normals, areas) ──
    t0 = std::chrono::steady_clock::now();
    auto ops = nxr::compute::assembleMeshOperators(ctx);
    std::cout << "[nxr-compute] assembled mesh operators (" << elapsedMs(t0) << " ms)" << std::endl;

    // ── 4. Eigensolve → normalize → removeDC ──────────────
    if (k >= surf.nV) {
        std::cerr << "nxr-compute precompute: requested k=" << k
                  << " but the mesh only has " << surf.nV << " vertices.\n";
        return 1;
    }
    t0 = std::chrono::steady_clock::now();
    auto eig = nxr::compute::solveEigenmodes(ops.stiffness, ops.mass, k, -1e-8, ctrl_c_token());
    eig.eigenvectors = nxr::compute::normalizeEigenmodes(eig.eigenvectors, ops.mass);
    eig = nxr::compute::removeDC(eig);
    std::cout << "[nxr-compute] eigensolve k=" << eig.k
              << " (" << elapsedMs(t0) << " ms)" << std::endl;

    // ── 5. Write Zarr ────────────────────────────────────
    t0 = std::chrono::steady_clock::now();
    // Group markers at every level — the renderer's zarr-fs.ts
    // refuses to open a store without `.zgroup` at the root.
    cxf_io::zarr::writeGroup(outputPath, "");
    cxf_io::zarr::writeGroup(outputPath, "manifold");
    cxf_io::zarr::writeGroup(outputPath, "manifold/eigenmodes");

    // Top-level store metadata.
    cxf_io::zarr::writeAttrs(outputPath, "",
        "{\"schema\":\"nxr-compute.session@0.1\",\"format\":\"zarr\",\"zarr_version\":2}");

    // manifold/.zattrs — describes the geometry block.
    {
        std::string a = "{";
        a += "\"schema\":\"nxr-compute.manifold@0.1\",";
        a += "\"numVertices\":" + std::to_string(surf.nV) + ",";
        a += "\"numFaces\":"    + std::to_string(surf.nF) + ",";
        a += "\"source\":\""    + inputPath + "\"";
        a += "}";
        cxf_io::zarr::writeAttrs(outputPath, "manifold", a);
    }

    // Vertices [V, 3] float32 — already row-major xyz triples.
    cxf_io::zarr::writeFloat32(
        outputPath, "manifold/vertices",
        {surf.nV, 3}, surf.vertices.data());

    // Faces [F, 3] uint32 — Zarr convention is unsigned for indices.
    // surf.faces is int32_t but always non-negative; copy through a
    // uint32_t buffer to be explicit about the cast.
    std::vector<std::uint32_t> facesU32(surf.faces.size());
    for (std::size_t i = 0; i < surf.faces.size(); i++) {
        facesU32[i] = static_cast<std::uint32_t>(surf.faces[i]);
    }
    cxf_io::zarr::writeUInt32(
        outputPath, "manifold/faces",
        {surf.nF, 3}, facesU32.data());

    // Eigenmodes block.
    {
        std::string a = "{";
        a += "\"schema\":\"nxr-compute.eigen@0.1\",";
        a += "\"numModes\":"    + std::to_string(eig.k) + ",";
        a += "\"numVertices\":" + std::to_string(surf.nV) + ",";
        a += "\"removedDC\":true,";
        a += "\"normalisation\":\"M-orthonormal\",";
        a += "\"operator\":\"Laplace-Beltrami\",";
        a += "\"basis\":\"P1-FEM\"";
        a += "}";
        cxf_io::zarr::writeAttrs(outputPath, "manifold/eigenmodes", a);
    }

    // Eigenvectors [V, K] float64 — Eigen storage is column-major,
    // Zarr expects row-major. Materialize a row-major copy.
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        evecsRowMajor = eig.eigenvectors;
    cxf_io::zarr::writeFloat64(
        outputPath, "manifold/eigenmodes/eigenvectors",
        {static_cast<int>(eig.eigenvectors.rows()),
         static_cast<int>(eig.eigenvectors.cols())},
        evecsRowMajor.data());

    // Eigenvalues [K, 1] float64. Eigen::VectorXd is K×1 column-major;
    // bytes are identical to row-major [K, 1], so dump directly.
    cxf_io::zarr::writeFloat64(
        outputPath, "manifold/eigenmodes/eigenvalues",
        {static_cast<int>(eig.eigenvalues.size()), 1},
        eig.eigenvalues.data());

    std::cout << "[nxr-compute] wrote Zarr store " << outputPath
              << " (" << elapsedMs(t0) << " ms)" << std::endl;
    std::cout << "[nxr-compute] done." << std::endl;
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    // Wire Ctrl-C → cancel on nxr-compute solvers. The default action would be
    // immediate process termination; the handler instead just sets the
    // flag so nxr-compute throws Error(Cancelled) and we exit cleanly,
    // closing files and surfacing a useful exit code.
    std::signal(SIGINT, onSigint);

    if (argc < 2) {
        printUsage();
        return 2;
    }

    std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help") {
        printUsage();
        return 0;
    }
    if (cmd == "-v" || cmd == "--version") {
        std::cout << kVersion << "\n";
        return 0;
    }

    if (cmd == "precompute") {
        try {
            return cmdPrecompute(argc - 2, argv + 2);
        } catch (const nxr::compute::Error& e) {
            std::cerr << "nxr-compute precompute: ["
                      << nxr::compute::errorCodeName(e.code()) << "] " << e.what() << "\n";
            if (!e.hint().empty()) {
                std::cerr << "  hint: " << e.hint() << "\n";
            }
            // Distinguish cancellation (130 = 128 + SIGINT, POSIX
            // convention) from other failures (1).
            return e.code() == nxr::compute::ErrorCode::Cancelled ? 130 : 1;
        } catch (const std::exception& e) {
            std::cerr << "nxr-compute precompute: " << e.what() << "\n";
            return 1;
        }
    }

    std::cerr << "nxr: unknown command \"" << cmd << "\"\n";
    printUsage();
    return 2;
}
