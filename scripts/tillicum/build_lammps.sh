#!/usr/bin/env bash
# Build ACEsuit/lammps (branch `mace`, pinned SHA) with libtorch — plus Kokkos/CUDA on a GPU
# partition — and install `lmp` into $B20_PREFIX/bin; $B20_PREFIX/bin/build.json records the
# SHA, cmake flags, versions and the build log path (SPEC.md section 8).
#
#   build_lammps.sh fetch   # login node (COMMANDS["lammps_fetch"]): clone + libtorch download
#   build_lammps.sh build   # compute node (the build_lammps.sbatch.j2 unit): cmake, make, install
#
# Environment: B20_PREFIX (required, e.g. <scratch>/b20-mlip), B20_LIBTORCH_URL (required),
# B20_LAMMPS_SHA (empty = head of `mace`; the SHA actually built is recorded either way),
# B20_CUDA (1 = Kokkos+CUDA, 0 = CPU/OpenMP), B20_KOKKOS_ARCH (HOPPER90 = H200), B20_JOBS (make -j),
# B20_FORCE=1 (rebuild although lmp exists), B20_NO_CPU_FALLBACK=1 (fail instead of retrying).
# Fallback (SPEC.md section 11): a failed CUDA build is retried once as a CPU/OpenMP build and
# recorded as "fallback": "cpu"; if that fails too the unit fails and lammps_cmd stays null.
set -euo pipefail

mode="${1:-build}"
: "${B20_PREFIX:?B20_PREFIX is required}"
: "${B20_LIBTORCH_URL:?B20_LIBTORCH_URL is required}"
B20_LAMMPS_SHA="${B20_LAMMPS_SHA:-}"
B20_CUDA="${B20_CUDA:-0}"
B20_KOKKOS_ARCH="${B20_KOKKOS_ARCH:-HOPPER90}"
B20_JOBS="${B20_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
LAMMPS_REPO="https://github.com/ACEsuit/lammps"
LAMMPS_BRANCH="mace"

work="$B20_PREFIX/build"
src="$work/lammps-src"
libtorch="$work/libtorch"
builddir="$work/lammps-build"
bindir="$B20_PREFIX/bin"
mkdir -p "$work" "$bindir"
FLAGS_USED=""
FALLBACK=""

fetch() {
  if [ ! -f "$libtorch/build-version" ]; then
    echo "libtorch: downloading $B20_LIBTORCH_URL"
    curl -L --fail --retry 3 -o "$work/libtorch.zip" "$B20_LIBTORCH_URL"
    rm -rf "$libtorch"
    unzip -q -o "$work/libtorch.zip" -d "$work"   # unpacks into $work/libtorch
    rm -f "$work/libtorch.zip"
  fi
  echo "libtorch: $(cat "$libtorch/build-version")"
  if [ ! -d "$src/.git" ]; then
    echo "lammps: cloning $LAMMPS_REPO (branch $LAMMPS_BRANCH, shallow)"
    git clone --branch "$LAMMPS_BRANCH" --depth=1 "$LAMMPS_REPO" "$src"
  fi
  if [ -n "$B20_LAMMPS_SHA" ]; then
    if ! git -C "$src" cat-file -e "${B20_LAMMPS_SHA}^{commit}" 2>/dev/null; then
      git -C "$src" fetch --depth=1 origin "$B20_LAMMPS_SHA" \
        || git -C "$src" fetch --unshallow origin "$LAMMPS_BRANCH"
    fi
    git -C "$src" checkout --quiet --detach "$B20_LAMMPS_SHA"
  else
    git -C "$src" fetch --depth=1 origin "$LAMMPS_BRANCH"
    git -C "$src" checkout --quiet --detach FETCH_HEAD
  fi
  echo "lammps: checkout $(git -C "$src" rev-parse HEAD)"
}

configure_and_make() {  # $1 = 1 for Kokkos+CUDA, 0 for CPU/OpenMP
  local cuda="$1"
  rm -rf "$builddir"
  mkdir -p "$builddir"
  cd "$builddir"
  local mpi=OFF
  command -v mpicxx >/dev/null 2>&1 && mpi=ON
  local flags=(
    -D CMAKE_BUILD_TYPE=Release
    -D CMAKE_INSTALL_PREFIX="$B20_PREFIX"
    # libtorch >= 2.9 headers use C++20 `requires` clauses (Tillicum 2026-09-19: gcc 13.4 with
    # -std=c++17 fails in ATen/core/TensorBase.h); the MACE docs' C++17 predates that.
    -D CMAKE_CXX_STANDARD="${B20_CXX_STANDARD:-20}"
    -D CMAKE_CXX_STANDARD_REQUIRED=ON
    -D BUILD_MPI="$mpi"
    -D BUILD_OMP=ON
    -D PKG_OPENMP=ON
    -D PKG_ML-MACE=ON
    -D CMAKE_PREFIX_PATH="$libtorch"
    # libtorch's caffe2::mkl target lists ${MKL_INCLUDE_DIR}; the zip ships no MKL headers and
    # CMake refuses a NOTFOUND include path, so point it at any existing directory (ML-MACE
    # community workaround; MKL is not used by the MACE pair style).
    -D MKL_INCLUDE_DIR="${B20_MKL_INCLUDE_DIR:-/usr/include}"
  )
  if [ "$cuda" = 1 ]; then
    flags+=(
      -D PKG_KOKKOS=ON
      -D Kokkos_ENABLE_CUDA=ON
      -D Kokkos_ENABLE_OPENMP=ON
      # in-tree Kokkos 4.x does not propagate its tpls/mdspan include dir to the LAMMPS
      # targets (atom.cpp: "mdspan/mdspan.hpp: No such file"); mdspan is experimental and unused
      -D Kokkos_ENABLE_IMPL_MDSPAN=OFF
      -D "Kokkos_ARCH_${B20_KOKKOS_ARCH}=ON"
      -D CMAKE_CXX_COMPILER="$src/lib/kokkos/bin/nvcc_wrapper"
      -D BUILD_SHARED_LIBS=ON
    )
  fi
  FLAGS_USED="${flags[*]}"
  echo "cmake ${flags[*]} $src/cmake"
  cmake "${flags[@]}" "$src/cmake"
  make -j "$B20_JOBS"
  make install
}

write_build_json() {
  local sha="$1" cuda_effective="$B20_CUDA"
  [ -n "$FALLBACK" ] && cuda_effective=0
  B20_JSON_SHA="$sha" \
  B20_JSON_FLAGS="$FLAGS_USED" \
  B20_JSON_CUDA="$cuda_effective" \
  B20_JSON_FALLBACK="$FALLBACK" \
  B20_JSON_LIBTORCH="$(cat "$libtorch/build-version" 2>/dev/null || echo unknown)" \
  B20_JSON_LOG="${B20_WORKDIR:-$work}/logs/${B20_UNIT:-build}.log" \
  B20_JSON_CMAKE="$(cmake --version 2>/dev/null | head -1 || echo unknown)" \
  B20_JSON_CXX="$(c++ --version 2>/dev/null | head -1 || echo unknown)" \
  B20_JSON_NVCC="$(nvcc --version 2>/dev/null | tail -1 || true)" \
  python3 - "$bindir/build.json" "$bindir/lmp" <<'PY'
import datetime, json, os, sys
out = {
    "lammps_repo": "https://github.com/ACEsuit/lammps",
    "branch": "mace",
    "sha": os.environ["B20_JSON_SHA"],
    "cmake_flags": os.environ["B20_JSON_FLAGS"].split(),
    "cuda": os.environ["B20_JSON_CUDA"] == "1",
    "fallback": os.environ["B20_JSON_FALLBACK"] or None,
    "kokkos_arch": os.environ.get("B20_KOKKOS_ARCH"),
    "libtorch": os.environ["B20_JSON_LIBTORCH"],
    "libtorch_url": os.environ.get("B20_LIBTORCH_URL"),
    "cmake": os.environ["B20_JSON_CMAKE"],
    "compiler": os.environ["B20_JSON_CXX"],
    "nvcc": os.environ["B20_JSON_NVCC"] or None,
    "log": os.environ["B20_JSON_LOG"],
    "lmp": sys.argv[2],
    "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
    fh.write("\n")
print(json.dumps(out, indent=2))
PY
}

build() {
  if [ -x "$bindir/lmp" ] && [ -f "$bindir/build.json" ] && [ "${B20_FORCE:-0}" != 1 ]; then
    echo "lmp already built (B20_FORCE=1 rebuilds):"
    cat "$bindir/build.json"
    return 0
  fi
  fetch
  local sha
  sha="$(git -C "$src" rev-parse HEAD)"
  if [ "$B20_CUDA" = 1 ]; then
    if ! configure_and_make 1; then
      if [ "${B20_NO_CPU_FALLBACK:-0}" = 1 ]; then
        echo "CUDA build failed and B20_NO_CPU_FALLBACK=1" >&2
        return 1
      fi
      echo "CUDA build failed; retrying once as a CPU/OpenMP build (SPEC.md section 11)"
      FALLBACK="cpu"
      configure_and_make 0
    fi
  else
    configure_and_make 0
  fi
  test -x "$bindir/lmp"
  write_build_json "$sha"
  echo "lmp installed: $bindir/lmp"
}

case "$mode" in
  fetch) fetch ;;
  build) build ;;
  *) echo "usage: build_lammps.sh fetch|build" >&2; exit 2 ;;
esac
