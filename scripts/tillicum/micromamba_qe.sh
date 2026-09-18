#!/usr/bin/env bash
# QE fallback when no quantum-espresso module exists (SPEC.md section 11): a static micromamba
# binary plus conda-forge qe=7.5 in <scratch>/qe.  Usage: micromamba_qe.sh <scratch>
# Idempotent: exits 0 at once when <scratch>/qe/bin/pw.x exists.  Executed by
# `b20mlip cluster bootstrap` only with cluster.install_qe / --install-qe (COMMANDS["micromamba_qe"]).
#
# /gpfs/scrubbed purges files untouched for 60 days, which can break conda environments there;
# prefer a /gpfs/projects scratch when the group has one, or simply re-run bootstrap.
set -euo pipefail
scratch="${1:?usage: micromamba_qe.sh <scratch>}"
scratch="${scratch%/}"
prefix="$scratch/qe"
mm="$scratch/bin/micromamba"
export MAMBA_ROOT_PREFIX="$scratch/.micromamba"
if [ -x "$prefix/bin/pw.x" ]; then
  echo "qe: present at $prefix/bin/pw.x"
  exit 0
fi
mkdir -p "$scratch/bin" "$MAMBA_ROOT_PREFIX"
if [ ! -x "$mm" ]; then
  echo "micromamba: downloading the static linux-64 binary"
  curl -Ls --fail --retry 3 \
    https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 \
    -o "$mm"
  chmod +x "$mm"
fi
"$mm" --version
"$mm" create -y -p "$prefix" -c conda-forge qe=7.5
test -x "$prefix/bin/pw.x"
echo "qe: installed at $prefix/bin/pw.x (conda-forge qe=7.5)"
