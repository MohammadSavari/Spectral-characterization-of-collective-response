#!/bin/bash
# Reassembles one (or all) of package_data.sh's chunked archives in data/ back
# into a whole tar.gz, verifies it against data/checksums_full.sha256, and
# extracts it into nets/.
#
# Archive members are paths relative to nets/ (e.g. LFC/240/ws/...), which is
# where every script and figure notebook expects to find the data.
#
# Usage:
#   bash reassemble.sh <archive-name>   # e.g. bash reassemble.sh LFC_240_ws_gt
#   bash reassemble.sh all              # reassemble + extract every archive
#
# Archives: LFC_240_ws_gt, LFC_240_mhk_gt, spec_div_top_pairs_gt,
# LFC_240_ws_csv, LFC_240_mhk_csv, spec_div_ws_csv, spec_div_mhk_csv,
# spec_div_pair_properties

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ARCHIVES=(
  LFC_240_ws_gt LFC_240_mhk_gt spec_div_top_pairs_gt
  LFC_240_ws_csv LFC_240_mhk_csv
  spec_div_ws_csv spec_div_mhk_csv spec_div_pair_properties
)

reassemble_one() {
  local name="$1"

  if ! compgen -G "data/${name}.tar.gz.part*" > /dev/null; then
    echo "No parts found for ${name} (expected data/${name}.tar.gz.part*)" >&2
    return 1
  fi

  echo "Reassembling data/${name}.tar.gz from parts"
  cat data/"${name}".tar.gz.part* > "data/${name}.tar.gz"

  echo "Verifying checksum"
  grep "data/${name}\.tar\.gz\$" data/checksums_full.sha256 | sha256sum -c -

  echo "Extracting data/${name}.tar.gz into nets/"
  mkdir -p nets
  tar xzf "data/${name}.tar.gz" -C nets

  # The reassembled tarball is a second copy of data now unpacked under nets/,
  # so drop it once extraction succeeded. The chunks are untouched.
  rm -f "data/${name}.tar.gz"

  echo "Done: extracted ${name} into nets/"
}

if [[ $# -ne 1 ]]; then
  echo "Usage: bash reassemble.sh <archive-name>|all" >&2
  echo "Archives: ${ARCHIVES[*]}" >&2
  exit 1
fi

if [[ "$1" == "all" ]]; then
  for name in "${ARCHIVES[@]}"; do
    reassemble_one "$name"
  done
else
  reassemble_one "$1"
fi
