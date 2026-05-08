#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/johnterry/Documents/Projects/Di-quark-pythia"
PYTHIA_ROOT="/Users/johnterry/pythia8312"

SRC="$ROOT/scripts/cpp/generate_dis_remnant_diagnostic.cc"
OUT="$ROOT/scripts/cpp/generate_dis_remnant_diagnostic"

g++ -std=c++17 -O2 \
  -I"$PYTHIA_ROOT/include" \
  "$SRC" \
  -L"$PYTHIA_ROOT/lib" -lpythia8 \
  -Wl,-rpath,"$PYTHIA_ROOT/lib" \
  -o "$OUT"

echo "Built: $OUT"

