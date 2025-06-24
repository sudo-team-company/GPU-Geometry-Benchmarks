#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR="build"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1
cmake --build "${BUILD_DIR}"
