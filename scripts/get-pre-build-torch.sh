#!/bin/bash
set -euo pipefail

DEFAULT_LIB_TORCH_VERSION="libtorch-shared-with-deps-2.8.0"
DEFAULT_OUTPUT_PATH="../pytorch"

LIB_TORCH_VERSION="${LIB_TORCH_VERSION:-$DEFAULT_LIB_TORCH_VERSION}"
OUTPUT_PATH="${OUTPUT_PATH:-$DEFAULT_OUTPUT_PATH}"

usage() {
  cat <<EOF
Usage: $0 [--version <libtorch-version>] [--output <path>]
Defaults: version=${DEFAULT_LIB_TORCH_VERSION}, output=${DEFAULT_OUTPUT_PATH}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--version)
      LIB_TORCH_VERSION="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

export LIB_TORCH_VERSION
export OUTPUT_PATH

if [ -d "${OUTPUT_PATH}" ]; then
  echo "Removing existing ${OUTPUT_PATH}"
  rm -rf "${OUTPUT_PATH}"
fi

ARCHIVE="${LIB_TORCH_VERSION}+cpu.zip"
URL="https://download.pytorch.org/libtorch/cpu/${LIB_TORCH_VERSION}%2Bcpu.zip"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}" "${ARCHIVE}"' EXIT

wget -O "${ARCHIVE}" "${URL}"
unzip "${ARCHIVE}" -d "${TMP_DIR}"
mv "${TMP_DIR}/libtorch" "${OUTPUT_PATH}"
