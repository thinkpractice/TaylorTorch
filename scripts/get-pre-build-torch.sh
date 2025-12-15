#!/bin/bash
set -euo pipefail

DEFAULT_LIB_TORCH_VERSION="2.9.1"
DEFAULT_OUTPUT_PATH="../pytorch"

LIB_TORCH_VERSION="${LIB_TORCH_VERSION:-$DEFAULT_LIB_TORCH_VERSION}"
OUTPUT_PATH="${OUTPUT_PATH:-$DEFAULT_OUTPUT_PATH}"

usage() {
  cat <<EOF
Usage: $0 [--version <version>] [--output <path>]
Defaults: version=${DEFAULT_LIB_TORCH_VERSION}, output=${DEFAULT_OUTPUT_PATH}
The version may be a bare number (e.g., 2.8.0) or a full libtorch package name.
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

PACKAGE_NAME="libtorch-shared-with-deps-${LIB_TORCH_VERSION}"

ARCHIVE="${PACKAGE_NAME}+cpu.zip"
URL="https://download.pytorch.org/libtorch/cpu/${PACKAGE_NAME}%2Bcpu.zip"
TMP_DIR="$(mktemp -d)"
ARCHIVE_PATH="${TMP_DIR}/${ARCHIVE}"
trap 'rm -rf "${TMP_DIR}"' EXIT

wget -O "${ARCHIVE_PATH}" "${URL}"
unzip "${ARCHIVE_PATH}" -d "${OUTPUT_PATH}"
