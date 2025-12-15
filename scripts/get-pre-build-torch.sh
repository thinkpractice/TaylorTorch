#!/bin/bash
set -euo pipefail

DEFAULT_LIB_TORCH_VERSION="2.9.1"
DEFAULT_OUTPUT_PATH=".."
DEFAULT_COMPUTE_PLATFORM="cpu"

detect_default_platform() {
  case "$(uname -s)" in
    Darwin)
      # Default to macOS arm64 build when running on macOS.
      echo "macos-arm64"
      ;;
    *)
      echo "${DEFAULT_COMPUTE_PLATFORM}"
      ;;
  esac
}

# Detect the platform automatically
DETECTED_PLATFORM="$(detect_default_platform)"

LIB_TORCH_VERSION="${LIB_TORCH_VERSION:-$DEFAULT_LIB_TORCH_VERSION}"
OUTPUT_PATH="${OUTPUT_PATH:-$DEFAULT_OUTPUT_PATH}"
COMPUTE_PLATFORM="${LIB_TORCH_COMPUTE_PLATFORM:-$DETECTED_PLATFORM}"

usage() {
  cat <<EOF
Usage: $0 [--version <version>] [--output <path>] [--platform <cpu|cu126|cu128|cu130|rocm6.4|macos-arm64>]
Defaults: version=${DEFAULT_LIB_TORCH_VERSION}, output=${DEFAULT_OUTPUT_PATH}, platform=${DETECTED_PLATFORM} (auto-detects macOS arm64)
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
    -p|--platform|--compute)
      COMPUTE_PLATFORM="$2"
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
export LIB_TORCH_COMPUTE_PLATFORM="${COMPUTE_PLATFORM}"

if [ -d "${OUTPUT_PATH}/libtorch" ]; then
  echo "Removing existing ${OUTPUT_PATH}/libtorch"
  rm -rf "${OUTPUT_PATH}/libtorch"
fi

case "${COMPUTE_PLATFORM}" in
  cpu|cu126|cu128|cu130|rocm6.4|macos-arm64)
    ;;
  *)
    echo "Unsupported compute platform: ${COMPUTE_PLATFORM}"
    echo "Supported platforms: cpu, cu126, cu128, cu130, rocm6.4, macos-arm64"
    exit 1
    ;;
esac

if [[ "${LIB_TORCH_VERSION}" == libtorch-* ]]; then
  PACKAGE_NAME="${LIB_TORCH_VERSION}"
else
  if [[ "${COMPUTE_PLATFORM}" == "macos-arm64" ]]; then
    PACKAGE_NAME="libtorch-macos-arm64-${LIB_TORCH_VERSION}"
  else
    PACKAGE_NAME="libtorch-shared-with-deps-${LIB_TORCH_VERSION}"
  fi
fi

if [[ "${COMPUTE_PLATFORM}" == "macos-arm64" ]]; then
  ARCHIVE="${PACKAGE_NAME}.zip"
  URL="https://download.pytorch.org/libtorch/cpu/${PACKAGE_NAME}.zip"
else
  ARCHIVE="${PACKAGE_NAME}+${COMPUTE_PLATFORM}.zip"
  URL="https://download.pytorch.org/libtorch/${COMPUTE_PLATFORM}/${PACKAGE_NAME}%2B${COMPUTE_PLATFORM}.zip"
fi

TMP_DIR="$(mktemp -d)"
ARCHIVE_PATH="${TMP_DIR}/${ARCHIVE}"
trap 'rm -rf "${TMP_DIR}"' EXIT

# Download archive into tmp directory and unzip it
wget -O "${ARCHIVE_PATH}" "${URL}"
unzip "${ARCHIVE_PATH}" -d "${OUTPUT_PATH}"
