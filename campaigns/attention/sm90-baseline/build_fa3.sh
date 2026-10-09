#!/usr/bin/env bash
set -euo pipefail

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$project/../../.." && pwd)
source "$repo/tools/env.sh"
: "${PIPELINK_CUDA_HOME:?Set PIPELINK_CUDA_HOME in the repository environment}"
export CUDA_HOME="$PIPELINK_CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export TMPDIR="$PIPELINK_SCRATCH_ROOT/tmp/sm90-baseline"
mkdir -p "$TMPDIR"

# Each nvcc process also uses two threads; cap jobs to avoid host-memory spikes.
cpus=$(nproc)
jobs=${MAX_JOBS:-4}
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_JOBS must be a positive integer" >&2
    exit 2
fi
(( jobs > cpus )) && jobs=$cpus
(( jobs > 8 )) && jobs=8
export MAX_JOBS=$jobs
export NVCC_THREADS=2
export TORCH_CUDA_ARCH_LIST=9.0a
export BUILD_TARGET=cuda
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE
export FLASH_ATTENTION_DISABLE_FP16=TRUE
export FLASH_ATTENTION_DISABLE_HDIM128=FALSE
export FLASH_ATTENTION_ENABLE_VCOLMAJOR=FALSE
for feature in BACKWARD SPLIT PAGEDKV APPENDKV FP8 \
    HDIM64 HDIM96 HDIM192 HDIM256 SM80 HDIMDIFF64 HDIMDIFF192; do
    export "FLASH_ATTENTION_DISABLE_${feature}=TRUE"
done

# Keep the shipped cluster dispatch and the bf16 GQA/local/softcap/varlen paths.
for feature in LOCAL SOFTCAP VARLEN PACKGQA CLUSTER; do
    export "FLASH_ATTENTION_DISABLE_${feature}=FALSE"
done
# Remote-only extensions must survive source mirrors, including --out syncs.
output="$PIPELINK_SCRATCH_ROOT/sm90-baseline/fa3"
mkdir -p "$output/flash_attn_3"
cd "$repo/third_party/flash-attention/hopper"
uv run --project "$project" python setup.py build_ext \
    --build-temp "$TMPDIR/build-fa3" --build-lib "$output" -j "$jobs"
cp flash_attn_3/__init__.py "$output/flash_attn_3/"
cp flash_attn_interface.py flash_attn_config.py "$output/"
printf 'Built shipped bf16 d128 FA3: %s\n' "$output"
