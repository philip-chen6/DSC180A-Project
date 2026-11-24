#!/usr/bin/env bash
# Source after activating the Python 3.12 venv installed from requirements.txt.
# CUDA's pip wheels use lib/ and versioned .so names; FlashInfer expects lib64/.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Activate the GPU Python venv before sourcing scripts/activate-gpu.sh" >&2
    return 1
fi
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "CUDA compiler not found in the active venv; install requirements.txt" >&2
    return 1
fi
if [[ ! -e "$CUDA_HOME/lib64" ]]; then
    ln -s lib "$CUDA_HOME/lib64" || return 1
fi
if [[ ! -e "$CUDA_HOME/lib/libcudart.so" ]]; then
    ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so" || return 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
if [[ -f /usr/lib/wsl/lib/libcuda.so ]]; then
    export LIBRARY_PATH="/usr/lib/wsl/lib:$LIBRARY_PATH"
fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Keep target arithmetic consistent between single-token decode and verification.
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
