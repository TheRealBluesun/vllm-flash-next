#!/usr/bin/env bash
# Serve RadixArk Qwen3.8-Flash-Next-NVFP4 via vLLM on 1x RTX PRO 6000 (sm_120).
#
# Tree: peakcrosser7/vllm release/qwen38next_offload @ 357e054 (+ local patches)
#   (qwen4_exp + PLE CPU offload + mixed NVFP4-experts/FP8-PLE dispatch).
# Does not touch /opt/d/vllm-qwen38 (the 27B venv).
#
# Split: NVFP4 experts + MTP + QSA/GDN on GPU; 51B n-gram/PLE table in host RAM
# (~48 GiB FP8). KV is cheap (12 QSA layers, BF16 only — do not pass fp8 KV).
#
# MTP k=3 and 262144 are mutually exclusive on 96 GiB (need ~7.6 GiB KV, MTP
# leaves ~6.7). Default is 184320 / max-num-seqs 4.
# Full native context: MAX_MODEL_LEN=262144 SPEC=none.
#
# Proven on this box (64 GiB swap at /opt/d/swapfile, locally compiled kernels):
# MTP k=3, FULL_DECODE_ONLY graphs, util 0.93, ~90 tok/s. Vision tower is ~1.2 GiB
# BF16 (model-bf16-00001); text-only left enough VRAM. LANGUAGE_MODEL_ONLY=1 skips it.
# Needs ~51 GiB host for the FP8 PLE table (pages into that swap). Do not omit
# swap. CUDA 13.1 (13.2 is the default /usr/local/cuda symlink; an earlier "13.2 crash"
# was likely host OOM before swap was added — untested since). Eager / no-MTP is SPEC=none ENFORCE_EAGER=1 GPU_MEMORY_UTILIZATION=0.85.
#
# Default listen is 127.0.0.1:8000. Hermes Flash-Next currently points at :8080;
# drop-in: PORT=8080. Refuses to start if the card already has >4 GiB unless FORCE=1.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
cd "$REPO_DIR"

if [[ ! -x "$REPO_DIR/.venv/bin/vllm" ]]; then
    printf 'vllm not installed in %s/.venv\n' "$REPO_DIR" >&2
    printf 'Install with:\n' >&2
    printf '  cd %s && source .venv/bin/activate\n' "$REPO_DIR" >&2
    printf '  PATH=/usr/local/cuda-13.1/bin:$PATH CUDA_HOME=/usr/local/cuda-13.1 \\\n' >&2
    printf '    VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=nightly \\\n' >&2
    printf '    uv pip install -e . --torch-backend=auto\n' >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$REPO_DIR/.venv/bin/activate"
# This workspace is often /opt/d/vllm-qwen38; an empty sys.path[0] would import
# that tree's vllm instead of this checkout.
export PYTHONPATH="$REPO_DIR"
echo "using vllm tree: $REPO_DIR ($(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown))"

# Only expose the RTX PRO 6000. With the 2080 SUPER (sm_75, desktop) visible,
# FlashInfer JITs for 75+120 into a new cache and rebuilds fused_moe from scratch.
# Pin by UUID so PCI/enumeration order changes cannot swap cards.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Set to your GPU UUID from `nvidia-smi -L`.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-REPLACE-WITH-YOUR-UUID}"

export PATH=/usr/local/cuda-13.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.1
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}

# $HOME is eCryptfs (143-byte filename limit) — redirect all ~/.cache off it.
mkdir -p /opt/d/caches /opt/d/caches/flashinfer-home
export XDG_CACHE_HOME=/opt/d/caches
export FLASHINFER_WORKSPACE_BASE=/opt/d/caches/flashinfer-home
# Keep this off /mnt/nvme1 (nearly full).
export HF_HOME="${HF_HOME:-/opt/d/caches/huggingface}"

export SAFETENSORS_FAST_GPU=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_PLE_CPU_OFFLOAD=1
# Let the PLE notification thread run while CUDA kernel launches wait.
export TORCHINDUCTOR_USE_FAST_TRITON_LAUNCHER="${TORCHINDUCTOR_USE_FAST_TRITON_LAUNCHER:-0}"
# In-startup JIT compiles (FlashInfer/Inductor) run while the ~48 GiB PLE table
# loads; 24 nvcc jobs + that OOM-killed the box (2026-09-23). 8 fits. Standalone
# prebuilds (ninja -C <cache dir>) can use 24.
export MAX_JOBS="${MAX_JOBS:-8}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"
# sm_120 workstation: FlashInfer CUTLASS NVFP4/FP8 SMEM-overflows; Marlin is the
# working path. DeepGEMM is slower than Marlin here even when it boots.
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
if [[ "${EXPANDABLE_SEGMENTS:-0}" == "1" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0f}"
# Optional. Must be a multiple of scheduler_block_size, which is config-dependent
# (800 with MTP graphs, 1568 on this eager TP1 profile). Leave unset unless you
# know the block size from a prior boot.
if [[ -n "${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-}" ]]; then
    export VLLM_PREFIX_CACHE_RETENTION_INTERVAL
fi

MODEL="${MODEL:-/opt/d/models/Qwen3.8-Flash-Next-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-Flash-Next}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-184320}"

if [[ ! -f "$MODEL/config.json" ]]; then
    printf 'model not found (missing config.json): %s\n' "$MODEL" >&2
    printf 'Download with:\n' >&2
    printf '  HF_HOME=/opt/d/caches/huggingface HF_XET_HIGH_PERFORMANCE=1 \\\n' >&2
    printf '    hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --local-dir %s\n' "$MODEL" >&2
    exit 1
fi

if [[ "${FORCE:-0}" != "1" ]]; then
    used_mib="$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    if [[ -n "$used_mib" && "$used_mib" -gt 4096 ]]; then
        printf 'GPU already has %s MiB in use; refusing to start (FORCE=1 to override).\n' "$used_mib" >&2
        nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv >&2 || true
        exit 1
    fi
fi

args=(
    --host "${HOST:-0.0.0.0}"
    --port "${PORT:-8000}"
    --served-model-name "$SERVED_MODEL_NAME"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.93}"
    --tensor-parallel-size "${TP:-1}"
    --distributed-executor-backend "${EXECUTOR:-mp}"
    --max-num-seqs "${MAX_NUM_SEQS:-4}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
    --linear-backend "${LINEAR_BACKEND:-marlin}"
    --enable-prefix-caching
    --no-enable-flashinfer-autotune
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_xml
    --enable-prompt-tokens-details
)

# Graphs are required for usable decode. ENFORCE_EAGER=1 is the crash-debug path.
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
    args+=(--enforce-eager)
else
    args+=(--compilation-config "${COMPILATION_CONFIG:-{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}}")
fi

# Vision on by default. LANGUAGE_MODEL_ONLY=1 skips the tower.
if [[ "${LANGUAGE_MODEL_ONLY:-0}" == "1" ]]; then
    args+=(--language-model-only)
fi

# Speculative decoding. The checkpoint ships an MTP head (mtp.* tensors).
# The target uses Marlin; the unquantized BF16 MTP draft selects its own backend.
SPEC="${SPEC:-mtp}"
case "$SPEC" in
    mtp)
        args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS:-3},\"moe_backend\":\"${DRAFT_MOE_BACKEND:-auto}\"}")
        ;;
    none|0)
        ;;
    *)
        echo "unknown SPEC=$SPEC (want mtp|none)" >&2
        exit 1
        ;;
esac

MOE_BACKEND="${MOE_BACKEND:-marlin}"
args+=(--moe-backend "$MOE_BACKEND")

# BLOCK_SIZE sets the KV block-size alignment. The hybrid attention block is rounded
# to a multiple of it. MTP k=5 needs a multiple of 48 (QSA ring capacity 12).
if [[ -n "${BLOCK_SIZE:-}" ]]; then
    args+=(--block-size "$BLOCK_SIZE")
fi

# Online-quantize the checkpoint's BF16 dense linears (the NVFP4 experts are
# untouched), e.g. DENSE_QUANT=fp8_per_block_static -> weight-only FP8 (Marlin).
# DENSE_QUANT_ACT (e.g. fp8_per_block_dynamic) makes it W8A8 for kernels that need it.
# DENSE_QUANT_SCHEME: an online scheme shorthand (e.g. mxfp4) instead of a weight key.
if [[ -n "${DENSE_QUANT_SCHEME:-}" && -z "${DENSE_QUANT:-}" ]]; then
    ign=""
    if [[ -n "${DENSE_QUANT_IGNORE:-}" ]]; then
        IFS=',' read -r -a ign_pats <<< "$DENSE_QUANT_IGNORE"
        ign=",\"ignore\":[$(printf '"%s",' "${ign_pats[@]}" | sed 's/,$//')]"
    fi
    args+=(--quantization-config "{\"linear\":\"${DENSE_QUANT_SCHEME}\"${ign}}")
fi
if [[ -n "${DENSE_QUANT:-}" ]]; then
    act="null"
    [[ -n "${DENSE_QUANT_ACT:-}" ]] && act="\"${DENSE_QUANT_ACT}\""
    # DENSE_QUANT_IGNORE: comma-separated fnmatch patterns kept in BF16 (small
    # layers where Marlin FP8 is slower than cuBLAS BF16, and the MoE router).
    ign=""
    if [[ -n "${DENSE_QUANT_IGNORE:-}" ]]; then
        IFS=',' read -r -a ign_pats <<< "$DENSE_QUANT_IGNORE"
        ign=",\"ignore\":[$(printf '"%s",' "${ign_pats[@]}" | sed 's/,$//')]"
    fi
    args+=(--quantization-config "{\"linear\":{\"weight\":\"${DENSE_QUANT}\",\"activation\":${act}}${ign}}")
fi

echo "VLLM_PLE_CPU_OFFLOAD=$VLLM_PLE_CPU_OFFLOAD MODEL=$MODEL MAX_MODEL_LEN=$MAX_MODEL_LEN SPEC=$SPEC ENFORCE_EAGER=${ENFORCE_EAGER:-0} LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-0} MOE_BACKEND=${MOE_BACKEND:-auto}"
exec vllm serve "$MODEL" "${args[@]}" "$@"
