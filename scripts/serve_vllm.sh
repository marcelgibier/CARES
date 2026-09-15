#!/usr/bin/env bash
# Serve an OpenAI-compatible vLLM endpoint from the cluster Apptainer image.
#
# The templates / scenarios / filter / subject-pairs stages talk to it through
# --base-url. One model at a time: the script blocks. Paths depend on the
# account, so override them through the environment rather than editing here.
#
#   ./scripts/serve_vllm.sh qwen
#   ./scripts/serve_vllm.sh deepseek
#   ./scripts/serve_vllm.sh mimo 0 10.10.1.105   # master node, 2 nodes
#   ./scripts/serve_vllm.sh mimo 1 10.10.1.105   # second node (headless)

set -euo pipefail

MODEL_KIND="${1:-qwen}"
NODE_RANK="${2:-0}"
MASTER_ADDR="${3:-}"

WORK="${CARES_WORK_DIR:?set CARES_WORK_DIR to your scratch directory}"
SIF_IMAGE="${CARES_SIF_IMAGE:-${WORK}/scripts/vllm_cu129_nightly_aarch64.sif}"
QWEN_PATH="${CARES_QWEN_PATH:-${WORK}/scripts/alm-grounding/Qwen}"
CACHE_DIR="${CARES_CACHE_DIR:-${WORK}/.cache}"
CUDA_HOME_HOST="${CARES_CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.0}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"

# FlashInfer compiles its kernels inside the read-only container: expose a
# writable copy of its cubins.
CUBINS_DIR="${CACHE_DIR}/flashinfer_cubins_override"
CUBINS_TARGET=/usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins
mkdir -p "${CUBINS_DIR}"
apptainer exec "${SIF_IMAGE}" bash -c \
    "cp -a ${CUBINS_TARGET}/. ${CUBINS_DIR}/ 2>/dev/null || true"

BIND_ARGS=(
    --bind "${CACHE_DIR}:${CACHE_DIR}:rw"
    --bind "${CUBINS_DIR}:${CUBINS_TARGET}:rw"
)

echo "[$(date)] === vLLM: ${MODEL_KIND} on ${VLLM_HOST}:${VLLM_PORT} ==="

case "${MODEL_KIND}" in
  qwen)
    apptainer exec --nv "${BIND_ARGS[@]}" "${SIF_IMAGE}" \
        vllm serve "${QWEN_PATH}" \
        --served-model-name Qwen \
        --trust-remote-code \
        --tensor-parallel-size 4 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --host "${VLLM_HOST}" --port "${VLLM_PORT}"
    ;;

  mimo)
    # Two nodes: rank 0 serves the API, rank 1 runs headless.
    if [[ -z "${MASTER_ADDR}" ]]; then
        echo "mimo: pass the master node address as the 3rd argument." >&2
        exit 2
    fi
    EXTRA=()
    if [[ "${NODE_RANK}" == "1" ]]; then
        EXTRA+=(--headless)
    else
        EXTRA+=(--host "${VLLM_HOST}" --port "${VLLM_PORT}")
    fi
    apptainer exec --nv "${BIND_ARGS[@]}" "${SIF_IMAGE}" \
        vllm serve XiaomiMiMo/MiMo-V2.5-Pro \
        --served-model-name mimo \
        --trust-remote-code \
        --generation-config vllm \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.80 \
        --nnodes 2 \
        --node-rank "${NODE_RANK}" \
        --master-addr "${MASTER_ADDR}" \
        --tool-call-parser mimo \
        --enable-auto-tool-choice \
        --reasoning-parser mimo \
        "${EXTRA[@]}"
    ;;

  deepseek)
    # DeepSeek-V4 JIT-compiles its MoE kernels: it needs the host nvcc.
    apptainer exec --nv \
        --bind "${CUDA_HOME_HOST}:${CUDA_HOME_HOST}" \
        --env "CUDA_HOME=${CUDA_HOME_HOST}" \
        --env "DG_JIT_NVCC_COMPILER=${CUDA_HOME_HOST}/bin/nvcc" \
        "${BIND_ARGS[@]}" "${SIF_IMAGE}" \
        vllm serve deepseek-ai/DeepSeek-V4-Flash \
        --served-model-name deepseekflash \
        --trust-remote-code \
        --kv-cache-dtype fp8 \
        --block-size 256 \
        --enable-expert-parallel \
        --tensor-parallel-size 4 \
        --attention_config.use_fp4_indexer_cache=True \
        --moe-backend deep_gemm_mega_moe \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --host "${VLLM_HOST}" --port "${VLLM_PORT}"
    ;;

  *)
    echo "Unknown model: ${MODEL_KIND} (expected: qwen | mimo | deepseek)" >&2
    exit 2
    ;;
esac
