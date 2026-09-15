#!/usr/bin/env bash
# ALM evaluation environment (`cares eval`) for aarch64 + NVIDIA B200.
#
# Blackwell (sm_100) needs PyTorch CUDA >= 12.8 wheels; since torch 2.7 the
# official index publishes aarch64 CUDA wheels. Everything follows: one uv venv,
# the two "in-house" model repositories cloned next to it, and the weights
# pre-downloaded, since compute nodes have no Internet access.
#
# Usage:
#   bash tools/install_alm_env.sh                   # everything, weights included
#   SKIP_DOWNLOAD=1 bash tools/install_alm_env.sh   # without the weights (~200 GB)
#
# Run from a machine with Internet access (login node), NOT from a compute node.
set -euo pipefail

# --- Settings --------------------------------------------------------------
#: Root of everything this script installs. On a cluster: $WORK or $SCRATCH,
#: never $HOME (the weights alone are ~200 GB).
ALM_ROOT="${ALM_ROOT:-$PWD/alm_env}"
#: Hugging Face cache, on a large volume visible from the compute nodes.
export HF_HOME="${HF_HOME:-$ALM_ROOT/hf_cache}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
#: Pick the index <= the driver CUDA version: the aarch64 wheels of a cuXXX
#: directory are not guaranteed to be a coherent torch/torchaudio pair.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
VENV="$ALM_ROOT/.venv"

say()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*"; }

# --- 0. Preflight ----------------------------------------------------------
say "Preflight checks"
arch="$(uname -m)"
[ "$arch" = "aarch64" ] || warn "architecture $arch (expected aarch64): the chosen torch wheels will not match"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || warn "nvidia-smi missing here (normal on a login node without GPU): the final CUDA check will be partial"
command -v ffmpeg >/dev/null 2>&1 \
  || warn "ffmpeg missing: REQUIRED by the whisper+llm ASR pipeline (module load ffmpeg, or conda/apt)"
command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 1; }
mkdir -p "$ALM_ROOT"

# --- 1. uv and virtual environment -----------------------------------------
say "uv and virtual environment ($VENV, Python $PYTHON_VERSION)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python "$PYTHON_VERSION" "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --- 2. torch (Blackwell) --------------------------------------------------
say "PyTorch CUDA (index $TORCH_INDEX)"
# The THREE packages in the same resolution: installed separately, nothing
# forces identical CUDA builds. torchvision is only here because
# qwen_omni_utils imports it unconditionally, even for audio only.
uv pip install --index-url "$TORCH_INDEX" "torch>=2.7" "torchaudio>=2.7" torchvision
python - <<'EOF'
import torch
print(f"torch {torch.__version__} | cuda build {torch.version.cuda}")
# Importing torchaudio IS the coherence test: it raises "compiled with
# different CUDA versions" when the pair is mismatched.
import torchaudio
print(f"torchaudio {torchaudio.__version__}: coherent torch/torchaudio pair")
archs = torch.cuda.get_arch_list()
print("compiled architectures:", archs)
assert any("sm_100" in a or "sm_10" in a for a in archs), (
    "No sm_100 in this torch wheel: a B200 will run degraded or not at all. "
    "Check TORCH_INDEX (cu128 or newer is needed).")
EOF

# --- 3. Common stack -------------------------------------------------------
say "transformers and common dependencies"
# Recent transformers: Qwen3-Omni (Qwen3OmniMoe*) and Audio Flamingo Next were
# only added lately. If the final check reports a missing class, install
# transformers from git.
uv pip install --upgrade \
  transformers accelerate \
  "huggingface_hub[cli]" \
  soundfile librosa einops sentencepiece \
  audioread \
  qwen-omni-utils
# audioread: imported by qwen_omni_utils without being one of its declared
# dependencies, and recent librosa no longer pulls it in.
# uv pip install "git+https://github.com/huggingface/transformers"   # missing classes

# flash-attention is OPTIONAL for Qwen3-Omni (SDPA works) but REQUIRED by
# MiMo-Audio, whose audio tokenizer imports it at module level with no fallback.
# No aarch64 wheel: it has to be compiled (see CLAUDE.md for the recipe that
# worked on aarch64 + B200 + torch 2.14/cu130). Do NOT stub
# `flash_attn_varlen_func`: it sits in the audio encoder.

# --- 4. In-house repositories ----------------------------------------------
say "MOSS-Audio and MiMo-Audio repositories (classes outside transformers -> --alm-repo)"
for repo in "OpenMOSS/MOSS-Audio" "XiaomiMiMo/MiMo-Audio"; do
  dest="$ALM_ROOT/$(basename "$repo")"
  [ -d "$dest" ] || git clone --depth 1 "https://github.com/$repo" "$dest"
done
# Their requirements sometimes pin torch/transformers, which would destroy the
# Blackwell stack just installed, so nothing is installed by default.
warn "MOSS/MiMo: requirements NOT installed by default (would overwrite torch cu128)."
warn "On the first run, install individually what the import asks for."

# --- 5. Weights ------------------------------------------------------------
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
  say "Pre-downloading the weights into $HF_HOME (~200 GB, long step)"
  for model in \
    nvidia/audio-flamingo-next-hf \
    Qwen/Qwen3-Omni-30B-A3B-Instruct \
    OpenMOSS-Team/MOSS-Audio-8B-Thinking \
    XiaomiMiMo/MiMo-Audio-7B-Instruct \
    XiaomiMiMo/MiMo-Audio-Tokenizer \
    openai/whisper-large-v3 \
    Qwen/Qwen3.6-35B-A3B
  do
    say "  $model"
    hf download "$model" >/dev/null
  done
else
  warn "SKIP_DOWNLOAD=1: weights not downloaded - nothing will run offline."
fi

# --- 6. Final check --------------------------------------------------------
say "Final check"
python - <<'EOF'
import importlib, torch, transformers
print(f"python OK | torch {torch.__version__} | transformers {transformers.__version__}")
print(f"cuda available here: {torch.cuda.is_available()} (false on a login node: normal)")
missing = [n for n in ("Qwen3OmniMoeForConditionalGeneration", "Qwen3OmniMoeProcessor")
           if not hasattr(transformers, n)]
if missing:
    print(f"!! classes missing from transformers: {missing}")
    print("   -> uv pip install 'git+https://github.com/huggingface/transformers'")
importlib.import_module("qwen_omni_utils")
print("qwen_omni_utils OK")
EOF

say "Done"
cat <<EOF

For a job (node WITHOUT Internet access):
  source $VENV/bin/activate
  export HF_HOME=$HF_HOME
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

  python -m cares eval --data-dir <data> --backend audio-flamingo
  python -m cares eval --data-dir <data> --backend qwen3-omni
  python -m cares eval --data-dir <data> --backend moss-audio --alm-repo $ALM_ROOT/MOSS-Audio
  python -m cares eval --data-dir <data> --backend mimo-audio --alm-repo $ALM_ROOT/MiMo-Audio
  python -m cares eval --data-dir <data> --backend whisper+llm

Weight sizes (bf16): Qwen3-Omni ~65 GB, Qwen3.6-35B ~70 GB, AF-Next ~15 GB,
MOSS 8B ~17 GB, MiMo 7B + tokenizer ~17 GB, Whisper ~3 GB. A B200 (180 GB HBM)
holds each one alone; do NOT load two at a time.
EOF
