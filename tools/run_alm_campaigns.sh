#!/usr/bin/env bash
# Run the five evaluation tasks for one or more ALM backends, in series.
#
#   tools/run_alm_campaigns.sh                  # all three, in series
#   tools/run_alm_campaigns.sh mimo-audio       # a single one
#   VALIDER=0 tools/run_alm_campaigns.sh        # skip validation (avoid)
#
# Each backend runs in the venv matching the transformers version its
# config.json declares, compilation caches are kept out of $HOME, and every
# campaign is preceded by a validation: loading is not working. Resuming is
# free - each task has its own `raw_<backend>_<task>` cache.
set -uo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

DATA=${DATA:-other/data/data_test}
VALIDER=${VALIDER:-1}
ENVS=${ENVS:-alm_env}

#: backend -> venv (the version its config.json declares) and cloned repository.
venv_for() {
  case "$1" in
    moss-audio)   echo "$ENVS/.venv_tf4" ;;    # transformers 4.57.1
    mimo-audio)   echo "$ENVS/.venv_tf449" ;;  # 4.49.0
    midasheng-lm) echo "$ENVS/.venv_tf452" ;;  # 4.52.4
    *)            echo "$ENVS/.venv" ;;        # 5.x, the original backends
  esac
}
repo_for() {
  case "$1" in
    moss-audio) echo "$ENVS/MOSS-Audio" ;;
    mimo-audio) echo "$ENVS/MiMo-Audio" ;;
    *)          echo "" ;;
  esac
}

CACHES=${CACHES:-$(cd .. && pwd)/.cache}
export HF_HOME="${HF_HOME:-$CACHES/huggingface}"
export TRITON_CACHE_DIR="$CACHES/triton" TRITON_HOME="$CACHES/triton_home"
export TORCHINDUCTOR_CACHE_DIR="$CACHES/inductor" CUDA_CACHE_PATH="$CACHES/nv"
export XDG_CACHE_HOME="$CACHES/xdg"
mkdir -p "$TRITON_CACHE_DIR" "$TRITON_HOME" "$TORCHINDUCTOR_CACHE_DIR" \
         "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"

BACKENDS=("$@")
# Cheapest first: moss-audio is a "Thinking" model and costs ~32 h.
[ ${#BACKENDS[@]} -eq 0 ] && BACKENDS=(mimo-audio midasheng-lm moss-audio)

failures=()
for backend in "${BACKENDS[@]}"; do
  venv=$(venv_for "$backend"); repo=$(repo_for "$backend")
  python="$venv/bin/python"
  echo "############ $backend  ($(date '+%H:%M'))  venv=$venv"
  if [ ! -x "$python" ]; then
    echo "  missing venv: $python" >&2; failures+=("$backend (venv)"); continue
  fi
  "$python" -c "import transformers; print('  transformers', transformers.__version__)"

  if [ "$VALIDER" = "1" ]; then
    echo "  --- validation (3 open questions)"
    args=(--backend "$backend" --data-dir "$DATA")
    [ -n "$repo" ] && args+=(--alm-repo "$repo")
    if ! "$python" tools/valider_backend.py "${args[@]}"; then
      echo "  VALIDATION FAILED: campaign NOT started for $backend" >&2
      failures+=("$backend (validation)"); continue
    fi
  fi

  ALM_REPO="$repo" PYTHON="$python" bash tools/run_evals.sh "$backend" "$DATA" \
    && echo "############ $backend DONE ($(date '+%H:%M'))" \
    || { echo "############ $backend FAILED ($(date '+%H:%M'))" >&2
         failures+=("$backend (eval)"); }
done

echo
if [ ${#failures[@]} -eq 0 ]; then
  echo "############ ALL PASSED ($(date '+%H:%M'))"
  exit 0
fi
echo "############ FAILURES: ${failures[*]}" >&2
exit 1
