#!/usr/bin/env bash
# Run every paper evaluation for one backend.
#
#   tools/run_evals.sh audio-flamingo
#   tools/run_evals.sh whisper+llm            # the text-only CONTROL, not optional
#
# --banks must stay on v3, the bank the dataset was generated from. Only the
# `summary` task calls an API (the text judge), and it runs last because it is
# the only task whose score depends on a second model.
set -euo pipefail

BACKEND=${1:?usage: run_evals.sh <backend> [data-dir]}
DATA=${2:-other/data/data_test}
BANKS=${BANKS:-cares/resources/scenario_banks_v3.json}

# One venv per transformers version: moss-audio and mimo-audio pin 4.x, the
# main venv carries the 5.x that qwen3-omni needs.
PYTHON=${PYTHON:-python}

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

[ -d "$DATA/audio_scenes" ] || { echo "No mixed scenes in $DATA" >&2; exit 1; }
if [ ! -d "$DATA/audio_cf0" ]; then
  echo "WARNING: no $DATA/audio_cf0 - the grounding task will be empty." >&2
fi

common=(--data-dir "$DATA" --backend "$BACKEND" --banks "$BANKS")
# moss-audio and mimo-audio load their classes from a cloned repository.
case "$BACKEND" in
  moss-audio) common+=(--alm-repo "${ALM_REPO:-alm_env/MOSS-Audio}") ;;
  mimo-audio) common+=(--alm-repo "${ALM_REPO:-alm_env/MiMo-Audio}") ;;
esac

echo "=== items (no model call)"
"$PYTHON" -m cares eval "${common[@]}" --task all --dump-items

# A failing task must not take the others down: each one has its own cache.
failures=()
for task in scene sounds-mcq reactions grounding; do
  echo "=== $task"
  "$PYTHON" -m cares eval "${common[@]}" --task "$task" || failures+=("$task")
done

echo "=== summary (ALM then text judge)"
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "  ANTHROPIC_API_KEY missing: summaries produced, NOT judged." >&2
  echo "  Re-running with the key and --scores-only will not pay for the audio again." >&2
  "$PYTHON" -m cares eval "${common[@]}" --task summary --no-judge
else
  "$PYTHON" -m cares eval "${common[@]}" --task summary
fi

echo
echo "Results: $DATA/eval_output/summary_$BACKEND.json"
if [ ${#failures[@]} -gt 0 ]; then
  echo "FAILED on: ${failures[*]}" >&2
  exit 1
fi
