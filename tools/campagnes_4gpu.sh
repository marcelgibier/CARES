#!/usr/bin/env bash
# Schedule the remaining `cares eval` campaigns over the 4 B200 of a single node.
#
#   setsid nohup bash tools/campagnes_4gpu.sh > logs/4gpu.log 2>&1 < /dev/null &
#
# `setsid` and not `nohup` alone: a `nohup` started from an ssh session dies
# with it. Phase A runs the short campaigns on the four cards, phase B runs
# moss-audio (the long pole) on the freed cards, so that the other results land
# by the third hour instead of the eleventh. The reference items_*.json is
# written once, up front: two concurrent run_shards.sh would write the same file
# through a deterministic temporary name.
set -uo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

DATA=${DATA:-other/data/data_test}
BANKS=${BANKS:-cares/resources/scenario_banks_v3.json}
ENVS=${ENVS:-alm_env}
CF="$DATA/audio_cf0"
LOGS=${LOGS:-logs}
mkdir -p "$LOGS"

export CAMPAIGN_START=$(date '+%s')
elapsed() { printf '%dh%02d' $(( ($(date '+%s') - CAMPAIGN_START) / 3600 )) \
                             $(( ($(date '+%s') - CAMPAIGN_START) % 3600 / 60 )); }

echo "############ 4 GPU campaigns - start $(date '+%F %H:%M')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
  | sed 's/^/  /' || echo "  (nvidia-smi unavailable)"

# ---------------------------------------------------------------------------
echo "=== reference: items_*.json, once for the four campaigns"
"$ENVS/.venv/bin/python" -m cares eval --data-dir "$DATA" --backend fake \
  --banks "$BANKS" --counterfactual-dirs "$CF" --task all --dump-items \
  || { echo "Reference not written: no completeness check would be possible." >&2
       echo "Stopping rather than spending 11 h of GPU blind." >&2
       exit 1; }

failures=()

# ---------------------------------------------------------------------------
echo "############ PHASE A ($(elapsed))"

# Card 0, in series: the text control first, it is short and the paper cannot
# do without that number.
(
  REFERENCE=0 VALIDATE=0 BACKEND=whisper+llm GPUS=0 TASKS=grounding \
    bash tools/run_shards.sh || exit 1
  REFERENCE=0 VALIDATE=0 BACKEND=mimo-audio GPUS=0 TASKS=summary \
    bash tools/run_shards.sh || exit 2
) > "$LOGS/A_card0.log" 2>&1 &
pid_a=$!

# Cards 1 to 3: MiDashengLM has never run a campaign, hence VALIDATE=1.
(
  REFERENCE=0 VALIDATE=1 BACKEND=midasheng-lm GPUS="1 2 3" \
    bash tools/run_shards.sh
) > "$LOGS/A_midasheng.log" 2>&1 &
pid_b=$!

wait "$pid_a" || failures+=("phase A card 0 (whisper+llm / mimo-audio)")
echo "  card 0 finished ($(elapsed))"
wait "$pid_b" || failures+=("phase A midasheng-lm")
echo "  midasheng-lm finished ($(elapsed))"

# ---------------------------------------------------------------------------
# MOSS starts even if phase A stumbled: the campaigns are independent, and this
# is the longest one. What failed is reported at the end, not buried.
echo "############ PHASE B: moss-audio on 4 cards ($(elapsed))"
REFERENCE=0 VALIDATE=1 BACKEND=moss-audio GPUS="0 1 2 3" \
  bash tools/run_shards.sh > "$LOGS/B_moss.log" 2>&1 \
  || failures+=("phase B moss-audio")

# ---------------------------------------------------------------------------
echo "############ END ($(elapsed), $(date '+%F %H:%M'))"
echo "=== final state of the caches"
"$ENVS/.venv/bin/python" tools/verifier_fusion.py "$DATA/eval_output" \
  whisper+llm --tasks grounding 2>&1 | sed 's/^/  /'
for b in mimo-audio midasheng-lm moss-audio; do
  "$ENVS/.venv/bin/python" tools/verifier_fusion.py "$DATA/eval_output" "$b" \
    2>&1 | sed "s/^/  [$b] /"
done

if [ ${#failures[@]} -gt 0 ]; then
  echo "############ FAILURES: ${failures[*]}" >&2
  echo "  The logs are in $LOGS/. Re-running the same command resumes where it" >&2
  echo "  stopped: every shard keeps its cache." >&2
  exit 1
fi
echo "############ ALL PASSED"
