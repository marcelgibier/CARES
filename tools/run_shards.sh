#!/usr/bin/env bash
# Run a `cares eval` campaign as disjoint shards, one per GPU.
#
#   BACKEND=moss-audio tools/run_shards.sh                   # 4 GPUs, 5 tasks
#   BACKEND=midasheng-lm GPUS="1 2 3" tools/run_shards.sh     # 3 GPUs
#   BACKEND=mimo-audio GPUS=0 TASKS=summary tools/run_shards.sh
#   DRY=1 ...                                                 # print, run nothing
#   STAGE=score ...                                           # shards done: merge+score
#   STAGE=merge ...                                           # merge only
#
# Each shard writes its own cache (one process per data-dir and per artefact),
# gets its own Triton/Inductor/remote-code caches, and is merged at the end in a
# single process. Splitting is by scene, never by item, so that a grounding pair
# and the three summary budgets of a scene stay together. No merge if a shard
# failed, and no scoring until tools/verifier_fusion.py is green: a truncated
# cache scores silently with exit code 0.
set -uo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

BACKEND=${BACKEND:-moss-audio}
DATA=${DATA:-other/data/data_test}
GPUS=${GPUS:-0 1 2 3}
BANKS=${BANKS:-cares/resources/scenario_banks_v3.json}
ENVS=${ENVS:-alm_env}
STAGE=${STAGE:-all}           # all | shards | score | merge
TASKS=${TASKS:-scene sounds-mcq reactions grounding summary}
DRY=${DRY:-0}
VALIDATE=${VALIDATE:-1}
REFERENCE=${REFERENCE:-1}

read -r -a CARDS <<< "$GPUS"
read -r -a TASK_LIST <<< "$TASKS"
NSHARDS=${#CARDS[@]}
[ "$NSHARDS" -gt 0 ] || { echo "GPUS is empty" >&2; exit 2; }

#: Each repository pins its own transformers version; running it under another
#: one produces failures that imitate model limits. Scoring loads no model.
case "$BACKEND" in
  moss-audio)   VENV=${VENV:-$ENVS/.venv_tf4} ;;    # transformers 4.57.1
  mimo-audio)   VENV=${VENV:-$ENVS/.venv_tf449} ;;  # 4.49.0
  midasheng-lm) VENV=${VENV:-$ENVS/.venv_tf452} ;;  # 4.52.4
  *)            VENV=${VENV:-$ENVS/.venv} ;;
esac
PYTHON=${PYTHON:-$VENV/bin/python}
PYTHON_SCORE=${PYTHON_SCORE:-$ENVS/.venv/bin/python}

case "$BACKEND" in
  moss-audio) ALM_REPO=${ALM_REPO:-$ENVS/MOSS-Audio} ;;
  mimo-audio) ALM_REPO=${ALM_REPO:-$ENVS/MiMo-Audio} ;;
  *)          ALM_REPO=${ALM_REPO:-} ;;
esac

# `audio_path_for` prefers the manifest field, which is relative to the repo
# root, hence the `cd` above.
case "$DATA" in /*) SCENES="$DATA/audio_scenes" ;;
                 *) SCENES="$root/$DATA/audio_scenes" ;; esac
CF="$DATA/audio_cf0"
OUT="$DATA/eval_output"
SHARDS="$DATA/eval_shards/$BACKEND"    # outside eval_output: scoring globs
LOGS="$SHARDS/logs"                    # `scores_*_*.json` in ONE directory

[ -d "$SCENES" ] || { echo "No mixed scenes in $SCENES" >&2; exit 1; }
[ -d "$CF" ] || echo "WARNING: no $CF - grounding will fail (exit 2)." >&2

CACHES=${CACHES:-$(cd .. && pwd)/.cache}
export HF_HOME="${HF_HOME:-$CACHES/huggingface}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# Setting libstdc++ here avoids the self-reinvocation of cares/alm/base.py.
if command -v gcc >/dev/null 2>&1; then
  _libdir=$(dirname "$(gcc -print-file-name=libstdc++.so.6 2>/dev/null)" 2>/dev/null)
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$_libdir:"*) ;;
    *) [ -n "$_libdir" ] && [ -d "$_libdir" ] && \
         export LD_LIBRARY_PATH="$_libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
fi

run() {   # echo under DRY=1, execute otherwise
  if [ "$DRY" = "1" ]; then printf '  [dry] %s\n' "$*"; return 0; fi
  "$@"
}

# --- Shards ----------------------------------------------------------------
# Manifests are dealt round-robin over the sorted list, which balances better
# than a hash (3.3 % max/min spread against 13.6 %).
make_shards() {
  mapfile -t manifests < <(find "$SCENES" -maxdepth 1 -name '*.manifest.json' | sort)
  [ ${#manifests[@]} -gt 0 ] || { echo "No manifest in $SCENES" >&2; exit 1; }
  for i in "${!CARDS[@]}"; do
    rm -rf "$SHARDS/$i/audio_scenes"
    mkdir -p "$SHARDS/$i/audio_scenes" "$SHARDS/$i/out" "$LOGS"
  done
  # `ln -s -t DIR TARGET...` and not `ln -s TARGET DIR/`: on this cluster
  # (coreutils 8.32, Lustre) the second form fails on a relative directory.
  for i in "${!CARDS[@]}"; do
    batch=()
    for idx in "${!manifests[@]}"; do
      [ $((idx % NSHARDS)) -eq "$i" ] && batch+=("${manifests[$idx]}")
    done
    [ ${#batch[@]} -gt 0 ] && ln -s -t "$SHARDS/$i/audio_scenes" "${batch[@]}"
  done

  echo "Shards: ${#manifests[@]} scenes over $NSHARDS GPU(s) [$GPUS]"
  total=0
  for i in "${!CARDS[@]}"; do
    n=$(find "$SHARDS/$i/audio_scenes" -name '*.manifest.json' | wc -l)
    printf '  shard %d (GPU %s): %s manifests\n' "$i" "${CARDS[$i]}" "$n"
    total=$((total + n))
    # An empty shard would evaluate zero scene: 0 item, exit 0, and the merge
    # would see nothing missing.
    [ "$n" -gt 0 ] || { echo "Shard $i is EMPTY: not starting the campaign." >&2
                        exit 1; }
  done
  [ "$total" -eq "${#manifests[@]}" ] || {
    echo "Incomplete links: $total out of ${#manifests[@]} manifests." >&2
    exit 1; }
}

# Free resume: what a previous run already paid for is copied into the shard.
# The central cache seeds the shard but must NOT win at merge time, otherwise
# re-running a shard with --retry-errors would have no effect.
seed_from_central_cache() {
  # UNION, never a conditional `cp`: a shard that already holds a cache would
  # keep it and ignore the central one, re-paying items another shard paid for
  # after a change in the number of GPUs. A real answer beats an error record,
  # same rule as `merge_caches`. `transcripts_whisper.json` is treated like an
  # answer cache: without it every whisper+llm shard re-transcribes everything.
  "$PYTHON_SCORE" - "$OUT" "$SHARDS" "$BACKEND" "$NSHARDS" <<'PY2'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from cares.jsonio import load_json, save_json

out, shards, backend, n = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
sources = sorted(out.glob(f"raw_{backend}_*.json"))
t = out / "transcripts_whisper.json"
if t.exists():
    sources.append(t)
for i in range(n):
    dst_dir = shards / str(i) / "out"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sources:
        central = load_json(src, default={}) or {}
        dst = dst_dir / src.name
        local = load_json(dst, default={}) or {}
        added = 0
        for k, v in central.items():
            previous = local.get(k)
            if previous is None or (isinstance(previous, dict) and previous.get("error")
                                    and isinstance(v, dict) and not v.get("error")):
                local[k] = v
                added += 1
        if added:
            save_json(dst, local)
            print(f"  shard {i}: {src.name} +{added} from the central cache "
                  f"({len(local)} in total)")
PY2
}

# Transcripts paid for by the shards go back to the central cache, otherwise the
# next run pays for them again. Plain union: the key is the wav path.
push_transcripts_up() {
  "$PYTHON_SCORE" - "$SHARDS" "$OUT" <<'PY2'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from cares.jsonio import load_json, save_json

shards, out = Path(sys.argv[1]), Path(sys.argv[2])
dst = out / "transcripts_whisper.json"
merged = dict(load_json(dst, default={}) or {})
before = len(merged)
for d in sorted(p for p in shards.glob("*/out") if p.is_dir()):
    merged.update(load_json(d / "transcripts_whisper.json", default={}) or {})
if len(merged) > before:
    save_json(dst, merged)
    print(f"  transcripts_whisper.json: {len(merged)} ({len(merged) - before} new)")
PY2
}

# The environment of a shard, in its own function because the VALIDATION must
# run in the same one: a validation that does not run in the environment of the
# campaign does not validate the campaign.
shard_env() {   # $1 = shard index
  local i=$1
  export CUDA_VISIBLE_DEVICES="${CARDS[$i]}"
  export TRITON_CACHE_DIR="$CACHES/triton/$BACKEND-$i"
  export TRITON_HOME="$CACHES/triton_home/$BACKEND-$i"
  export TORCHINDUCTOR_CACHE_DIR="$CACHES/inductor/$BACKEND-$i"
  export CUDA_CACHE_PATH="$CACHES/nv/$BACKEND-$i"
  export XDG_CACHE_HOME="$CACHES/xdg/$BACKEND-$i"
  export HF_MODULES_CACHE="$CACHES/hf_modules/$BACKEND-$i"
  mkdir -p "$TRITON_CACHE_DIR" "$TRITON_HOME" "$TORCHINDUCTOR_CACHE_DIR" \
           "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$HF_MODULES_CACHE"

  # A per-shard remote-code cache removes the race in `get_cached_module_file`,
  # but an empty one is fatal under HF_HUB_OFFLINE=1, so seed it from the
  # shared cache.
  local module_source="${HF_HOME:-$HOME/.cache/huggingface}/modules"
  if [ -d "$module_source" ] && [ -z "$(ls -A "$HF_MODULES_CACHE" 2>/dev/null)" ]; then
    cp -r "$module_source/." "$HF_MODULES_CACHE/" 2>/dev/null \
      || echo "  WARNING: remote-code cache not seeded for shard $i" >&2
  fi
}

run_shard() {   # $1 = shard index
  local i=$1 rc=0
  local scenes="$SHARDS/$i/audio_scenes" out="$SHARDS/$i/out"
  shard_env "$i"

  # The same options as tools/run_evals.sh, plus the three imposed by the
  # split. Everything else is left at its default; `record["run"]` is what
  # records them, and verifier_fusion.py refuses a cache that mixes them.
  local common=(--data-dir "$DATA" --backend "$BACKEND" --banks "$BANKS"
                --scenes-dir "$scenes" --output-dir "$out")
  [ -n "$ALM_REPO" ] && common+=(--alm-repo "$ALM_REPO")

  for task in "${TASK_LIST[@]}"; do
    local args=("${common[@]}" --task "$task")
    [ "$task" = "grounding" ] && args+=(--counterfactual-dirs "$CF")
    # The summary judge runs once, after the merge: it is text, not GPU, and
    # four concurrent judges would write the same judged_*.json.
    [ "$task" = "summary" ] && args+=(--no-judge)
    echo "--- shard $i: $task ($(date '+%H:%M'))"
    run "$PYTHON" -m cares eval "${args[@]}" || { rc=1; break; }
  done

  # These files carry the canonical names but only cover a quarter of the
  # dataset: leaving them would be read later as a complete run.
  rm -f "$out"/scores_*.json "$out"/summary_*.json
  return $rc
}

# Merge: UNION of the shards, with one business rule - an answer from a shard
# beats an error record from the central cache.
merge_caches() {
  "$PYTHON_SCORE" - "$SHARDS" "$OUT" "$BACKEND" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from cares.jsonio import load_json, save_json

shards, out, backend = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
out.mkdir(parents=True, exist_ok=True)
# `sorted(glob)` and not `range(n)`: a previous 4-shard run leaves PAID answers
# that re-running with 2 shards would drop without a word.
dirs = sorted(p for p in shards.glob("*/out") if p.is_dir())
if not dirs:
    print("No shard to merge."); raise SystemExit(1)
# `judged_` as well as `raw_`: a shard re-run without --no-judge carries
# verdicts that are paid for too.
names = sorted({f.name for d in dirs
                for pattern in (f"raw_{backend}_*.json", f"judged_{backend}_*.json")
                for f in d.glob(pattern)})
if not names:
    print(f"No {backend} cache in the shards."); raise SystemExit(1)
for name in names:
    dst = out / name
    merged = dict(load_json(dst, default={}) or {})
    central_keys, before, duplicates, repaired = set(merged), len(merged), 0, 0
    for d in dirs:
        for k, v in (load_json(d / name, default={}) or {}).items():
            previous = merged.get(k)
            if previous is None:
                merged[k] = v
            elif (isinstance(previous, dict) and previous.get("error")
                  and isinstance(v, dict) and not v.get("error")):
                merged[k] = v          # a real answer beats an error
                repaired += 1
            else:
                duplicates += k not in central_keys
    errors = sum(1 for v in merged.values() if isinstance(v, dict) and v.get("error"))
    save_json(dst, merged)
    print(f"  {name}: {len(merged)} items ({before} already central, "
          f"{repaired} errors repaired, {duplicates} duplicates between shards, "
          f"{errors} in error)")
PY
}

# Scoring: a single process, on the WHOLE item list (the data-dir, not a shard).
# --scores-only loads no model; `summary` judges the cached summaries without
# paying for a second of audio again.
score() {
  local rc=0
  for task in "${TASK_LIST[@]}"; do
    echo "--- scores: $task"
    local args=(--data-dir "$DATA" --backend "$BACKEND" --banks "$BANKS"
                --task "$task" --scores-only)
    [ "$task" = "grounding" ] && args+=(--counterfactual-dirs "$CF")
    run "$PYTHON_SCORE" -m cares eval "${args[@]}" || rc=1
  done
  return $rc
}

# ---------------------------------------------------------------------------
echo "############ $BACKEND on $NSHARDS GPU(s) [$GPUS] - $DATA ($(date '+%H:%M'))"
echo "  tasks   : $TASKS"
echo "  shards  : $SHARDS"
echo "  python  : $PYTHON   (scoring: $PYTHON_SCORE)"

# `merge` and `score` both start from shards that are already computed: neither
# must rebuild the symlink farm nor re-run the shards.
if [ "$STAGE" != "score" ] && [ "$STAGE" != "merge" ]; then
  # The reference item list, once and without splitting. `--backend fake` on
  # purpose: the backend changes no item, and it is the only way to write the
  # reference without loading the weights. REFERENCE=0 when TWO backends run at
  # the same time on different GPUs, since they would write the same
  # items_*.json through a deterministic temporary name.
  if [ "$REFERENCE" = "1" ]; then
  echo "=== reference: items_*.json (no model call)"
  run "$PYTHON_SCORE" -m cares eval --data-dir "$DATA" --backend fake \
    --banks "$BANKS" --counterfactual-dirs "$CF" --task all --dump-items \
    || { echo "Reference not written: no completeness check possible, stopping." >&2
         exit 1; }
  fi

  if [ "$VALIDATE" = "1" ]; then
    echo "=== validation (3 open questions, GPU ${CARDS[0]})"
    echo "    in the EXACT environment of a shard - otherwise it validates nothing"
    val=(--backend "$BACKEND" --data-dir "$DATA")
    [ -n "$ALM_REPO" ] && val+=(--alm-repo "$ALM_REPO")
    if ! ( shard_env 0; run "$PYTHON" tools/valider_backend.py "${val[@]}" ); then
      echo "VALIDATION FAILED: campaign NOT started." >&2; exit 2
    fi
  fi

  make_shards
  seed_from_central_cache

  pids=(); for i in "${!CARDS[@]}"; do
    run_shard "$i" > "$LOGS/shard$i.log" 2>&1 &
    pids[i]=$!
    echo "  shard $i started (pid ${pids[i]}, GPU ${CARDS[$i]}) -> $LOGS/shard$i.log"
  done
  failures=()
  for i in "${!CARDS[@]}"; do
    wait "${pids[i]}" || failures+=("shard $i")
  done
  echo "############ shards finished ($(date '+%H:%M'))"
  # Merging despite a dead shard would produce a full set of scores over a
  # fraction of the dataset, exit 0, without a warning.
  if [ ${#failures[@]} -gt 0 ]; then
    echo "  FAILED: ${failures[*]} (see $LOGS)." >&2
    echo "  NO MERGE, NO SCORES. Re-running the same command resumes where it" >&2
    echo "  stopped: every shard has its own cache." >&2
    exit 1
  fi
fi

[ "$STAGE" = "shards" ] && exit 0

echo "=== merging the caches -> $OUT"
merge_caches || exit 1
push_transcripts_up

# STAGE=merge: merge WITHOUT checking or scoring. This is the step to run before
# CHANGING THE NUMBER OF GPUs of a running campaign: the split is a round-robin
# on the scene rank, so a new shard would not be seeded by the answers another
# shard already paid for. The completeness check makes no sense here, since the
# campaign is unfinished by construction.
if [ "$STAGE" = "merge" ]; then
  echo "############ merge only done. Re-run with the new GPUS."
  exit 0
fi

echo "=== completeness check (blocking)"
run "$PYTHON_SCORE" tools/verifier_fusion.py "$OUT" "$BACKEND" \
  --tasks "${TASK_LIST[@]}" || exit 1

if [ -z "${ANTHROPIC_API_KEY:-}" ] && [[ " $TASKS " == *" summary "* ]]; then
  echo "ANTHROPIC_API_KEY missing: the summary task will stay unjudged," >&2
  echo "  hence unscored (a mention rate without a judge is 0 by construction)." >&2
  echo "  Re-running with STAGE=score and the key will not pay for the audio again." >&2
fi

echo "=== scores (single process, whole dataset)"
score || exit 1
echo "############ $BACKEND DONE ($(date '+%H:%M'))"
