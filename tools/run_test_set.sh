#!/usr/bin/env bash
# Finish the test split: forced alignment, audible mix, counterfactual mix.
#
# The steps are sequential and each one reads the previous. Three options are
# load-bearing: --dialogues (mix reads the unfiltered dataset by default),
# --alignments-dir (the TTS sidecar drifts up to 6.4 s) and --loudness-from
# (the muted member must reuse the master gain of the audible one).
set -euo pipefail

DATA=${DATA:-other/data/data_test}
SOUNDS=${SOUNDS:-../cares_sounds}
DIALOGUES="$DATA/filter_output/dialogues_filtered.json"
WORKERS=${WORKERS:-8}

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

[ -f "$DIALOGUES" ] || { echo "Filtered dataset not found: $DIALOGUES" >&2; exit 1; }
[ -d "$SOUNDS/sound_events" ] || { echo "Sound takes not found: $SOUNDS" >&2; exit 1; }

# Disk space, before the first computation: ~11 MB per scene, doubled by the
# counterfactual mix.
scenes=$(python - "$DIALOGUES" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(len(d if isinstance(d, list) else d))
PY
)
needed=$(( (scenes * 11 * 2) / 1000 + 1 ))          # GB, margin included
free=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
echo "Disk: ${free} GB free, ~${needed} GB needed for ${scenes} scenes"
if [ "$free" -lt "$needed" ]; then
  echo "NOT ENOUGH DISK SPACE: $((needed - free)) GB missing. Nothing was started." >&2
  echo "Free some space, or mix elsewhere with --output-dir." >&2
  exit 1
fi

echo "=== 1/3  forced alignment (MMS_FA, GPU)"
python -m cares align \
  --data-dir "$DATA" \
  --dialogues "$DIALOGUES" \
  --voices-dir "$DATA/out_voices" \
  --output-dir "$DATA/alignments"

echo "=== 2/3  audible mix"
python -m cares mix \
  --data-dir "$DATA" \
  --dialogues "$DIALOGUES" \
  --voices-dir "$DATA/out_voices" \
  --alignments-dir "$DATA/alignments" \
  --events-root "$SOUNDS/sound_events" \
  --backgrounds-root "$SOUNDS/scenes" \
  --output-dir "$DATA/audio_scenes" \
  --workers "$WORKERS"

echo "=== 3/3  counterfactual mix (rank 0 muted) - the grounding task"
python -m cares mix \
  --data-dir "$DATA" \
  --dialogues "$DIALOGUES" \
  --voices-dir "$DATA/out_voices" \
  --alignments-dir "$DATA/alignments" \
  --events-root "$SOUNDS/sound_events" \
  --backgrounds-root "$SOUNDS/scenes" \
  --output-dir "$DATA/audio_cf0" \
  --mute-event-rank 0 \
  --loudness-from "$DATA/audio_scenes" \
  --workers "$WORKERS"

echo "=== check: no take shared between splits"
python -m cares mix --data-dir "$DATA" --output-dir "$DATA/audio_scenes" --audit-splits

echo
echo "audible scenes : $(ls "$DATA"/audio_scenes/*.wav 2>/dev/null | wc -l)"
echo "muted scenes   : $(ls "$DATA"/audio_cf0/*.wav 2>/dev/null | wc -l)"
