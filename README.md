<div align="center">
  <img src="assets/logo.svg" alt="" width="120" height="120">
  <h1>CARES</h1>
  <p><strong>Controlled Audio Reactions in Environmental Scenes</strong></p>
</div>

CARES generates two-speaker audio scenes in which environmental sounds are placed
inside a conversation. Each sound carries a **reaction type** that the dialogue
must realise without ever naming the sound. Everything structural (scene,
roles, sound identifiers, reaction types, split) is drawn from a fixed seed
before the first model call, so the dataset is reproducible.

The repository also contains an evaluation harness for audio-language models
(`cares eval`) and a text-only control built from Whisper plus an LLM.

## Reaction types

| Type | What the dialogue does |
|---|---|
| `pivot` | The sound takes over, it becomes the subject for several turns. |
| `verbal` | One speaker acknowledges it in passing, then the conversation resumes. |
| `behavioral` | Nobody mentions it, but it visibly disrupts the speaking. |
| `ambient` | Nobody reacts. |

## Install

Python ≥ 3.10. The core needs only `numpy` and `tqdm`, everything else is an
extra, grouped by stage.

```bash
pip install -e '.[all]'            
```

Without installing, `python -m cares <command>` works from the repository root. A
stage whose dependencies are missing does not block the others, the error is
raised only when that command is invoked.

API keys are read from the environment, for example :

```bash
export ANTHROPIC_API_KEY=sk-ant-...     
export ELEVENLABS_API_KEY=sk_...         
```

Stages served by vLLM take `--base-url` and `--model`.

## Pipeline

Every stage writes under one data root (`./data` by default, or `--data-dir`, or
`$CARES_DATA_DIR`) and resumes where it stopped.

```bash
# 1. Templates
cares templates --base-url http://localhost:8000/v1 --model Qwen --concurrency 32

# 2. Scenarios: deterministic pre-allocation 
cares scenarios --base-url http://localhost:8000/v1 --model Qwen --concurrency 64

# 3. Dialogues. --batch halves the cost
cares dialogues --model claude-opus-4-8 --concurrency 8

# 4. Rejection sampling
cares filter --base-url http://localhost:8000/v1 --model maverick --tau 0.82
cares filter --finalize            # writes filter_output/dialogues_filtered.json

# 5. Voices then forced alignment
cares tts --input data/filter_output/dialogues_filtered.json --voices "id1,id2,id3"
cares align

# 6. Scene mixing
cares mix --events-root <sounds> --backgrounds-root <ambiences> \
        --dialogues data/filter_output/dialogues_filtered.json --workers 8
```

`cares --help` lists every command, `cares <command> --help` its options.

## Evaluation

`cares eval` runs five tasks on the mixed scenes: `scene`, `sounds-mcq`,
`reactions`, `grounding` and `summary`. 

```bash
cares eval --data-dir <data> --backend <name> \
         --task all --dump-items      # write the questions, call no model
cares eval --data-dir <data> --backend <name> --task reactions
```

`tools/run_evals.sh` freezes the full sequence.

Backends: `audio-flamingo`, `qwen3-omni`, `kimi-audio`, `moss-audio`,
`mimo-audio`, `midasheng-lm`, `whisper+llm` (the text-only control), and `fake`
(deterministic, no GPU, the default).

## What each stage writes

Everything lives under `--data-dir`. are Keep files marked as caches (rerun free).


| Stage | Directory | Main output |
|---|---|---|
| `templates` | `template_generation_output/` | `templates.json` |
| `scenarios` | `scenario_generation_output/` | `scenarios.json`, `splits.json` |
| `dialogues` | `dialogue_generation_output/` | `dialogues.json` |
| `filter` | `filter_output/` | `dialogues_filtered.json` (the dataset) |
| `tts` | `out_voices/` | `<sid>.mp3`, `<sid>.meta.json` |
| `align` | `alignments/` | `<sid>.json` |
| `mix` | `audio_scenes/` | `<sid>.wav`, `<sid>.manifest.json` |
| `eval` | `eval_output/` | `scores_<backend>_<task>.json` |
