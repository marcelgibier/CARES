from __future__ import annotations

import argparse
from typing import Any

from ..config import THEMES, Paths
from ..jsonio import load_json, save_json
from ..jsonparse import find_list
from ..llm import OpenAIChat
from ..log import get_logger
from ..runner import Checkpoint, run_async, run_pool

log = get_logger(__name__)

N_TEMPLATES_PER_THEME = 10

GEN_TEMPERATURE = 0.9
GEN_TOP_P = 0.95
GEN_MAX_NEW_TOKENS = 3000
GEN_MAX_RETRIES = 5

SAVE_EVERY = 10
DEFAULT_CONCURRENCY = 32
HTTP_TIMEOUT = 600.0

LIST_ALIASES = ("templates", "sub_topics", "subtopics", "sub_topic_templates",
                "items", "list", "topics", "results", "data")


SYSTEM_PROMPT = """You are a creative assistant designing diverse audio scene \
templates for a dataset of everyday two-speaker conversations. Your goal is to \
produce sub-topics that are concrete, mutually distinct, and that together cover \
the natural breadth of what real people typically discuss around a given theme.

You answer with strict JSON only."""

USER_PROMPT = """Generate exactly {n} templates for the following everyday-life \
conversation theme.

THEME CATEGORY : {category}
THEME          : {theme}

Each template will later guide the generation of a realistic two-speaker audio \
scene (Alice and Bob talking). The {n} templates together should span the FULL \
BREADTH of what people genuinely talk about around this theme — different angles, \
life situations, contexts, concerns, age groups, social settings. They must feel \
meaningfully different from each other.

For each template, produce :
  - title       : a short heading (3 to 8 words)
  - description : a single line listing 2 to 4 concrete facets or examples \
(roughly 8 to 18 words)

REQUIREMENTS :
  1. Produce EXACTLY {n} items, no more, no less.
  2. No duplicates and no near-duplicates. Each title must be distinctive.
  3. Stay grounded in everyday life. No abstract philosophy, no fringe scenarios.
  4. Cover varied angles : practical, emotional, social, financial, generational, etc.
  5. Description must NOT just rephrase the title.
  6. Natural, plain English. No emojis, no markdown bullets, no numbering inside the JSON values.

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT — STRICT
═══════════════════════════════════════════════════════════════════
Return ONLY a JSON object with this EXACT structure. The top-level key MUST be \
"templates" (NOT "sub_topics", NOT "items", NOT "list"). No preamble, no \
postamble, no code fences, no comments.

EXAMPLE OUTPUT — for the theme "Raising children" :
{{
  "templates": [
    {{"title": "Discipline and setting boundaries", "description": "approaches to rules, consequences, and positive discipline"}},
    {{"title": "Education and academic support", "description": "school choices, homework help, learning styles"}},
    {{"title": "Emotional development and mental health", "description": "managing feelings, building resilience, anxiety"}},
    {{"title": "Screen time and technology use", "description": "managing devices, social media, online safety"}},
    {{"title": "Nutrition and healthy eating habits", "description": "meal planning, picky eaters, food relationships"}},
    {{"title": "Sleep routines and bedtime", "description": "establishing schedules, sleep problems, age-appropriate needs"}},
    {{"title": "Sibling relationships and rivalry", "description": "managing conflicts, fostering bonds, fairness"}},
    {{"title": "Communication and active listening", "description": "age-appropriate conversations, difficult topics"}},
    {{"title": "Values, morals, and character building", "description": "teaching empathy, honesty, responsibility"}},
    {{"title": "Work-life balance for parents", "description": "juggling careers, quality time, parental self-care"}}
  ]
}}

Now produce the same structure (top-level key "templates", exactly {n} items) \
for the theme above."""


TEMPLATE_SCHEMA = {
    "type": "object",
    "properties": {
        "templates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["title", "description"],
            },
        }
    },
    "required": ["templates"],
}


def theme_key(category: str, theme: str) -> str:
    return f"{category}::{theme}"


def flatten_themes(input_data: dict) -> list[dict]:
    flat = []
    for block in input_data.get("topics", []):
        for theme in block["items"]:
            flat.append({"category": block["category"], "theme": theme})
    return flat


def validate_templates(templates: Any) -> tuple[bool, str]:
    if not isinstance(templates, list):
        return False, f"templates is not a list but a {type(templates).__name__}"
    if len(templates) != N_TEMPLATES_PER_THEME:
        return False, f"wrong count: {len(templates)} (expected {N_TEMPLATES_PER_THEME})"

    seen_titles = set()
    for i, item in enumerate(templates):
        if not isinstance(item, dict):
            return False, f"item {i} is not an object"
        title = (item.get("title") or "").strip()
        desc = (item.get("description") or "").strip()
        if not title:
            return False, f"item {i}: empty title"
        if not desc:
            return False, f"item {i}: empty description"
        if len(desc.split()) < 4:
            return False, f"item {i}: description too short ('{desc}')"
        key = title.lower().strip(" .,-—–")
        if key in seen_titles:
            return False, f"duplicate title: '{title}'"
        seen_titles.add(key)

    return True, ""


def _postprocess(parsed: Any) -> tuple[list | None, str]:
    templates = find_list(parsed, LIST_ALIASES)
    if templates is None:
        keys = list(parsed.keys())[:5] if isinstance(parsed, dict) else type(parsed).__name__
        return None, f"no_list_found_in_json (top_keys={keys})"
    ok, reason = validate_templates(templates)
    return (templates, "") if ok else (None, reason)


def _cache_is_valid(record: dict) -> bool:
    templates = record.get("templates")
    return bool(templates) and validate_templates(templates)[0]


async def _generate(themes: list[dict], chat: OpenAIChat, cache: Checkpoint,
                    concurrency: int) -> dict:
    log.info("Resume: %d themes already cached", len(cache))
    todo = cache.pending(themes, key=lambda t: theme_key(t["category"], t["theme"]),
                         is_valid=_cache_is_valid)
    log.info("To generate: %d themes (concurrency=%d, %d templates/theme)",
             len(todo), concurrency, N_TEMPLATES_PER_THEME)
    if not todo:
        return cache.data

    async def worker(entry: dict) -> None:
        category, theme = entry["category"], entry["theme"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                n=N_TEMPLATES_PER_THEME, category=category, theme=theme)},
        ]
        result = await chat.generate(messages, _postprocess, label=theme,
                                     max_retries=GEN_MAX_RETRIES)
        if not result.ok:
            log.warning("  failure '%s': %s", theme, result.status)
        await cache.record(theme_key(category, theme), {
            "category": category,
            "theme": theme,
            "templates": result.value,
            "raw_generation": result.raw,
            "generation_status": result.status,
        })

    await run_pool(todo, worker, concurrency=concurrency, desc="Template generation")
    cache.flush()
    n_ok = sum(1 for r in cache.data.values() if r.get("templates"))
    log.info("Generation done: %d valid themes out of %d processed", n_ok, len(cache))
    return cache.data


def finalize(results: dict, paths: Paths) -> None:
    by_category: dict[str, dict[str, list[dict]]] = {}
    flat: list[dict] = []
    n_themes_ok = 0

    for record in results.values():
        if not record.get("templates"):
            continue
        category, theme = record["category"], record["theme"]
        by_category.setdefault(category, {})[theme] = record["templates"]
        n_themes_ok += 1
        for i, template in enumerate(record["templates"]):
            flat.append({
                "template_id": f"{category}__{theme}__{i + 1:02d}".replace(" ", "_"),
                "category": category,
                "theme": theme,
                "index": i + 1,
                "title": template["title"],
                "description": template["description"],
            })

    save_json(paths.templates, by_category)
    save_json(paths.templates_flat, flat)
    log.info("Final dataset: %d themes / %d templates", n_themes_ok, len(flat))
    log.info("  -> %s", paths.templates)
    log.info("  -> %s", paths.templates_flat)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default=str(THEMES),
                        help="Theme JSON {topics: [{category, items}]}")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default=None, help="required except with --finalize-only")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=None,
                        help="Debug: process N themes only")
    parser.add_argument("--struct-mode",
                        choices=["auto", "response_format", "guided_json", "json_object", "none"],
                        default="auto",
                        help="Server-side structured output; 'auto' tries "
                             "response_format -> guided_json -> json_object")
    parser.add_argument("--finalize-only", action="store_true",
                        help="No generation, rebuild the final files from the cache")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Drop the themes without a template from the cache, then re-run")


def run(args: argparse.Namespace) -> int:
    paths = Paths.resolve(args.data_dir)
    cache = Checkpoint(paths.raw_templates, save_every=SAVE_EVERY)

    if args.finalize_only:
        if not cache.data:
            log.error("No %s found.", paths.raw_templates)
            return 1
        finalize(cache.data, paths)
        return 0

    if not args.model:
        log.error("--model is required (except with --finalize-only)")
        return 2

    themes_data = load_json(args.input)
    if themes_data is None:
        log.error("Input file not found: %s", args.input)
        return 1
    themes = flatten_themes(themes_data)
    if args.limit:
        themes = themes[: args.limit]
    log.info("Themes loaded: %d", len(themes))

    if args.retry_failed:
        removed = cache.drop(lambda r: bool(r.get("templates")))
        log.info("--retry-failed: %d failures dropped from the cache", removed)

    chat = OpenAIChat(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        schema=TEMPLATE_SCHEMA,
        schema_name="templates_response",
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        max_tokens=GEN_MAX_NEW_TOKENS,
        struct_mode=args.struct_mode,
        max_retries=GEN_MAX_RETRIES,
        timeout=HTTP_TIMEOUT,
    )

    results = run_async(_generate(themes, chat, cache, args.concurrency))
    finalize(results, paths)
    return 0
