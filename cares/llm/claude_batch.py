from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from ..jsonio import load_json, save_json
from ..log import get_logger
from .claude import parsed_from_message

log = get_logger(__name__)

T = TypeVar("T")

MAX_REQUESTS_PER_BATCH = 100_000

MAX_BATCH_BYTES = 256 * 1024 * 1024

BATCH_BYTES_MARGIN = 0.9

POLL_INTERVAL = 60


def safe_custom_id(item_id: str, used: dict[str, str]) -> str:
    cid = re.sub(r"[^a-zA-Z0-9_-]", "-", item_id)[:64] or "id"
    if used.get(cid) not in (None, item_id):
        digest = hashlib.sha1(item_id.encode()).hexdigest()[:8]
        cid = (cid[:55] + "-" + digest)[:64]
    used[cid] = item_id
    return cid


def poll_batch(client: Any, batch_id: str, wait: bool, interval: int) -> Any:
    batch = client.messages.batches.retrieve(batch_id)
    while wait and batch.processing_status != "ended":
        counts = batch.request_counts
        log.info("  batch %s: %s (succeeded=%s errored=%s processing=%s "
                 "canceled=%s expired=%s)",
                 batch_id, batch.processing_status, counts.succeeded, counts.errored,
                 counts.processing, counts.canceled, counts.expired)
        time.sleep(interval)
        batch = client.messages.batches.retrieve(batch_id)
    return batch


def collect_batch(
    client: Any,
    batch_id: str,
    id_map: dict[str, str],
    results: dict[str, Any],
    *,
    tool_name: str,
    parse_result: Callable[[str, Any | None, str], tuple[Any | None, str]],
    make_record: Callable[[str, Any | None, str, str], dict],
) -> int:
    n_ok = 0
    for res in client.messages.batches.results(batch_id):
        item_id = id_map.get(res.custom_id, res.custom_id)
        rtype = res.result.type
        value, raw, status = None, "", f"batch:{rtype}"

        if rtype == "succeeded":
            parsed, raw = parsed_from_message(res.result.message, tool_name)
            value, status = parse_result(item_id, parsed, raw)
            if value is not None:
                n_ok += 1
        elif rtype == "errored":
            try:
                status = f"batch:errored:{res.result.error.error.type}"
                raw = str(res.result.error)[:400]
            except Exception:  # noqa: BLE001
                status = "batch:errored:unknown"

        results[item_id] = make_record(item_id, value, raw, status)
    return n_ok


def _chunks(items: list[T], build_params: Callable[[T], dict],
            item_id: Callable[[T], str], max_per_batch: int):
    budget = int(MAX_BATCH_BYTES * BATCH_BYTES_MARGIN)
    chunk: list[tuple[T, dict]] = []
    size = 0

    for item in items:
        try:
            params = build_params(item)
        except ValueError as exc:
            log.warning("  skip '%s': %s", item_id(item), exc)
            continue
        weight = len(json.dumps(params, ensure_ascii=False).encode("utf-8"))
        if weight > budget:
            log.warning("  skip '%s': %d bytes, over the budget of one batch (%d)",
                        item_id(item), weight, budget)
            continue
        if chunk and (len(chunk) >= max_per_batch or size + weight > budget):
            yield chunk
            chunk, size = [], 0
        chunk.append((item, params))
        size += weight

    if chunk:
        yield chunk


def run_batch(
    *,
    client: Any,
    items: list[T],
    item_id: Callable[[T], str],
    build_params: Callable[[T], dict],
    results: dict[str, Any],
    results_path: Path,
    state_path: Path,
    needs_work: Callable[[T, Any], bool],
    tool_name: str,
    parse_result: Callable[[str, Any | None, str], tuple[Any | None, str]],
    make_record: Callable[[str, Any | None, str, str], dict],
    max_per_batch: int = MAX_REQUESTS_PER_BATCH,
    wait: bool = True,
    interval: int = POLL_INTERVAL,
    desc: str = "requests",
) -> dict[str, Any]:
    state = load_json(state_path, default={}) or {}

    in_flight: set[str] = set()
    for batch_id, info in list(state.items()):
        if info.get("status") == "collected":
            continue
        log.info("Existing batch %s: checking...", batch_id)
        batch = poll_batch(client, batch_id, wait, interval)
        if batch.processing_status != "ended":
            in_flight.update(info.get("id_map", {}).values())
            log.info("  %s still running (%d requests) -> rerun --batch later "
                     "to collect.", batch_id, len(info.get("id_map", {})))
            continue
        n = collect_batch(client, batch_id, info.get("id_map", {}), results,
                          tool_name=tool_name, parse_result=parse_result,
                          make_record=make_record)
        info["status"] = "collected"
        save_json(results_path, results)
        save_json(state_path, state)
        log.info("  %s collected: %d valid results", batch_id, n)

    pending = [item for item in items
               if item_id(item) not in in_flight
               and needs_work(item, results.get(item_id(item)))]
    log.info("To submit in batch: %d %s (wait=%s)", len(pending), desc, wait)
    if in_flight:
        log.info("  (%d already in flight in an uncollected batch)", len(in_flight))
    if not pending:
        return results

    for chunk in _chunks(pending, build_params, item_id, max_per_batch):
        used: dict[str, str] = {}
        requests = [{"custom_id": safe_custom_id(item_id(item), used), "params": params}
                    for item, params in chunk]
        if not requests:
            continue

        batch = client.messages.batches.create(requests=requests)
        state[batch.id] = {"status": "submitted", "id_map": used}
        save_json(state_path, state)
        log.info("Batch submitted: %s (%d requests). Rerun --batch to collect.",
                 batch.id, len(requests))

        batch = poll_batch(client, batch.id, wait, interval)
        if batch.processing_status != "ended":
            log.info("  %s still running -> rerun --batch later.", batch.id)
            continue
        n = collect_batch(client, batch.id, used, results, tool_name=tool_name,
                          parse_result=parse_result, make_record=make_record)
        state[batch.id]["status"] = "collected"
        save_json(results_path, results)
        save_json(state_path, state)
        log.info("  %s collected: %d valid results", batch.id, n)

    return results
