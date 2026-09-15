from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..dataset import REACTIONS, stratum_key
from ..log import get_logger

log = get_logger(__name__)


def safe_div(a: float | None, b: float | None) -> float | None:
    return a / b if b else None


def _fmt(x: Any) -> str:
    return f"{x:5.2f}" if isinstance(x, (int, float)) else "  n/a"


def build_confusion(pairs: list[tuple]) -> dict:
    counts: Counter = Counter(pairs)
    has_missing = any(p == "missing" for _, p in pairs)
    pred_labels = list(REACTIONS) + (["missing"] if has_missing else [])
    matrix = [[counts.get((g, p), 0) for p in pred_labels] for g in REACTIONS]
    return {
        "labels_gold": list(REACTIONS),
        "labels_pred": pred_labels,
        "matrix": matrix,
        "counts": {f"{g}->{p}": c for (g, p), c in counts.items()},
    }


def per_class_metrics(counts: Counter) -> dict:
    out = {}
    f1_scores = []
    for label in REACTIONS:
        tp = counts.get((label, label), 0)
        fn = sum(c for (g, p), c in counts.items() if g == label and p != label)
        fp = sum(c for (g, p), c in counts.items() if g != label and p == label)
        support = sum(c for (g, _), c in counts.items() if g == label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        if precision and recall and (precision + recall) > 0:
            f1 = safe_div(2 * precision * recall, precision + recall)
        else:
            f1 = 0.0 if support and tp == 0 else None
        out[label] = {"precision": precision, "recall": recall, "f1": f1,
                      "support": support, "tp": tp, "fp": fp, "fn": fn}
        if support:
            f1_scores.append(f1 if f1 is not None else 0.0)

    n_supported = sum(1 for k in REACTIONS if out[k]["support"])
    return {
        "per_class": out,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else None,
        "macro_precision": safe_div(
            sum(out[k]["precision"] or 0 for k in REACTIONS if out[k]["support"]), n_supported),
        "macro_recall": safe_div(
            sum(out[k]["recall"] or 0 for k in REACTIONS if out[k]["support"]), n_supported),
    }


def cohen_kappa(pairs: list[tuple]) -> float | None:
    valid = [(g, p) for g, p in pairs if p in REACTIONS]
    n = len(valid)
    if n == 0:
        return None
    observed = sum(1 for g, p in valid if g == p) / n
    gold_counts = Counter(g for g, _ in valid)
    pred_counts = Counter(p for _, p in valid)
    expected = sum((gold_counts[k] / n) * (pred_counts[k] / n) for k in REACTIONS)
    return (observed - expected) / (1 - expected) if (1 - expected) > 0 else None


def compute_metrics(scenarios: list[dict], raw_inferences: dict, tau: float) -> dict:
    pairs_all: list[tuple] = []
    cos_values: list[float] = []
    topic_hits = 0
    n_eval = 0
    n_missing_total = 0
    n_spurious_total = 0
    n_reaction_exact = 0
    n_accepted = 0

    by_stratum: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "react_ok": 0, "topic_ok": 0, "accept": 0, "pairs": []})

    for scenario in scenarios:
        record = raw_inferences.get(scenario["scenario_id"])
        if not record or not record.get("eval"):
            continue
        evaluation = record["eval"]
        if (evaluation.get("subject_cos") is None and not evaluation.get("pairs")
                and evaluation.get("n_sounds", 0) > 0):
            continue

        n_eval += 1
        stratum = stratum_key(scenario)
        pairs = [tuple(p) for p in evaluation.get("pairs", [])]
        pairs_all.extend(pairs)
        by_stratum[stratum]["pairs"].extend(pairs)

        cos = evaluation.get("subject_cos")
        if cos is not None:
            cos_values.append(cos)
        topic_ok = cos is not None and cos >= tau
        reactions_ok = bool(evaluation.get("reactions_ok"))
        accepted = topic_ok and reactions_ok

        topic_hits += int(topic_ok)
        n_reaction_exact += int(reactions_ok)
        n_accepted += int(accepted)
        n_missing_total += evaluation.get("n_missing", 0)
        n_spurious_total += evaluation.get("n_spurious", 0)

        stats = by_stratum[stratum]
        stats["n"] += 1
        stats["react_ok"] += int(reactions_ok)
        stats["topic_ok"] += int(topic_ok)
        stats["accept"] += int(accepted)

    counts = Counter(pairs_all)
    class_metrics = per_class_metrics(counts)
    metrics = {
        "n_scenarios_evaluated": n_eval,
        "tau": tau,
        "n_sounds_total": len(pairs_all),
        "n_sounds_typed": sum(1 for _, p in pairs_all if p in REACTIONS),
        "n_missing": n_missing_total,
        "n_spurious": n_spurious_total,
        "reaction_typing": {
            "micro_accuracy": safe_div(sum(counts.get((k, k), 0) for k in REACTIONS),
                                       len(pairs_all)),
            "macro_f1": class_metrics["macro_f1"],
            "macro_precision": class_metrics["macro_precision"],
            "macro_recall": class_metrics["macro_recall"],
            "cohen_kappa": cohen_kappa(pairs_all),
            "per_class": class_metrics["per_class"],
        },
        "confusion_matrix": build_confusion(pairs_all),
        "topic_recovery": {
            "recovery_rate": safe_div(topic_hits, n_eval),
            "mean_cosine": safe_div(sum(cos_values), len(cos_values)),
            "n_with_subject": len(cos_values),
        },
        "exact_match_rate_last": safe_div(n_accepted, n_eval),
        "reaction_exact_rate_last": safe_div(n_reaction_exact, n_eval),
        "per_stratum": {},
    }

    for stratum, stats in sorted(by_stratum.items()):
        stratum_counts = Counter(stats["pairs"])
        metrics["per_stratum"][stratum] = {
            "n": stats["n"],
            "reaction_exact_rate": safe_div(stats["react_ok"], stats["n"]),
            "topic_recovery_rate": safe_div(stats["topic_ok"], stats["n"]),
            "accept_rate": safe_div(stats["accept"], stats["n"]),
            "sound_accuracy": safe_div(
                sum(stratum_counts.get((k, k), 0) for k in REACTIONS),
                sum(stratum_counts.values()) or 0) if stats["pairs"] else None,
        }
    return metrics


def _tex_escape(text: str, tex: bool) -> str:
    if not tex:
        return text
    for char, escaped in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                          ("&", r"\&"), ("#", r"\#"), ("$", r"\$")):
        text = text.replace(char, escaped)
    return text


def plot_confusion_matrix(confusion: dict, png_path: Path, pdf_path: Path | None = None,
                          title: str = "Reaction-typing confusion (gold vs recovered)",
                          usetex: bool = True) -> None:
    try:
        import matplotlib
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotting requires `pip install 'cares[analysis]'`") from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels_gold = confusion["labels_gold"]
    labels_pred = confusion["labels_pred"]
    matrix = np.array(confusion["matrix"], dtype="float64")
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
    n_gold, n_pred = matrix.shape

    def percentage(i: int, j: int, tex: bool) -> str:
        if row_sums[i] == 0:
            return ""
        value = normalized[i, j] * 100.0
        return rf"{value:.0f}\%" if tex else f"{value:.0f}%"

    def draw(tex: bool) -> None:
        rc = {"text.usetex": tex, "font.family": "serif",
              "mathtext.fontset": "cm", "axes.unicode_minus": False}
        with matplotlib.rc_context(rc):
            fig, ax = plt.subplots(figsize=(1.6 + 1.05 * n_pred, 1.5 + 1.0 * n_gold))
            image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
            ax.set_xticks(range(n_pred))
            ax.set_yticks(range(n_gold))
            ax.set_xticklabels([_tex_escape(x, tex) for x in labels_pred],
                               rotation=30, ha="right")
            ax.set_yticklabels([_tex_escape(x, tex) for x in labels_gold])
            ax.set_xlabel("Recovered ($g$)" if tex else "Recovered (g)")
            ax.set_ylabel("Gold")
            if title:
                ax.set_title(_tex_escape(title, tex))
            for i in range(n_gold):
                for j in range(n_pred):
                    ax.text(j, i, percentage(i, j, tex), ha="center", va="center",
                            fontsize=11,
                            color="white" if normalized[i, j] > 0.6 else "#1a1a1a")
            colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            colorbar.set_label("row-normalized (recall)")
            fig.tight_layout()
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png_path, dpi=200, bbox_inches="tight")
            if pdf_path is not None:
                fig.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)

    try:
        draw(usetex)
        used_tex = usetex
    except Exception as exc:  # noqa: BLE001
        if not usetex:
            raise
        log.warning("LaTeX rendering unavailable (%s) -> mathtext fallback.", str(exc)[:120])
        draw(False)
        used_tex = False

    log.info("Confusion matrix plotted -> %s%s%s", png_path,
             f" et {pdf_path}" if pdf_path else "",
             " [LaTeX]" if used_tex else " [mathtext CM]")


def log_ascii_confusion(confusion: dict) -> None:
    header = " " * 12 + "".join(f"{p[:9]:>10s}" for p in confusion["labels_pred"])
    log.info("Confusion (rows=gold, columns=predicted):")
    log.info("%s", header)
    for gold, row in zip(confusion["labels_gold"], confusion["matrix"], strict=True):
        log.info("%s", f"{gold:>11s} " + "".join(f"{v:>10d}" for v in row))


def log_metrics(metrics: dict) -> None:
    typing = metrics["reaction_typing"]
    log.info("=" * 64)
    log.info("REACTION-TYPING METRICS")
    log.info("=" * 64)
    log.info("Evaluated: %d  sounds: %d (typed %d, missing %d, spurious %d)  (tau=%s)",
             metrics["n_scenarios_evaluated"], metrics["n_sounds_total"],
             metrics["n_sounds_typed"], metrics["n_missing"], metrics["n_spurious"],
             metrics["tau"])
    log.info("micro-acc=%s  macro-F1=%s  macro-P=%s  macro-R=%s  kappa=%s",
             _fmt(typing["micro_accuracy"]), _fmt(typing["macro_f1"]),
             _fmt(typing["macro_precision"]), _fmt(typing["macro_recall"]),
             _fmt(typing["cohen_kappa"]))
    log.info("  %-12s %5s %5s %5s %6s", "class", "P", "R", "F1", "supp")
    for label in REACTIONS:
        per_class = typing["per_class"][label]
        log.info("  %-12s %s %s %s %6d", label, _fmt(per_class["precision"]),
                 _fmt(per_class["recall"]), _fmt(per_class["f1"]), per_class["support"])

    recovery = metrics["topic_recovery"]
    log.info("Topic recovery (cos>=tau): %s  (mean cos %s)",
             _fmt(recovery["recovery_rate"]), _fmt(recovery["mean_cosine"]))
    log.info("Exact match (topic+typing, last inf.): %s   exact typing: %s",
             _fmt(metrics["exact_match_rate_last"]), _fmt(metrics["reaction_exact_rate_last"]))
    log_ascii_confusion(metrics["confusion_matrix"])
    log.info("-" * 64)
    log.info("  %-8s %5s %7s %7s %6s %7s", "stratum", "n", "react%", "topic%", "acc%", "sndAcc")
    for stratum, stats in metrics["per_stratum"].items():
        log.info("  %-8s %5d %5.1f%% %6.1f%% %5.1f%% %s", stratum, stats["n"],
                 (stats["reaction_exact_rate"] or 0) * 100,
                 (stats["topic_recovery_rate"] or 0) * 100,
                 (stats["accept_rate"] or 0) * 100,
                 _fmt(stats["sound_accuracy"]))
