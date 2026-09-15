from __future__ import annotations

import re

VOCAL_TAGS: dict[str, str] = {
    "cough": "[coughs]",
    "coughing": "[coughs]",
    "sneeze": "[sneezes]",
    "sneezing": "[sneezes]",
    "laugh": "[laughs]",
    "laughter": "[laughs]",
    "laughing": "[laughs]",
    "chuckle": "[chuckles]",
    "giggle": "[giggles]",
    "sigh": "[sighs]",
    "sighing": "[sighs]",
    "gasp": "[gasps]",
    "groan": "[groans]",
    "yawn": "[yawns]",
    "sniff": "[sniffs]",
    "sniffle": "[sniffs]",
    "snort": "[snorts]",
    "hiccup": "[hiccups]",
    "throat": "[clears throat]",
    "clearing": "[clears throat]",
    "clear": "[clears throat]",
    "cry": "[crying]",
    "crying": "[crying]",
    "sob": "[crying]",
    "gulp": "[gulps]",
    "swallow": "[swallows]",
    "whistle": "[whistles]",
    "burp": "[burps]",
}

SFX_TAGS: dict[str, str] = {
    "gunshot": "[gunshot]",
    "gun": "[gunshot]",
    "explosion": "[explosion]",
    "applause": "[applause]",
    "clap": "[clapping]",
    "clapping": "[clapping]",
}


def event_leaf(event_id: str) -> str:
    return event_id.split("/", 1)[1] if "/" in event_id else event_id


def event_to_tag(
    event_id: str,
    *,
    include_sfx: bool = False,
    extra: dict[str, str] | None = None,
) -> str | None:
    if not event_id:
        return None

    table: dict[str, str] = dict(VOCAL_TAGS)
    if include_sfx:
        table.update(SFX_TAGS)
    if extra:
        table.update(extra)

    if event_id in table:
        return table[event_id]
    if event_id.lower() in table:
        return table[event_id.lower()]

    leaf = event_leaf(event_id).lower()

    if leaf in table:
        return table[leaf]

    tokens = [t for t in re.split(r"[_\-\s]+", leaf) if t]
    canonical = {k: v for k, v in table.items() if "/" not in k and " " not in k}
    if tokens and all(t in canonical for t in tokens):
        return canonical[tokens[0]]
    return None


def is_performable(
    event_id: str,
    *,
    include_sfx: bool = False,
    extra: dict[str, str] | None = None,
) -> bool:
    return event_to_tag(event_id, include_sfx=include_sfx, extra=extra) is not None
