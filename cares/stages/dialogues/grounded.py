from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Any

from ...allocation import (
    GENDER_WORDS,
    SLOT_WORDS,
    register_move,
    register_required_tags,
)
from ...dataset import REACTION_TAG, normalize_scenario, strip_prosody_tags, utterances
from .common import (
    MANNER_TAG_WORDS,
    MAX_TURNS,
    MIN_TURNS,
    VOCAL_TAG_SPELLINGS,
    WEAK_TAG_WORDS,
    WORN_PHRASINGS,
    add_common_arguments,
    check_turn_counts,
    max_turns_for,
    repair_item_types,
    run_variant,
    scan_timeline,
    strengthen_prosody_tags,
)

MIN_PIVOT_REACTION_TURNS = 4


SYSTEM_PROMPT = """You are a dialogue writer for a TTS audio-scene dataset. \
You write realistic spoken English between two adult speakers who are \
physically together at a given location. Specific acoustic events are woven \
into the timeline, and each event carries a REACTION TYPE that your dialogue \
must realize. You answer with strict JSON only, matching the schema given by \
the runtime."""


USER_PROMPT = """Write a realistic dialogue scene between Speaker A and \
Speaker B. The subject, the events, and each event's reaction type are all \
PRE-ALLOCATED. Your job is to write the actual lines and choose where each \
acoustic event falls in the timeline.

═══════════════════════════════════════════════════════════════════
SCENARIO
═══════════════════════════════════════════════════════════════════
Theme           : {theme_title}
Theme details   : {theme_description}
Scene/Location  : {scene}
Speaker A       : {role_a} ({gender_a})
Speaker B       : {role_b} ({gender_b})

CONVERSATION SUBJECT — what the two speakers are talking about:
    {subject}

The dialogue must be ABOUT this subject. Both speakers take part naturally: \
one may bring it up, the other reacts, asks follow-ups, disagrees, adds their \
own angle. YOU decide who says what — there is no fixed speaker assignment for \
the subject.

The two speakers are PHYSICALLY TOGETHER at the location above. They are \
NEVER on a phone call with each other. NEVER use phrases like "are you \
there", "hold on a sec, I'm on a call", "can you hear me", "speakerphone". \
They can see and hear each other directly.

═══════════════════════════════════════════════════════════════════
ACOUSTIC EVENTS AND HOW THE SPEAKERS REACT
═══════════════════════════════════════════════════════════════════
Each event below carries a REACTION TYPE that you MUST realize in the \
dialogue. YOU choose WHERE each event falls in the timeline. The four types:

  • PIVOT — the sound INTERRUPTS and the conversation TURNS TO IT. After it \
occurs, the speakers react to it and the talk is ABOUT it (and its aftermath) \
for AT LEAST {min_pivot_turns} of the following turns; the original subject \
becomes secondary from that point on. There is at most one pivot in a scene.
  • VERBAL — a speaker briefly REACTS ALOUD to the sound, then the \
conversation CONTINUES on its course. It is a passing remark, never the \
centre, no real follow-up. The reaction may refer to the sound, but should \
RARELY name it by its label: someone hearing a plane take off is far likelier \
to react to its sheer power, or to having to raise their voice over it, than \
to announce that a plane took off. React to some ASPECT of it — what it was \
like, what it means, what it did to the room — and let the label stay unsaid \
unless the drawn register calls for it.
  • BEHAVIORAL — the sound is never mentioned, NOT EVEN OBLIQUELY, but it \
visibly DERAILS the speaking at the moment it occurs. The speaker who is \
talking loses their thread, and then picks the SAME subject back up. The \
giveaway is a break in the flow tied to the sound, followed by a return to \
what was being said. A smooth, coherent change of subject is NOT behavioral: \
it is indistinguishable from normal conversation and will read as ambient. \
The disruption must be audible as a break, not a tidy topic switch. It often \
works best if the SAME speaker keeps the turn across the sound, so the derail \
shows inside their own continued line. The break can live in the WORDS (a \
sentence that comes apart, a word hunted for out loud) or in the VOICE alone \
(a hole in the middle of a phrase, a caught breath) — the drawn register below \
says which, and a register that names required tags admits no other way.
  • AMBIENT — pure background. No one reacts, no one refers to it, the \
dialogue continues exactly as if it were not there.

Events to place (each MUST appear exactly once in the timeline):
{events_block}

REALIZING EACH REACTION — do this carefully, it is the point of the scene:
  • Every event must get the reaction its type calls for, clearly tied to the \
moment it occurs in the timeline (the reaction lands in the very next turn or \
two for pivot/verbal/behavioral).
  • If there are SEVERAL reacted-to events (verbal and/or behavioral), give \
EACH its OWN distinct, separately placed reaction. A common mistake is to make \
the first one land and let the next slip by as background. Space them out at \
clearly different points so each has room. Do not bundle two sounds into one \
shared "what was all that".
  • Each reacted-to event below carries a REGISTER: the move its reaction \
must make. Realise that move. Registers describe what the speaker DOES, never \
what they say — invent the wording yourself, and do not reach for the phrasing \
that first comes to mind, it is the one every other scene will already have \
used. Two events with different registers must read as visibly different \
reactions.
  • For BEHAVIORAL specifically, the derail must be audible as a break in the \
line and the speaker must return to the SAME subject. A clean switch to a new \
topic is NOT behavioral and will read as ambient.
  • BEHAVIORAL — WHERE the break falls. The sound is heard just before the \
turn that follows it, so the break must come in the FIRST SENTENCE of that \
turn. A turn that runs fluently for two or three sentences and only then \
stumbles reads as an unrelated hesitation: by then the listener has no reason \
to connect it to anything.
  • BEHAVIORAL — never explain the interruption. The speaker may show what \
they are doing, but must NEVER give a reason for stopping. "hold on... there, \
okay" is right; "hold on, this chair leg is sinking into the grass" is wrong, \
because it hands the listener a cause that is not the sound, and the sound \
then explains nothing.
  • BEHAVIORAL — do NOT use the stock ways of asking for a lost place. These \
exact turns of phrase are REJECTED and the scene will be thrown away: \
{worn_phrasings}. The move — getting the other speaker to hand your place \
back — is fine and often good; it is the wording that is worn out. Invent \
another way in: name the last concrete thing you remember and ask what came \
after it, start the sentence again from further back, or hand the floor over \
without admitting anything.
  • Before finishing, check each event one by one against its type: does the \
dialogue do exactly what that type requires (turn to it / briefly name it / \
silently adapt / ignore it)? If any event is mistyped, the scene is not done.

{pivot_directive}
PLACEMENT AND TONAL FIT:
  • WHERE each event goes is ASSIGNED, not yours to choose: every event below \
carries a PLACEMENT band, and it must occur inside that third of the \
conversation. Events are listed in the order they must occur.
  • The band is fixed; the exact beat inside it is yours. Pick the moment \
within the band where the sound sits best, and never let a light or trivial \
sound (glasses clinking, a cheerful chime) land on a heavy moment (grief, bad \
news, conflict) — write the conversation so the band and the mood agree.
  • Do not bunch events together: they are in different bands, keep them apart.

═══════════════════════════════════════════════════════════════════
NEVER NAME THE ACOUSTIC EVENTS BY THEIR LABEL
═══════════════════════════════════════════════════════════════════
The utterances must NEVER directly name or describe an acoustic event by its \
category. Do not say "the coffee machine just hissed", "that was a gunshot", \
"I heard glass shatter".
  • For PIVOT and VERBAL events, the speakers may REFER to the sound obliquely \
— "what was that?", "did you hear that?", "that came from outside", "that's so \
loud" — but never name the category itself.
  • For BEHAVIORAL and AMBIENT events, the speakers do not refer to the sound \
at all.
The audio listener infers the sound from the audio alone; we test whether a \
model correctly LINKS the conversation to the unnamed event. (All event_ids \
you are given are ENVIRONMENTAL sounds; none is a vocal sound made by a \
speaker.)

═══════════════════════════════════════════════════════════════════
LENGTH AND STRUCTURE
═══════════════════════════════════════════════════════════════════
- Produce between {min_turns} and {max_turns} utterance turns total \
(events do NOT count toward this turn count).
- Alternate speakers naturally. Both A and B must speak. A single speaker \
may have two consecutive turns occasionally if it feels natural, but most of \
the time speakers alternate.
- The dialogue starts in the middle of an existing interaction (in medias \
res) — NOT with a formal greeting like "Hi", "Hello", "Hey, how are you" \
unless that genuinely fits the relation. Real conversations rarely start cold.

═══════════════════════════════════════════════════════════════════
STYLE — REALISTIC SPOKEN ENGLISH
═══════════════════════════════════════════════════════════════════
- Natural spoken English, casual register, full of contractions ("I'm", \
"don't", "we've", "yeah").
- Conversational markers: "right", "yeah", "you know", "I mean", "honestly", \
"actually". Use them sparingly so the dialogue does not feel parodic.
- Allow interruptions, false starts, reformulations: "Wait no, I meant the \
other one", "It was just, you know, kind of weird".
- The relation between the two speakers must shape the register. Two close \
friends speak differently from a retired aunt and her adult nephew.
- NEVER use hesitation fillers like "uh", "um", "er", "erm", "hmm".
- NEVER use em dashes (—), semicolons (;), or parentheses (()) anywhere \
in the text. Stick to periods, commas, question marks, exclamation marks, \
and apostrophes.
- NO emojis, NO sound effects spelled out, NO speaker labels inside text. \
NO prose stage directions describing actions (do not write things like \
"she laughs" or "he leans back"). The ONLY bracketed markers allowed inside \
the text are the prosody / non-verbal vocal tags described next.

═══════════════════════════════════════════════════════════════════
EMOTIONAL PROSODY TAGS (optional, for expressive TTS)
═══════════════════════════════════════════════════════════════════
The speech synthesizer understands inline tags in square brackets that shape \
HOW a line is delivered. You MAY add these inside the "text" of an utterance, \
wherever they make delivery more natural. Which lines you tag is a matter of \
judgement: tag where emotion, energy, or pacing would realistically colour the \
line, and leave plain lines plain. Do not tag every line.

What these tags are:
  • They describe the EMOTION, VOLUME, ENERGY, or PACING of the delivery. \
Examples (not limited to this list): [excited], [nervous], [frustrated], \
[tired], [amused], [hesitant], [questioning], [sarcastic], [gently], \
[reluctantly], [whispering], [quietly], [pauses], [startled]. \
A short combined form is fine, e.g. [angrily, fed up].
  • Place a tag immediately before the part of the line it affects. It can \
open a line ("[nervously] I don't know if I should say this."), sit between \
sentences ("That's fine. [quietly] I guess.") or fall INSIDE a sentence, \
between two words that belong together ("I left them under the [pauses] mat"). \
That last position is the strongest: it is the only one the ear cannot mistake \
for ordinary breathing, and some registers below REQUIRE it.
  • A startled reaction to a pivot/verbal sound is a natural place for \
[startled] or [alarmed].

STRENGTH — this matters more than which tag you choose:
  • The synthesizer acts on a BLUNT direction and ignores a hedged one. A \
tag that is qualified away is WORSE than no tag at all: it stays in the \
written line, so the transcript claims the delivery changed, while the audio \
is indistinguishable from a plain reading.
  • NEVER put any of these words inside a tag: {weak_tag_words}.
  • Write [furious], not [somewhat annoyed]. Write [sighs], not \
[a small sigh]. Write [whispering], not [a little quiet].
  • Prefer a tag that names a DEFINITE state — [furious], [startled], \
[whispering], [pauses], [sighs] — over one that names a shade of a state. If \
the delivery does not warrant a blunt tag, write no tag.

MANNER IS NOT A TAG EITHER — it goes in what the speaker says and does:
  • The synthesizer sets an EMOTION over a whole line, and it inserts an EVENT \
(a laugh, a breath, a silence). It does NOT change the MANNER of a delivery \
partway through, because that asks for a contrast with what came before and \
there is nothing to contrast against. Measured: nine acoustic parameters, none \
of them moves.
  • NEVER put any of these words inside a tag: {manner_tag_words}. A tag built \
on one of them is silently dropped before the dialogue is saved.
  • [flat, clipped] then [warm again] is the exact shape that fails: it reads \
as a delivery change in the transcript and is inaudible in the recording. If a \
speaker goes cold, show it in what they say, or use a definite emotion tag.

VOLUME IS NOT A TAG — write it into the words:
  • The synthesizer does NOT raise the voice for [raising voice], [shouting] \
or [loudly]. It DOES raise it for CAPITAL LETTERS and exclamation marks. \
Never write a volume tag; write the loud part in CAPITALS instead and end it \
with "!".
  • "[raising voice] Like clockwork, that." must be written \
"LIKE CLOCKWORK, THAT!" with no tag at all.
  • Capitalise only the words actually raised — a phrase, rarely a whole \
sentence — and leave the rest of the turn in normal case. A turn written \
entirely in capitals reads as one long shout.
  • Exclamation marks are the second lever and can be used a little more \
freely than ordinary writing would allow. Question marks keep their "?".
  • Tags remain the right tool for EMOTION ([furious], [startled], [amused]), \
for PACING ([pauses], [hesitant]) and for VOCAL SOUNDS ([sighs], [gasps]) — \
VOLUME moves into the spelling, and MANNER is not written as a tag at all.

CRITICAL — these prosody tags are NOT acoustic events:
  • A prosody tag changes how a HUMAN VOICE sounds. It lives INSIDE the "text" \
string of an utterance.
  • An acoustic event is a NON-SPEECH ENVIRONMENTAL sound and is represented \
ONLY as a separate timeline item {{"type": "event", "event_id": "..."}}, never \
as a tag in the text.
  • Never name or describe an acoustic event through a tag either.

═══════════════════════════════════════════════════════════════════
NON-VERBAL VOCAL SOUNDS (optional, inline tags — for expressive TTS)
═══════════════════════════════════════════════════════════════════
The synthesizer can also produce NON-VERBAL VOCAL SOUNDS made by the speaker: \
a laugh, a sigh, a sniff, a throat-clear. These are written the SAME way as \
prosody tags — inline, in square brackets, inside the "text" — and make the \
speaker emit that sound. Use a LIGHT TOUCH (a few across the whole dialogue).

Use these EXACT tag spellings:
  {vocal_tag_spellings}
A combined form with an emotion is fine, e.g. "[nervous] I... [gulps] okay."

IMPORTANT — keep these subtle and in character:
  • A non-verbal vocal sound is NOT a topic and NOT an event to be discussed. \
The OTHER speaker must NOT react to it or turn it into a subject (no "are you \
OK?", "bless you", "why are you laughing?"). It happens inside one line and \
the conversation flows on.
  • These vocal tags are part of the SPEECH. They are NEVER timeline events \
and must NEVER be written as {{"type": "event", ...}}. They live only inside \
"text".
  • Every utterance must still contain actual words to be spoken.

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT — strict JSON, no preamble, no postamble, no code fences
═══════════════════════════════════════════════════════════════════
A single JSON object with one key "timeline" mapping to an ORDERED array. \
Each array item is one of:

  An utterance:
    {{"type": "utterance", "speaker": "A" or "B", "text": "..."}}

  An acoustic event placeholder (the event_id is given to you above):
    {{"type": "event", "event_id": "<one of the event_ids listed above>"}}

The timeline interleaves utterances and events in the order they happen. \
All event_ids listed above MUST appear EXACTLY ONCE somewhere in the \
timeline. The number of utterance items must be between {min_turns} and \
{max_turns}.

Example shape (illustrative, ignore content; note the optional prosody tag \
inside the text, and the event as a separate item):
{{
  "timeline": [
    {{"type": "utterance", "speaker": "A", "text": "So I finally tried it last week."}},
    {{"type": "event", "event_id": "cafe_restaurant/coffee_machine"}},
    {{"type": "utterance", "speaker": "B", "text": "[curious] Yeah? And how was it really?"}},
    {{"type": "utterance", "speaker": "A", "text": "[a little embarrassed] Honestly better than I expected."}}
  ]
}}"""


PIVOT_DIRECTIVE_TEMPLATE = """═══════════════════════════════════════════════════════════════════
PIVOT EVENT — {pivot_event_id}
═══════════════════════════════════════════════════════════════════
This is the single PIVOT of the scene. Handle it with care:
  • Spend the FIRST part of the dialogue (roughly the first third to half) on \
the subject as normal, treating any verbal / behavioral / ambient events \
naturally as they fall.
  • Then place the pivot event in the timeline.
  • From that point on, the conversation TURNS TO the sound: the speakers \
react to what they just heard and the talk is ABOUT it and its aftermath for \
AT LEAST {min_pivot_turns} of the remaining turns. The original subject \
recedes and clearly becomes secondary.
  • The reaction must be tied to the moment the sound occurs (the very next \
turn reacts), and the pivot must be obvious to a listener.
  • The speakers refer to the sound obliquely but NEVER name it by label \
("what was that?", "that sounded close", not "that was a siren").{rare_extra}

"""


RARE_PIVOT_EXTRA = """
  • This pivot is a RARE, DISRUPTIVE sound. The reaction is strong: startled, \
alarmed, worried, asking each other what just happened, maybe deciding to do \
something about it. The interruption is sharp and the pivot unmistakable."""


def _format_events_block(events: list[dict]) -> str:
    if not events:
        return ("    (none — there is no acoustic event in this scene; just "
                "write the conversation about the subject)")
    lines = []
    for event in events:
        tag = REACTION_TAG.get(event.get("reaction"), str(event.get("reaction")).upper())
        suffix = "  (rare, disruptive)" if event.get("is_rare") else ""
        slot = SLOT_WORDS.get(event.get("slot") or "")
        placement = f"   [PLACEMENT: {slot}]" if slot else ""
        lines.append(f"    - {event['event_id']}  →  {tag}{suffix}{placement}")
        reaction = event.get("reaction") or ""
        register = event.get("register") or ""
        move = register_move(reaction, register)
        if move:
            lines.append(f"        REGISTER — {register} : {move}")
            required = register_required_tags(reaction, register)
            if required:
                lines.append(
                    "        DELIVERY TAG REQUIRED — this move lives in the VOICE "
                    "and leaves NO trace in the words. The reacting line MUST "
                    f"carry exactly one of these tags: {', '.join(required)} — "
                    "spelled that way, nothing else. Put it INSIDE a phrase, "
                    "between a word and the one it belongs with, never after a "
                    "full stop, and never at either end of the line: that "
                    "position is what makes the break audible as a break.\n"
                    "        NOTHING IN THE WRITING MAY ANNOUNCE IT. Do NOT put "
                    "an ellipsis before the tag, and do NOT repeat the word "
                    "across it — the sentence resumes exactly where it stopped, "
                    "saying nothing twice. Read the line with the tag deleted: "
                    "it must be one clean, ordinary sentence that nobody would "
                    "think was interrupted. If a reader can tell that something "
                    "happened, this register has failed and the dialogue is "
                    "rejected. Without the tag the reaction does not exist in "
                    "the audio; with a written hint it does not exist as this "
                    "register.")
    return "\n".join(lines)


def _event_id_to_words(event_id: str) -> list[str]:
    name = event_id.split("/", 1)[1] if "/" in event_id else event_id
    return [p.lower() for p in re.split(r"[_\-\s]+", name) if len(p) >= 4]


def check_event_naming(timeline: list[dict], event_ids: list[str]) -> tuple[bool, str]:
    event_terms = {}
    for event_id in event_ids:
        terms = _event_id_to_words(event_id)
        if len(terms) >= 2:
            event_terms[event_id] = terms
    if not event_terms:
        return True, ""

    for i, item in enumerate(timeline):
        if item.get("type") != "utterance":
            continue
        text = strip_prosody_tags(item.get("text") or "").lower()
        for event_id, terms in event_terms.items():
            hits = sum(1 for t in terms if re.search(r"\b" + re.escape(t) + r"\b", text))
            if hits >= 2:
                return False, (f"utterance[{i}] names the event '{event_id}' "
                               f"(words detected: {terms})")
    return True, ""


SLOT_TOLERANCE = 0.10


def check_event_slots(timeline: list[dict], expected_events: list[dict]) -> tuple[bool, str]:
    from ...allocation import PLACEMENT_SLOTS

    slots = {e["event_id"]: e.get("slot") for e in expected_events if e.get("slot")}
    if not slots:
        return True, ""
    n_utt = sum(1 for it in timeline if it.get("type") == "utterance")
    if n_utt < len(PLACEMENT_SLOTS):
        return True, ""
    seen = 0
    for i, item in enumerate(timeline):
        if item.get("type") == "utterance":
            seen += 1
            continue
        if item.get("type") != "event":
            continue
        slot = slots.get(item.get("event_id"))
        if not slot:
            continue
        position = seen / n_utt
        band = PLACEMENT_SLOTS.index(slot)
        low = band / len(PLACEMENT_SLOTS) - SLOT_TOLERANCE
        high = (band + 1) / len(PLACEMENT_SLOTS) + SLOT_TOLERANCE
        if not (low <= position <= high):
            return False, (f"timeline[{i}]: '{item['event_id']}' should be "
                           f"'{slot}' (position {position:.2f}, expected "
                           f"{max(0.0, low):.2f}-{min(1.0, high):.2f})")
    return True, ""


_SENTENCE_END_BEFORE_TAG = re.compile(r"""[.!?]["'\u201d\u2019)\]]*\s*$""")

_NOT_A_SENTENCE_END = re.compile(
    r"""(?: \.\.                                      # ellipse "..."
          | \u2026                                    # ellipse "…"
          | \b(?:Mr|Mrs|Ms|Dr|Prof|St|Sgt|Lt|vs|etc)  # abbreviation "Dr."
          | \b[A-Z](?:\.[A-Z])*                      # acronym "U.S."
          | \d                                       # decimal "3.5"
        )\.["'\u201d\u2019)\]]*\s*$""", re.VERBOSE)


def _ends_a_sentence(fragment: str) -> bool:
    return bool(_SENTENCE_END_BEFORE_TAG.search(fragment)
                and not _NOT_A_SENTENCE_END.search(fragment))


_WORDS_RE = re.compile(r"[a-z']+")


def check_required_tags(timeline: list[dict],
                        expected_events: list[dict]) -> tuple[bool, str]:
    from ...allocation import register_required_tags

    need = {}
    for e in expected_events:
        tags = register_required_tags(e.get("reaction") or "", e.get("register") or "")
        if tags:
            need[e["event_id"]] = tags
    if not need:
        return True, ""

    for i, item in enumerate(timeline):
        if item.get("type") != "event" or item.get("event_id") not in need:
            continue
        tags = need[item["event_id"]]
        nxt = next((t for t in timeline[i + 1:] if t.get("type") == "utterance"), None)
        text = (nxt or {}).get("text") or ""
        low = text.lower()
        found = sorted((low.index(t.lower()), t) for t in tags if t.lower() in low)
        if not found:
            return False, (f"'{item['event_id']}': delivery register, the reacting "
                           f"turn must carry one of {', '.join(tags)}, "
                           "INSIDE a sentence")
        cut, hit = found[0]
        before, after = text[:cut], text[cut + len(hit):]
        if not strip_prosody_tags(before).strip():
            return False, (f"'{item['event_id']}': {hit} opens the turn; it must "
                           "fall INSIDE a sentence, between two words")
        if not strip_prosody_tags(after).strip():
            return False, (f"'{item['event_id']}': {hit} ends the turn; it must "
                           "fall INSIDE a sentence, between two words")
        if _ends_a_sentence(before):
            return False, (f"'{item['event_id']}': {hit} sits after a sentence "
                           "end; it must fall INSIDE a sentence")
        clean_before = strip_prosody_tags(before).rstrip()
        if clean_before.endswith(("...", "\u2026")):
            return False, (f"'{item['event_id']}': {hit} follows an ellipsis; the "
                           "break must exist IN THE VOICE ALONE, so nothing may "
                           "announce it in the text")
        head = _WORDS_RE.findall(clean_before.lower())
        tail = _WORDS_RE.findall(strip_prosody_tags(after).lower())
        if head and tail and head[-1] == tail[0]:
            return False, (f"'{item['event_id']}': the word '{tail[0]}' is repeated "
                           f"on both sides of {hit}; the sentence must RESUME where "
                           "it stopped, saying nothing twice")
    return True, ""


def validate_timeline(parsed: dict, expected_events: list[dict],
                      min_turns: int = MIN_TURNS,
                      max_turns: int = MAX_TURNS) -> tuple[bool, str]:
    if not isinstance(parsed, dict):
        return False, "root is not a dict"

    expected_ids = [e["event_id"] for e in expected_events]
    pivot_id = next((e["event_id"] for e in expected_events
                     if e.get("reaction") == "pivot"), None)

    def check_event(index: int, event_id: Any) -> str | None:
        if not isinstance(event_id, str) or not event_id:
            return f"timeline[{index}]: missing event_id"
        if event_id not in expected_ids:
            return (f"timeline[{index}]: unexpected event_id '{event_id}' "
                    f"(expected: {expected_ids})")
        return None

    timeline = parsed.get("timeline")
    reason, stats = scan_timeline(timeline, check_event)
    if reason:
        return False, reason

    reason = check_turn_counts(stats, min_turns,
                               max_turns_for(expected_events, max_turns))
    if reason:
        return False, reason

    seen = [event_id for _, event_id in stats.events]
    if set(expected_ids) != set(seen):
        missing = set(expected_ids) - set(seen)
        extra = set(seen) - set(expected_ids)
        return False, (f"inconsistent events: missing={list(missing)}, "
                       f"extra={list(extra)}")
    if len(seen) != len(expected_ids):
        return False, (f"duplicated events: got {len(seen)} placements "
                       f"for {len(expected_ids)} expected")

    if pivot_id is not None:
        pivot_pos = next((i for i, event_id in stats.events if event_id == pivot_id), None)
        if pivot_pos is None:
            return False, f"pivot '{pivot_id}' missing from the timeline"
        turns_after = sum(1 for item in timeline[pivot_pos + 1:]
                          if item.get("type") == "utterance")
        if turns_after < MIN_PIVOT_REACTION_TURNS:
            return False, (f"pivot '{pivot_id}': only {turns_after} turns after "
                           f"it (min {MIN_PIVOT_REACTION_TURNS} for the "
                           f"conversation to turn to it)")

    ok, reason = check_event_slots(timeline, expected_events)
    if not ok:
        return False, reason

    ok, reason = check_event_naming(timeline, expected_ids)
    if not ok:
        return False, reason

    ok, reason = check_required_tags(timeline, expected_events)
    if not ok:
        return False, reason

    if stats.speakers != {"A", "B"}:
        return False, f"missing speakers: present={stats.speakers}"

    return True, ""


class GroundedVariant:
    name = "grounded"
    mode = "grounded"
    system_prompt = SYSTEM_PROMPT
    max_turns = MAX_TURNS

    def build_prompt(self, scenario: dict) -> tuple[str, list[dict]]:
        n = normalize_scenario(scenario)

        missing = [k for k in ("scene", "role_a", "role_b", "subject") if not n.get(k)]
        if missing:
            raise ValueError(
                f"scenario '{n.get('scenario_id')}': missing field(s) {missing}. "
                f"Expected: scene, role_a/role_b (or speakers{{A,B}}), and subject.")

        events = n["events"]
        pivots = [e for e in events if e.get("reaction") == "pivot"]
        if len(pivots) > 1:
            raise ValueError(f"scenario '{n['scenario_id']}': {len(pivots)} pivots "
                             f"(at most 1 expected).")

        pivot_directive = ""
        if pivots:
            pivot_directive = PIVOT_DIRECTIVE_TEMPLATE.format(
                pivot_event_id=pivots[0]["event_id"],
                min_pivot_turns=MIN_PIVOT_REACTION_TURNS,
                rare_extra=(RARE_PIVOT_EXTRA if pivots[0]["is_rare"] else ""),
            )

        user = USER_PROMPT.format(
            theme_title=n["theme_title"],
            theme_description=n["theme_description"],
            scene=n["scene"],
            role_a=n["role_a"],
            role_b=n["role_b"],
            gender_a=GENDER_WORDS.get(n.get("gender_a"), "unspecified"),
            gender_b=GENDER_WORDS.get(n.get("gender_b"), "unspecified"),
            subject=n["subject"],
            events_block=_format_events_block(events),
            pivot_directive=pivot_directive,
            min_turns=MIN_TURNS,
            max_turns=max_turns_for(events, self.max_turns),
            min_pivot_turns=MIN_PIVOT_REACTION_TURNS,
            weak_tag_words=", ".join(WEAK_TAG_WORDS),
            manner_tag_words=", ".join(MANNER_TAG_WORDS),
            vocal_tag_spellings="  ".join(VOCAL_TAG_SPELLINGS),
            worn_phrasings="; ".join(f'"{ph}"' for ph, _ in WORN_PHRASINGS),
        )
        return user, events

    def validate(self, parsed: Any, context: list[dict]) -> tuple[bool, str]:
        return validate_timeline(parsed, context, MIN_TURNS, self.max_turns)

    def final_record(self, scenario: dict, timeline: list[dict]) -> dict:
        n = normalize_scenario(scenario)
        timeline = strengthen_prosody_tags(repair_item_types(timeline))
        turns = utterances(timeline)
        reactions = Counter(e["reaction"] for e in n["events"])
        record = {
            "scenario_id": n["scenario_id"],
            "template_id": n["template_id"],
            "category": n["category"],
            "theme": {"title": n["theme_title"], "description": n["theme_description"]},
            "theme_label": n["theme_label"],
            "scene": n["scene"],
            "speakers": {"A": n["role_a"], "B": n["role_b"]},
            "subject": n["subject"],
            "events": n["events"],
            "has_rare_event": n["has_rare_event"],
            "timeline": timeline,
            "metadata": {
                "scenario_idx_in_template": n["scenario_idx_in_template"],
                "n_events": len(n["events"]),
                "n_pivot": reactions["pivot"],
                "n_verbal": reactions["verbal"],
                "n_behavioral": reactions["behavioral"],
                "n_ambient": reactions["ambient"],
                "n_reacted_to": reactions["pivot"] + reactions["verbal"]
                + reactions["behavioral"],
                "n_turns_total": len(turns),
                "n_turns_A": sum(1 for u in turns if u["speaker"] == "A"),
                "n_turns_B": sum(1 for u in turns if u["speaker"] == "B"),
                "n_events_in_timeline": sum(1 for it in timeline if it["type"] == "event"),
            },
        }
        if scenario.get("split"):
            record["split"] = scenario["split"]
        for key in ("gender_a", "gender_b"):
            if scenario.get(key):
                record[key] = scenario[key]
        return record


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)


def run(args: argparse.Namespace) -> int:
    variant = GroundedVariant()
    variant.max_turns = getattr(args, "max_turns", MAX_TURNS)
    return run_variant(variant, args)
