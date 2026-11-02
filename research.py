"""Telco AI Research Agent: answers questions about telco AI using only the industry reports Rohit has shared.

1. Retrieve: a BM25 keyword search picks the fact notes (data/reports.json) that best match the question (RAG).
2. Answer: a free LLM writes a short answer from those notes only, citing each one, e.g. [F21].
3. Each [F..] is shown with its report and page, so every claim can be checked.
If nothing matches, the agent says the reports don't cover it. With no AI key, it lists the best matching notes.
"""

import json
import math
import re
from pathlib import Path

from agent import GEMINI_MODEL, GROQ_MODEL, ask_gemini, ask_groq, error_text

LIBRARY = json.loads((Path(__file__).parent / "data" / "reports.json").read_text(encoding="utf-8"))
REPORTS = LIBRARY["reports"]
FACTS = {f["id"]: f for f in LIBRARY["facts"]}

_STOP = set("""a an and are as at be by can do does for from has have how i in is it its me my of on or our
should so that the their them they this to us was we what when which who why will
with you your about into more than ai agent agents telco telcos telecom telecoms""".split())


def _stem(w):
    # Very small stemmer so "results", "reported" and "reporting" match "result" and "report".
    for end in ("ing", "ed", "es", "s"):
        if len(w) > len(end) + 3 and w.endswith(end):
            return w[: -len(end)]
    return w


def _words(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


# Topic and tag words count three times, so a question about "security" finds the security notes first.
_DOCS = {fid: _words((f["topic"] + " " + f.get("tags", "") + " ") * 3 + f["text"]) for fid, f in FACTS.items()}
_AVG = sum(len(d) for d in _DOCS.values()) / len(_DOCS)


def search(question, k=6):
    q = set(_words(question))
    scores = {}
    for fid, doc in _DOCS.items():
        s = 0.0
        for w in q:
            tf = doc.count(w)
            if tf:
                df = sum(1 for d in _DOCS.values() if w in d)
                idf = math.log(1 + (len(_DOCS) - df + 0.5) / (df + 0.5))
                s += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(doc) / _AVG))
        if s > 0:
            scores[fid] = s
    return [FACTS[fid] for fid in sorted(scores, key=scores.get, reverse=True)[:k]]


def source(fact):
    r = REPORTS[fact["report"]]
    return f"{r['publisher']}, {r['title']} ({r['year']}), p.{fact['page']}"


SYSTEM = (
    "You are the Telco AI Research Agent. You answer questions about AI in the telecom industry using ONLY "
    "the NOTES given, which come from published industry reports. Cite every fact with its note id in square "
    "brackets, e.g. [F21]. Use the exact numbers and company names from the notes and never add facts, numbers "
    "or examples that are not in them. If the notes do not answer the question, say plainly that the shared "
    "reports don't cover it. Start with a one-sentence direct answer, then up to four short bullets. "
    "Keep it under 150 words, in plain English."
)


def build_prompt(question, notes):
    lines = "\n".join(f"[{f['id']}] ({source(f)}) {f['topic']}: {f['text']}" for f in notes)
    return f"NOTES\n{lines}\n\nQUESTION: {question}"


def built_in_answer(notes):
    return "Here is what the shared reports say:\n\n" + "\n".join(f"- {f['text']} [{f['id']}]" for f in notes[:4])


def cited(answer, notes):
    """The notes the answer actually cites, in order; falls back to all retrieved notes."""
    ids = list(dict.fromkeys(re.findall(r"\[(F\d+)\]", answer)))
    return [FACTS[i] for i in ids if i in FACTS] or notes


def respond(question, history=(), groq_key=None, gemini_key=None):
    notes = search(question)
    if not notes:
        return {"answer": "The reports shared so far don't cover that. Try asking about customer service, "
                          "network operations, security, adoption numbers or skills.",
                "engine": "Search (no AI needed)", "notes": [], "cited": [], "prompt": None}
    prompt = build_prompt(question, notes)
    last_error = None
    for name, key, fn in (("Groq " + GROQ_MODEL, groq_key, ask_groq),
                          ("Gemini " + GEMINI_MODEL, gemini_key, ask_gemini)):
        if key:
            try:
                answer = fn(key, SYSTEM, list(history)[-4:], prompt)
                return {"answer": answer, "engine": name, "notes": notes, "cited": cited(answer, notes),
                        "prompt": prompt}
            except Exception as e:  # rate limit, network, bad key: fall through to the next option
                last_error = f"{name} unavailable ({error_text(e)})"
    answer = built_in_answer(notes)
    return {"answer": answer, "engine": f"Built-in search ({last_error or 'no AI key set'})", "notes": notes,
            "cited": cited(answer, notes), "prompt": prompt}
