"""GenAI Customer Service Agent for a mobile operator.

1. Route: work out what the customer needs (bill issue, recharge, internet/WiFi, other).
2. Bill issue (live):
   a. Compare this bill with last month's and with card payments, in plain code (exact numbers,
      and a double-charge check).
   b. Retrieve the price guide sections that explain the changed lines, plus any that match the question (RAG).
   c. Ask a free LLM to answer using only those sources, quoting their ids, e.g. [B3] [T4].
   d. If it finds a double charge, log a complaint ticket and hand it to the Master AI Agent (simulated).
3. Recharge and internet/WiFi are the next use cases: the agent recognises them and says so.
If no API key is set, or the LLM call fails, a built-in explainer writes the answer from the same sources.
"""

import json
import math
import re
from pathlib import Path

import requests

DATA = Path(__file__).parent / "data"
TARIFF = json.loads((DATA / "tariff.json").read_text(encoding="utf-8"))
CUSTOMERS = json.loads((DATA / "bills.json").read_text(encoding="utf-8"))["customers"]
SECTIONS = {s["id"]: s for s in TARIFF["sections"]}

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.5-flash"

INTENTS = {
    "bill": ("Bill issue", "Live"),
    "recharge": ("Recharge / top-up", "Next"),
    "internet": ("Internet or WiFi problem", "Next"),
    "other": ("Other question", "Next"),
}
_INTENT_WORDS = {
    "bill": r"\bbill|\bcharge|\bprice|\bcost|\bpay\b|\bpaid\b|refund|expensive|\bhigh|roaming|discount|wrong|overcharg|invoice|\$",
    "recharge": r"recharge|top.?up|topup|credit|voucher|balance|prepaid",
    "internet": r"internet|wifi|wi-fi|broadband|router|slow|signal|no data|connection|network|outage|4g|5g",
}


def money(x):
    return f"-${abs(x):.2f}" if x < 0 else f"${x:.2f}"


def total(bill):
    return round(sum(line["amount"] for line in bill["lines"]), 2)


def detect_intent(question):
    """Keyword router: free, instant and predictable. The most matched intent wins."""
    q = question.lower()
    hits = {k: len(re.findall(p, q)) for k, p in _INTENT_WORDS.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] else "other"


def _key(item):
    # Lines are matched across months by their tariff section and first two words.
    return " ".join(re.findall(r"[a-z0-9]+", item.lower())[:2])


def compare(customer):
    """Line-by-line difference between last month's bill and this month's, plus a double-charge check."""
    prev, cur = customer["bills"]["previous"], customer["bills"]["current"]
    prev_by_key = {(_key(l["item"]), l["tariff"]): l for l in prev["lines"]}
    changes, seen = [], set()
    for line in cur["lines"]:
        k = (_key(line["item"]), line["tariff"])
        old = prev_by_key.get(k)
        seen.add(k)
        before = old["amount"] if old else 0.0
        if round(line["amount"] - before, 2) != 0:
            changes.append({"line": line, "before": before, "change": round(line["amount"] - before, 2),
                            "status": "changed" if old else "new"})
    for k, old in prev_by_key.items():
        if k not in seen:
            changes.append({"line": old, "before": old["amount"], "change": round(-old["amount"], 2),
                            "status": "gone"})
    changes.sort(key=lambda c: -c["change"])
    # A bill line that matches a card payment already made (same first words, same amount) is a double charge.
    doubles = [(line, pay) for line in cur["lines"] for pay in customer.get("payments", [])
               if line["item"].lower().startswith(pay["item"].lower()) and line["amount"] == pay["amount"]]
    return {"previous_total": total(prev), "current_total": total(cur),
            "difference": round(total(cur) - total(prev), 2), "changes": changes, "doubles": doubles}


# ---------- Retrieval: small BM25 keyword search over the price guide ----------

def _words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


# Title words count three times, so "roaming" finds the Roaming section first.
_DOCS = {sid: _words((s["title"] + " ") * 3 + s["text"]) for sid, s in SECTIONS.items()}
_AVG = sum(len(d) for d in _DOCS.values()) / len(_DOCS)
_STOP = {"my", "is", "the", "a", "why", "what", "i", "me", "this", "bill", "so", "do", "how", "it", "to", "of"}


def search_tariff(query, k=2):
    q = set(_words(query)) - _STOP
    scores = {}
    for sid, doc in _DOCS.items():
        s = 0.0
        for w in q:
            tf = doc.count(w)
            if tf:
                df = sum(1 for d in _DOCS.values() if w in d)
                idf = math.log(1 + (len(_DOCS) - df + 0.5) / (df + 0.5))
                s += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(doc) / _AVG))
        if s > 0:
            scores[sid] = s
    return sorted(scores, key=scores.get, reverse=True)[:k]


def gather_sources(customer, question):
    diff = compare(customer)
    ids = []
    for c in diff["changes"]:
        ids.append(c["line"]["tariff"])
    if diff["doubles"]:
        ids += ["T12", "T11"]
    ids += search_tariff(question)
    if re.search(r"wrong|dispute|refund|complain|mistake|twice", question, re.I):
        ids.append("T11")
    ids.append("T9")  # always offer the spend-cap tip
    return diff, [SECTIONS[sid] for sid in dict.fromkeys(ids)]


# ---------- Answer writing ----------

SYSTEM = (
    "You are the GenAI Customer Service Agent of a mobile operator, helping with a bill question. "
    "Answer ONLY from the SOURCES given. Every fact must cite its source id in square brackets, "
    "e.g. [B3] for a bill line, [PAY1] for a payment or [T4] for a price guide section. "
    "Use the exact amounts from the sources and never invent charges, prices or rules. "
    "If a DOUBLE CHARGE is listed, say so clearly, apologise, and say complaint ticket {ticket} has been logged "
    "and passed to the Master AI Agent to confirm and arrange the refund. "
    "If the sources do not answer the question, say so and offer to pass it to a person. "
    "Structure: one sentence with the total change, then a short bullet per reason (biggest first), "
    "then one 'How to avoid this next time' tip from the price guide (skip it for a double charge). "
    "Keep it under 170 words. Be warm and plain."
)


def ticket_id(customer):
    return f"CS-{customer['id']}-001"


def build_prompt(customer, question, diff, sections):
    cur, prev = customer["bills"]["current"], customer["bills"]["previous"]
    lines = "\n".join(f"[{l['id']}] {l['item']}: {money(l['amount'])}" for l in cur["lines"])
    changes = "\n".join(
        f"- [{c['line']['id']}] {c['line']['item']}: {money(c['before'])} -> "
        f"{money(0 if c['status'] == 'gone' else c['line']['amount'])} ({c['status']}, change {money(c['change'])})"
        for c in diff["changes"])
    pays = "\n".join(f"[{p['id']}] {p['date']}: {p['item']}, {money(p['amount'])}, {p['method']}"
                     for p in customer.get("payments", [])) or "none"
    doubles = "\n".join(f"- [{l['id']}] {l['item']} {money(l['amount'])} was already paid as [{p['id']}]"
                        for l, p in diff["doubles"]) or "none found"
    tariff = "\n".join(f"[{s['id']}] {s['title']}: {s['text']}" for s in sections)
    return (f"CUSTOMER: {customer['name']} ({customer['segment']}), plan {customer['plan']}\n\n"
            f"SOURCES\nThis bill ({cur['period']}), total {money(diff['current_total'])}:\n{lines}\n\n"
            f"Last bill ({prev['period']}) total {money(diff['previous_total'])}. "
            f"Difference {money(diff['difference'])}. Changed lines (computed exactly):\n{changes or '- none'}\n\n"
            f"Card payments already made:\n{pays}\n\nDOUBLE CHARGE check (computed exactly):\n{doubles}\n\n"
            f"Price guide sections:\n{tariff}\n\nQUESTION: {question}")


def ask_groq(key, system, history, prompt):
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": GROQ_MODEL, "temperature": 0.1,
                            "messages": [{"role": "system", "content": system}, *history,
                                         {"role": "user", "content": prompt}]},
                      timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ask_gemini(key, system, history, prompt):
    turns = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
             for m in history]
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                      headers={"x-goog-api-key": key},
                      json={"systemInstruction": {"parts": [{"text": system}]},
                            "contents": [*turns, {"role": "user", "parts": [{"text": prompt}]}],
                            "generationConfig": {"temperature": 0.1}},
                      timeout=30)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def best_sentence(section, item):
    """The price guide sentence that shares the most words with the bill line."""
    # Numbers (dates, minutes) are left out: they match the wrong sentences too easily.
    item_words = {w for w in _words(item) if not w.isdigit()} - _STOP - {"and", "for", "by", "you", "your", "in", "on"}
    sentences = re.split(r"(?<=\.) ", section["text"])
    return max(sentences, key=lambda t: len(item_words & set(_words(t))))


def built_in_answer(customer, diff):
    """No-AI fallback that writes the answer from the same sources."""
    d = diff["difference"]
    if not diff["changes"]:
        return f"Your bill is {money(diff['current_total'])}, the same as last month. Nothing has changed."
    out = [f"Your bill is **{money(diff['current_total'])}**, which is **{money(abs(d))} "
           f"{'more' if d > 0 else 'less'}** than last month ({money(diff['previous_total'])}). Here is why:"]
    for line, pay in diff["doubles"]:
        out.append(f"- **This looks like a double charge.** {line['item']}, {money(line['amount'])} [{line['id']}] "
                   f"was already paid by card on {pay['date']} [{pay['id']}]. "
                   f"{SECTIONS['T12']['text']} [T12]")
    doubled = {line["id"] for line, _ in diff["doubles"]}
    for c in diff["changes"]:
        line, sec = c["line"], SECTIONS[c["line"]["tariff"]]
        if line["id"] in doubled:
            continue
        if c["status"] == "gone":
            what = f"{line['item']} ({money(line['amount'])}) is no longer on your bill [{line['id']}]"
        elif c["status"] == "new":
            what = f"New charge: {line['item']}, {money(line['amount'])} [{line['id']}]"
        else:
            what = f"{line['item']} changed by {money(c['change'])} [{line['id']}]"
        out.append(f"- {what}. {best_sentence(sec, line['item'])} [{sec['id']}]")
    if diff["doubles"]:
        out.append(f"\nI'm sorry about this. I've logged complaint ticket **{ticket_id(customer)}** and passed it "
                   f"to the Master AI Agent to confirm and arrange your refund. [T11]")
    else:
        out.append(f"\n**How to avoid this next time:** {best_sentence(SECTIONS['T9'], 'set a monthly spend cap')} [T9]")
    return "\n".join(out)


def not_yet(intent):
    label = INTENTS[intent][0]
    return (f"I can see this is about **{label.lower()}**. In this demo I can only help with **bill issues** so far. "
            f"{label} help is the next use case being added. Try asking me about your bill, for example "
            f"*\"Why is my bill higher this month?\"*")


def respond(customer, question, history=(), groq_key=None, gemini_key=None):
    """Returns a dict: answer, intent, engine, and (for bill issues) diff, sections, prompt, ticket."""
    intent = detect_intent(question)
    result = {"intent": intent, "engine": "Router (no AI needed)", "ticket": None}
    if intent != "bill":
        result["answer"] = not_yet(intent)
        return result
    diff, sections = gather_sources(customer, question)
    prompt = build_prompt(customer, question, diff, sections)
    system = SYSTEM.format(ticket=ticket_id(customer))
    result.update(diff=diff, sections=sections, prompt=prompt,
                  ticket=ticket_id(customer) if diff["doubles"] else None)
    last_error = None
    for name, key, fn in (("Groq " + GROQ_MODEL, groq_key, ask_groq),
                          ("Gemini " + GEMINI_MODEL, gemini_key, ask_gemini)):
        if key:
            try:
                result.update(answer=fn(key, system, list(history)[-6:], prompt), engine=name)
                return result
            except Exception as e:  # rate limit, network, bad key: fall through to the next option
                last_error = f"{name} unavailable ({type(e).__name__})"
    result.update(answer=built_in_answer(customer, diff),
                  engine=f"Built-in explainer ({last_error or 'no AI key set'})")
    return result
