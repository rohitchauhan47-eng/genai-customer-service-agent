"""AI Hyper-personalisation Agent: picks the right remedy for a customer case, a person approves it.

1. Rules in code: add what the customer is owed (refund of a confirmed billing error, outage credit).
2. Rules in code: list the extra offers that suit this customer and fit the goodwill budget for their segment.
3. Value in code: for each extra offer, value = yearly spend x churn-risk cut - cost. Only offers worth more
   than they cost are kept, best first.
4. LLM: picks one of those offers (it cannot invent one), explains why, citing facts [F..] and offers [O..],
   and drafts the message to the customer.
5. Human in the loop: a supervisor approves, edits or rejects before anything reaches the customer.
If no API key is set, or the LLM call fails, the best-value offer is used with a built-in message.
"""

import json
import re
from pathlib import Path

from agent import GEMINI_MODEL, GROQ_MODEL, ask_gemini, ask_groq, error_text, money

DATA = Path(__file__).parent / "data"
CASES = json.loads((DATA / "cases.json").read_text(encoding="utf-8"))["cases"]
CATALOGUE = json.loads((DATA / "offers.json").read_text(encoding="utf-8"))
OFFERS = {o["id"]: o for o in CATALOGUE["offers"]}


def profile(case):
    return {f["key"]: f["value"] for f in case["facts"]}


def fact_id(case, key):
    return next(f["id"] for f in case["facts"] if f["key"] == key)


def offer_cost(offer, p):
    c = offer["cost"]
    if c == "billing_error":
        return round(p.get("billing_error", 0), 2)
    if c == "outage":
        return round(p["arpu"] / 30 * p.get("outage_days", 0), 2)
    if c == "loyalty":
        return round(p["arpu"] * 0.10 * 12, 2)
    return c


def fits(offer, p):
    for key, need in offer["fit"].items():
        have = p.get(key, 0)
        if isinstance(need, str):
            if have != need:
                return False
        elif have < need:
            return False
    return True


def plan(case):
    """Everything worked out in code: what is owed, which extras are allowed, and what each is worth."""
    p = profile(case)
    owed = [{"offer": OFFERS[oid], "cost": offer_cost(OFFERS[oid], p)} for oid in ("O1", "O2")
            if offer_cost(OFFERS[oid], p) > 0]
    budget = CATALOGUE["goodwill_budget"][p["segment"]]
    yearly = p["arpu"] * 12
    options, ruled_out = [], []
    for o in CATALOGUE["offers"]:
        if o["kind"] != "extra":
            continue
        cost = offer_cost(o, p)
        value = round(yearly * o["lift"] - cost, 2)
        row = {"offer": o, "cost": cost, "value": value}
        if not fits(o, p):
            ruled_out.append({**row, "why": "does not suit this customer"})
        elif cost > budget:
            ruled_out.append({**row, "why": f"costs more than the {money(budget)} goodwill budget"})
        elif value <= 0:
            ruled_out.append({**row, "why": "costs more than it is worth"})
        else:
            options.append(row)
    options.sort(key=lambda r: -r["value"])
    return {"profile": p, "owed": owed, "options": options, "ruled_out": ruled_out, "budget": budget}


SYSTEM = (
    "You are the AI Hyper-personalisation Agent of a mobile operator. A complaint case has come in. "
    "Pick exactly ONE extra offer from ALLOWED OFFERS (or none if the list is empty) that best suits this customer, "
    "on top of what they are OWED. Use only the FACTS and offers given; cite them like [F4] or [O3]. "
    "Reply with JSON only: {\"offer\": \"O3\" or null, \"reason\": \"2-3 sentences for the supervisor, with citations\", "
    "\"message\": \"a warm message to the customer, under 80 words, no citations, that apologises if something went "
    "wrong, states what they get with exact amounts, and signs off as 'Your TelcoDemo team'\"}"
)


def build_prompt(case, pl):
    facts = "\n".join(f"[{f['id']}] {f['text']}" for f in case["facts"])
    owed = "\n".join(f"[{r['offer']['id']}] {r['offer']['name']}: {money(r['cost'])}" for r in pl["owed"]) or "nothing"
    allowed = "\n".join(f"[{r['offer']['id']}] {r['offer']['name']} (cost {money(r['cost'])}, "
                        f"estimated value {money(r['value'])}; {r['offer']['rule']})" for r in pl["options"]) or "none"
    return (f"CASE {case['case']}: {case['summary']}\n\nFACTS\n{facts}\n\nOWED (always given, decided by rules)\n{owed}\n\n"
            f"ALLOWED OFFERS (passed the rules and the {money(pl['budget'])} goodwill budget, best value first)\n{allowed}")


def built_in(case, pl, pick):
    p = pl["profile"]
    reason = []
    for r in pl["owed"]:
        reason.append(f"{r['offer']['name']} ({money(r['cost'])}) is owed under the rules [{r['offer']['id']}].")
    if pick:
        reason.append(f"{pick['offer']['name']} has the best value of the allowed offers ({money(pick['value'])}) "
                      f"[{pick['offer']['id']}], with churn risk at {int(p['churn_risk'] * 100)}% [{fact_id(case, 'churn_risk')}].")
    items = [r["offer"]["say"].format(cost=money(r["cost"])) for r in pl["owed"] + ([pick] if pick else [])]
    sorry = "we're sorry for the trouble" if pl["owed"] else "thanks for getting in touch"
    message = (f"Hi, {sorry}. We've added " + " and ".join(items) + " to your account. "
               "Thank you for being with us. Your TelcoDemo team")
    return " ".join(reason), message


def recommend(case, groq_key=None, gemini_key=None):
    """Returns dict: plan, pick (option row or None), reason, message, engine, prompt."""
    pl = plan(case)
    prompt = build_prompt(case, pl)
    best = pl["options"][0] if pl["options"] else None
    last_error = None
    for name, key, fn in (("Groq " + GROQ_MODEL, groq_key, ask_groq),
                          ("Gemini " + GEMINI_MODEL, gemini_key, ask_gemini)):
        if not key:
            continue
        try:
            raw = fn(key, SYSTEM, [], prompt)
            out = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            allowed = {r["offer"]["id"]: r for r in pl["options"]}
            if out.get("offer") and out["offer"] not in allowed:
                raise ValueError("picked an offer the rules do not allow")  # guardrail
            pick = allowed.get(out.get("offer"))
            return {"plan": pl, "pick": pick, "reason": out["reason"], "message": out["message"],
                    "engine": name, "prompt": prompt}
        except Exception as e:
            last_error = f"{name} unavailable ({error_text(e)})"
    reason, message = built_in(case, pl, best)
    return {"plan": pl, "pick": best, "reason": reason, "message": message,
            "engine": f"Built-in rules ({last_error or 'no AI key set'})", "prompt": prompt}
