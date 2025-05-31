# ----------------------------------------------------------------
# fill_state_and_action.py  –  carry‑over state + heuristic action
# ----------------------------------------------------------------
"""Usage
python fill_state_and_action.py --input step2.json --output step3.json
"""
import json, argparse, pathlib, re
from copy import deepcopy
from typing import Dict, Any, List

REQ_PAT = re.compile(r"\b(can you|could you|i need|i want|please|looking for)\b", re.I)
INF_PAT = re.compile(r"\b(is|are|it is|here is|here are|the)\b", re.I)
SLOT_HINT = re.compile(r"\b(area|price|food|people|phone|number|reference|ref|book|time|date)\b", re.I)


def regex_extract(u: str) -> Dict[str, str]:
    # naive pattern extractor, extend as needed
    out = {}
    if m := re.search(r"(north|south|east|west|centre|center)", u, re.I):
        out["area"] = {"value": m.group(1).lower(), "slot_type": "string"}
    if m := re.search(r"\b(cheap|inexpensive|moderate|expensive)\b", u, re.I):
        out["price"] = {"value": m.group(1).lower(), "slot_type": "string"}
    if m := re.search(r"\b(\d{11,})\b", u):
        out["phone"] = {"value": "<phone_any>", "slot_type": "string"}
    if m := re.search(r"\b[A-Z0-9]{6,}\b", u):
        out["booking_ref"] = {"value": "<ref_any>", "slot_type": "string"}
    if m := re.search(r"\b(\d{1,2})\s*people\b", u):
        out["people"] = {"value": m.group(1), "slot_type": "numeric"}
    return out


def add_action(turn: Dict[str, Any]):
    utt = turn["utterance"].lower()
    if turn.get("action"):
        return
    if REQ_PAT.search(utt):
        turn["action"] = [{"type": "request", "slot": "", "value": ""}]
    elif INF_PAT.search(utt) and SLOT_HINT.search(utt):
        turn["action"] = [{"type": "inform", "slot": "", "value": ""}]
    else:
        turn["action"] = []


def process_dialogue(dlg: Dict[str, Any]):
    running = {}
    turns_sorted = sorted(dlg["annotation"].items(), key=lambda kv: int(kv[1]["turn_idx"]))
    for _, turn in turns_sorted:
        turn["state_before"] = deepcopy(running)
        explicit = regex_extract(turn["utterance"])
        running.update(explicit)
        turn["state_after"] = deepcopy(running)
        add_action(turn)


def main(inp: pathlib.Path, out: pathlib.Path):
    data = json.load(inp.open())
    for dlg in data:
        process_dialogue(dlg)
    json.dump(data, out.open("w"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    main(pathlib.Path(args.input), pathlib.Path(args.output))

