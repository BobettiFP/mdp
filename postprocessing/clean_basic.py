# =============================================================
# ontology_free_postprocess_pipeline
# -------------------------------------------------------------
#  Four standalone Python modules + helper shell snippet that turn
#  raw ontology‑free LLM annotations into a cleaned, compact MDP
#  dataset.  Each module streams JSON from <input> to <output> so
#  they can be chained easily.
# =============================================================

# ----------------------------------------------------------------
# clean_basic.py  –  surface cleanup + value wrapping
# ----------------------------------------------------------------
"""Usage
python clean_basic.py --input raw.json --output step1.json
"""

import re, json, argparse, sys, pathlib
from typing import Dict, Any, List

PHONE_PAT  = re.compile(r"\d{11,}")
REF_PAT    = re.compile(r"(?=.*[A-Za-z])[A-Za-z0-9]{6,}")
DATE_PAT   = re.compile(r"\d{4}-\d{2}-\d{2}")
NUMBER_PAT = re.compile(r"^\d+(\.\d+)?$")


def snake_case(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)  # spaces, camel hump splitter later
    # camelCase → snake_case
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return re.sub(r"_+", "_", s).lower().strip("_")


def wrap_value(val: str) -> Dict[str, str]:
    orig = val
    if PHONE_PAT.fullmatch(val):
        val = "<phone_any>"
        _type = "string"
    elif REF_PAT.fullmatch(val):
        val = "<ref_any>"
        _type = "string"
    elif DATE_PAT.fullmatch(val):
        _type = "date_iso"
    elif NUMBER_PAT.fullmatch(val):
        _type = "numeric"
    else:
        _type = "string"
    return {"value": val, "slot_type": _type}


def process_turn(turn: Dict[str, Any]):
    for field in ("state_before", "state_after"):
        state = turn.get(field, {}) or {}
        cleaned = {}
        for k, v in state.items():
            k2 = snake_case(k)
            if isinstance(v, dict) and "value" in v:
                cleaned[k2] = v  # already wrapped
            else:
                cleaned[k2] = wrap_value(str(v))
        turn[field] = cleaned

    # ensure reward/transition exist even if missing
    turn.setdefault("reward", {"score": 0, "justification": "auto-added"})
    turn.setdefault("transition", {"slots_added": {}, "slots_removed": [],
                                     "goal_progress": "unchanged",
                                     "is_valid_transition": True,
                                     "justification": "auto-added"})


def main(inp: pathlib.Path, out: pathlib.Path):
    data = json.load(inp.open())
    for dlg in data:
        for t in dlg["annotation"].values():
            process_turn(t)
    json.dump(data, out.open("w"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    main(pathlib.Path(args.input), pathlib.Path(args.output))

