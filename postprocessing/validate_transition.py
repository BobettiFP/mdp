# ----------------------------------------------------------------
# validate_transition.py  –  recompute diff & flag invalid
# ----------------------------------------------------------------
"""Usage
python validate_transition.py --input step3.json --output final.json
"""
import json, argparse, pathlib
from typing import Dict, Any


def diff(prev: Dict[str, Any], after: Dict[str, Any]):
    added = {k: v for k, v in after.items() if prev.get(k) != v}
    removed = [k for k in prev if k not in after]
    return added, removed


def validate_turn(turn: Dict[str, Any]):
    added, removed = diff(turn["state_before"], turn["state_after"])
    trans = turn["transition"]
    # compare only keys
    rec_added  = trans.get("slots_added", {})
    rec_removed = trans.get("slots_removed", [])
    ok = added == rec_added and sorted(removed) == sorted(rec_removed)
    trans["is_valid_transition"] = ok
    if not ok:
        trans["justification"] = "auto‑validator mismatch"
        trans["slots_added"] = added
        trans["slots_removed"] = removed


def main(inp: pathlib.Path, out: pathlib.Path):
    data = json.load(inp.open())
    for dlg in data:
        for turn in dlg["annotation"].values():
            validate_turn(turn)
    json.dump(data, out.open("w"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    main(pathlib.Path(args.input), pathlib.Path(args.output))

# ----------------------------------------------------------------
# pipeline.sh – chain everything (bash snippet)
# ----------------------------------------------------------------
#
#   INPUT=raw_llm.json
#   python clean_basic.py           --input $INPUT      --output step1.json
#   python merge_slot_names.py      --input step1.json  --output step2.json  --top 300
#   python fill_state_and_action.py --input step2.json  --output step3.json
#   python validate_transition.py   --input step3.json  --output final.json
#   echo "✅  Finished → final.json"
# ----------------------------------------------------------------
