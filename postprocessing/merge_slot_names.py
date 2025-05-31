# ----------------------------------------------------------------
# merge_slot_names.py  –  light‑weight similarity clustering
# ----------------------------------------------------------------
"""Usage
python merge_slot_names.py --input step1.json --output step2.json --top 300
"""
import json, argparse, pathlib, math
from collections import Counter, defaultdict
from typing import Dict, Any
import difflib

SIM_TH = 0.9  # Jaro similarity threshold


def jaro(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=a, b=b).ratio()


def build_alias_map(data, top_n: int):
    cnt = Counter()
    for dlg in data:
        for turn in dlg["annotation"].values():
            for field in ("state_before", "state_after"):
                cnt.update(turn[field].keys())
    commons = [w for w, _ in cnt.most_common(top_n)]

    alias = {}
    for i, s in enumerate(commons):
        for t in commons[i + 1:]:
            if jaro(s, t) >= SIM_TH:
                master = min(s, t, key=len)
                alias[t] = master
                alias[s] = master
    return alias


def rename_state(state: Dict[str, Any], alias: Dict[str, str]):
    return {alias.get(k, k): v for k, v in state.items()}


def main(inp: pathlib.Path, out: pathlib.Path, top: int):
    data = json.load(inp.open())
    alias = build_alias_map(data, top)
    for dlg in data:
        for turn in dlg["annotation"].values():
            turn["state_before"] = rename_state(turn["state_before"], alias)
            turn["state_after"] = rename_state(turn["state_after"], alias)
    json.dump(data, out.open("w"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--top", type=int, default=300)
    args = ap.parse_args()
    main(pathlib.Path(args.input), pathlib.Path(args.output), args.top)

