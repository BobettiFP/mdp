#!/usr/bin/env python3
"""
Dynamic Slot Analyzer — transition‑aware
---------------------------------------
CLI
$ python dynamic_slot_analyzer.py \
        --human dialogues_001.json \
        --llm   full_annotation.json \
        --export processed_annotations.json

Changes 2025‑06‑06:
  • Added support for schema2qa/SGD‑style `state` objects (nested by domain)
    so LLM annotations are properly parsed.
  • `_extract_turn_slots` now unifies three formats:
      1. Simple `{"slots": {...}}`
      2. MultiWOZ `frames[i].state.slot_values`
      3. Schema‑style `state` `{domain: {slot: value}}`
"""
import json, pathlib, argparse, collections
from typing import Dict, List, Tuple

# ----- canonicalisation ------------------------------------------------------
_OVERRIDES = {
    "postcode": "post_code",
    "pricerange": "price_range",
    "price range": "price_range",
    "post code": "post_code",
}

def canon(slot: str) -> str:
    s = slot.lower().replace("-", "_").replace(" ", "_")
    return _OVERRIDES.get(s, s)

# ----- load raw dialogue files ----------------------------------------------

def _extract_turn_slots(turn: Dict) -> Dict[str, str]:
    """Return flat {domain_slot: value} mapping for a single turn.

    Handles:
      • Simple `"slots"` dict (e.g. ConvLab human annotations)
      • MultiWOZ `"frames"[i]["state"]["slot_values"]`
      • Schema‑style `"state"` nested by domain (LLM output)
    """
    slots: Dict[str, str] = {}

    # 1) Direct "slots" field
    if "slots" in turn and isinstance(turn["slots"], dict):
        slots.update({canon(k): v for k, v in turn["slots"].items()})

    # 2) MultiWOZ frames -> state -> slot_values
    for f in turn.get("frames", []):
        sv = f.get("state", {}).get("slot_values", {})
        for k, vals in sv.items():
            key = canon(k)
            if isinstance(vals, list) and vals:
                val = vals[-1]
                if isinstance(val, dict) and "value" in val:
                    val = val["value"]
                slots[key] = val
            elif isinstance(vals, str):
                slots[key] = vals

    # 3) Schema / SGD‑style nested "state"
    if "state" in turn and isinstance(turn["state"], dict):
        for domain, dom_slots in turn["state"].items():
            if not isinstance(dom_slots, dict):
                continue
            for raw_slot, val in dom_slots.items():
                # unwrap typical {"value": "..."} structures
                if isinstance(val, dict) and "value" in val:
                    val = val["value"]
                if isinstance(val, list):
                    if val:
                        last = val[-1]
                        val = last["value"] if isinstance(last, dict) and "value" in last else last
                key = canon(f"{domain}_{raw_slot}")
                slots[key] = val

    return slots


def load_turns(path: pathlib.Path, ann_type: str) -> List[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {path}: {e}")
        return []

    # Normalise to list[dialogue]
    if isinstance(data, dict) and ("dialogue" in data or "turns" in data):
        data = [data]

    turns: List[Dict] = []
    for dlg in data:
        did = dlg.get("dialogue_id") or dlg.get("dialogueID") or "unknown"
        dialogue_turns = dlg.get("dialogue") or dlg.get("turns") or []
        for i, t in enumerate(dialogue_turns):
            t = dict(t)  # shallow copy
            t["dialogue_id"] = did
            t["annotation_type"] = ann_type
            # turn_id robust cast to int
            try:
                t["turn_id"] = int(t.get("turn_id", i))
            except (ValueError, TypeError):
                t["turn_id"] = i
            turns.append(t)

    return turns

# ----- build (state_before, action, state_after) -----------------------------

def build_records(turns: List[Dict]) -> Tuple[List[Dict], Dict[str, str]]:
    by_dlg = collections.defaultdict(list)
    for t in turns:
        by_dlg[(t["annotation_type"], t["dialogue_id"])].append(t)

    recs: List[Dict] = []
    canon_map: Dict[str, str] = {}

    for (_atype, _did), tl in by_dlg.items():
        tl.sort(key=lambda x: x["turn_id"])
        prev = {}

        for tr in tl:
            cur = _extract_turn_slots(tr)

            # Track canonicalisation (best‑effort)
            raw_keys = set()
            if "slots" in tr and isinstance(tr["slots"], dict):
                raw_keys.update(tr["slots"].keys())
            if "state" in tr and isinstance(tr["state"], dict):
                for dom, dom_slots in tr["state"].items():
                    for r in dom_slots.keys():
                        raw_keys.add(f"{dom}_{r}")
            for raw in raw_keys:
                canon_map[raw] = canon(raw)

            # diff
            add  = {k: v for k, v in cur.items() if prev.get(k) != v}
            dele = {k: None for k in prev if k not in cur}

            if add and not dele:
                act = "inform"
            elif dele and not add:
                act = "delete"
            elif add or dele:
                act = "update"
            else:
                act = "noop"

            recs.append({
                "state_before": dict(prev),
                "action": act,
                "state_after": dict(cur),
                "annotation_type": tr["annotation_type"],
                "dialogue_id": tr["dialogue_id"],
                "turn_id": tr["turn_id"],
                "utterance": tr.get("utterance", ""),
            })
            prev = cur

    return recs, canon_map

# ----- main ------------------------------------------------------------------

def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True, help="Human annotation file path")
    ap.add_argument("--llm", required=True, help="LLM annotation file path")
    ap.add_argument("--export", required=True, help="Output file path")
    args = ap.parse_args()

    print(f"Loading human annotations from {args.human} …")
    human = load_turns(pathlib.Path(args.human), "human")
    print(f"Loading LLM annotations from {args.llm} …")
    llm = load_turns(pathlib.Path(args.llm), "llm")

    print(f"Processing {len(human)} human turns and {len(llm)} LLM turns …")
    recs, cmap = build_records(human + llm)

    out = {"annotations": recs, "canonical_map": cmap}
    output_path = pathlib.Path(args.export)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✔ {args.export} written — {len(recs):,} records (human={len(human)}, llm={len(llm)})")

if __name__ == "__main__":
    _cli()
