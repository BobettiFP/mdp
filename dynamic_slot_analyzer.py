#!/usr/bin/env python3
"""
Dynamic Slot Comparison Toolkit (clean version)
==============================================
Compares human‑annotated MultiWOZ dialogues (50_dialogues_001.json)
with LLM‑generated annotations (v3_results_first_50.json).

Steps
-----
1. Ingest both files, extracting user‑turn slots from the various
   schema variants we’ve seen (frames/state/belief_state/etc.).
2. Cluster nearly‑identical slot names using rapidfuzz Q‑ratio ≥ 90.
3. Compute overlap / innovation metrics.
4. Optionally export a merged JSON for downstream RL.

Usage
-----
$ python dynamic_slot_comparison.py \
      --human 50_dialogues_001.json \
      --llm   v3_results_first_50.json \
      --export processed_annotations.json

Dependencies
------------
    pip install rapidfuzz sentence-transformers scikit-learn numpy
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
from rapidfuzz import fuzz, process

###############################################################################
# Helpers                                                                     #
###############################################################################

def _normalise_token(token: str) -> str:
    token = token.lower().replace("-", "_")
    token = token.replace("postcode", "post_code").replace("pricerange", "price_range")
    # strip common domain prefixes for clustering similarity
    for prefix in (
        "restaurant_",
        "hotel_",
        "train_",
        "taxi_",
        "attraction_",
        "bus_",
        "hospital_",
        "police_",
    ):
        if token.startswith(prefix):
            token = token[len(prefix):]
    return token

###############################################################################
# Slot clustering                                                             #
###############################################################################

class SlotClusterer:
    """Group slot names that are nearly identical using rapidfuzz."""

    def __init__(self, similarity_threshold: float = 90.0):
        self.similarity_threshold = similarity_threshold
        self._representative: Dict[str, str] = {}

    # ------------------------------------------------------------------
    def fit(self, slot_names: Iterable[str]):
        reps: List[str] = []
        for raw in slot_names:
            norm = _normalise_token(raw)

            if not reps:
                reps.append(norm)
                self._representative[raw] = norm
                continue

            res = process.extractOne(norm, reps, scorer=fuzz.QRatio)
            if res and res[1] >= self.similarity_threshold:
                best = res[0]
            else:
                reps.append(norm)
                best = norm

            self._representative[raw] = best

    # ------------------------------------------------------------------
    def canonical(self, slot_name: str) -> str:
        return self._representative.get(slot_name, _normalise_token(slot_name))

###############################################################################
# Data model                                                                  #
###############################################################################

@dataclass
class DynamicAnnotation:
    utterance: str
    slots: Dict[str, str]
    annotation_type: str  # "human" | "llm"
    dialogue_id: str | None
    turn_id: int | str | None
    metadata: Dict = field(default_factory=dict)

###############################################################################
# Main analyzer                                                               #
###############################################################################

class DynamicSlotAnalyzer:
    def __init__(self):
        self.human_annotations: List[DynamicAnnotation] = []
        self.llm_annotations: List[DynamicAnnotation] = []
        self._clusterer: Optional[SlotClusterer] = None
        self.log = logging.getLogger("SlotAnalyzer")

    # ─────────────────────────── ingest ────────────────────────────────
    def ingest_multiwoz(self, path: Path, annotation_type: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            if not isinstance(item, dict):
                continue
            if "error" in item and ("annotation" not in item and "turns" not in item):
                continue  # skip stub/error entries

            dialogue_id = item.get("dialogue_id") or item.get("dialogueID")

            if "annotation" in item and isinstance(item["annotation"], dict):
                turns = item["annotation"].values()
            elif "turns" in item and isinstance(item["turns"], list):
                turns = item["turns"]
            else:
                continue  # unknown structure

            for turn in turns:
                speaker = (turn.get("speaker") or turn.get("speaker_role") or turn.get("role") or "user").lower()
                if speaker not in ("user", "usr", "customer"):
                    continue  # we analyse only user utterances

                utter = turn.get("utterance") or turn.get("text") or ""
                slots = self._extract_slots_from_turn(turn)
                if not slots:
                    continue

                ann = DynamicAnnotation(
                    utterance=utter,
                    slots=slots,
                    annotation_type=annotation_type,
                    dialogue_id=dialogue_id,
                    turn_id=turn.get("turn_id") or turn.get("turn_index"),
                )
                if annotation_type == "human":
                    self.human_annotations.append(ann)
                else:
                    self.llm_annotations.append(ann)

    # ────────────────────── slot extraction helpers ─────────────────────
    def _extract_slots_from_turn(self, turn: Dict) -> Dict[str, str]:
        slots: Dict[str, str] = {}

        # 1) frames/state/slot_values (MultiWOZ 2.2)
        if "frames" in turn:
            for frame in turn["frames"]:
                sv = frame.get("state", {}).get("slot_values", {})
                for k, v in sv.items():
                    if isinstance(v, list):
                        v = v[0] if v else ""
                    slots[k] = v

        # 2) flat or nested state dict
        if "state" in turn and isinstance(turn["state"], dict):
            st = turn["state"]
            nested = all(isinstance(val, dict) for val in st.values())
            if nested:
                for dom, sv in st.items():
                    for k, v in sv.items():
                        slots[f"{dom}_{k}"] = v
            else:
                slots.update(st)

        # 3) belief_state variant
        if "belief_state" in turn and isinstance(turn["belief_state"], dict):
            for dom, sv in turn["belief_state"].items():
                if isinstance(sv, dict):
                    for k, v in sv.items():
                        slots[f"{dom}_{k}"] = v

        # 4) state_before / state_after (our MDP annotator)
        for key in ("state_before", "state_after"):
            if key in turn and isinstance(turn[key], dict):
                sd = turn[key]
                nested = all(isinstance(val, dict) for val in sd.values())
                if nested:
                    for dom, sv in sd.items():
                        for k, v in sv.items():
                            slots[f"{dom}_{k}"] = v
                else:
                    slots.update(sd)

        # 5) dialogue_acts fallback
        if not slots and "dialogue_acts" in turn:
            for act in turn["dialogue_acts"]:
                slot = act.get("slot")
                value = act.get("value")
                if slot is None or value is None:
                    continue
                dom = act.get("domain", "")
                key = f"{dom}_{slot}" if dom else slot
                slots[key] = value

        return slots

    # ─────────────────────── cluster + analyse ──────────────────────────
    def fit_slot_clusters(self):
        all_slot_names: Set[str] = {
            s for ann in (self.human_annotations + self.llm_annotations) for s in ann.slots
        }
        self._clusterer = SlotClusterer()
        self._clusterer.fit(all_slot_names)
        self.log.info("Built canonical map for %d raw slot names", len(all_slot_names))

    def analyse(self):
        assert self._clusterer is not None, "Run fit_slot_clusters() first"
        h = {self._clusterer.canonical(s) for a in self.human_annotations for s in a.slots}
        l = {self._clusterer.canonical(s) for a in self.llm_annotations for s in a.slots}
        return {
            "unique_human": len(h),
            "unique_llm": len(l),
            "overlap": len(h & l),
            "llm_innovation_rate": (len(l - h) / len(l)) if l else 0,
        }

    # ─────────────────────────── export ────────────────────────────────
    def export(self, path: Path):
        out = {
            "cluster_canonical_map": self._clusterer._representative if self._clusterer else {},
            "human_annotations": [a.__dict__ for a in self.human_annotations],
            "llm_annotations": [a.__dict__ for a in self.llm_annotations],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        self.log.info("Exported merged JSON → %s", path)

###############################################################################
# CLI entry                                                                   #
###############################################################################

def _parse_args():
    p = argparse.ArgumentParser(description="Compare slot coverage")
    p.add_argument("--human", required=True, type=Path, help="Human‑annotated JSON path")
    p.add_argument("--llm", required=True, type=Path, help="LLM‑annotated JSON path")
    p.add_argument("--export", type=Path, help="Optional merged JSON output path")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = _parse_args()

    analyzer = DynamicSlotAnalyzer()
    analyzer.ingest_multiwoz(args.human, "human")
    analyzer.ingest_multiwoz(args.llm, "llm")

    analyzer.fit_slot_clusters()
    metrics = analyzer.analyse()

    print("\n===== Slot Comparison Summary =====")
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').title()}: {v}")

    if args.export:
        analyzer.export(args.export)


if __name__ == "__main__":
    main()
