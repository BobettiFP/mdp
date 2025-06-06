#!/usr/bin/env python3
"""
Dynamic Slot Comparison — no-human variant
========================================
Clusters *all* slot names fully automatically, using a hybrid of
string-edit distance (rapidfuzz QRatio) and semantic embedding similarity
(SBERT).  No manual alias tables required.

Run:
-----
$ python dynamic_slot_comparison.py \
      --human 50_dialogues_001.json \
      --llm   v3_results_first_50.json \
      --export processed_annotations.json

Dependencies:
-------------
    pip install rapidfuzz sentence-transformers scikit-learn numpy networkx
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import networkx as nx
import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

###############################################################################
# Normalisation helper                                                        #
###############################################################################

def _normalise_token(token: str) -> str:
    """Lower-case, snake-case and strip obvious domain prefixes."""
    token = token.lower().replace("-", "_")
    token = token.replace("postcode", "post_code").replace("pricerange", "price_range")
    for prefix in (
        "restaurant_", "hotel_", "train_", "taxi_", "attraction_", "bus_",
        "hospital_", "police_",
    ):
        if token.startswith(prefix):
            token = token[len(prefix):]
    return token

###############################################################################
# Slot clustering                                                             #
###############################################################################

class SlotClusterer:
    """Cluster slot names via hybrid string+semantic similarity graph."""

    def __init__(self,
                 str_thresh_high: float = 90.0,
                 str_thresh_low: float = 80.0,
                 sem_thresh_high: float = 0.80,
                 sem_thresh_low: float = 0.70):
        self.str_hi = str_thresh_high
        self.str_lo = str_thresh_low
        self.sem_hi = sem_thresh_high
        self.sem_lo = sem_thresh_low
        self._rep: Dict[str, str] = {}
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    # ------------------------------------------------------------------
    def fit(self, slot_names: Iterable[str]):
        names = list(slot_names)
        if not names:
            return
        norm = [_normalise_token(n) for n in names]
        embeds = self._model.encode(norm, show_progress_bar=False, convert_to_numpy=True)

        G = nx.Graph()
        G.add_nodes_from(range(len(names)))

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                s_ratio = fuzz.QRatio(norm[i], norm[j])
                if s_ratio >= self.str_hi:
                    G.add_edge(i, j)
                    continue
                sim = self._cosine(embeds[i], embeds[j])
                if sim >= self.sem_hi:
                    G.add_edge(i, j)
                    continue
                if s_ratio >= self.str_lo and sim >= self.sem_lo:
                    G.add_edge(i, j)

        for comp in nx.connected_components(G):
            canon = min((norm[idx] for idx in comp), key=len)
            for idx in comp:
                self._rep[names[idx]] = canon

    # ------------------------------------------------------------------
    def canonical(self, raw: str) -> str:
        """Return canonical representative for raw slot name."""
        return self._rep.get(raw, _normalise_token(raw))

###############################################################################
# Data model                                                                  #
###############################################################################

@dataclass
class DynamicAnnotation:
    utterance: str
    slots: Dict[str, str]
    annotation_type: str  # "human" | "llm"
    dialogue_id: Optional[str]
    turn_id: Optional[int | str]
    metadata: Dict = field(default_factory=dict)

###############################################################################
# Analyzer                                                                    #
###############################################################################

class DynamicSlotAnalyzer:
    def __init__(self):
        self.human_annotations: List[DynamicAnnotation] = []
        self.llm_annotations: List[DynamicAnnotation] = []
        self._clusterer: Optional[SlotClusterer] = None
        self.log = logging.getLogger("SlotAnalyzer")

    # ───────────────────────── ingest ────────────────────────────
    def ingest_multiwoz(self, path: Path, annotation_type: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if not isinstance(item, dict):
                continue
            if "error" in item and ("annotation" not in item and "turns" not in item):
                continue  # skip stub entries
            dialogue_id = item.get("dialogue_id") or item.get("dialogueID")
            if "annotation" in item and isinstance(item["annotation"], dict):
                turns = item["annotation"].values()
            elif "turns" in item and isinstance(item["turns"], list):
                turns = item["turns"]
            else:
                continue
            for turn in turns:
                role = (turn.get("speaker") or turn.get("speaker_role") or turn.get("role") or "user").lower()
                if role not in ("user", "usr", "customer"):
                    continue
                utt = turn.get("utterance") or turn.get("text") or ""
                slots = self._extract_slots(turn)
                if not slots:
                    continue
                self._store(annotation_type, utt, slots, dialogue_id, turn)

    def _store(self, a_type: str, utt: str, slots: Dict[str, str], dlg_id: Optional[str], turn: Dict):
        ann = DynamicAnnotation(
            utterance=utt,
            slots=slots,
            annotation_type=a_type,
            dialogue_id=dlg_id,
            turn_id=turn.get("turn_id") or turn.get("turn_index"),
        )
        (self.human_annotations if a_type == "human" else self.llm_annotations).append(ann)

    # ───────────────────── slot extraction helpers ───────────────────
    def _extract_slots(self, turn: Dict) -> Dict[str, str]:
        slots: Dict[str, str] = {}
        # frames/state/slot_values (schema-V2)
        if "frames" in turn:
            for fr in turn["frames"]:
                for k, v in fr.get("state", {}).get("slot_values", {}).items():
                    v = v[0] if isinstance(v, list) and v else v
                    slots[k] = v
        # state flat or nested
        if "state" in turn and isinstance(turn["state"], dict):
            st = turn["state"]
            if all(isinstance(v, dict) for v in st.values()):
                for d, sv in st.items():
                    for k, v in sv.items():
                        slots[f"{d}_{k}"] = v
            else:
                slots.update(st)
        # belief_state variant
        if "belief_state" in turn and isinstance(turn["belief_state"], dict):
            for d, sv in turn["belief_state"].items():
                if isinstance(sv, dict):
                    for k, v in sv.items():
                        slots[f"{d}_{k}"] = v
        # state_before / state_after
        for key in ("state_before", "state_after"):
            if key in turn and isinstance(turn[key], dict):
                sd = turn[key]
                if all(isinstance(v, dict) for v in sd.values()):
                    for d, sv in sd.items():
                        for k, v in sv.items():
                            slots[f"{d}_{k}"] = v
                else:
                    slots.update(sd)
        # dialogue_acts fallback
        if not slots and "dialogue_acts" in turn:
            for act in turn["dialogue_acts"]:
                slot = act.get("slot"); val = act.get("value")
                if slot and val is not None:
                    dom = act.get("domain", "")
                    key = f"{dom}_{slot}" if dom else slot
                    slots[key] = val
        return slots

    # ─────────────────── cluster & analyse ────────────────────────
    def fit_slot_clusters(self):
        all_slots: Set[str] = {s for ann in (self.human_annotations + self.llm_annotations) for s in ann.slots}
        self._clusterer = SlotClusterer()
        self._clusterer.fit(all_slots)
        self.log.info("Hybrid clustering applied to %d raw slot names", len(all_slots))

    def analyse(self):
        if not self._clusterer:
            raise RuntimeError("fit_slot_clusters() must be called first")
        h = {self._clusterer.canonical(s) for a in self.human_annotations for s in a.slots}
        l = {self._clusterer.canonical(s) for a in self.llm_annotations for s in a.slots}
        return {
            "unique_human": len(h),
            "unique_llm": len(l),
            "overlap": len(h & l),
            "llm_innovation_rate": round((len(l - h) / len(l)), 3) if l else 0.0,
        }

    # ───────────────────────── export ────────────────────────────
    def export(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "canonical_map": self._clusterer._rep if self._clusterer else {},
            "human_annotations": [a.__dict__ for a in self.human_annotations],
            "llm_annotations": [a.__dict__ for a in self.llm_annotations],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.log.info("Exported merged JSON → %s", path)

###############################################################################
# CLI                                                                        #
###############################################################################

def _parse_args():
    p = argparse.ArgumentParser(description="Compare slot coverage with hybrid clustering")
    p.add_argument("--human", type=Path, required=True, help="Path to human-annotated JSON")
    p.add_argument("--llm", type=Path, required=True, help="Path to LLM-annotated JSON")
    p.add_argument("--export", type=Path, help="Optional output JSON with canonical map")
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
