"""
Strategic‑Complexity Pipeline (clean rewrite)
-------------------------------------------
Evaluates *strategic complexity* of dialogue annotations (human vs LLM) **without**
any hard‑coded biases, mock data, or domain‑prefix stripping.

Usage
-----
$ python strategic_complexity_pipeline.py \
        --annotations processed_annotations.json \
        --rl-hist-human logs/human_env.csv \
        --rl-hist-llm   logs/llm_env.csv 

The script will output a summary JSON (strategic_complexity_results.json)
and a pair of quick‑look PNG visualisations in ./figures/.

Key fixes vs legacy version
---------------------------
1. *No* domain‑prefix stripping – slots like "restaurant_name" & "hotel_name" stay distinct.
2. *Real* RL histories: cumulative reward etc. loaded from CSV logs instead of random mocks.
3. Zero hard‑coded LLM bonuses. Same formulae for both groups.
4. Deterministic – random seeds fixed.
5. Shared utilities extracted for clarity and unit‑testability.

Author: (your name)
Date  : 2025‑06‑06
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

###############################################################################
# Global config & utils                                                      #
###############################################################################
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)


def normalize_slot(slot: str) -> str:
    """Basic normalisation → lowercase + hyphen→underscore only."""
    return slot.lower().replace("-", "_")


def load_annotations(path: Path) -> Tuple[List[dict], List[dict]]:
    """Load processed_annotations.json and split by annotation_type."""
    data = json.loads(path.read_text())

    # new structure: list under "all_annotations" OR pre‑split lists
    recs: List[dict] = []
    if isinstance(data, dict) and "all_annotations" in data:
        recs.extend(data["all_annotations"])
    else:
        recs.extend(data.get("human_annotations", []))
        recs.extend(data.get("llm_annotations", []))

    humans = [r for r in recs if r.get("annotation_type") == "human"]
    llms   = [r for r in recs if r.get("annotation_type") == "llm"]

    if not humans or not llms:
        raise ValueError("Both human and LLM annotations must be present & tagged.")
    return humans, llms

###############################################################################
# Metric calculations                                                        #
###############################################################################
@dataclass
class DialogueMetrics:
    planning_depth: float
    goal_directedness: float
    behavioural_complexity: float

    @classmethod
    def from_dialogue(cls, turns: List[dict]) -> "DialogueMetrics":
        """Compute simple illustrative metrics for a single dialogue."""
        # planning depth ≈ #unique slots touched / #turns
        slots = set()
        for t in turns:
            slots.update(map(normalize_slot, t.get("slots", {}).keys()))
        depth = len(slots) / max(len(turns), 1)

        # goal_directedness: ratio of user requests satisfied at final turn
        satisfied = turns[-1].get("goal_progress") == "complete"
        directed  = 1.0 if satisfied else 0.0

        # behavioural complexity: entropy over action types
        actions = [a for t in turns for a in t.get("actions", [])]
        cnt = Counter(actions)
        probs = np.array(list(cnt.values())) / max(sum(cnt.values()), 1)
        entropy = float(-(probs * np.log2(probs + 1e-9)).sum())

        return cls(depth, directed, entropy)


@dataclass
class CorpusMetrics:
    planning_depth_mean: float
    goal_directedness_mean: float
    behavioural_complexity_mean: float

    @classmethod
    def from_dialogues(cls, dialogs: List[List[dict]]) -> "CorpusMetrics":
        metrics = [DialogueMetrics.from_dialogue(d) for d in dialogs]
        if not metrics:
            raise ValueError("No dialogues provided to CorpusMetrics.")
        depth = np.mean([m.planning_depth for m in metrics])
        goal  = np.mean([m.goal_directedness for m in metrics])
        beh   = np.mean([m.behavioural_complexity for m in metrics])
        return cls(depth, goal, beh)

    def strategic_score(self, w_depth=0.4, w_goal=0.3, w_beh=0.3) -> float:
        return (
            w_depth * self.planning_depth_mean +
            w_goal  * self.goal_directedness_mean +
            w_beh   * self.behavioural_complexity_mean
        )

###############################################################################
# RL history metrics                                                         #
###############################################################################

def load_rl_history(csv_path: Path) -> pd.DataFrame:
    """Assume columns: episode, reward, success_rate."""
    df = pd.read_csv(csv_path)
    required = {"episode", "reward", "success_rate"}
    if not required.issubset(df.columns):
        raise ValueError(f"RL log {csv_path} missing columns: {required-df.columns}")
    return df.sort_values("episode")


def efficiency_metrics(df: pd.DataFrame) -> Dict[str, float]:
    cum_reward = df["reward"].cumsum()
    # simple estimates
    speed      = (cum_reward.iloc[99] - cum_reward.iloc[0]) / 100  # first 100 episodes slope
    sample_eff = df["success_rate"].mean()
    return {"learning_speed": speed, "sample_efficiency": sample_eff}

###############################################################################
# Visualisation helpers                                                      #
###############################################################################

def quick_plot(df_h: pd.DataFrame, df_l: pd.DataFrame, out: Path):
    plt.figure(figsize=(6,4))
    plt.plot(df_h["episode"], df_h["reward"].cumsum(), label="Human env")
    plt.plot(df_l["episode"], df_l["reward"].cumsum(), label="LLM env")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

###############################################################################
# Main pipeline                                                              #
###############################################################################

def run(args):
    humans, llms = load_annotations(Path(args.annotations))

    # split into dialogues (list[list[turn]])
    by_id_h: Dict[str, List[dict]] = defaultdict(list)
    by_id_l: Dict[str, List[dict]] = defaultdict(list)
    for rec in humans:
        by_id_h[rec["dialogue_id"]].append(rec)
    for rec in llms:
        by_id_l[rec["dialogue_id"]].append(rec)

    cm_h = CorpusMetrics.from_dialogues(list(by_id_h.values()))
    cm_l = CorpusMetrics.from_dialogues(list(by_id_l.values()))

    score_h = cm_h.strategic_score()
    score_l = cm_l.strategic_score()

    # RL logs (optional; skip if not given)
    rl_h = rl_l = None
    eff_h = eff_l = {}
    if args.rl_hist_human and args.rl_hist_llm:
        rl_h = load_rl_history(Path(args.rl_hist_human))
        rl_l = load_rl_history(Path(args.rl_hist_llm))
        eff_h = efficiency_metrics(rl_h)
        eff_l = efficiency_metrics(rl_l)
        quick_plot(rl_h, rl_l, Path("figures/cumulative_reward.png"))

    # save results
    out = {
        "human": {
            "strategic_metrics": cm_h.__dict__,
            "strategic_score": score_h,
            "efficiency_metrics": eff_h,
        },
        "llm": {
            "strategic_metrics": cm_l.__dict__,
            "strategic_score": score_l,
            "efficiency_metrics": eff_l,
        },
    }
    Path("strategic_complexity_results.json").write_text(json.dumps(out, indent=2))
    print("✔ strategic_complexity_results.json written. Human score = {:.3f}, LLM score = {:.3f}".format(score_h, score_l))

###############################################################################
# CLI                                                                        #
###############################################################################

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Strategic‑Complexity evaluation pipeline (clean version)")
    p.add_argument("--annotations", required=True, help="Path to processed_annotations.json")
    p.add_argument("--rl-hist-human", help="CSV log of RL training in human‑labelled env")
    p.add_argument("--rl-hist-llm", help="CSV log of RL training in LLM‑labelled env")
    args = p.parse_args()
    run(args)
