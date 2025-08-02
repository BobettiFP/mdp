from __future__ import annotations

import argparse, json, os, random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------- goal achievement by slot satisfaction --------------------
def goal_achieved(records: List[dict]) -> float:
    requested_slots = set()
    satisfied_slots = set()

    for r in records:
        if r["turn_id"] % 2 == 0 and r["action"] == "inform":
            requested_slots.update(r.get("state_after", {}).keys())

    for r in records:
        sa = r.get("state_after", {})
        for slot in requested_slots:
            if slot in sa:
                satisfied_slots.add(slot)

    return float(bool(satisfied_slots & requested_slots))

# ---------------- strategic metrics per dialogue ---------------------------
def _normalize(slot: str) -> str:
    return slot.lower().replace("-", "_")

def metrics_per_dialog(records: List[dict]) -> Dict[str, float]:
    slots = set()
    actions = []
    for r in records:
        slots.update(map(_normalize, r.get("state_after", {}).keys()))
        actions.append(r["action"])

    turns = len(records)
    depth = len(slots) / max(turns, 1)
    goal = goal_achieved(records)

    from collections import Counter
    cnt = Counter(actions)
    p = np.array(list(cnt.values())) / max(sum(cnt.values()), 1)
    beh = float(-(p * np.log2(p + 1e-9)).sum())

    return {
        "planning_depth": depth,
        "goal_directedness": goal,
        "behavioural_complexity": beh
    }

# ---------------- corpus aggregation ---------------------------------------
def corpus_metrics(dialogs: Dict[str, List[dict]]):
    ms = [metrics_per_dialog(d) for d in dialogs.values()]
    arr = {k: np.mean([m[k] for m in ms]) for k in ms[0].keys()}
    score = 0.4 * arr["planning_depth"] + 0.3 * arr["goal_directedness"] + 0.3 * arr["behavioural_complexity"]
    return arr, score

# ---------------- RL logs (optional) ---------------------------------------
def load_rl(csv: Path):
    df = pd.read_csv(csv)
    need = {"episode", "reward"}
    if not need.issubset(df.columns):
        raise ValueError(f"{csv} missing {need - set(df.columns)}")
    return df.sort_values("episode")

def learning_efficiency(df: pd.DataFrame):
    speed = (df.reward.cumsum().iloc[199] - df.reward.cumsum().iloc[0]) / 200 if len(df) > 200 else 0
    sample_eff = df.reward.gt(0).mean()
    return {"learning_speed": speed, "sample_efficiency": sample_eff}

# ---------------- plotting --------------------------------------------------
def plot_metrics(h: Dict[str, float], l: Dict[str, float], out: Path):
    keys = list(h.keys())
    x = np.arange(len(keys))
    plt.figure(figsize=(6, 4))
    plt.bar(x - 0.15, [h[k] for k in keys], width=0.3, label="Human")
    plt.bar(x + 0.15, [l[k] for k in keys], width=0.3, label="LLM")
    plt.xticks(x, keys, rotation=15)
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def plot_cumreward(h_df: pd.DataFrame, l_df: pd.DataFrame, out: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(h_df.episode, h_df.reward.cumsum(), label="Human env")
    plt.plot(l_df.episode, l_df.reward.cumsum(), label="LLM env")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def write_report(out: dict, path: Path):
    lines = []
    for src in ["human", "llm"]:
        m = out[src]["strategic_metrics"]
        lines.append(f"[{src.upper()}]")
        for k, v in m.items():
            lines.append(f"{k:<25}: {v:.4f}")
        lines.append(f"strategic_score          : {out[src]['strategic_score']:.4f}\n")

    if "efficiency_metrics" in out["human"]:
        for src in ["human", "llm"]:
            e = out[src]["efficiency_metrics"]
            lines.append(f"[{src.upper()} RL Efficiency]")
            for k, v in e.items():
                lines.append(f"{k:<25}: {v:.4f}")
            lines.append("")

    path.write_text("\n".join(lines))

# ---------------- main ------------------------------------------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--annotations", required=True)
    pa.add_argument("--rl-hist-human", dest="rl_h")
    pa.add_argument("--rl-hist-llm", dest="rl_l")
    pa.add_argument("--outdir", default="strategic_complexity_results")
    args = pa.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data = json.load(open(args.annotations))
    recs = data["annotations"] if isinstance(data, dict) else data
    human = [r for r in recs if r.get("annotation_type") == "human"]
    llm = [r for r in recs if r.get("annotation_type") == "llm"]

    by_h = defaultdict(list)
    by_l = defaultdict(list)
    for r in human:
        by_h[r["dialogue_id"]].append(r)
    for r in llm:
        by_l[r["dialogue_id"]].append(r)

    met_h, score_h = corpus_metrics(by_h)
    met_l, score_l = corpus_metrics(by_l)

    out = {
        "human": {"strategic_metrics": met_h, "strategic_score": score_h},
        "llm": {"strategic_metrics": met_l, "strategic_score": score_l},
    }

    if args.rl_h and args.rl_l:
        df_h = load_rl(Path(args.rl_h))
        df_l = load_rl(Path(args.rl_l))
        out["human"]["efficiency_metrics"] = learning_efficiency(df_h)
        out["llm"]["efficiency_metrics"] = learning_efficiency(df_l)
        plot_cumreward(df_h, df_l, Path(args.outdir)/"cumulative_reward.png")

    Path(args.outdir, "strategic_complexity_results.json").write_text(json.dumps(out, indent=2))
    plot_metrics(met_h, met_l, Path(args.outdir)/"strategic_metrics.png")
    write_report(out, Path(args.outdir)/"summary.txt")
    print(f"✔ results + plots saved in {args.outdir}  |  Human={score_h:.3f}  LLM={score_l:.3f}")

if __name__ == "__main__":
    main()
