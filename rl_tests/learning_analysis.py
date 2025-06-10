#!/usr/bin/env python3
"""
learning_analysis.py
====================
Visualise PPO training logs for **human‑annotated** vs **LLM‑annotated**
Dialog MDP environments.

Expected CSV columns
--------------------
* **episode**  (int)        – 0‑based episode index
* **reward**   (float)      – episodic return
* (opt) **f1** or **success_rate**           – quality metric
* (opt) **epsilon** / **exploration_rate**   – exploration schedule

Two figures are produced in `--outdir` (created if absent):
1. `learning_efficiency_analysis.png`  – 2×2 composite
2. `reward_quality_analysis.png`       – 2×3 composite

CLI examples
------------
```bash
python rl_tests/learning_analysis.py \
    --human-log logs/human_env.csv \
    --llm-log   logs/llm_env.csv   \
    --outdir    analysis_plots     # full log

python rl_tests/learning_analysis.py \
    --human-log logs/human_env.csv \
    --llm-log   logs/llm_env.csv   \
    --outdir    analysis_plots \
    --head 100                      # first 100 episodes only
```
"""

from __future__ import annotations
import argparse, os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def downsample(df: pd.DataFrame, target: int = 2000) -> pd.DataFrame:
    """Return *df* if short enough, else keep every k‑th row so len ≤ target."""
    if len(df) <= target:
        return df
    step = int(np.ceil(len(df) / target))
    return df.iloc[::step].reset_index(drop=True)


def roll_mean(s: pd.Series, win: int):
    return s.rolling(win, min_periods=1).mean()


def roll_var(s: pd.Series, win: int):
    return s.rolling(win, min_periods=1).var()


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def plot_learning(axs, df: pd.DataFrame, label: str, color: str):
    """Fill a 2×2 `axs` grid."""
    ep = df["episode"]

    # (0,0) cumulative average reward
    axs[0, 0].plot(ep, df["reward"].expanding().mean(), color=color, label=label)

    # (0,1) metric convergence (F1 or success_rate)
    metric_col = _find_col(df, "f1", "success_rate")
    if metric_col:
        axs[0, 1].plot(ep, roll_mean(df[metric_col], 20), color=color, label=label)
    else:
        axs[0, 1].text(0.5, 0.5, "<metric missing>", ha="center", va="center", transform=axs[0, 1].transAxes)

    # (1,0) exploration rate
    eps_col = _find_col(df, "epsilon", "exploration_rate", "eps", "explore", "e")
    if eps_col:
        axs[1, 0].plot(ep, df[eps_col], color=color, label=label)
    else:
        axs[1, 0].text(0.5, 0.5, "<exploration rate missing>", ha="center", va="center", transform=axs[1, 0].transAxes, fontsize=9, color="gray")
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 0].set_frame_on(False)

    # (1,1) reward variance (30‑episode)
    axs[1, 1].plot(ep, roll_var(df["reward"], 30), color=color, label=label)


def plot_reward(axs, df: pd.DataFrame, label: str, color: str):
    """Fill a 2×3 `axs` grid."""
    ep = df["episode"]

    # (0,0) reward distribution
    axs[0, 0].hist(df["reward"], bins=40, alpha=0.6, density=True, color=color, label=label)

    # (0,1) reward evolution (MA‑100)
    axs[0, 1].plot(ep, roll_mean(df["reward"], 100), linewidth=1.2, color=color, label=label)

    # (0,2) reward variance (50‑ep)
    axs[0, 2].plot(ep, roll_var(df["reward"], 50), color=color, label=label)

    # (1,0) cumulative reward
    axs[1, 0].plot(ep, df["reward"].cumsum(), color=color, label=label)

    # (1,1) reward Δ (MA‑100 of diff)
    axs[1, 1].plot(ep, roll_mean(df["reward"].diff(), 100), linewidth=1.2, color=color, label=label)

    # (1,2) FFT magnitude spectrum (demeaned)
    demeaned = (df["reward"] - df["reward"].mean()).values
    yf = np.abs(rfft(demeaned))
    xf = rfftfreq(len(demeaned), d=1)
    axs[1, 2].plot(xf, yf, color=color, label=label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_csv(path: str, role: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{role} log not found: {path}")
    df = pd.read_csv(path)
    if not {"episode", "reward"}.issubset(df.columns):
        raise ValueError(f"{role} log must have 'episode' and 'reward' columns")
    df = df.sort_values("episode").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description="Plot learning & reward diagnostics")
    ap.add_argument("--human-log", required=True, help="CSV path for human‑annotated run")
    ap.add_argument("--llm-log",   required=True, help="CSV path for LLM‑annotated run")
    ap.add_argument("--outdir", default="analysis_plots", help="Output directory for images")
    ap.add_argument("--max-points", type=int, default=2000, help="Downsample each curve to at most N points")
    ap.add_argument("--head", type=int, default=None, help="Use only the first N episodes")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    h_df = load_csv(args.human_log, "Human")
    l_df = load_csv(args.llm_log,   "LLM")

    if args.head:
        h_df = h_df[h_df["episode"] < args.head]
        l_df = l_df[l_df["episode"] < args.head]

    h_df = downsample(h_df, target=args.max_points)
    l_df = downsample(l_df, target=args.max_points)

    # ---------------- Learning efficiency (2×2) ----------------
    fig1, axs1 = plt.subplots(2, 2, figsize=(10, 8))
    plot_learning(axs1, h_df, "Human", "royalblue")
    plot_learning(axs1, l_df, "LLM",   "orange")

    axs1[0, 0].set_title("Cumulative Average Reward (Learning Efficiency)")
    axs1[0, 0].set_xlabel("Episode"); axs1[0, 0].set_ylabel("Cumulative Avg Reward")

    axs1[0, 1].set_title("Metric Convergence (MA‑20)"); axs1[0, 1].set_xlabel("Episode")
    axs1[1, 0].set_title("Exploration Rate")
    axs1[1, 1].set_title("Learning Stability (30‑episode variance)"); axs1[1, 1].set_xlabel("Episode")