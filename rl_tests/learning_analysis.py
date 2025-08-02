#!/usr/bin/env python3
"""
learning_analysis.py
====================
Visualise PPO training logs for **human‑annotated** vs **LLM‑annotated**
Dialog MDP environments.

Expected CSV columns
--------------------
* **episode**  (int)        – 0‑based episode index
* **reward**   (float)      – episodic return
* (opt) **f1** or **success_rate**           – quality metric (if missing, shows MA-100 trend)
* (opt) **epsilon** / **exploration_rate**   – exploration schedule (if missing, computed metrics are used)

Two figures are produced in `--outdir` (created if absent):
1. `learning_efficiency_analysis.png`  – 2×2 learning trends & stability
2. `detailed_reward_analysis.png`      – 2×3 comprehensive reward analysis

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
    """Fill a 2×2 `axs` grid with learning efficiency metrics."""
    ep = df["episode"]

    # (0,0) cumulative average reward - 전체 학습 효율성
    axs[0, 0].plot(ep, df["reward"].expanding().mean(), color=color, label=label)

    # (0,1) long-term trend (MA-100) - 장기 트렌드  
    metric_col = _find_col(df, "f1", "success_rate")
    if metric_col:
        axs[0, 1].plot(ep, roll_mean(df[metric_col], 20), color=color, label=label)
    else:
        axs[0, 1].plot(ep, roll_mean(df["reward"], 100), color=color, label=label)

    # (1,0) short-term trend (MA-20) - 단기 트렌드
    axs[1, 0].plot(ep, roll_mean(df["reward"], 20), color=color, label=label)

    # (1,1) learning stability (variance) - 학습 안정성
    axs[1, 1].plot(ep, roll_var(df["reward"], 30), color=color, label=label)


def plot_reward(axs, df: pd.DataFrame, label: str, color: str):
    """Fill a 2×3 `axs` grid with detailed reward analysis."""
    ep = df["episode"]

    # (0,0) reward distribution histogram - 보상 분포
    axs[0, 0].hist(df["reward"], bins=40, alpha=0.6, density=True, color=color, label=label)

    # (0,1) reward distribution over time (percentiles) - 시간에 따른 분포 변화
    rolling_median = df["reward"].rolling(200, min_periods=1).median()
    rolling_25 = df["reward"].rolling(200, min_periods=1).quantile(0.25)
    rolling_75 = df["reward"].rolling(200, min_periods=1).quantile(0.75)
    
    axs[0, 1].plot(ep, rolling_median, color=color, label=f"{label} (median)", linewidth=2)
    axs[0, 1].plot(ep, rolling_25, color=color, alpha=0.5, linestyle='--', linewidth=1)
    axs[0, 1].plot(ep, rolling_75, color=color, alpha=0.5, linestyle='--', linewidth=1)
    axs[0, 1].fill_between(ep, rolling_25, rolling_75, color=color, alpha=0.1)

    # (0,2) reward variance (50‑ep) - 변동성
    axs[0, 2].plot(ep, roll_var(df["reward"], 50), color=color, label=label)

    # (1,0) cumulative reward - 누적 보상
    axs[1, 0].plot(ep, df["reward"].cumsum(), color=color, label=label)

    # (1,1) reward change (MA‑100 of diff) - 변화율
    axs[1, 1].plot(ep, roll_mean(df["reward"].diff(), 100), linewidth=1.2, color=color, label=label)

    # (1,2) FFT magnitude spectrum (demeaned) - 주파수 분석
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
    ap.add_argument("--head", type=int, default=50000, help="Use only the first N episodes (default: 50000)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    h_df = load_csv(args.human_log, "Human")
    l_df = load_csv(args.llm_log,   "LLM")

    if args.head:
        h_df = h_df[h_df["episode"] < args.head]
        l_df = l_df[l_df["episode"] < args.head]
        print(f"📊 Analyzing first {args.head} episodes")

    h_df = downsample(h_df, target=args.max_points)
    l_df = downsample(l_df, target=args.max_points)

    # ---------------- Learning efficiency (2×2) ----------------
    fig1, axs1 = plt.subplots(2, 2, figsize=(10, 8))
    plot_learning(axs1, h_df, "Human", "royalblue")
    plot_learning(axs1, l_df, "LLM",   "orange")

    axs1[0, 0].set_title("Cumulative Average Reward (Learning Efficiency)")
    axs1[0, 0].set_xlabel("Episode"); axs1[0, 0].set_ylabel("Cumulative Avg Reward")
    axs1[0, 0].legend()

    axs1[0, 1].set_title("Metric Convergence (MA‑20)"); axs1[0, 1].set_xlabel("Episode")
    axs1[0, 1].legend()
    
    axs1[1, 0].set_title("Exploration Rate")
    axs1[1, 0].set_xlabel("Episode")
    axs1[1, 0].legend()
    
    axs1[1, 1].set_title("Learning Stability (30‑episode variance)")
    axs1[1, 1].set_xlabel("Episode")
    axs1[1, 1].set_ylabel("Reward Variance")
    axs1[1, 1].legend()

    plt.tight_layout()
    fig1_path = outdir / "learning_efficiency_analysis.png"
    plt.savefig(fig1_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved learning efficiency plot: {fig1_path}")

    # ---------------- Reward quality analysis (2×3) ----------------
    fig2, axs2 = plt.subplots(2, 3, figsize=(15, 8))
    plot_reward(axs2, h_df, "Human", "royalblue")
    plot_reward(axs2, l_df, "LLM",   "orange")

    axs2[0, 0].set_title("Reward Distribution")
    axs2[0, 0].set_xlabel("Reward"); axs2[0, 0].set_ylabel("Density")
    axs2[0, 0].legend()

    axs2[0, 1].set_title("Reward Evolution (MA‑100)")
    axs2[0, 1].set_xlabel("Episode"); axs2[0, 1].set_ylabel("Reward")
    axs2[0, 1].legend()

    axs2[0, 2].set_title("Reward Variance (50‑episode)")
    axs2[0, 2].set_xlabel("Episode"); axs2[0, 2].set_ylabel("Variance")
    axs2[0, 2].legend()

    axs2[1, 0].set_title("Cumulative Reward")
    axs2[1, 0].set_xlabel("Episode"); axs2[1, 0].set_ylabel("Cumulative Reward")
    axs2[1, 0].legend()

    axs2[1, 1].set_title("Reward Change (MA‑100)")
    axs2[1, 1].set_xlabel("Episode"); axs2[1, 1].set_ylabel("Reward Δ")
    axs2[1, 1].legend()

    axs2[1, 2].set_title("FFT Magnitude Spectrum")
    axs2[1, 2].set_xlabel("Frequency"); axs2[1, 2].set_ylabel("Magnitude")
    axs2[1, 2].legend()

    plt.tight_layout()
    fig2_path = outdir / "detailed_reward_analysis.png"
    plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved detailed analysis plot: {fig2_path}")

    plt.show()  # 선택사항: GUI에서 플롯 보기
    print(f"\n📊 Analysis complete! Plots saved in: {outdir}")
    print(f"   📈 Learning trends: learning_efficiency_analysis.png")
    print(f"   🔍 Detailed analysis: detailed_reward_analysis.png")


if __name__ == "__main__":
    main()