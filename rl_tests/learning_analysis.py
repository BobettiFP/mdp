#!/usr/bin/env python3
"""
learning_analysis.py
====================
Visualise PPO training logs for **human‑annotated** vs **LLM‑annotated**
Dialog MDP environments.

Expected CSV columns
--------------------
* **episode**  (int)        – 0‑based episode index (required)
* **reward**   (float)      – episodic return (required)
* (opt) **f1** or **success_rate**           – quality metric (if missing, uses reward trends)
* (opt) **epsilon** / **exploration_rate**   – exploration schedule (if missing, computed automatically)

Two sets of figures are produced in `--outdir` (created if absent):

Learning Efficiency Analysis (4 files):
1. `learning_efficiency.png`       – overall learning progress
2. `long_term_trend.png`           – MA-100 smoothed trend
3. `short_term_trend.png`          – MA-20 responsive trend  
4. `learning_stability.png`        – variance-based stability

Detailed Reward Analysis (6 files):
5. `reward_distribution.png`       – histogram of reward values
6. `distribution_over_time.png`    – temporal distribution changes
7. `reward_variance.png`           – 50-episode variance
8. `cumulative_reward.png`         – total accumulated reward
9. `reward_change_rate.png`        – rate of reward improvement
10. `frequency_analysis.png`       – FFT spectrum analysis

CLI examples
------------
```bash
python rl_tests/learning_analysis.py \
    --human-log logs/human_env.csv \
    --llm-log   logs/llm_env.csv   \
    --outdir    analysis_plots     # creates 10 individual PNG files

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


def _find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def save_individual_plot(plot_func, h_df, l_df, title, filename, outdir, figsize=(8, 6)):
    """개별 subplot을 별도 파일로 저장"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_func(ax, h_df, "Human", "royalblue")
    plot_func(ax, l_df, "LLM", "orange")
    
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    
    filepath = outdir / f"{filename}.png"
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()  # 메모리 절약
    print(f"✓ Saved: {filename}.png")


def plot_cumulative_avg(ax, df, label, color):
    """누적 평균 보상"""
    ep = df["episode"]
    ax.plot(ep, df["reward"].expanding().mean(), color=color, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Avg Reward")


def plot_long_term_trend(ax, df, label, color):
    """장기 트렌드 (MA-100)"""
    ep = df["episode"]
    metric_col = _find_col(df, "f1", "success_rate")
    if metric_col:
        ax.plot(ep, roll_mean(df[metric_col], 20), color=color, label=label)
        ax.set_ylabel("Metric Value")
    else:
        ax.plot(ep, roll_mean(df["reward"], 100), color=color, label=label)
        ax.set_ylabel("Reward (MA-100)")
    ax.set_xlabel("Episode")


def plot_short_term_trend(ax, df, label, color):
    """단기 트렌드 (MA-20)"""
    ep = df["episode"]
    ax.plot(ep, roll_mean(df["reward"], 20), color=color, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (MA-20)")


def plot_learning_stability(ax, df, label, color):
    """학습 안정성 (variance)"""
    ep = df["episode"]
    ax.plot(ep, roll_var(df["reward"], 30), color=color, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward Variance")


def plot_reward_distribution(ax, df, label, color):
    """보상 분포 히스토그램"""
    ax.hist(df["reward"], bins=40, alpha=0.6, density=True, color=color, label=label)
    ax.set_xlabel("Reward")
    ax.set_ylabel("Density")


def plot_distribution_over_time(ax, df, label, color):
    """시간에 따른 분포 변화"""
    ep = df["episode"]
    rolling_median = df["reward"].rolling(200, min_periods=1).median()
    rolling_25 = df["reward"].rolling(200, min_periods=1).quantile(0.25)
    rolling_75 = df["reward"].rolling(200, min_periods=1).quantile(0.75)
    
    ax.plot(ep, rolling_median, color=color, label=f"{label} (median)", linewidth=2)
    ax.plot(ep, rolling_25, color=color, alpha=0.5, linestyle='--', linewidth=1)
    ax.plot(ep, rolling_75, color=color, alpha=0.5, linestyle='--', linewidth=1)
    ax.fill_between(ep, rolling_25, rolling_75, color=color, alpha=0.1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward Percentiles")


def plot_reward_variance(ax, df, label, color):
    """보상 변동성"""
    ep = df["episode"]
    ax.plot(ep, roll_var(df["reward"], 50), color=color, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Variance")


def plot_cumulative_reward(ax, df, label, color):
    """누적 보상"""
    ep = df["episode"]
    ax.plot(ep, df["reward"].cumsum(), color=color, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")


def plot_reward_change_rate(ax, df, label, color):
    """보상 변화율"""
    ep = df["episode"]
    ax.plot(ep, roll_mean(df["reward"].diff(), 100), linewidth=1.2, color=color, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward Δ")


def plot_frequency_analysis(ax, df, label, color):
    """주파수 분석 (FFT)"""
    demeaned = (df["reward"] - df["reward"].mean()).values
    yf = np.abs(rfft(demeaned))
    xf = rfftfreq(len(demeaned), d=1)
    ax.plot(xf, yf, color=color, label=label)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(0, 0.5)  # Nyquist frequency까지만 표시


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

    print(f"📊 Saving individual plots to: {outdir}")
    
    # ---------------- Learning Efficiency Plots ----------------
    print("\n📈 Learning Efficiency Analysis:")
    save_individual_plot(plot_cumulative_avg, h_df, l_df, 
                        "Overall Learning Efficiency", "learning_efficiency", outdir)
    save_individual_plot(plot_long_term_trend, h_df, l_df, 
                        "Long-term Trend (MA-100)", "long_term_trend", outdir)
    save_individual_plot(plot_short_term_trend, h_df, l_df, 
                        "Short-term Trend (MA-20)", "short_term_trend", outdir)
    save_individual_plot(plot_learning_stability, h_df, l_df, 
                        "Learning Stability (30-episode variance)", "learning_stability", outdir)

    # ---------------- Detailed Reward Analysis ----------------
    print("\n🔍 Detailed Reward Analysis:")
    save_individual_plot(plot_reward_distribution, h_df, l_df, 
                        "Reward Distribution", "reward_distribution", outdir)
    save_individual_plot(plot_distribution_over_time, h_df, l_df, 
                        "Distribution over Time (Median + IQR)", "distribution_over_time", outdir)
    save_individual_plot(plot_reward_variance, h_df, l_df, 
                        "Reward Variance (50-episode)", "reward_variance", outdir)
    save_individual_plot(plot_cumulative_reward, h_df, l_df, 
                        "Cumulative Reward", "cumulative_reward", outdir)
    save_individual_plot(plot_reward_change_rate, h_df, l_df, 
                        "Reward Change Rate (MA-100)", "reward_change_rate", outdir)
    save_individual_plot(plot_frequency_analysis, h_df, l_df, 
                        "Frequency Analysis (FFT)", "frequency_analysis", outdir)
    
    print(f"\n🎉 Analysis complete! All plots saved in: {outdir}")
    print(f"📊 Total files created: 10 individual PNG files")
    print(f"\n📈 Learning Efficiency Files:")
    print(f"   • learning_efficiency.png")
    print(f"   • long_term_trend.png") 
    print(f"   • short_term_trend.png")
    print(f"   • learning_stability.png")
    print(f"\n🔍 Detailed Analysis Files:")
    print(f"   • reward_distribution.png")
    print(f"   • distribution_over_time.png")
    print(f"   • reward_variance.png")
    print(f"   • cumulative_reward.png")
    print(f"   • reward_change_rate.png")
    print(f"   • frequency_analysis.png")


if __name__ == "__main__":
    main()