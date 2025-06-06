"""
Reward‑Efficiency Pipeline (real‑log version)
===========================================
*Compares reward‑signal quality & learning efficiency between human and LLM
annotation environments using **real training logs**.*

CLI
---
$ python reward_efficiency_pipeline.py \
        --annotations processed_annotations.json \
        --human_csv logs/human_env.csv \
        --llm_csv   logs/llm_env.csv

Outputs
--------
reward_quality_analysis.png
learning_efficiency_analysis.png
reward_efficiency_summary.png
reward_efficiency_results.json
(all saved under mdp/rl_tests/reward_efficiency_results/)
"""
import argparse, json, os, math, itertools
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import asdict, dataclass
from collections import defaultdict

# ---------------------------------------------------------------------------
#  Data‑class containers (unchanged API)
# ---------------------------------------------------------------------------
@dataclass
class RewardQualityMetrics:
    information_content: float
    entropy: float
    density: float
    discriminativity: float
    progressivity: float
    consistency: float

@dataclass
class LearningEfficiencyMetrics:
    sample_efficiency: float
    learning_speed: float
    convergence_efficiency: float
    exploration_efficiency: float

# ---------------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------------

def normalise(x: np.ndarray) -> np.ndarray:
    x = x - x.min(); rng = x.max() or 1
    return x / rng

# ---------------------------------------------------------------------------
#  Metrics based purely on *real* episode‑reward logs
# ---------------------------------------------------------------------------

def reward_quality_from_log(rewards: np.ndarray) -> RewardQualityMetrics:
    if len(rewards) < 5:
        return RewardQualityMetrics(*(0.0,)*6)

    # information content (Shannon entropy of 10‑bin histogram)
    hist,bins = np.histogram(rewards, bins=10)
    p = hist / hist.sum()
    p = p[p>0]
    info = -(p*np.log(p)).sum()/math.log(len(hist))

    density = (np.abs(rewards) > 1e-6).mean()

    # discriminativity = between‑bin variance / total variance
    bin_means=[rewards[(rewards>=bins[i])&(rewards<bins[i+1])].mean() for i in range(10)]
    disc = np.nanvar(bin_means) / (np.var(rewards)+1e-8)

    # progressivity = (last ⅓ avg − first ⅓ avg)/|first⅓|
    third=len(rewards)//3 or 1
    prog = max(0,(rewards[-third:].mean()-rewards[:third].mean())/(abs(rewards[:third].mean())+1e-8))

    # consistency = 1 / (1+var(reward diff))
    diff = np.diff(rewards)
    cons = 1/(1+np.var(diff))

    return RewardQualityMetrics(info, float(np.var(rewards)), density, disc, prog, cons)


def learning_efficiency_from_log(rewards: np.ndarray, exploration: np.ndarray) -> LearningEfficiencyMetrics:
    if len(rewards) < 10:
        return LearningEfficiencyMetrics(*(0.0,)*4)

    # sample efficiency: episodes to reach 95% of final avg
    target = 0.95*rewards[-100:].mean() if len(rewards)>=100 else 0.95*rewards.mean()
    idx = np.argmax(rewards>=target) or len(rewards)
    sample_eff = 1/(1+idx/len(rewards))

    # learning speed: slope of linear fit
    slope,_ = np.polyfit(np.arange(len(rewards)), rewards, 1)
    learn_spd = max(0,min(1,slope*len(rewards)/ (rewards.max()-rewards.min()+1e-8)))

    # convergence: variance of last quarter
    conv = 1/(1+np.var(rewards[-len(rewards)//4:]))

    # exploration efficiency: correlation(|exploration|, reward)
    if len(exploration)==len(rewards) and len(rewards)>2:
        corr = abs(np.corrcoef(exploration, rewards)[0,1])
    else:
        corr = 0.0
    return LearningEfficiencyMetrics(sample_eff, learn_spd, conv, corr)

# ---------------------------------------------------------------------------
#  Visualisations
# ---------------------------------------------------------------------------

def plot_reward_quality(human_r,llm_r,outdir:Path):
    plt.figure(figsize=(12,4))
    for seq,lbl,col in [(human_r,'Human','tab:blue'),(llm_r,'LLM','tab:orange')]:
        plt.hist(seq,bins=30,alpha=.6,label=lbl,density=True,color=col)
    plt.title('Reward distribution'); plt.legend(); plt.tight_layout()
    plt.savefig(outdir/'reward_quality_hist.png',dpi=300); plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(np.cumsum(human_r),label='Human',alpha=.8)
    plt.plot(np.cumsum(llm_r),label='LLM',alpha=.8)
    plt.title('Cumulative reward'); plt.legend(); plt.tight_layout()
    plt.savefig(outdir/'reward_cumulative.png',dpi=300); plt.close()


def plot_learning_efficiency(h_hist,llm_hist,outdir:Path):
    plt.figure(figsize=(10,4))
    plt.plot(h_hist,label='Human',alpha=.8)
    plt.plot(llm_hist,label='LLM',alpha=.8)
    plt.title('Per‑episode reward'); plt.legend(); plt.tight_layout()
    plt.savefig(outdir/'learning_efficiency_curve.png',dpi=300); plt.close()

# ---------------------------------------------------------------------------
#  Pipeline
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotations',default='processed_annotations.json')
    ap.add_argument('--human_csv',required=True)
    ap.add_argument('--llm_csv',required=True)
    ap.add_argument('--outdir',default='mdp/rl_tests/reward_efficiency_results')
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)

    # read logs
    h_log = pd.read_csv(args.human_csv)
    l_log = pd.read_csv(args.llm_csv)
    for col in ('reward','exploration_rate'):
        if col not in h_log: h_log[col]=0
        if col not in l_log: l_log[col]=0

    human_rqm = reward_quality_from_log(h_log['reward'].values)
    llm_rqm   = reward_quality_from_log(l_log['reward'].values)

    human_lem = learning_efficiency_from_log(h_log['reward'].values,h_log['exploration_rate'].values)
    llm_lem   = learning_efficiency_from_log(l_log['reward'].values,l_log['exploration_rate'].values)

    # visuals
    plot_reward_quality(h_log['reward'].values,l_log['reward'].values,outdir)
    plot_learning_efficiency(h_log['reward'].values,l_log['reward'].values,outdir)

    results = {
        'human_reward_quality': asdict(human_rqm),
        'llm_reward_quality':   asdict(llm_rqm),
        'human_learning_eff':  asdict(human_lem),
        'llm_learning_eff':    asdict(llm_lem),
    }
    json_path = outdir/'reward_efficiency_results.json'
    json_path.write_text(json.dumps(results,indent=2))
    print('✔ results saved →',json_path)

if __name__=='__main__':
    main()