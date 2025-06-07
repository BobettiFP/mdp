"""
Reward‑Efficiency Pipeline (rev 2)
---------------------------------
Accepts both kebab‑case and snake‑case CLI options.
CLI Example (one line):
python rl_tests/reward_efficiency_pipeline.py --annotations processed_annotations.json --human-log logs/human_env.csv --llm-log logs/llm_env.csv --outdir mdp/rl_tests/reward_efficiency_results
"""
import argparse, os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# ---------- argument parsing ------------------------------------------------
pa = argparse.ArgumentParser()
pa.add_argument("--annotations", default="processed_annotations.json")
pa.add_argument("--human-log", dest="human_log")      # kebab‑case
pa.add_argument("--human_log", dest="human_log")       # snake‑case alias
pa.add_argument("--llm-log",   dest="llm_log")
pa.add_argument("--llm_log",   dest="llm_log")
pa.add_argument("--outdir",    dest="outdir", default="reward_efficiency_results")
pa.add_argument("--out",       dest="outdir")          # backward compat
args = pa.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# ---------- load CSV --------------------------------------------------------
h_df = pd.read_csv(args.human_log)
l_df = pd.read_csv(args.llm_log)

# ---------- helper functions -----------------------------------------------
def describe_rewards(df):
    return {
        "mean":   df["reward"].mean(),
        "std":    df["reward"].std(ddof=0),
        "median": df["reward"].median(),
        "max":    df["reward"].max(),
    }

def sample_efficiency(df, thr=0.8):
    target = thr * df["reward"].max()
    idx = df[ df["reward"] >= target ].index
    return float(idx.min()+1) if not idx.empty else float("inf")

# ---------- compute metrics -------------------------------------------------
report = {
    "human": {
        "reward_stats": describe_rewards(h_df),
        "sample_efficiency": sample_efficiency(h_df),
    },
    "llm": {
        "reward_stats": describe_rewards(l_df),
        "sample_efficiency": sample_efficiency(l_df),
    }
}

# ---------- save JSON -------------------------------------------------------
out_json = os.path.join(args.outdir, "reward_efficiency_results.json")
json.dump(report, open(out_json, "w"), indent=2)
print("✔ JSON written →", out_json)

# ---------- plots -----------------------------------------------------------
plt.figure();
plt.plot(h_df.reward.rolling(20).mean(), label="Human 20‑ep MA")
plt.plot(l_df.reward.rolling(20).mean(), label="LLM 20‑ep MA")
plt.xlabel("Episode"); plt.ylabel("Reward 20‑MA"); plt.legend()
plt.title("Reward Curve (smoothed)")
plt.savefig(os.path.join(args.outdir, "learning_efficiency.png"), dpi=150)
print("✔ plot → learning_efficiency.png")
