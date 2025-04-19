"""
mdp_pipeline_minipilot.py
─────────────────────────
Mini‑pilot: MultiWOZ Restaurant 대화 ① GPT‑4o로 MDP 주석(병렬) → ② MDP 그래프 구축
→ ③ Dirichlet 스무딩으로 P(s′|s,a) 추정 → ④ Plotly 히트맵 저장.

USAGE
-----
python mdp_pipeline_minipilot.py dataset/train/dialogues_001.json
"""

import os, json, argparse, hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import beta
import plotly.express as px

# --------------------- CONFIG ---------------------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")      # .env에 저장
CLIENT      = OpenAI(api_key=OPENAI_KEY)

MODEL       = "gpt-4o"
TEMPERATURE = 0.1
DIRICHLET_ALPHA = 1.0      # smoothing prior
TOP_NEXT        = 3        # 히트맵에서 (state,action) 당 상위 next‑state 수
MAX_WORKERS     = 6        # 동시 GPT 호출 스레드 수
# --------------------------------------------------

PROMPT_TEMPLATE = """
You are given a task‑oriented dialogue.
For each turn output a JSON object with:
1) state_id       : SHA256 of the canonicalised belief_state (sorted slot–value pairs, lowercase)
2) speaker        : "user" | "system"
3) utterance      : original text
4) belief_state   : list of {domain, slots:[[slot,value],...]}
5) system_action  : main dialogue act(s)
6) transitions    : {state_1, action, state_2}  (previous → current)
Return a single JSON object keyed by "turn_1", "turn_2", ...
Output must be valid JSON **only**.
"""

# ────────────────── GPT 호출 ──────────────────
def gpt_annotate(text: str) -> dict:
    """Call GPT‑4o to annotate a single dialogue; return parsed dict."""
    resp = CLIENT.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": PROMPT_TEMPLATE},
            {"role": "user",   "content": f"Below is a conversation. Annotate:\n{text}"}
        ]
    )
    content = resp.choices[0].message.content
    content = content.replace("```", " ").replace("json", " ").strip()
    return json.loads(content)

# ────────────────── 그래프 처리 ──────────────────
def add_turn_to_graph(g: nx.DiGraph, s1: str, act: str, s2: str):
    if not g.has_node(s1):
        g.add_node(s1)
    if not g.has_node(s2):
        g.add_node(s2)
    if g.has_edge(s1, s2):
        g[s1][s2]["count"] += 1
    else:
        g.add_edge(s1, s2, action=act, count=1)

def process_dialogues_parallel(file_path: Path,
                               n_dialogues: int = 300,
                               max_workers: int = MAX_WORKERS) -> nx.DiGraph:
    """Annotate dialogues in parallel threads and build a merged MDP graph."""
    with open(file_path) as f:
        data = json.load(f)
    dialogs = data[:n_dialogues]

    def _annotate_single(dlg):
        text = "\n".join(turn["utterance"] for turn in dlg["turns"])
        try:
            return gpt_annotate(text)
        except Exception as e:
            print("GPT error:", e)
            return None

    graph = nx.DiGraph()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_annotate_single, dlg): dlg for dlg in dialogs}
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Annotating (parallel)"):
            ann = fut.result()
            if ann is None:
                continue
            for turn in ann.values():
                s1 = turn["transitions"]["state_1"]
                s2 = turn["transitions"]["state_2"]
                act = turn["transitions"]["action"]
                add_turn_to_graph(graph, s1, act, s2)
    return graph

# ───── Dirichlet 스무딩으로 확률 & 95 % CI ─────
def apply_dirichlet(graph: nx.DiGraph, alpha: float = DIRICHLET_ALPHA):
    out_totals = defaultdict(int)
    for u, v, d in graph.edges(data=True):
        out_totals[u] += d["count"]

    for u, v, d in graph.edges(data=True):
        denom = out_totals[u] + alpha * len(graph.successors(u))
        d["prob"] = (d["count"] + alpha) / denom
        a = d["count"] + alpha
        b = denom - a
        d["ci_low"], d["ci_high"] = beta.interval(0.95, a, b)

# ───────────── 히트맵 시각화 ─────────────
def build_heatmap(graph: nx.DiGraph,
                  out_html: str = "transition_heatmap.html",
                  top_n: int = TOP_NEXT):
    rows = [
        {
            "state": u,
            "action": d["action"],
            "next": v,
            "prob": d["prob"],
            "ci": d["ci_high"] - d["ci_low"],
        }
        for u, v, d in graph.edges(data=True)
    ]
    df = pd.DataFrame(rows)
    df_top = (
        df.groupby(["state", "action"])
          .apply(lambda x: x.nlargest(top_n, "prob"))
          .reset_index(drop=True)
    )
    fig = px.density_heatmap(
        df_top,
        x="action",
        y="state",
        z="prob",
        color_continuous_scale="Viridis",
        hover_data=["next", "ci"],
        height=800,
    )
    fig.write_html(out_html)
    print(f"Heatmap saved to {out_html}")

# ────────────────── Main ──────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Path to MultiWOZ Restaurant JSON file")
    parser.add_argument("--dialogs", type=int, default=300,
                        help="Number of dialogues to process (default 300)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help="Parallel GPT threads (default 6)")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"{args.data} not found")

    graph = process_dialogues_parallel(Path(args.data),
                                       n_dialogues=args.dialogs,
                                       max_workers=args.workers)
    apply_dirichlet(graph)
    nx.write_graphml(graph, "mdp_graph.graphml")
    build_heatmap(graph)

if __name__ == "__main__":
    main()
