import json, networkx as nx, powerlaw, collections
from tqdm import tqdm

DATA = "mdp/annotation_results/results_first10_try5.json"

g = nx.DiGraph()           # 방향 MDP 그래프

with open(DATA, encoding="utf-8") as f:
    for dlg in json.load(f):
        for t in dlg["annotation"].values():
            src = t["state_id_prev"]
            dst = t["state_id_next"]
            # action 하나만 가정. 여러 개면 loop 돌려도 됨
            act = t["action"][0]["type"] if t["action"] else "none"

            if g.has_edge(src, dst):
                g[src][dst]["weight"] += 1
            else:
                g.add_edge(src, dst,
                           weight=1,
                           action=collections.Counter())
            g[src][dst]["action"][act] += 1

# -------- 기본 지표 --------
num_nodes = g.number_of_nodes()
num_edges = g.number_of_edges()
avg_deg   = sum(dict(g.out_degree()).values()) / num_nodes
density   = nx.density(g)
print(f"nodes {num_nodes}, edges {num_edges}, avg_deg {avg_deg:.2f}, density {density:.5f}")

# -------- 군집계수 --------
clust = nx.average_clustering(g.to_undirected())
print("average clustering", clust)

# -------- in-degree 분포 및 파워 로 우 검사 --------
deg_seq = [d for _, d in g.in_degree()]
fit = powerlaw.Fit(deg_seq, discrete=True, verbose=False)
print("power-law alpha", fit.power_law.alpha,
      "xmin", fit.power_law.xmin)

# -------- 상위 action 빈도 --------
edge_actions = collections.Counter()
for _, _, d in g.edges(data=True):
    edge_actions += d["action"]
print("top actions:", edge_actions.most_common(10)[:5])
