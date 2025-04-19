"""
build_graph.py
❱ python build_graph.py annotated_turns.ldjson.gz --outdir ./graph_out
"""

import gzip, csv, json, argparse, pickle, os
from collections import defaultdict
import networkx as nx
from tqdm import tqdm

# ----------------- 2.1  그래프 초기화 -----------------
G = nx.MultiDiGraph()          # 방향 그래프

def add_edge(prev_id: str, next_id: str, action: str,
             user_utt: str, sys_utt: str):
    """
    노드·엣지 삽입 및 속성 누적
    """
    if not G.has_node(prev_id):
        G.add_node(prev_id)
    if not G.has_node(next_id):
        G.add_node(next_id)

    if G.has_edge(prev_id, next_id, key=action):
        # 멀티에지(동일 action)를 1개만 쓰고 count만 올린다
        G[prev_id][next_id][action]["count"] += 1
    else:
        G.add_edge(
            prev_id,
            next_id,
            key=action,
            count=1,
            example_user=user_utt,
            example_system=sys_utt
        )

# ----------------- 2.2  LDJSON 스트림 파싱 -----------------
def build_graph_from_ldjson(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Building graph"):
            turn = json.loads(line)
            tr = turn["transition"]
            add_edge(
                tr["prev_state_id"],
                tr["next_state_id"],
                tr["action"],
                turn["utterances"]["user"],
                turn["utterances"]["system"]
            )

# ----------------- 2.3  직렬화 -----------------
def save_outputs(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # ① Pickle
    with open(os.path.join(out_dir, "graph.pkl"), "wb") as fp:
        pickle.dump(G, fp)
    print("[✓] Pickle saved")

    # ② GraphML (node/edge 데이터 시각 툴 호환)
    nx.write_graphml(G, os.path.join(out_dir, "graph.graphml"))
    print("[✓] GraphML saved")

    # ③ CSV edge list
    with open(os.path.join(out_dir, "edge_list.csv"), "w", newline='', encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["src", "dst", "action", "count",
                         "example_user", "example_system"])
        for u, v, k, data in G.edges(keys=True, data=True):
            writer.writerow([
                u, v, k, data["count"],
                data["example_user"], data["example_system"]
            ])
    print("[✓] CSV edge list saved")

# ----------------- main -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ldjson_gz", help="annotated_turns.ldjson.gz")
    parser.add_argument("--outdir", default="graph_outputs")
    args = parser.parse_args()

    build_graph_from_ldjson(args.ldjson_gz)
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    save_outputs(args.outdir)
