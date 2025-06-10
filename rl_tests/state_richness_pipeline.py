#!/usr/bin/env python3
"""
State-Richness pipeline rev 4 (state classification 및 대표 state 추출 포함)
"""
import argparse, json, random, os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.stats import entropy
from scipy.spatial.distance import cdist

# --------------------- util --------------------------------------------------
def _to_py(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x

def vecs(recs: List[dict], slot2i: Dict[str,int], val2i: Dict[str,int]):
    dim=len(slot2i)+len(val2i); arr=[]
    for r in recs:
        v=np.zeros(dim,dtype=np.int8)
        for k in r["state_after"]: v[slot2i[k]]=1
        for vstr in map(str, r["state_after"].values()):
            v[len(slot2i)+val2i[vstr]]=1
        arr.append(v)
    return np.vstack(arr)

def find_optimal_clusters(X, max_clusters=None):
    """엘보우 방법과 실루엣 스코어를 통해 최적 클러스터 수 결정"""
    if len(X) < 3:
        return 1
    
    max_k = max_clusters or min(10, len(X) // 2)
    max_k = max(2, min(max_k, len(X) - 1))
    
    if max_k < 2:
        return 1
    
    # 실루엣 스코어로 최적 클러스터 수 찾기
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=0)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    if silhouette_scores:
        optimal_k = K_range[np.argmax(silhouette_scores)]
        return optimal_k
    
    return 2

def classify_states(X, recs, optimal_k):
    """상태들을 클러스터링하고 각 클러스터의 대표 상태 선택"""
    if len(X) < 2:
        return {
            'labels': np.zeros(len(X)),
            'representatives': [0] if len(X) > 0 else [],
            'cluster_info': {},
            'n_clusters': 1 if len(X) > 0 else 0
        }
    
    # 클러스터링 수행
    kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=0)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    # 각 클러스터의 대표 상태 선택 (centroid에 가장 가까운 실제 상태)
    representatives = []
    cluster_info = {}
    
    for cluster_id in range(optimal_k):
        cluster_mask = labels == cluster_id
        cluster_points = X[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_points) == 0:
            continue
            
        # centroid와의 거리 계산
        distances = cdist([centroids[cluster_id]], cluster_points, metric='euclidean')[0]
        rep_idx_in_cluster = np.argmin(distances)
        rep_idx_global = cluster_indices[rep_idx_in_cluster]
        
        representatives.append(rep_idx_global)
        
        # 클러스터 정보 저장
        cluster_info[f'cluster_{cluster_id}'] = {
            'size': int(np.sum(cluster_mask)),
            'representative_idx': int(rep_idx_global),
            'representative_state': recs[rep_idx_global]['state_after'],
            'centroid_distance': float(distances[rep_idx_in_cluster]),
            'avg_distance_to_centroid': float(np.mean(distances))
        }
    
    return {
        'labels': labels,
        'representatives': representatives,
        'cluster_info': cluster_info,
        'n_clusters': optimal_k,
        'centroids': centroids
    }

def richness(X, slot2i, val2i):
    uniq=len(np.unique(X,axis=0))
    ent=entropy(list(Counter(map(bytes,X)).values()))
    cov=uniq/len(X)
    pca=PCA().fit(X)
    eff=(pca.explained_variance_ratio_.cumsum()>=0.95).argmax()+1
    dens=sep=0.0
    if uniq>10:
        km=KMeans(n_clusters=min(10,uniq//2),n_init=5,random_state=0).fit(X)
        dens=silhouette_score(X,km.labels_)
    if uniq>1:
        smp=np.unique(X,axis=0)[:1500]
        sep=float(np.linalg.norm(smp[:,None]-smp,axis=-1).mean())
    return dict(unique_states=uniq,state_entropy=ent,coverage_ratio=cov,
                effective_dim=eff,density_uniformity=dens,cluster_separation=sep,
                slot_vocab=len(slot2i),value_vocab=len(val2i))

def tsne_fig_with_clusters(X, labels, representatives, title, path):
    """클러스터링 결과를 포함한 t-SNE 시각화"""
    if len(X)<3: return
    try: 
        emb=TSNE(n_components=2,random_state=0,perplexity=min(30,len(X)//3)).fit_transform(X)
    except: 
        emb=PCA(n_components=2).fit_transform(X)
    
    plt.figure(figsize=(8,6))
    
    # 클러스터별로 다른 색상으로 표시
    if labels is not None and len(np.unique(labels)) > 1:
        scatter = plt.scatter(emb[:,0], emb[:,1], c=labels, s=20, alpha=0.6, cmap='tab10')
        plt.colorbar(scatter, label='Cluster')
        
        # 대표 상태들을 강조 표시
        if representatives:
            rep_emb = emb[representatives]
            plt.scatter(rep_emb[:,0], rep_emb[:,1], c='red', s=100, marker='*', 
                       edgecolors='black', linewidth=1, label='Representatives', alpha=0.9)
            
            # 대표 상태에 번호 표시
            for i, (x, y) in enumerate(rep_emb):
                plt.annotate(f'R{i}', (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold', color='white')
        
        plt.legend()
    else:
        plt.scatter(emb[:,0], emb[:,1], s=20, alpha=0.6)
    
    plt.title(f'{title} State Space (Clustered)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def tsne_fig(X,title,path):
    """기존 t-SNE 시각화 (호환성 유지)"""
    tsne_fig_with_clusters(X, None, None, title, path)

# --------------------- main --------------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--annotations",required=True)
    ap.add_argument("--outdir",default="mdp/rl_tests/state_richness_results")
    ap.add_argument("--max_clusters", type=int, default=None, 
                    help="최대 클러스터 수 (None이면 자동 결정)")
    a=ap.parse_args(); Path(a.outdir).mkdir(parents=True,exist_ok=True)

    data=json.load(open(a.annotations))
    recs=data["annotations"] if isinstance(data,dict) else data
    H=[r for r in recs if r.get("annotation_type")=="human"]
    L=[r for r in recs if r.get("annotation_type")=="llm"]
    if not H or not L: raise RuntimeError("annotation_type 라벨이 부족합니다.")

    slotH={s:i for i,s in enumerate(sorted({k for r in H for k in r["state_after"]}))}
    valH ={v:i for i,v in enumerate(sorted({str(x) for r in H for x in r["state_after"].values()}))}
    slotL={s:i for i,s in enumerate(sorted({k for r in L for k in r["state_after"]}))}
    valL ={v:i for i,v in enumerate(sorted({str(x) for r in L for x in r["state_after"].values()}))}

    Xh,Xl=vecs(H,slotH,valH),vecs(L,slotL,valL)
    Rh,Rl=richness(Xh,slotH,valH),richness(Xl,slotL,valL)

    # 상태 분류 및 대표 상태 추출
    print("🔍 Human states 클러스터링 중...")
    optimal_k_h = find_optimal_clusters(Xh, a.max_clusters)
    classification_h = classify_states(Xh, H, optimal_k_h)
    print(f"   └ {optimal_k_h}개 클러스터 발견, {len(classification_h['representatives'])}개 대표 상태 선택")

    print("🔍 LLM states 클러스터링 중...")
    optimal_k_l = find_optimal_clusters(Xl, a.max_clusters)
    classification_l = classify_states(Xl, L, optimal_k_l)
    print(f"   └ {optimal_k_l}개 클러스터 발견, {len(classification_l['representatives'])}개 대표 상태 선택")

    # 클러스터링 결과 시각화
    tsne_fig_with_clusters(Xh, classification_h['labels'], classification_h['representatives'],
                          "Human", Path(a.outdir)/"state_space_human_clustered.png")
    tsne_fig_with_clusters(Xl, classification_l['labels'], classification_l['representatives'],
                          "LLM", Path(a.outdir)/"state_space_llm_clustered.png")

    # 기존 시각화도 유지
    tsne_fig(Xh,"Human",Path(a.outdir)/"state_space_human.png")
    tsne_fig(Xl,"LLM",  Path(a.outdir)/"state_space_llm.png")

    # bar plot (기존)
    feat=["unique_states","coverage_ratio","effective_dim"]
    plt.figure(figsize=(5,3))
    x=np.arange(len(feat));w=.35
    plt.bar(x-w/2,[Rh[f] for f in feat],w,label="Human")
    plt.bar(x+w/2,[Rl[f] for f in feat],w,label="LLM")
    plt.xticks(x,feat); plt.legend(); plt.tight_layout()
    plt.savefig(Path(a.outdir)/"richness_compare.png",dpi=300); plt.close()

    # 클러스터 수 비교 차트
    plt.figure(figsize=(6,4))
    categories = ['Human', 'LLM']
    cluster_counts = [classification_h['n_clusters'], classification_l['n_clusters']]
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = plt.bar(categories, cluster_counts, color=colors, alpha=0.7)
    plt.ylabel('Number of Clusters')
    plt.title('State Clusters Comparison')
    plt.ylim(0, max(cluster_counts) * 1.2)
    
    # 막대 위에 숫자 표시
    for bar, count in zip(bars, cluster_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(Path(a.outdir)/"cluster_comparison.png", dpi=300)
    plt.close()

    # 결과 저장
    out=Path(a.outdir)/"state_richness_metrics.json"
    results = {
        "human": {
            "richness_metrics": {k:_to_py(v) for k,v in Rh.items()},
            "clustering": {
                "n_clusters": int(classification_h['n_clusters']),
                "cluster_info": {k: {
                    **v, 
                    "representative_state": v["representative_state"]
                } for k, v in classification_h['cluster_info'].items()},
                "representative_indices": [int(x) for x in classification_h['representatives']]
            }
        },
        "llm": {
            "richness_metrics": {k:_to_py(v) for k,v in Rl.items()},
            "clustering": {
                "n_clusters": int(classification_l['n_clusters']),
                "cluster_info": {k: {
                    **v,
                    "representative_state": v["representative_state"]
                } for k, v in classification_l['cluster_info'].items()},
                "representative_indices": [int(x) for x in classification_l['representatives']]
            }
        }
    }
    
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("✔ saved →",out)
    
    # 대표 상태들 요약 출력
    print("\n📊 Human 대표 상태들:")
    for i, (cluster_name, info) in enumerate(classification_h['cluster_info'].items()):
        print(f"  {cluster_name} (크기: {info['size']}): {info['representative_state']}")
    
    print("\n📊 LLM 대표 상태들:")
    for i, (cluster_name, info) in enumerate(classification_l['cluster_info'].items()):
        print(f"  {cluster_name} (크기: {info['size']}): {info['representative_state']}")

if __name__=="__main__":
    np.random.seed(0); random.seed(0); main()