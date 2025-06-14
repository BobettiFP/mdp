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
    """엘보우 방법, 실루엣 스코어, 칼린스키-하라바즈 지수를 종합한 최적 클러스터 수 결정"""
    if len(X) < 3:
        return 1
    
    # 고유한 상태 수 기반으로 더 현실적인 최대 클러스터 수 설정
    unique_states = len(np.unique(X, axis=0))
    max_k = max_clusters or min(max(unique_states // 3, 8), unique_states - 1)
    max_k = max(2, min(max_k, len(X) - 1))
    
    if max_k < 2:
        return 1
    
    K_range = range(2, max_k + 1)
    silhouette_scores = []
    inertias = []
    calinski_scores = []
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = kmeans.fit_predict(X)
        
        # 실루엣 스코어
        sil_score = silhouette_score(X, labels)
        silhouette_scores.append(sil_score)
        
        # 엘보우 방법을 위한 inertia
        inertias.append(kmeans.inertia_)
        
        # 칼린스키-하라바즈 지수
        from sklearn.metrics import calinski_harabasz_score
        ch_score = calinski_harabasz_score(X, labels)
        calinski_scores.append(ch_score)
    
    if not silhouette_scores:
        return 2
    
    # 엘보우 방법으로 급격한 변화점 찾기
    def find_elbow(inertias):
        if len(inertias) < 3:
            return 0
        
        # 2차 차분으로 변곡점 찾기
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        if len(second_diffs) == 0:
            return 0
        return np.argmax(second_diffs) + 2  # K_range 시작이 2이므로
    
    # 각 방법의 추천값
    sil_optimal = K_range[np.argmax(silhouette_scores)]
    elbow_k = find_elbow(inertias)
    ch_optimal = K_range[np.argmax(calinski_scores)]
    
    # 가중 평균으로 최종 결정 (실루엣 스코어에 더 큰 가중치)
    candidates = []
    if sil_optimal: candidates.extend([sil_optimal] * 3)  # 실루엣 스코어 3배 가중
    if elbow_k and 2 <= elbow_k <= max_k: candidates.append(elbow_k)
    if ch_optimal: candidates.append(ch_optimal)
    
    if candidates:
        # 가장 빈번한 값 선택, 동률이면 중간값
        from collections import Counter
        counter = Counter(candidates)
        most_common = counter.most_common()
        
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # 동률이면 더 큰 클러스터 수 선택 (더 세분화)
            optimal_k = max([k for k, count in most_common if count == most_common[0][1]])
        else:
            optimal_k = most_common[0][0]
        
        # 최소 3개 클러스터 보장 (단순한 이진 분류 방지)
        optimal_k = max(3, min(optimal_k, max_k))
        return optimal_k
    
    return max(3, min(max_k, len(K_range) // 2 + 2))

def classify_states(X, recs, optimal_k):
    """다중 방법으로 상태들을 클러스터링하고 각 클러스터의 대표 상태 선택"""
    if len(X) < 2:
        return {
            'labels': np.zeros(len(X)),
            'representatives': [0] if len(X) > 0 else [],
            'cluster_info': {},
            'n_clusters': 1 if len(X) > 0 else 0,
            'method': 'single_state'
        }
    
    # 여러 클러스터링 방법 시도
    methods = {}
    
    # 1. K-Means
    kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=0)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_score = silhouette_score(X, kmeans_labels) if len(np.unique(kmeans_labels)) > 1 else -1
    methods['kmeans'] = (kmeans_labels, kmeans_score, kmeans.cluster_centers_)
    
    # 2. 계층적 클러스터링
    try:
        from sklearn.cluster import AgglomerativeClustering
        agg = AgglomerativeClustering(n_clusters=optimal_k)
        agg_labels = agg.fit_predict(X)
        agg_score = silhouette_score(X, agg_labels) if len(np.unique(agg_labels)) > 1 else -1
        methods['hierarchical'] = (agg_labels, agg_score, None)
    except:
        pass
    
    # 3. DBSCAN (밀도 기반)
    try:
        from sklearn.cluster import DBSCAN
        # eps 자동 조정
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=4)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        eps = np.percentile(distances[:, -1], 90)  # 90퍼센타일을 eps로 사용
        
        dbscan = DBSCAN(eps=eps, min_samples=max(3, len(X) // 50))
        dbscan_labels = dbscan.fit_predict(X)
        
        # 노이즈 포인트(-1)가 너무 많으면 제외
        if len(np.unique(dbscan_labels[dbscan_labels != -1])) >= 2:
            dbscan_score = silhouette_score(X[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
            methods['dbscan'] = (dbscan_labels, dbscan_score, None)
    except:
        pass
    
    # 최고 점수의 방법 선택
    best_method = 'kmeans'
    best_score = methods['kmeans'][1]
    
    for method_name, (labels, score, centers) in methods.items():
        if score > best_score:
            best_method = method_name
            best_score = score
    
    labels, _, centroids = methods[best_method]
    
    # DBSCAN의 경우 노이즈 포인트 처리
    if best_method == 'dbscan':
        # 노이즈 포인트들을 가장 가까운 클러스터에 할당
        noise_mask = labels == -1
        if np.any(noise_mask):
            valid_labels = labels[~noise_mask]
            valid_points = X[~noise_mask]
            noise_points = X[noise_mask]
            
            # 각 노이즈 포인트를 가장 가까운 클러스터에 할당
            for i, noise_point in enumerate(noise_points):
                distances_to_clusters = []
                for cluster_id in np.unique(valid_labels):
                    cluster_points = valid_points[valid_labels == cluster_id]
                    min_dist = np.min(cdist([noise_point], cluster_points))
                    distances_to_clusters.append(min_dist)
                
                closest_cluster = np.unique(valid_labels)[np.argmin(distances_to_clusters)]
                labels[noise_mask][i] = closest_cluster
    
    # 클러스터 중심 계산 (centroids가 없는 경우)
    if centroids is None:
        unique_labels = np.unique(labels)
        centroids = []
        for cluster_id in unique_labels:
            cluster_points = X[labels == cluster_id]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
    
    # 각 클러스터의 대표 상태 선택
    representatives = []
    cluster_info = {}
    actual_clusters = len(np.unique(labels))
    
    for i, cluster_id in enumerate(np.unique(labels)):
        cluster_mask = labels == cluster_id
        cluster_points = X[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_points) == 0:
            continue
            
        # centroid와의 거리 계산
        if len(centroids) > i:
            distances = cdist([centroids[i]], cluster_points, metric='euclidean')[0]
        else:
            # centroids가 부족한 경우 클러스터 중심 직접 계산
            cluster_center = np.mean(cluster_points, axis=0)
            distances = cdist([cluster_center], cluster_points, metric='euclidean')[0]
        
        rep_idx_in_cluster = np.argmin(distances)
        rep_idx_global = cluster_indices[rep_idx_in_cluster]
        
        representatives.append(rep_idx_global)
        
        # 클러스터 내 다양성 계산
        intra_cluster_distances = cdist(cluster_points, cluster_points)
        diversity = np.mean(intra_cluster_distances[np.triu_indices_from(intra_cluster_distances, k=1)])
        
        # 클러스터 정보 저장
        cluster_info[f'cluster_{cluster_id}'] = {
            'size': int(np.sum(cluster_mask)),
            'representative_idx': int(rep_idx_global),
            'representative_state': recs[rep_idx_global]['state_after'],
            'centroid_distance': float(distances[rep_idx_in_cluster]),
            'avg_distance_to_centroid': float(np.mean(distances)),
            'intra_cluster_diversity': float(diversity) if not np.isnan(diversity) else 0.0
        }
    
    return {
        'labels': labels,
        'representatives': representatives,
        'cluster_info': cluster_info,
        'n_clusters': actual_clusters,
        'centroids': centroids,
        'method': best_method,
        'silhouette_score': best_score
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
    print(f"   └ 방법: {classification_h['method']}, {classification_h['n_clusters']}개 클러스터, "
          f"실루엣 스코어: {classification_h.get('silhouette_score', 0):.3f}")

    print("🔍 LLM states 클러스터링 중...")
    optimal_k_l = find_optimal_clusters(Xl, a.max_clusters)
    classification_l = classify_states(Xl, L, optimal_k_l)
    print(f"   └ 방법: {classification_l['method']}, {classification_l['n_clusters']}개 클러스터, "
          f"실루엣 스코어: {classification_l.get('silhouette_score', 0):.3f}")

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
                "method": classification_h.get('method', 'unknown'),
                "silhouette_score": float(classification_h.get('silhouette_score', 0)),
                "n_clusters": int(classification_h['n_clusters']),
                "cluster_info": {k: {
                    **{kk: (vv if not isinstance(vv, dict) else vv) for kk, vv in v.items()},
                    "representative_state": v["representative_state"]
                } for k, v in classification_h['cluster_info'].items()},
                "representative_indices": [int(x) for x in classification_h['representatives']]
            }
        },
        "llm": {
            "richness_metrics": {k:_to_py(v) for k,v in Rl.items()},
            "clustering": {
                "method": classification_l.get('method', 'unknown'),
                "silhouette_score": float(classification_l.get('silhouette_score', 0)),
                "n_clusters": int(classification_l['n_clusters']),
                "cluster_info": {k: {
                    **{kk: (vv if not isinstance(vv, dict) else vv) for kk, vv in v.items()},
                    "representative_state": v["representative_state"]
                } for k, v in classification_l['cluster_info'].items()},
                "representative_indices": [int(x) for x in classification_l['representatives']]
            }
        }
    }
    
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("✔ saved →",out)
    
    # 대표 상태들 요약 출력
    print(f"\n📊 Human 대표 상태들 ({classification_h['method']} 방법):")
    for i, (cluster_name, info) in enumerate(classification_h['cluster_info'].items()):
        diversity = info.get('intra_cluster_diversity', 0)
        print(f"  {cluster_name} (크기: {info['size']}, 다양성: {diversity:.3f}): {info['representative_state']}")
    
    print(f"\n📊 LLM 대표 상태들 ({classification_l['method']} 방법):")
    for i, (cluster_name, info) in enumerate(classification_l['cluster_info'].items()):
        diversity = info.get('intra_cluster_diversity', 0)
        print(f"  {cluster_name} (크기: {info['size']}, 다양성: {diversity:.3f}): {info['representative_state']}")
    
    print(f"\n🎯 클러스터링 품질:")
    print(f"  Human - 실루엣 스코어: {classification_h.get('silhouette_score', 0):.3f}")
    print(f"  LLM   - 실루엣 스코어: {classification_l.get('silhouette_score', 0):.3f}")

if __name__=="__main__":
    np.random.seed(0); random.seed(0); main()