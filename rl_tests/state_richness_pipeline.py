#!/usr/bin/env python3
"""
State-Richness pipeline rev 5 (완전 설정 가능 + UMAP)
모든 하드코딩 제거, UMAP 기본 사용, 완전 설정 가능
"""
import argparse, json, random, os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 백엔드 설정으로 폰트 문제 방지
plt.rcParams['font.family'] = 'DejaVu Sans'  # 영어 폰트 강제 설정
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.stats import entropy
from scipy.spatial.distance import cdist

# 동적 import - 사용 가능한 차원 축소 방법들
AVAILABLE_METHODS = ['pca']
try:
    import umap
    AVAILABLE_METHODS.append('umap')
except ImportError:
    pass

try:
    from sklearn.manifold import TSNE
    AVAILABLE_METHODS.append('tsne')
except ImportError:
    pass

class ConfigurableStateAnalyzer:
    """완전히 설정 가능한 상태 분석기"""
    
    def __init__(self, config: dict = None):
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
    
    def _load_default_config(self) -> dict:
        """기본 설정 - 하드코딩 없이 적응적"""
        return {
            # 샘플링 설정
            'max_samples_clustering': None,  # None = 자동 결정
            'max_samples_visualization': None,  # None = 자동 결정
            'sampling_strategy': 'smart',  # 'smart', 'random', 'stratified'
            
            # 차원 축소 설정
            'optimal_dimensions': True,  # 최적 차원 자동 결정
            'dimension_method': 'pca',  # 'pca', 'elbow' 
            'variance_threshold': 0.95,  # PCA 분산 설명 임계값
            'max_components': None,  # None = 자동 결정
            'min_components': 3,  # 최소 차원 수
            
            # 클러스터링 설정
            'clustering_methods': ['kmeans'],  # ['kmeans', 'hierarchical', 'dbscan']
            'cluster_range': 'auto',  # 'auto' or [min, max]
            'max_clusters_search': None,  # None = 자동 결정
            'use_reduced_dims_for_clustering': True,  # 차원 축소 후 클러스터링
            'kmeans_config': {
                'n_init': 'auto',  # 'auto' = 데이터 크기에 따라
                'max_iter': 'auto',
                'tol': 1e-4
            },
            
            # 시각화 설정 - 모든 방법 실행
            'generate_all_visualizations': True,  # 모든 차원 축소 방법 실행
            'visualization_methods': ['pca', 'umap', 'tsne'],  # 실행할 모든 방법
            'fallback_method': 'pca',
            'umap_config': {
                'n_neighbors': 'auto',  # 'auto' = sqrt(n_samples)
                'min_dist': 0.1,
                'metric': 'euclidean',
                'n_epochs': 'auto'
            },
            'tsne_config': {
                'perplexity': 'auto',  # 'auto' = min(30, n_samples/4)
                'max_iter': 'auto',  # 기존 n_iter을 max_iter로 변경
                'early_exaggeration': 12.0
            },
            'pca_config': {
                'n_components': 2  # 시각화는 항상 2D
            },
            
            # 성능 설정
            'performance_mode': 'balanced',  # 'fast', 'balanced', 'accurate'
            'adaptive_sampling': True,
            'parallel_processing': True,
            
            # 출력 설정
            'plot_dpi': 200,
            'save_formats': ['png'],  # ['png', 'pdf', 'svg']
            'include_sample_info': True
        }
    
    def _auto_configure_for_data_size(self, n_samples: int):
        """데이터 크기에 따른 자동 설정"""
        
        # 샘플링 크기 자동 결정
        if self.config['max_samples_clustering'] is None:
            if n_samples > 50000:
                self.config['max_samples_clustering'] = 15000
            elif n_samples > 10000:
                self.config['max_samples_clustering'] = 8000
            else:
                self.config['max_samples_clustering'] = n_samples
        
        if self.config['max_samples_visualization'] is None:
            if n_samples > 10000:
                self.config['max_samples_visualization'] = 5000  # 5000개로 증가
            elif n_samples > 5000:
                self.config['max_samples_visualization'] = 2000
            else:
                self.config['max_samples_visualization'] = n_samples
        
        # 클러스터 탐색 범위 자동 결정
        if self.config['max_clusters_search'] is None:
            unique_estimate = min(n_samples // 10, 20)  # 추정
            self.config['max_clusters_search'] = max(5, min(unique_estimate, 15))
        
        # K-means 설정 자동 조정
        if self.config['kmeans_config']['n_init'] == 'auto':
            if n_samples > 10000:
                self.config['kmeans_config']['n_init'] = 5
            elif n_samples > 1000:
                self.config['kmeans_config']['n_init'] = 10
            else:
                self.config['kmeans_config']['n_init'] = 20
        
        if self.config['kmeans_config']['max_iter'] == 'auto':
            if n_samples > 10000:
                self.config['kmeans_config']['max_iter'] = 100
            else:
                self.config['kmeans_config']['max_iter'] = 300
        
        # UMAP 설정 자동 조정
        if self.config['umap_config']['n_neighbors'] == 'auto':
            self.config['umap_config']['n_neighbors'] = max(5, min(int(np.sqrt(n_samples)), 50))
        
        if self.config['umap_config']['n_epochs'] == 'auto':
            if n_samples > 10000:
                self.config['umap_config']['n_epochs'] = 200
            else:
                self.config['umap_config']['n_epochs'] = 500
        
        # t-SNE 설정 자동 조정
        if self.config['tsne_config']['perplexity'] == 'auto':
            self.config['tsne_config']['perplexity'] = min(30, max(5, n_samples // 4))
        
        if self.config['tsne_config']['max_iter'] == 'auto':
            if n_samples > 5000:
                self.config['tsne_config']['max_iter'] = 250
            else:
                self.config['tsne_config']['max_iter'] = 1000
        
        # 성능 모드별 추가 조정
        if self.config['performance_mode'] == 'fast':
            self.config['max_samples_clustering'] = min(self.config['max_samples_clustering'], 5000)
            self.config['max_samples_visualization'] = min(self.config['max_samples_visualization'], 3000)  # 증가
            self.config['max_clusters_search'] = min(self.config['max_clusters_search'], 8)
            # 빠른 모드에서는 t-SNE 제외
            self.config['visualization_methods'] = [m for m in self.config['visualization_methods'] if m != 'tsne']
        elif self.config['performance_mode'] == 'accurate':
            self.config['max_samples_clustering'] = min(n_samples, 20000)
            self.config['max_samples_visualization'] = min(n_samples, 8000)  # 증가
            # 정확 모드에서는 모든 방법 사용
            if 'tsne' not in self.config['visualization_methods']:
                self.config['visualization_methods'].append('tsne')

def smart_sample(X, recs, max_samples, strategy='smart', seed=42):
    """다양한 샘플링 전략"""
    if len(X) <= max_samples:
        return X, recs, np.arange(len(X))
    
    np.random.seed(seed)
    
    if strategy == 'smart':
        # 유니크한 상태들을 우선적으로 포함
        unique_indices = []
        seen_states = set()
        
        for i, state_vec in enumerate(X):
            state_key = tuple(state_vec)
            if state_key not in seen_states:
                unique_indices.append(i)
                seen_states.add(state_key)
        
        # 유니크 상태 + 랜덤 추가
        if len(unique_indices) >= max_samples:
            selected = np.random.choice(unique_indices, max_samples, replace=False)
        else:
            remaining = max_samples - len(unique_indices)
            other_indices = [i for i in range(len(X)) if i not in unique_indices]
            if other_indices:
                additional = np.random.choice(other_indices, 
                                            min(remaining, len(other_indices)), 
                                            replace=False)
                selected = np.concatenate([unique_indices, additional])
            else:
                selected = unique_indices
    
    elif strategy == 'stratified':
        # 클러스터링 기반 층화 샘플링
        try:
            from sklearn.cluster import MiniBatchKMeans
            n_strata = min(max_samples // 10, 50)
            mini_kmeans = MiniBatchKMeans(n_clusters=n_strata, random_state=seed)
            strata_labels = mini_kmeans.fit_predict(X)
            
            samples_per_stratum = max_samples // n_strata
            selected = []
            
            for stratum in range(n_strata):
                stratum_indices = np.where(strata_labels == stratum)[0]
                if len(stratum_indices) > 0:
                    n_samples = min(samples_per_stratum, len(stratum_indices))
                    stratum_sample = np.random.choice(stratum_indices, n_samples, replace=False)
                    selected.extend(stratum_sample)
            
            selected = np.array(selected[:max_samples])
        except:
            # 실패시 랜덤 샘플링
            selected = np.random.choice(len(X), max_samples, replace=False)
    
    else:  # random
        selected = np.random.choice(len(X), max_samples, replace=False)
    
    selected = np.sort(selected)
    return X[selected], [recs[i] for i in selected], selected

def find_optimal_dimensions(X, method='pca', max_components=None, variance_threshold=0.95):
    """최적 차원 수 자동 결정"""
    if max_components is None:
        max_components = min(50, X.shape[1], X.shape[0] - 1)
    
    if method == 'pca':
        pca = PCA()
        pca.fit(X)
        
        # 분산 설명 비율로 결정
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        optimal_dims = np.argmax(cumvar >= variance_threshold) + 1
        
        # 최소 3차원, 최대 제한 적용
        optimal_dims = max(3, min(optimal_dims, max_components))
        
        return optimal_dims, pca.explained_variance_ratio_[:optimal_dims]
    
    elif method == 'elbow':
        # PCA 기반 엘보우 방법
        pca = PCA()
        pca.fit(X)
        
        explained_var = pca.explained_variance_ratio_
        
        # 2차 차분으로 엘보우 찾기
        if len(explained_var) < 3:
            return 3, explained_var[:3]
        
        diffs = np.diff(explained_var)
        second_diffs = np.diff(diffs)
        
        if len(second_diffs) > 0:
            elbow_point = np.argmax(second_diffs) + 3  # 최소 3차원
            elbow_point = min(elbow_point, max_components)
        else:
            elbow_point = min(10, max_components)
        
        return elbow_point, explained_var[:elbow_point]
    
    return min(10, max_components), None

def reduce_dimensions_for_clustering(X, config: dict):
    """클러스터링을 위한 차원 축소 (최적 차원)"""
    if not config['use_reduced_dims_for_clustering']:
        return X, X.shape[1], "Original dimensions"
    
    if config['optimal_dimensions']:
        optimal_dims, explained_var = find_optimal_dimensions(
            X, 
            method=config['dimension_method'],
            max_components=config['max_components'],
            variance_threshold=config['variance_threshold']
        )
        
        # 최소 차원 보장
        optimal_dims = max(config['min_components'], optimal_dims)
        
        pca = PCA(n_components=optimal_dims)
        X_reduced = pca.fit_transform(X)
        
        total_explained = np.sum(explained_var) if explained_var is not None else np.sum(pca.explained_variance_ratio_)
        
        return X_reduced, optimal_dims, f"PCA-{optimal_dims}D (explained: {total_explained:.1%})"
    else:
        # 설정된 차원 수 사용
        n_components = config.get('clustering_dimensions', min(10, X.shape[1]))
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        total_explained = np.sum(pca.explained_variance_ratio_)
        
        return X_reduced, n_components, f"PCA-{n_components}D (explained: {total_explained:.1%})"
    """데이터 크기와 사용 가능한 방법에 따라 최적의 시각화 방법 선택"""
    
    if preference != 'auto' and preference in available_methods:
        return preference
    
    # 자동 선택 로직
    if 'umap' in available_methods:
        return 'umap'  # UMAP이 있으면 우선 선택
    elif n_samples > 1000 and 'tsne' in available_methods:
        return 'pca'  # 큰 데이터에서는 t-SNE 대신 PCA
    elif 'tsne' in available_methods:
        return 'tsne'
    else:
        return 'pca'

def apply_all_dimensionality_reductions(X, config: dict) -> Dict[str, Tuple[np.ndarray, str]]:
    """모든 사용 가능한 차원 축소 방법 실행"""
    results = {}
    
    # 실행할 방법들 결정
    methods_to_run = []
    for method in config['visualization_methods']:
        if method == 'umap' and 'umap' in AVAILABLE_METHODS:
            methods_to_run.append('umap')
        elif method == 'tsne' and 'tsne' in AVAILABLE_METHODS:
            methods_to_run.append('tsne')
        elif method == 'pca':
            methods_to_run.append('pca')
    
    # PCA는 항상 포함 (fallback)
    if 'pca' not in methods_to_run:
        methods_to_run.append('pca')
    
    print(f"   📈 Dimensionality reduction methods: {', '.join(methods_to_run)}")
    
    for method in methods_to_run:
        try:
            if method == 'umap':
                print(f"     🔄 UMAP running...")
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=config['umap_config']['n_neighbors'],
                    min_dist=config['umap_config']['min_dist'],
                    metric=config['umap_config']['metric'],
                    n_epochs=config['umap_config']['n_epochs'],
                    random_state=42,
                    verbose=False
                )
                embedding = reducer.fit_transform(X)
                results[method] = (embedding, "UMAP")
                print(f"     ✅ UMAP completed")
                
            elif method == 'tsne':
                print(f"     🔄 t-SNE running...")
                reducer = TSNE(
                    n_components=2,
                    perplexity=config['tsne_config']['perplexity'],
                    max_iter=config['tsne_config']['max_iter'],  # 수정: n_iter -> max_iter
                    early_exaggeration=config['tsne_config']['early_exaggeration'],
                    random_state=42,
                    verbose=0
                )
                embedding = reducer.fit_transform(X)
                results[method] = (embedding, "t-SNE")
                print(f"     ✅ t-SNE completed")
                
            elif method == 'pca':
                print(f"     🔄 PCA running...")
                pca = PCA(n_components=2)
                embedding = pca.fit_transform(X)
                explained_var = np.sum(pca.explained_variance_ratio_)
                results[method] = (embedding, f"PCA (explained: {explained_var:.1%})")
                print(f"     ✅ PCA completed (explained: {explained_var:.1%})")
                
        except Exception as e:
            print(f"     ❌ {method.upper()} failed: {e}")
            # PCA fallback
            if method != 'pca':
                try:
                    pca = PCA(n_components=2)
                    embedding = pca.fit_transform(X)
                    explained_var = np.sum(pca.explained_variance_ratio_)
                    results[f"{method}_fallback"] = (embedding, f"PCA (fallback, explained: {explained_var:.1%})")
                except:
                    continue
    
    return results
def visualize_all_methods(X, labels, representatives, title, output_dir, config: dict):
    """모든 차원 축소 방법으로 시각화 생성"""
    if len(X) < 3:
        return {}
    
    # 시각화용 샘플링
    max_vis_samples = config['max_samples_visualization']
    if len(X) > max_vis_samples:
        X_vis, _, vis_indices = smart_sample(X, [{}] * len(X), max_vis_samples, 
                                           config['sampling_strategy'])
        if labels is not None:
            labels_vis = labels[vis_indices]
        else:
            labels_vis = None
        
        if representatives:
            rep_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(vis_indices)}
            representatives_vis = [rep_mapping[rep] for rep in representatives if rep in rep_mapping]
        else:
            representatives_vis = []
        
        sample_info = f'Showing: {len(X_vis):,}/{len(X):,} states'
    else:
        X_vis = X
        labels_vis = labels
        representatives_vis = representatives
        sample_info = None
    
    # 모든 차원 축소 방법 실행
    all_embeddings = apply_all_dimensionality_reductions(X_vis, config)
    
    visualization_results = {}
    
    # 각 방법별로 시각화 생성
    for method, (embedding, method_name) in all_embeddings.items():
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if labels_vis is not None and len(np.unique(labels_vis)) > 1:
                scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels_vis, 
                                   s=30, alpha=0.7, cmap='tab10')
                plt.colorbar(scatter, label='Cluster', ax=ax)
                
                if representatives_vis:
                    rep_emb = embedding[representatives_vis]
                    ax.scatter(rep_emb[:, 0], rep_emb[:, 1], c='red', s=150, marker='*', 
                             edgecolors='black', linewidth=2, label='Representatives', alpha=0.9)
                
                ax.legend()
            else:
                ax.scatter(embedding[:, 0], embedding[:, 1], s=30, alpha=0.7)
            
            ax.set_title(f'{title} State Space ({method_name})')
            ax.set_xlabel(f'{method_name} 1')
            ax.set_ylabel(f'{method_name} 2')
            
            # 샘플링 정보
            if config['include_sample_info'] and sample_info:
                ax.text(0.02, 0.98, sample_info, 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            plt.tight_layout()
            
            # 다중 형식 저장 - 더 구체적인 파일명 사용
            for fmt in config['save_formats']:
                # title을 그대로 사용 (이미 "human_clustered" 같은 형태)
                save_path = output_dir / f"{title}_{method}.{fmt}"
                plt.savefig(save_path, dpi=config['plot_dpi'], bbox_inches='tight')
                print(f"     💾 Saved: {save_path}")  # 저장된 파일 확인용
            
            plt.close()
            
            visualization_results[method] = {
                'method_name': method_name,
                'embedding_shape': embedding.shape,
                'file_saved': True
            }
            
        except Exception as e:
            print(f"     ❌ {method} visualization failed: {e}")
            visualization_results[method] = {
                'method_name': method,
                'error': str(e),
                'file_saved': False
            }
    
    return visualization_results
def create_comparison_plots(human_results, llm_results, output_dir, config: dict):
    """차원 축소 방법들의 비교 플롯 생성"""
    
    # 성공한 방법들만 추출
    successful_methods = set(human_results.keys()) & set(llm_results.keys())
    successful_methods = {m for m in successful_methods 
                         if human_results[m].get('file_saved', False) and 
                            llm_results[m].get('file_saved', False)}
    
    if len(successful_methods) < 2:
        print("   ⚠️ Skipping comparison plot (insufficient successful methods)")
        return
    
    # 방법별 성능 비교 차트
    plt.figure(figsize=(12, 6))
    
    methods = list(successful_methods)
    method_names = [human_results[m]['method_name'] for m in methods]
    
    # 실행 성공 여부를 시각화
    success_counts = [2 if m in successful_methods else 0 for m in methods]  # Human + LLM
    
    bars = plt.bar(range(len(methods)), success_counts, 
                  color=['#2E86AB', '#A23B72', '#F18F01'][:len(methods)], 
                  alpha=0.8)
    
    plt.xlabel('Dimensionality Reduction Method')
    plt.ylabel('Successfully Executed Environments')
    plt.title('Execution Results by Dimensionality Reduction Method')
    plt.xticks(range(len(methods)), [m.upper() for m in methods])
    plt.ylim(0, 2.5)
    
    # 막대 위에 정보 표시
    for i, (bar, method) in enumerate(zip(bars, methods)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'Both Envs', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison.png", dpi=config['plot_dpi'])
    plt.close()
    
    # 방법별 요약 테이블 생성
    summary_data = []
    for method in methods:
        summary_data.append({
            'method': method.upper(),
            'method_name': human_results[method]['method_name'],
            'human_success': human_results[method].get('file_saved', False),
            'llm_success': llm_results[method].get('file_saved', False)
        })
    
    return summary_data

def find_optimal_clusters(X, config: dict):
    """설정 가능한 클러스터 수 결정"""
    if len(X) < 3:
        return 1
    
    unique_states = len(np.unique(X, axis=0))
    
    if config['cluster_range'] == 'auto':
        min_k = 2
        max_k = min(config['max_clusters_search'], unique_states // 2, len(X) - 1)
    else:
        min_k, max_k = config['cluster_range']
    
    max_k = max(min_k, min(max_k, len(X) - 1))
    
    if max_k < min_k:
        return min_k
    
    K_range = range(min_k, max_k + 1)
    best_score = -1
    best_k = min_k
    
    for k in K_range:
        try:
            kmeans = KMeans(
                n_clusters=k, 
                n_init=config['kmeans_config']['n_init'],
                max_iter=config['kmeans_config']['max_iter'],
                tol=config['kmeans_config']['tol'],
                random_state=42
            )
            labels = kmeans.fit_predict(X)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        except:
            continue
    
    return best_k

def classify_states(X, recs, config: dict):
    """설정 가능한 상태 분류 (최적 차원에서)"""
    if len(X) < 2:
        return {
            'labels': np.zeros(len(X)),
            'representatives': [0] if len(X) > 0 else [],
            'cluster_info': {},
            'n_clusters': 1 if len(X) > 0 else 0,
            'method': 'single_state',
            'dimensions_used': X.shape[1],
            'dimension_info': "원본 차원"
        }
    
    # 클러스터링을 위한 차원 축소
    X_for_clustering, n_dims, dim_info = reduce_dimensions_for_clustering(X, config)
    
    # 최적 클러스터 수 결정
    optimal_k = find_optimal_clusters(X_for_clustering, config)
    
    # K-means 클러스터링
    kmeans = KMeans(
        n_clusters=optimal_k,
        n_init=config['kmeans_config']['n_init'],
        max_iter=config['kmeans_config']['max_iter'],
        tol=config['kmeans_config']['tol'],
        random_state=42
    )
    labels = kmeans.fit_predict(X_for_clustering)
    
    try:
        silhouette = silhouette_score(X_for_clustering, labels) if len(np.unique(labels)) > 1 else -1
    except:
        silhouette = -1
    
    # 대표 상태 선택 (원본 공간에서)
    representatives = []
    cluster_info = {}
    
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_points_reduced = X_for_clustering[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_points_reduced) == 0:
            continue
        
        # 축소된 공간에서 centroid와의 거리 계산
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_points_reduced - centroid, axis=1)
        rep_idx_in_cluster = np.argmin(distances)
        rep_idx_global = cluster_indices[rep_idx_in_cluster]
        
        representatives.append(rep_idx_global)
        
        cluster_info[f'cluster_{cluster_id}'] = {
            'size': int(np.sum(cluster_mask)),
            'representative_idx': int(rep_idx_global),
            'representative_state': recs[rep_idx_global]['state_after'],
            'avg_distance_to_centroid': float(np.mean(distances)),
            'cluster_compactness': float(np.std(distances))
        }
    
    return {
        'labels': labels,
        'representatives': representatives,
        'cluster_info': cluster_info,
        'n_clusters': len(np.unique(labels)),
        'centroids': kmeans.cluster_centers_,
        'method': 'kmeans_optimal_dims',
        'silhouette_score': silhouette,
        'dimensions_used': n_dims,
        'dimension_info': dim_info,
        'X_clustered': X_for_clustering  # 클러스터링에 사용된 데이터
    }

def vecs(recs: List[dict], slot2i: Dict[str,int], val2i: Dict[str,int]):
    """벡터 변환 (기존과 동일)"""
    dim=len(slot2i)+len(val2i); arr=[]
    for r in recs:
        v=np.zeros(dim,dtype=np.int8)
        for k in r["state_after"]: 
            if k in slot2i:
                v[slot2i[k]]=1
        for vstr in map(str, r["state_after"].values()):
            if vstr in val2i:
                v[len(slot2i)+val2i[vstr]]=1
        arr.append(v)
    return np.vstack(arr)

def richness_metrics(X, slot2i, val2i, config: dict):
    """Richness 메트릭 계산"""
    uniq = len(np.unique(X, axis=0))
    
    # Entropy
    state_counts = Counter(map(bytes, X))
    ent = entropy(list(state_counts.values()))
    
    cov = uniq / len(X)
    
    # PCA
    sample_size = min(len(X), 5000)  # PCA용 샘플링
    if len(X) > sample_size:
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X
    
    try:
        pca = PCA().fit(X_sample)
        eff = (pca.explained_variance_ratio_.cumsum() >= 0.95).argmax() + 1
    except:
        eff = min(10, X.shape[1])
    
    # 클러스터 기반 density
    dens = 0.0
    if uniq > 3:
        try:
            n_clusters = min(8, uniq // 2)
            sample_for_clustering = X_sample[:min(3000, len(X_sample))]
            km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42).fit(sample_for_clustering)
            dens = silhouette_score(sample_for_clustering, km.labels_)
        except:
            dens = 0.0
    
    # Separation
    sep = 0.0
    if uniq > 1:
        unique_states = np.unique(X, axis=0)
        if len(unique_states) > 200:
            sample_idx = np.random.choice(len(unique_states), 200, replace=False)
            unique_states = unique_states[sample_idx]
        
        if len(unique_states) > 1:
            try:
                distances = cdist(unique_states, unique_states)
                sep = float(distances[np.triu_indices_from(distances, k=1)].mean())
            except:
                sep = 0.0
    
    return dict(
        unique_states=uniq,
        state_entropy=ent,
        coverage_ratio=cov,
        effective_dim=eff,
        density_uniformity=dens,
        cluster_separation=sep,
        slot_vocab=len(slot2i),
        value_vocab=len(val2i)
    )

def _to_py(x):
    """NumPy 타입을 Python 타입으로 안전하게 변환"""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()  # ndarray를 list로 변환
    if isinstance(x, (list, tuple)):
        return [_to_py(item) for item in x]
    if isinstance(x, dict):
        return {key: _to_py(value) for key, value in x.items()}
    return x

def safe_json_serialize(obj):
    """JSON 직렬화를 위한 안전한 변환"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {key: safe_json_serialize(value) for key, value in obj.items()}
    # 알 수 없는 타입은 문자열로 변환
    return str(obj)

def visualize_states(X, labels, representatives, title, path, config: dict):
    """레거시 호환용 - 실제로는 모든 방법 실행"""
    output_dir = path.parent if hasattr(path, 'parent') else Path('.')
    return visualize_all_methods(X, labels, representatives, title, output_dir, config)

def main():
    parser = argparse.ArgumentParser(description="설정 가능한 State-Richness 분석기")
    parser.add_argument("--annotations", required=True, help="어노테이션 JSON 파일")
    parser.add_argument("--outdir", default="state_richness_results", help="출력 디렉토리")
    parser.add_argument("--config", help="설정 JSON 파일")
    
    # 성능 관련
    parser.add_argument("--performance", choices=['fast', 'balanced', 'accurate'], 
                       default='balanced', help="성능 모드")
    parser.add_argument("--max-samples", type=int, help="최대 샘플 수 (클러스터링)")
    parser.add_argument("--max-vis-samples", type=int, help="최대 샘플 수 (시각화)")
    
    # 방법 선택
    parser.add_argument("--sampling", choices=['smart', 'random', 'stratified'], 
                       default='smart', help="샘플링 전략")
    parser.add_argument("--save-formats", nargs='+', choices=['png', 'pdf', 'svg'],
                       default=['png'], help="저장할 이미지 형식들")
    
    # 클러스터링 설정
    parser.add_argument("--max-clusters", type=int, help="최대 클러스터 수")
    parser.add_argument("--cluster-range", nargs=2, type=int, help="클러스터 범위 [min max]")
    parser.add_argument("--no-dim-reduction", action="store_true", help="차원 축소 없이 클러스터링")
    parser.add_argument("--variance-threshold", type=float, default=0.95, help="PCA 분산 설명 임계값")
    parser.add_argument("--min-components", type=int, default=3, help="최소 차원 수")
    parser.add_argument("--dimension-method", choices=['pca', 'elbow'], default='pca', help="차원 결정 방법")
    
    args = parser.parse_args()
    
    # 설정 로드
    analyzer = ConfigurableStateAnalyzer()
    
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            user_config = json.load(f)
        analyzer.config.update(user_config)
    
    # CLI 인수로 설정 오버라이드
    analyzer.config['performance_mode'] = args.performance
    analyzer.config['sampling_strategy'] = args.sampling
    
    if args.save_formats:
        analyzer.config['save_formats'] = args.save_formats
    
    if args.max_samples:
        analyzer.config['max_samples_clustering'] = args.max_samples
    if args.max_vis_samples:
        analyzer.config['max_samples_visualization'] = args.max_vis_samples
    if args.max_clusters:
        analyzer.config['max_clusters_search'] = args.max_clusters
    if args.cluster_range:
        analyzer.config['cluster_range'] = args.cluster_range
    if args.no_dim_reduction:
        analyzer.config['use_reduced_dims_for_clustering'] = False
    if args.variance_threshold:
        analyzer.config['variance_threshold'] = args.variance_threshold
    if args.min_components:
        analyzer.config['min_components'] = args.min_components
    if args.dimension_method:
        analyzer.config['dimension_method'] = args.dimension_method
    
    # 출력 디렉토리 생성
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드
    print("📊 데이터 로드 중...")
    with open(args.annotations) as f:
        data = json.load(f)
    
    recs = data["annotations"] if isinstance(data, dict) else data
    H = [r for r in recs if r.get("annotation_type") == "human"]
    L = [r for r in recs if r.get("annotation_type") == "llm"]
    
    if not H or not L:
        raise RuntimeError("Insufficient Human or LLM annotation data.")
    
    print(f"   Human: {len(H):,}, LLM: {len(L):,}")
    
    # 데이터 크기에 따른 자동 설정
    total_samples = len(H) + len(L)
    analyzer._auto_configure_for_data_size(total_samples)
    
    print(f"⚙️  Configuration: {analyzer.config['performance_mode']} mode")
    print(f"   Clustering samples: {analyzer.config['max_samples_clustering']:,}")
    print(f"   Visualization samples: {analyzer.config['max_samples_visualization']:,}")
    print(f"   Visualization methods: {', '.join(analyzer.config['visualization_methods']).upper()}")
    print(f"   Available methods: {', '.join(AVAILABLE_METHODS)}")
    print(f"   Save formats: {', '.join(analyzer.config['save_formats'])}")
    
    # Vocabulary 구축
    slotH = {s: i for i, s in enumerate(sorted({k for r in H for k in r["state_after"]}))}
    valH = {v: i for i, v in enumerate(sorted({str(x) for r in H for x in r["state_after"].values()}))}
    slotL = {s: i for i, s in enumerate(sorted({k for r in L for k in r["state_after"]}))}
    valL = {v: i for i, v in enumerate(sorted({str(x) for r in L for x in r["state_after"].values()}))}
    
    # 벡터 변환
    Xh_full = vecs(H, slotH, valH)
    Xl_full = vecs(L, slotL, valL)
    print(f"   벡터 차원: Human {Xh_full.shape}, LLM {Xl_full.shape}")
    
    # 클러스터링용 샘플링
    max_clustering = analyzer.config['max_samples_clustering']
    Xh, H_sample, h_indices = smart_sample(Xh_full, H, max_clustering, 
                                          analyzer.config['sampling_strategy'])
    Xl, L_sample, l_indices = smart_sample(Xl_full, L, max_clustering, 
                                          analyzer.config['sampling_strategy'])
    
    if len(Xh) < len(Xh_full) or len(Xl) < len(Xl_full):
        print(f"   샘플링됨: Human {len(Xh):,}/{len(Xh_full):,}, LLM {len(Xl):,}/{len(Xl_full):,}")
    
    # Richness 계산
    print("📊 Richness 메트릭 계산 중...")
    Rh = richness_metrics(Xh_full, slotH, valH, analyzer.config)
    Rl = richness_metrics(Xl_full, slotL, valL, analyzer.config)
    
    # 클러스터링
    print("🔍 상태 클러스터링 중...")
    
    print("   Human 분석...")
    classification_h = classify_states(Xh, H_sample, analyzer.config)
    print(f"   └ {classification_h['dimensions_used']}D에서 {classification_h['n_clusters']}개 클러스터")
    print(f"     차원 정보: {classification_h['dimension_info']}")
    print(f"     실루엣 스코어: {classification_h.get('silhouette_score', 0):.3f}")
    
    print("   LLM 분석...")
    classification_l = classify_states(Xl, L_sample, analyzer.config)
    print(f"   └ {classification_l['dimensions_used']}D에서 {classification_l['n_clusters']}개 클러스터")
    print(f"     차원 정보: {classification_l['dimension_info']}")
    print(f"     실루엣 스코어: {classification_l.get('silhouette_score', 0):.3f}")
    
    # 시각화
    print("📈 시각화 생성 중...")
    visualize_states(Xh, classification_h['labels'], classification_h['representatives'],
                    "Human", output_dir / "human_clustered", analyzer.config)
    visualize_states(Xl, classification_l['labels'], classification_l['representatives'],
                    "LLM", output_dir / "llm_clustered", analyzer.config)
    
    # 전체 데이터 시각화 (클러스터 없이)
    visualize_states(Xh_full, None, None, "Human", output_dir / "human_overview", analyzer.config)
    visualize_states(Xl_full, None, None, "LLM", output_dir / "llm_overview", analyzer.config)
    
    # 비교 차트들
    print("📊 Generating comparison charts...")
    
    # Richness 비교
    features = ["unique_states", "coverage_ratio", "effective_dim"]
    plt.figure(figsize=(8, 5))
    x = np.arange(len(features))
    width = 0.35
    
    plt.bar(x - width/2, [Rh[f] for f in features], width, label="Human", alpha=0.8)
    plt.bar(x + width/2, [Rl[f] for f in features], width, label="LLM", alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('State Richness Comparison')
    plt.xticks(x, features)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "richness_comparison.png", dpi=analyzer.config['plot_dpi'])
    plt.close()
    
    # 클러스터 수 비교
    plt.figure(figsize=(8, 5))
    categories = ['Human', 'LLM']
    cluster_counts = [classification_h['n_clusters'], classification_l['n_clusters']]
    colors = ['#2E86AB', '#A23B72']
    
    bars = plt.bar(categories, cluster_counts, color=colors, alpha=0.8)
    plt.ylabel('Number of Clusters')
    plt.title('State Clusters Comparison')
    
    for bar, count in zip(bars, cluster_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_count_comparison.png", dpi=analyzer.config['plot_dpi'])
    plt.close()
    
    # 결과 저장
    print("💾 Saving results...")
    
    # 시각화 결과 안전하게 수집
    viz_results = {}
    if 'human_viz_results' in locals():
        viz_results['human_clustered'] = safe_json_serialize(human_viz_results)
    if 'llm_viz_results' in locals():
        viz_results['llm_clustered'] = safe_json_serialize(llm_viz_results)
    if 'human_overview_results' in locals():
        viz_results['human_overview'] = safe_json_serialize(human_overview_results)
    if 'llm_overview_results' in locals():
        viz_results['llm_overview'] = safe_json_serialize(llm_overview_results)
    if 'comparison_summary' in locals():
        viz_results['comparison_summary'] = safe_json_serialize(comparison_summary)
    
    # 클러스터링 결과 안전하게 변환
    human_clustering_safe = {}
    for k, v in classification_h.items():
        if k not in ['centroids', 'X_clustered']:  # 큰 배열들 제외
            human_clustering_safe[k] = safe_json_serialize(v)
    human_clustering_safe["representative_indices"] = [int(x) for x in classification_h['representatives']]
    
    llm_clustering_safe = {}
    for k, v in classification_l.items():
        if k not in ['centroids', 'X_clustered']:  # 큰 배열들 제외
            llm_clustering_safe[k] = safe_json_serialize(v)
    llm_clustering_safe["representative_indices"] = [int(x) for x in classification_l['representatives']]
    
    results = {
        "configuration": safe_json_serialize(analyzer.config),
        "data_info": {
            "total_human_states": len(Xh_full),
            "total_llm_states": len(Xl_full),
            "analyzed_human_states": len(Xh),
            "analyzed_llm_states": len(Xl),
            "available_methods": AVAILABLE_METHODS
        },
        "visualization_results": viz_results,
        "human": {
            "richness_metrics": {k: safe_json_serialize(v) for k, v in Rh.items()},
            "clustering": human_clustering_safe
        },
        "llm": {
            "richness_metrics": {k: safe_json_serialize(v) for k, v in Rl.items()},
            "clustering": llm_clustering_safe
        }
    }
    
    output_file = output_dir / "analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=True)  # ensure_ascii=True로 변경
    
    print(f"✅ Analysis completed: {output_file}")
    print(f"\n📊 Results Summary:")
    print(f"   Human: {Rh['unique_states']:,} unique states → {classification_h['dimensions_used']}D → {classification_h['n_clusters']} clusters")
    print(f"   LLM:   {Rl['unique_states']:,} unique states → {classification_l['dimensions_used']}D → {classification_l['n_clusters']} clusters")
    print(f"   Dimensionality reduction: {'Used' if analyzer.config['use_reduced_dims_for_clustering'] else 'Not used'}")
    
    # 생성된 시각화 방법들 요약
    successful_methods = set()
    generated_files = []
    
    if viz_results:
        for data_type, methods_result in viz_results.items():
            if isinstance(methods_result, dict):
                for method, result in methods_result.items():
                    if isinstance(result, dict) and result.get('file_saved', False):
                        successful_methods.add(method)
                        # 실제 생성된 파일 목록 구성
                        for fmt in analyzer.config['save_formats']:
                            generated_files.append(f"{data_type}_{method}.{fmt}")
    
    if successful_methods:
        print(f"   Visualization methods: {', '.join(sorted(successful_methods)).upper()}")
        print(f"   📁 Generated {len(generated_files)} visualization files:")
        for i, file in enumerate(sorted(generated_files)):
            if i < 10:  # 처음 10개만 표시
                print(f"      {file}")
            elif i == 10:
                print(f"      ... and {len(generated_files) - 10} more files")
                break
    else:
        print(f"   ⚠️ No visualization method info available")
        print(f"   💡 Files may still be generated. Check the output directory:")
        print(f"      {output_dir}")
        
    print(f"   📁 Comparison charts:")
    for name in ['richness_comparison.png', 'cluster_count_comparison.png']:
        print(f"      {name}")

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    main()