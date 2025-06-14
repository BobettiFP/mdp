#!/usr/bin/env python3
"""
State‑Richness pipeline rev 7  — 시각화 개선 버전
* 클러스터 분리 개선을 위한 전처리 및 매개변수 튜닝
* 더 나은 시각화를 위한 점 크기, 투명도, 이상치 처리
* 개선된 차원축소 매개변수들
"""

import argparse, json, random, os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend, avoids font issues
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import entropy
from scipy.spatial.distance import cdist

# ───────────────────────── AVAILABLE DIM‑REDUCTION BACKENDS ────────────────────
AVAILABLE_METHODS: List[str] = ["pca"]
try:
    import umap  # type: ignore
    AVAILABLE_METHODS.append("umap")
except ImportError:
    pass

try:
    from sklearn.manifold import TSNE  # noqa: F401
    AVAILABLE_METHODS.append("tsne")
except ImportError:
    pass

# ───────────────────────────────── CONFIG CLASS ───────────────────────────────
class ConfigurableStateAnalyzer:
    """Fully configurable state‑space analyzer with improved visualization."""

    def __init__(self, config: Optional[dict] = None):
        self.config = self._load_default_config()
        if config:
            self.config.update(config)

    def _load_default_config(self) -> dict:
        return {
            # sampling
            "max_samples_clustering": None,
            "max_samples_visualization": None,
            "sampling_strategy": "smart",  # smart | random | stratified

            # 전처리 개선 (안전한 설정)
            "preprocessing": {
                "remove_duplicates": False,  # 인덱싱 문제 방지를 위해 비활성화
                "scale_features": True,
                "scaler_type": "robust",  # standard | robust | none
                "variance_filter": False,  # 인덱싱 문제 방지를 위해 비활성화
                "min_variance": 0.01,
            },

            # dimensionality reduction for clustering
            "optimal_dimensions": True,
            "dimension_method": "pca",  # pca | elbow
            "variance_threshold": 0.95,
            "max_components": None,
            "min_components": 3,

            # clustering
            "clustering_methods": ["kmeans"],
            "cluster_range": "auto",
            "max_clusters_search": None,
            "use_reduced_dims_for_clustering": True,
            "kmeans_config": {
                "n_init": "auto",
                "max_iter": "auto",
                "tol": 1e-4,
            },

            # visualization
            "generate_all_visualizations": True,
            "visualization_methods": ["pca", "umap", "tsne"],
            "fallback_method": "pca",
            
            # 개선된 UMAP 설정
            "umap_config": {
                "n_neighbors": "auto",
                "min_dist": 0.01,  # 더 작게 설정하여 클러스터 분리 개선
                "metric": "euclidean",
                "n_epochs": "auto",
                "spread": 2.0,
                "local_connectivity": 2.0,
            },
            
            # 개선된 t-SNE 설정
            "tsne_config": {
                "perplexity": "auto",
                "max_iter": "auto",
                "early_exaggeration": 20.0,  # 기본값보다 높게
                "learning_rate": "auto",
                "init": "pca",  # random 대신 pca 초기화
            },
            
            "pca_config": {
                "n_components": 2,
            },

            # 시각화 개선
            "visualization_enhancement": {
                "alpha_adjustment": True,
                "point_size_adjustment": True,
                "outlier_handling": True,
                "jitter_amount": 0.02,
            },

            # performance
            "performance_mode": "balanced",  # fast | balanced | accurate
            "adaptive_sampling": True,
            "parallel_processing": True,

            # output
            "plot_dpi": 300,  # 해상도 높임
            "save_formats": ["png"],
            "include_sample_info": True,
            "figure_size": (12, 10),  # 그림 크기 키움
        }

    def _auto_configure_for_data_size(self, n_samples: int):
        cfg = self.config

        # clustering/visualization sample caps
        if cfg["max_samples_clustering"] is None:
            cfg["max_samples_clustering"] = (
                15000 if n_samples > 50_000 else 8000 if n_samples > 10_000 else n_samples
            )
        if cfg["max_samples_visualization"] is None:
            cfg["max_samples_visualization"] = (
                8000 if n_samples > 10_000 else 5000 if n_samples > 5000 else n_samples
            )

        # cluster search range
        if cfg["max_clusters_search"] is None:
            cfg["max_clusters_search"] = max(5, min(n_samples // 10, 20))

        # k‑means auto n_init / max_iter
        if cfg["kmeans_config"]["n_init"] == "auto":
            cfg["kmeans_config"]["n_init"] = 5 if n_samples > 10_000 else 10 if n_samples > 1000 else 20
        if cfg["kmeans_config"]["max_iter"] == "auto":
            cfg["kmeans_config"]["max_iter"] = 100 if n_samples > 10_000 else 300

        # 개선된 UMAP auto adjustments
        if cfg["umap_config"]["n_neighbors"] == "auto":
            cfg["umap_config"]["n_neighbors"] = max(5, min(int(np.sqrt(n_samples) * 0.5), 30))
        if cfg["umap_config"]["n_epochs"] == "auto":
            cfg["umap_config"]["n_epochs"] = 500 if n_samples > 10_000 else 1000

        # 개선된 t‑SNE auto adjustments
        if cfg["tsne_config"]["perplexity"] == "auto":
            cfg["tsne_config"]["perplexity"] = max(10, min(50, n_samples // 4, n_samples - 1))
        if cfg["tsne_config"]["max_iter"] == "auto":
            cfg["tsne_config"]["max_iter"] = 1000 if n_samples > 5000 else 2000
        if cfg["tsne_config"]["learning_rate"] == "auto":
            cfg["tsne_config"]["learning_rate"] = max(50, min(1000, n_samples / 12))

        # performance modes
        if cfg["performance_mode"] == "fast":
            cfg["max_samples_clustering"] = min(cfg["max_samples_clustering"], 5000)
            cfg["max_samples_visualization"] = min(cfg["max_samples_visualization"], 3000)
            cfg["max_clusters_search"] = min(cfg["max_clusters_search"], 8)
            cfg["visualization_methods"] = [m for m in cfg["visualization_methods"] if m != "tsne"]
        elif cfg["performance_mode"] == "accurate" and "tsne" not in cfg["visualization_methods"]:
            cfg["visualization_methods"].append("tsne")

# ─────────────────────────────── 전처리 개선 ────────────────────────────────────

def preprocess_features(X: np.ndarray, cfg: dict):
    """향상된 전처리 파이프라인"""
    X_processed = X.copy().astype(float)
    preprocessing_info = {}
    
    # 1. 중복 제거
    if cfg["preprocessing"]["remove_duplicates"]:
        unique_rows, unique_idx = np.unique(X_processed, axis=0, return_index=True)
        if len(unique_rows) < len(X_processed):
            duplicates_removed = len(X_processed) - len(unique_rows)
            X_processed = unique_rows
            preprocessing_info["duplicates_removed"] = duplicates_removed
        else:
            preprocessing_info["duplicates_removed"] = 0
    
    # 2. 분산 필터링
    if cfg["preprocessing"]["variance_filter"] and X_processed.shape[1] > 1:
        variances = np.var(X_processed, axis=0)
        high_var_features = variances > cfg["preprocessing"]["min_variance"]
        if np.sum(high_var_features) > 0:
            X_processed = X_processed[:, high_var_features]
            preprocessing_info["low_variance_features_removed"] = np.sum(~high_var_features)
        else:
            preprocessing_info["low_variance_features_removed"] = 0
    
    # 3. 스케일링
    if cfg["preprocessing"]["scale_features"] and X_processed.shape[1] > 1:
        if cfg["preprocessing"]["scaler_type"] == "standard":
            scaler = StandardScaler()
        elif cfg["preprocessing"]["scaler_type"] == "robust":
            scaler = RobustScaler()
        else:
            scaler = None
            
        if scaler is not None:
            X_processed = scaler.fit_transform(X_processed)
            preprocessing_info["scaling_applied"] = cfg["preprocessing"]["scaler_type"]
    
    return X_processed, preprocessing_info

# ─────────────────────────────── SAMPLING ────────────────────────────────────

def smart_sample(
    X: np.ndarray,
    recs: List[dict],
    max_samples: int,
    strategy: str = "smart",
    seed: int = 42,
):
    if len(X) <= max_samples:
        return X, recs, np.arange(len(X))

    rng = np.random.default_rng(seed)

    if strategy == "smart":
        seen, uniq_idx = set(), []
        for i, vec in enumerate(X):
            key = tuple(vec.round(6) if vec.dtype.kind in 'fc' else vec)
            if key not in seen:
                seen.add(key)
                uniq_idx.append(i)
        
        if len(uniq_idx) >= max_samples:
            sel = rng.choice(uniq_idx, max_samples, replace=False)
        else:
            remaining = max_samples - len(uniq_idx)
            others = [i for i in range(len(X)) if i not in uniq_idx]
            if len(others) > 0:
                add = rng.choice(others, min(remaining, len(others)), replace=False)
                sel = np.concatenate([uniq_idx, add])
            else:
                sel = np.array(uniq_idx)

    elif strategy == "stratified":
        try:
            from sklearn.cluster import MiniBatchKMeans
            n_strata = max(2, min(max_samples // 20, 100))
            labels = MiniBatchKMeans(n_clusters=n_strata, random_state=seed).fit_predict(X)
            per = max_samples // n_strata
            sel = []
            for s in range(n_strata):
                idx = np.where(labels == s)[0]
                if idx.size:
                    sel.extend(rng.choice(idx, min(per, idx.size), replace=False))
            sel = np.array(sel[:max_samples])
        except Exception:
            sel = rng.choice(len(X), max_samples, replace=False)

    else:  # random
        sel = rng.choice(len(X), max_samples, replace=False)

    sel.sort()
    return X[sel], [recs[i] for i in sel], sel

# ─────────────────────── DIMENSIONALITY REDUCTION ────────────────────────────

def find_optimal_dimensions(X, method: str, max_components: int, variance_threshold: float):
    max_components = min(max_components, X.shape[1], X.shape[0] - 1)
    if method == "elbow":
        pca = PCA().fit(X)
        ev = pca.explained_variance_ratio_
        if len(ev) < 3:
            return max(3, len(ev)), ev[: len(ev)]
        second = np.diff(np.diff(ev))
        elbow = np.argmax(second) + 3
        return min(max(3, elbow), max_components), ev[: elbow]
    else:  # pca cumulative variance
        pca = PCA().fit(X)
        cum = np.cumsum(pca.explained_variance_ratio_)
        k = np.argmax(cum >= variance_threshold) + 1
        k = max(3, min(k, max_components))
        return k, pca.explained_variance_ratio_[:k]


def reduce_dimensions_for_clustering(X, cfg):
    if not cfg["use_reduced_dims_for_clustering"]:
        return X, X.shape[1], "Original"

    max_comp = cfg["max_components"] or 50
    k, ev = find_optimal_dimensions(
        X,
        cfg["dimension_method"],
        max_components=max_comp,
        variance_threshold=cfg["variance_threshold"],
    )
    k = max(cfg["min_components"], k)
    pca = PCA(n_components=k)
    Xr = pca.fit_transform(X)
    return Xr, k, f"PCA‑{k}D (cum var {np.sum(ev):.1%})"

# ───────────────────────────── 개선된 VISUALIZATION ─────────────────────────────────

def apply_all_dimred(X: np.ndarray, cfg: dict):
    """개선된 차원축소 적용"""
    methods = []
    for m in cfg["visualization_methods"]:
        if m in ("pca",) or (m == "umap" and "umap" in AVAILABLE_METHODS) or (
            m == "tsne" and "tsne" in AVAILABLE_METHODS
        ):
            methods.append(m)
    if "pca" not in methods:
        methods.append("pca")  # always have fallback

    results = {}
    for m in methods:
        try:
            if m == "pca":
                pca = PCA(n_components=2)
                emb = pca.fit_transform(X)
                results[m] = (emb, f"PCA ({np.sum(pca.explained_variance_ratio_):.1%})")
                
            elif m == "umap":
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=cfg["umap_config"]["n_neighbors"],
                    min_dist=cfg["umap_config"]["min_dist"],
                    metric=cfg["umap_config"]["metric"],
                    n_epochs=cfg["umap_config"]["n_epochs"],
                    spread=cfg["umap_config"]["spread"],
                    local_connectivity=cfg["umap_config"]["local_connectivity"],
                    random_state=42,
                    verbose=False,
                )
                results[m] = (reducer.fit_transform(X), "UMAP")
                
            elif m == "tsne":
                reducer = TSNE(
                    n_components=2,
                    perplexity=cfg["tsne_config"]["perplexity"],
                    n_iter=cfg["tsne_config"]["max_iter"],
                    early_exaggeration=cfg["tsne_config"]["early_exaggeration"],
                    learning_rate=cfg["tsne_config"]["learning_rate"],
                    init=cfg["tsne_config"]["init"],
                    random_state=42,
                    verbose=0,
                )
                results[m] = (reducer.fit_transform(X), "t‑SNE")
        except Exception as e:
            print(f"    ❌ {m} failed: {e}")
    return results


def add_jitter(X, amount=0.02):
    """겹치는 점들에 작은 노이즈 추가"""
    noise = np.random.normal(0, amount, X.shape)
    return X + noise


def detect_outliers(X, method="iqr", threshold=1.5):
    """이상치 감지"""
    if method == "iqr":
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        outliers = np.any((X < lower) | (X > upper), axis=1)
    else:  # z-score
        z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        outliers = np.any(z_scores > threshold, axis=1)
    return outliers


def visualize_states(X, labels, reps, title: str, out_dir: Path, cfg):
    """개선된 시각화 함수"""
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # subsample for drawing first (원본 데이터에서)
    if len(X) > cfg["max_samples_visualization"]:
        X_draw, _, idx = smart_sample(X, [{}] * len(X), 
                                    cfg["max_samples_visualization"], cfg["sampling_strategy"])
        labels_draw = labels[idx] if labels is not None else None
        reps_draw = [np.where(idx == r)[0][0] for r in reps if r in idx] if reps else []
        info = f"Showing {len(X_draw):,}/{len(X):,}"
    else:
        X_draw, labels_draw, reps_draw, info = X, labels, reps, None

    # 샘플링 후 전처리 적용 (시각화용 - 크기 변경 없는 처리만)
    X_processed = X_draw.copy().astype(float)
    preprocess_info = {}
    
    # 시각화에서는 스케일링만 적용 (크기가 변하지 않음)
    if cfg["preprocessing"]["scale_features"] and X_processed.shape[1] > 1:
        if cfg["preprocessing"]["scaler_type"] == "standard":
            scaler = StandardScaler()
        elif cfg["preprocessing"]["scaler_type"] == "robust":
            scaler = RobustScaler()
        else:
            scaler = None
            
        if scaler is not None:
            X_processed = scaler.fit_transform(X_processed)
            preprocess_info["scaling_applied"] = cfg["preprocessing"]["scaler_type"]

    embeddings = apply_all_dimred(X_processed, cfg)
    vis_result = {}
    
    for m, (E, name) in embeddings.items():
        # 지터 추가 (옵션)
        if cfg["visualization_enhancement"]["jitter_amount"] > 0:
            E = add_jitter(E, cfg["visualization_enhancement"]["jitter_amount"])
        
        fig, ax = plt.subplots(figsize=cfg["figure_size"])
        
        # 이상치 감지
        if cfg["visualization_enhancement"]["outlier_handling"]:
            outliers = detect_outliers(E)
            normal_points = ~outliers
        else:
            outliers = np.zeros(len(E), dtype=bool)
            normal_points = np.ones(len(E), dtype=bool)
        
        # 점 크기와 투명도 조정
        if cfg["visualization_enhancement"]["point_size_adjustment"]:
            base_size = 20 if len(E) < 1000 else 15 if len(E) < 5000 else 10
        else:
            base_size = 30
            
        if cfg["visualization_enhancement"]["alpha_adjustment"]:
            alpha = 0.8 if len(E) < 1000 else 0.6 if len(E) < 5000 else 0.4
        else:
            alpha = 0.7
        
        if labels_draw is not None and len(np.unique(labels_draw)) > 1:
            # 일반 점들 그리기
            if np.any(normal_points):
                sc = ax.scatter(E[normal_points, 0], E[normal_points, 1], 
                              c=labels_draw[normal_points], s=base_size, alpha=alpha, 
                              cmap="tab20", edgecolors='none')
                plt.colorbar(sc, ax=ax, label="Cluster")
            
            # 이상치 점들 그리기
            if np.any(outliers):
                ax.scatter(E[outliers, 0], E[outliers, 1], 
                          c=labels_draw[outliers], s=base_size*1.5, alpha=alpha*1.2, 
                          cmap="tab20", marker='s', edgecolors='black', linewidth=0.5)
            
            # 대표점들 그리기
            if reps_draw:
                ax.scatter(
                    E[reps_draw, 0], E[reps_draw, 1],
                    c="red", s=base_size*3, marker="*",
                    edgecolors="black", linewidths=1.5,
                    label="Representatives", alpha=1.0, zorder=10
                )
                ax.legend()
        else:
            # 클러스터 라벨이 없는 경우
            if np.any(normal_points):
                ax.scatter(E[normal_points, 0], E[normal_points, 1], 
                          s=base_size, alpha=alpha, c='steelblue', edgecolors='none')
            if np.any(outliers):
                ax.scatter(E[outliers, 0], E[outliers, 1], 
                          s=base_size*1.5, alpha=alpha*1.2, c='orange', 
                          marker='s', edgecolors='black', linewidth=0.5)
        
        ax.set_title(f"{title} State Space ({name})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Component 1", fontsize=12)
        ax.set_ylabel("Component 2", fontsize=12)
        
        # 격자와 테두리 개선
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if info and cfg["include_sample_info"]:
            ax.text(0.01, 0.99, info, transform=ax.transAxes, va="top", 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                   fontsize=9)
        
        plt.tight_layout()
        
        for fmt in cfg["save_formats"]:
            path = out_dir.with_suffix(f".{fmt}").parent / f"{title.lower()}_{m}.{fmt}"
            path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(path, dpi=cfg["plot_dpi"], bbox_inches="tight", 
                       facecolor='white', edgecolor='none')
        plt.close(fig)
        
        vis_result[m] = {
            "method": name, 
            "file_saved": True, 
            "shape": E.shape,
            "preprocessing_info": preprocess_info
        }
    
    return vis_result

# ─────────────────────────── CLUSTERING ───────────────────────────────────────

def find_optimal_clusters(X: np.ndarray, cfg):
    uniq = len(np.unique(X, axis=0))
    if uniq < 3:
        return 1
    if cfg["cluster_range"] == "auto":
        min_k, max_k = 2, min(cfg["max_clusters_search"], uniq // 2, len(X) - 1)
    else:
        min_k, max_k = cfg["cluster_range"]
    best_k, best_score = min_k, -1
    for k in range(min_k, max_k + 1):
        try:
            lbl = KMeans(n_clusters=k, **cfg["kmeans_config"], random_state=42).fit_predict(X)
            if len(np.unique(lbl)) > 1:
                sc = silhouette_score(X, lbl)
                if sc > best_score:
                    best_k, best_score = k, sc
        except Exception:
            continue
    return best_k


def classify_states(X: np.ndarray, recs: List[dict], cfg):
    if len(X) < 2:
        return {
            "labels": np.zeros(len(X)),
            "representatives": [0] if len(X) else [],
            "n_clusters": 1 if len(X) else 0,
            "dimensions_used": X.shape[1],
            "dimension_info": "N/A",
            "silhouette_score": -1,
        }
    
    # 클러스터링을 위한 전처리 (크기 변경 없는 처리만)
    X_processed = X.copy().astype(float)
    if cfg["preprocessing"]["scale_features"] and X_processed.shape[1] > 1:
        if cfg["preprocessing"]["scaler_type"] == "standard":
            scaler = StandardScaler()
        elif cfg["preprocessing"]["scaler_type"] == "robust":
            scaler = RobustScaler()
        else:
            scaler = None
            
        if scaler is not None:
            X_processed = scaler.fit_transform(X_processed)
    
    Xr, ndims, info = reduce_dimensions_for_clustering(X_processed, cfg)
    k = find_optimal_clusters(Xr, cfg)

    km = KMeans(n_clusters=k, **cfg["kmeans_config"], random_state=42)
    lbl = km.fit_predict(Xr)
    sil = silhouette_score(Xr, lbl) if k > 1 else -1

    reps, cinfo = [], {}
    for cid in np.unique(lbl):
        idx = np.where(lbl == cid)[0]
        centroid = km.cluster_centers_[cid]
        dist = np.linalg.norm(Xr[idx] - centroid, axis=1)
        ridx_local = np.argmin(dist)
        ridx = idx[ridx_local]
        reps.append(ridx)
        cinfo[cid] = {
            "size": int(idx.size),
            "rep_index": int(ridx),
            "avg_dist": float(dist.mean()),
        }
    return {
        "labels": lbl,
        "representatives": reps,
        "cluster_info": cinfo,
        "n_clusters": int(k),
        "dimensions_used": ndims,
        "dimension_info": info,
        "silhouette_score": float(sil),
    }

# ───────────────────────── RICHNESS METRICS ─────────────────────────────────

def vecs(recs: List[dict], slot2i: Dict[str, int], val2i: Dict[str, int]):
    dim = len(slot2i) + len(val2i)
    arr = []
    for r in recs:
        v = np.zeros(dim, np.int8)
        for k in r["state_after"]:
            if k in slot2i:
                v[slot2i[k]] = 1
        for val_str in map(str, r["state_after"].values()):
            if val_str in val2i:
                v[len(slot2i) + val2i[val_str]] = 1
        arr.append(v)
    return np.vstack(arr)


def richness_metrics(X: np.ndarray, slot2i, val2i):
    uniq = len(np.unique(X, axis=0))
    ent = entropy(list(Counter(map(bytes, X)).values()))
    cov = uniq / len(X)

    sample_idx = np.random.choice(len(X), min(len(X), 5000), replace=False)
    eff = PCA().fit(X[sample_idx]).explained_variance_ratio_.cumsum().searchsorted(0.95) + 1

    dens = 0.0
    try:
        tmp = X[sample_idx][:3000]
        lbl = KMeans(n_clusters=min(8, uniq // 2), n_init=3, random_state=42).fit_predict(tmp)
        dens = silhouette_score(tmp, lbl)
    except Exception:
        pass

    sep = 0.0
    try:
        uniq_states = np.unique(X, axis=0)
        if len(uniq_states) > 1:
            if len(uniq_states) > 200:
                uniq_states = uniq_states[np.random.choice(len(uniq_states), 200, replace=False)]
            d = cdist(uniq_states, uniq_states)
            sep = float(d[np.triu_indices_from(d, 1)].mean())
    except Exception:
        pass

    return {
        "unique_states": uniq,
        "state_entropy": ent,
        "coverage_ratio": cov,
        "effective_dim": eff,
        "density_uniformity": dens,
        "cluster_separation": sep,
        "slot_vocab": len(slot2i),
        "value_vocab": len(val2i),
    }

# ──────────────────────────── UTILS ─────────────────────────────────────────

def safe_json(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [safe_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    return str(obj)

# ─────────────────────────── MAIN ────────────────────────────────────────────

def create_comparison_plots(h_res, l_res, out_dir: Path, cfg):
    common = {m for m in h_res if h_res[m]["file_saved"] and l_res.get(m, {}).get("file_saved")}
    if len(common) < 2:
        return None
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(common)), [2] * len(common))
    plt.xticks(range(len(common)), [m.upper() for m in common])
    plt.ylabel("# Successful Envs")
    plt.title("Dim‑Reduction Success (Human+LLM)")
    plt.tight_layout()
    path = out_dir / "method_comparison.png"
    plt.savefig(path, dpi=cfg["plot_dpi"])
    plt.close()
    return {m: True for m in common}


def main():
    parser = argparse.ArgumentParser("State‑Richness Analyzer (rev 7 - Improved Visualization)")
    parser.add_argument("--annotations", required=True, help="annotation JSON file")
    parser.add_argument("--outdir", default="state_richness_results")
    parser.add_argument("--config")
    parser.add_argument("--performance", choices=["fast", "balanced", "accurate"], default="balanced")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-vis-samples", type=int)
    parser.add_argument("--sampling", choices=["smart", "random", "stratified"], default="smart")
    parser.add_argument("--save-formats", nargs="+", choices=["png", "pdf", "svg"], default=["png"])
    parser.add_argument("--max-clusters", type=int)
    parser.add_argument("--cluster-range", nargs=2, type=int)
    parser.add_argument("--no-dim-reduction", action="store_true")
    parser.add_argument("--variance-threshold", type=float)
    parser.add_argument("--min-components", type=int)
    parser.add_argument("--dimension-method", choices=["pca", "elbow"])
    args = parser.parse_args()

    analyzer = ConfigurableStateAnalyzer()
    if args.config and Path(args.config).is_file():
        analyzer.config.update(json.load(open(args.config)))

    # apply CLI overrides
    cfg = analyzer.config
    cfg["performance_mode"] = args.performance
    cfg["sampling_strategy"] = args.sampling
    cfg["save_formats"] = args.save_formats or cfg["save_formats"]
    if args.max_samples:
        cfg["max_samples_clustering"] = args.max_samples
    if args.max_vis_samples:
        cfg["max_samples_visualization"] = args.max_vis_samples
    if args.max_clusters:
        cfg["max_clusters_search"] = args.max_clusters
    if args.cluster_range:
        cfg["cluster_range"] = args.cluster_range
    if args.no_dim_reduction:
        cfg["use_reduced_dims_for_clustering"] = False
    if args.variance_threshold:
        cfg["variance_threshold"] = args.variance_threshold
    if args.min_components:
        cfg["min_components"] = args.min_components
    if args.dimension_method:
        cfg["dimension_method"] = args.dimension_method

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("📊 Loading annotations …")
    data = json.load(open(args.annotations))
    recs = data["annotations"] if isinstance(data, dict) else data
    H = [r for r in recs if r.get("annotation_type") == "human"]
    L = [r for r in recs if r.get("annotation_type") == "llm"]
    if not H or not L:
        raise RuntimeError("Both human and LLM annotations required.")

    total = len(H) + len(L)
    analyzer._auto_configure_for_data_size(total)

    print(f"   Human {len(H):,}, LLM {len(L):,}, total {total:,}")
    print(f"📋 Configuration: {cfg['performance_mode']} mode, {cfg['sampling_strategy']} sampling")

    # vocab & vectors
    slotH = {s: i for i, s in enumerate(sorted({k for r in H for k in r["state_after"]}))}
    valH = {v: i for i, v in enumerate(sorted({str(x) for r in H for x in r["state_after"].values()}))}
    slotL = {s: i for i, s in enumerate(sorted({k for r in L for k in r["state_after"]}))}
    valL = {v: i for i, v in enumerate(sorted({str(x) for r in L for x in r["state_after"].values()}))}
    Xh_full, Xl_full = vecs(H, slotH, valH), vecs(L, slotL, valL)

    Xh, Hs, _ = smart_sample(Xh_full, H, cfg["max_samples_clustering"], cfg["sampling_strategy"])
    Xl, Ls, _ = smart_sample(Xl_full, L, cfg["max_samples_clustering"], cfg["sampling_strategy"])

    print("📈 Computing richness metrics …")
    Rh, Rl = richness_metrics(Xh_full, slotH, valH), richness_metrics(Xl_full, slotL, valL)

    print("🔍 Clustering …")
    Ch = classify_states(Xh, Hs, cfg)
    Cl = classify_states(Xl, Ls, cfg)

    print("🖼  Visualizing with improved settings …")
    print(f"   UMAP: min_dist={cfg['umap_config']['min_dist']}, n_neighbors={cfg['umap_config']['n_neighbors']}")
    print(f"   t-SNE: perplexity={cfg['tsne_config']['perplexity']}, early_exag={cfg['tsne_config']['early_exaggeration']}")
    
    hv = visualize_states(Xh, Ch["labels"], Ch["representatives"], "Human", out_dir, cfg)
    lv = visualize_states(Xl, Cl["labels"], Cl["representatives"], "LLM", out_dir, cfg)
    hv_over = visualize_states(Xh_full, None, [], "Human_overview", out_dir, cfg)
    lv_over = visualize_states(Xl_full, None, [], "LLM_overview", out_dir, cfg)

    cmp = create_comparison_plots(hv, lv, out_dir, cfg)

    print("💾 Saving JSON …")
    res = {
        "configuration": safe_json(cfg),
        "data_info": {
            "total_human_states": len(Xh_full),
            "total_llm_states": len(Xl_full),
            "sampled_human": len(Xh),
            "sampled_llm": len(Xl),
            "available_methods": AVAILABLE_METHODS,
        },
        "visualization": safe_json({
            "human_clustered": hv,
            "llm_clustered": lv,
            "human_overview": hv_over,
            "llm_overview": lv_over,
            "comparison": cmp,
        }),
        "human": {
            "richness": Rh,
            "clustering": safe_json(Ch),
        },
        "llm": {
            "richness": Rl,
            "clustering": safe_json(Cl),
        },
    }
    out_json = out_dir / "analysis_results.json"
    json.dump(res, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("✅ Done →", out_json)
    print(f"📁 Results saved to: {out_dir}")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    main()