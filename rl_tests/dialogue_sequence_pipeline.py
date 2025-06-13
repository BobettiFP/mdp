#!/usr/bin/env python3
"""
Dialogue Sequence Analysis Pipeline
각 dialogue의 state-action sequence를 분석하고 유사한 구조의 dialogue를 찾음
"""
import argparse, json, random, os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ===================== Sequence Representation Methods =====================

class SequenceEncoder:
    """다양한 방법으로 dialogue sequence를 벡터로 인코딩"""
    
    def __init__(self, method='combined'):
        self.method = method
        self.fitted = False
        
    def fit(self, sequences: List[List[Dict]], actions: List[List[str]] = None):
        """모든 sequences를 보고 vocabulary 구축"""
        self.all_states = []
        self.all_actions = []
        
        for seq in sequences:
            for state in seq:
                self.all_states.append(state)
        
        if actions:
            for action_seq in actions:
                self.all_actions.extend(action_seq)
        
        # State vocabulary 구축
        self.state_keys = sorted(set(key for state in self.all_states for key in state.keys()))
        self.state_values = sorted(set(str(val) for state in self.all_states for val in state.values()))
        
        # Action vocabulary 구축
        if self.all_actions:
            self.action_vocab = sorted(set(self.all_actions))
        else:
            self.action_vocab = []
            
        self.fitted = True
        return self
    
    def _state_to_vector(self, state: Dict) -> np.ndarray:
        """개별 state를 벡터로 변환"""
        # Slot presence + Value presence 방식 (기존 코드와 동일)
        vec = np.zeros(len(self.state_keys) + len(self.state_values), dtype=np.int8)
        
        for key in state:
            if key in self.state_keys:
                vec[self.state_keys.index(key)] = 1
                
        for val in state.values():
            val_str = str(val)
            if val_str in self.state_values:
                vec[len(self.state_keys) + self.state_values.index(val_str)] = 1
                
        return vec
    
    def _encode_statistical_features(self, sequence: List[Dict], actions: List[str] = None) -> np.ndarray:
        """통계적 특성으로 sequence 인코딩"""
        features = []
        
        # 시퀀스 길이
        features.append(len(sequence))
        
        # 유니크 state 개수
        unique_states = len(set(str(sorted(state.items())) for state in sequence))
        features.append(unique_states)
        
        # State diversity (entropy)
        state_counts = Counter(str(sorted(state.items())) for state in sequence)
        if len(state_counts) > 1:
            features.append(entropy(list(state_counts.values())))
        else:
            features.append(0.0)
            
        # State transition diversity
        if len(sequence) > 1:
            transitions = []
            for i in range(len(sequence) - 1):
                s1 = str(sorted(sequence[i].items()))
                s2 = str(sorted(sequence[i+1].items()))
                transitions.append(f"{s1}->{s2}")
            
            unique_transitions = len(set(transitions))
            features.append(unique_transitions)
            
            # Transition entropy
            trans_counts = Counter(transitions)
            features.append(entropy(list(trans_counts.values())))
        else:
            features.extend([0.0, 0.0])
            
        # State repetition ratio
        if len(sequence) > 0:
            features.append(unique_states / len(sequence))
        else:
            features.append(0.0)
            
        # Action diversity (if available)
        if actions:
            unique_actions = len(set(actions))
            features.append(unique_actions)
            
            if len(actions) > 1:
                action_counts = Counter(actions)
                features.append(entropy(list(action_counts.values())))
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
            
        return np.array(features, dtype=np.float32)
    
    def _encode_ngram_features(self, sequence: List[Dict], n=2) -> np.ndarray:
        """N-gram 기반 sequence 인코딩"""
        if len(sequence) < n:
            return np.zeros(100, dtype=np.float32)  # 고정 크기 벡터
            
        # State를 string으로 변환
        state_strings = [str(sorted(state.items())) for state in sequence]
        
        # N-gram 생성
        ngrams = []
        for i in range(len(state_strings) - n + 1):
            ngram = tuple(state_strings[i:i+n])
            ngrams.append(ngram)
            
        # N-gram frequency vector
        ngram_counts = Counter(ngrams)
        
        # 고정 크기 벡터로 변환 (해싱 트릭 사용)
        vec = np.zeros(100, dtype=np.float32)
        for ngram, count in ngram_counts.items():
            hash_val = hash(str(ngram)) % 100
            vec[hash_val] += count
            
        # 정규화
        if np.sum(vec) > 0:
            vec = vec / np.sum(vec)
            
        return vec
    
    def _encode_sequence_embedding(self, sequence: List[Dict]) -> np.ndarray:
        """Sequence를 고정 크기 embedding으로 변환"""
        if not sequence:
            return np.zeros(self.get_embedding_dim())
            
        # 각 state를 벡터로 변환
        state_vectors = [self._state_to_vector(state) for state in sequence]
        state_matrix = np.array(state_vectors)
        
        # 다양한 aggregation 방법들
        embeddings = []
        
        # 1. Mean pooling
        embeddings.append(np.mean(state_matrix, axis=0))
        
        # 2. Max pooling
        embeddings.append(np.max(state_matrix, axis=0))
        
        # 3. First and last states
        embeddings.append(state_matrix[0])
        embeddings.append(state_matrix[-1])
        
        # 4. Std pooling (variation)
        if len(state_matrix) > 1:
            embeddings.append(np.std(state_matrix, axis=0))
        else:
            embeddings.append(np.zeros(state_matrix.shape[1]))
            
        return np.concatenate(embeddings)
    
    def get_embedding_dim(self) -> int:
        """임베딩 차원 계산"""
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
            
        state_dim = len(self.state_keys) + len(self.state_values)
        
        if self.method == 'statistical':
            return 8  # 통계적 특성 개수
        elif self.method == 'ngram':
            return 100  # 고정 크기
        elif self.method == 'embedding':
            return state_dim * 5  # mean, max, first, last, std
        elif self.method == 'combined':
            return 8 + 100 + state_dim * 5  # 모든 방법 결합
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def transform(self, sequences: List[List[Dict]], actions: List[List[str]] = None) -> np.ndarray:
        """Sequences를 벡터들로 변환"""
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
            
        encoded_sequences = []
        
        for i, sequence in enumerate(sequences):
            action_seq = actions[i] if actions else None
            
            if self.method == 'statistical':
                enc = self._encode_statistical_features(sequence, action_seq)
            elif self.method == 'ngram':
                enc = self._encode_ngram_features(sequence)
            elif self.method == 'embedding':
                enc = self._encode_sequence_embedding(sequence)
            elif self.method == 'combined':
                stat_enc = self._encode_statistical_features(sequence, action_seq)
                ngram_enc = self._encode_ngram_features(sequence)
                emb_enc = self._encode_sequence_embedding(sequence)
                enc = np.concatenate([stat_enc, ngram_enc, emb_enc])
            else:
                raise ValueError(f"Unknown method: {self.method}")
                
            encoded_sequences.append(enc)
            
        return np.array(encoded_sequences)

# ===================== Sequence Distance Metrics =====================

def dtw_distance(seq1: List[Dict], seq2: List[Dict], state_encoder) -> float:
    """Dynamic Time Warping distance between sequences"""
    # State sequences를 벡터로 변환
    vec1 = np.array([state_encoder._state_to_vector(s) for s in seq1])
    vec2 = np.array([state_encoder._state_to_vector(s) for s in seq2])
    
    n, m = len(vec1), len(vec2)
    
    # DTW matrix 초기화
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # DTW 계산
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(vec1[i-1] - vec2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )
    
    return dtw_matrix[n, m] / max(n, m)  # 정규화

def edit_distance(seq1: List[Dict], seq2: List[Dict]) -> float:
    """Edit distance between state sequences"""
    # State를 string으로 변환
    str1 = [str(sorted(s.items())) for s in seq1]
    str2 = [str(sorted(s.items())) for s in seq2]
    
    n, m = len(str1), len(str2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    # 초기화
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    # DP 계산
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[n][m] / max(n, m)  # 정규화

# ===================== Main Analysis Functions =====================

def extract_dialogue_sequences(data: List[Dict]) -> Tuple[List[List[Dict]], List[List[str]], List[str]]:
    """데이터에서 dialogue별 sequence 추출"""
    dialogues = defaultdict(list)
    dialogue_ids = []
    
    # annotation_type별로 다른 처리
    human_data = [r for r in data if r.get('annotation_type') == 'human']
    llm_data = [r for r in data if r.get('annotation_type') == 'llm']
    
    # Human 데이터 처리 (기존 방식)
    for record in human_data:
        dialogue_id = record.get('dialogue_id', record.get('id', 'unknown'))
        dialogues[f"human_{dialogue_id}"].append(record)
    
    # LLM 데이터 처리 (특별한 로직 적용)
    llm_dialogues = group_llm_dialogues(llm_data)
    for dialogue_id, records in llm_dialogues.items():
        dialogues[f"llm_{dialogue_id}"] = records
    
    sequences = []
    action_sequences = []
    
    for dialogue_id, records in dialogues.items():
        # Turn 순서로 정렬 (turn_id 우선, 없으면 turn, step 순)
        records.sort(key=lambda x: x.get('turn_id', x.get('turn', x.get('step', 0))))
        
        # State sequence 추출
        state_seq = [r['state_after'] for r in records if 'state_after' in r]
        action_seq = [r.get('action', r.get('annotation', '')) for r in records]
        
        if state_seq:  # 빈 시퀀스 제외
            sequences.append(state_seq)
            action_sequences.append(action_seq)
            dialogue_ids.append(dialogue_id)
    
    return sequences, action_sequences, dialogue_ids

def group_llm_dialogues(llm_data: List[Dict]) -> Dict[str, List[Dict]]:
    """LLM 데이터를 dialogue로 그룹화하는 특별한 로직"""
    dialogues = defaultdict(list)
    
    # 방법 1: state transition을 기반으로 dialogue 구분
    current_dialogue_id = 0
    current_dialogue = []
    
    for i, record in enumerate(llm_data):
        current_dialogue.append(record)
        
        # 새로운 dialogue 시작 조건들을 체크
        is_new_dialogue = False
        
        # 조건 1: state_before가 비어있고 이전에 데이터가 있었다면
        if (record.get('state_before', {}) == {} and 
            len(current_dialogue) > 1):
            is_new_dialogue = True
        
        # 조건 2: 이전 record의 state_after와 현재 record의 state_before가 완전히 다르다면
        elif (i > 0 and 
              current_dialogue and len(current_dialogue) > 1):
            prev_record = llm_data[i-1]
            prev_state = prev_record.get('state_after', {})
            curr_state_before = record.get('state_before', {})
            
            # 도메인이 완전히 바뀌었는지 체크
            prev_domains = set(key.split('_')[0] for key in prev_state.keys())
            curr_domains = set(key.split('_')[0] for key in curr_state_before.keys())
            
            if (prev_domains and curr_domains and 
                not prev_domains.intersection(curr_domains)):
                is_new_dialogue = True
        
        # 조건 3: 일정 길이마다 강제로 dialogue 구분 (너무 긴 dialogue 방지)
        elif len(current_dialogue) > 20:  # 20턴 넘으면 새 dialogue
            is_new_dialogue = True
            
        # 새 dialogue 시작
        if is_new_dialogue and len(current_dialogue) > 1:
            # 현재 record는 새 dialogue에 포함
            dialogues[f"auto_{current_dialogue_id}"] = current_dialogue[:-1]
            current_dialogue = [record]
            current_dialogue_id += 1
    
    # 마지막 dialogue 추가
    if current_dialogue:
        dialogues[f"auto_{current_dialogue_id}"] = current_dialogue
    
    # 각 dialogue의 길이가 1인 경우들을 별도 처리
    single_turn_dialogues = {}
    multi_turn_dialogues = {}
    
    for dialogue_id, records in dialogues.items():
        if len(records) == 1:
            # 단일 턴은 개별 dialogue로 처리
            single_turn_dialogues[f"single_{dialogue_id}"] = records
        else:
            multi_turn_dialogues[dialogue_id] = records
    
    # 결과 합치기
    result = {**multi_turn_dialogues, **single_turn_dialogues}
    
    print(f"   └ LLM dialogue 자동 구분: {len(result)}개 dialogues 생성")
    print(f"     - Multi-turn: {len(multi_turn_dialogues)}개")
    print(f"     - Single-turn: {len(single_turn_dialogues)}개")
    
    return result

def compute_sequence_similarities(sequences: List[List[Dict]], 
                                method: str = 'combined',
                                distance_metric: str = 'euclidean') -> np.ndarray:
    """Sequence 간 유사도 행렬 계산"""
    
    # Sequence encoder 초기화 및 학습
    encoder = SequenceEncoder(method=method)
    encoder.fit(sequences)
    
    # Sequences를 벡터로 변환
    if method in ['statistical', 'ngram', 'embedding', 'combined']:
        X = encoder.transform(sequences)
        # 거리 행렬 계산
        distances = pairwise_distances(X, metric=distance_metric)
        
    elif method == 'dtw':
        # DTW distance 계산
        n = len(sequences)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = dtw_distance(sequences[i], sequences[j], encoder)
                distances[i, j] = distances[j, i] = dist
                
    elif method == 'edit':
        # Edit distance 계산
        n = len(sequences)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = edit_distance(sequences[i], sequences[j])
                distances[i, j] = distances[j, i] = dist
    
    # Distance를 similarity로 변환 (옵션)
    # similarities = 1 / (1 + distances)
    
    return distances, X if method in ['statistical', 'ngram', 'embedding', 'combined'] else None

def find_optimal_clusters_for_sequences(X: np.ndarray, max_clusters: int = None) -> int:
    """Sequence clustering을 위한 최적 클러스터 수 결정"""
    if len(X) < 3:
        return 1
    
    max_k = max_clusters or min(max(len(X) // 5, 3), len(X) - 1)
    max_k = max(2, min(max_k, len(X) - 1))
    
    if max_k < 2:
        return 1
    
    K_range = range(2, max_k + 1)
    silhouette_scores = []
    inertias = []
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(X, labels)
            silhouette_scores.append(sil_score)
            inertias.append(kmeans.inertia_)
        else:
            silhouette_scores.append(-1)
            inertias.append(float('inf'))
    
    if not silhouette_scores or max(silhouette_scores) < 0:
        return 2
    
    # 최고 실루엣 스코어를 가진 클러스터 수 선택
    optimal_k = K_range[np.argmax(silhouette_scores)]
    return optimal_k

def cluster_dialogues(X: np.ndarray, dialogue_ids: List[str], optimal_k: int) -> Dict:
    """Dialogue sequence clustering"""
    if len(X) < 2:
        return {
            'labels': np.zeros(len(X)),
            'cluster_info': {},
            'n_clusters': 1 if len(X) > 0 else 0,
            'method': 'single_dialogue'
        }
    
    # K-means clustering
    kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # 실루엣 스코어 계산
    silhouette = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1
    
    # 클러스터 정보 구성
    cluster_info = {}
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_dialogues = [dialogue_ids[i] for i in np.where(cluster_mask)[0]]
        cluster_points = X[cluster_mask]
        
        # 클러스터 중심에 가장 가까운 dialogue 찾기
        center = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_points - center, axis=1)
        representative_idx = np.argmin(distances)
        representative_dialogue = cluster_dialogues[representative_idx]
        
        cluster_info[f'cluster_{cluster_id}'] = {
            'size': int(np.sum(cluster_mask)),
            'dialogues': cluster_dialogues,
            'representative_dialogue': representative_dialogue,
            'avg_distance_to_center': float(np.mean(distances)),
            'intra_cluster_diversity': float(np.std(distances))
        }
    
    return {
        'labels': labels,
        'cluster_info': cluster_info,
        'n_clusters': len(np.unique(labels)),
        'centroids': kmeans.cluster_centers_,
        'method': 'kmeans',
        'silhouette_score': silhouette
    }

def visualize_dialogue_space(X: np.ndarray, labels: np.ndarray, dialogue_ids: List[str], 
                           title: str, output_path: Path):
    """Dialogue space t-SNE 시각화"""
    if len(X) < 3:
        return
    
    try:
        # t-SNE 임베딩
        perplexity = min(30, len(X) // 3)
        if perplexity < 2:
            perplexity = 2
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embedding = tsne.fit_transform(X)
    except:
        # t-SNE 실패시 PCA 사용
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(X)
    
    plt.figure(figsize=(12, 8))
    
    # 클러스터별로 다른 색상과 마커 사용
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, cluster_id in enumerate(unique_labels):
        cluster_mask = labels == cluster_id
        cluster_points = embedding[cluster_mask]
        
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=[colors[i]], label=f'Cluster {cluster_id}', 
                   s=50, alpha=0.7)
        
        # 몇 개 dialogue ID 표시 (너무 많으면 생략)
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) <= 10:  # 10개 이하만 라벨 표시
            for idx in cluster_indices:
                plt.annotate(dialogue_ids[idx], 
                           (embedding[idx, 0], embedding[idx, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f'{title} - Dialogue Sequence Clustering')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ===================== Main Function =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="JSON file with dialogue annotations")
    parser.add_argument("--outdir", default="dialogue_sequence_results", help="Output directory")
    parser.add_argument("--encoding_method", default="combined", 
                       choices=['statistical', 'ngram', 'embedding', 'combined', 'dtw', 'edit'],
                       help="Sequence encoding method")
    parser.add_argument("--distance_metric", default="euclidean",
                       choices=['euclidean', 'cosine', 'manhattan'],
                       help="Distance metric for clustering")
    parser.add_argument("--max_clusters", type=int, default=None,
                       help="Maximum number of clusters")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드
    with open(args.annotations, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        records = data.get("annotations", [])
    else:
        records = data
    
    # Human과 LLM 데이터 분리
    human_records = [r for r in records if r.get("annotation_type") == "human"]
    llm_records = [r for r in records if r.get("annotation_type") == "llm"]
    
    if not human_records or not llm_records:
        raise ValueError("annotation_type이 'human' 또는 'llm'인 데이터가 부족합니다.")
    
    print(f"📊 데이터 로드 완료: Human {len(human_records)}, LLM {len(llm_records)}")
    
    # Dialogue sequence 추출
    print("🔍 Dialogue sequences 추출 중...")
    all_sequences, all_actions, all_dialogue_ids = extract_dialogue_sequences(records)
    
    # Human과 LLM sequences 분리
    human_sequences, human_actions, human_dialogue_ids = [], [], []
    llm_sequences, llm_actions, llm_dialogue_ids = [], [], []
    
    for seq, acts, did in zip(all_sequences, all_actions, all_dialogue_ids):
        if did.startswith('human_'):
            human_sequences.append(seq)
            human_actions.append(acts)
            human_dialogue_ids.append(did)
        elif did.startswith('llm_'):
            llm_sequences.append(seq)
            llm_actions.append(acts)
            llm_dialogue_ids.append(did)
    
    print(f"   Human dialogues: {len(human_sequences)}")
    print(f"   LLM dialogues: {len(llm_sequences)}")
    
    # 분석 및 클러스터링
    results = {}
    
    for data_type, sequences, dialogue_ids in [
        ("human", human_sequences, human_dialogue_ids),
        ("llm", llm_sequences, llm_dialogue_ids)
    ]:
        print(f"\n🔍 {data_type.upper()} sequences 분석 중...")
        
        if len(sequences) < 2:
            print(f"   ⚠️  {data_type} dialogue 수가 부족합니다 (최소 2개 필요)")
            continue
        
        # 유사도 계산
        distances, X = compute_sequence_similarities(
            sequences, 
            method=args.encoding_method,
            distance_metric=args.distance_metric
        )
        
        if X is not None:
            # 클러스터링
            optimal_k = find_optimal_clusters_for_sequences(X, args.max_clusters)
            clustering_result = cluster_dialogues(X, dialogue_ids, optimal_k)
            
            print(f"   └ 클러스터 수: {clustering_result['n_clusters']}, "
                  f"실루엣 스코어: {clustering_result['silhouette_score']:.3f}")
            
            # 시각화
            visualize_dialogue_space(
                X, clustering_result['labels'], dialogue_ids,
                data_type.upper(), output_dir / f"dialogue_space_{data_type}.png"
            )
            
            # 결과 저장
            results[data_type] = {
                'n_dialogues': len(sequences),
                'encoding_method': args.encoding_method,
                'clustering': {
                    'method': clustering_result['method'],
                    'n_clusters': clustering_result['n_clusters'],
                    'silhouette_score': clustering_result['silhouette_score'],
                    'cluster_info': clustering_result['cluster_info']
                }
            }
            
            # 클러스터 정보 출력
            print(f"\n📊 {data_type.upper()} 클러스터 분석:")
            for cluster_name, info in clustering_result['cluster_info'].items():
                print(f"  {cluster_name}: {info['size']}개 dialogues")
                print(f"    대표 dialogue: {info['representative_dialogue']}")
                print(f"    다양성: {info['intra_cluster_diversity']:.3f}")
    
    # 결과 저장
    output_file = output_dir / "dialogue_sequence_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 분석 완료. 결과 저장: {output_file}")

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    main()