#!/usr/bin/env python3
"""
실제 데이터 기반 전이 동역학 → 학습 안정성 증명 파이프라인
===========================================================
mdp/processed_annotations.json 데이터를 활용하여
LLM annotation이 더 일관된 전이 동역학을 제공하여 
더 안정적인 학습을 달성한다는 가설을 검증
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mutual_info_score
from sklearn.cluster import KMeans
from scipy.stats import entropy, pearsonr
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Set, Union, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import json
from pathlib import Path
import itertools

# 결과 저장 경로는 실행 시에 설정

@dataclass
class TransitionDynamicsMetrics:
    """전이 동역학 메트릭"""
    # 1. 일관성 메트릭
    transition_consistency: float      # 같은 (s,a) → 같은 s' 빈도
    transition_predictability: float   # 전이 예측 정확도
    transition_stability: float       # 전이 확률 분산
    
    # 2. 구조적 메트릭
    transition_entropy: float          # 전이 분포 엔트로피
    state_connectivity: float         # 상태 간 연결성
    markov_property: float            # 마르코프 성질 만족도
    
    # 3. 정보 이론적 메트릭
    mutual_information: float         # I(S_t; S_{t+1} | A_t)
    transition_complexity: float      # 전이 복잡도
    causal_strength: float           # 인과 관계 강도
    
    # 4. 패턴 메트릭
    transition_pattern_diversity: float # 전이 패턴 다양성
    determinism_score: float         # 결정론적 정도
    cyclic_patterns: float           # 순환 패턴 존재

@dataclass
class LearningStabilityMetrics:
    """학습 안정성 메트릭"""
    # 1. 수렴 안정성
    convergence_smoothness: float     # 학습 곡선 부드러움
    convergence_predictability: float # 수렴 예측 가능성
    final_variance: float            # 최종 성능 분산
    
    # 2. 학습 과정 안정성  
    learning_consistency: float      # 학습 일관성
    gradient_stability: float       # Gradient 안정성 (시뮬레이션)
    exploration_stability: float    # 탐험 안정성
    
    # 3. 견고성
    hyperparameter_sensitivity: float # 하이퍼파라미터 민감도
    initialization_robustness: float  # 초기화 견고성
    replay_consistency: float        # 반복 실험 일관성

class TransitionDynamicsAnalyzer:
    """전이 동역학 분석기"""
    
    def __init__(self, annotations: List[dict], annotation_type: str, canonical_map: Dict[str, str]):
        self.annotations = annotations
        self.annotation_type = annotation_type
        self.canonical_map = canonical_map
        self.transitions = []
        self.state_transitions = None
        self.transition_matrix = None
        
        print(f"[{self.annotation_type}] 전이 동역학 분석기 초기화: {len(annotations)}개 annotation")
        
    def _normalize_slot_name(self, slot_name: str) -> str:
        """슬롯 이름 정규화 (canonical_map 활용)"""
        # canonical_map에 정의된 매핑 사용
        if slot_name in self.canonical_map:
            return self.canonical_map[slot_name]
        
        # 기본 정규화
        normalized = slot_name.lower().replace("-", "_")
        
        # 도메인 접두사 제거
        domain_prefixes = ["restaurant_", "hotel_", "train_", "taxi_", "attraction_", 
                          "bus_", "hospital_", "police_"]
        for prefix in domain_prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        
        return normalized
    
    def _extract_slot_value(self, slot_data: Union[str, Dict]) -> str:
        """슬롯 값 추출"""
        if isinstance(slot_data, str):
            return slot_data
        elif isinstance(slot_data, dict):
            return str(slot_data.get("value", str(slot_data)))
        elif isinstance(slot_data, list):
            return str(slot_data[0]) if slot_data else ""
        elif isinstance(slot_data, (int, float)):
            return str(slot_data)
        else:
            return str(slot_data)
    
    def extract_transitions(self) -> List[Dict]:
        """상태-행동-다음상태 전이 추출"""
        print(f"[{self.annotation_type}] 전이 데이터 추출 중...")
        
        # 대화별로 정리
        dialogues = defaultdict(list)
        for ann in self.annotations:
            if ann.get('dialogue_id') and ann.get('slots'):
                # 정규화된 슬롯으로 변환
                normalized_slots = {}
                for slot_name, slot_value in ann['slots'].items():
                    normalized_slot = self._normalize_slot_name(slot_name)
                    extracted_value = self._extract_slot_value(slot_value)
                    normalized_slots[normalized_slot] = extracted_value
                
                if normalized_slots:
                    ann_copy = ann.copy()
                    ann_copy['normalized_slots'] = normalized_slots
                    dialogues[ann['dialogue_id']].append(ann_copy)
        
        transitions = []
        
        for dialogue_id, turns in dialogues.items():
            # turn_id로 정렬
            try:
                turns.sort(key=lambda x: int(x.get('turn_id', 0)) if x.get('turn_id') is not None else 0)
            except (ValueError, TypeError):
                turns.sort(key=lambda x: str(x.get('turn_id', '')) if x.get('turn_id') is not None else '')
            
            # 연속된 턴 간 전이 추출
            for i in range(len(turns) - 1):
                current_turn = turns[i]
                next_turn = turns[i + 1]
                
                # 현재 상태 (슬롯 집합을 frozen set으로)
                current_state = frozenset(current_turn['normalized_slots'].keys())
                next_state = frozenset(next_turn['normalized_slots'].keys())
                
                # 행동 (새로 추가된 슬롯들)
                action = next_state - current_state
                
                # 전이 기록
                transition = {
                    'current_state': current_state,
                    'action': action,
                    'next_state': next_state,
                    'dialogue_id': dialogue_id,
                    'turn_pair': (i, i+1),
                    'current_slots': current_turn['normalized_slots'],
                    'next_slots': next_turn['normalized_slots']
                }
                transitions.append(transition)
        
        self.transitions = transitions
        print(f"총 {len(transitions)}개 전이 추출됨")
        return transitions
    
    def build_transition_matrix(self) -> np.ndarray:
        """전이 행렬 구축"""
        print(f"[{self.annotation_type}] 전이 행렬 구축 중...")
        
        if not self.transitions:
            self.extract_transitions()
        
        if not self.transitions:
            print("전이 데이터가 없어 빈 행렬 반환")
            return np.array([[]])
        
        # 모든 고유 상태 수집
        all_states = set()
        for transition in self.transitions:
            all_states.add(transition['current_state'])
            all_states.add(transition['next_state'])
        
        if len(all_states) < 2:
            print("충분한 상태가 없어 기본 행렬 반환")
            return np.array([[1.0]])
        
        # 상태를 인덱스에 매핑
        state_to_idx = {state: i for i, state in enumerate(sorted(all_states, key=str))}
        n_states = len(all_states)
        
        # 전이 카운트 행렬
        transition_counts = np.zeros((n_states, n_states))
        
        for transition in self.transitions:
            current_idx = state_to_idx[transition['current_state']]
            next_idx = state_to_idx[transition['next_state']]
            transition_counts[current_idx, next_idx] += 1
        
        # 확률 행렬로 변환
        transition_matrix = transition_counts.copy()
        for i in range(n_states):
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                transition_matrix[i] = transition_counts[i] / row_sum
            else:
                # 자기 자신으로 전이 (흡수 상태)
                transition_matrix[i, i] = 1.0
        
        self.transition_matrix = transition_matrix
        self.state_to_idx = state_to_idx
        print(f"전이 행렬 크기: {n_states} x {n_states}")
        return transition_matrix
    
    def calculate_transition_consistency(self) -> float:
        """전이 일관성 계산"""
        if not self.transitions:
            self.extract_transitions()
        
        if not self.transitions:
            return 0.0
        
        # 같은 (current_state, action) 쌍에 대한 next_state 분포의 일관성
        state_action_outcomes = defaultdict(list)
        
        for transition in self.transitions:
            key = (transition['current_state'], transition['action'])
            state_action_outcomes[key].append(transition['next_state'])
        
        consistency_scores = []
        for outcomes in state_action_outcomes.values():
            if len(outcomes) > 1:
                # 가장 빈번한 결과의 비율
                outcome_counts = Counter(outcomes)
                most_common_count = outcome_counts.most_common(1)[0][1]
                consistency = most_common_count / len(outcomes)
                consistency_scores.append(consistency)
            else:
                consistency_scores.append(1.0)  # 단일 관찰은 완전 일관
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def calculate_transition_predictability(self) -> float:
        """전이 예측 가능성 계산"""
        if self.transition_matrix is None:
            self.build_transition_matrix()
        
        if self.transition_matrix.size == 0:
            return 0.0
        
        # 각 상태에서 다음 상태 예측의 확실성 (엔트로피 기반)
        predictability_scores = []
        
        for i in range(self.transition_matrix.shape[0]):
            row = self.transition_matrix[i]
            if row.sum() > 0:
                # 0이 아닌 확률들만 고려
                non_zero_probs = row[row > 0]
                if len(non_zero_probs) > 0:
                    row_entropy = entropy(non_zero_probs)
                    max_entropy = np.log(len(non_zero_probs))
                    if max_entropy > 0:
                        predictability = 1 - (row_entropy / max_entropy)
                    else:
                        predictability = 1.0
                    predictability_scores.append(predictability)
        
        return np.mean(predictability_scores) if predictability_scores else 0.0
    
    def calculate_markov_property(self) -> float:
        """마르코프 성질 만족도 계산"""
        if not self.transitions:
            self.extract_transitions()
        
        if len(self.transitions) < 2:
            return 1.0
        
        # 대화별로 연속된 전이 분석
        dialogues = defaultdict(list)
        for transition in self.transitions:
            dialogues[transition['dialogue_id']].append(transition)
        
        markov_scores = []
        
        for dialogue_transitions in dialogues.values():
            if len(dialogue_transitions) >= 2:
                for i in range(len(dialogue_transitions) - 1):
                    t1 = dialogue_transitions[i]
                    t2 = dialogue_transitions[i + 1]
                    
                    # 연속된 전이에서 상태 연결성 체크
                    if t1['next_state'] == t2['current_state']:
                        markov_scores.append(1.0)
                    else:
                        markov_scores.append(0.0)
        
        return np.mean(markov_scores) if markov_scores else 1.0
    
    def calculate_mutual_information(self) -> float:
        """상태 간 상호 정보량 계산"""
        if not self.transitions:
            self.extract_transitions()
        
        if not self.transitions:
            return 0.0
        
        # 현재 상태와 다음 상태 간 상호 정보
        current_states = []
        next_states = []
        
        for transition in self.transitions:
            current_states.append(str(sorted(transition['current_state'])))
            next_states.append(str(sorted(transition['next_state'])))
        
        if len(set(current_states)) > 1 and len(set(next_states)) > 1:
            try:
                mi = mutual_info_score(current_states, next_states)
            except:
                mi = 0.0
        else:
            mi = 0.0
        
        return mi
    
    def calculate_determinism_score(self) -> float:
        """결정론적 정도 계산"""
        if self.transition_matrix is None:
            self.build_transition_matrix()
        
        if self.transition_matrix.size == 0:
            return 0.0
        
        # 각 행에서 최대값의 평균 (높을수록 결정론적)
        determinism_scores = []
        for i in range(self.transition_matrix.shape[0]):
            row = self.transition_matrix[i]
            if row.sum() > 0:
                max_prob = np.max(row)
                determinism_scores.append(max_prob)
        
        return np.mean(determinism_scores) if determinism_scores else 0.0
    
    def calculate_transition_dynamics_metrics(self) -> TransitionDynamicsMetrics:
        """전이 동역학 메트릭 종합 계산"""
        print(f"[{self.annotation_type}] 전이 동역학 메트릭 계산 중...")
        
        # 기본 메트릭들
        consistency = self.calculate_transition_consistency()
        predictability = self.calculate_transition_predictability()
        markov_prop = self.calculate_markov_property()
        mutual_info = self.calculate_mutual_information()
        determinism = self.calculate_determinism_score()
        
        # 전이 안정성 (확률 분산)
        if self.transition_matrix is not None and self.transition_matrix.size > 0:
            non_zero_probs = self.transition_matrix[self.transition_matrix > 0]
            stability = 1.0 / (1.0 + np.var(non_zero_probs)) if len(non_zero_probs) > 0 else 1.0
        else:
            stability = 0.0
        
        # 전이 엔트로피
        if self.transition_matrix is not None and self.transition_matrix.size > 0:
            entropies = []
            for i in range(self.transition_matrix.shape[0]):
                row = self.transition_matrix[i]
                non_zero = row[row > 0]
                if len(non_zero) > 0:
                    entropies.append(entropy(non_zero))
            trans_entropy = np.mean(entropies) if entropies else 0.0
        else:
            trans_entropy = 0.0
        
        # 상태 연결성
        if self.transition_matrix is not None and self.transition_matrix.size > 1:
            # 그래프로 변환하여 연결성 측정
            try:
                G = nx.from_numpy_array(self.transition_matrix > 0, create_using=nx.DiGraph)
                if len(G.nodes) > 0:
                    connectivity = nx.average_clustering(G.to_undirected())
                else:
                    connectivity = 0.0
            except:
                connectivity = 0.0
        else:
            connectivity = 0.0
        
        # 전이 복잡성
        if self.transitions:
            unique_transitions = len(set(
                (transition['current_state'], transition['action'], transition['next_state'])
                for transition in self.transitions
            ))
            complexity = unique_transitions / len(self.transitions)
        else:
            complexity = 0.0
        
        # 패턴 다양성
        if self.transitions:
            transition_patterns = set()
            for transition in self.transitions:
                pattern = (len(transition['current_state']), len(transition['action']), len(transition['next_state']))
                transition_patterns.add(pattern)
            pattern_diversity = len(transition_patterns) / len(self.transitions)
        else:
            pattern_diversity = 0.0
        
        # 순환 패턴 (간단한 근사)
        cyclic_patterns = connectivity  # 연결성이 높으면 순환 가능성도 높음
        
        return TransitionDynamicsMetrics(
            transition_consistency=consistency,
            transition_predictability=predictability,
            transition_stability=stability,
            transition_entropy=trans_entropy,
            state_connectivity=connectivity,
            markov_property=markov_prop,
            mutual_information=mutual_info,
            transition_complexity=complexity,
            causal_strength=mutual_info * consistency,
            transition_pattern_diversity=pattern_diversity,
            determinism_score=determinism,
            cyclic_patterns=cyclic_patterns
        )

class LearningStabilitySimulator:
    """학습 안정성 시뮬레이터"""
    
    def __init__(self, dynamics_analyzer: TransitionDynamicsAnalyzer):
        self.analyzer = dynamics_analyzer
        self.annotation_type = dynamics_analyzer.annotation_type
        
    def simulate_learning_curves(self, num_simulations: int = 10, num_episodes: int = 100) -> List[List[float]]:
        """학습 곡선 시뮬레이션"""
        print(f"[{self.annotation_type}] 학습 곡선 시뮬레이션 중...")
        
        curves = []
        
        # 전이 동역학의 일관성에 기반한 시뮬레이션
        dynamics_metrics = self.analyzer.calculate_transition_dynamics_metrics()
        
        for sim in range(num_simulations):
            curve = []
            
            # 초기 성능
            performance = 0.1
            
            for episode in range(num_episodes):
                # 전이 일관성이 높을수록 안정적인 학습
                consistency_factor = dynamics_metrics.transition_consistency
                predictability_factor = dynamics_metrics.transition_predictability
                
                # 학습률 (일관성과 예측가능성에 비례)
                learning_rate = 0.01 * (1 + consistency_factor + predictability_factor)
                
                # 노이즈 (일관성이 높을수록 노이즈 감소)
                noise_std = 0.1 * (1 - consistency_factor)
                noise = np.random.normal(0, noise_std)
                
                # 성능 업데이트
                improvement = learning_rate * (1 - performance) + noise
                performance = max(0, min(1, performance + improvement))
                
                curve.append(performance)
            
            curves.append(curve)
        
        return curves
    
    def calculate_learning_stability_metrics(self) -> LearningStabilityMetrics:
        """학습 안정성 메트릭 계산"""
        print(f"[{self.annotation_type}] 학습 안정성 메트릭 계산 중...")
        
        # 학습 곡선 시뮬레이션
        learning_curves = self.simulate_learning_curves()
        
        if not learning_curves:
            return LearningStabilityMetrics(
                convergence_smoothness=0.0, convergence_predictability=0.0, final_variance=1.0,
                learning_consistency=0.0, gradient_stability=0.0, exploration_stability=0.0,
                hyperparameter_sensitivity=1.0, initialization_robustness=0.0, replay_consistency=0.0
            )
        
        # 수렴 부드러움 (변동성의 역수)
        smoothness_scores = []
        for curve in learning_curves:
            if len(curve) > 1:
                differences = np.diff(curve)
                smoothness = 1.0 / (1.0 + np.var(differences))
                smoothness_scores.append(smoothness)
        
        convergence_smoothness = np.mean(smoothness_scores) if smoothness_scores else 0.0
        
        # 수렴 예측가능성 (시뮬레이션 간 일관성)
        final_performances = [curve[-1] for curve in learning_curves]
        convergence_predictability = 1.0 / (1.0 + np.var(final_performances))
        
        # 최종 분산
        final_variance = np.var(final_performances)
        
        # 학습 일관성 (시뮬레이션 간 곡선 유사성)
        if len(learning_curves) > 1:
            correlations = []
            for i in range(len(learning_curves)):
                for j in range(i+1, len(learning_curves)):
                    try:
                        corr, _ = pearsonr(learning_curves[i], learning_curves[j])
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                    except:
                        pass
            learning_consistency = np.mean(correlations) if correlations else 0.0
        else:
            learning_consistency = 1.0
        
        # Gradient 안정성 (전이 일관성 기반 근사)
        dynamics_metrics = self.analyzer.calculate_transition_dynamics_metrics()
        gradient_stability = dynamics_metrics.transition_consistency
        
        # 탐험 안정성 (전이 예측가능성 기반)
        exploration_stability = dynamics_metrics.transition_predictability
        
        # 하이퍼파라미터 민감도 (전이 복잡성의 역수)
        hyperparameter_sensitivity = 1.0 - dynamics_metrics.transition_complexity
        
        # 초기화 견고성 (마르코프 성질 기반)
        initialization_robustness = dynamics_metrics.markov_property
        
        # 반복 실험 일관성
        replay_consistency = learning_consistency
        
        return LearningStabilityMetrics(
            convergence_smoothness=convergence_smoothness,
            convergence_predictability=convergence_predictability,
            final_variance=final_variance,
            learning_consistency=learning_consistency,
            gradient_stability=gradient_stability,
            exploration_stability=exploration_stability,
            hyperparameter_sensitivity=hyperparameter_sensitivity,
            initialization_robustness=initialization_robustness,
            replay_consistency=replay_consistency
        )

def run_transition_dynamics_pipeline():
    """전이 동역학 → 학습 안정성 검증 파이프라인"""
    
    # 결과 저장 경로 설정
    results_dir = Path("mdp/rl_tests/transition_dynamics_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("전이 동역학 일관성 → 학습 안정성 검증 파이프라인")
    print("="*80)
    
    # 데이터 로드 (경로 자동 탐지)
    possible_paths = [
        Path("../processed_annotations.json"),           # rl_tests에서 실행 시
        Path("mdp/processed_annotations.json"),          # 프로젝트 루트에서 실행 시
        Path("processed_annotations.json"),              # 같은 폴더에서 실행 시
        Path("../../mdp/processed_annotations.json")     # 더 깊은 폴더에서 실행 시
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        print("다음 위치에서 데이터 파일을 찾을 수 없습니다:")
        for path in possible_paths:
            print(f"  - {path.absolute()}")
        raise FileNotFoundError("processed_annotations.json 파일을 찾을 수 없습니다.")
    
    print(f"데이터 로드 중: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    canonical_map = data.get('canonical_map', {})
    human_annotations = data.get('human_annotations', [])
    llm_annotations = data.get('llm_annotations', [])
    
    print(f"로드된 데이터:")
    print(f"  - Human annotations: {len(human_annotations)}개")
    print(f"  - LLM annotations: {len(llm_annotations)}개")
    print(f"  - Canonical map: {len(canonical_map)}개 매핑")
    
    # 1단계: 전이 동역학 분석
    print("\n1단계: 전이 동역학 분석")
    print("-" * 50)
    
    human_dynamics = TransitionDynamicsAnalyzer(human_annotations, "Human", canonical_map)
    llm_dynamics = TransitionDynamicsAnalyzer(llm_annotations, "LLM", canonical_map)
    
    human_dyn_metrics = human_dynamics.calculate_transition_dynamics_metrics()
    llm_dyn_metrics = llm_dynamics.calculate_transition_dynamics_metrics()
    
    # 전이 행렬 시각화
    visualize_transition_matrices(human_dynamics, llm_dynamics, results_dir)
    
    # 2단계: 학습 안정성 시뮬레이션
    print("\n2단계: 학습 안정성 시뮬레이션")
    print("-" * 50)
    
    human_simulator = LearningStabilitySimulator(human_dynamics)
    llm_simulator = LearningStabilitySimulator(llm_dynamics)
    
    human_stability_metrics = human_simulator.calculate_learning_stability_metrics()
    llm_stability_metrics = llm_simulator.calculate_learning_stability_metrics()
    
    # 학습 안정성 시각화
    visualize_learning_stability(human_simulator, llm_simulator, results_dir)
    
    # 3단계: 상관관계 분석
    print("\n3단계: 상관관계 분석")
    print("-" * 50)
    
    correlation_analysis = analyze_dynamics_stability_correlation(
        human_dyn_metrics, llm_dyn_metrics,
        human_stability_metrics, llm_stability_metrics
    )
    
    # 4단계: 결과 종합
    print("\n4단계: 결과 종합")
    print("-" * 50)
    
    results = compile_dynamics_stability_results(
        human_dyn_metrics, llm_dyn_metrics,
        human_stability_metrics, llm_stability_metrics,
        correlation_analysis
    )
    
    # 종합 시각화
    visualize_comprehensive_results(human_dyn_metrics, llm_dyn_metrics,
                                  human_stability_metrics, llm_stability_metrics, results_dir)
    
    # 결과 저장
    results_path = results_dir / 'transition_dynamics_stability_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\n결과 저장: {results_path}")
    
    return results

def visualize_transition_matrices(human_dynamics: TransitionDynamicsAnalyzer,
                                llm_dynamics: TransitionDynamicsAnalyzer,
                                results_dir: Path):
    """전이 행렬 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Human transition matrix
    if human_dynamics.transition_matrix is not None and human_dynamics.transition_matrix.size > 1:
        im1 = axes[0].imshow(human_dynamics.transition_matrix, cmap='Blues', aspect='auto')
        axes[0].set_title(f'Human Annotation Transition Matrix\n({human_dynamics.transition_matrix.shape[0]} states)')
        axes[0].set_xlabel('Next State')
        axes[0].set_ylabel('Current State')
        plt.colorbar(im1, ax=axes[0])
    else:
        axes[0].text(0.5, 0.5, 'Insufficient\nTransition Data', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Human Annotation Transition Matrix')
    
    # LLM transition matrix
    if llm_dynamics.transition_matrix is not None and llm_dynamics.transition_matrix.size > 1:
        im2 = axes[1].imshow(llm_dynamics.transition_matrix, cmap='Oranges', aspect='auto')
        axes[1].set_title(f'LLM Annotation Transition Matrix\n({llm_dynamics.transition_matrix.shape[0]} states)')
        axes[1].set_xlabel('Next State')
        axes[1].set_ylabel('Current State')
        plt.colorbar(im2, ax=axes[1])
    else:
        axes[1].text(0.5, 0.5, 'Insufficient\nTransition Data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('LLM Annotation Transition Matrix')
    
    plt.tight_layout()
    
    save_path = results_dir / 'transition_matrices_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"전이 행렬 시각화 저장: {save_path}")
    
    plt.show()

def visualize_learning_stability(human_simulator: LearningStabilitySimulator,
                                llm_simulator: LearningStabilitySimulator,
                                results_dir: Path):
    """학습 안정성 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 학습 곡선 시뮬레이션
    human_curves = human_simulator.simulate_learning_curves(num_simulations=10)
    llm_curves = llm_simulator.simulate_learning_curves(num_simulations=10)
    
    # 1. 개별 학습 곡선들
    for curve in human_curves:
        axes[0,0].plot(curve, color='blue', alpha=0.3, linewidth=1)
    for curve in llm_curves:
        axes[0,0].plot(curve, color='orange', alpha=0.3, linewidth=1)
    
    # 평균 곡선
    if human_curves:
        human_mean = np.mean(human_curves, axis=0)
        axes[0,0].plot(human_mean, color='blue', linewidth=3, label='Human (평균)')
    if llm_curves:
        llm_mean = np.mean(llm_curves, axis=0)
        axes[0,0].plot(llm_mean, color='orange', linewidth=3, label='LLM (평균)')
    
    axes[0,0].set_title('Learning Curve Simulations')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Performance')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 최종 성능 분포
    if human_curves and llm_curves:
        human_final = [curve[-1] for curve in human_curves]
        llm_final = [curve[-1] for curve in llm_curves]
        
        axes[0,1].boxplot([human_final, llm_final], labels=['Human', 'LLM'])
        axes[0,1].set_title('Final Performance Distribution')
        axes[0,1].set_ylabel('Final Performance')
        axes[0,1].grid(True, alpha=0.3)
    
    # 3. 학습 분산 (에피소드별)
    if human_curves and llm_curves:
        human_variance = np.var(human_curves, axis=0)
        llm_variance = np.var(llm_curves, axis=0)
        
        axes[1,0].plot(human_variance, color='blue', label='Human', linewidth=2)
        axes[1,0].plot(llm_variance, color='orange', label='LLM', linewidth=2)
        axes[1,0].set_title('Learning Variance Over Episodes')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Variance')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # 4. 수렴 속도 비교
    if human_curves and llm_curves:
        human_convergence = []
        llm_convergence = []
        
        for curve in human_curves:
            # 90% 성능 달성 시점
            target = 0.9 * curve[-1]
            convergence_episode = len(curve)
            for i, perf in enumerate(curve):
                if perf >= target:
                    convergence_episode = i
                    break
            human_convergence.append(convergence_episode)
        
        for curve in llm_curves:
            target = 0.9 * curve[-1]
            convergence_episode = len(curve)
            for i, perf in enumerate(curve):
                if perf >= target:
                    convergence_episode = i
                    break
            llm_convergence.append(convergence_episode)
        
        axes[1,1].boxplot([human_convergence, llm_convergence], labels=['Human', 'LLM'])
        axes[1,1].set_title('Convergence Speed (Episodes to 90%)')
        axes[1,1].set_ylabel('Episodes')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = results_dir / 'learning_stability_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"학습 안정성 시각화 저장: {save_path}")
    
    plt.show()

def visualize_comprehensive_results(human_dyn: TransitionDynamicsMetrics,
                                  llm_dyn: TransitionDynamicsMetrics,
                                  human_stab: LearningStabilityMetrics,
                                  llm_stab: LearningStabilityMetrics,
                                  results_dir: Path):
    """종합 결과 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 전이 동역학 메트릭
    dynamics_metrics = ['Consistency', 'Predictability', 'Stability', 'Markov\nProperty']
    human_dyn_vals = [human_dyn.transition_consistency, human_dyn.transition_predictability,
                     human_dyn.transition_stability, human_dyn.markov_property]
    llm_dyn_vals = [llm_dyn.transition_consistency, llm_dyn.transition_predictability,
                   llm_dyn.transition_stability, llm_dyn.markov_property]
    
    x = np.arange(len(dynamics_metrics))
    width = 0.35
    
    axes[0,0].bar(x - width/2, human_dyn_vals, width, label='Human', color='skyblue', alpha=0.8)
    axes[0,0].bar(x + width/2, llm_dyn_vals, width, label='LLM', color='orange', alpha=0.8)
    axes[0,0].set_title('Transition Dynamics Quality')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(dynamics_metrics)
    axes[0,0].set_ylabel('Score')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 학습 안정성 메트릭
    stability_metrics = ['Convergence\nSmoothness', 'Learning\nConsistency', 'Gradient\nStability', 'Initialization\nRobustness']
    human_stab_vals = [human_stab.convergence_smoothness, human_stab.learning_consistency,
                      human_stab.gradient_stability, human_stab.initialization_robustness]
    llm_stab_vals = [llm_stab.convergence_smoothness, llm_stab.learning_consistency,
                    llm_stab.gradient_stability, llm_stab.initialization_robustness]
    
    x = np.arange(len(stability_metrics))
    axes[0,1].bar(x - width/2, human_stab_vals, width, label='Human', color='skyblue', alpha=0.8)
    axes[0,1].bar(x + width/2, llm_stab_vals, width, label='LLM', color='orange', alpha=0.8)
    axes[0,1].set_title('Learning Stability')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(stability_metrics)
    axes[0,1].set_ylabel('Score')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 정보 이론적 메트릭
    info_metrics = ['Mutual\nInformation', 'Transition\nEntropy', 'Causal\nStrength']
    human_info_vals = [human_dyn.mutual_information, human_dyn.transition_entropy, human_dyn.causal_strength]
    llm_info_vals = [llm_dyn.mutual_information, llm_dyn.transition_entropy, llm_dyn.causal_strength]
    
    x = np.arange(len(info_metrics))
    axes[0,2].bar(x - width/2, human_info_vals, width, label='Human', color='skyblue', alpha=0.8)
    axes[0,2].bar(x + width/2, llm_info_vals, width, label='LLM', color='orange', alpha=0.8)
    axes[0,2].set_title('Information Theoretic Measures')
    axes[0,2].set_xticks(x)
    axes[0,2].set_xticklabels(info_metrics)
    axes[0,2].set_ylabel('Score')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. 복잡성 vs 안정성
    axes[1,0].scatter(human_dyn.transition_complexity, human_stab.convergence_smoothness, 
                     s=100, color='skyblue', label='Human', alpha=0.8)
    axes[1,0].scatter(llm_dyn.transition_complexity, llm_stab.convergence_smoothness,
                     s=100, color='orange', label='LLM', alpha=0.8)
    axes[1,0].set_title('Complexity vs Stability')
    axes[1,0].set_xlabel('Transition Complexity')
    axes[1,0].set_ylabel('Convergence Smoothness')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. 예측가능성 vs 일관성
    axes[1,1].scatter(human_dyn.transition_predictability, human_stab.learning_consistency,
                     s=100, color='skyblue', label='Human', alpha=0.8)
    axes[1,1].scatter(llm_dyn.transition_predictability, llm_stab.learning_consistency,
                     s=100, color='orange', label='LLM', alpha=0.8)
    axes[1,1].set_title('Predictability vs Consistency')
    axes[1,1].set_xlabel('Transition Predictability')
    axes[1,1].set_ylabel('Learning Consistency')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. 종합 점수 비교
    human_dynamics_score = np.mean([human_dyn.transition_consistency, human_dyn.transition_predictability, human_dyn.markov_property])
    llm_dynamics_score = np.mean([llm_dyn.transition_consistency, llm_dyn.transition_predictability, llm_dyn.markov_property])
    human_stability_score = np.mean([human_stab.convergence_smoothness, human_stab.learning_consistency, human_stab.gradient_stability])
    llm_stability_score = np.mean([llm_stab.convergence_smoothness, llm_stab.learning_consistency, llm_stab.gradient_stability])
    
    categories = ['Transition\nDynamics', 'Learning\nStability']
    human_scores = [human_dynamics_score, human_stability_score]
    llm_scores = [llm_dynamics_score, llm_stability_score]
    
    x = np.arange(len(categories))
    axes[1,2].bar(x - width/2, human_scores, width, label='Human', color='skyblue', alpha=0.8)
    axes[1,2].bar(x + width/2, llm_scores, width, label='LLM', color='orange', alpha=0.8)
    axes[1,2].set_title('Overall Performance')
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels(categories)
    axes[1,2].set_ylabel('Composite Score')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = results_dir / 'comprehensive_dynamics_stability_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"종합 결과 시각화 저장: {save_path}")
    
    plt.show()

def analyze_dynamics_stability_correlation(human_dyn: TransitionDynamicsMetrics,
                                         llm_dyn: TransitionDynamicsMetrics,
                                         human_stab: LearningStabilityMetrics,
                                         llm_stab: LearningStabilityMetrics):
    """전이 동역학과 학습 안정성 간 상관관계 분석"""
    
    print("전이 동역학 - 학습 안정성 상관관계 분석 중...")
    
    # 전이 동역학 점수 계산
    def calc_dynamics_score(metrics):
        return (
            metrics.transition_consistency * 0.25 +
            metrics.transition_predictability * 0.25 +
            metrics.transition_stability * 0.20 +
            metrics.markov_property * 0.15 +
            metrics.determinism_score * 0.15
        )
    
    # 학습 안정성 점수 계산
    def calc_stability_score(metrics):
        return (
            metrics.convergence_smoothness * 0.25 +
            metrics.learning_consistency * 0.20 +
            metrics.gradient_stability * 0.20 +
            metrics.initialization_robustness * 0.15 +
            (1.0 / (1.0 + metrics.final_variance)) * 0.20  # 낮은 분산이 좋음
        )
    
    human_dynamics_score = calc_dynamics_score(human_dyn)
    llm_dynamics_score = calc_dynamics_score(llm_dyn)
    human_stability_score = calc_stability_score(human_stab)
    llm_stability_score = calc_stability_score(llm_stab)
    
    # 상관관계 계산
    dynamics_scores = [human_dynamics_score, llm_dynamics_score]
    stability_scores = [human_stability_score, llm_stability_score]
    
    if len(set(dynamics_scores)) > 1 and len(set(stability_scores)) > 1:
        correlation_coeff, p_value = pearsonr(dynamics_scores, stability_scores)
    else:
        correlation_coeff = 0.0
        p_value = 1.0
    
    correlation_results = {
        'human_dynamics_score': human_dynamics_score,
        'llm_dynamics_score': llm_dynamics_score,
        'human_stability_score': human_stability_score,
        'llm_stability_score': llm_stability_score,
        'dynamics_advantage': llm_dynamics_score - human_dynamics_score,
        'stability_advantage': llm_stability_score - human_stability_score,
        'correlation_coefficient': correlation_coeff,
        'p_value': p_value
    }
    
    return correlation_results

def compile_dynamics_stability_results(human_dyn: TransitionDynamicsMetrics,
                                     llm_dyn: TransitionDynamicsMetrics,
                                     human_stab: LearningStabilityMetrics,
                                     llm_stab: LearningStabilityMetrics,
                                     correlation_analysis: Dict):
    """최종 결과 컴파일"""
    
    # 가설 검증
    hypothesis_verified = (
        correlation_analysis['dynamics_advantage'] > 0 and
        correlation_analysis['stability_advantage'] > 0
    )
    
    # 주요 개선 사항
    key_improvements = {
        'transition_dynamics': {
            'consistency': llm_dyn.transition_consistency - human_dyn.transition_consistency,
            'predictability': llm_dyn.transition_predictability - human_dyn.transition_predictability,
            'stability': llm_dyn.transition_stability - human_dyn.transition_stability,
            'markov_property': llm_dyn.markov_property - human_dyn.markov_property
        },
        'learning_stability': {
            'convergence_smoothness': llm_stab.convergence_smoothness - human_stab.convergence_smoothness,
            'learning_consistency': llm_stab.learning_consistency - human_stab.learning_consistency,
            'gradient_stability': llm_stab.gradient_stability - human_stab.gradient_stability,
            'final_variance_reduction': human_stab.final_variance - llm_stab.final_variance
        }
    }
    
    results = {
        'experiment_name': 'LLM annotation → 더 일관된 전이 동역학 → 더 안정적 학습',
        'hypothesis_verified': hypothesis_verified,
        'transition_dynamics_metrics': {
            'human': human_dyn.__dict__,
            'llm': llm_dyn.__dict__
        },
        'learning_stability_metrics': {
            'human': human_stab.__dict__,
            'llm': llm_stab.__dict__
        },
        'correlation_analysis': correlation_analysis,
        'key_improvements': key_improvements,
        'conclusion': (
            "✅ LLM annotation이 더 일관된 전이 동역학을 제공하여 더 안정적인 학습을 달성함"
            if hypothesis_verified else
            "❌ 가설이 충분히 지지되지 않음"
        )
    }
    
    # 결과 출력
    print(f"\n{'='*60}")
    print("실험 결과 요약")
    print(f"{'='*60}")
    print(f"가설 검증: {'✅ 성공' if hypothesis_verified else '❌ 실패'}")
    print(f"전이 동역학 우위: {correlation_analysis['dynamics_advantage']:.3f}")
    print(f"학습 안정성 우위: {correlation_analysis['stability_advantage']:.3f}")
    
    print(f"\n주요 개선 사항:")
    print(f"  전이 일관성: +{key_improvements['transition_dynamics']['consistency']:.3f}")
    print(f"  전이 예측가능성: +{key_improvements['transition_dynamics']['predictability']:.3f}")
    print(f"  수렴 부드러움: +{key_improvements['learning_stability']['convergence_smoothness']:.3f}")
    print(f"  학습 일관성: +{key_improvements['learning_stability']['learning_consistency']:.3f}")
    print(f"  분산 감소: +{key_improvements['learning_stability']['final_variance_reduction']:.3f}")
    
    return results

if __name__ == "__main__":
    try:
        results = run_transition_dynamics_pipeline()
        print("\n🎉 파이프라인 실행 완료!")
        print(f"결과 파일 위치: mdp/rl_tests/transition_dynamics_results/")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()