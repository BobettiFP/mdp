#!/usr/bin/env python3
"""
공정한 대화 시스템 RL 학습 비교 실험
===================================
보상 함수를 개선하여 Human vs LLM annotation을 공정하게 비교
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import random
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

@dataclass
class DialogueState:
    """대화 상태 표현"""
    utterance: str
    slots: Dict[str, str]
    turn_id: int
    dialogue_id: str
    
    def to_vector(self, slot_vocabulary: Dict[str, int], value_vocabulary: Dict[str, int]) -> np.ndarray:
        """상태를 벡터로 변환"""
        vector = np.zeros(len(slot_vocabulary) + len(value_vocabulary))
        
        for slot, value in self.slots.items():
            if slot in slot_vocabulary:
                vector[slot_vocabulary[slot]] = 1
            
            # 값이 리스트인 경우 첫 번째 요소 사용
            if isinstance(value, list):
                value = value[0] if value else ""
            
            value_str = str(value)
            if value_str in value_vocabulary:
                vector[len(slot_vocabulary) + value_vocabulary[value_str]] = 1
                
        return vector

@dataclass
class TrainingMetrics:
    """학습 과정 메트릭 저장"""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    action_distributions: List[np.ndarray] = field(default_factory=list)
    loss_values: List[float] = field(default_factory=list)
    slot_f1_scores: List[float] = field(default_factory=list)  # 정확도 대신 F1
    slot_precision: List[float] = field(default_factory=list)
    slot_recall: List[float] = field(default_factory=list)
    exploration_rate: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)

class DialogueEnvironment:
    """개선된 대화 환경"""
    
    def __init__(self, annotations: List[dict], annotation_type: str, canonical_map: Dict[str, str] = None):
        self.annotations = annotations
        self.annotation_type = annotation_type
        self.canonical_map = canonical_map or {}
        self.dialogues = self._organize_by_dialogue()
        self.current_dialogue = None
        self.current_turn = 0
        self.episode_stats = defaultdict(int)
        
        # 정규화된 슬롯 vocabulary
        self.slot_vocab = {}
        self.value_vocab = {}
        self._build_vocabulary()
        
        # 통계 정보
        self.slot_statistics = self._calculate_statistics()
        
    def _normalize_slot(self, slot: str) -> str:
        """슬롯 이름 정규화"""
        # canonical_map 사용
        if slot in self.canonical_map:
            return self.canonical_map[slot]
        
        # 기본 정규화
        slot = slot.lower().replace("-", "_")
        for prefix in ["restaurant_", "hotel_", "train_", "taxi_", "attraction_", 
                      "bus_", "hospital_", "police_"]:
            if slot.startswith(prefix):
                slot = slot[len(prefix):]
        return slot
    
    def _organize_by_dialogue(self) -> Dict[str, List[dict]]:
        """대화별로 주석 정리"""
        dialogues = defaultdict(list)
        for ann in self.annotations:
            if ann['dialogue_id']:
                dialogues[ann['dialogue_id']].append(ann)
        
        for did in dialogues:
            dialogues[did].sort(key=lambda x: int(x.get('turn_id') or 0) if x.get('turn_id') is not None else 0)
            
        return dict(dialogues)
    
    def _build_vocabulary(self):
        """정규화된 슬롯으로 vocabulary 구축"""
        all_slots = set()
        all_values = set()
        
        for ann in self.annotations:
            for slot, value in ann.get('slots', {}).items():
                normalized_slot = self._normalize_slot(slot)
                all_slots.add(normalized_slot)
                
                if isinstance(value, list):
                    value = value[0] if value else ""
                all_values.add(str(value))
        
        self.slot_vocab = {slot: i for i, slot in enumerate(sorted(all_slots))}
        self.value_vocab = {value: i for i, value in enumerate(sorted(all_values))}
    
    def _calculate_statistics(self) -> Dict:
        """데이터셋 통계 계산"""
        slot_counts = defaultdict(int)
        slot_values_per_turn = []
        total_turns = 0
        
        for ann in self.annotations:
            normalized_slots = {}
            for slot, value in ann.get('slots', {}).items():
                normalized_slot = self._normalize_slot(slot)
                normalized_slots[normalized_slot] = value
                slot_counts[normalized_slot] += 1
            
            slot_values_per_turn.append(len(normalized_slots))
            total_turns += 1
        
        return {
            'slot_counts': dict(slot_counts),
            'total_turns': total_turns,
            'avg_slots_per_turn': np.mean(slot_values_per_turn),
            'std_slots_per_turn': np.std(slot_values_per_turn),
            'slot_distribution': {s: c/total_turns for s, c in slot_counts.items()}
        }
    
    def reset(self) -> DialogueState:
        """새 대화 시작"""
        self.current_dialogue = random.choice(list(self.dialogues.keys()))
        self.current_turn = 0
        self.episode_stats['total_episodes'] += 1
        
        if self.dialogues[self.current_dialogue]:
            ann = self.dialogues[self.current_dialogue][0]
            # 정규화된 슬롯 사용
            normalized_slots = {}
            for slot, value in ann['slots'].items():
                normalized_slots[self._normalize_slot(slot)] = value
                
            return DialogueState(
                utterance=ann['utterance'],
                slots=normalized_slots,
                turn_id=int(ann.get('turn_id') or 0) if ann.get('turn_id') is not None else 0,
                dialogue_id=self.current_dialogue
            )
        else:
            return self.reset()
    
    def step(self, predicted_slots: Set[str]) -> Tuple[DialogueState, float, bool, Dict]:
        """개선된 step - 슬롯 집합을 예측"""
        current_ann = self.dialogues[self.current_dialogue][self.current_turn]
        
        # 실제 슬롯 (정규화)
        actual_slots = set()
        for slot in current_ann.get('slots', {}):
            actual_slots.add(self._normalize_slot(slot))
        
        # F1 기반 보상 계산
        reward, info = self._calculate_f1_reward(predicted_slots, actual_slots)
        
        self.current_turn += 1
        done = self.current_turn >= len(self.dialogues[self.current_dialogue])
        
        if not done:
            ann = self.dialogues[self.current_dialogue][self.current_turn]
            # 정규화된 슬롯 사용
            normalized_slots = {}
            for slot, value in ann['slots'].items():
                normalized_slots[self._normalize_slot(slot)] = value
                
            next_state = DialogueState(
                utterance=ann['utterance'],
                slots=normalized_slots,
                turn_id=int(ann.get('turn_id') or self.current_turn) if ann.get('turn_id') is not None else self.current_turn,
                dialogue_id=self.current_dialogue
            )
        else:
            next_state = None
            
        return next_state, reward, done, info
    
    def _calculate_f1_reward(self, predicted: Set[str], actual: Set[str]) -> Tuple[float, Dict]:
        """F1 스코어 기반 보상"""
        if not predicted and not actual:
            precision = recall = f1 = 1.0
        elif not predicted or not actual:
            precision = recall = f1 = 0.0
        else:
            tp = len(predicted & actual)
            fp = len(predicted - actual)
            fn = len(actual - predicted)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 보상은 F1 스코어 자체 (0~1 범위로 정규화됨)
        reward = f1
        
        info = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predicted_slots': predicted,
            'actual_slots': actual,
            'tp': len(predicted & actual) if predicted and actual else 0,
            'fp': len(predicted - actual) if predicted else 0,
            'fn': len(actual - predicted) if actual else 0
        }
        
        return reward, info

class SlotPredictionNetwork(nn.Module):
    """슬롯 예측 네트워크 (멀티 레이블 분류)"""
    
    def __init__(self, input_size: int, hidden_size: int, num_slots: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_slots)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return torch.sigmoid(x)  # 멀티 레이블을 위한 sigmoid

class FairRLAgent:
    """공정한 비교를 위한 RL 에이전트"""
    
    def __init__(self, env: DialogueEnvironment, learning_rate: float = 0.001, threshold: float = 0.5):
        self.env = env
        self.threshold = threshold
        input_size = len(env.value_vocab)  # 값만 사용 (슬롯은 예측 대상)
        num_slots = len(env.slot_vocab)
        
        self.policy = SlotPredictionNetwork(input_size, 256, num_slots)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
        self.metrics = TrainingMetrics()
        self.episode_count = 0
        
        # 슬롯 인덱스 역매핑
        self.idx_to_slot = {v: k for k, v in env.slot_vocab.items()}
    
    def predict_slots(self, state: DialogueState, training: bool = True) -> Tuple[Set[str], torch.Tensor]:
        """슬롯 집합 예측"""
        # 값만으로 입력 벡터 생성
        value_vector = np.zeros(len(self.env.value_vocab))
        for value in state.slots.values():
            if isinstance(value, list):
                value = value[0] if value else ""
            value_str = str(value)
            if value_str in self.env.value_vocab:
                value_vector[self.env.value_vocab[value_str]] = 1
        
        x = torch.FloatTensor(value_vector).unsqueeze(0)
        
        # 슬롯 예측
        slot_probs = self.policy(x)
        
        # 임계값 기반 슬롯 선택
        if training:
            # 학습 중에는 확률적 선택
            predicted_indices = (torch.rand_like(slot_probs) < slot_probs).squeeze().nonzero().flatten()
        else:
            # 평가 시에는 임계값 사용
            predicted_indices = (slot_probs > self.threshold).squeeze().nonzero().flatten()
        
        predicted_slots = {self.idx_to_slot[idx.item()] for idx in predicted_indices}
        
        return predicted_slots, slot_probs
    
    def train_episode(self) -> Tuple[float, Dict]:
        """한 에피소드 학습"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        predictions = []
        targets = []
        infos = []
        
        while state is not None:
            # 슬롯 예측
            predicted_slots, slot_probs = self.predict_slots(state, training=True)
            
            # 환경에서 보상 받기
            next_state, reward, done, info = self.env.step(predicted_slots)
            
            # 타겟 생성 (실제 슬롯)
            target = torch.zeros(len(self.env.slot_vocab))
            for slot in info['actual_slots']:
                if slot in self.env.slot_vocab:
                    target[self.env.slot_vocab[slot]] = 1
            
            predictions.append(slot_probs)
            targets.append(target.unsqueeze(0))
            infos.append(info)
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
            state = next_state
        
        # 배치 학습
        if predictions:
            pred_batch = torch.cat(predictions, dim=0)
            target_batch = torch.cat(targets, dim=0)
            
            loss = self.criterion(pred_batch, target_batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient norm
            total_norm = 0
            for p in self.policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 메트릭 기록
            self.metrics.episode_rewards.append(episode_reward)
            self.metrics.episode_lengths.append(episode_length)
            self.metrics.loss_values.append(loss.item())
            self.metrics.gradient_norms.append(total_norm)
            
            # F1 스코어 평균
            avg_f1 = np.mean([info['f1'] for info in infos])
            avg_precision = np.mean([info['precision'] for info in infos])
            avg_recall = np.mean([info['recall'] for info in infos])
            
            self.metrics.slot_f1_scores.append(avg_f1)
            self.metrics.slot_precision.append(avg_precision)
            self.metrics.slot_recall.append(avg_recall)
            
            # 엔트로피 (탐험 정도)
            entropy = -torch.mean(torch.sum(pred_batch * torch.log(pred_batch + 1e-8) + 
                                           (1 - pred_batch) * torch.log(1 - pred_batch + 1e-8), dim=1))
            self.metrics.exploration_rate.append(entropy.item())
        
        self.episode_count += 1
        
        return episode_reward, {
            'length': episode_length,
            'f1': avg_f1 if predictions else 0,
            'precision': avg_precision if predictions else 0,
            'recall': avg_recall if predictions else 0,
            'loss': loss.item() if predictions else 0
        }

def run_fair_comparison(processed_annotations_path: Path, 
                       num_episodes: int = 2000,
                       eval_interval: int = 100):
    """공정한 비교 실험"""
    
    # 데이터 로드
    with open(processed_annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    canonical_map = data.get('canonical_map', {})
    
    # 환경 및 에이전트 생성
    human_env = DialogueEnvironment(data['human_annotations'], 'human', canonical_map)
    llm_env = DialogueEnvironment(data['llm_annotations'], 'llm', canonical_map)
    
    human_agent = FairRLAgent(human_env)
    llm_agent = FairRLAgent(llm_env)
    
    # 데이터셋 통계 출력
    print("="*80)
    print("DATASET STATISTICS (After Normalization)")
    print("="*80)
    print(f"\nHuman Dataset:")
    print(f"  Unique slots: {len(human_env.slot_vocab)}")
    print(f"  Unique values: {len(human_env.value_vocab)}")
    print(f"  Avg slots/turn: {human_env.slot_statistics['avg_slots_per_turn']:.2f}")
    print(f"  Total turns: {human_env.slot_statistics['total_turns']}")
    
    print(f"\nLLM Dataset:")
    print(f"  Unique slots: {len(llm_env.slot_vocab)}")
    print(f"  Unique values: {len(llm_env.value_vocab)}")
    print(f"  Avg slots/turn: {llm_env.slot_statistics['avg_slots_per_turn']:.2f}")
    print(f"  Total turns: {llm_env.slot_statistics['total_turns']}")
    print("-" * 80)
    
    # 학습 진행
    for episode in range(num_episodes):
        # 학습
        human_reward, human_info = human_agent.train_episode()
        llm_reward, llm_info = llm_agent.train_episode()
        
        # 진행상황 출력
        if (episode + 1) % eval_interval == 0:
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"Human - Reward: {human_reward:.3f}, F1: {human_info['f1']:.3f}, "
                  f"P/R: {human_info['precision']:.3f}/{human_info['recall']:.3f}")
            print(f"LLM   - Reward: {llm_reward:.3f}, F1: {llm_info['f1']:.3f}, "
                  f"P/R: {llm_info['precision']:.3f}/{llm_info['recall']:.3f}")
    
    # 분석 및 시각화
    analyze_fair_results(human_agent.metrics, llm_agent.metrics, human_env, llm_env)
    
    return evaluate_fair_performance(human_agent, llm_agent, human_env, llm_env)

def analyze_fair_results(human_metrics: TrainingMetrics, llm_metrics: TrainingMetrics,
                        human_env: DialogueEnvironment, llm_env: DialogueEnvironment):
    """공정한 결과 분석 및 시각화"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. F1 스코어 (주요 지표)
    ax1 = plt.subplot(2, 3, 1)
    window = 50
    human_f1_ma = np.convolve(human_metrics.slot_f1_scores, np.ones(window)/window, mode='valid')
    llm_f1_ma = np.convolve(llm_metrics.slot_f1_scores, np.ones(window)/window, mode='valid')
    ax1.plot(human_f1_ma, label='Human', alpha=0.8, color='blue')
    ax1.plot(llm_f1_ma, label='LLM', alpha=0.8, color='orange')
    ax1.set_title('Slot Prediction F1 Score')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('F1 Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision vs Recall
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(human_metrics.slot_precision[-100:], human_metrics.slot_recall[-100:], 
                alpha=0.5, label='Human', color='blue')
    ax2.scatter(llm_metrics.slot_precision[-100:], llm_metrics.slot_recall[-100:], 
                alpha=0.5, label='LLM', color='orange')
    ax2.set_title('Precision vs Recall (Last 100 episodes)')
    ax2.set_xlabel('Precision')
    ax2.set_ylabel('Recall')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 보상 곡선 (정규화됨)
    ax3 = plt.subplot(2, 3, 3)
    human_reward_ma = np.convolve(human_metrics.episode_rewards, np.ones(window)/window, mode='valid')
    llm_reward_ma = np.convolve(llm_metrics.episode_rewards, np.ones(window)/window, mode='valid')
    ax3.plot(human_reward_ma, label='Human', alpha=0.8, color='blue')
    ax3.plot(llm_reward_ma, label='LLM', alpha=0.8, color='orange')
    ax3.set_title('Episode Rewards (F1-based)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 학습 손실
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(human_metrics.loss_values, label='Human', alpha=0.6, color='blue')
    ax4.plot(llm_metrics.loss_values, label='LLM', alpha=0.6, color='orange')
    ax4.set_title('Training Loss')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('BCE Loss')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 최종 F1 분포
    ax5 = plt.subplot(2, 3, 5)
    final_f1_human = human_metrics.slot_f1_scores[-200:]
    final_f1_llm = llm_metrics.slot_f1_scores[-200:]
    ax5.boxplot([final_f1_human, final_f1_llm], labels=['Human', 'LLM'])
    ax5.set_title('Final F1 Score Distribution')
    ax5.set_ylabel('F1 Score')
    ax5.grid(True, alpha=0.3)
    
    # 6. 정규화된 vocabulary 크기
    ax6 = plt.subplot(2, 3, 6)
    categories = ['Original\nSlots', 'Normalized\nSlots', 'Values']
    human_counts = [
        32,  # 원래 Human 슬롯 수
        len(human_env.slot_vocab),
        len(human_env.value_vocab)
    ]
    llm_counts = [
        445,  # 원래 LLM 슬롯 수
        len(llm_env.slot_vocab),
        len(llm_env.value_vocab)
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    ax6.bar(x - width/2, human_counts, width, label='Human', color='blue')
    ax6.bar(x + width/2, llm_counts, width, label='LLM', color='orange')
    ax6.set_xticks(x)
    ax6.set_xticklabels(categories)
    ax6.set_title('Vocabulary Comparison')
    ax6.set_ylabel('Count')
    ax6.legend()
    ax6.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('fair_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_fair_performance(human_agent: FairRLAgent, llm_agent: FairRLAgent,
                             human_env: DialogueEnvironment, llm_env: DialogueEnvironment) -> Dict:
    """공정한 최종 성능 평가"""
    
    # 최종 200 에피소드 성능
    human_final_f1 = human_agent.metrics.slot_f1_scores[-200:]
    llm_final_f1 = llm_agent.metrics.slot_f1_scores[-200:]
    
    # 통계적 검증
    t_stat, p_value = stats.ttest_ind(human_final_f1, llm_final_f1)
    
    results = {
        'human_final_f1_mean': np.mean(human_final_f1),
        'human_final_f1_std': np.std(human_final_f1),
        'llm_final_f1_mean': np.mean(llm_final_f1),
        'llm_final_f1_std': np.std(llm_final_f1),
        'difference': np.mean(llm_final_f1) - np.mean(human_final_f1),
        't_statistic': t_stat,
        'p_value': p_value,
        'human_final_precision': np.mean(human_agent.metrics.slot_precision[-200:]),
        'human_final_recall': np.mean(human_agent.metrics.slot_recall[-200:]),
        'llm_final_precision': np.mean(llm_agent.metrics.slot_precision[-200:]),
        'llm_final_recall': np.mean(llm_agent.metrics.slot_recall[-200:])
    }
    
    # 결과 출력
    print("\n" + "="*80)
    print("FAIR COMPARISON RESULTS")
    print("="*80)
    print(f"\nF1 Score Performance:")
    print(f"  Human: {results['human_final_f1_mean']:.3f} ± {results['human_final_f1_std']:.3f}")
    print(f"  LLM:   {results['llm_final_f1_mean']:.3f} ± {results['llm_final_f1_std']:.3f}")
    print(f"  Difference: {results['difference']:.3f} (p={results['p_value']:.4f})")
    
    print(f"\nPrecision/Recall:")
    print(f"  Human: P={results['human_final_precision']:.3f}, R={results['human_final_recall']:.3f}")
    print(f"  LLM:   P={results['llm_final_precision']:.3f}, R={results['llm_final_recall']:.3f}")
    
    # 통계적 유의성 판단
    if results['p_value'] < 0.05:
        if results['difference'] > 0:
            print(f"\n✅ LLM annotations show significantly better performance")
        else:
            print(f"\n✅ Human annotations show significantly better performance")
    else:
        print(f"\n⚖️  No significant difference in performance (p={results['p_value']:.3f})")
    
    # 결과 저장
    with open('fair_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # 공정한 비교 실험 실행
    results = run_fair_comparison(
        Path("processed_annotations.json"),
        num_episodes=2000,
        eval_interval=100
    )