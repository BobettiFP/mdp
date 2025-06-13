#!/usr/bin/env python3
"""
진짜 대화 시스템 강화학습
======================
실제 대화 태스크를 위한 RL 구현
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import random
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

@dataclass
class DialogueState:
    """대화 상태"""
    dialogue_history: List[Dict]  # 지금까지의 대화 내역
    current_utterance: str        # 현재 사용자 발화
    belief_state: Dict[str, str]  # 현재까지 추적한 슬롯-값
    turn_number: int
    dialogue_id: str

@dataclass 
class DialogueAction:
    """시스템 액션"""
    action_type: str  # 'request', 'inform', 'confirm', 'bye'
    slot: Optional[str] = None
    value: Optional[str] = None

class DialogueSimulator:
    """대화 시뮬레이터 - 사용자 역할"""
    
    def __init__(self, dialogues: List[Dict], annotation_type: str):
        self.dialogues = self._organize_dialogues(dialogues)
        self.annotation_type = annotation_type
        self.current_dialogue = None
        self.goal_slots = {}
        self.satisfied_slots = set()
        
    def _organize_dialogues(self, annotations: List[Dict]) -> Dict:
        """대화별로 정리"""
        dialogues = defaultdict(list)
        for ann in annotations:
            if ann['dialogue_id']:
                dialogues[ann['dialogue_id']].append(ann)
        
        # 각 대화의 모든 슬롯을 수집 (목표로 사용)
        dialogue_goals = {}
        for did, turns in dialogues.items():
            all_slots = {}
            for turn in turns:
                for slot, value in turn.get('slots', {}).items():
                    if isinstance(value, list):
                        value = value[0] if value else ""
                    all_slots[slot] = str(value)
            dialogue_goals[did] = all_slots
            
        self.dialogue_goals = dialogue_goals
        return dict(dialogues)
    
    def start_dialogue(self) -> Tuple[str, Dict[str, str]]:
        """새 대화 시작"""
        self.current_dialogue = random.choice(list(self.dialogues.keys()))
        self.goal_slots = self.dialogue_goals[self.current_dialogue].copy()
        self.satisfied_slots = set()
        
        # 첫 발화 생성
        first_utterance = self._generate_user_utterance()
        return first_utterance, self.goal_slots
    
    def _generate_user_utterance(self) -> str:
        """사용자 발화 생성 (간단한 템플릿 기반)"""
        unsatisfied = set(self.goal_slots.keys()) - self.satisfied_slots
        
        if not unsatisfied:
            return "Thank you, that's all I need."
        
        # 1-3개의 슬롯 정보를 포함한 발화 생성
        slots_to_mention = random.sample(list(unsatisfied), 
                                       min(random.randint(1, 3), len(unsatisfied)))
        
        utterance_parts = []
        for slot in slots_to_mention:
            value = self.goal_slots[slot]
            # 간단한 템플릿
            if 'area' in slot:
                utterance_parts.append(f"in the {value} area")
            elif 'price' in slot:
                utterance_parts.append(f"with {value} price")
            elif 'name' in slot:
                utterance_parts.append(f"named {value}")
            else:
                utterance_parts.append(f"{slot} is {value}")
        
        return f"I'm looking for something {' and '.join(utterance_parts)}."
    
    def respond_to_action(self, action: DialogueAction) -> Tuple[str, float, bool]:
        """시스템 액션에 대한 응답"""
        reward = 0
        done = False
        
        if action.action_type == 'request':
            # 정보 요청에 응답
            if action.slot in self.goal_slots:
                value = self.goal_slots[action.slot]
                response = f"The {action.slot} should be {value}."
                reward = 0.1  # 적절한 질문
            else:
                response = "I don't have a preference for that."
                reward = -0.1  # 불필요한 질문
                
        elif action.action_type == 'confirm':
            # 확인 요청에 응답
            if action.slot in self.goal_slots and action.value == self.goal_slots[action.slot]:
                response = "Yes, that's correct."
                self.satisfied_slots.add(action.slot)
                reward = 0.5  # 올바른 확인
            else:
                response = "No, that's not right."
                reward = -0.3  # 잘못된 확인
                
        elif action.action_type == 'inform':
            # 정보 제공에 응답
            if self._check_constraints_satisfied(action):
                response = "That sounds good!"
                reward = 0.3
            else:
                response = "That doesn't match what I'm looking for."
                reward = -0.2
                
        elif action.action_type == 'bye':
            # 대화 종료
            if len(self.satisfied_slots) >= len(self.goal_slots) * 0.8:
                response = "Great, thank you!"
                reward = 1.0  # 성공적 종료
                done = True
            else:
                response = "Wait, I still need help with some things."
                reward = -0.5  # 너무 이른 종료
        
        else:
            response = self._generate_user_utterance()
            
        return response, reward, done
    
    def _check_constraints_satisfied(self, action: DialogueAction) -> bool:
        """제약조건 만족 여부 확인"""
        # 간단한 체크 - 실제로는 더 복잡한 로직 필요
        return random.random() > 0.3

class DialogueEnvironment:
    """강화학습을 위한 대화 환경"""
    
    def __init__(self, annotations: List[Dict], annotation_type: str, max_turns: int = 20):
        self.simulator = DialogueSimulator(annotations, annotation_type)
        self.max_turns = max_turns
        self.current_state = None
        self.dialogue_history = []
        self.belief_state = {}
        self.turn_count = 0
        
        # 슬롯과 액션 vocabulary
        self.slots = self._extract_slots(annotations)
        self.action_types = ['request', 'inform', 'confirm', 'bye']
        self.action_space_size = len(self.action_types) * (len(self.slots) + 1)
        
    def _extract_slots(self, annotations: List[Dict]) -> List[str]:
        """모든 슬롯 추출"""
        slots = set()
        for ann in annotations:
            slots.update(ann.get('slots', {}).keys())
        return sorted(list(slots))
    
    def reset(self) -> DialogueState:
        """환경 초기화"""
        first_utterance, goal_slots = self.simulator.start_dialogue()
        self.dialogue_history = [{'speaker': 'user', 'utterance': first_utterance}]
        self.belief_state = {}
        self.turn_count = 0
        
        self.current_state = DialogueState(
            dialogue_history=self.dialogue_history.copy(),
            current_utterance=first_utterance,
            belief_state=self.belief_state.copy(),
            turn_number=self.turn_count,
            dialogue_id=self.simulator.current_dialogue
        )
        
        return self.current_state
    
    def step(self, action_idx: int) -> Tuple[DialogueState, float, bool, Dict]:
        """액션 실행"""
        # 액션 인덱스를 DialogueAction으로 변환
        action = self._idx_to_action(action_idx)
        
        # 시뮬레이터로부터 응답 받기
        user_response, reward, done = self.simulator.respond_to_action(action)
        
        # 대화 기록 업데이트
        self.dialogue_history.append({
            'speaker': 'system',
            'action': action,
            'utterance': f"System: {action.action_type} {action.slot or ''} {action.value or ''}"
        })
        self.dialogue_history.append({
            'speaker': 'user',
            'utterance': user_response
        })
        
        # Belief state 업데이트
        if action.action_type == 'confirm' and 'correct' in user_response:
            self.belief_state[action.slot] = action.value
        
        self.turn_count += 1
        
        # 최대 턴 수 체크
        if self.turn_count >= self.max_turns:
            done = True
            reward -= 0.5  # 너무 긴 대화 페널티
        
        # 다음 상태
        self.current_state = DialogueState(
            dialogue_history=self.dialogue_history.copy(),
            current_utterance=user_response,
            belief_state=self.belief_state.copy(),
            turn_number=self.turn_count,
            dialogue_id=self.simulator.current_dialogue
        )
        
        info = {
            'action': action,
            'goal_slots': self.simulator.goal_slots,
            'satisfied_slots': self.simulator.satisfied_slots,
            'success': done and reward > 0
        }
        
        return self.current_state, reward, done, info
    
    def _idx_to_action(self, idx: int) -> DialogueAction:
        """인덱스를 액션으로 변환"""
        num_slots = len(self.slots)
        action_type_idx = idx // (num_slots + 1)
        slot_idx = idx % (num_slots + 1)
        
        action_type = self.action_types[action_type_idx]
        slot = self.slots[slot_idx] if slot_idx < num_slots else None
        
        # 간단한 값 생성 (실제로는 더 복잡한 로직 필요)
        value = None
        if action_type == 'confirm' and slot:
            # Belief state나 히스토리에서 값 추출
            value = self.belief_state.get(slot, "some_value")
        
        return DialogueAction(action_type, slot, value)
    
    def state_to_vector(self, state: DialogueState) -> np.ndarray:
        """상태를 벡터로 변환"""
        # 간단한 구현: 현재 발화의 단어 존재 여부 + belief state
        vector_size = 100 + len(self.slots) * 2  # 단어 특징 + 슬롯 특징
        vector = np.zeros(vector_size)
        
        # 발화에서 단어 특징 추출 (간단히 구현)
        words = state.current_utterance.lower().split()
        for i, word in enumerate(words[:100]):
            vector[i] = 1
        
        # Belief state 인코딩
        for i, slot in enumerate(self.slots):
            if slot in state.belief_state:
                vector[100 + i*2] = 1  # 슬롯 존재
                vector[100 + i*2 + 1] = 0.5  # 값 존재 (간단히)
        
        return vector

class DialoguePolicyNetwork(nn.Module):
    """대화 정책 네트워크"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

class DialogueRLAgent:
    """대화 강화학습 에이전트"""
    
    def __init__(self, env: DialogueEnvironment, learning_rate: float = 0.001):
        self.env = env
        state_size = 100 + len(env.slots) * 2
        
        self.policy = DialoguePolicyNetwork(
            state_size, 
            256, 
            env.action_space_size
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # 메트릭
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = deque(maxlen=100)
        self.action_counts = defaultdict(int)
        
    def select_action(self, state: DialogueState, epsilon: float = 0.1) -> Tuple[int, torch.Tensor]:
        """액션 선택 (ε-greedy)"""
        state_vector = torch.FloatTensor(self.env.state_to_vector(state)).unsqueeze(0)
        
        if random.random() < epsilon:
            # 탐험
            action = random.randint(0, self.env.action_space_size - 1)
            probs = self.policy(state_vector)
            log_prob = torch.log(probs[0, action])
        else:
            # 활용
            probs = self.policy(state_vector)
            dist = Categorical(probs)
            action_tensor = dist.sample()
            action = action_tensor.item()
            log_prob = dist.log_prob(action_tensor)
        
        return action, log_prob
    
    def train_episode(self, epsilon: float = 0.1) -> Dict:
        """REINFORCE로 한 에피소드 학습"""
        state = self.env.reset()
        log_probs = []
        rewards = []
        
        episode_reward = 0
        
        while True:
            # 액션 선택
            action, log_prob = self.select_action(state, epsilon)
            self.action_counts[action] += 1
            
            # 환경 스텝
            next_state, reward, done, info = self.env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            episode_reward += reward
            
            if done:
                self.success_rate.append(1 if info['success'] else 0)
                break
                
            state = next_state
        
        # 정책 업데이트
        self._update_policy(log_probs, rewards)
        
        # 메트릭 기록
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(self.env.turn_count)
        
        return {
            'reward': episode_reward,
            'length': self.env.turn_count,
            'success': info['success'],
            'success_rate': np.mean(self.success_rate) if self.success_rate else 0
        }
    
    def _update_policy(self, log_probs: List[torch.Tensor], rewards: List[float]):
        """REINFORCE 업데이트"""
        if not log_probs:
            return
            
        # 할인된 보상 계산
        discounted_rewards = []
        gamma = 0.99
        R = 0
        
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        
        # 정규화
        discounted_rewards = torch.tensor(discounted_rewards)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # 손실 계산
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        
        loss = torch.cat(policy_loss).sum()
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

def compare_dialogue_rl(processed_annotations_path: Path, num_episodes: int = 1000):
    """Human vs LLM 대화 RL 비교"""
    
    with open(processed_annotations_path, 'r') as f:
        data = json.load(f)
    
    # 환경과 에이전트 생성
    human_env = DialogueEnvironment(data['human_annotations'], 'human')
    llm_env = DialogueEnvironment(data['llm_annotations'], 'llm')
    
    human_agent = DialogueRLAgent(human_env)
    llm_agent = DialogueRLAgent(llm_env)
    
    # 학습
    human_metrics = []
    llm_metrics = []
    
    print("Training Dialogue RL Agents...")
    for episode in range(num_episodes):
        # 탐험률 감소
        epsilon = max(0.01, 0.5 * (0.995 ** episode))
        
        # 학습
        human_result = human_agent.train_episode(epsilon)
        llm_result = llm_agent.train_episode(epsilon)
        
        human_metrics.append(human_result)
        llm_metrics.append(llm_result)
        
        # 진행 상황 출력
        if (episode + 1) % 100 == 0:
            print(f"\nEpisode {episode + 1}")
            print(f"Human - Reward: {human_result['reward']:.2f}, "
                  f"Success Rate: {human_result['success_rate']:.2%}, "
                  f"Turns: {human_result['length']}")
            print(f"LLM   - Reward: {llm_result['reward']:.2f}, "
                  f"Success Rate: {llm_result['success_rate']:.2%}, "
                  f"Turns: {llm_result['length']}")
    
    # 결과 시각화
    visualize_dialogue_results(human_metrics, llm_metrics, human_agent, llm_agent)
    
    return human_metrics, llm_metrics

def visualize_dialogue_results(human_metrics: List[Dict], llm_metrics: List[Dict],
                              human_agent: DialogueRLAgent, llm_agent: DialogueRLAgent):
    """결과 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 에피소드 보상
    ax = axes[0, 0]
    window = 50
    human_rewards = [m['reward'] for m in human_metrics]
    llm_rewards = [m['reward'] for m in llm_metrics]
    
    human_ma = np.convolve(human_rewards, np.ones(window)/window, mode='valid')
    llm_ma = np.convolve(llm_rewards, np.ones(window)/window, mode='valid')
    
    ax.plot(human_ma, label='Human', alpha=0.8)
    ax.plot(llm_ma, label='LLM', alpha=0.8)
    ax.set_title('Episode Rewards (Moving Average)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 성공률
    ax = axes[0, 1]
    human_success = [m['success_rate'] for m in human_metrics]
    llm_success = [m['success_rate'] for m in llm_metrics]
    
    ax.plot(human_success, label='Human', alpha=0.8)
    ax.plot(llm_success, label='LLM', alpha=0.8)
    ax.set_title('Task Success Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 대화 길이
    ax = axes[1, 0]
    human_lengths = [m['length'] for m in human_metrics]
    llm_lengths = [m['length'] for m in llm_metrics]
    
    ax.hist(human_lengths[-200:], bins=20, alpha=0.5, label='Human', density=True)
    ax.hist(llm_lengths[-200:], bins=20, alpha=0.5, label='LLM', density=True)
    ax.set_title('Dialogue Length Distribution (Last 200)')
    ax.set_xlabel('Number of Turns')
    ax.set_ylabel('Density')
    ax.legend()
    
    # 4. 액션 분포
    ax = axes[1, 1]
    action_types = ['request', 'inform', 'confirm', 'bye']
    human_action_dist = defaultdict(int)
    llm_action_dist = defaultdict(int)
    
    # 액션 타입별로 집계
    for action_idx, count in human_agent.action_counts.items():
        action_type_idx = action_idx // (len(human_agent.env.slots) + 1)
        if action_type_idx < len(action_types):
            human_action_dist[action_types[action_type_idx]] += count
    
    for action_idx, count in llm_agent.action_counts.items():
        action_type_idx = action_idx // (len(llm_agent.env.slots) + 1)
        if action_type_idx < len(action_types):
            llm_action_dist[action_types[action_type_idx]] += count
    
    x = np.arange(len(action_types))
    width = 0.35
    
    human_counts = [human_action_dist[a] for a in action_types]
    llm_counts = [llm_action_dist[a] for a in action_types]
    
    # 정규화
    human_counts = np.array(human_counts) / sum(human_counts)
    llm_counts = np.array(llm_counts) / sum(llm_counts)
    
    ax.bar(x - width/2, human_counts, width, label='Human')
    ax.bar(x + width/2, llm_counts, width, label='LLM')
    ax.set_xticks(x)
    ax.set_xticklabels(action_types)
    ax.set_title('Action Type Distribution')
    ax.set_ylabel('Proportion')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('dialogue_rl_results.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # 실행
    human_metrics, llm_metrics = compare_dialogue_rl(
        Path("processed_annotations.json"),
        num_episodes=1000
    )
    
    # 최종 통계
    print("\n" + "="*60)
    print("FINAL STATISTICS (Last 100 episodes)")
    print("="*60)
    
    human_final_rewards = [m['reward'] for m in human_metrics[-100:]]
    llm_final_rewards = [m['reward'] for m in llm_metrics[-100:]]
    human_final_success = [m['success_rate'] for m in human_metrics[-100:]]
    llm_final_success = [m['success_rate'] for m in llm_metrics[-100:]]
    
    print(f"\nAverage Reward:")
    print(f"  Human: {np.mean(human_final_rewards):.3f} ± {np.std(human_final_rewards):.3f}")
    print(f"  LLM:   {np.mean(llm_final_rewards):.3f} ± {np.std(llm_final_rewards):.3f}")
    
    print(f"\nSuccess Rate:")
    print(f"  Human: {np.mean(human_final_success):.2%}")
    print(f"  LLM:   {np.mean(llm_final_success):.2%}")
