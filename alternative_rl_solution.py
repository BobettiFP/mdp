"""
ConvLab 대신 사용할 수 있는 대안적 RL 대화 시스템
최신 라이브러리와 호환되며 더 간단한 구현
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Tuple
import random
from collections import deque, defaultdict
import pickle

class DialogueState:
    """대화 상태 관리"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.state = {
            'hotel': {},
            'restaurant': {},
            'train': {},
            'taxi': {},
            'attraction': {},
            'general': {}
        }
        self.history = []
        self.turn_count = 0
        
    def update(self, domain: str, slot: str, value: str):
        """상태 업데이트"""
        if domain not in self.state:
            self.state[domain] = {}
        self.state[domain][slot] = value
        
    def get_state_vector(self) -> np.ndarray:
        """상태를 벡터로 변환"""
        # 간단한 상태 벡터화 (실제로는 더 정교한 인코딩 필요)
        vector = []
        
        # 각 도메인별로 상태 정보 수집
        for domain in ['hotel', 'restaurant', 'train', 'taxi', 'attraction']:
            domain_state = self.state.get(domain, {})
            
            # 주요 슬롯들의 존재 여부를 이진값으로 표현
            key_slots = {
                'hotel': ['name', 'area', 'price_range', 'stars', 'people', 'nights'],
                'restaurant': ['name', 'area', 'food_type', 'price_range', 'people', 'time'],
                'train': ['departure', 'destination', 'day', 'time', 'people'],
                'taxi': ['departure', 'destination', 'time'],
                'attraction': ['name', 'area', 'type']
            }
            
            for slot in key_slots.get(domain, []):
                vector.append(1.0 if slot in domain_state else 0.0)
                
        # 턴 수 정보 추가 (정규화)
        vector.append(min(self.turn_count / 20.0, 1.0))
        
        return np.array(vector, dtype=np.float32)

class DialogueAction:
    """대화 액션 정의"""
    
    def __init__(self):
        # 가능한 액션 타입들
        self.action_types = [
            'inform', 'request', 'confirm', 'select', 'recommend',
            'book', 'negate', 'affirm', 'thank', 'bye', 'greet'
        ]
        
        # 도메인별 주요 슬롯들
        self.domain_slots = {
            'hotel': ['name', 'area', 'price_range', 'stars', 'internet', 'parking'],
            'restaurant': ['name', 'area', 'food_type', 'price_range'],
            'train': ['departure', 'destination', 'day', 'time'],
            'taxi': ['departure', 'destination', 'time'],
            'attraction': ['name', 'area', 'type'],
            'general': ['none']
        }
        
        # 액션 인덱스 매핑 생성
        self.action_to_idx = {}
        self.idx_to_action = {}
        idx = 0
        
        for domain in self.domain_slots:
            for action_type in self.action_types:
                for slot in self.domain_slots[domain]:
                    action_key = f"{domain}-{action_type}-{slot}"
                    self.action_to_idx[action_key] = idx
                    self.idx_to_action[idx] = action_key
                    idx += 1
                    
        self.action_dim = len(self.action_to_idx)
    
    def action_to_vector(self, action_dict: Dict) -> np.ndarray:
        """액션을 벡터로 변환"""
        vector = np.zeros(self.action_dim)
        
        for act_type, slots in action_dict.items():
            if '-' in act_type:
                domain, intent = act_type.split('-', 1)
                for slot_value in slots:
                    slot = slot_value[0] if slot_value else 'none'
                    action_key = f"{domain}-{intent}-{slot}"
                    if action_key in self.action_to_idx:
                        vector[self.action_to_idx[action_key]] = 1.0
                        
        return vector
    
    def vector_to_action(self, vector: np.ndarray, threshold: float = 0.5) -> Dict:
        """벡터를 액션으로 변환"""
        action_dict = {}
        
        for idx, prob in enumerate(vector):
            if prob > threshold:
                action_key = self.idx_to_action[idx]
                domain, intent, slot = action_key.split('-')
                act_type = f"{domain}-{intent}"
                
                if act_type not in action_dict:
                    action_dict[act_type] = []
                action_dict[act_type].append([slot, ""])
                
        return action_dict

class PolicyNetwork(nn.Module):
    """정책 네트워크"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # 0-1 범위의 액션 확률
        )
        
    def forward(self, state):
        return self.network(state)

class ValueNetwork(nn.Module):
    """가치 네트워크 (Critic)"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        return self.network(state)

class PPOAgent:
    """PPO 에이전트"""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 네트워크 초기화
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)
        
        # 옵티마이저
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # PPO 하이퍼파라미터
        self.epsilon = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # 경험 버퍼
        self.buffer = []
        
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """액션 선택"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            value = self.value_net(state_tensor)
            
        # 확률적 액션 샘플링
        action = torch.bernoulli(action_probs).cpu().numpy()[0]
        
        return action, value.item()
    
    def store_experience(self, state, action, reward, next_state, done, value):
        """경험 저장"""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'value': value
        })
    
    def update(self):
        """정책 업데이트"""
        if len(self.buffer) < 32:  # 최소 배치 크기
            return
            
        # GAE 계산
        states, actions, rewards, values, advantages = self._compute_gae()
        
        # PPO 업데이트
        self._ppo_update(states, actions, rewards, values, advantages)
        
        # 버퍼 클리어
        self.buffer.clear()
    
    def _compute_gae(self):
        """Generalized Advantage Estimation 계산"""
        states = torch.FloatTensor([exp['state'] for exp in self.buffer]).to(self.device)
        actions = torch.FloatTensor([exp['action'] for exp in self.buffer]).to(self.device)
        rewards = [exp['reward'] for exp in self.buffer]
        values = [exp['value'] for exp in self.buffer]
        
        # GAE 계산
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
                
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        return states, actions, returns, torch.FloatTensor(values).to(self.device), advantages
    
    def _ppo_update(self, states, actions, returns, old_values, advantages):
        """PPO 정책 업데이트"""
        # 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 여러 에포크 업데이트
        for _ in range(4):
            # 정책 네트워크 업데이트
            new_action_probs = self.policy_net(states)
            
            # 정책 비율 계산 (간단화된 버전)
            old_action_probs = actions  # 이전 액션 확률로 근사
            ratio = new_action_probs / (old_action_probs + 1e-8)
            
            # PPO loss
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages.unsqueeze(1)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # 가치 네트워크 업데이트
            new_values = self.value_net(states).squeeze()
            value_loss = nn.MSELoss()(new_values, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

class DialogueEnvironment:
    """대화 환경"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
        self.state = DialogueState()
        self.action_space = DialogueAction()
        self.current_dialogue = None
        self.current_turn = 0
        
    def reset(self) -> np.ndarray:
        """환경 리셋"""
        self.state.reset()
        self.current_dialogue = random.choice(self.data)
        self.current_turn = 0
        return self.state.get_state_vector()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """환경 스텝"""
        self.current_turn += 1
        self.state.turn_count = self.current_turn
        
        # 액션을 대화 액션으로 변환
        action_dict = self.action_space.vector_to_action(action)
        
        # 보상 계산
        reward = self._calculate_reward(action_dict)
        
        # 종료 조건 확인
        done = self.current_turn >= 20 or self._is_dialogue_complete()
        
        # 다음 상태
        next_state = self.state.get_state_vector()
        
        info = {
            'turn': self.current_turn,
            'action': action_dict
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, action_dict: Dict) -> float:
        """보상 계산"""
        reward = 0
        
        # 정보 제공 시 보상
        if any('inform' in act for act in action_dict.keys()):
            reward += 1
        
        # 예약 완료 시 큰 보상
        if any('book' in act for act in action_dict.keys()):
            reward += 5
        
        # 대화 종료 시 보상
        if any('bye' in act for act in action_dict.keys()):
            reward += 2
        
        # 턴이 길어질수록 페널티
        reward -= 0.1
        
        return reward
    
    def _is_dialogue_complete(self) -> bool:
        """대화 완료 여부 확인"""
        # 간단한 완료 조건 (실제로는 더 정교한 로직 필요)
        return self.current_turn > 10

class DialogueTrainer:
    """대화 시스템 훈련기"""
    
    def __init__(self, data_path: str):
        # 데이터 로드
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 환경 및 에이전트 설정
        self.env = DialogueEnvironment(self.data)
        state_dim = len(self.env.state.get_state_vector())
        action_dim = self.env.action_space.action_dim
        
        self.agent = PPOAgent(state_dim, action_dim)
        
        # 훈련 통계
        self.episode_rewards = []
        self.episode_lengths = []
    
    def train(self, num_episodes: int = 1000):
        """훈련 실행"""
        print(f"Starting training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # 액션 선택
                action, value = self.agent.select_action(state)
                
                # 환경 스텝
                next_state, reward, done, info = self.env.step(action)
                
                # 경험 저장
                self.agent.store_experience(state, action, reward, next_state, done, value)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # 에이전트 업데이트
            self.agent.update()
            
            # 통계 저장
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # 주기적 출력
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.1f}")
    
    def save_model(self, path: str):
        """모델 저장"""
        torch.save({
            'policy_net': self.agent.policy_net.state_dict(),
            'value_net': self.agent.value_net.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path)
        self.agent.policy_net.load_state_dict(checkpoint['policy_net'])
        self.agent.value_net.load_state_dict(checkpoint['value_net'])
        print(f"Model loaded from {path}")

def main():
    """메인 실행 함수"""
    print("=== Alternative RL Dialogue System ===")
    
    # 훈련 시작
    trainer = DialogueTrainer("test_result.json")
    trainer.train(num_episodes=2000)
    
    # 모델 저장
    trainer.save_model("dialogue_policy.pth")
    
    print("Training completed!")

if __name__ == "__main__":
    main()
