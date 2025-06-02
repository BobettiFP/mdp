#!/usr/bin/env python3
"""
Dynamic Slot Generation RL 대화 시스템 (수정된 버전)
상태 벡터 차원 불일치 문제 해결
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Set
import random
from collections import deque, defaultdict
import pickle
import os
from datetime import datetime
import re

# 시드 설정
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class DynamicSlotExtractor:
    """LLM을 사용한 동적 슬롯 추출기"""
    
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.discovered_slots = {
            'hotel': set(),
            'restaurant': set(), 
            'train': set(),
            'taxi': set(),
            'attraction': set(),
            'general': set()
        }
        
        # 기본 도메인만 정의 (슬롯은 동적 생성)
        self.domains = ['hotel', 'restaurant', 'train', 'taxi', 'attraction', 'general']
        
        # 고정된 최대 슬롯 수 설정 (차원 일관성을 위해)
        self.max_slots_per_domain = 20
        
    def extract_slots_from_utterance(self, utterance: str, context: str = "") -> Dict[str, Set[str]]:
        """발화에서 동적으로 슬롯 추출"""
        
        try:
            # LLM 대신 규칙 기반 추출 사용 (안정성을 위해)
            slots = self._extract_slots_rule_based(utterance)
            
            # 발견된 슬롯들을 저장 (최대 개수 제한)
            for domain, domain_slots in slots.items():
                current_slots = self.discovered_slots[domain]
                new_slots = domain_slots - current_slots
                
                # 최대 슬롯 수 제한
                if len(current_slots) < self.max_slots_per_domain:
                    slots_to_add = list(new_slots)[:self.max_slots_per_domain - len(current_slots)]
                    self.discovered_slots[domain].update(slots_to_add)
                
            return slots
            
        except Exception as e:
            print(f"슬롯 추출 실패: {e}")
            return self._fallback_slot_extraction(utterance)
    
    def _extract_slots_rule_based(self, utterance: str) -> Dict[str, Set[str]]:
        """규칙 기반 슬롯 추출 (LLM 대체)"""
        slots = {domain: set() for domain in self.domains}
        utterance_lower = utterance.lower()
        
        # 호텔 관련 슬롯
        if any(word in utterance_lower for word in ['hotel', 'accommodation', 'room']):
            if 'parking' in utterance_lower:
                slots['hotel'].add('parking_availability')
            if any(word in utterance_lower for word in ['internet', 'wifi']):
                slots['hotel'].add('internet_access')
            if any(word in utterance_lower for word in ['expensive', 'luxury', 'premium']):
                slots['hotel'].add('price_range_high')
            if any(word in utterance_lower for word in ['cheap', 'budget', 'affordable']):
                slots['hotel'].add('price_range_low')
            if any(word in utterance_lower for word in ['area', 'location', 'centre', 'center']):
                slots['hotel'].add('location_preference')
            if any(word in utterance_lower for word in ['business', 'amenities']):
                slots['hotel'].add('business_facilities')
            if any(word in utterance_lower for word in ['executive', 'lounge']):
                slots['hotel'].add('executive_services')
                
        # 레스토랑 관련 슬롯
        if any(word in utterance_lower for word in ['restaurant', 'food', 'dining']):
            if any(word in utterance_lower for word in ['chinese', 'korean', 'italian', 'french']):
                slots['restaurant'].add('cuisine_type')
            if any(word in utterance_lower for word in ['expensive', 'fine', 'upscale']):
                slots['restaurant'].add('price_range_high')
            if any(word in utterance_lower for word in ['cheap', 'budget']):
                slots['restaurant'].add('price_range_low')
            if any(word in utterance_lower for word in ['centre', 'center', 'area']):
                slots['restaurant'].add('location_preference')
            if any(word in utterance_lower for word in ['private', 'room']):
                slots['restaurant'].add('private_dining')
            if any(word in utterance_lower for word in ['organic', 'eco']):
                slots['restaurant'].add('sustainability_focus')
            if any(word in utterance_lower for word in ['sommelier', 'wine']):
                slots['restaurant'].add('wine_services')
                
        # 기차 관련 슬롯
        if 'train' in utterance_lower:
            if any(word in utterance_lower for word in ['tuesday', 'wednesday', 'saturday']):
                slots['train'].add('departure_day')
            if any(word in utterance_lower for word in ['cambridge', 'london', 'birmingham']):
                slots['train'].add('route_specification')
            if any(word in utterance_lower for word in ['wifi', 'internet']):
                slots['train'].add('connectivity_services')
            if any(word in utterance_lower for word in ['first', 'class']):
                slots['train'].add('service_class')
            if any(word in utterance_lower for word in ['catering', 'food']):
                slots['train'].add('onboard_services')
                
        # 택시 관련 슬롯
        if 'taxi' in utterance_lower:
            if any(word in utterance_lower for word in ['child', 'safety', 'seat']):
                slots['taxi'].add('child_safety_features')
            if any(word in utterance_lower for word in ['pet', 'friendly']):
                slots['taxi'].add('pet_accommodation')
            if any(word in utterance_lower for word in ['leather', 'premium']):
                slots['taxi'].add('comfort_features')
            if any(word in utterance_lower for word in ['climate', 'air']):
                slots['taxi'].add('climate_control')
                
        # 관광지 관련 슬롯
        if any(word in utterance_lower for word in ['attraction', 'museum', 'tour']):
            if any(word in utterance_lower for word in ['educational', 'interactive']):
                slots['attraction'].add('educational_value')
            if any(word in utterance_lower for word in ['vip', 'guided']):
                slots['attraction'].add('tour_services')
            if any(word in utterance_lower for word in ['photography', 'photo']):
                slots['attraction'].add('photography_permissions')
            if any(word in utterance_lower for word in ['family', 'children']):
                slots['attraction'].add('family_friendly')
                
        return slots
    
    def _fallback_slot_extraction(self, utterance: str) -> Dict[str, Set[str]]:
        """폴백: 기본 슬롯 추출"""
        slots = {domain: set() for domain in self.domains}
        utterance_lower = utterance.lower()
        
        if 'hotel' in utterance_lower:
            slots['hotel'].add('basic_requirement')
        if 'restaurant' in utterance_lower:
            slots['restaurant'].add('basic_requirement')
        if 'train' in utterance_lower:
            slots['train'].add('basic_requirement')
            
        return slots
    
    def get_all_discovered_slots(self) -> Dict[str, List[str]]:
        """지금까지 발견된 모든 슬롯 반환"""
        return {domain: list(slots) for domain, slots in self.discovered_slots.items()}
    
    def get_current_slot_vocabulary_size(self) -> int:
        """현재 슬롯 어휘 크기"""
        total_slots = sum(len(slots) for slots in self.discovered_slots.values())
        return total_slots
    
    def get_fixed_slot_vector_size(self) -> int:
        """고정된 슬롯 벡터 크기 반환"""
        return len(self.domains) * self.max_slots_per_domain

class DynamicDialogueState:
    """동적 슬롯을 지원하는 대화 상태 (고정 크기 벡터)"""
    
    def __init__(self, slot_extractor: DynamicSlotExtractor):
        self.slot_extractor = slot_extractor
        self.fixed_state_size = 200  # 고정된 상태 벡터 크기
        self.reset()
    
    def reset(self):
        """상태 초기화"""
        self.state = {domain: {} for domain in self.slot_extractor.domains}
        self.history = []
        self.turn_count = 0
        self.active_slots = set()
        
    def update_from_utterance(self, utterance: str, speaker: str = "USER"):
        """발화에서 동적으로 상태 업데이트"""
        self.turn_count += 1
        
        # 슬롯 추출
        context = " ".join([h['utterance'] for h in self.history[-3:]])
        extracted_slots = self.slot_extractor.extract_slots_from_utterance(utterance, context)
        
        # 상태 업데이트
        for domain, slots in extracted_slots.items():
            for slot in slots:
                if slot:
                    self.state[domain][slot] = self._extract_value_for_slot(utterance, slot)
                    self.active_slots.add(f"{domain}.{slot}")
        
        # 히스토리 업데이트
        self.history.append({
            'speaker': speaker,
            'utterance': utterance,
            'extracted_slots': extracted_slots
        })
        
    def _extract_value_for_slot(self, utterance: str, slot: str) -> str:
        """슬롯에 대한 값 추출"""
        if 'price' in slot or 'budget' in slot:
            if 'expensive' in utterance.lower() or 'luxury' in utterance.lower():
                return "expensive"
            elif 'cheap' in utterance.lower() or 'budget' in utterance.lower():
                return "cheap"
            else:
                return "mentioned"
        elif 'location' in slot or 'area' in slot:
            areas = ['centre', 'center', 'south', 'north', 'east', 'west', 'cambridge']
            for area in areas:
                if area in utterance.lower():
                    return area
            return "specified"
        else:
            return "mentioned"
    
    def get_dynamic_state_vector(self) -> np.ndarray:
        """고정 크기 상태 벡터 반환"""
        vector = np.zeros(self.fixed_state_size, dtype=np.float32)
        
        # 도메인별 슬롯 인코딩 (각 도메인당 30개 슬롯 할당)
        idx = 0
        for domain in self.slot_extractor.domains:
            domain_slots = list(self.slot_extractor.discovered_slots[domain])
            
            # 각 도메인마다 최대 30개 슬롯
            for i in range(min(30, len(domain_slots))):
                if idx < self.fixed_state_size - 10:  # 메타 정보를 위한 공간 확보
                    slot_key = f"{domain}.{domain_slots[i]}"
                    if slot_key in self.active_slots:
                        vector[idx] = 1.0
                    idx += 1
        
        # 메타 정보 (마지막 10개 차원)
        meta_start = self.fixed_state_size - 10
        if meta_start > 0:
            vector[meta_start] = min(self.turn_count / 20.0, 1.0)     # 정규화된 턴 수
            vector[meta_start + 1] = min(len(self.active_slots) / 50.0, 1.0)  # 활성 슬롯 비율
            vector[meta_start + 2] = min(len(self.history) / 10.0, 1.0)       # 히스토리 길이
            
            # 도메인별 활성화 정보
            for i, domain in enumerate(self.slot_extractor.domains[:6]):
                if meta_start + 3 + i < self.fixed_state_size:
                    domain_active = sum(1 for slot in self.active_slots if slot.startswith(f"{domain}."))
                    vector[meta_start + 3 + i] = min(domain_active / 10.0, 1.0)
        
        return vector

class DynamicActionSpace:
    """동적 액션 공간 (고정 크기)"""
    
    def __init__(self, slot_extractor: DynamicSlotExtractor):
        self.slot_extractor = slot_extractor
        self.action_types = ['inform', 'request', 'confirm', 'select', 'recommend', 
                           'book', 'negate', 'affirm', 'thank', 'bye', 'greet']
        self.fixed_action_size = 1000  # 고정된 액션 공간 크기
        self.update_action_space()
        
    def update_action_space(self):
        """동적으로 액션 공간 업데이트"""
        self.action_to_idx = {}
        self.idx_to_action = {}
        idx = 0
        
        # 도메인-액션 타입 조합
        for domain in self.slot_extractor.domains:
            for action_type in self.action_types:
                # 발견된 슬롯들에 대한 액션 (제한된 수)
                domain_slots = list(self.slot_extractor.discovered_slots[domain])
                for slot in domain_slots[:10]:  # 도메인당 최대 10개 슬롯
                    if idx < self.fixed_action_size and slot:
                        action_key = f"{domain}-{action_type}-{slot}"
                        self.action_to_idx[action_key] = idx
                        self.idx_to_action[idx] = action_key
                        idx += 1
                
                # 일반적인 도메인 액션
                if idx < self.fixed_action_size:
                    action_key = f"{domain}-{action_type}-general"
                    self.action_to_idx[action_key] = idx
                    self.idx_to_action[idx] = action_key
                    idx += 1
        
        # 나머지 공간을 플레이스홀더로 채움
        while idx < self.fixed_action_size:
            self.action_to_idx[f"placeholder_{idx}"] = idx
            self.idx_to_action[idx] = f"placeholder_{idx}"
            idx += 1
            
        self.action_dim = self.fixed_action_size
        
    def get_action_dim(self) -> int:
        """액션 차원 반환"""
        return self.action_dim
    
    def vector_to_action(self, vector: np.ndarray, threshold: float = 0.5) -> Dict:
        """벡터를 액션으로 변환"""
        action_dict = {}
        
        for idx, prob in enumerate(vector):
            if idx < len(self.idx_to_action) and prob > threshold:
                action_key = self.idx_to_action[idx]
                
                if not action_key.startswith("placeholder"):
                    parts = action_key.split('-')
                    if len(parts) >= 3:
                        domain, intent, slot = parts[0], parts[1], '-'.join(parts[2:])
                        act_type = f"{domain}-{intent}"
                        
                        if act_type not in action_dict:
                            action_dict[act_type] = []
                        action_dict[act_type].append([slot, ""])
                
        return action_dict

class DynamicPolicyNetwork(nn.Module):
    """동적 정책 네트워크 (고정 크기 입출력)"""
    
    def __init__(self, state_dim: int = 200, action_dim: int = 1000, hidden_dim: int = 512):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 고정 크기 네트워크
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()
        )
        
    def forward(self, state):
        # 입력 크기 검증
        if state.shape[-1] != self.state_dim:
            print(f"Warning: Expected state dim {self.state_dim}, got {state.shape[-1]}")
            # 크기 조정
            if state.shape[-1] < self.state_dim:
                # 패딩
                padding = torch.zeros(*state.shape[:-1], self.state_dim - state.shape[-1], device=state.device)
                state = torch.cat([state, padding], dim=-1)
            else:
                # 자르기
                state = state[..., :self.state_dim]
        
        return self.network(state)

class DynamicDialogueEnvironment:
    """동적 슬롯을 지원하는 대화 환경"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
        self.slot_extractor = DynamicSlotExtractor()
        self.state = DynamicDialogueState(self.slot_extractor)
        self.action_space = DynamicActionSpace(self.slot_extractor)
        self.current_dialogue = None
        self.current_turn_idx = 0
        
        # 데이터에서 슬롯 사전 학습
        self._bootstrap_slots_from_data()
        
    def _bootstrap_slots_from_data(self):
        """데이터에서 슬롯 사전 학습"""
        print("Bootstrapping slots from training data...")
        
        sample_utterances = [
            "I need a hotel with parking and internet",
            "Looking for a cheap restaurant in the centre", 
            "Want a train to Cambridge on Tuesday",
            "Need a taxi for multiple passengers",
            "Searching for educational attractions"
        ]
        
        for utterance in sample_utterances:
            self.slot_extractor.extract_slots_from_utterance(utterance)
        
        # 훈련 데이터에서도 일부 추출
        for dialogue in self.data[:10]:
            for turn in dialogue.get('turns', []):
                if turn.get('speaker') == 'USER':
                    utterance = turn.get('utterance', '')
                    if utterance:
                        self.slot_extractor.extract_slots_from_utterance(utterance)
        
        # 액션 공간 업데이트
        self.action_space.update_action_space()
        
        print(f"Discovered slots: {self.slot_extractor.get_current_slot_vocabulary_size()}")
        print(f"Action space size: {self.action_space.get_action_dim()}")
        
    def reset(self) -> np.ndarray:
        """환경 리셋"""
        self.state.reset()
        self.current_dialogue = random.choice(self.data)
        self.current_turn_idx = 0
        
        # 첫 번째 사용자 발화 처리
        user_turns = [turn for turn in self.current_dialogue.get('turns', []) 
                     if turn.get('speaker') == 'USER']
        
        if user_turns:
            first_utterance = user_turns[0].get('utterance', '')
            if first_utterance:
                self.state.update_from_utterance(first_utterance, 'USER')
        
        state_vector = self.state.get_dynamic_state_vector()
        print(f"Reset - State vector shape: {state_vector.shape}")
        return state_vector
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """환경 스텝"""
        self.current_turn_idx += 1
        
        # 액션 변환
        action_dict = self.action_space.vector_to_action(action, threshold=0.3)
        
        # 다음 사용자 발화
        next_user_utterance = self._get_next_user_utterance()
        
        if next_user_utterance:
            old_slot_count = self.slot_extractor.get_current_slot_vocabulary_size()
            self.state.update_from_utterance(next_user_utterance, 'USER')
            new_slot_count = self.slot_extractor.get_current_slot_vocabulary_size()
            
            # 새 슬롯 발견 시 보상
            if new_slot_count > old_slot_count:
                self.action_space.update_action_space()
                discovery_bonus = (new_slot_count - old_slot_count) * 2.0
            else:
                discovery_bonus = 0.0
        else:
            discovery_bonus = 0.0
            next_user_utterance = ""
        
        # 보상 계산
        reward = self._calculate_reward(action_dict, discovery_bonus)
        
        # 종료 조건
        done = (self.current_turn_idx >= 15 or 
                'bye' in str(action_dict).lower() or
                self._is_goal_achieved())
        
        # 다음 상태
        next_state = self.state.get_dynamic_state_vector()
        
        info = {
            'turn': self.current_turn_idx,
            'action': action_dict,
            'discovered_slots': self.slot_extractor.get_current_slot_vocabulary_size(),
            'active_slots': len(self.state.active_slots),
            'user_utterance': next_user_utterance,
            'slot_discovery_bonus': discovery_bonus
        }
        
        return next_state, reward, done, info
    
    def _get_next_user_utterance(self) -> str:
        """다음 사용자 발화 가져오기"""
        user_turns = [turn for turn in self.current_dialogue.get('turns', []) 
                     if turn.get('speaker') == 'USER']
        
        if self.current_turn_idx < len(user_turns):
            return user_turns[self.current_turn_idx].get('utterance', '')
        return ""
    
    def _calculate_reward(self, action_dict: Dict, discovery_bonus: float) -> float:
        """보상 계산"""
        reward = 0.0
        
        # 기본 액션 보상
        for act_type in action_dict.keys():
            if 'inform' in act_type:
                reward += 1.0
            elif 'request' in act_type:
                reward += 0.5
            elif 'book' in act_type:
                reward += 3.0
            elif 'bye' in act_type:
                reward += 1.0
        
        # 슬롯 발견 보상
        reward += discovery_bonus
        
        # 활성 슬롯 활용 보상
        if len(self.state.active_slots) > 0:
            reward += len(self.state.active_slots) * 0.1
        
        # 턴 길이 페널티
        reward -= 0.1
        
        return reward
    
    def _is_goal_achieved(self) -> bool:
        """목표 달성 여부"""
        return len(self.state.active_slots) >= 3

class DynamicDialogueTrainer:
    """동적 슬롯 대화 시스템 훈련기"""
    
    def __init__(self, data_path: str):
        print("Loading data for dynamic slot learning...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} dialogues")
        
        # 환경 및 네트워크 초기화
        self.env = DynamicDialogueEnvironment(self.data)
        self.policy_net = DynamicPolicyNetwork(state_dim=200, action_dim=1000)
        self.value_net = nn.Sequential(
            nn.Linear(200, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 옵티마이저
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=3e-4)
        
        # 훈련 통계
        self.episode_rewards = []
        self.slot_discoveries = []
        self.action_space_sizes = []
        
        # 로그 디렉토리
        self.log_dir = f"dynamic_logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
        
    def train(self, num_episodes: int = 2000):
        """훈련 실행"""
        print("=" * 60)
        print("🚀 Starting Dynamic Slot Discovery Training")
        print("=" * 60)
        
        for episode in range(num_episodes):
            try:
                state = self.env.reset()
                episode_reward = 0
                
                # 상태 벡터 크기 검증
                if len(state) != 200:
                    print(f"Warning: State vector size {len(state)}, expected 200")
                    continue
                
                for step in range(15):  # 최대 15스텝
                    # 정책 네트워크로 액션 선택
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    
                    with torch.no_grad():
                        action_probs = self.policy_net(state_tensor)
                        action = torch.bernoulli(action_probs).cpu().numpy()[0]
                    
                    # 환경 스텝
                    next_state, reward, done, info = self.env.step(action)
                    
                    episode_reward += reward
                    state = next_state
                    
                    if done:
                        break
                
                # 통계 저장
                self.episode_rewards.append(episode_reward)
                self.slot_discoveries.append(self.env.slot_extractor.get_current_slot_vocabulary_size())
                self.action_space_sizes.append(self.env.action_space.get_action_dim())
                
                # 주기적 출력
                if episode % 100 == 0:
                    self._log_progress(episode)
                    
            except Exception as e:
                print(f"Episode {episode} failed: {e}")
                continue
        
        # 최종 결과 저장
        self._save_results()
        print("🎉 Training completed!")
    
    def _log_progress(self, episode: int):
        """진행 상황 로깅"""
        window = 100
        recent_rewards = self.episode_rewards[-window:] if len(self.episode_rewards) >= window else self.episode_rewards
        
        if recent_rewards:
            avg_reward = np.mean(recent_rewards)
            current_slots = self.slot_discoveries[-1] if self.slot_discoveries else 0
            
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Discovered Slots: {current_slots:3d}")
    
    def _save_results(self):
        """결과 저장"""
        results = {
            'episode_rewards': self.episode_rewards,
            'slot_discoveries': self.slot_discoveries,
            'action_space_sizes': self.action_space_sizes,
            'final_discovered_slots': self.env.slot_extractor.get_all_discovered_slots()
        }
        
        with open(f"{self.log_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # 시각화
        if len(self.episode_rewards) > 0:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(self.episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            
            plt.subplot(2, 2, 2)
            plt.plot(self.slot_discoveries)
            plt.title('Slot Discovery Over Time')
            plt.xlabel('Episode')
            plt.ylabel('Total Discovered Slots')
            
            plt.subplot(2, 2, 3)
            if len(self.slot_discoveries) > 1:
                discovery_rate = np.diff(self.slot_discoveries)
                plt.plot(discovery_rate)
                plt.title('Slot Discovery Rate')
                plt.xlabel('Episode')
                plt.ylabel('New Slots per Episode')
            
            plt.subplot(2, 2, 4)
            recent_rewards = []
            window_size = 50
            for i in range(len(self.episode_rewards)):
                start = max(0, i - window_size + 1)
                window_avg = np.mean(self.episode_rewards[start:i+1])
                recent_rewards.append(window_avg)
            plt.plot(recent_rewards)
            plt.title('Moving Average Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            
            plt.tight_layout()
            plt.savefig(f"{self.log_dir}/training_results.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Results saved to {self.log_dir}/")
    
    def test_dynamic_slot_generation(self, test_utterances: List[str]):
        """동적 슬롯 생성 테스트"""
        print("\n" + "=" * 60)
        print("🧪 DYNAMIC SLOT GENERATION TEST")
        print("=" * 60)
        
        for i, utterance in enumerate(test_utterances, 1):
            print(f"\n{i}. Test Utterance: '{utterance}'")
            
            # 슬롯 추출
            extracted_slots = self.env.slot_extractor.extract_slots_from_utterance(utterance)
            
            print("   Generated Slots:")
            for domain, slots in extracted_slots.items():
                if slots:
                    print(f"     {domain}: {list(slots)}")
            
            # 상태 업데이트 시뮬레이션
            test_state = DynamicDialogueState(self.env.slot_extractor)
            test_state.update_from_utterance(utterance)
            
            print(f"   Active Slots: {len(test_state.active_slots)}")
            print(f"   State Vector Dim: {len(test_state.get_dynamic_state_vector())}")
        
        # 전체 발견된 슬롯 요약
        print(f"\n📊 Total Discovered Slots: {self.env.slot_extractor.get_current_slot_vocabulary_size()}")
        print("\n🏷️  Discovered Slots by Domain:")
        for domain, slots in self.env.slot_extractor.get_all_discovered_slots().items():
            if slots:
                print(f"  {domain.upper()}: {len(slots)} slots")
                print(f"    Examples: {list(slots)[:3]}")

class SlotEvolutionAnalyzer:
    """슬롯 진화 분석기"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        
    def analyze_slot_evolution(self):
        """슬롯 진화 과정 분석"""
        print("\n" + "=" * 60)
        print("🔬 SLOT EVOLUTION ANALYSIS")
        print("=" * 60)
        
        # 결과 파일 로드
        results_path = os.path.join(self.log_dir, 'results.json')
        if not os.path.exists(results_path):
            print(f"❌ Results file not found: {results_path}")
            return
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        slot_discoveries = results.get('slot_discoveries', [])
        episode_rewards = results.get('episode_rewards', [])
        final_slots = results.get('final_discovered_slots', {})
        
        if not slot_discoveries:
            print("❌ No slot discovery data found")
            return
        
        # 기본 통계
        initial_slots = slot_discoveries[0] if slot_discoveries else 0
        final_slots_count = slot_discoveries[-1] if slot_discoveries else 0
        total_growth = final_slots_count - initial_slots
        
        print(f"📈 Slot Discovery Statistics:")
        print(f"  Initial Slots: {initial_slots}")
        print(f"  Final Slots: {final_slots_count}")
        print(f"  Total Growth: {total_growth}")
        
        if len(slot_discoveries) > 1:
            growth_rate = total_growth / len(slot_discoveries)
            print(f"  Average Growth Rate: {growth_rate:.3f} slots/episode")
        
        # 도메인별 분석
        print(f"\n🏷️  Final Slot Distribution:")
        total_final = 0
        for domain, slots in final_slots.items():
            slot_count = len(slots) if isinstance(slots, list) else 0
            total_final += slot_count
            if slot_count > 0:
                print(f"  {domain.upper()}: {slot_count} slots")
                # 몇 가지 예시 슬롯 출력
                example_slots = slots[:3] if isinstance(slots, list) else []
                if example_slots:
                    print(f"    Examples: {example_slots}")
        
        # 혁신성 분석
        print(f"\n🚀 Innovation Analysis:")
        innovative_keywords = ['preference', 'requirement', 'specification', 'flexibility', 'sensitivity']
        innovative_count = 0
        
        for domain, slots in final_slots.items():
            if isinstance(slots, list):
                for slot in slots:
                    if any(keyword in slot.lower() for keyword in innovative_keywords):
                        innovative_count += 1
        
        innovation_ratio = innovative_count / total_final if total_final > 0 else 0
        print(f"  Innovative Slots: {innovative_count}/{total_final} ({innovation_ratio:.1%})")
        
        # 시각화
        self._create_evolution_plots(slot_discoveries, episode_rewards)
    
    def _create_evolution_plots(self, slot_discoveries, episode_rewards):
        """진화 과정 시각화"""
        plt.figure(figsize=(15, 10))
        
        # 슬롯 발견 곡선
        plt.subplot(2, 3, 1)
        plt.plot(slot_discoveries, 'b-', linewidth=2)
        plt.title('Slot Discovery Curve', fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('Total Discovered Slots')
        plt.grid(True, alpha=0.3)
        
        # 슬롯 발견 속도
        if len(slot_discoveries) > 1:
            plt.subplot(2, 3, 2)
            discovery_rate = np.diff(slot_discoveries)
            plt.plot(discovery_rate, 'g-', linewidth=2)
            plt.title('Slot Discovery Rate', fontweight='bold')
            plt.xlabel('Episode')
            plt.ylabel('New Slots per Episode')
            plt.grid(True, alpha=0.3)
        
        # 누적 보상
        if episode_rewards:
            plt.subplot(2, 3, 3)
            cumulative_rewards = np.cumsum(episode_rewards)
            plt.plot(cumulative_rewards, 'r-', linewidth=2)
            plt.title('Cumulative Rewards', fontweight='bold')
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Reward')
            plt.grid(True, alpha=0.3)
        
        # 슬롯 발견과 보상의 상관관계
        if episode_rewards and len(episode_rewards) == len(slot_discoveries):
            plt.subplot(2, 3, 4)
            plt.scatter(slot_discoveries, episode_rewards, alpha=0.6)
            plt.title('Slots vs Rewards', fontweight='bold')
            plt.xlabel('Discovered Slots')
            plt.ylabel('Episode Reward')
            plt.grid(True, alpha=0.3)
        
        # 이동평균 보상
        if episode_rewards:
            plt.subplot(2, 3, 5)
            window = min(50, len(episode_rewards) // 4)
            if window > 1:
                moving_avg = []
                for i in range(len(episode_rewards)):
                    start = max(0, i - window + 1)
                    avg = np.mean(episode_rewards[start:i+1])
                    moving_avg.append(avg)
                plt.plot(moving_avg, 'purple', linewidth=2)
                plt.title(f'Moving Average Rewards (window={window})', fontweight='bold')
                plt.xlabel('Episode')
                plt.ylabel('Average Reward')
                plt.grid(True, alpha=0.3)
        
        # 학습 효율성 (보상/슬롯 비율)
        if episode_rewards and slot_discoveries:
            plt.subplot(2, 3, 6)
            efficiency = []
            for i, (reward, slots) in enumerate(zip(episode_rewards, slot_discoveries)):
                if slots > 0:
                    efficiency.append(reward / slots)
                else:
                    efficiency.append(0)
            
            if efficiency:
                plt.plot(efficiency, 'orange', linewidth=2)
                plt.title('Learning Efficiency', fontweight='bold')
                plt.xlabel('Episode')
                plt.ylabel('Reward per Slot')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/slot_evolution_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Evolution analysis saved to {self.log_dir}/slot_evolution_analysis.png")

def main():
    """메인 실행 함수"""
    print("🌟 Dynamic Slot Generation RL Dialogue System (Fixed Version)")
    print("=" * 70)
    
    try:
        # 훈련 시작
        trainer = DynamicDialogueTrainer("test_result.json")
        trainer.train(num_episodes=1000)  # 더 안정적인 훈련을 위해 줄임
        
        # 테스트 발화들
        test_utterances = [
            "I need a hotel with exceptional parking facilities and high-speed internet for business travelers",
            "Looking for a restaurant with authentic cuisine and romantic ambiance for anniversary dinner",
            "Want a train with flexible booking options and comfortable seating for elderly passengers",
            "Need a taxi service with child safety features and environmentally friendly vehicles",
            "Searching for tourist attractions with accessibility support and educational value for families"
        ]
        
        # 동적 슬롯 생성 테스트
        trainer.test_dynamic_slot_generation(test_utterances)
        
        print(f"\n🎉 Training completed! Check {trainer.log_dir} for results.")
        
        # 분석 실행
        analyzer = SlotEvolutionAnalyzer(trainer.log_dir)
        analyzer.analyze_slot_evolution()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()