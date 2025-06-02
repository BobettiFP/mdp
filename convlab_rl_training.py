import json
import os
from typing import Dict, List, Any
import torch
import numpy as np
from convlab.policy import PPO
from convlab.util.multiwoz.state import default_state
from convlab.dst import RuleDST
from convlab.nlg import TemplateNLG  
from convlab.nlu import MILU
from convlab.policy.rule import RulePolicy
from convlab.evaluator.multiwoz_eval import MultiWozEvaluator

class DataConverter:
    """기존 annotation 데이터를 ConvLab 형식으로 변환"""
    
    def __init__(self):
        self.domain_mapping = {
            'hotel': 'hotel',
            'restaurant': 'restaurant', 
            'train': 'train',
            'taxi': 'taxi',
            'attraction': 'attraction'
        }
        
    def convert_annotation_to_convlab(self, data: List[Dict]) -> Dict:
        """annotation 데이터를 ConvLab 형식으로 변환"""
        convlab_data = {}
        
        for dialogue in data:
            dialogue_id = dialogue['dialogue_id']
            convlab_dialogue = {
                'goal': self._extract_goal(dialogue),
                'log': self._convert_turns(dialogue['turns'])
            }
            convlab_data[dialogue_id] = convlab_dialogue
            
        return convlab_data
    
    def _extract_goal(self, dialogue: Dict) -> Dict:
        """대화에서 goal 정보 추출"""
        goal = {}
        services = dialogue.get('services', [])
        
        # 마지막 상태에서 goal 정보 추출
        if dialogue['turns']:
            last_turn = dialogue['turns'][-1]
            state = last_turn.get('state', {})
            
            for domain in services:
                if domain in state:
                    goal[domain] = {
                        'info': {},
                        'book': {},
                        'reqt': []
                    }
                    
                    domain_state = state[domain]
                    for slot, value in domain_state.items():
                        if 'book' in slot or slot in ['people', 'nights', 'duration']:
                            goal[domain]['book'][slot] = value
                        else:
                            goal[domain]['info'][slot] = value
                            
        return goal
    
    def _convert_turns(self, turns: List[Dict]) -> List[Dict]:
        """turn 데이터를 ConvLab 형식으로 변환"""
        convlab_turns = []
        
        for turn in turns:
            if turn['speaker'] == 'USER':
                # 사용자 턴
                user_turn = {
                    'text': turn['utterance'],
                    'metadata': self._convert_state(turn.get('state', {})),
                    'dialog_act': self._convert_dialogue_acts(turn.get('dialogue_acts', []))
                }
                convlab_turns.append(user_turn)
                
            elif turn['speaker'] == 'SYSTEM':
                # 시스템 턴
                system_turn = {
                    'text': turn['utterance'],
                    'metadata': {},
                    'dialog_act': self._convert_dialogue_acts(turn.get('dialogue_acts', []))
                }
                convlab_turns.append(system_turn)
                
        return convlab_turns
    
    def _convert_state(self, state: Dict) -> Dict:
        """상태 정보를 ConvLab metadata 형식으로 변환"""
        metadata = {}
        
        for domain, domain_state in state.items():
            if domain in self.domain_mapping:
                metadata[domain] = {
                    'book': {},
                    'semi': {}
                }
                
                for slot, value in domain_state.items():
                    if isinstance(value, dict) and 'value' in value:
                        value = value['value']
                    
                    if 'book' in slot or slot in ['people', 'nights']:
                        metadata[domain]['book'][slot] = value
                    else:
                        metadata[domain]['semi'][slot] = value
                        
        return metadata
    
    def _convert_dialogue_acts(self, dialogue_acts: List[Dict]) -> Dict:
        """dialogue acts를 ConvLab 형식으로 변환"""
        acts = {}
        
        for act in dialogue_acts:
            intent = act.get('intent', '')
            domain = act.get('domain', 'general')
            slot = act.get('slot', '')
            value = act.get('value', '')
            
            if isinstance(value, dict) and 'value' in value:
                value = value['value']
                
            act_key = f"{domain}-{intent}"
            if act_key not in acts:
                acts[act_key] = []
                
            acts[act_key].append([slot, value])
            
        return acts

class ConvLabRLTrainer:
    """ConvLab RL 훈련 클래스"""
    
    def __init__(self, data_path: str = "converted_data.json"):
        self.data_path = data_path
        self.setup_components()
        
    def setup_components(self):
        """ConvLab 컴포넌트 설정"""
        # DST (Dialogue State Tracking)
        self.dst = RuleDST()
        
        # NLU (Natural Language Understanding) 
        self.nlu = MILU(mode='usr')
        
        # NLG (Natural Language Generation)
        self.nlg = TemplateNLG(is_user=False)
        
        # Policy (강화학습으로 훈련할 부분)
        self.policy = PPO()
        
        # User simulator (rule-based)
        self.user_policy = RulePolicy(character='usr')
        
        # Evaluator
        self.evaluator = MultiWozEvaluator()
        
    def load_converted_data(self) -> Dict:
        """변환된 데이터 로드"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def create_environment(self):
        """RL 환경 생성"""
        from convlab.env import Environment
        
        env = Environment(
            dst=self.dst,
            policy=self.policy,
            nlg=self.nlg,
            user_nlu=self.nlu,
            user_dst=RuleDST(),
            user_policy=self.user_policy,
            user_nlg=TemplateNLG(is_user=True)
        )
        return env
        
    def train_policy(self, num_episodes: int = 1000):
        """정책 네트워크 훈련"""
        env = self.create_environment()
        data = self.load_converted_data()
        
        # 훈련 루프
        for episode in range(num_episodes):
            # 환경 초기화
            env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # 현재 상태 가져오기
                state = env.get_state()
                
                # 정책에서 액션 선택
                action = self.policy.predict(state)
                
                # 액션 실행
                next_state, reward, done, info = env.step(action)
                
                # 경험 저장
                self.policy.update(state, action, reward, next_state, done)
                
                episode_reward += reward
                
            # 주기적으로 평가
            if episode % 100 == 0:
                self.evaluate_policy(episode, episode_reward)
                
    def evaluate_policy(self, episode: int, reward: float):
        """정책 평가"""
        print(f"Episode {episode}: Reward = {reward:.2f}")
        
        # 평가 메트릭 계산
        success_rate = self.evaluator.task_success()
        inform_f1 = self.evaluator.inform_F1()
        
        print(f"Success Rate: {success_rate:.3f}")
        print(f"Inform F1: {inform_f1:.3f}")
        
    def save_model(self, save_path: str = "trained_policy.pkl"):
        """훈련된 모델 저장"""
        self.policy.save(save_path)
        print(f"Model saved to {save_path}")

def main():
    """메인 실행 함수"""
    # 1. 데이터 변환
    print("Converting annotation data to ConvLab format...")
    
    # annotation 데이터 로드 (test_result.json)
    with open('test_result.json', 'r', encoding='utf-8') as f:
        annotation_data = json.load(f)
    
    # 데이터 변환
    converter = DataConverter()
    convlab_data = converter.convert_annotation_to_convlab(annotation_data)
    
    # 변환된 데이터 저장
    with open('converted_data.json', 'w', encoding='utf-8') as f:
        json.dump(convlab_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(convlab_data)} dialogues")
    
    # 2. RL 훈련 시작
    print("Starting RL training...")
    
    trainer = ConvLabRLTrainer("converted_data.json")
    trainer.train_policy(num_episodes=1000)
    
    # 3. 모델 저장
    trainer.save_model("multiwoz_policy.pkl")
    
    print("Training completed!")

# 추가: 훈련된 모델 테스트
def test_trained_model():
    """훈련된 모델 테스트"""
    from convlab.policy import PPO
    
    # 저장된 모델 로드
    policy = PPO()
    policy.load("multiwoz_policy.pkl")
    
    # 테스트 대화
    test_utterances = [
        "I need a hotel in Cambridge with free parking",
        "I want an expensive restaurant in the centre", 
        "Book me a train to London on Saturday"
    ]
    
    dst = RuleDST()
    
    for utterance in test_utterances:
        print(f"\nUser: {utterance}")
        
        # 상태 업데이트
        state = dst.update(utterance)
        
        # 정책에서 응답 생성
        action = policy.predict(state)
        print(f"System action: {action}")

if __name__ == "__main__":
    # 기본 훈련 실행
    main()
    
    # 훈련된 모델 테스트 (선택사항)
    # test_trained_model()
