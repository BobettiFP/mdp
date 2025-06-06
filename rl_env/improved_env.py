# ------------------------------  improved_env.py  -------------------------------------
"""
개선된 DialogueMDP - 더 현실적이고 도전적인 환경
"""
import json, random
from typing import Dict, Tuple, List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

def _vec(idx: Tuple[int,...], dim: int) -> np.ndarray:
    """Convert state indices to binary vector."""
    v = np.zeros(dim, dtype=np.int8)
    for i in idx: 
        if 0 <= i < dim:
            v[i] = 1
    return v

def state_to_idx(sd: Dict[str,str], vocab: Dict[str,int]) -> Tuple[int,...]:
    """Convert state dict to sorted tuple of vocab indices."""
    if not sd:
        return tuple()
    return tuple(sorted(vocab[s] for s in sd if s in vocab))

class ImprovedDialogueMDP(gym.Env):
    """개선된 대화 MDP - 더 현실적인 보상과 도전"""
    
    metadata = {"render_modes": []}

    def __init__(self, trans: Dict, slot_vocab: Dict[str,int], act_vocab: Dict[str,int], 
                 difficulty: str = "normal"):
        super().__init__()
        self.trans = trans
        self.slot_vocab = slot_vocab
        self.action_vocab = act_vocab
        self.difficulty = difficulty
        
        # 공간 정의
        self.observation_space = spaces.MultiBinary(len(slot_vocab))
        self.action_space = spaces.Discrete(len(act_vocab))
        
        # 가능한 상태들
        self._states = list({s for (s, _) in trans.keys()})
        if not self._states:
            self._states = [tuple()]
            
        self._current_state = None
        self._episode_step = 0
        self._max_episode_steps = 20  # 에피소드 길이 제한
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
            random.seed(seed)
        
        self._current_state = random.choice(self._states)
        self._episode_step = 0
        obs = _vec(self._current_state, len(self.slot_vocab))
        return obs, {}

    def step(self, action: int):
        """Take action with improved reward shaping."""
        self._episode_step += 1
        key = (self._current_state, int(action))
        
        # 기본 transition
        if key not in self.trans:
            obs = _vec(self._current_state, len(self.slot_vocab))
            reward = -0.1  # 유효하지 않은 액션 페널티
            return obs, reward, True, False, {"invalid_action": True}
        
        next_state, base_reward = self.trans[key]
        
        # 개선된 보상 설계
        reward = self._calculate_reward(
            self._current_state, action, next_state, base_reward
        )
        
        self._current_state = next_state
        obs = _vec(self._current_state, len(self.slot_vocab))
        
        # 종료 조건
        done = (
            base_reward > 0 or  # 성공적인 inform
            self._episode_step >= self._max_episode_steps  # 최대 스텝 도달
        )
        
        truncated = self._episode_step >= self._max_episode_steps
        
        return obs, reward, done, truncated, {
            "episode_step": self._episode_step,
            "base_reward": base_reward
        }
    
    def _calculate_reward(self, prev_state: Tuple, action: int, 
                         next_state: Tuple, base_reward: float) -> float:
        """개선된 보상 계산"""
        reward = base_reward
        
        # 난이도별 조정
        if self.difficulty == "hard":
            # 어려운 환경: 보상 감소, 더 까다로운 조건
            reward *= 0.5
            
            # 상태 크기에 따른 보상 조정
            state_complexity = len(next_state)
            if state_complexity > 3:  # 복잡한 상태
                reward *= 0.8
                
        elif self.difficulty == "easy":
            # 쉬운 환경: 보상 증가
            reward *= 1.5
            
        # 진행 보상 (상태 변화가 있을 때)
        if prev_state != next_state:
            reward += 0.1
            
        # 스텝 페널티 (효율성 장려)
        reward -= 0.01 * self._episode_step
        
        # 노이즈 추가 (확률적 환경)
        if self.difficulty != "deterministic":
            noise = np.random.normal(0, 0.05)  # 작은 노이즈
            reward += noise
            
        return max(reward, -1.0)  # 최소 보상 제한

def create_challenging_transitions(base_trans: Dict, difficulty: str = "normal") -> Dict:
    """더 도전적인 transition 생성"""
    enhanced_trans = {}
    
    for (state, action), (next_state, reward) in base_trans.items():
        # 기본 transition 추가
        enhanced_trans[(state, action)] = (next_state, reward)
        
        # 확률적 실패 추가 (difficulty에 따라)
        if difficulty == "hard" and reward > 0:
            # 성공 액션도 때때로 실패
            failure_prob = 0.2
            if random.random() < failure_prob:
                enhanced_trans[(state, action)] = (state, 0.0)  # 제자리
        
        # 추가 랜덤 transition (exploration 장려)
        if difficulty != "deterministic" and len(state) > 0:
            # 일부 상태에서 랜덤 transition 추가
            if random.random() < 0.1:  # 10% 확률
                random_state = tuple(random.sample(state, max(1, len(state)//2)))
                enhanced_trans[(state, action)] = (random_state, reward * 0.5)
    
    return enhanced_trans

def build_improved_env(proc_json: str, ann_type: str, difficulty: str = "normal") -> ImprovedDialogueMDP:
    """개선된 환경 빌더"""
    # 기존 환경 구축 로직
    with open(proc_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "annotations" in data:
        recs = data["annotations"]
    else:
        recs = data
    
    recs = [r for r in recs if r.get("annotation_type") == ann_type]
    
    if not recs:
        raise RuntimeError(f"No records found for annotation type '{ann_type}'")
    
    # LLM 데이터 개선
    if ann_type == "llm":
        recs = enhance_llm_data(recs, difficulty)
    
    # Vocabulary 구축
    all_slots = set()
    all_actions = set()
    
    for r in recs:
        sb, sa = r.get("state_before", {}), r.get("state_after", {})
        all_slots.update(sb.keys())
        all_slots.update(sa.keys())
        all_actions.add(r.get("action", "noop"))
    
    if not all_slots:
        all_slots = {"dummy_slot"}
    if not all_actions:
        all_actions = {"noop", "inform", "update", "delete"}  # 최소 액션 세트
        
    slot_vocab = {s: i for i, s in enumerate(sorted(all_slots))}
    act_vocab = {a: i for i, a in enumerate(sorted(all_actions))}
    
    # 기본 transition 구축
    base_trans = {}
    for r in recs:
        sb, sa = r.get("state_before", {}), r.get("state_after", {})
        action = r.get("action", "noop")
        
        s_idx = state_to_idx(sb, slot_vocab)
        ns_idx = state_to_idx(sa, slot_vocab)
        act_idx = act_vocab[action]
        
        # 개선된 보상 설계
        if action == "inform" and sb != sa:
            reward = 1.0
        elif action == "update" and sb != sa:
            reward = 0.6
        elif action == "delete":
            reward = 0.3
        else:
            reward = 0.0
            
        base_trans[(s_idx, act_idx)] = (ns_idx, reward)
    
    # 도전적인 transition 생성
    enhanced_trans = create_challenging_transitions(base_trans, difficulty)
    
    print(f"Built enhanced {ann_type} environment:")
    print(f"  - Difficulty: {difficulty}")
    print(f"  - Slots: {len(slot_vocab)}")
    print(f"  - Actions: {len(act_vocab)}")
    print(f"  - Transitions: {len(enhanced_trans)}")
    
    return ImprovedDialogueMDP(enhanced_trans, slot_vocab, act_vocab, difficulty)

def enhance_llm_data(recs: List[Dict], difficulty: str) -> List[Dict]:
    """LLM 데이터 품질 개선"""
    enhanced = []
    
    # 더 다양한 슬롯 조합 생성
    slot_templates = {
        "restaurant": ["area", "pricerange", "food", "name"],
        "hotel": ["area", "pricerange", "type", "parking", "internet"],
        "taxi": ["departure", "destination", "leaveat", "arriveby"],
        "train": ["departure", "destination", "day", "leaveat"]
    }
    
    values = {
        "area": ["centre", "north", "south", "east", "west"],
        "pricerange": ["cheap", "moderate", "expensive"],
        "food": ["chinese", "indian", "european", "italian", "thai"],
        "type": ["hotel", "guesthouse"],
        "parking": ["yes", "no"],
        "internet": ["yes", "no"]
    }
    
    for i, r in enumerate(recs):
        # 원본 레코드 유지
        enhanced.append(r)
        
        # 추가 변형 생성 (difficulty에 따라)
        if difficulty == "hard":
            variations = 3
        elif difficulty == "easy":
            variations = 1
        else:
            variations = 2
            
        for v in range(variations):
            domain = random.choice(list(slot_templates.keys()))
            slots = random.sample(slot_templates[domain], 
                                k=random.randint(1, len(slot_templates[domain])//2))
            
            state = {}
            for slot in slots:
                full_slot = f"{domain}_{slot}"
                if slot in values:
                    state[full_slot] = random.choice(values[slot])
                else:
                    state[full_slot] = f"value_{random.randint(1,10)}"
            
            # 노이즈 있는 transition 생성
            if random.random() < 0.3:  # 30% 확률로 부분 실패
                incomplete_state = dict(random.sample(list(state.items()), 
                                                    max(1, len(state)//2)))
                enhanced.append({
                    "state_before": {},
                    "state_after": incomplete_state,
                    "action": "update",  # inform 대신 update
                    "annotation_type": r["annotation_type"],
                    "dialogue_id": f"{r['dialogue_id']}_var_{i}_{v}",
                    "turn_id": 0,
                    "utterance": f"partial request {i}_{v}"
                })
            else:
                enhanced.append({
                    "state_before": {},
                    "state_after": state,
                    "action": "inform",
                    "annotation_type": r["annotation_type"],
                    "dialogue_id": f"{r['dialogue_id']}_var_{i}_{v}",
                    "turn_id": 0,
                    "utterance": f"enhanced request {i}_{v}"
                })
    
    return enhanced