# ------------------------------  env.py  -------------------------------------
"""
DialogueMDP Gym-compatible env built from processed_annotations.json
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
        if 0 <= i < dim:  # 범위 체크
            v[i] = 1
    return v

def state_to_idx(sd: Dict[str,str], vocab: Dict[str,int]) -> Tuple[int,...]:
    """Convert state dict to sorted tuple of vocab indices."""
    if not sd:  # 빈 상태 처리
        return tuple()
    return tuple(sorted(vocab[s] for s in sd if s in vocab))  # vocab에 있는 것만

class DialogueMDP(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, trans: Dict, slot_vocab: Dict[str,int], act_vocab: Dict[str,int]):
        super().__init__()
        self.trans = trans
        self.slot_vocab = slot_vocab
        self.action_vocab = act_vocab
        
        # 공간 정의
        self.observation_space = spaces.MultiBinary(len(slot_vocab))
        self.action_space = spaces.Discrete(len(act_vocab))
        
        # 가능한 상태들 추출
        self._states = list({s for (s, _) in trans.keys()})
        if not self._states:  # 빈 상태 리스트 체크
            self._states = [tuple()]  # 최소한 빈 상태 하나는 있어야 함
            
        self._current_state = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
            random.seed(seed)
        
        self._current_state = random.choice(self._states)
        obs = _vec(self._current_state, len(self.slot_vocab))
        return obs, {}

    def step(self, action: int):
        """Take action and return next state, reward, done, truncated, info."""
        key = (self._current_state, int(action))
        
        if key not in self.trans:
            # 유효하지 않은 액션인 경우
            obs = _vec(self._current_state, len(self.slot_vocab))
            return obs, -0.1, True, False, {"invalid_action": True}
        
        next_state, reward = self.trans[key]
        self._current_state = next_state
        obs = _vec(self._current_state, len(self.slot_vocab))
        
        # 성공적인 inform 액션으로 에피소드 종료
        done = reward > 0
        
        return obs, reward, done, False, {}

# ---------------------------------------------------------------------------
def build_env(proc_json: str, ann_type: str) -> DialogueMDP:
    """Build environment from processed annotations."""
    try:
        with open(proc_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load {proc_json}: {e}")
    
    # 데이터 구조 확인
    if isinstance(data, dict) and "annotations" in data:
        recs = data["annotations"]
    elif isinstance(data, list):
        recs = data
    else:
        raise RuntimeError(f"Invalid data format in {proc_json}")
    
    # 해당 타입의 레코드만 필터링
    recs = [r for r in recs if r.get("annotation_type") == ann_type]
    
    if not recs:
        raise RuntimeError(f"No records found for annotation type '{ann_type}'")
    
    # LLM 데이터의 경우 utterance에서 슬롯 추출 시도
    if ann_type == "llm":
        enhanced_recs = []
        for r in recs:
            utterance = r.get("utterance", "").lower()
            
            # 간단한 슬롯 추출 로직 (실제로는 더 정교한 NER이 필요)
            extracted_slots = {}
            
            # 가격 관련
            if "expensive" in utterance:
                extracted_slots["restaurant_pricerange"] = "expensive"
            elif "cheap" in utterance:
                extracted_slots["restaurant_pricerange"] = "cheap"
            elif "moderate" in utterance:
                extracted_slots["restaurant_pricerange"] = "moderate"
            
            # 지역 관련
            if "centre" in utterance or "center" in utterance:
                extracted_slots["restaurant_area"] = "centre"
            elif "north" in utterance:
                extracted_slots["restaurant_area"] = "north"
            elif "south" in utterance:
                extracted_slots["restaurant_area"] = "south"
            elif "east" in utterance:
                extracted_slots["restaurant_area"] = "east"
            elif "west" in utterance:
                extracted_slots["restaurant_area"] = "west"
            
            # 음식 타입
            if "chinese" in utterance:
                extracted_slots["restaurant_food"] = "chinese"
            elif "european" in utterance:
                extracted_slots["restaurant_food"] = "european"
            elif "indian" in utterance:
                extracted_slots["restaurant_food"] = "indian"
            
            # 호텔 관련
            if "hotel" in utterance:
                extracted_slots["hotel_type"] = "hotel"
            if "parking" in utterance:
                extracted_slots["hotel_parking"] = "yes"
            if "internet" in utterance:
                extracted_slots["hotel_internet"] = "yes"
            
            # 추출된 슬롯이 있으면 상태 업데이트
            if extracted_slots:
                new_record = dict(r)
                new_record["state_after"] = extracted_slots
                new_record["action"] = "inform"
                enhanced_recs.append(new_record)
            else:
                enhanced_recs.append(r)
        
        recs = enhanced_recs
        print(f"Enhanced {len([r for r in recs if r.get('state_after')])} LLM records with extracted slots")
    
    # 의미있는 레코드만 유지
    meaningful_recs = []
    for r in recs:
        sb, sa = r.get("state_before", {}), r.get("state_after", {})
        action = r.get("action", "noop")
        
        # 상태 변화가 있거나, inform 액션이거나, 빈 상태가 아닌 경우
        if sb != sa or action == "inform" or sb or sa:
            meaningful_recs.append(r)
    
    if not meaningful_recs:
        # 의미있는 레코드가 없으면 원본 레코드 중 일부라도 사용
        print(f"Warning: No meaningful records found for {ann_type}, creating synthetic data")
        
        # LLM을 위한 합성 데이터 생성
        if ann_type == "llm":
            synthetic_recs = []
            base_slots = ["restaurant_pricerange", "restaurant_area", "restaurant_food"]
            values = {
                "restaurant_pricerange": ["cheap", "moderate", "expensive"],
                "restaurant_area": ["centre", "north", "south", "east", "west"],
                "restaurant_food": ["chinese", "indian", "european"]
            }
            
            for i in range(20):  # 20개의 합성 레코드 생성
                selected_slots = random.sample(base_slots, k=random.randint(1, 2))
                state = {slot: random.choice(values[slot]) for slot in selected_slots}
                
                synthetic_recs.append({
                    "state_before": {},
                    "state_after": state,
                    "action": "inform",
                    "annotation_type": ann_type,
                    "dialogue_id": f"synthetic_{i}",
                    "turn_id": 0,
                    "utterance": f"synthetic utterance {i}"
                })
            
            meaningful_recs = synthetic_recs
            print(f"Created {len(meaningful_recs)} synthetic records for {ann_type}")
        else:
            meaningful_recs = recs[:10]  # 최대 10개만 사용
    
    recs = meaningful_recs
    
    # Vocabulary 구축
    all_slots = set()
    all_actions = set()
    
    for r in recs:
        sb, sa = r.get("state_before", {}), r.get("state_after", {})
        all_slots.update(sb.keys())
        all_slots.update(sa.keys())
        all_actions.add(r.get("action", "noop"))
    
    # 최소한의 vocabulary 보장
    if not all_slots:
        all_slots = {"dummy_slot"}  # 더미 슬롯 추가
    if not all_actions:
        all_actions = {"noop"}  # 기본 액션 추가
        
    slot_vocab = {s: i for i, s in enumerate(sorted(all_slots))}
    act_vocab = {a: i for i, a in enumerate(sorted(all_actions))}
    
    print(f"Built vocab: {len(slot_vocab)} slots, {len(act_vocab)} actions")
    print(f"Actions: {list(act_vocab.keys())}")
    
    # Transition 구축
    trans = {}
    reward_count = 0
    
    for r in recs:
        sb, sa = r.get("state_before", {}), r.get("state_after", {})
        action = r.get("action", "noop")
        
        s_idx = state_to_idx(sb, slot_vocab)
        ns_idx = state_to_idx(sa, slot_vocab)
        act_idx = act_vocab[action]
        
        # 보상 설계 개선
        if action == "inform" and sb != sa:
            reward = 1.0
            reward_count += 1
        elif action == "update" and sb != sa:
            reward = 0.5
        elif action == "delete":
            reward = 0.1
        else:
            reward = 0.0
            
        trans[(s_idx, act_idx)] = (ns_idx, reward)
    
    print(f"Built {len(trans)} transitions, {reward_count} positive rewards")
    
    if not trans:
        # 최소한의 transition 생성
        empty_state = tuple()
        noop_action = act_vocab.get("noop", 0)
        trans[(empty_state, noop_action)] = (empty_state, 0.0)
        print("Warning: Created minimal transition set")
    
    return DialogueMDP(trans, slot_vocab, act_vocab)