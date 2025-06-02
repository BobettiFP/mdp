import gym, json, random
import numpy as np
from state_encoder import encode_state, slot_vocab

class DialogueEnv(gym.Env):
    def __init__(self, dataset_path):
        self.dials = json.load(open(dataset_path))
        self.action_space = gym.spaces.Discrete(6)       # inform, request, ask, confirm, cancel, none
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(len(slot_vocab)*2,), dtype=np.float32)

    def reset(self):
        self.dlg = random.choice(self.dials)
        self.turns = list(self.dlg["turns"])
        self.ptr = 0
        return encode_state(self.turns[0]["state_before"])

    def step(self, act_idx):
        self.ptr += 1
        done = self.ptr >= len(self.turns)
        reward = self.turns[self.ptr-1]["reward"]["score"] if not done else 0
        obs = encode_state(self.turns[self.ptr-1]["state_after"]) if not done else np.zeros_like(self.observation_space.low)
        return obs, reward, done, {}
