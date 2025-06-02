import json, numpy as np

slot_vocab = json.load(open("slot_vocab.json"))
token_vocab = {tok: i for i, tok in enumerate(json.load(open("token_vocab.json")))}

def encode_state(state):
    vec = []
    for slot in slot_vocab:
        if slot not in state:
            vec.extend([0.0, 0.0])
            continue
        vobj = state[slot]
        vtype, val = vobj["slot_type"], vobj["value"]
        # 존재 마스크 1
        if vtype == "numeric":
            vec.extend([1.0, np.tanh(float(val))])  # scale to -1~1
        elif vtype == "time_min":
            vec.extend([1.0, float(val)/1440])      # 0~1
        elif vtype == "weekday":
            vec.extend([1.0, val/6.0])              # 0~1
        elif vtype == "list_string":
            tok = val[0] if val else "<unk>"
            vec.extend([1.0, token_vocab.get(tok, 0)/len(token_vocab)])
        else:  # string/date_iso etc.
            vec.extend([1.0, token_vocab.get(val, 0)/len(token_vocab)])
    return np.array(vec, dtype=np.float32)
