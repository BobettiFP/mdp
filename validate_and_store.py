import json, gzip, sys, re, hashlib
from collections import defaultdict
from jsonschema import validate, Draft202012Validator, ValidationError

# ---------------- 1.1  스트리밍 저장 -----------------
def stream_save(turn_json: dict, fout):
    fout.write((json.dumps(turn_json, ensure_ascii=False) + "\n").encode("utf-8"))

# ---------------- 1.2  JSON‑Schema ------------------
TURN_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "turn_idx", "utterances",
        "state_repr_prev", "state_id_prev",
        "system_action",
        "state_repr_next", "state_id_next",
        "transition"
    ],
    "properties": {
        "turn_idx": {"type": "integer", "minimum": 1},
        "utterances": {
            "type": "object",
            "required": ["user", "system"],
            "properties": {
                "user": {"type": "string"},
                "system": {"type": "string"}
            }
        },
        "state_repr_prev": {"type": "string"},
        "state_id_prev":   {"type": "string", "pattern": "^[a-f0-9]{64}$"},
        "system_action":   {"type": "string"},
        "state_repr_next": {"type": "string"},
        "state_id_next":   {"type": "string", "pattern": "^[a-f0-9]{64}$"},
        "transition": {
            "oneOf": [
                {   # 객체 1개
                    "type": "object",
                    "required": ["prev_state_id", "action", "next_state_id"]
                },
                {   # 객체 배열
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["prev_state_id", "action", "next_state_id"]
                    }
                }
            ]
        }
    }
}
validator = Draft202012Validator(TURN_SCHEMA)

# ---------------- 1.3  해시 충돌 검사 -------------
hash_map = defaultdict(set)
def check_hash_collision(turn_json, dlg_tag):
    for sid_key, srepr_key in [("state_id_prev", "state_repr_prev"),
                               ("state_id_next", "state_repr_next")]:
        sid = turn_json[sid_key]
        srepr = turn_json[srepr_key]
        hash_map[sid].add(srepr)
        if len(hash_map[sid]) > 1:
            print(f"[WARNING] Hash collision {sid} : {hash_map[sid]}", file=sys.stderr)

# ---------------- 대화별 처리 ----------------------
def process_dialogue(idx: int, dlg_obj: dict, fout):
    # 오류 대화 건너뛰기
    if "error" in dlg_obj:
        print(f"[SKIP] dlg#{idx} has error: {dlg_obj['error']}", file=sys.stderr)
        return

    for tkey, turn in dlg_obj.items():
        # ** error 턴 건너뜀 **
        if not isinstance(turn, dict):
            print(f"[SKIP] dlg#{idx} {tkey} is not dict", file=sys.stderr)
            continue

        # transition 필수 보정
        if "transition" not in turn or not turn["transition"]:
            turn["transition"] = {}

        try:
            validate(turn, TURN_SCHEMA)
        except ValidationError as ve:
            print(f"[SCHEMA ERR] dlg#{idx} {tkey}: {ve.message}", file=sys.stderr)
            continue

        check_hash_collision(turn, f"dlg#{idx} {tkey}")
        stream_save(turn, fout)

# ---------------- 실행 -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("--out", default="annotated_turns60.ldjson.gz")
    args = parser.parse_args()

    with open(args.input_json, encoding="utf-8") as fp:
        data = json.load(fp)

    with gzip.open(args.out, "wb") as fout:
        for dlg_idx, dlg in data.items():
            process_dialogue(dlg_idx, dlg, fout)

    print("Finished. Stored to", args.out)
