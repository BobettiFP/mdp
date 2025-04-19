import json, gzip, hashlib, sys
from pathlib import Path
from collections import defaultdict
from jsonschema import validate, Draft202012Validator, ValidationError

# --------------- 1.1  결과 스트리밍 저장 -----------------
def stream_save(turn_json: dict, fout):
    """
    주석된 single-turn JSON을 한 줄로 gzip 파일에 기록
    (= line‑delimited JSON L‑JSON) 
    """
    line = json.dumps(turn_json, ensure_ascii=False)
    fout.write((line + "\n").encode("utf-8"))

# --------------- 1.2  JSONSchema 로 구조 검증 -----------------
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
            "type": "object",
            "required": ["prev_state_id", "action", "next_state_id"]
        }
    }
}
validator = Draft202012Validator(TURN_SCHEMA)

# --------------- 1.3  SHA‑256 충돌(역매핑) 체크 --------------
hash_map = defaultdict(set)   # state_id -> {state_repr}

def check_hash_collision(turn_json: dict, dialogue_id: str):
    for key in ("state_id_prev", "state_id_next"):
        sid = turn_json[key]
        srepr = (
            turn_json["state_repr_prev"]
            if key == "state_id_prev" else
            turn_json["state_repr_next"]
        )
        hash_map[sid].add(srepr)
        if len(hash_map[sid]) > 1:
            print(f"[WARNING] Hash collision in {dialogue_id} for {sid}: "
                  f"{hash_map[sid]}", file=sys.stderr)

# --------------- 메인: 전체 대화 JSON 처리 -------------------
def process_dialogue(dialogue_idx: int, dialogue_obj: dict, fout):
    """
    주석기 결과가 대화 단위 JSON( turn_1, turn_2 … )인 경우
    -> 턴별로 분해해서 검증·저장
    """
    for tkey, turn in dialogue_obj.items():
        try:
            validate(turn, TURN_SCHEMA)
        except ValidationError as ve:
            print(f"[SCHEMA ERR] dlg#{dialogue_idx} {tkey}: {ve.message}",
                  file=sys.stderr)
            continue
        check_hash_collision(turn, f"dlg#{dialogue_idx} {tkey}")
        stream_save(turn, fout)

# ---------------------------- 실행 예시 -------------------------
if __name__ == "__main__":
    """
    usage:
        python validate_and_store.py results_first30.json
               --out results_first30.ldjson.gz
    """
    import argparse, gzip
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="annotated dialogue JSON file")
    parser.add_argument("--out", default="annotated_turns.ldjson.gz")
    args = parser.parse_args()

    data = json.load(open(args.input_json, encoding="utf-8"))
    total_turns, total_err = 0, 0

    with gzip.open(args.out, "wb") as fout:
        for dlg_idx, dlg in data.items():
            process_dialogue(dlg_idx, dlg, fout)

    print("Finished. Stored to", args.out)
