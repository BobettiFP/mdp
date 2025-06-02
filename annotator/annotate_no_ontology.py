#!/usr/bin/env python3
# annotate_no_ontology.py
# -----------------------------------------------
# Task-oriented dialogue → MDP element annotation
# (ontology-free version, gpt-4o-mini backend)
# -----------------------------------------------

import json, os, hashlib, concurrent.futures as cf
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

# ---------- 0. 환경 설정 ----------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment.")
client = OpenAI(api_key=api_key)

# ---------- 1. 시스템 프롬프트 ----------
SYSTEM_PROMPT = """
You are “MDP-Annotator-v2”, an expert annotator for task-oriented dialogues.
For each turn in the dialogue, extract MDP elements in separate fields.

STRICT RULES:
- DO NOT infer or assume anything that is not explicitly present in the dialogue text.
- Use only observable, textual evidence for all elements.
- ALL slot names must be lowercase snake_case (spaces→_ , camelCase→snake_case).
- You must extract at least one state and action unless the utterance is totally meaningless.

Normalize values:
  cheap|inexpensive→cheap
  centre|center→centre
  \d{11,} → "<phone_any>"      # 11+ digits = phone
  (?=.*[A-Za-z])[A-Za-z0-9]{6,} → "<ref_any>" # 6+ alnum incl. letter = ref

If a value matches a pattern, wrap it as:
  {"value": "<normalized>", "slot_type": "numeric|date_iso|string"}

state_before/state_after must be dictionaries whose values follow the above object format.

If a value matches a pattern, assign a slot_type field:
  - 70 %+ digits → "numeric"
  - yyyy-mm-dd → "date_iso"
Return: { "slot": "price", "value": "65", "slot_type": "numeric" }

Return valid JSON only (no markdown). Acceptable top-level formats:
  1) a list of turn objects, or
  2) {"turns": [...]}

Each turn object must contain exactly:
  "turn_idx"       : integer (starts at 1)
  "speaker"        : "user" or "system"
  "utterance"      : string (verbatim)
  "state_before"   : dict of known slot-values
  "action"         : list of {type, slot, value}
  "state_after"    : dict of updated slot-values
  "transition"     : {
                       "slots_added": {...},
                       "slots_removed": [...],
                       "goal_progress": "partial"|"complete"|"unchanged",
                       "is_valid_transition": true|false,
                       "justification": string
                     }
  "reward"         : {
                       "score": -1|0|1,
                       "justification": string
                     }
"""

USER_PROMPT_TMPL = """
DIALOGUE HISTORY:

{dialogue_text}

TASK:
Annotate each turn according to the definitions and format in the system prompt.
Do not assume anything not written in the dialogue.
"""

# ---------- 2. 유틸 함수 ----------
def hash_state(state: Dict[str, str]) -> str:
    """JSON 직렬화를 통해 key 순서와 무관하게 state 해시 생성."""
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()

def join_utterances(dialogue: Dict[str, Any]) -> str:
    """turns 배열 → 'speaker: utterance' 줄바꿈 블록."""
    return "\n".join(f"{t['speaker']}: {t['utterance']}" for t in dialogue["turns"])

def normalize_turns(raw_json: Any) -> Dict[str, Dict[str, Any]]:
    """
    LLM 응답을 { "turn_1": {...}, ... } 형태로 정규화.

    허용 구조:
      1) [ {...}, {...} ]                       → list
      2) { "turns": [ {...}, ... ] }           → dict + key "turns"
      3) { "turn_1": {...}, ... }              → 이미 dict-of-turns
    """
    # 1) 바로 리스트
    if isinstance(raw_json, list):
        return {f"turn_{i+1}": t for i, t in enumerate(raw_json)}

    # 2) {"turns": [...]}
    if isinstance(raw_json, dict) and "turns" in raw_json and isinstance(raw_json["turns"], list):
        return {f"turn_{i+1}": t for i, t in enumerate(raw_json["turns"])}

    # 3) dict-of-turns이면 그대로
    if isinstance(raw_json, dict):
        return raw_json

    raise ValueError("Un-parsable JSON structure from LLM")

# ---------- 3. LLM 호출 및 결과 가공 ----------
def annotate_dialogue(idx: int, dialogue: Dict[str, Any]) -> Dict[str, Any]:
    text_block = join_utterances(dialogue)
    user_prompt = USER_PROMPT_TMPL.format(dialogue_text=text_block)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            top_p=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )
        raw_json = json.loads(resp.choices[0].message.content)

        # *** 핵심: 어떤 형태든 dict-of-turns로 변환 ***
        turns_dict = normalize_turns(raw_json)

        # state 해시 붙이기 + 기본 타입 보정
        for turn in turns_dict.values():
            if not isinstance(turn.get("state_before"), dict):
                turn["state_before"] = {}
            if not isinstance(turn.get("state_after"), dict):
                turn["state_after"] = {}

            turn["state_id_prev"] = hash_state(turn["state_before"])
            turn["state_id_next"] = hash_state(turn["state_after"])

        return {
            "dialogue_id": dialogue["dialogue_id"],
            "annotation": turns_dict
        }

    except Exception as e:
        return {
            "dialogue_id": dialogue.get("dialogue_id", f"idx_{idx}"),
            "error": str(e)
        }

# ---------- 4. 병렬 실행 ----------
def annotate_batch(dialogues: List[Dict[str, Any]], max_workers: int = 20) -> List[Dict[str, Any]]:
    results = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(annotate_dialogue, i, dlg) for i, dlg in enumerate(dialogues)]
        for f in tqdm(cf.as_completed(futures), total=len(futures), desc="Annotating"):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"error": f"Future failed: {e}"})
    return results

# ---------- 5. CLI ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ontology-free MDP annotator.")
    parser.add_argument("--input", required=True, help="Input dialogue JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--threads", type=int, default=20, help="Thread pool size")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        dialogues = json.load(f)
    if isinstance(dialogues, dict):             # MultiWOZ dict → list 변환
        dialogues = list(dialogues.values())

    print(f"[INFO] Loaded {len(dialogues)} dialogues")

    annotated = annotate_batch(dialogues, max_workers=args.threads)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved annotated dialogues to {args.output}")
