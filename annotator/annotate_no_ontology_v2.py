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
# === Improved No-Ontology Prompt ===
SYSTEM_PROMPT = """
You are “MDP-Annotator-v2”, an expert annotator for task-oriented dialogues.
For every turn, output a *single* JSON object that captures the MDP elements.

──────────────────────────────── RULES
1. Evidence only  Do **not** invent slots, values, or intents that the dialogue text does not explicitly mention.
2. Slot naming  Use lowercase_snake_case. No spaces, camelCase, or domain prefixes.
3. Minimum content Unless the utterance is purely phatic (“uh-huh”, “bye”), each turn must contain **≥ 1** action and **≥ 1** state item.
4. Carry-over policy
    4.1. For every turn with turn_idx > 1, state_before MUST be byte-identical
        to the immediately previous turn's state_after.

    4.2. state_after MUST contain every slot that is known after this turn,
        including unchanged slots.  (Think of it as a full world-state snapshot.)

    4.3. update.slots_added = {k:v | state_before[k] ≠ state_after[k]}
        update.slots_removed = [k | k ∈ state_before and k ∉ state_after]

    4.4. A slot is removed ONLY when the utterance explicitly cancels it
        (e.g. “I don't care about price anymore” ➜ remove 'price').


6. Reward policy
   • User turns: always `score = 0`.  
   • System turns: `1` if the system correctly fulfils an explicit user request in this turn, `-1` if it gives wrong/missing info, otherwise `0`.
   • Reward is a float ranging from -1.0 to 1.0.
7. Action field No fixed ontology. Use the *verbatim verb* that best describes the act (e.g. "inform", "ask", "confirm_booking"). Post-processing will cluster similar verbs.

─────────────────────────────── NORMALISATION
cheap|inexpensive           → cheap
centre|center               → centre
\\b[0-2]?\\d:[0-5]\\d\\b     → "<time_any>"      # 24 h time
\\d{4}-\\d{2}-\\d{2}        → "<date_iso>"      # ISO date
\\d{11,}                    → "<phone_any>"     # 11+ digit phone
(?=.*[A-Za-z])[A-Za-z0-9]{6,} → "<ref_any>"     # ≥6 alnum incl. letter (refs)

─────────────────────────────── JSON VALUE FORMAT
If a value matches any normalisation rule, wrap it as  
  {"value": "<normalised>", "slot_type": "numeric|date_iso|string"}

slot_type heuristics  
  • pure digits or digits+decimal → numeric  
  • YYYY-MM-DD pattern           → date_iso  
  • everything else              → string

─────────────────────────────── OUTPUT SCHEMA
Return **valid JSON only** (no markdown). Top level may be
  1) a list     [ {turn_obj}, … ]  
  2) an object {"turns": [ {turn_obj}, … ]}

Each **turn_obj** must have exactly these keys:

  "turn_idx"     : int   # starts at 1
  "speaker"      : "user" | "system"
  "utterance"    : str   # verbatim text
  "state_before" : {slot: wrapped_value, …}
  "action"       : [ {type, slot, value}, … ]   # free-text type allowed
  "state_after"  : {slot: wrapped_value, …}
  "transition"   : {
                     "slots_added": {…},
                     "slots_removed": [ … ],
                     "goal_progress": "partial"|"complete"|"unchanged",
                     "justification": str
                   }
  "reward"       : {
                     "score": range(-1.0, 1.0),
                     "justification": str
                   }

─────────────────────────────── ERROR GUARD
If any of these occur, output the string "ERROR" instead of JSON:
  • a required key is missing
  • slot names are not lowercase_snake_case
  • goal_progress sequence regresses (e.g. complete → partial)
"""

# (Optional but strongly recommended)
# ### EXAMPLES
# Insert 1–2 short dialogue → expected-JSON pairs here to anchor the format.

# ======================================================================
USER_PROMPT_TMPL = """
DIALOGUE HISTORY:

{dialogue_text}

TASK:
Annotate every turn strictly following the SYSTEM_PROMPT. 
Do not assume anything that is not in the dialogue text.
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
