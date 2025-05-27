import json, os, hashlib, concurrent.futures as cf
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

# ----------------- 0. 환경 설정 -----------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment.")
client = OpenAI(api_key=api_key)

# ----------------- 시스템 프롬프트 (온톨로지 없이 작동) -----------------
SYSTEM_PROMPT = """
You are “MDP‑Annotator‑v2”, an expert annotator for task-oriented dialogues.
For each turn in the dialogue, extract MDP elements in separate fields.

STRICT RULES:
- DO NOT infer or assume anything that is not explicitly present in the dialogue text.
- Use only observable, textual evidence for all elements.

For each turn, output a JSON object with the following fields:
  "turn_idx": integer (starts at 1)
  "speaker": "user" or "system"
  "utterance": original text of the turn
  "state_before": dictionary of known slot-values before this turn
  "action": list of intent-slot-value triplets (e.g., [{"type": "inform", "slot": "area", "value": "west"}])
  "state_after": dictionary of updated slot-values after applying action
  "transition": {
    "slots_added": {...},
    "slots_removed": [...],
    "goal_progress": "partial" or "complete" or "unchanged",
    "is_valid_transition": true or false,
    "justification": textual explanation
  },
  "reward": {
    "score": -1 | 0 | 1,
    "justification": short reason
  }

Ensure the JSON output is valid. Output only one top-level JSON object.
"""

USER_PROMPT_TMPL = """
DIALOGUE HISTORY:

{dialogue_text}

TASK:
Annotate each turn according to the definitions and format in the system prompt.
Do not assume anything not written in the dialogue.
"""

# ----------------- 유틸 함수 -----------------

def hash_state(state: Dict[str, str]) -> str:
    """JSON 직렬화를 통해 key 순서에 관계없이 hash 생성"""
    state_json = json.dumps(state, sort_keys=True)
    return hashlib.sha256(state_json.encode()).hexdigest()

def join_utterances(dialogue: Dict[str, Any]) -> str:
    return "\n".join(f"{t['speaker']}: {t['utterance']}" for t in dialogue["turns"])

# ----------------- LLM 호출 및 결과 가공 -----------------

def annotate_dialogue(idx: int, dialogue: Dict[str, Any]) -> Dict[str, Any]:
    text_block = join_utterances(dialogue)
    user_prompt = USER_PROMPT_TMPL.format(dialogue_text=text_block)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )
        dlg_json = json.loads(resp.choices[0].message.content)

        # 리스트 형태 응답이면 dict로 변환
        if isinstance(dlg_json, list):
            dlg_json = {f"turn_{i+1}": turn for i, turn in enumerate(dlg_json)}

        for turn_id, turn in dlg_json.items():
            if not isinstance(turn.get("state_before"), dict):
                turn["state_before"] = {}
            if not isinstance(turn.get("state_after"), dict):
                turn["state_after"] = {}

            turn["state_id_prev"] = hash_state(turn["state_before"])
            turn["state_id_next"] = hash_state(turn["state_after"])

        return {
            "dialogue_id": dialogue["dialogue_id"],
            "annotation": dlg_json
        }

    except Exception as e:
        return {
            "dialogue_id": dialogue["dialogue_id"],
            "error": str(e)
        }

# ----------------- 배치 실행 -----------------

def annotate_batch(dialogues: List[Dict[str, Any]], max_workers: int = 4) -> List[Dict[str, Any]]:
    results = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(annotate_dialogue, i, dlg) for i, dlg in enumerate(dialogues)]
        for f in tqdm(cf.as_completed(futures), total=len(futures), desc="Annotating"):
            try:
                result = f.result()
                results.append(result)
            except Exception as e:
                print(f"[ERROR] Future failed: {e}")
    return results


# ----------------- 실행부 -----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input dialogue JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    print(f"[INFO] Loaded {len(dialogues)} dialogues")

    annotated = annotate_batch(dialogues)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved annotated dialogues to {args.output}")
