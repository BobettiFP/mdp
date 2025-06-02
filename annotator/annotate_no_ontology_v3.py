#!/usr/bin/env python3
# annotate_convlab3_standard.py
# -----------------------------------------------
# Task-oriented dialogue → ConvLab-3 standard format annotation
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
You are "ConvLab3-Standard-Annotator", creating standard ConvLab-3 unified format for RL comparison research.

──────────────────────────────── CORE TASK
Extract dialogue acts, maintain belief state, infer API calls from task-oriented dialogue turns. Use ONLY information explicitly mentioned in utterance text.

──────────────────────────────── NAMING CONVENTIONS
• Domains: restaurant, hotel, train, taxi, attraction, hospital, police, general, shopping, spa, etc.
• Slots: lowercase_snake_case (name, phone, area, book_time, food_type, etc.)
• Intents: inform, request, confirm, recommend, book, select, negate, affirm, thank, bye, greet
• Values: normalize using patterns below, otherwise keep original

──────────────────────────────── VALUE NORMALIZATION
Apply these transformations:
cheap|inexpensive → cheap
centre|center → centre
\\b[0-2]?\\d:[0-5]\\d\\b → "<time_any>"
\\d{4}-\\d{2}-\\d{2} → "<date_iso>"
\\d{11,} → "<phone_any>"
(?=.*[A-Za-z])[A-Za-z0-9]{6,} → "<ref_any>"

Wrap normalized values: {"value": "<pattern>", "slot_type": "time_any|date_iso|phone_any|ref_any|string|numeric"}

──────────────────────────────── STATE MANAGEMENT
• state: Cumulative belief state (all information mentioned so far)
• state_update: Only slots changed in this turn
• Formula: previous_state + state_update = current_state
• services: List of domains mentioned anywhere in dialogue

──────────────────────────────── API INFERENCE
Add api_call when system provides specific entities or search results:
• Trigger phrases: "I found...", "There are X...", "I recommend...", booking confirmations
• Method format: "{domain}.{action}" (restaurant.search, hotel.book, train.inform)
• Parameters: User constraints that triggered the query
• Results: Specific entities with their attributes

──────────────────────────────── JSON OUTPUT FORMAT
Return your response as valid JSON only. The output must be a single JSON object with this exact structure:

{
  "dialogue_id": "string",
  "services": ["domain1", "domain2"],
  "turns": [
    {
      "turn_id": 0,
      "speaker": "USER"|"SYSTEM",
      "utterance": "exact text",
      "dialogue_acts": [
        {"intent": "str", "domain": "str", "slot": "str", "value": "str|obj"}
      ],
      "state": {"domain": {"slot": "value"}},
      "state_update": {"domain": {"slot": "value"}},
      "api_call": {"method": "domain.action", "parameters": {"slot": "value"}},
      "api_call_result": [{"slot": "value"}]
    }
  ]
}

──────────────────────────────── REQUIREMENTS
• turn_id starts at 0, increments sequentially
• Include api_call/api_call_result only when system provides specific information
• Every turn needs ≥1 dialogue_act unless purely phatic ("ok", "bye")
• State consistency: previous_state + state_update = current_state

──────────────────────────────── VALIDATION
Output JSON object {"error": "description"} if:
• Required keys missing
• turn_id sequence broken  
• State math incorrect (previous + update ≠ current)

Always respond with properly formatted JSON - no other text or formatting.
"""

USER_PROMPT_TMPL = """
DIALOGUE HISTORY:

{dialogue_text}

TASK:
Annotate every turn strictly following the ConvLab-3 standard format. 
Extract dialogue acts, maintain cumulative belief state, and infer API calls when the system provides specific information.
Do not assume anything that is not explicitly mentioned in the dialogue text.
"""

# ---------- 2. 유틸 함수 ----------
def hash_state(state: Dict[str, Any]) -> str:
    """JSON 직렬화를 통해 key 순서와 무관하게 state 해시 생성."""
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:12]

def join_utterances(dialogue: Dict[str, Any]) -> str:
    """turns 배열 → 'speaker: utterance' 줄바꿈 블록."""
    return "\n".join(f"{t['speaker']}: {t['utterance']}" for t in dialogue["turns"])

def validate_convlab3_format(data: Dict[str, Any]) -> bool:
    """ConvLab-3 표준 형식 검증"""
    required_keys = ["dialogue_id", "services", "turns"]
    if not all(key in data for key in required_keys):
        return False
    
    if not isinstance(data["services"], list):
        return False
        
    for turn in data["turns"]:
        turn_required = ["turn_id", "speaker", "utterance", "dialogue_acts", "state", "state_update"]
        if not all(key in turn for key in turn_required):
            return False
            
        if turn["speaker"] not in ["USER", "SYSTEM"]:
            return False
            
        if not isinstance(turn["dialogue_acts"], list):
            return False
            
    return True

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

        # 에러 응답 체크
        if "error" in raw_json:
            return {
                "dialogue_id": dialogue.get("dialogue_id", f"idx_{idx}"),
                "error": raw_json["error"]
            }

        # ConvLab-3 형식 검증
        if not validate_convlab3_format(raw_json):
            return {
                "dialogue_id": dialogue.get("dialogue_id", f"idx_{idx}"),
                "error": "Invalid ConvLab-3 format"
            }

        # dialogue_id 설정
        if "dialogue_id" not in raw_json:
            raw_json["dialogue_id"] = dialogue.get("dialogue_id", f"idx_{idx}")

        # turn_id 순서 검증 및 보정
        for i, turn in enumerate(raw_json["turns"]):
            if turn.get("turn_id") != i:
                turn["turn_id"] = i

        return raw_json

    except json.JSONDecodeError as e:
        return {
            "dialogue_id": dialogue.get("dialogue_id", f"idx_{idx}"),
            "error": f"JSON decode error: {str(e)}"
        }
    except Exception as e:
        return {
            "dialogue_id": dialogue.get("dialogue_id", f"idx_{idx}"),
            "error": f"Processing error: {str(e)}"
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

# ---------- 5. 결과 분석 ----------
def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """결과 분석 및 통계"""
    total = len(results)
    errors = sum(1 for r in results if "error" in r)
    success = total - errors
    
    # 성공한 결과에서 통계 수집
    services_count = {}
    avg_turns = 0
    total_turns = 0
    
    for r in results:
        if "error" not in r and "turns" in r:
            total_turns += len(r["turns"])
            for service in r.get("services", []):
                services_count[service] = services_count.get(service, 0) + 1
    
    if success > 0:
        avg_turns = total_turns / success
    
    return {
        "total_dialogues": total,
        "successful": success,
        "errors": errors,
        "success_rate": success / total if total > 0 else 0,
        "average_turns": avg_turns,
        "services_distribution": services_count
    }

# ---------- 6. CLI ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ConvLab-3 standard format annotator.")
    parser.add_argument("--input", required=True, help="Input dialogue JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--threads", type=int, default=20, help="Thread pool size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dialogues to process")
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    args = parser.parse_args()

    # 입력 파일 로드
    with open(args.input, "r", encoding="utf-8") as f:
        dialogues = json.load(f)
    
    # MultiWOZ dict → list 변환
    if isinstance(dialogues, dict):
        dialogues = list(dialogues.values())
    
    # 제한 적용
    if args.limit:
        dialogues = dialogues[:args.limit]

    print(f"[INFO] Loaded {len(dialogues)} dialogues")

    # 어노테이션 실행
    annotated = annotate_batch(dialogues, max_workers=args.threads)

    # 결과 저장
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved annotated dialogues to {args.output}")

    # 통계 출력
    if args.stats:
        stats = analyze_results(annotated)
        print("\n[STATISTICS]")
        print(f"Total dialogues: {stats['total_dialogues']}")
        print(f"Successful: {stats['successful']}")
        print(f"Errors: {stats['errors']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Average turns per dialogue: {stats['average_turns']:.1f}")
        print(f"Services found: {list(stats['services_distribution'].keys())}")