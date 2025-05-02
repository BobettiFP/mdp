"""
annotate_parallel.py
GPT‑4o로 MultiWOZ 대화를 주석 → state_repr 정규화 → SHA‑256 해시 부여
멀티스레드 병렬 실행 후 results_first20.json 저장
"""

import json, os, re, hashlib, concurrent.futures as cf
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

# ----------------- 1. 온톨로지 · 정규화 규칙 -----------------
SLOT_ORDER = [
    "area", "price", "food", "people",
    "day", "time", "booking_ref"
]

SYNONYM_MAP = {
    "centre": "centre", "center": "centre",
    "cheap": "cheap", "inexpensive": "cheap",
    "north": "north", "south": "south",
    "expensive": "expensive", "moderate": "moderate"
}

OPEN_VAL_MAP = {
    r"^\+?\d{5,}$":      "phone_any",   # 전화번호
    r"^[A-Z0-9]{6,}$":   "ref_any"      # 예약코드
}

def canonicalize(raw: str) -> str:
    """slot=value|… 문자열을 완전·정렬 형태로 변환"""
    slot_dict = {s: "none" for s in SLOT_ORDER}

    for pair in raw.split("|"):
        if "=" not in pair:
            continue
        k, v = [x.strip().lower() for x in pair.split("=", 1)]
        if k not in slot_dict:
            continue
        v = SYNONYM_MAP.get(v, v) or "none"
        for pat, token in OPEN_VAL_MAP.items():
            if re.match(pat, v):
                v = token
                break
        slot_dict[k] = v

    return "|".join(f"{k}={slot_dict[k]}" for k in SLOT_ORDER)

def hash_state(canonical: str) -> str:
    return hashlib.sha256(canonical.encode()).hexdigest()

# ----------------- 2. GPT 시스템 프롬프트 -----------------
SYSTEM_PROMPT = """
You are “MDP‑Annotator‑v1”, a precise labeling agent for task‑oriented dialogues.
Return a SINGLE JSON object whose top‑level keys are "turn_1", "turn_2", …

For every turn include:
  "turn_idx"        : integer starting at 1
  "utterances"      : {"user": "...", "system": "..."}
  "state_repr_prev" : canonical string BEFORE the user utterance
  "system_action"   : one ontology label
  "state_repr_next" : canonical string AFTER the system utterance
  "transition"      : { "action": "<system_action>" }

Ontology:
  slots  = [area, price, food, people, day, time, booking_ref]
  system_actions = [greet, goodbye, request(slot), inform(slot), confirm(slot),
                    offer(slot_set), book(slot_set), explain_no_match, apologize]

Canonical‑state rules:
  * lower‑case everything; unknown slot → value "none"
  * synonyms: cheap/inexpensive→cheap, centre/center→centre,
              tomorrow→date_any, 6 pm/18:00→time_any
  * open‑value slots: phone→phone_any, booking reference→ref_any, names→name_any
  * alphabetical slot order joined by "|"

STRICT:
  – output MUST be valid JSON, no ``` fences
  – never invent slot names
  – retry internally until JSON validates
"""

USER_PROMPT_TMPL = """
DIALOGUE HISTORY:

{dialogue_text}

TASK:
Annotate ALL turns following the format in the system prompt.
"""

# ----------------- 3. LLM 호출 -----------------
def join_utterances(dialogue: Dict[str, Any]) -> str:
    return "\n".join(f"{t['speaker']}: {t['utterance']}"
                     for t in dialogue["turns"])

def annotate_dialogue(idx: int, dialogue: Dict[str, Any]) -> Dict[str, Any]:
    text_block = join_utterances(dialogue)
    user_prompt = USER_PROMPT_TMPL.format(dialogue_text=text_block)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )
        dlg_json = json.loads(resp.choices[0].message.content)

        # -------- 3‑1 해시·정규화 후속 처리 --------
        for turn in dlg_json.values():
            # prev
            canon_prev = canonicalize(turn["state_repr_prev"])
            sid_prev   = hash_state(canon_prev)
            turn["state_repr_prev"] = canon_prev
            turn["state_id_prev"]   = sid_prev
            # next
            canon_next = canonicalize(turn["state_repr_next"])
            sid_next   = hash_state(canon_next)
            turn["state_repr_next"] = canon_next
            turn["state_id_next"]   = sid_next
            # transition 보강
            tr = turn["transition"]
            tr["prev_state_id"] = sid_prev
            tr["next_state_id"] = sid_next

        return dlg_json
    except Exception as e:
        return {"error": str(e)}

# ----------------- 4. 병렬 주석 -----------------
def parallel_annotate(dialogues: List[Dict[str, Any]],
                      num_threads: int = 10) -> Dict[int, Dict]:
    results: Dict[int, Dict] = {}
    with cf.ThreadPoolExecutor(max_workers=num_threads) as exe:
        futures = {exe.submit(annotate_dialogue, idx, dlg): idx
                   for idx, dlg in enumerate(dialogues)}
        for fut in tqdm(cf.as_completed(futures),
                        total=len(futures),
                        desc="Annotating"):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                results[idx] = {"error": str(exc)}
    return results

# ----------------- 5. 데이터 로드 -----------------
def load_multiwoz_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else list(data.values())

# ----------------- 6. 메인 -----------------
if __name__ == "__main__":
    FILE_PATH   = "dataset/train/dialogues_001.json"
    NUM_THREADS = 20
    MAX_FIRST   = 60     # 앞에서 20개만 테스트

    dialogues = load_multiwoz_json(FILE_PATH)
    print(f"원본 대화 수: {len(dialogues)}")
    dialogues = dialogues
    print(f"이번 주석 대상: {len(dialogues)}개")

    annotated = parallel_annotate(dialogues, NUM_THREADS)

    with open("results_dialogue001.json", "w", encoding="utf-8") as fp:
        json.dump(annotated, fp, ensure_ascii=False, indent=2)

    print("완료: results_dialogue001.json 저장")
