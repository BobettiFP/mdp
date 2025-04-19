import json, os, hashlib, concurrent.futures as cf
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

# ---------------- 환경 설정 ----------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment.")
client = OpenAI(api_key=api_key)

# ------------- 고정 시스템 프롬프트 -------------
SYSTEM_PROMPT = """
You are “MDP‑Annotator‑v1”, a precise labeling agent for task‑oriented dialogues.
Your output must be a SINGLE JSON object whose top‑level keys are "turn_1", "turn_2", …

For every turn include:
  "turn_idx"          (int, starting at 1)
  "utterances": {"user": "...", "system": "..."}
  "state_repr_prev"   (canonical string BEFORE the user utterance)
  "state_id_prev"     (lower‑case sha256 of state_repr_prev)
  "system_action"     (one of the ontology labels)
  "state_repr_next"   (canonical string AFTER the system utterance)
  "state_id_next"     (sha256 of state_repr_next)
  "transition": { "prev_state_id": "...", "action": "...", "next_state_id": "..." }

Ontology:
  slots = [area, price, food, people, day, time, booking_ref]
  system_actions = [greet, goodbye, request(slot), inform(slot), confirm(slot),
                    offer(slot_set), book(slot_set), explain_no_match, apologize]

Canonical‑state rules:
  * lower‑case everything; unknown slot → value "none"
  * synonyms: cheap/inexpensive→cheap, centre/center→centre,
              tomorrow→date_any, 6 pm/18:00→time_any
  * open‑value slots: phone→phone_any, booking reference→ref_any, names→name_any
  * alphabetical slot order joined by "|"
  * state_id = sha256(canonical string)

STRICT:
  – output MUST be valid JSON, nothing else
  – never invent slot names
  – retry internally until JSON validates
"""

# ---------- 사용자 프롬프트 템플릿 ----------
USER_PROMPT_TMPL = """
DIALOGUE HISTORY:

{dialogue_text}

TASK:
Return the JSON annotation for ALL turns using the format in the system prompt.
"""

# ---------- 도우미 함수 ----------
def join_utterances(dialogue: Dict[str, Any]) -> str:
    """turn 리스트를 'speaker: utterance' 형식으로 연결"""
    lines = []
    for t in dialogue["turns"]:
        speaker = t.get("speaker", "unknown")
        utt = t.get("utterance", "")
        lines.append(f"{speaker}: {utt}")
    return "\n".join(lines)

def annotate_dialogue(idx: int, dialogue: Dict[str, Any]) -> Dict[str, Any]:
    """하나의 대화를 GPT‑4o로 주석"""
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
        raw = resp.choices[0].message.content
        return json.loads(raw)
    except Exception as e:
        # 오류 발생 시 원본 응답/메시지 함께 저장
        return {"error": str(e), "raw": locals().get("raw", "")}

# ---------- 멀티스레드 실행 ----------
def parallel_annotate(dialogues: List[Dict[str, Any]],
                      num_threads: int = 10) -> Dict[int, Dict]:
    results: Dict[int, Dict] = {}
    with cf.ThreadPoolExecutor(max_workers=num_threads) as exe:
        futures = {
            exe.submit(annotate_dialogue, idx, dlg): idx
            for idx, dlg in enumerate(dialogues)
        }
        for fut in tqdm(cf.as_completed(futures),
                        total=len(futures),
                        desc="Annotating"):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                results[idx] = {"error": str(exc)}
    return results

# ---------- 데이터 로드 ----------
def load_multiwoz_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else list(data.values())

# ----------------- 메인 -----------------
if __name__ == "__main__":
    FILE_PATH   = "dataset/train/dialogues_001.json"
    NUM_THREADS = 20
    MAX_FIRST   = 20              # ← 확인용으로 30개만!

    dialogues = load_multiwoz_json(FILE_PATH)
    print(f"원본 대화 수: {len(dialogues)}")

    # ---- ① 앞에서 30개만 잘라서 사용 ----
    dialogues = dialogues[:MAX_FIRST]
    print(f"이번 주석 대상: {len(dialogues)}개")

    # ---- ② 병렬 주석 실행 ----
    annotated = parallel_annotate(dialogues, NUM_THREADS)

    with open("results_first20.json", "w", encoding="utf-8") as fp:
        json.dump(annotated, fp, ensure_ascii=False, indent=2)

    print("완료: results_first30.json 저장")
