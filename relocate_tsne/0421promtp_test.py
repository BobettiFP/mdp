import json
import os
import threading
from queue import Queue
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# OpenAI API 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 시스템 프롬프트 설정
PROMPT_TEMPLATE = """
##########################
##  SYSTEM  PROMPT
##########################
You are “Universal‑MDP‑Annotator‑GPT”, an expert in dialogue theory
(speech acts, QUD, information‑state) and reinforcement learning.
Your job is to convert **any multi‑turn human dialogue** into Markov‑Decision‑Process
elements using the **have / need** abstraction.

──────────
★ CORE IDEA
──────────
Every utterance reveals two things:
  1. What the speaker ALREADY HAS (facts, commitments, offers).
  2. What the speaker STILL NEEDS (information, action, resource).

──────────
★ SCHEMA (one JSON object only)
──────────
{
  "dialogue_id": "<string>",
  "turns": [
    {
      "turn_id": <int>,

      // exactly ONE of the following two utterance fields must appear:
      "utterance_user":   "<raw text>"   OR
      "utterance_system": "<raw text>",

      // if the turn is by USER, fill *_user fields and omit *_system (and vice‑versa)
      "state_user":   { "have": [ "<triple|fact>", … ],
                        "need": [ "<goal(arg1,arg2)>", … ] },

      "action_user":  { "intent": "<intent‑string>",
                        "slots":  { "<slot>": "<value>", … } },

      "state_system": { ... same structure … },   // only if system speaks
      "action_system":{ ... same structure … },   // only if system speaks

      "reward": <float>,   // +1 fulfils speaker need, –1 blocks, 0 neutral
      "gamma":  <float>    // 0.99 normally, 0.0 on terminal turn
    }
  ]
}

──────────
★ INTENT CONVENTION
──────────
• Use these 9 universal intents                                (case‑exact):
    "inform", "confirm", "deny", "request([...])",
    "offer", "accept", "reject", "greet", "bye"
• For every request, list the targets inside brackets:
      request(["restaurant"])
      request(["symptom","duration"])   // multiple requests
• Combine acts if needed:  "inform+request([...])"

──────────
★ STATE RULES
──────────
1.  state.have  –  list triples **subject:attribute:value** OR succinct facts the
    speaker now possesses / has asserted / can provide.
2.  state.need  –  list goals as **function‑style strings**; e.g.
        restaurant(cheap,north)        confirm_symptom(cough)
    · Split logical OR into separate goals: confirm_symptom(cough),
                                             confirm_symptom(sore_throat)
3.  Keep arrays empty ([]) if nothing applies; never null.

──────────
★ REWARD & GAMMA
──────────
• reward = +1  if the turn clearly advances the speaker toward clearing a need.
• reward = –1  if it obstructs or contradicts.
• reward =  0  otherwise.   Last turn gets gamma = 0.0.

──────────
★ OUTPUT RULES (CRITICAL)
──────────
1. **Return only the JSON object above** – no prose, no markdown.
2. JSON must be valid UTF‑8, double‑quoted keys, no trailing commas.
3. Think step‑by‑step internally but **never reveal** your chain‑of‑thought.

──────────
★ EXAMPLE (abridged turn)
──────────
USER  : "Any cheap restaurant in the north?"
⇒ state_user.have  = []
   state_user.need = ["restaurant(cheap,north)"]
   action_user     = request(["restaurant"])
"""

def get_utterances_from_dialogue(dialogue: Dict[str, Any]) -> List[str]:
    """
    한 대화에서 모든 발화(utterance)를 추출합니다.

    Args:
        dialogue: 단일 대화 데이터

    Returns:
        해당 대화의 모든 발화 리스트
    """
    utterances = []
    
    for turn in dialogue.get('turns', []):
        if 'utterance' in turn:
            # 화자 정보와 함께 발화 저장
            speaker = turn.get('speaker', '')
            utterance = turn.get('utterance', '')
            utterances.append(f"{speaker}: {utterance}")
    
    return utterances

def annotate_text(dialogue_text: str) -> str:
    """OpenAI API를 사용하여 대화 텍스트에 주석을 추가합니다."""
    user_prompt = f"""
You will receive a multi‑turn dialogue.  
1. Assign a unique dialogue_id.  
2. Produce a single JSON object exactly following the **Schema‑Free MDP JSON**
   defined in the system instructions.  
3. Remember to keep your chain‑of‑thought private.
    Text:
    {dialogue_text}
    """
    
    try:
        response = client.chat.completions.create(
            model='gpt-4o',
            temperature=0.1,
            messages=[
                {"role": "system", "content": PROMPT_TEMPLATE},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API 호출 중 오류 발생: {e}")
        return f"Error: {str(e)}"

def refine_result(result: str) -> str:
    """코드 블록과 json 태그를 제거하고 결과를 정제합니다."""
    return result.replace("```", "").replace("json", "").strip()

def worker(queue: Queue, results: Dict[int, Dict], pbar: tqdm):
    """작업자 스레드 함수"""
    while True:
        try:
            dialogue_idx, dialogue = queue.get()
            
            if dialogue is None:  # 종료 신호
                queue.task_done()
                break
                
            text = get_utterances_from_dialogue(dialogue)
            dialogue_text = "\n".join(text)
            
            annotation = annotate_text(dialogue_text)
            
            try:
                json_result = json.loads(refine_result(annotation))
                results[dialogue_idx] = json_result
            except json.JSONDecodeError as e:
                print(f"JSON 디코딩 오류 (idx={dialogue_idx}): {e}")
                results[dialogue_idx] = {"error": "JSON 파싱 오류", "raw_text": refine_result(annotation)}
            
            pbar.update(1)
            queue.task_done()
        except Exception as e:
            print(f"작업자 오류: {e}")
            queue.task_done()

def parallel_annotate(dialogues: List[Dict[str, Any]], num_threads: int=5, max_dialogues: int=None) -> Dict[int, Dict]:
    """대화 데이터를 병렬로 annotate합니다.
    
    Args:
        dialogues: 대화 데이터 리스트
        num_threads: 사용할 스레드 수
        max_dialogues: 처리할 최대 대화 수 (None이면 모두 처리)
        
    Returns:
        주석이 달린 대화 결과 딕셔너리
    """
    if max_dialogues is not None:
        dialogues_to_process = dialogues[:max_dialogues]
    else:
        dialogues_to_process = dialogues
    
    # 결과를 저장할 딕셔너리
    results = {}
    
    # 작업 큐 생성
    queue = Queue()
    
    # tqdm 진행률 표시기
    pbar = tqdm(total=len(dialogues_to_process), desc="Annotating dialogues")
    
    # 작업자 스레드 생성
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(queue, results, pbar))
        t.daemon = True
        t.start()
        threads.append(t)
    
    # 작업 큐에 작업 추가
    for idx, dialogue in enumerate(dialogues_to_process):
        queue.put((idx, dialogue))
    
    # 작업 완료 대기
    queue.join()
    
    # 모든 작업자 스레드 종료
    for _ in range(num_threads):
        queue.put((None, None))
    
    for t in threads:
        t.join()
    
    pbar.close()
    
    return results

def load_multiwoz_dialogues(file_path: str) -> List[Dict[str, Any]]:
    """
    MultiWOZ JSON 파일에서 모든 대화 데이터를 로드합니다.
    """
    # JSON 파일 로드
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # data가 리스트 형태인지 확인
    if isinstance(data, list):
        return data
    # data가 딕셔너리 형태라면 (일부 MultiWOZ 버전에서 가능)
    elif isinstance(data, dict):
        # 각 대화를 리스트로 반환
        return list(data.values())
    else:
        raise ValueError(f"예상치 못한 데이터 형식입니다: {type(data)}")

# 메인 함수
def main():
    # MultiWOZ 데이터 파일 경로
    file_path = "dataset/train/dialogues_001.json"  # 실제 파일 경로로 변경해주세요
    
    # 모든 대화 데이터 로드
    dialogues = load_multiwoz_dialogues(file_path)
    print(f"총 {len(dialogues)}개의 대화를 로드했습니다.")
    
    # 병렬 처리 설정
    num_threads = 20  # 사용할 스레드 수
    max_dialogues = None  # 처리할 최대 대화 수 (None으로 설정하면 모든 대화 처리)
    
    # 병렬로 대화 주석 처리
    results = parallel_annotate(dialogues, num_threads, max_dialogues)
    
    # 결과 저장
    with open('./results_parallel.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"결과가 'results_parallel.json' 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()