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
You are an expert annotator for multi-domain dialogues.
Below are the annotation rules you MUST follow when I provide a conversation.

I want you to produce a single top-level dictionary with keys for each turn:
  "turn_1", "turn_2", "turn_3", ...

Each "turn_x" is a dictionary containing:

1) "turn_idx"
   - The turn number (starting from 1).

2) "speaker"
   - Either "user" or "system".

3) "utterance"
   - The exact text for that turn.

4) "dialogue_acts"
   - An object whose keys are act types (e.g., "Inform", "Request", "Offer", etc.),
     and whose values are arrays of [domain, slot, value] tuples.

5) "belief_state"
   - An array of objects, each with:
       "domain": (e.g., "hotel", "restaurant", "train", "context", etc.)
       "slots": a list of [slot, value] pairs
   - **Important**: If user or system mentions any global context 
     (e.g., "I am currently at Cambridge station," "I only have 1 hour left," etc.),
     store that info under `{"domain": "context", "slots": [...]}`
   - Keep domain-specific info in their respective domain objects (e.g. "hotel", "restaurant").

6) "transitions"
   - Show how the turn's final state changes to the next turn's state:
       {
         "state_1": [...],
         "action": "...",
         "state_2": [...]
       }
   - "state_1" = the final belief_state after processing this turn.
   - "action" = main dialogue act(s) that changed the state.
   - "state_2" = the belief_state for the next turn.

7) "reward"
   - "positive", "completed", or "negative" if the user or system indicates success/failure/dissatisfaction. Otherwise "NA".

### Output Must Be Valid JSON

Your final output must be a single JSON dictionary, not an array, with top-level keys "turn_1", "turn_2", etc.
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
    Below is a natural language conversation between a user and a system. Please extract, define, and annotate the dialogue elements based on the guidelines provided.
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
    file_path = "mdp/dataset/train/dialogues_001.json"  # 실제 파일 경로로 변경해주세요
    
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