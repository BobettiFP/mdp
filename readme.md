# ConvLab-3 표준 형식 어노테이션 실행 가이드

## 1. 필요한 패키지 설치

```bash
pip install openai python-dotenv tqdm
```

## 2. 환경 변수 설정

`.env` 파일 생성:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## 3. 기본 실행 명령어

### 소규모 테스트 (10개 대화)
```bash
python annotate_convlab3_standard.py \
    --input input_dialogues.json \
    --output convlab3_annotated.json \
    --limit 10 \
    --threads 5 \
    --stats
```

### 전체 데이터셋 처리
```bash
python annotate_convlab3_standard.py \
    --input multiwoz_dialogues.json \
    --output convlab3_multiwoz_reannotated.json \
    --threads 20 \
    --stats
```

### 대용량 데이터 배치 처리
```bash
# 메모리 절약을 위해 배치로 처리
python annotate_convlab3_standard.py \
    --input large_dataset.json \
    --output batch_1.json \
    --limit 1000 \
    --threads 15

python annotate_convlab3_standard.py \
    --input large_dataset.json \
    --output batch_2.json \
    --limit 1000 \
    --threads 15
```

## 4. 결과 검증 명령어

### JSON 형식 검증
```bash
python -m json.tool convlab3_annotated.json > /dev/null && echo "Valid JSON" || echo "Invalid JSON"
```

### 기본 통계 확인
```bash
python -c "
import json
with open('convlab3_annotated.json', 'r') as f:
    data = json.load(f)
print(f'Total dialogues: {len(data)}')
errors = sum(1 for d in data if 'error' in d)
print(f'Errors: {errors}')
print(f'Success rate: {(len(data)-errors)/len(data)*100:.1f}%')
"
```

## 5. ConvLab-3와 호환성 테스트

### ConvLab-3 설치
```bash
git clone https://github.com/ConvLab/ConvLab-3.git
cd ConvLab-3
pip install -e .
```

### 데이터 로드 테스트
```bash
python -c "
from convlab.util.unified_datasets_util import load_dataset
import json

# 우리 데이터 로드 테스트
with open('convlab3_annotated.json', 'r') as f:
    our_data = json.load(f)

print('Data loaded successfully!')
print(f'Sample dialogue keys: {list(our_data[0].keys())}')
print(f'Sample turn keys: {list(our_data[0][\"turns\"][0].keys())}')
"
```

## 6. 성능 모니터링

### 실시간 진행상황 모니터링
```bash
# 터미널 1: 실행
python annotate_convlab3_standard.py --input data.json --output result.json --threads 20

# 터미널 2: 진행상황 확인
watch -n 5 "wc -l result.json && tail -1 result.json"
```

### 에러 로그 분석
```bash
python -c "
import json
with open('convlab3_annotated.json', 'r') as f:
    data = json.load(f)

errors = [d for d in data if 'error' in d]
for i, err in enumerate(errors[:5]):  # 처음 5개 에러만
    print(f'Error {i+1}: {err[\"error\"]}')
"
```

## 7. 메모리 최적화 실행

### 대용량 데이터용 (메모리 제한 환경)
```bash
# 작은 배치 크기로 실행
python annotate_convlab3_standard.py \
    --input huge_dataset.json \
    --output output.json \
    --threads 10 \
    --limit 500
```

## 8. 결과 병합 (배치 처리한 경우)

```bash
python -c "
import json

# 여러 배치 파일 병합
all_data = []
for i in range(1, 6):  # batch_1.json ~ batch_5.json
    with open(f'batch_{i}.json', 'r') as f:
        all_data.extend(json.load(f))

with open('merged_convlab3_data.json', 'w') as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f'Merged {len(all_data)} dialogues')
"
```

## 9. 품질 체크 스크립트

```bash
python -c "
import json

with open('convlab3_annotated.json', 'r') as f:
    data = json.load(f)

# 필수 필드 체크
issues = []
for i, d in enumerate(data):
    if 'error' in d:
        continue
    
    # ConvLab-3 필수 필드 확인
    required = ['dialogue_id', 'services', 'turns']
    missing = [k for k in required if k not in d]
    if missing:
        issues.append(f'Dialogue {i}: missing {missing}')
    
    # Turn 필수 필드 확인
    for j, turn in enumerate(d.get('turns', [])):
        turn_required = ['turn_id', 'speaker', 'utterance', 'dialogue_acts', 'state']
        turn_missing = [k for k in turn_required if k not in turn]
        if turn_missing:
            issues.append(f'Dialogue {i}, Turn {j}: missing {turn_missing}')

print(f'Quality check: {len(issues)} issues found')
for issue in issues[:10]:  # 첫 10개만 출력
    print(f'  - {issue}')
"
```

## 10. 최종 RL 실험 준비

```bash
# ConvLab-3 RL 실험용 데이터 변환
python -c "
import json

# 우리 데이터를 ConvLab-3 unified format으로 최종 변환
with open('convlab3_annotated.json', 'r') as f:
    our_data = json.load(f)

# 에러가 없는 대화만 필터링
clean_data = [d for d in our_data if 'error' not in d]

print(f'Clean dialogues: {len(clean_data)}')
print(f'Ready for RL training!')

# ConvLab-3 형식으로 저장
with open('convlab3_rl_ready.json', 'w') as f:
    json.dump(clean_data, f, indent=2)
"
```