#!/usr/bin/env python3
"""
슬롯 유사도 분석 및 시각화
=======================
processed_annotations.json에서 슬롯들을 읽어 유사도를 계산하고
CSV 형태로 출력합니다.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import pandas as pd

def load_annotations(path: Path):
    """processed_annotations.json 파일 로드"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_all_slots(data):
    """모든 슬롯 추출 및 출처 정보 수집"""
    slot_sources = defaultdict(set)  # slot -> {human, llm}
    
    # Human annotations
    for ann in data.get('human_annotations', []):
        for slot in ann.get('slots', {}):
            slot_sources[slot].add('human')
    
    # LLM annotations
    for ann in data.get('llm_annotations', []):
        for slot in ann.get('slots', {}):
            slot_sources[slot].add('llm')
    
    return slot_sources

def normalise_slot(slot: str) -> str:
    """슬롯 이름 정규화"""
    slot = slot.lower().replace("-", "_")
    slot = slot.replace("postcode", "post_code").replace("pricerange", "price_range")
    
    # 도메인 접두사 제거
    for prefix in ["restaurant_", "hotel_", "train_", "taxi_", "attraction_", "bus_", "hospital_", "police_"]:
        if slot.startswith(prefix):
            slot = slot[len(prefix):]
    
    return slot

def calculate_similarities(slots, model):
    """모든 슬롯 쌍의 유사도 계산"""
    slot_list = list(slots)
    n = len(slot_list)
    
    # 정규화된 슬롯 이름
    norm_slots = [normalise_slot(s) for s in slot_list]
    
    # 임베딩 계산
    embeddings = model.encode(norm_slots, show_progress_bar=True, convert_to_numpy=True)
    
    # 유사도 매트릭스
    similarities = []
    
    for i in range(n):
        for j in range(i + 1, n):
            # 문자열 유사도
            str_sim = fuzz.QRatio(norm_slots[i], norm_slots[j]) / 100.0
            
            # 의미적 유사도
            sem_sim = float(np.dot(embeddings[i], embeddings[j]) / 
                           (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8))
            
            # 종합 점수 (둘 중 높은 값)
            combined = max(str_sim, sem_sim)
            
            similarities.append({
                'slot1': slot_list[i],
                'slot2': slot_list[j],
                'slot1_norm': norm_slots[i],
                'slot2_norm': norm_slots[j],
                'string_similarity': round(str_sim, 3),
                'semantic_similarity': round(sem_sim, 3),
                'combined_similarity': round(combined, 3),
                'slot1_source': ', '.join(sorted(slots[slot_list[i]])),
                'slot2_source': ', '.join(sorted(slots[slot_list[j]])),
                'same_canonical': data.get('canonical_map', {}).get(slot_list[i]) == 
                                data.get('canonical_map', {}).get(slot_list[j])
            })
    
    return similarities

def find_potential_matches(similarities, threshold=0.7):
    """임계값 이상의 유사도를 가진 슬롯 쌍 찾기"""
    matches = [s for s in similarities if s['combined_similarity'] >= threshold]
    return sorted(matches, key=lambda x: x['combined_similarity'], reverse=True)

def save_to_csv(similarities, output_path):
    """유사도 결과를 CSV로 저장"""
    df = pd.DataFrame(similarities)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved to {output_path}")

def print_summary(slot_sources, similarities):
    """요약 정보 출력"""
    human_only = [s for s, src in slot_sources.items() if src == {'human'}]
    llm_only = [s for s, src in slot_sources.items() if src == {'llm'}]
    both = [s for s, src in slot_sources.items() if 'human' in src and 'llm' in src]
    
    print("\n===== 슬롯 분포 =====")
    print(f"Human only: {len(human_only)}")
    print(f"LLM only: {len(llm_only)}")
    print(f"Both: {len(both)}")
    print(f"Total unique slots: {len(slot_sources)}")
    
    # 높은 유사도 쌍 출력
    high_sim = find_potential_matches(similarities, 0.8)
    print(f"\n===== 높은 유사도 (>0.8) 슬롯 쌍: {len(high_sim)}개 =====")
    for match in high_sim[:10]:  # 상위 10개만
        print(f"{match['slot1']} <-> {match['slot2']}: "
              f"str={match['string_similarity']}, sem={match['semantic_similarity']}, "
              f"sources=({match['slot1_source']} vs {match['slot2_source']})")

if __name__ == "__main__":
    # 설정
    input_path = Path("processed_annotations.json")
    output_csv = Path("slot_similarities.csv")
    output_high_sim = Path("high_similarity_slots.csv")
    
    # 데이터 로드
    print("Loading data...")
    data = load_annotations(input_path)
    slot_sources = extract_all_slots(data)
    
    # 모델 로드
    print("Loading SBERT model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 유사도 계산
    print("Calculating similarities...")
    similarities = calculate_similarities(slot_sources, model)
    
    # 전체 결과 저장
    save_to_csv(similarities, output_csv)
    
    # 높은 유사도만 따로 저장
    high_sim = find_potential_matches(similarities, 0.7)
    save_to_csv(high_sim, output_high_sim)
    
    # 요약 출력
    print_summary(slot_sources, similarities)
    
    # 클러스터링 문제 진단
    print("\n===== 클러스터링 진단 =====")
    canonical_map = data.get('canonical_map', {})
    
    # 같은 canonical을 가져야 할 것 같은데 다른 경우
    missed_clusters = []
    for sim in similarities:
        if (sim['combined_similarity'] >= 0.8 and 
            not sim['same_canonical'] and
            sim['slot1'] in canonical_map and 
            sim['slot2'] in canonical_map):
            missed_clusters.append(sim)
    
    if missed_clusters:
        print(f"\n놓친 클러스터 후보 ({len(missed_clusters)}개):")
        for m in missed_clusters[:10]:
            print(f"  {m['slot1']} ({canonical_map.get(m['slot1'])}) <-> "
                  f"{m['slot2']} ({canonical_map.get(m['slot2'])}): {m['combined_similarity']}")
