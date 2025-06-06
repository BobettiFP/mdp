#!/usr/bin/env python3
"""
개선된 슬롯 정규화 및 재분석
========================
하이픈/언더스코어 문제를 해결하고 실제 겹침을 확인
"""

import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

def improved_normalise(slot: str) -> str:
    """개선된 정규화 함수"""
    # 1. 소문자 변환
    slot = slot.lower()
    
    # 2. 하이픈을 언더스코어로 통일
    slot = slot.replace("-", "_")
    
    # 3. 복합어 분리
    slot = slot.replace("postcode", "post_code")
    slot = slot.replace("pricerange", "price_range")
    slot = slot.replace("bookday", "book_day")
    slot = slot.replace("bookstay", "book_stay")
    slot = slot.replace("bookpeople", "book_people")
    slot = slot.replace("leaveat", "leave_at")
    slot = slot.replace("arriveby", "arrive_by")
    
    # 4. 도메인 접두사 제거 (마지막에 처리)
    for prefix in ["restaurant_", "hotel_", "train_", "taxi_", "attraction_", 
                   "bus_", "hospital_", "police_"]:
        if slot.startswith(prefix):
            slot = slot[len(prefix):]
    
    return slot

def analyze_overlap(data):
    """실제 겹침 분석"""
    # 원본 슬롯과 정규화된 슬롯 매핑
    human_slots_raw = defaultdict(set)  # normalized -> {raw slots}
    llm_slots_raw = defaultdict(set)
    
    # Human annotations
    for ann in data.get('human_annotations', []):
        for slot in ann.get('slots', {}):
            normalized = improved_normalise(slot)
            human_slots_raw[normalized].add(slot)
    
    # LLM annotations  
    for ann in data.get('llm_annotations', []):
        for slot in ann.get('slots', {}):
            normalized = improved_normalise(slot)
            llm_slots_raw[normalized].add(slot)
    
    # 겹침 분석
    human_norm = set(human_slots_raw.keys())
    llm_norm = set(llm_slots_raw.keys())
    
    overlap = human_norm & llm_norm
    human_only = human_norm - llm_norm
    llm_only = llm_norm - human_norm
    
    print("\n===== 개선된 정규화 후 분석 =====")
    print(f"Human 고유 슬롯 (정규화): {len(human_only)}")
    print(f"LLM 고유 슬롯 (정규화): {len(llm_only)}")
    print(f"겹치는 슬롯 (정규화): {len(overlap)}")
    print(f"LLM innovation rate: {len(llm_only) / len(llm_norm):.3f}")
    
    # 겹치는 슬롯의 원본 표기 확인
    print("\n===== 겹치는 슬롯의 다양한 표기 =====")
    overlap_details = []
    for norm_slot in sorted(overlap)[:20]:  # 상위 20개
        human_variants = sorted(human_slots_raw[norm_slot])
        llm_variants = sorted(llm_slots_raw[norm_slot])
        print(f"\n{norm_slot}:")
        print(f"  Human: {', '.join(human_variants)}")
        print(f"  LLM: {', '.join(llm_variants)}")
        
        overlap_details.append({
            'normalized_slot': norm_slot,
            'human_variants': ', '.join(human_variants),
            'llm_variants': ', '.join(llm_variants),
            'total_variants': len(human_variants) + len(llm_variants)
        })
    
    # LLM만 찾은 슬롯 중 흥미로운 것들
    print("\n===== LLM만 찾은 주요 슬롯 =====")
    llm_unique_slots = []
    for norm_slot in sorted(llm_only)[:30]:
        raw_slots = sorted(llm_slots_raw[norm_slot])
        print(f"{norm_slot}: {', '.join(raw_slots)}")
        llm_unique_slots.append({
            'normalized_slot': norm_slot,
            'raw_slots': ', '.join(raw_slots)
        })
    
    # CSV로 저장
    pd.DataFrame(overlap_details).to_csv('slot_overlap_analysis.csv', index=False)
    pd.DataFrame(llm_unique_slots).to_csv('llm_unique_slots.csv', index=False)
    
    return {
        'human_only': human_only,
        'llm_only': llm_only,
        'overlap': overlap,
        'human_slots_raw': human_slots_raw,
        'llm_slots_raw': llm_slots_raw
    }

def check_clustering_issues(data, analysis_result):
    """클러스터링 문제 진단"""
    canonical_map = data.get('canonical_map', {})
    
    print("\n===== 클러스터링 문제 진단 =====")
    
    # 같은 정규화 슬롯인데 다른 canonical을 가진 경우
    issues = defaultdict(list)
    
    for norm_slot in analysis_result['overlap']:
        human_raws = analysis_result['human_slots_raw'][norm_slot]
        llm_raws = analysis_result['llm_slots_raw'][norm_slot]
        
        # 각 raw 슬롯의 canonical 확인
        canonicals = set()
        for raw in human_raws | llm_raws:
            canon = canonical_map.get(raw, improved_normalise(raw))
            canonicals.add(canon)
        
        if len(canonicals) > 1:
            issues[norm_slot] = {
                'raw_slots': human_raws | llm_raws,
                'canonicals': canonicals
            }
    
    print(f"\n같은 슬롯인데 다른 canonical로 분류된 경우: {len(issues)}개")
    for norm_slot, info in list(issues.items())[:10]:
        print(f"\n{norm_slot}:")
        print(f"  Raw slots: {info['raw_slots']}")
        print(f"  Canonicals: {info['canonicals']}")

if __name__ == "__main__":
    # 데이터 로드
    with open('processed_annotations.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 분석 실행
    analysis_result = analyze_overlap(data)
    
    # 클러스터링 문제 확인
    check_clustering_issues(data, analysis_result)
