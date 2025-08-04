#!/usr/bin/env python3
"""
Dynamic Key-Value State Analyzer — LLM annotation 단일 파일 처리
------------------------------------------------
• LLM annotation 결과에서 key-value 조합을 state로 취급하여 병합
• edit distance AND semantic similarity가 모두 임계값을 만족할 때만 key-value 상태 병합
"""
import json, pathlib, argparse, collections, os, re, itertools
from typing import Dict, List, Tuple, Set

# ────────────────────────────── 설정 ──────────────────────────────
EDIT_DIST_MAX  = 2       # Levenshtein 거리 임계값 (<=)
EMBED_SIM_MIN  = 0.90    # 코사인 유사도 임계값 (>=)
EMBED_MODEL_ID = os.getenv("SENTENCE_MODEL", "all-MiniLM-L6-v2")

# RapidFuzz 우선, 없으면 difflib
try:
    from rapidfuzz.distance import Levenshtein
    _lev = lambda a, b: Levenshtein.distance(a, b)
except ImportError:
    from difflib import SequenceMatcher
    _lev = lambda a, b: int(round((1 - SequenceMatcher(None, a, b).ratio()) * max(len(a), len(b))))

# sentence-transformers (없으면 임베딩 병합 불가)
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _embed_model = SentenceTransformer(EMBED_MODEL_ID)
except ImportError:
    SentenceTransformer = None
    _embed_model = None

# 보호 접미사 ― 절대 다른 슬롯과 합치지 않는다
PROTECTED_SUFFIX = {"name", "phone", "ref", "reference", "reference_number"}

# 동적 매핑
_DYNAMIC_MAP: Dict[str, str] = {}

# ────────────────────────────── 기본 정규화 ──────────────────────────────
_SPLIT = re.compile(r"[-\s]+")
def _canon_basic(s: str) -> str:
    # key=value 형태 분리
    if "=" in s:
        key_part, value_part = s.split("=", 1)
        # key 부분만 정규화 (value는 그대로 유지)
        key_normalized = _SPLIT.sub("_", key_part.lower())
        key_normalized = re.sub(r"_+", "_", key_normalized).strip("_")
        return f"{key_normalized}={value_part}"
    else:
        # = 없으면 전체 정규화
        s = _SPLIT.sub("_", s.lower())
        s = re.sub(r"_+", "_", s).strip("_")
        return s

def canon(slot_key_value: str) -> str:
    b = _canon_basic(slot_key_value)
    return _DYNAMIC_MAP.get(b, b)

# ────────────────────────────── 원본 슬롯 수집 ──────────────────────────────
def _raw_slot_names(turn: Dict) -> List[str]:
    """key-value 조합으로 슬롯 이름 수집"""
    res = []
    
    # state에서 key-value 조합 추출
    if isinstance(turn.get("state"), dict):
        for domain, slots in turn["state"].items():
            if isinstance(slots, dict):
                for slot_name, slot_value in slots.items():
                    # 값이 딕셔너리이고 'value' 키가 있으면 그 값 사용
                    if isinstance(slot_value, dict) and "value" in slot_value:
                        slot_value = slot_value["value"]
                    
                    # key=value 형태로 조합
                    if slot_value is not None:
                        key_value = f"{domain}_{slot_name}={str(slot_value)}"
                        res.append(key_value)
    
    # state_update에서 key-value 조합 추출
    if isinstance(turn.get("state_update"), dict):
        for domain, slots in turn["state_update"].items():
            if isinstance(slots, dict):
                for slot_name, slot_value in slots.items():
                    # 값이 딕셔너리이고 'value' 키가 있으면 그 값 사용
                    if isinstance(slot_value, dict) and "value" in slot_value:
                        slot_value = slot_value["value"]
                    
                    # key=value 형태로 조합
                    if slot_value is not None:
                        key_value = f"{domain}_{slot_name}={str(slot_value)}"
                        res.append(key_value)
    
    return res

# ────────────────────────────── 병합 로직 ──────────────────────────────
def _build_dynamic_map(raw_names: Set[str]) -> Dict[str, str]:
    basics = sorted({_canon_basic(n) for n in raw_names})
    if len(basics) <= 1:
        return {n: n for n in basics}

    if _embed_model:
        emb = _embed_model.encode(basics, normalize_embeddings=True)
    else:
        emb = None

    parent = {n: n for n in basics}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 모든 쌍 비교
    for i, a in enumerate(basics):
        # key=value에서 key 부분만 추출하여 접미사 확인
        key_a = a.split("=")[0] if "=" in a else a
        suf_a = key_a.split('_')[-1]
        
        for j in range(i + 1, len(basics)):
            b = basics[j]
            # key=value에서 key 부분만 추출하여 접미사 확인
            key_b = b.split("=")[0] if "=" in b else b
            suf_b = key_b.split('_')[-1]

            # 보호 접미사 충돌 차단
            if suf_a in PROTECTED_SUFFIX or suf_b in PROTECTED_SUFFIX:
                continue

            # ① edit distance
            if _lev(a, b) > EDIT_DIST_MAX:
                continue

            # ② embedding (없으면 병합 불가)
            if emb is None:
                continue
            sim = float(np.dot(emb[i], emb[j]))
            if sim < EMBED_SIM_MIN:
                continue

            union(a, b)

    # 대표 슬롯: 언더바 많은 것(도메인 포함) → 길이 긴 것 → 알파벳
    clusters: Dict[str, Set[str]] = collections.defaultdict(set)
    for n in basics:
        clusters[find(n)].add(n)

    mapping: Dict[str, str] = {}
    for root, members in clusters.items():
        rep = max(members, key=lambda x: (x.count('_'), len(x), x))
        for m in members:
            mapping[m] = rep
    return mapping

# ────────────────────────────── 슬롯 추출 ──────────────────────────────
def _extract_turn_slots(turn: Dict) -> Dict[str, str]:
    """key-value 조합으로 슬롯 추출 (key는 정규화, value는 원본 유지)"""
    out: Dict[str, str] = {}
    
    # state에서 슬롯 값 추출
    if isinstance(turn.get("state"), dict):
        for domain, slots in turn["state"].items():
            if not isinstance(slots, dict):
                continue
            for slot_name, slot_value in slots.items():
                # 값이 딕셔너리이고 'value' 키가 있으면 그 값 사용
                if isinstance(slot_value, dict) and "value" in slot_value:
                    slot_value = slot_value["value"]
                
                # 문자열로 변환
                if slot_value is not None:
                    key_value = f"{domain}_{slot_name}={str(slot_value)}"
                    canonical_key_value = canon(key_value)
                    out[canonical_key_value] = str(slot_value)  # value는 원본 유지
    
    return out

# ────────────────────────────── 파일 로드 ──────────────────────────────
def load_turns(path: pathlib.Path, ann_type: str) -> List[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {path}: {e}")
        return []

    # 데이터가 리스트가 아니면 리스트로 감싸기
    if not isinstance(data, list):
        data = [data]

    turns: List[Dict] = []
    for dlg in data:
        # dialogue_id 추출
        did = dlg.get("dialogue_id", "unknown")
        
        # turns 리스트 추출
        dturns = dlg.get("turns", [])
        
        for turn in dturns:
            turn = dict(turn)
            turn["dialogue_id"] = did
            turn["annotation_type"] = ann_type
            
            # turn_id 정규화
            try:
                turn["turn_id"] = int(turn.get("turn_id", 0))
            except (ValueError, TypeError):
                turn["turn_id"] = 0
                
            turns.append(turn)
    
    return turns

# ────────────────────────────── 레코드 생성 ──────────────────────────────
def build_records(turns: List[Dict]) -> List[Dict]:
    by = collections.defaultdict(list)
    for t in turns:
        by[(t["annotation_type"], t["dialogue_id"])].append(t)

    recs: List[Dict] = []
    for key, tl in by.items():
        tl.sort(key=lambda x: x["turn_id"])
        prev = set()  # 이전 turn의 key-value 조합들
        
        for tr in tl:
            cur_slots = _extract_turn_slots(tr)
            cur = set(cur_slots.keys())  # 현재 turn의 key-value 조합들
            
            add = cur - prev    # 새로 추가된 key-value 조합들
            dele = prev - cur   # 제거된 key-value 조합들

            if add and not dele:
                act = "inform"
            elif dele and not add:
                act = "delete"  
            elif add or dele:
                act = "update"
            else:
                act = "noop"

            # state_before와 state_after를 key-value 조합 리스트로 표현
            recs.append({
                "state_before": sorted(list(prev)),
                "action": act,
                "state_after": sorted(list(cur)),
                "added_states": sorted(list(add)),
                "deleted_states": sorted(list(dele)),
                "annotation_type": tr["annotation_type"],
                "dialogue_id": tr["dialogue_id"],
                "turn_id": tr["turn_id"],
                "utterance": tr.get("utterance", ""),
            })
            prev = cur
    return recs

# ────────────────────────────── 실험 분석 ──────────────────────────────
def analyze_thresholds(raw_names: Set[str], edit_dist_range: List[int], embed_sim_range: List[float]) -> Dict:
    """다양한 임계값 조합으로 key-value 상태 병합 실험"""
    results = []
    original_count = len({_canon_basic(n) for n in raw_names})
    
    print(f"Original unique key-value states (after basic normalization): {original_count}")
    print("\nRunning threshold analysis...")
    print("EditDist | EmbedSim | Final States | Reduction% | Clusters")
    print("-" * 65)
    
    for edit_max in edit_dist_range:
        for embed_min in embed_sim_range:
            # 임시로 전역 변수 변경
            global EDIT_DIST_MAX, EMBED_SIM_MIN
            orig_edit, orig_embed = EDIT_DIST_MAX, EMBED_SIM_MIN
            EDIT_DIST_MAX, EMBED_SIM_MIN = edit_max, embed_min
            
            # 병합 실행
            mapping = _build_dynamic_map(raw_names)
            final_states = set(mapping.values())
            final_count = len(final_states)
            reduction_pct = (original_count - final_count) / original_count * 100
            
            # 클러스터 분석
            clusters = collections.defaultdict(list)
            for orig, final in mapping.items():
                clusters[final].append(orig)
            cluster_sizes = [len(members) for members in clusters.values()]
            avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
            
            result = {
                "edit_dist_max": edit_max,
                "embed_sim_min": embed_min,
                "original_count": original_count,
                "final_count": final_count,
                "reduction_percent": reduction_pct,
                "avg_cluster_size": avg_cluster_size,
                "mapping": dict(mapping)
            }
            results.append(result)
            
            print(f"   {edit_max:2d}    |   {embed_min:.2f}   |     {final_count:3d}     |   {reduction_pct:5.1f}%   |   {avg_cluster_size:.1f}")
            
            # 전역 변수 복원
            EDIT_DIST_MAX, EMBED_SIM_MIN = orig_edit, orig_embed
    
    return {"analysis_results": results, "original_keyvalue_states": list(raw_names)}

def show_detailed_clusters(raw_names: Set[str], edit_dist: int, embed_sim: float) -> None:
    """특정 임계값에서의 key-value 상태 클러스터 상세 정보 출력"""
    global EDIT_DIST_MAX, EMBED_SIM_MIN
    orig_edit, orig_embed = EDIT_DIST_MAX, EMBED_SIM_MIN
    EDIT_DIST_MAX, EMBED_SIM_MIN = edit_dist, embed_sim
    
    mapping = _build_dynamic_map(raw_names)
    
    # 클러스터 그룹화
    clusters = collections.defaultdict(list)
    for orig, final in mapping.items():
        clusters[final].append(orig)
    
    print(f"\n=== Detailed Key-Value State Clusters (EditDist≤{edit_dist}, EmbedSim≥{embed_sim:.2f}) ===")
    
    # 큰 클러스터부터 정렬
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    for final_state, members in sorted_clusters:
        if len(members) > 1:  # 병합된 것만 표시
            print(f"\n[{final_state}] ← {len(members)} key-value states merged:")
            for member in sorted(members):
                print(f"  • {member}")
    
    # 전역 변수 복원
    EDIT_DIST_MAX, EMBED_SIM_MIN = orig_edit, orig_embed

# ────────────────────────────── CLI ──────────────────────────────
def _cli() -> None:
    ap = argparse.ArgumentParser(description="LLM annotation 결과를 처리하여 key-value 상태를 병합합니다.")
    ap.add_argument("--llm", required=True, help="LLM annotation 파일 경로")
    ap.add_argument("--export", help="결과 출력 파일 경로 (분석 모드에서는 선택사항)")
    
    # 실험 모드 옵션들
    ap.add_argument("--analyze", action="store_true", help="key-value 상태 임계값 분석 모드 실행")
    ap.add_argument("--edit-range", default="0,1,2,3,4,5", help="Edit distance 범위 (예: 0,1,2,3)")
    ap.add_argument("--embed-range", default="0.7,0.8,0.85,0.9,0.95", help="Embedding similarity 범위 (예: 0.8,0.9,0.95)")
    ap.add_argument("--detailed", help="특정 임계값의 상세 key-value 상태 클러스터 보기 (예: --detailed 2,0.9)")
    
    args = ap.parse_args()

    llm_turns = load_turns(pathlib.Path(args.llm), "llm")

    raw: Set[str] = set()
    for t in llm_turns:
        raw.update(_raw_slot_names(t))

    if args.analyze:
        # 실험 분석 모드
        edit_range = [int(x.strip()) for x in args.edit_range.split(",")]
        embed_range = [float(x.strip()) for x in args.embed_range.split(",")]
        
        analysis_results = analyze_thresholds(raw, edit_range, embed_range)
        
        if args.export:
            analysis_path = pathlib.Path(args.export).with_suffix('.analysis.json')
            with analysis_path.open("w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=2)
            print(f"\n✔ Analysis results saved to {analysis_path}")
        
    elif args.detailed:
        # 상세 클러스터 보기 모드
        try:
            edit_val, embed_val = args.detailed.split(",")
            edit_val, embed_val = int(edit_val.strip()), float(embed_val.strip())
            show_detailed_clusters(raw, edit_val, embed_val)
        except ValueError:
            print("Error: --detailed 옵션은 '편집거리,임베딩유사도' 형식이어야 합니다 (예: 2,0.9)")
            return
        
    else:
        # 기본 처리 모드
        if not args.export:
            print("Error: 기본 모드에서는 --export가 필요합니다")
            return
            
        global _DYNAMIC_MAP
        _DYNAMIC_MAP = _build_dynamic_map(raw)

        recs = build_records(llm_turns)
        raw_to_final = {n: canon(n) for n in raw}

        out = {"annotations": recs, "canonical_keyvalue_map": raw_to_final}
        out_path = pathlib.Path(args.export)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        print(f"✔ {args.export} written  ({len(recs):,} records, {len(set(raw_to_final.values()))} unique key-value states)")

if __name__ == "__main__":
    _cli()