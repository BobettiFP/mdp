#!/usr/bin/env python3
"""
Dynamic Slot Analyzer — self-merging edition
-------------------------------------------
• 사람·LLM 주석 JSON → (state_before, action, state_after) 레코드
• 슬롯 표준화:
    ① 기본 정규화(lower + 공백·하이픈 → 언더바)
    ② edit distance ≤ EDIT_DIST_MAX  → 병합
    ③ 임베딩 코사인 유사도 ≥ EMBED_SIM_MIN → 병합
      *sentence-transformers가 설치돼 있을 때만 사용

CLI
$ python dynamic_slot_analyzer.py \
        --human dialogues_001.json \
        --llm   full_annotation.json \
        --export processed_annotations.json
"""
import json, pathlib, argparse, collections, itertools, os, re
from typing import Dict, List, Tuple, Set

# ──────────────────────────────────────────────────────────────────────────────
# 옵션: 임계치 / 모델
EDIT_DIST_MAX  = 2       # <= 2이면 같은 슬롯으로 간주
EMBED_SIM_MIN  = 0.90    # >= 0.90이면 같은 슬롯
EMBED_MODEL_ID = os.getenv("SENTENCE_MODEL", "all-MiniLM-L6-v2")

# RapidFuzz(권장) → 실패 시 difflib 대체
try:
    from rapidfuzz.distance import Levenshtein
    _levenshtein = lambda a, b: Levenshtein.distance(a, b)
except ImportError:
    from difflib import SequenceMatcher
    _levenshtein = lambda a, b: int(round((1 - SequenceMatcher(None, a, b).ratio()) * max(len(a), len(b))))

# sentence-transformers(선택)
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _embed_model = SentenceTransformer(EMBED_MODEL_ID)
except ImportError:
    SentenceTransformer = None
    _embed_model = None

# 동적 매핑 전역
_DYNAMIC_MAP: Dict[str, str] = {}

# ────────────────────────────────────────────── 1. 기본 정규화
_COMMON_SPLIT = re.compile(r"[-\s]+")
def _canon_basic(slot: str) -> str:
    s = _COMMON_SPLIT.sub("_", slot.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def canon(slot: str) -> str:
    """스크립트 전역에서 쓰이는 공식 Canon 함수"""
    basic = _canon_basic(slot)
    return _DYNAMIC_MAP.get(basic, basic)

# ────────────────────────────────────────────── 2. 슬롯 추출(원본 이름 수집용)
def _raw_slot_names(turn: Dict) -> List[str]:
    names: List[str] = []
    if "slots" in turn and isinstance(turn["slots"], dict):
        names.extend(turn["slots"].keys())
    for f in turn.get("frames", []):
        sv = f.get("state", {}).get("slot_values", {})
        names.extend(sv.keys())
    if "state" in turn and isinstance(turn["state"], dict):
        for dom, dom_slots in turn["state"].items():
            if isinstance(dom_slots, dict):
                names.extend(f"{dom}_{k}" for k in dom_slots.keys())
    return names

# ────────────────────────────────────────────── 3. 동적 병합 매핑 생성
def _build_dynamic_map(raw_names: Set[str]) -> Dict[str, str]:
    basic_names = sorted({_canon_basic(n) for n in raw_names})
    if len(basic_names) <= 1:
        return {n: n for n in basic_names}

    # 유사도 계산 준비(임베딩은 옵션)
    if _embed_model:
        emb = _embed_model.encode(basic_names, normalize_embeddings=True)
    else:
        emb = None

    # Union–Find
    parent = {n: n for n in basic_names}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 모든 쌍 비교(소규모 슬롯 집합 가정)
    for i, a in enumerate(basic_names):
        for j in range(i + 1, len(basic_names)):
            b = basic_names[j]
            similar = False

            # ① edit distance
            if _levenshtein(a, b) <= EDIT_DIST_MAX:
                similar = True
            # ② embedding
            elif emb is not None:
                sim = float(np.dot(emb[i], emb[j]))
                if sim >= EMBED_SIM_MIN:
                    similar = True

            if similar:
                union(a, b)

    # 대표값(짧은 문자열 우선) 결정
    clusters: Dict[str, Set[str]] = collections.defaultdict(set)
    for n in basic_names:
        clusters[find(n)].add(n)

    mapping: Dict[str, str] = {}
    for root, members in clusters.items():
        rep = sorted(members, key=lambda x: (len(x), x))[0]
        for m in members:
            mapping[m] = rep
    return mapping

# ────────────────────────────────────────────── 4. 턴별 슬롯 채집
def _extract_turn_slots(turn: Dict) -> Dict[str, str]:
    slots: Dict[str, str] = {}

    # 1) direct "slots"
    if "slots" in turn and isinstance(turn["slots"], dict):
        slots.update({canon(k): v for k, v in turn["slots"].items()})

    # 2) MultiWOZ frames
    for f in turn.get("frames", []):
        sv = f.get("state", {}).get("slot_values", {})
        for k, vals in sv.items():
            key = canon(k)
            if isinstance(vals, list) and vals:
                val = vals[-1]
                if isinstance(val, dict) and "value" in val:
                    val = val["value"]
                slots[key] = val
            elif isinstance(vals, str):
                slots[key] = vals

    # 3) Schema/SGD nested "state"
    if "state" in turn and isinstance(turn["state"], dict):
        for dom, dom_slots in turn["state"].items():
            if not isinstance(dom_slots, dict):
                continue
            for raw_slot, val in dom_slots.items():
                if isinstance(val, dict) and "value" in val:
                    val = val["value"]
                if isinstance(val, list) and val:
                    last = val[-1]
                    val = last["value"] if isinstance(last, dict) and "value" in last else last
                key = canon(f"{dom}_{raw_slot}")
                slots[key] = val

    return slots

# ────────────────────────────────────────────── 5. 파일 로드
def load_turns(path: pathlib.Path, ann_type: str) -> List[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {path}: {e}")
        return []

    if isinstance(data, dict) and ("dialogue" in data or "turns" in data):
        data = [data]

    turns: List[Dict] = []
    for dlg in data:
        did = dlg.get("dialogue_id") or dlg.get("dialogueID") or "unknown"
        dialogue_turns = dlg.get("dialogue") or dlg.get("turns") or []
        for i, t in enumerate(dialogue_turns):
            t = dict(t)
            t["dialogue_id"] = did
            t["annotation_type"] = ann_type
            try:
                t["turn_id"] = int(t.get("turn_id", i))
            except (ValueError, TypeError):
                t["turn_id"] = i
            turns.append(t)
    return turns

# ────────────────────────────────────────────── 6. 레코드 구축
def build_records(turns: List[Dict]) -> List[Dict]:
    by_dlg = collections.defaultdict(list)
    for t in turns:
        by_dlg[(t["annotation_type"], t["dialogue_id"])].append(t)

    recs: List[Dict] = []
    for (_atype, _did), tl in by_dlg.items():
        tl.sort(key=lambda x: x["turn_id"])
        prev = {}
        for tr in tl:
            cur = _extract_turn_slots(tr)

            add  = {k: v for k, v in cur.items() if prev.get(k) != v}
            dele = {k: None for k in prev if k not in cur}

            if add and not dele:
                act = "inform"
            elif dele and not add:
                act = "delete"
            elif add or dele:
                act = "update"
            else:
                act = "noop"

            recs.append({
                "state_before": dict(prev),
                "action": act,
                "state_after": dict(cur),
                "annotation_type": tr["annotation_type"],
                "dialogue_id": tr["dialogue_id"],
                "turn_id": tr["turn_id"],
                "utterance": tr.get("utterance", ""),
            })
            prev = cur
    return recs

# ────────────────────────────────────────────── 7. CLI
def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True, help="Human annotation file path")
    ap.add_argument("--llm",   required=True, help="LLM annotation file path")
    ap.add_argument("--export", required=True, help="Output file path")
    args = ap.parse_args()

    print(f"Loading human annotations from {args.human} …")
    human = load_turns(pathlib.Path(args.human), "human")
    print(f"Loading LLM annotations from {args.llm} …")
    llm   = load_turns(pathlib.Path(args.llm),   "llm")

    all_turns = human + llm
    print(f"Collected {len(all_turns):,} turns → building slot map …")

    # ① 모든 원본 슬롯 이름 수집
    raw_names: Set[str] = set()
    for t in all_turns:
        raw_names.update(_raw_slot_names(t))

    # ② 동적 매핑 생성 & 전역 등록
    global _DYNAMIC_MAP
    _DYNAMIC_MAP = _build_dynamic_map(raw_names)

    # ③ 레코드 생성
    recs = build_records(all_turns)

    # ④ raw → 최종 Canon 매핑(json 출력용)
    raw_to_final = {name: canon(name) for name in raw_names}

    out = {"annotations": recs, "canonical_map": raw_to_final}
    output_path = pathlib.Path(args.export)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✔ {args.export} written — {len(recs):,} records "
          f"(human={len(human)}, llm={len(llm)})")

if __name__ == "__main__":
    _cli()
