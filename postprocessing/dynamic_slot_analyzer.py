#!/usr/bin/env python3
"""
Dynamic Slot Analyzer — strict double-check merge
------------------------------------------------
• edit distance AND semantic similarity가 모두 임계값을 만족할 때만 슬롯 병합
• 입력·출력·CLI 형식은 기존과 동일
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
    s = _SPLIT.sub("_", s.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def canon(slot: str) -> str:
    b = _canon_basic(slot)
    return _DYNAMIC_MAP.get(b, b)

# ────────────────────────────── 원본 슬롯 수집 ──────────────────────────────
def _raw_slot_names(turn: Dict) -> List[str]:
    res = []
    if isinstance(turn.get("slots"), dict):
        res.extend(turn["slots"].keys())
    for f in turn.get("frames", []):
        res.extend(f.get("state", {}).get("slot_values", {}).keys())
    if isinstance(turn.get("state"), dict):
        for dom, m in turn["state"].items():
            if isinstance(m, dict):
                res.extend(f"{dom}_{k}" for k in m.keys())
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
        suf_a = a.split('_')[-1]
        for j in range(i + 1, len(basics)):
            b = basics[j]
            suf_b = b.split('_')[-1]

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
    out: Dict[str, str] = {}
    # direct "slots"
    if isinstance(turn.get("slots"), dict):
        out.update({canon(k): v for k, v in turn["slots"].items()})
    # MultiWOZ frames
    for f in turn.get("frames", []):
        sv = f.get("state", {}).get("slot_values", {})
        for k, vals in sv.items():
            key = canon(k)
            if isinstance(vals, list) and vals:
                val = vals[-1]
                if isinstance(val, dict) and "value" in val:
                    val = val["value"]
                out[key] = val
            elif isinstance(vals, str):
                out[key] = vals
    # Schema/SGD state
    if isinstance(turn.get("state"), dict):
        for dom, m in turn["state"].items():
            if not isinstance(m, dict):
                continue
            for raw, val in m.items():
                if isinstance(val, dict) and "value" in val:
                    val = val["value"]
                if isinstance(val, list) and val:
                    last = val[-1]
                    val = last["value"] if isinstance(last, dict) and "value" in last else last
                key = canon(f"{dom}_{raw}")
                out[key] = val
    return out

# ────────────────────────────── 파일 로드 ──────────────────────────────
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
        dturns = dlg.get("dialogue") or dlg.get("turns") or []
        for i, t in enumerate(dturns):
            t = dict(t)
            t["dialogue_id"] = did
            t["annotation_type"] = ann_type
            try:
                t["turn_id"] = int(t.get("turn_id", i))
            except (ValueError, TypeError):
                t["turn_id"] = i
            turns.append(t)
    return turns

# ────────────────────────────── 레코드 생성 ──────────────────────────────
def build_records(turns: List[Dict]) -> List[Dict]:
    by = collections.defaultdict(list)
    for t in turns:
        by[(t["annotation_type"], t["dialogue_id"])].append(t)

    recs: List[Dict] = []
    for key, tl in by.items():
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

# ────────────────────────────── CLI ──────────────────────────────
def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--llm",   required=True)
    ap.add_argument("--export", required=True)
    args = ap.parse_args()

    human = load_turns(pathlib.Path(args.human), "human")
    llm   = load_turns(pathlib.Path(args.llm),   "llm")
    all_turns = human + llm

    raw: Set[str] = set()
    for t in all_turns:
        raw.update(_raw_slot_names(t))

    global _DYNAMIC_MAP
    _DYNAMIC_MAP = _build_dynamic_map(raw)

    recs = build_records(all_turns)
    raw_to_final = {n: canon(n) for n in raw}

    out = {"annotations": recs, "canonical_map": raw_to_final}
    out_path = pathlib.Path(args.export)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✔ {args.export} written  ({len(recs):,} records)")

if __name__ == "__main__":
    _cli()
