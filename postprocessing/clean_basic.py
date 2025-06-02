# clean_basic.py  —  value wrapping (extended) + 구조 자동 탐색
# -----------------------------------------------------------------
# 사용 예:
#   python clean_basic.py --input raw.json --output step1.json
# -----------------------------------------------------------------
import re, json, argparse, pathlib, sys
from typing import Dict, Any, List

# ---------- 패턴 ----------
PHONE_PAT  = re.compile(r"\d{11,}$")
REF_PAT    = re.compile(r"(?=.*[A-Za-z])[A-Za-z0-9]{6,}$")
DATE_ISO   = re.compile(r"\d{4}-\d{2}-\d{2}$")
DATE_SLASH = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$")
NUMBER_PAT = re.compile(r"^\d+(\.\d+)?$")
TIME_24H   = re.compile(r"^(\d{1,2}):(\d{2})$")
TIME_12H   = re.compile(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)$", re.I)
MONEY_PAT  = re.compile(r"^[£$€]\s?\d+(\.\d+)?$")
WDAY_PAT   = re.compile(r"^(next|this)?\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$", re.I)
ENTITY_HINT= re.compile(r"\b(hotel|restaurant|inn|lodge|guest house|b&b|cafe|bar)\b", re.I)
DELIM_RE   = re.compile(r"[|,]")

WDAY_LIST = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

# ---------- 헬퍼 ----------
def snake_case(s: str) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return re.sub(r"_+", "_", s).lower().strip("_")

def _wrap_single(v: str) -> Dict[str, Any]:
    v = v.strip()
    # 시간
    if TIME_24H.fullmatch(v):
        h, m = map(int, TIME_24H.match(v).groups())
        return {"value": h*60 + m, "slot_type": "time_min"}
    if TIME_12H.fullmatch(v):
        h, m, ap = TIME_12H.match(v).groups()
        h = int(h) % 12 + (12 if ap.lower()=="pm" else 0)
        m = int(m or 0)
        return {"value": h*60 + m, "slot_type": "time_min"}
    # 요일
    if WDAY_PAT.fullmatch(v):
        day = WDAY_PAT.match(v).group(2).lower()
        idx = WDAY_LIST.index(day)
        return {"value": idx, "slot_type": "weekday"}
    # 숫자
    if NUMBER_PAT.fullmatch(v):
        return {"value": float(v), "slot_type": "numeric"}
    # 날짜
    if DATE_ISO.fullmatch(v):
        return {"value": v, "slot_type": "date_iso"}
    if DATE_SLASH.fullmatch(v):
        d, mth, y = map(int, DATE_SLASH.match(v).groups())
        if y < 100: y += 2000
        return {"value": f"{y:04d}-{mth:02d}-{d:02d}", "slot_type": "date_iso"}
    # 돈
    if MONEY_PAT.fullmatch(v):
        amt = float(re.sub(r"[£$€\s]", "", v))
        return {"value": amt, "slot_type": "numeric"}
    # 전화/참조
    if PHONE_PAT.fullmatch(v):
        return {"value": "<phone_any>", "slot_type": "string"}
    if REF_PAT.fullmatch(v):
        return {"value": "<ref_any>", "slot_type": "string"}
    # 엔티티
    if ENTITY_HINT.search(v) or (len(v) > 20 and " " in v):
        return {"value": "<entity_any>", "slot_type": "string"}
    # 그 외
    return {"value": v, "slot_type": "string"}

def wrap_value(raw: str) -> Dict[str, Any]:
    if DELIM_RE.search(raw):
        parts = [p.strip() for p in DELIM_RE.split(raw) if p.strip()]
        return {"value": [_wrap_single(p)["value"] for p in parts],
                "slot_type": "list_string"}
    return _wrap_single(raw)

def clean_state(state: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for k, v in state.items():
        k_std = snake_case(k)
        if isinstance(v, dict) and "value" in v:
            cleaned[k_std] = v
        else:
            cleaned[k_std] = wrap_value(str(v))
    return cleaned

def ensure_turn_fields(turn: Dict[str, Any]):
    turn.setdefault("reward", {"score": 0, "justification": "auto-added"})
    turn.setdefault("transition", {
        "slots_added": {}, "slots_removed": [],
        "goal_progress": "unchanged",
        "is_valid_transition": True,
        "justification": "auto-added"
    })

# ---------- JSON 구조 유틸 ----------
def iter_turns(dialogue: Any):
    """
    Yields every turn dict regardless of dialogue structure:
    - {"annotation": {turn_id: turn_dict, ...}}
    - {"turns": [turn_dict, ...]}
    - direct list of turns
    """
    if isinstance(dialogue, dict):
        if "annotation" in dialogue and isinstance(dialogue["annotation"], dict):
            yield from dialogue["annotation"].values()
        elif "turns" in dialogue and isinstance(dialogue["turns"], list):
            yield from dialogue["turns"]
    elif isinstance(dialogue, list):
        yield from dialogue
    else:
        return

# ---------- main ----------
def main(src: pathlib.Path, dst: pathlib.Path):
    data_raw = json.load(src.open())
    # 데이터 리스트 통일
    dialogues: List[Any]
    if isinstance(data_raw, list):
        dialogues = data_raw
    elif isinstance(data_raw, dict):
        dialogues = list(data_raw.values())
    else:
        sys.exit("Unsupported JSON top-level structure")

    for dlg in dialogues:
        for turn in iter_turns(dlg):
            if not isinstance(turn, dict):
                continue
            if "state_before" in turn:
                turn["state_before"] = clean_state(turn["state_before"])
            if "state_after" in turn:
                turn["state_after"] = clean_state(turn["state_after"])
            ensure_turn_fields(turn)

    json.dump(dialogues, dst.open("w"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    main(pathlib.Path(args.input), pathlib.Path(args.output))
