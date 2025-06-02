#!/usr/bin/env python3
import json, argparse, pathlib, collections

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--slot_vocab", default="slot_vocab.json")
ap.add_argument("--token_vocab", default="token_vocab.json")
ap.add_argument("--token_top", type=int, default=1000)
args = ap.parse_args()

slot_counter, tok_counter = collections.Counter(), collections.Counter()

def iter_turns(dlg):
    if isinstance(dlg, dict) and "annotation" in dlg:
        yield from dlg["annotation"].values()
    elif isinstance(dlg, dict) and "turns" in dlg:
        yield from dlg["turns"]
    elif isinstance(dlg, list):
        yield from dlg
    else:
        return

for dlg in json.load(open(args.input, encoding="utf8")):
    for turn in iter_turns(dlg):
        for st in ("state_before", "state_after"):
            for k, v in turn.get(st, {}).items():
                slot_counter[k] += 1
                tok = v["value"]
                # list_string 형태는 원소마다 토큰 빈도 증가
                if isinstance(tok, list):
                    tok_counter.update(tok)
                else:
                    tok_counter[tok] += 1

json.dump([s for s, _ in slot_counter.most_common()],
          open(args.slot_vocab, "w"), ensure_ascii=False, indent=2)
json.dump(["<unk>"] + [t for t, _ in tok_counter.most_common(args.token_top)],
          open(args.token_vocab, "w"), ensure_ascii=False, indent=2)
print("saved:", args.slot_vocab, args.token_vocab)
