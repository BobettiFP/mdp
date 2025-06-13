#!/usr/bin/env python3
"""
Transition Dynamics → Learning Stability  rev 4
⚙️  모든 key 구성요소를 str로 캐스팅해 unhashable 오류 제거
"""
import argparse, json, math, random
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# ------------------------- helpers ------------------------------------------
def load_recs(p: Path):
    data=json.load(p.open())
    recs=data["annotations"] if isinstance(data,dict) else data
    H=[r for r in recs if r.get("annotation_type")=="human"]
    L=[r for r in recs if r.get("annotation_type")=="llm"]
    if not H or not L:
        raise RuntimeError("annotation_type 라벨이 부족합니다.")
    return H,L

def _state_tuple(sd: dict)->tuple:
    """dict → tuple of sorted (slot, str(value))"""
    return tuple(sorted((k, str(v)) for k,v in sd.items()))

def build_trans(recs):
    C=Counter()
    for r in recs:
        sb=_state_tuple(r["state_before"])
        sa=_state_tuple(r["state_after"])
        act=str(r.get("action","noop"))
        C[(sb, act, sa)]+=1
    return C

# ------------------------- metrics ------------------------------------------
def metrics(C: Counter):
    total=sum(C.values())
    consistency=max(C.values())/total

    by_sa=defaultdict(list)
    for (s,a,ns),f in C.items():
        by_sa[(s,a)].append(f)

    ent_norm=[]
    for freqs in by_sa.values():
        if len(freqs)==1:
            ent_norm.append(0.0)            # 완전 예측 가능
        else:
            ent_norm.append(
                entropy(freqs,base=2)/math.log2(len(freqs))
            )
    predictability=1.0-np.mean(ent_norm) if ent_norm else 0.0
    stability=total/len(by_sa)              # 전이 당 average freq

    return {"consistency":consistency,
            "predictability":predictability,
            "stability":stability}

# ------------------------- plotting -----------------------------------------
def barplot(h,ll,out,title):
    labels=list(h.keys()); x=np.arange(len(labels)); w=.35
    plt.figure(figsize=(5,3))
    plt.bar(x-w/2,[h[k] for k in labels],w,label="Human")
    plt.bar(x+w/2,[ll[k] for k in labels],w,label="LLM")
    plt.xticks(x,labels); plt.ylim(0,1); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()

# ------------------------- main ---------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--annotations",required=True)
    ap.add_argument("--outdir",default="mdp/rl_tests/transition_dynamics_results")
    args=ap.parse_args(); outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)

    Hrec,Lrec=load_recs(Path(args.annotations))
    Ch,Cl=build_trans(Hrec),build_trans(Lrec)
    print(f"[human] usable transitions: {sum(Ch.values()):,} | states: {len({s for (s,_,_) in Ch})}")
    print(f"[llm]   usable transitions: {sum(Cl.values()):,} | states: {len({s for (s,_,_) in Cl})}")

    Mh,Ml=metrics(Ch),metrics(Cl)
    adv={k: Ml[k]-Mh[k] for k in Mh}

    (outdir/"transition_dynamics_metrics.json").write_text(
        json.dumps({"human_metrics":Mh,"llm_metrics":Ml,"dynamics_advantage":adv},indent=2)
    )
    barplot(Mh,Ml,outdir/"dynamics_compare.png","Transition-Dynamics Comparison")
    print("✔ 결과 저장:",outdir)

if __name__=="__main__":
    np.random.seed(0); random.seed(0); main()
