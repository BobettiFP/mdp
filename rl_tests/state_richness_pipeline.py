#!/usr/bin/env python3
"""
State-Richness pipeline  rev 3  (NumPy scalar → Python scalar 변환 포함)
"""
import argparse, json, random, os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.stats import entropy

# --------------------- util --------------------------------------------------
def _to_py(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x

def vecs(recs: List[dict], slot2i: Dict[str,int], val2i: Dict[str,int]):
    dim=len(slot2i)+len(val2i); arr=[]
    for r in recs:
        v=np.zeros(dim,dtype=np.int8)
        for k in r["state_after"]: v[slot2i[k]]=1
        for vstr in map(str, r["state_after"].values()):
            v[len(slot2i)+val2i[vstr]]=1
        arr.append(v)
    return np.vstack(arr)

def richness(X, slot2i, val2i):
    uniq=len(np.unique(X,axis=0))
    ent=entropy(list(Counter(map(bytes,X)).values()))
    cov=uniq/len(X)
    pca=PCA().fit(X)
    eff=(pca.explained_variance_ratio_.cumsum()>=0.95).argmax()+1
    dens=sep=0.0
    if uniq>10:
        km=KMeans(n_clusters=min(10,uniq//2),n_init=5,random_state=0).fit(X)
        dens=silhouette_score(X,km.labels_)
    if uniq>1:
        smp=np.unique(X,axis=0)[:1500]
        sep=float(np.linalg.norm(smp[:,None]-smp,axis=-1).mean())
    return dict(unique_states=uniq,state_entropy=ent,coverage_ratio=cov,
                effective_dim=eff,density_uniformity=dens,cluster_separation=sep,
                slot_vocab=len(slot2i),value_vocab=len(val2i))

def tsne_fig(X,title,path):
    if len(X)<3: return
    try: emb=TSNE(n_components=2,random_state=0,perplexity=min(30,len(X)//3)).fit_transform(X)
    except: emb=PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(5,4)); plt.scatter(emb[:,0],emb[:,1],s=6,alpha=.6)
    plt.title(title); plt.axis("off"); plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()

# --------------------- main --------------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--annotations",required=True)
    ap.add_argument("--outdir",default="mdp/rl_tests/state_richness_results")
    a=ap.parse_args(); Path(a.outdir).mkdir(parents=True,exist_ok=True)

    data=json.load(open(a.annotations))
    recs=data["annotations"] if isinstance(data,dict) else data
    H=[r for r in recs if r.get("annotation_type")=="human"]
    L=[r for r in recs if r.get("annotation_type")=="llm"]
    if not H or not L: raise RuntimeError("annotation_type 라벨이 부족합니다.")

    slotH={s:i for i,s in enumerate(sorted({k for r in H for k in r["state_after"]}))}
    valH ={v:i for i,v in enumerate(sorted({str(x) for r in H for x in r["state_after"].values()}))}
    slotL={s:i for i,s in enumerate(sorted({k for r in L for k in r["state_after"]}))}
    valL ={v:i for i,v in enumerate(sorted({str(x) for r in L for x in r["state_after"].values()}))}

    Xh,Xl=vecs(H,slotH,valH),vecs(L,slotL,valL)
    Rh,Rl=richness(Xh,slotH,valH),richness(Xl,slotL,valL)

    tsne_fig(Xh,"Human",Path(a.outdir)/"state_space_human.png")
    tsne_fig(Xl,"LLM",  Path(a.outdir)/"state_space_llm.png")

    # bar plot
    feat=["unique_states","coverage_ratio","effective_dim"]
    plt.figure(figsize=(5,3))
    x=np.arange(len(feat));w=.35
    plt.bar(x-w/2,[Rh[f] for f in feat],w,label="Human")
    plt.bar(x+w/2,[Rl[f] for f in feat],w,label="LLM")
    plt.xticks(x,feat); plt.legend(); plt.tight_layout()
    plt.savefig(Path(a.outdir)/"richness_compare.png",dpi=300); plt.close()

    out=Path(a.outdir)/"state_richness_metrics.json"
    out.write_text(json.dumps({"human":{k:_to_py(v) for k,v in Rh.items()},
                               "llm":  {k:_to_py(v) for k,v in Rl.items()}},
                              indent=2,ensure_ascii=False))
    print("✔ saved →",out)

if __name__=="__main__":
    np.random.seed(0); random.seed(0); main()
