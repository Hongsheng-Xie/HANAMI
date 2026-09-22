#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HANAMI_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = HANAMI_ROOT / "results/clinical_concordance/score_generation"
if str(HANAMI_ROOT) not in sys.path:
    sys.path.insert(0, str(HANAMI_ROOT))

from utils_our_bio_allgene import prepare_data
from base_gcn import GCN_binary_SAGE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0,10,20,30,40,50,60,70,80,90]
DISE_NUM, DRUG_NUM, GENE_NUM = 694,1272,4519
TOTAL = DISE_NUM + DRUG_NUM + GENE_NUM
TARGETS = [
    ("Cariprazine-DRD2-Depressive Disorder", (162,1507,3372), 18891),
    ("Propranolol-ADRB2-Prostatic Neoplasms", (2,961,3910), 26436),
    ("Lasofoxifene-ESR1-Breast Neoplasms", (6,1907,2038), 2296),
    ("Prednisolone-NR3C1-Endometriosis", (1,1142,2047), 2998),
    ("Diazepam-GABRA1-Seizures", (112,1121,4504), 33823),
    ("Esmolol-ADRB1-Hypertension", (93,729,1983), 147),
]


def log(x):
    print(x,flush=True)


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b=f.read(8*1024*1024)
            if not b: break
            h.update(b)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def args_for(seed,epochs):
    return SimpleNamespace(
        data_name="ms",task="binary",input_dir=str(HANAMI_ROOT/"data/ms")+os.sep,
        test_ratio=0.1,val_ratio=0.1,seed=seed,lr=0.001,weight_decay=0.00001,
        hidden_channels=256,batch_size=5000,batch_num=10,epoch_num=epochs,
        tau=1000.0,lam=0.1,abla_edge=False,abla_basic=False,
        dise_num=DISE_NUM,drug_num=DRUG_NUM,gene_num=GENE_NUM,
    )


def npair_loss(z1,z2,margin=1.0):
    a=F.normalize(z1,dim=1)
    p=F.normalize(z2,dim=1)
    sim=torch.matmul(a,p.T)
    positive=torch.diag(sim)
    diff=sim-positive.unsqueeze(1)
    diff.masked_fill_(torch.eye(z1.size(0),device=z1.device).bool(),float("-inf"))
    loss=torch.log1p(torch.exp(diff).sum(dim=1)).mean()
    if margin>0:
        loss=loss+margin*positive.mean()
    return loss


def logits(model,h,edges,args):
    feat=model.pred(h,edges)
    if not args.abla_edge:
        feat=torch.cat((feat,model.pooling2(h,edges)),dim=1)
    return model.classifier(feat)


def state_cpu(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def auc_margin(pos_logits,neg_logits):
    p=(pos_logits[:,1]-pos_logits[:,0]).detach().cpu().numpy()
    n=(neg_logits[:,1]-neg_logits[:,0]).detach().cpu().numpy()
    y=np.r_[np.ones(len(p),dtype=np.int8),np.zeros(len(n),dtype=np.int8)]
    return float(roc_auc_score(y,np.r_[p,n]))


def train_epoch(model,opt,criterion,x,edge_index,pos,neg,args):
    model.train()
    adjmask=torch.ones_like(edge_index[0],dtype=torch.bool,device=DEVICE)
    total=0.0
    for batch_id in range(args.batch_num):
        opt.zero_grad(set_to_none=True)
        perm=torch.randperm(len(edge_index[0]))
        adjmask[perm[:args.batch_size]] = False
        h=model(x,edge_index)
        hm=model(x,edge_index[:,adjmask])
        loss=args.lam*npair_loss(h,hm)
        start=len(pos)//args.batch_num*batch_id
        end=len(pos)//args.batch_num*(batch_id+1)
        lp=logits(model,h,pos[start:end],args)
        ln=logits(model,h,neg[start:end],args)
        lpm=logits(model,hm,pos[start:end],args)
        lnm=logits(model,hm,neg[start:end],args)
        labels=torch.cat((
            torch.ones(lp.size(0),dtype=torch.long,device=DEVICE),
            torch.zeros(ln.size(0),dtype=torch.long,device=DEVICE),
        ))
        loss=loss+criterion(torch.cat((lp,ln)),labels)
        if args.lam>0:
            loss=loss+criterion(torch.cat((lpm,lnm)),labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss in batch {batch_id}")
        loss.backward()
        opt.step()
        total+=float(loss.detach())
    return total/args.batch_num


def assert_targets_unseen(poslist,neglist,pool):
    targets={tuple(pool[idx].tolist()) for _,_,idx in TARGETS}
    for collection_name,collection in (("positive",poslist),("negative",neglist)):
        for split_id,split in enumerate(collection):
            for motif,rows in split.items():
                got={tuple(x) for x in rows.detach().cpu().numpy().tolist()}
                overlap=targets & got
                if overlap:
                    raise RuntimeError(f"targets leaked into {collection_name} split={split_id} motif={motif}: {overlap}")


def tie_rank(scores,idx):
    target=scores[idx]
    ties=int(np.count_nonzero(scores==target))
    rank=1.0+int(np.count_nonzero(scores>target))+0.5*(ties-1)
    return rank,100.0*rank/len(scores),ties,float(target)


def run_seed(seed,epochs,pool,pool_t,out,score_batch):
    started=time.time()
    args=args_for(seed,epochs)
    data,poslist,neglist=prepare_data(args)
    data=data.to(DEVICE)
    if tuple(data.x.shape)!=(TOTAL,1792):
        raise RuntimeError(f"expected final HANAMI features {(TOTAL,1792)}, got {tuple(data.x.shape)}")
    if tuple(data.train_graph.shape)[0]!=2:
        raise RuntimeError(f"unexpected graph shape {tuple(data.train_graph.shape)}")
    assert_targets_unseen(poslist,neglist,pool)

    # Mirror the final paper script's seed reset and its discarded outer model
    # initialization before the clique-specific model is instantiated.
    set_seed(seed)
    discarded=GCN_binary_SAGE(data.x.shape[1],args.hidden_channels,args.hidden_channels,TOTAL,args).to(DEVICE)
    del discarded
    model=GCN_binary_SAGE(data.x.shape[1],args.hidden_channels,args.hidden_channels,TOTAL,args).to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    criterion=torch.nn.CrossEntropyLoss()

    key="clique"
    edge_index=data.train_graph
    pos=poslist[0][key]
    neg=neglist[0][key]
    val_pos=poslist[1][key]
    val_neg=neglist[1][key]

    best_auc,best_epoch,best_state=-math.inf,-1,None
    for epoch in range(epochs):
        loss=train_epoch(model,optimizer,criterion,data.x,edge_index,pos,neg,args)
        model.eval()
        with torch.no_grad():
            h=model(data.x,edge_index)
            va=auc_margin(logits(model,h,val_pos,args),logits(model,h,val_neg,args))
        if va>best_auc:
            best_auc,best_epoch,best_state=va,epoch,state_cpu(model)
        if epoch==0 or (epoch+1)%10==0:
            log(f"seed={seed} epoch={epoch+1}/{epochs} loss={loss:.6f} val_auc={va:.6f} best={best_auc:.6f}@{best_epoch+1}")

    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    scores=np.empty(len(pool),dtype=np.float64)
    with torch.no_grad():
        h=model(data.x,edge_index)
        for start in range(0,len(pool),score_batch):
            trip=pool_t[start:start+score_batch].to(DEVICE)
            out_logits=logits(model,h,trip,args)
            scores[start:start+len(trip)]=(out_logits[:,1]-out_logits[:,0]).cpu().numpy()
    if not np.isfinite(scores).all():
        raise RuntimeError("non-finite pool scores")

    checkpoint=out/f"checkpoint_seed{seed}.pt"
    torch.save({
        "model_state_dict":best_state,
        "seed":seed,
        "best_epoch_zero_based":best_epoch,
        "best_validation_auc":best_auc,
        "args":vars(args),
    },checkpoint)
    np.save(out/f"scores_seed{seed}.npy",scores)
    targets=[]
    for name,trip,idx in TARGETS:
        if tuple(pool[idx])!=trip:
            raise RuntimeError(f"target index mismatch {name}")
        rank,pct,ties,score=tie_rank(scores,idx)
        targets.append({
            "target":name,"global_triplet":list(trip),"pool_index_zero_based":idx,
            "pool_size":len(pool),"score":score,"rank":rank,"top_percent":pct,"ties":ties,
        })
    meta={
        "method":"HANAMI","seed":seed,"epochs":epochs,
        "best_epoch_zero_based":best_epoch,"best_epoch_one_based":best_epoch+1,
        "best_validation_auc":best_auc,"runtime_seconds":time.time()-started,
        "score":"class-1 logit margin z1-z0",
        "rank_formula":"1 + count(score>target) + 0.5*(count(score==target)-1)",
        "top_percent_formula":"100*rank/46704; lower is better",
        "checkpoint_rule":"earliest strict maximum validation AUROC",
        "data_loader":"utils_our_bio_allgene.prepare_data",
        "feature_shape":list(data.x.shape),
        "feature_files":["dise_All.pth","drug_All.pth","gene_All.pth"],
        "architecture":"GCN_binary_SAGE: SAGEConv(max), SAGEConv(sum), node and edge pooling",
        "optimizer":{"name":"Adam","lr":args.lr,"weight_decay":args.weight_decay},
        "contrastive_loss":"paper npair_loss, lambda=0.1",
        "edge_masking":"cumulative across 10 minibatches, matching final paper runner",
        "message_passing_graph":"data.train_graph as coded in final paper runner",
        "discarded_outer_initialization_mirrored":True,
        "score_file":str(out/f"scores_seed{seed}.npy"),
        "checkpoint_file":str(checkpoint),
        "targets":targets,
    }
    with open(out/f"meta_seed{seed}.json","w",encoding="utf8") as f:
        json.dump(meta,f,indent=2)
    return meta


def collate(out,seeds):
    rows=[]
    for seed in seeds:
        p=out/f"meta_seed{seed}.json"
        if not p.exists(): continue
        meta=json.load(open(p,encoding="utf8"))
        for x in meta["targets"]:
            rows.append({
                "method":"HANAMI","seed":seed,"target":x["target"],
                "rank":x["rank"],"top_percent":x["top_percent"],"score":x["score"],"ties":x["ties"],
                "best_epoch_one_based":meta["best_epoch_one_based"],
                "best_validation_auc":meta["best_validation_auc"],
                "runtime_seconds":meta["runtime_seconds"],
            })
    if not rows: return [],[]
    with open(out/"hanami_per_seed.csv","w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for name,trip,idx in TARGETS:
        rr=[x for x in rows if x["target"]==name]
        vals=np.array([x["top_percent"] for x in rr],dtype=float)
        ranks=np.array([x["rank"] for x in rr],dtype=float)
        summary.append({
            "method":"HANAMI","target":name,"n_seeds":len(rr),
            "median_top_percent":float(np.median(vals)),
            "q1_top_percent":float(np.percentile(vals,25)),
            "q3_top_percent":float(np.percentile(vals,75)),
            "mean_top_percent":float(np.mean(vals)),
            "sd_top_percent":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0,
            "min_top_percent":float(np.min(vals)),"max_top_percent":float(np.max(vals)),
            "median_rank":float(np.median(ranks)),
            "q1_rank":float(np.percentile(ranks,25)),
            "q3_rank":float(np.percentile(ranks,75)),
        })
    with open(out/"hanami_ten_seed_summary.csv","w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    return rows,summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    ap.add_argument("--seeds",default=",".join(map(str,SEEDS)))
    ap.add_argument("--epochs",type=int,default=150)
    ap.add_argument("--score-batch",type=int,default=2048)
    ap.add_argument("--force",action="store_true")
    ap.add_argument("--smoke",action="store_true")
    cli=ap.parse_args()
    seeds=[int(x) for x in cli.seeds.split(",") if x.strip()]
    epochs=cli.epochs
    out=cli.output_dir.resolve()
    if cli.smoke:
        seeds=seeds[:1];epochs=min(epochs,2);out=out/"smoke"
    out.mkdir(parents=True,exist_ok=True)
    if DEVICE.type!="cuda":
        raise RuntimeError("CUDA is required")
    probe=torch.randn(256,256,device=DEVICE);_=probe@probe;torch.cuda.synchronize()
    pool_path=HANAMI_ROOT/"data/ms/tup_cenofgene.npy"
    pool=np.load(pool_path,allow_pickle=False).astype(np.int64,copy=False)
    if pool.shape!=(46704,3) or len(np.unique(pool,axis=0))!=46704:
        raise RuntimeError(f"unexpected pool {pool.shape}")
    pool_t=torch.from_numpy(pool)
    for name,trip,idx in TARGETS:
        if tuple(pool[idx])!=trip: raise RuntimeError(f"target mismatch {name}")
    manifest={
        "status":"running","hanami_root":str(HANAMI_ROOT),"output_dir":str(out),
        "runner":str(Path(__file__).resolve()),"seeds":seeds,"epochs":epochs,
        "pool_file":str(pool_path),"pool_sha256":sha256_file(pool_path),"pool_size":len(pool),
        "torch_version":torch.__version__,"cuda_runtime":torch.version.cuda,
        "gpu":torch.cuda.get_device_name(0),"data_loader":"utils_our_bio_allgene.prepare_data",
        "paper_runner":str(HANAMI_ROOT/"main.py"),
        "provisional_extractor":"extract_repurposing_candidates.py (historical reference; not used)",
        "notes":[
            "Provisional extractor was not used.",
            "Exact full pool is byte-identical to the four-baseline pool.",
            "Only the clique binary classifier is trained because it is first and independent in the paper's seven-motif loop.",
        ],
    }
    with open(out/"run_manifest.json","w",encoding="utf8") as f:json.dump(manifest,f,indent=2)
    for seed in seeds:
        paths=[out/f"meta_seed{seed}.json",out/f"scores_seed{seed}.npy",out/f"checkpoint_seed{seed}.pt"]
        if not cli.force and all(p.exists() for p in paths):
            log(f"SKIP seed={seed}: complete outputs exist")
            continue
        log(f"START seed={seed} epochs={epochs} device={DEVICE}")
        meta=run_seed(seed,epochs,pool,pool_t,out,cli.score_batch)
        line=" | ".join(f"{x['target'].split('-')[0]}={x['top_percent']:.3f}%" for x in meta["targets"])
        log(f"DONE seed={seed} best_epoch={meta['best_epoch_one_based']} val_auc={meta['best_validation_auc']:.6f} seconds={meta['runtime_seconds']:.1f} :: {line}")
        collate(out,seeds)
        torch.cuda.empty_cache()
    rows,summary=collate(out,seeds)
    manifest["status"]="complete" if len({x["seed"] for x in rows})==len(seeds) else "partial"
    manifest["completed_seed_count"]=len({x["seed"] for x in rows})
    with open(out/"run_manifest.json","w",encoding="utf8") as f:json.dump(manifest,f,indent=2)
    log(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
