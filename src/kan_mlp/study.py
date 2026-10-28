"""Validation-only selection and stopping, independent final evaluation."""
import argparse
import copy
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, mean_squared_error
from .models import MODELS,architecture,build
from .data import ROOT,TARGETS,synthetic,biological


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def tensor_split(data,key):
    x,y,_=data[key]
    return torch.from_numpy(x),torch.from_numpy(y)


def loss_fn(pred,y,task):
    return torch.nn.functional.cross_entropy(pred,y) if task=="classification" else (pred-y).square().mean()


def fit(kind,dims,data,cfg,seed,lr):
    seed_all(seed)
    model=build(kind,dims,cfg["terms"])
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=cfg["weight_decay"])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=cfg["lr_patience"],threshold=cfg["min_delta"],threshold_mode="rel")
    tx,ty=tensor_split(data,"train");vx,vy=tensor_split(data,"val")
    best=float("inf");reference=float("inf");stale=0;history=[];checkpoint=None;best_epoch=0
    start=time.perf_counter();reason="epoch_limit"
    for epoch in range(1,cfg["max_epochs"]+1):
        model.train();opt.zero_grad();loss=loss_fn(model(tx),ty,data["task"])
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);opt.step()
        if epoch%cfg["check_every"] and epoch!=cfg["max_epochs"]: continue
        model.eval()
        with torch.no_grad(): val=float(loss_fn(model(vx),vy,data["task"]))
        if not np.isfinite(val): raise FloatingPointError("Nonfinite validation loss")
        scheduler.step(val)
        history.append({"epoch":epoch,"seconds":time.perf_counter()-start,"train_loss":float(loss.detach()),"val_loss":val,"lr":opt.param_groups[0]["lr"]})
        if val<best:
            best=val;checkpoint=copy.deepcopy(model.state_dict());best_epoch=epoch
        if val < reference*(1-cfg["min_delta"]): reference=val;stale=0
        else: stale+=1
        if stale>=cfg["patience"]:
            reason="validation_plateau";break
    seconds=time.perf_counter()-start
    if checkpoint is None: raise RuntimeError("No validation checkpoint")
    model.load_state_dict(checkpoint);model.eval()
    return model,{"val_loss":best,"epochs":epoch,"best_epoch":best_epoch,"stop_reason":reason,"fit_seconds":seconds,"lr":lr,"history":history}


def evaluate(model,data):
    results={}
    for split in ["inside","outside"]:
        if split not in data: continue
        x,y,truth=data[split];xx=torch.from_numpy(x)
        with torch.no_grad():
            for _ in range(3): model(xx)
            times=[]
            for _ in range(10):
                start=time.perf_counter(); pred=model(xx);times.append(time.perf_counter()-start)
            pred=pred.numpy()
        metric={"predict_us_per_sample":float(np.median(times)*1e6/len(x))}
        if data["task"]=="regression":
            metric.update(rmse=float(np.sqrt(mean_squared_error(y,pred))),clean_rmse=float(np.sqrt(mean_squared_error(truth,pred))))
        else:
            probs=torch.softmax(torch.from_numpy(pred),-1).numpy();labels=probs.argmax(1)
            metric.update(accuracy=float(accuracy_score(y,labels)),balanced_accuracy=float(balanced_accuracy_score(y,labels)),macro_f1=float(f1_score(y,labels,labels=np.arange(data["outputs"]),average="macro",zero_division=0)),log_loss=float(log_loss(y,probs,labels=np.arange(data["outputs"]))))
        results[split]=metric
    return results


def run(cfg,out):
    torch.set_num_threads(cfg.get("threads",1));torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    config_path=out/"config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError("Output directory contains a different config; use a new directory")
    config_path.write_text(json.dumps(cfg,indent=2))
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,stderr=subprocess.DEVNULL,text=True).strip()
    except subprocess.CalledProcessError: commit=None
    metadata={"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"platform":platform.platform(),"processor":platform.processor(),"threads":torch.get_num_threads(),"device":"cpu","commit":commit}
    (out/"environment.json").write_text(json.dumps(metadata,indent=2))
    rows=[]
    for target in cfg["datasets"]:
      for seed in cfg["seeds"]:
        data=synthetic(target,seed,cfg["n_train"],cfg["noise"]) if target in TARGETS else biological(target,seed)
        (out/f"split_{target}_{seed}.json").write_text(json.dumps({k:data[k] for k in ["split_ids","preprocessing"] if k in data},indent=2))
        for depth in cfg["depths"]:
          for budget in cfg["budgets"]:
            # Rotate execution order by seed to reduce systematic ordering bias.
            order=cfg["models"];r=seed%len(order);order=order[r:]+order[:r]
            for kind in order:
                dims,p=architecture(kind,data["inputs"],data["outputs"],depth,budget,cfg["terms"])
                error=abs(p-budget)/budget
                if error>cfg["budget_tolerance"]:
                    raise ValueError(f"Cannot match {kind} to {budget} within tolerance: {p}")
                ident=f"{target}_{seed}_{depth}_{budget}_{kind}"
                saved=out/f"{ident}.json"
                if saved.exists() and (out/f"{ident}.pt").exists():
                    rows.append(json.loads(saved.read_text()));continue
                candidates=[];winner=None;selected=None;cost=0
                for lr in cfg["learning_rates"]:
                    model,record=fit(kind,dims,data,cfg,seed+100000,lr)
                    cost+=record["fit_seconds"];candidates.append(record)
                    if selected is None or record["val_loss"]<selected["val_loss"]:
                        winner=model;selected=record
                # Test data first consulted after validation-only selection.
                result={"dataset":target,"seed":seed,"depth":depth,"budget":budget,"model":kind,"dims":dims,"parameters":p,"budget_relative_error":error,"search_seconds":cost,"selected":selected,"candidates":candidates,"metrics":evaluate(winner,data)}
                ident=f"{target}_{seed}_{depth}_{budget}_{kind}"
                torch.save({"dims":dims,"kind":kind,"terms":cfg["terms"],"state_dict":winner.state_dict()},out/f"{ident}.pt")
                (out/f"{ident}.json").write_text(json.dumps(result,indent=2));rows.append(result)
                print(ident,selected["stop_reason"],selected["epochs"],flush=True)
    (out/"results.json").write_text(json.dumps(rows,indent=2))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--out",required=True)
    args=parser.parse_args();run(json.loads(Path(args.config).read_text()),args.out)

if __name__=="__main__":main()
