from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT=Path(__file__).resolve().parents[2]
TARGETS={
 "linear":lambda x:0.8*x[:,0]-0.5*x[:,1],
 "additive":lambda x:np.sin(np.pi*x[:,0])+0.5*x[:,1]**2,
 "interaction":lambda x:1.5*x[:,0]*x[:,1],
 "waves":lambda x:np.sin(np.pi*x[:,0])*np.cos(np.pi*x[:,1]),
 "radial":lambda x:np.exp(-5*((x[:,0]-0.25)**2+(x[:,1]+0.2)**2)),
 "kink":lambda x:np.maximum(0,x[:,0]+x[:,1])-0.4*x[:,0]
}


def synthetic(target, seed, n=1000, noise=0.05):
    data={"task":"regression","outputs":1,"inputs":2}
    streams=np.random.SeedSequence(seed).spawn(4)
    for split,stream,size in zip(["train","val","inside","outside"],streams,[n,400,1000,1000]):
        rng=np.random.default_rng(stream)
        if split=="outside":
            chunks=[]
            while sum(len(x) for x in chunks)<size:
                x=rng.uniform(-1.5,1.5,(size,2));chunks.append(x[(np.abs(x)>1).any(1)])
            x=np.concatenate(chunks)[:size]
        else: x=rng.uniform(-1,1,(size,2))
        truth=TARGETS[target](x).reshape(-1,1)
        y=truth+rng.normal(0,noise,truth.shape)
        data[split]=(x.astype("float32"),y.astype("float32"),truth.astype("float32"))
    data["split_ids"]={k:hashlib.sha256(data[k][0].tobytes()).hexdigest() for k in ["train","val","inside","outside"]}
    return data


def fetch_yeast():
    dest=ROOT/"data/raw"; dest.mkdir(parents=True,exist_ok=True)
    path=dest/"yeast.data"
    if not path.exists():
        archive=dest/"yeast.zip"
        urllib.request.urlretrieve("https://archive.ics.uci.edu/static/public/110/yeast.zip",archive)
        with zipfile.ZipFile(archive) as z:
            for name in ["yeast.data","yeast.names"]: (dest/name).write_bytes(z.read(name))
    return path


def biological(name,seed):
    if name=="yeast":
        rows=[line.split() for line in fetch_yeast().read_text().splitlines() if line.strip()]
        x=np.array([r[1:9] for r in rows],dtype=float)
        y=LabelEncoder().fit_transform([r[-1] for r in rows]); ids=np.array([r[0] for r in rows])
    elif name=="breast_cancer":
        ds=load_breast_cancer();x,y=ds.data,ds.target;ids=np.arange(len(y)).astype(str)
    else: raise ValueError(name)
    # 60/20/20 stratified split. Rare Yeast classes remain included.
    train,hold=train_test_split(np.arange(len(y)),test_size=0.4,stratify=y,random_state=seed)
    val,test=train_test_split(hold,test_size=0.5,stratify=y[hold],random_state=seed+10000)
    scaler=StandardScaler().fit(x[train])
    xx=scaler.transform(x).astype("float32")
    data={"task":"classification","outputs":int(y.max()+1),"inputs":x.shape[1],"split_ids":{}}
    for split,index in [("train",train),("val",val),("inside",test)]:
        data[split]=(xx[index],y[index].astype("int64"),y[index].astype("int64"))
        data["split_ids"][split]=ids[index].tolist()
    data["preprocessing"]={"mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist()}
    return data
