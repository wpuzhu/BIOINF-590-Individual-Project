"""Wrappers for published deep-learning domain methods (GPU recommended).

All wrappers have the workflow signature f(ctx, k, seed) -> labels, build a fresh
AnnData from ctx.counts / ctx.coords (so they also work on count-split and
subsampled replicates), and cluster the learned embedding with a Gaussian
mixture (tied covariance, the analogue of mclust 'EEE' used by both papers'
tutorials). This avoids R/mclust in Colab.

  GraphST   Long et al., Nat Commun 2023   (pip: GraphST, POT)
  STAGATE   Dong & Zhang, Nat Commun 2022  (git: QIFEIDKN/STAGATE_pyG; torch_geometric)
  SpaGCN    Hu et al., Nat Methods 2021    (pip: SpaGCN; CPU is fine)

A wrapper that fails to import or run raises; workflow.run_members records the
error and continues with the remaining methods.
"""
import numpy as np


def _adata(ctx):
    import anndata as ad
    import scipy.sparse as sp
    a = ad.AnnData(X=sp.csr_matrix(ctx.counts.astype(np.float32)))
    a.var_names = [str(g) for g in (ctx.genes or range(ctx.counts.shape[1]))]
    a.var_names_make_unique()
    a.obsm["spatial"] = np.asarray(ctx.coords, float)
    return a


def _gmm(emb, k, seed, n_pcs=20):
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    z = PCA(min(n_pcs, emb.shape[1]), random_state=seed).fit_transform(emb) if emb.shape[1] > n_pcs else emb
    return GaussianMixture(k, covariance_type="tied", n_init=3, random_state=seed).fit_predict(z)


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def graphst(ctx, k, seed):
    from GraphST import GraphST
    a = _adata(ctx)
    model = GraphST.GraphST(a, device=_device(), random_seed=int(seed) + 41)
    a = model.train()
    return _gmm(np.asarray(a.obsm["emb"]), k, seed)


def stagate(ctx, k, seed):
    import scanpy as sc
    import STAGATE_pyG
    a = _adata(ctx)
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=min(3000, a.n_vars - 1))
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    STAGATE_pyG.Cal_Spatial_Net(a, model="KNN", k_cutoff=6, verbose=False)
    a = STAGATE_pyG.train_STAGATE(a, random_seed=int(seed), device=_device(), verbose=False)
    return _gmm(np.asarray(a.obsm["STAGATE"]), k, seed)


def spagcn(ctx, k, seed):
    import random
    import torch
    import scanpy as sc
    import SpaGCN as spg
    a = _adata(ctx)
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    if ctx.cluster_genes is not None:
        a = a[:, np.asarray(ctx.cluster_genes)].copy()
    a.X = np.asarray(a.X.toarray() if hasattr(a.X, "toarray") else a.X, dtype=np.float32)  # SpaGCN calls .A on sparse
    x, y = ctx.coords[:, 0].tolist(), ctx.coords[:, 1].tolist()
    adj = spg.calculate_adj_matrix(x=x, y=y, histology=False)
    l = spg.search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    clf = spg.SpaGCN(); clf.set_l(l)
    clf.train(a, adj, init_spa=True, init="kmeans", n_clusters=k, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, _ = clf.predict()
    return np.asarray(y_pred)


GPU_METHODS = {"GraphST": graphst, "STAGATE": stagate, "SpaGCN": spagcn}


def available(names=None, ctx=None, k=5):
    """Import-check each method (and optionally a tiny smoke run) and return the working subset."""
    ok, problems = {}, {}
    for n in (names or GPU_METHODS):
        try:
            if n == "GraphST":
                import GraphST  # noqa: F401
                import ot  # noqa: F401
            elif n == "STAGATE":
                import STAGATE_pyG  # noqa: F401
            elif n == "SpaGCN":
                import SpaGCN  # noqa: F401
            ok[n] = GPU_METHODS[n]
        except Exception as e:
            problems[n] = f"{type(e).__name__}: {e}"
    return ok, problems
