"""
pipeline_v2.py — the PAPER's literal architecture. DROP-IN replacement for pipeline.py.
=======================================================================================
    import pipeline_v2 as P          # instead of: import pipeline as P
...and every cell of run_replication.ipynb works unchanged: same `Config`, same `run()`,
same `run_rank_sweep()`, same `compare_eval_modes()`, same outputs / metrics / tables /
figures.

ONLY the network differs. Data prep (`load_and_prepare`) and evaluation (`evaluate`) are
imported from pipeline.py rather than duplicated, so a v1-vs-v2 difference is purely
architectural.

v1 (pipeline.py)                          v2 (this file, paper §3.3)
---------------------------------------   ------------------------------------------
sample = ONE security's 8-qtr sequence    sample = the WHOLE cross-section, 8 qtrs
X: [batch, 8, F]                          X: [batch, 8, N*F]     <- N flattened into input
LSTM(F -> hidden) -> Linear(hidden, 3)    LSTM(N*F -> numcell) -> Linear(numcell, N*3)
output: [batch, 3]                        output: [batch, N, 3]  <- the paper's (N x 3)
N = batch dimension                       N = ARCHITECTURE dimension
weights shared across securities          one weight set per fund-window
~hundreds of samples per fund-window      ~13 samples per fund-window (!)

Paper quotes implemented here
-----------------------------
  "Each sample is represented as a three-dimensional tensor of shape (T,N,F) ... the
   number of outputs is N (equal to the number of distinct security identifiers retained
   in that window)"
  "The output of the recurrent layer is passed through a dense transformation, reshaped,
   and mapped to a probability grid of dimension (N x 3) via a softmax activation."
  "The hidden dimension, denoted numcell, scales with the size of the realized
   cross-section."
  time-step mask  : "At any quarter in which ... all securities in the retained output set
                     are padding indicators (sh_past = 0 ...), we mask the entire time step
                     from recurrence."
  feasibility mask: "If the security lacks continuous presence over the full eight-quarter
                     input horizon ... zero out the sell probability p_-1 and renormalize."
  naive baseline  : "the naive classifier predicts the max(frequency) class observed across
                     all securities and time steps."

Honest caveats
--------------
1. Heavily over-parameterised. N=25, F=20 -> LSTM input 500 wide, head emits N*3=75
   (~50k+ params) trained on only ~13 sequences per window (20 train quarters minus the
   8-quarter lookback). N=100 -> ~840k params on 13 samples. Expect overfitting and
   likely WEAKER results than v1. That is a property of the paper's design, not a bug.
   Use `max_stock` to cap N.
2. PyTorch's nn.LSTM has no true *recurrent* dropout (its `dropout` arg only applies
   between stacked layers), so "dropout and recurrent dropout, each 0.25" is approximated
   with input + output dropout.
3. `model_mode="global"` is not applicable here (N is fund-specific); it is accepted and
   ignored with a note, so the notebook's comparison cell still runs.
"""
from __future__ import annotations
import os, json
from dataclasses import dataclass, asdict, replace
from typing import List
import numpy as np
import pandas as pd

import pipeline as V1
# reuse the shared layers verbatim -> identical data prep + identical evaluation
from pipeline import load_and_prepare, evaluate, compare_eval_modes, build_sequences  # noqa: F401


# ============================================================ CONFIG (same name & API)
@dataclass
class Config(V1.Config):
    """Everything from pipeline.Config, plus the paper-architecture knobs."""
    # Cap on N (retained security identifiers per fund-window). The paper ties the cap
    # "dynamically to the realized cross-sectional size in that window"; None = every
    # distinct security held in the window. Lower it to tame the parameter count.
    max_stock: int = None
    numcell_cap: int = 128        # numcell = clip(N, 16, numcell_cap)
    min_samples: int = 6          # skip a fund-window with fewer sequences than this
    val_frac: float = 0.2         # last X% of TRAIN sequences (chronological) = validation


ConfigV2 = Config          # alias


# ============================================================ TENSOR BUILDER
def build_window_tensors(fp: pd.DataFrame, feat: List[str], cfg: Config,
                         tr_lo: int, te_hi: int):
    """Build the paper's (T, N, F) samples for ONE fund-window.
    Returns X [S,T,N,F], held_seq [S,T,N], step_mask [S,T], feas [S,N],
            y [S,N] ({0,1,2}, -1 = ignore), qi_lab [S], secs (len N) — or None.
    """
    seq = cfg.seq_len
    qmin, qmax = tr_lo - seq + 1, te_hi - 1
    sub = fp[(fp["qi"] >= qmin) & (fp["qi"] <= qmax)]
    if sub.empty:
        return None
    # retained security set = "distinct security identifiers retained in that window"
    if cfg.max_stock is not None and sub["security"].nunique() > cfg.max_stock:
        keep = (sub.groupby("security", observed=True)["weight"].mean()
                   .sort_values(ascending=False).head(cfg.max_stock).index)
        sub = sub[sub["security"].isin(keep)]
    secs = list(pd.unique(sub["security"]))
    N, F = len(secs), len(feat)
    if N == 0:
        return None
    Q = qmax - qmin + 1
    si = {s: i for i, s in enumerate(secs)}

    G = np.zeros((Q, N, F), dtype=np.float32)
    held = np.zeros((Q, N), dtype=np.float32)
    Yg = np.full((Q, N), np.nan, dtype=np.float32)
    pq = (sub["qi"].to_numpy() - qmin).astype(int)
    ps = sub["security"].map(si).to_numpy().astype(int)
    G[pq, ps, :] = sub[feat].to_numpy(dtype="float32", na_value=0.0)
    held[pq, ps] = 1.0
    Yg[pq, ps] = sub["Y"].to_numpy(dtype="float32", na_value=np.nan)

    labs = [t for t in range(tr_lo, te_hi) if (t - qmin) >= seq - 1]
    if not labs:
        return None
    idx = [t - qmin for t in labs]
    X = np.stack([G[i - seq + 1:i + 1] for i in idx])              # [S,T,N,F]
    hs = np.stack([held[i - seq + 1:i + 1] for i in idx])          # [S,T,N]
    step_mask = (hs.sum(axis=2) > 0).astype(np.float32)            # [S,T] all-padding -> 0
    feas = (hs.min(axis=1) > 0)                                    # [S,N] present all T steps
    ylab = np.stack([Yg[i] for i in idx])
    hlab = np.stack([held[i] for i in idx])
    y = np.where((hlab > 0) & ~np.isnan(ylab), ylab + 1, -1).astype(np.int64)
    return X, hs, step_mask, feas, y, np.array(labs), secs


# ============================================================ MODEL
def _make_panel_lstm(N, F, numcell, dropout):
    import torch.nn as nn

    class PanelLSTM(nn.Module):
        """(T, N*F) -> LSTM -> dense -> reshape (N, 3): the paper's design."""
        def __init__(self):
            super().__init__()
            self.N = N
            self.in_drop = nn.Dropout(dropout)      # stands in for recurrent dropout
            self.lstm = nn.LSTM(N * F, numcell, batch_first=True)
            self.out_drop = nn.Dropout(dropout)
            self.head = nn.Linear(numcell, N * 3)

        def forward(self, x, step_mask):
            x = x * step_mask.unsqueeze(-1)         # mask fully-synthetic quarters
            o, _ = self.lstm(self.in_drop(x))
            h = self.out_drop(o[:, -1, :])
            return self.head(h).view(-1, self.N, 3)
    return PanelLSTM()


def _train_predict_v2(X, hs, step_mask, feas, y, labs, secs, tr, te, cfg, dev):
    """Fit one PanelLSTM on the train sequences; return test predictions as rows."""
    import torch
    S, T, N, F = X.shape
    Xtr, htr = X[tr], hs[tr].astype(bool)
    flat, pres = Xtr.reshape(-1, F), htr.reshape(-1)
    if pres.sum() < 20:
        return None
    # standardize on TRAIN using only real (held) entries; re-zero padding afterwards
    mu, sd = flat[pres].mean(0), flat[pres].std(0) + 1e-6
    Xz = ((X - mu) / sd).astype(np.float32) * hs[..., None]
    Xf = Xz.reshape(S, T, N * F)

    numcell = int(np.clip(N, 16, cfg.numcell_cap))       # "scales with the cross-section"
    model = _make_panel_lstm(N, F, numcell, cfg.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-1)   # -1 = padded / unlabelled slot

    Xt = torch.from_numpy(Xf); yt = torch.from_numpy(y); mt = torch.from_numpy(step_mask)
    tr_i = np.where(tr)[0]
    nval = int(len(tr_i) * cfg.val_frac)                 # chronological split (time series)
    use_val = nval >= 1 and len(tr_i) - nval >= 2
    trn_i, val_i = (tr_i[:-nval], tr_i[-nval:]) if use_val else (tr_i, tr_i[:0])

    def _loss(idx):
        lg = model(Xt[idx].to(dev), mt[idx].to(dev))
        return lossf(lg.reshape(-1, 3), yt[idx].reshape(-1).to(dev))

    best, best_state, bad = 1e9, None, 0
    for _ in range(cfg.max_epochs):
        model.train()
        for b in range(0, len(trn_i), cfg.batch):
            bi = trn_i[b:b + cfg.batch]
            opt.zero_grad(); _loss(bi).backward(); opt.step()
        if use_val:
            model.eval()
            with torch.no_grad():
                vl = float(_loss(val_i))
            if vl < best - 1e-4:
                best, best_state, bad = vl, {k: v.detach().cpu().clone()
                                             for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    te_i = np.where(te)[0]
    if len(te_i) == 0:
        return None
    with torch.no_grad():
        P = torch.softmax(model(Xt[te_i].to(dev), mt[te_i].to(dev)), dim=2).cpu().numpy()

    # naive: max(frequency) class across ALL securities and time steps of the train window
    ytr = y[tr]; ytr = ytr[ytr >= 0]
    naive = float(np.bincount(ytr, minlength=3).argmax() - 1) if ytr.size else 0.0

    rows = []
    for k, s in enumerate(te_i):
        p = P[k].copy()
        p[~feas[s], 0] = 0.0                      # feasibility: can't sell what you don't hold
        p = p / p.sum(1, keepdims=True)
        valid = np.where(y[s] >= 0)[0]
        if valid.size == 0:
            continue
        rows.append(pd.DataFrame({
            "security": [secs[j] for j in valid],
            "qi": int(labs[s]),
            "y_pred": p[valid].argmax(1) - 1,
            "p_sell": p[valid, 0], "p_hold": p[valid, 1], "p_buy": p[valid, 2],
            "feasible": feas[s][valid],
            "y_naive": naive,
        }))
    return pd.concat(rows, ignore_index=True) if rows else None


# ============================================================ PER-FUND DRIVER
def _fund_task_v2(fund, fp, feat, cfg, dev="cpu"):
    V1._set_threads(1)
    ql = np.array(sorted(fp["qi"].unique()))
    if len(ql) < cfg.window_q:
        return []
    outs = []
    for tr_lo, tr_hi, te_lo, te_hi in V1._windows(int(ql[0]), int(ql[-1]), cfg):
        t = build_window_tensors(fp, feat, cfg, tr_lo, te_hi)
        if t is None:
            continue
        X, hs, sm, feas, y, labs, secs = t
        tr = (labs >= tr_lo) & (labs < tr_hi)
        te = (labs >= te_lo) & (labs < te_hi)
        if tr.sum() < cfg.min_samples or te.sum() == 0:
            continue
        out = _train_predict_v2(X, hs, sm, feas, y, labs, secs, tr, te, cfg, dev)
        if out is not None and len(out):
            out["fund"] = fund
            outs.append(out)
    return outs


def run_model(panel: pd.DataFrame, cfg: Config, verbose=True):
    """Walk-forward, one PanelLSTM per fund-window. Same output schema as
    pipeline.run_model, so pipeline.evaluate() works unchanged."""
    import torch
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = V1._device(cfg)
    feat = [f for f in cfg.features if f in panel.columns]
    ncore = os.cpu_count() or 1
    njobs = ncore if cfg.n_jobs in (-1, 0) else cfg.n_jobs
    n_funds = int(panel["fund"].nunique())
    if cfg.model_mode == "global":
        print("[model-v2] note: the paper architecture is inherently per fund-window "
              "(N is fund-specific) -> 'global' not applicable; running per_fund.")
    if verbose:
        print(f"[model-v2] paper (T,N,F)->(N x 3) | device={dev} F={len(feat)} "
              f"funds={n_funds} cores={ncore}")

    preds = []
    if dev != "cpu" or njobs == 1 or cfg.parallel_backend == "serial":
        V1._set_threads(cfg.torch_threads or ncore)
        for i, (f, fp) in enumerate(V1._iter_funds(panel), 1):
            preds += _fund_task_v2(f, fp, feat, cfg, dev)
            if verbose and i % 50 == 0:
                print(f"  [v2] {i}/{n_funds} funds")
    else:
        from concurrent.futures import ThreadPoolExecutor
        V1._set_threads(1)
        if verbose:
            print(f"  [v2] {n_funds} funds on {njobs} THREADS (shared memory, low RAM)")

        def _wrk(a):
            f, ix = a
            return _fund_task_v2(f, panel.take(ix), feat, cfg, "cpu")
        done = 0
        with ThreadPoolExecutor(max_workers=njobs) as ex:
            for outs in ex.map(_wrk, V1._fund_index_order(panel)):
                preds += outs; done += 1
                if verbose and done % 100 == 0:
                    print(f"  [v2] {done}/{n_funds} funds done")
    if not preds:
        raise RuntimeError("v2 produced no predictions -- window too short, or raise max_stock "
                           "/ lower min_samples")
    out = pd.concat(preds, ignore_index=True)
    # attach the evaluation columns (Y, returns, weight, rank, inv_type) from the panel
    cols = [c for c in ["fund", "security", "qi", "yq", "Y", "fwd_1q", "fwd_2q", "fwd_3q",
                        "inv_type", "weight", "rank"] if c in panel.columns]
    return out.merge(panel[cols], on=["fund", "security", "qi"], how="left")


# ============================================================ ORCHESTRATE (same API)
def run(cfg: Config = None, verbose=True):
    """Identical contract to pipeline.run() — returns
    {panel, predictions, metrics, tables, figures, config}."""
    cfg = cfg or Config()
    if cfg.save_outputs:
        os.makedirs(cfg.out_dir, exist_ok=True)
    panel = load_and_prepare(cfg)
    if verbose:
        bal = panel.dropna(subset=["Y"])["Y"].value_counts(normalize=True).round(3).to_dict()
        print(f"[data] rows={len(panel):,} funds={panel.fund.nunique()} "
              f"quarters={panel.qi.nunique()} max_rank={panel.attrs.get('eff_rank')} "
              f"class_balance={bal}")
    preds = run_model(panel, cfg, verbose=verbose)
    metrics, tables = evaluate(preds, cfg)
    figs = V1._figures(metrics, cfg)
    clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
    clean["architecture"] = "v2_paper_(T,N,F)->(Nx3)"
    if verbose:
        print("\n=== PREDICTABILITY (v2 = paper architecture) ===")
        print(f"  LSTM  precision: pooled {clean['lstm_precision_pooled']:.3f} | "
              f"fund-avg {clean['lstm_precision_fundavg']:.3f}   (paper 0.71)")
        print(f"  Naive precision: pooled {clean['naive_precision_pooled']:.3f} | "
              f"fund-avg {clean['naive_precision_fundavg']:.3f}   (paper 0.52)")
        print(f"\n  [eval_mode={clean['eval_mode']} | return={clean['eval_return']} | "
              f"sort_lag={clean['eval_sort_lag']}]")
        print(f"  Table XII Q1-Q5 = {clean['tableXII_Q1mQ5']*100:+.2f}%/qtr "
              f"(t={clean['tableXII_Q1mQ5_t']:.2f})   (paper +1.06%, t=5.74)")
        print(f"  Table XI  corr-incorr = {clean['correct_minus_incorrect']*100:+.2f}%/qtr "
              f"(t={clean['correct_minus_incorrect_t']:.2f})   (paper -0.23%, t=-12.4)")
    if cfg.save_outputs:
        json.dump(clean, open(f"{cfg.out_dir}/metrics.json", "w"), indent=2)
        for nm, tb in tables.items():
            tb.to_csv(f"{cfg.out_dir}/{nm}.csv", index=False)
        preds.to_parquet(f"{cfg.out_dir}/predictions.parquet", index=False)
    return {"panel": (panel if cfg.keep_panel else None), "predictions": preds,
            "metrics": clean, "tables": tables, "figures": figs, "config": asdict(cfg)}


def run_rank_sweep(cfg: Config = None, ranks=(10, 25), verbose=True):
    """Same API as pipeline.run_rank_sweep, but trains the v2 architecture."""
    cfg = cfg or Config()
    rows, out = [], {}
    for r in ranks:
        if verbose:
            print(f"\n{'='*20} max_rank = {r} {'='*20}")
        res = run(replace(cfg, max_rank=r, out_dir=f"{cfg.out_dir}_rank{r}"), verbose=verbose)
        m = res["metrics"]
        rows.append({"max_rank": r, "n_funds": m["n_funds"], "n_pred": m["n_predictions"],
                     "LSTM_prec": m["lstm_precision_fundavg"], "naive_prec": m["naive_precision_fundavg"],
                     "LSTM_minus_naive": m["lstm_precision_fundavg"] - m["naive_precision_fundavg"],
                     "XII_Q1mQ5": m["tableXII_Q1mQ5"], "XII_t": m["tableXII_Q1mQ5_t"],
                     "XI_diff": m["correct_minus_incorrect"], "XI_t": m["correct_minus_incorrect_t"]})
        out[r] = res
    return pd.DataFrame(rows), out


def compare_v1_v2(cfg: Config = None, verbose=True):
    """Train BOTH architectures on the same panel + same evaluation -> any difference is
    purely architectural."""
    cfg = cfg or Config()
    r2 = run(cfg, verbose=verbose)
    cfg_v1 = V1.Config(**{k: v for k, v in asdict(cfg).items()
                          if k in V1.Config.__dataclass_fields__})
    r1 = V1.run(cfg_v1, verbose=verbose)
    rows = []
    for nm, r in (("v1 (shared weights, per security)", r1),
                  ("v2 (paper, (T,N,F)->(Nx3))", r2)):
        m = r["metrics"]
        rows.append({"architecture": nm, "n_pred": m["n_predictions"], "n_funds": m["n_funds"],
                     "LSTM_prec": m["lstm_precision_fundavg"],
                     "naive_prec": m["naive_precision_fundavg"],
                     "LSTM_minus_naive": m["lstm_precision_fundavg"] - m["naive_precision_fundavg"],
                     "XII_Q1mQ5": m["tableXII_Q1mQ5"], "XII_t": m["tableXII_Q1mQ5_t"]})
    return pd.DataFrame(rows), {"v1": r1, "v2": r2}


if __name__ == "__main__":
    run(Config())
