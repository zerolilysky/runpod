"""
pipeline_v2.py — Mimicking Finance replication, the PAPER's architecture.
=========================================================================
Fully self-contained end-to-end pipeline: one holdings parquet in, predictability +
portfolio sorts out. No external pulls, no dependency on any other module.

    import pipeline_v2 as P
    cfg = P.Config(data_path="...parquet")
    res = P.run(cfg)

Architecture (paper §3.3) — one LSTM per fund-window over the WHOLE cross-section:

    X: [batch, T=8, N*F]  ->  LSTM(N*F -> numcell)  ->  Linear(numcell, N*3)
                          ->  reshape [batch, N, 3] -> softmax

PANEL LAYOUT — top-N AT THE LABEL QUARTER, ALIGNED BY SECURITY IDENTITY
----------------------------------------------------------------------
For each label quarter t: take the fund's **top-`max_rank` securities as ranked at t**,
give each one a column (column 0 = largest at t), then walk the SAME security back through
t-7 ... t. A quarter where that security was not held is padding (`sh_past = 0`).

    N        = max_rank                    -> fixed width, never blows up with turnover
    column j = ONE security, tracked over its own 8-quarter history
    padding  = that security absent in that quarter (or fewer than N holdings)

This satisfies both of the paper's (mutually inconsistent) descriptions where they agree,
and is forced by the feasibility mask, which only makes sense if a column IS a security:

  §3.1 "one row per quarter and one column slot per rank (id) ... ids 76-100 are empty on
        purpose - referred to as padding. This gives us a consistent width across time."
        -> width = max_rank, ordered by rank.  [we order columns by rank AT t]
  §3.3 "the number of outputs is N (equal to the number of distinct security identifiers
        retained in that window)"
        -> columns are security identifiers.   [we track identities, not slots]
  §3.3 "For a given identifier ... if THE SECURITY lacks CONTINUOUS PRESENCE over the full
        eight-quarter input horizon (i.e., if any period exhibits sh_past = 0), we ... zero
        out the sell probability"
        -> decisive: continuous presence of a *security* is only definable if a column
           follows one security through time.

(A pure rank-slot reading — column j = whoever is rank j each quarter — would make each
column a chimera of different stocks and would render "the security lacks continuous
presence" meaningless, so it is not used.)

Within one sample the N columns are a FIXED set of N securities; across samples (different
label quarters) the set changes, because each label quarter picks its own top-N.

OTHER PAPER DETAILS IMPLEMENTED
-------------------------------
  "The output of the recurrent layer is passed through a dense transformation, reshaped,
   and mapped to a probability grid of dimension (N x 3) via a softmax activation."
  "The hidden dimension, denoted numcell, scales with the size of the realized
   cross-section."                              -> hidden = clip(N, 16, hidden_cap)
  time-step mask  : "At any quarter in which ... all securities in the retained output set
                     are padding indicators (sh_past = 0 ...), we mask the entire time step
                     from recurrence."
  feasibility mask: "If the security lacks continuous presence over the full eight-quarter
                     input horizon ... zero out the sell probability p_-1 and renormalize."
  naive baseline  : "the naive classifier predicts the max(frequency) class observed across
                     all securities and time steps."   -> computed PER FUND per window.

NO LOOK-AHEAD
-------------
* Target Y_t = sign of the t -> t+1 share change (a label, never a feature).
* Every feature is observable at quarter t (own past trades via `pdsh`; peer rates lagged).
* Forward returns are used ONLY for evaluation, and `eval_mode` controls their timing.
* Rolling windows split chronologically; the scaler is fit on train rows only.

HONEST CAVEATS
--------------
1. This design is heavily over-parameterised: N=25,F=20 -> LSTM input 500 wide, head emits
   N*3=75 (~50k params) trained on only ~13 sequences per window (20 train quarters minus
   the 8-quarter lookback). Expect overfitting. Two levers: lower `max_rank` (N is the
   template width), or use `model_mode="global"`, which pools every fund's samples into
   one model per window (~13 x #funds samples) — only possible because N is fixed at
   max_rank, so all funds share the same (N*F) -> (N*3) shape.
2. torch's nn.LSTM has no true *recurrent* dropout (its `dropout` arg only applies between
   stacked layers), so "dropout and recurrent dropout, each 0.25" is approximated with
   input + output dropout.
"""
from __future__ import annotations
import os, json, warnings
from dataclasses import dataclass, field, asdict, replace
from typing import List
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================ CONFIG
@dataclass
class Config:
    # ---- data ----
    data_path: str = ("manager_holdings/master_batches_returnfiltered/"
                      "panel_holdings_All_Funds_filter_rank.parquet")
    out_dir: str = "outputs_v2"
    col_map: dict = field(default_factory=lambda: {
        "fund": "fund", "date": "date", "security": "security", "shares": "shares",
        "position_value": "position_value", "market_cap": "market_cap", "isUs": "isUs",
        "quarterly_ret": "quarterly_ret", "past_1q_ret": "past_1q_ret",
        "future_1q_ret": "future_1q_ret", "future_2q_ret": "future_2q_ret",
        "future_3q_ret": "future_3q_ret", "InvTypeCode": "inv_type",
        "portfolio_value": "portfolio_value", "weight": "weight", "rank": "rank",
        "n_holdings": "n_holdings",
    })
    # Keep only these InvTypeCode values (e.g. (401,)). None = all.
    # With >1 type surviving, InvTypeCode also becomes the CATEGORY for peer activity
    # rates and for benchmark-adjusting fund returns (the paper's Morningstar category).
    inv_type_codes: tuple = None

    # ---- evaluation timing --------------------------------------------------
    # accuracy(t) needs shares[t+1] -> OBSERVABLE at t+1, PUBLIC ~45-60 days later.
    #   "tradeable"       : accuracy(t), return t+2->t+3 (`future_3q_ret`) - clears filing delay
    #   "predictive"      : accuracy(t), return t+1->t+2 (`future_2q_ret`) - no overlap. DEFAULT
    #   "contemporaneous" : accuracy(t), return t->t+1   - OVERLAPS -> biased benchmark only
    #   "lagged"          : accuracy(t-1), return t->t+1 - clean but staler
    # Training is independent of this -> compare_eval_modes() gives all four from one run.
    eval_mode: str = "predictive"

    # ---- sample filters (paper §3.1) ----
    us_only: bool = True
    min_years: int = 7          # >7 calendar years of history
    min_holdings: int = 10      # >=10 securities per quarter (auto-capped at max_rank)
    max_rank: int = None        # keep rank <= max_rank. None = whatever the file has
    change_band: float = 0.01   # +-1% dead-band around zero share change

    # ---- rolling design (paper Fig 2) ----
    window_q: int = 28          # observation window
    train_q: int = 20           # train quarters within a window
    test_q: int = 8             # test quarters within a window
    seq_len: int = 8            # LSTM input sequence length (quarters)
    step: int = 8               # quarters to advance each window (1 = fully overlapping)

    # ---- model (paper architecture) ----
    # N is fixed at max_rank, so every fund has the SAME input
    # (N*F) and output (N*3) shape -> weights can be shared across funds.
    #   "per_fund" : one LSTM per fund-window (the paper's design; ~13 samples each)
    #   "global"   : one LSTM per window, pooled over all funds (~13 x #funds samples ->
    #                far less overfitting, but more memory: all funds' tensors per window)
    model_mode: str = "per_fund"
    # The paper calls this `numcell`: hidden = clip(N, 16, hidden_cap), i.e. it "scales
    # with the size of the realized cross-section". Same field name as pipeline.py so the
    # SAME notebook runs against either module.
    hidden_cap: int = 128
    dropout: float = 0.25
    max_epochs: int = 50        # paper caps at 50
    patience: int = 50          # paper's early-stopping patience
    lr: float = 3e-3
    batch: int = 64             # sequences per step (there are only ~13 per window)
    min_samples: int = 6        # skip a fund-window with fewer train sequences
    val_frac: float = 0.2       # last X% of TRAIN sequences (chronological) = validation
    device: str = "auto"        # "auto" | "cpu" | "cuda"
    seed: int = 42

    # ---- accepted for drop-in compatibility with pipeline.py, UNUSED here ----
    # (kept so the identical notebook/config runs against either module)
    hidden: int = 64             # v1 only: v2 derives hidden = clip(N, 16, hidden_cap)
    min_seq_per_fund: int = 120  # v1 only: v2 uses `min_samples` (panel-sequences, ~13/window)
    min_train_global: int = 2000 # v1 only: same
    numcell_cap: int = None      # alias for hidden_cap; if set, it wins

    def __post_init__(self):
        if self.numcell_cap is not None:      # allow the paper's own name
            self.hidden_cap = self.numcell_cap

    # ---- CPU performance / memory ----
    n_jobs: int = -1                     # funds in parallel (-1 = all cores). 1 = serial
    torch_threads: int = 0               # intra-op threads. 0 = auto
    parallel_backend: str = "threading"  # "threading" (SHARED memory, low RAM) | "serial"
    downcast: bool = True                # float32 + categoricals -> ~halves panel RAM
    keep_panel: bool = False             # include the (large) panel in run() output
    save_outputs: bool = True

    @property
    def features(self) -> List[str]:
        return ["weight", "w_lag1", "dw", "rank", "log_posval", "log_pv", "log_mktcap",
                "quarterly_ret", "past_1q_ret", "pdsh", "pdsh_sign", "pdsh_lag1",
                "sh_lag1", "sh_lag2", "sh_lag3",
                "peer_buy", "peer_sell", "peer_hold", "n_holdings", "fund_ret_l1"]


# ============================================================ DATA + FEATURES
def load_and_prepare(cfg: Config) -> pd.DataFrame:
    """Load the panel, build the target and all (lagged) features. Memory-lean:
    reads only needed columns, float32 throughout, prunes in place."""
    inv = {v: k for k, v in cfg.col_map.items()}
    want = ["fund", "date", "security", "shares", "position_value", "market_cap",
            "quarterly_ret", "past_1q_ret", "future_1q_ret", "future_2q_ret",
            "future_3q_ret", "inv_type", "portfolio_value", "weight", "rank",
            "n_holdings", "isUs"]
    want_raw = [inv[c] for c in want if c in inv]
    try:
        import pyarrow.parquet as _pq
        avail = set(_pq.ParquetFile(cfg.data_path).schema.names)
        use = [c for c in want_raw if c in avail]
        df = pd.read_parquet(cfg.data_path, columns=use or None)
    except Exception:
        df = pd.read_parquet(cfg.data_path)
    df = df.rename(columns={inv[c]: c for c in cfg.col_map.values() if inv[c] in df.columns})
    miss = [c for c in ["fund", "date", "security", "shares"] if c not in df.columns]
    if miss:
        raise ValueError(f"missing required columns after mapping: {miss}")

    df["date"] = pd.to_datetime(df["date"])
    df["yq"] = df["date"].dt.to_period("Q")
    if cfg.us_only and "isUs" in df.columns:
        df = df[df["isUs"].astype(bool)]
    if cfg.inv_type_codes is not None and "inv_type" in df.columns:
        codes = {str(c) for c in cfg.inv_type_codes}
        before = len(df)
        df = df[df["inv_type"].astype(str).isin(codes)]
        print(f"[data] InvTypeCode filter {sorted(codes)}: {before:,} -> {len(df):,} rows")
        if len(df) == 0:
            raise ValueError(f"no rows for InvTypeCode {sorted(codes)}")
    df = df.sort_values("date").drop_duplicates(["fund", "yq", "security"], keep="last")
    eff_rank = None
    if "rank" in df.columns:
        if cfg.max_rank is not None:
            df = df[df["rank"] <= cfg.max_rank]
        eff_rank = int(df["rank"].max()) if len(df) else 0
    df.drop(columns=[c for c in ("date", "isUs") if c in df.columns], inplace=True)

    # float32 up front -> derived features stay float32; pd.NA/nullable -> np.nan
    F32 = "float32" if cfg.downcast else "float64"
    raw_num = ["shares", "position_value", "market_cap", "quarterly_ret", "past_1q_ret",
               "future_1q_ret", "future_2q_ret", "future_3q_ret", "portfolio_value",
               "weight", "rank", "n_holdings"]
    for c in raw_num:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(F32) if c in df.columns \
            else np.array(np.nan, dtype=F32)
    df = df.sort_values(["fund", "security", "yq"]).reset_index(drop=True)
    g = df.groupby(["fund", "security"], sort=False)

    for k in (1, 2, 3):
        df[f"sh_lag{k}"] = g["shares"].shift(k).astype(F32)
    df["w_lag1"] = g["weight"].shift(1).astype(F32)
    df["dw"] = (df["weight"] - df["w_lag1"]).astype(F32)
    # PAST realised share change INTO quarter t (known at t) -- the paper's core signal
    df["pdsh"] = ((df["shares"] - df["sh_lag1"]) / (df["sh_lag1"].abs() + 1.0)).astype(F32)
    df["pdsh_sign"] = np.sign(df["pdsh"]).fillna(0.0).astype(F32)
    df["pdsh_lag1"] = g["pdsh"].shift(1).astype(F32)
    df["log_posval"] = np.log(df["position_value"].abs() + 1.0).astype(F32)
    df["log_pv"] = np.log(df["portfolio_value"].abs() + 1.0).astype(F32)
    df["log_mktcap"] = np.log(df["market_cap"].abs() + 1.0).astype(F32)

    # TARGET: sign of NEXT-quarter fractional share change (+-band). label, not a feature.
    sh_next = g["shares"].shift(-1)
    dsh = (sh_next - df["shares"]) / (df["shares"].abs() + 1.0)
    df["Y"] = np.select([dsh <= -cfg.change_band, dsh >= cfg.change_band],
                        [-1.0, 1.0], default=0.0).astype(F32)
    df.loc[dsh.isna(), "Y"] = np.nan
    # realised returns -- EVALUATION ONLY. Carry all three so every eval_mode works.
    df["fwd_1q"] = df["future_1q_ret"]                    # t   -> t+1
    df["fwd_2q"] = df["future_2q_ret"]                    # t+1 -> t+2 (after acc(t) known)
    df["fwd_3q"] = df["future_3q_ret"]                    # t+2 -> t+3 (after filing delay)

    # fund-level filters (min_holdings capped at the rank cutoff)
    mh = cfg.min_holdings
    if eff_rank and mh > eff_rank:
        print(f"[data] min_holdings {mh} > max rank {eff_rank} -> capped to {eff_rank}")
        mh = eff_rank
    cnt = df.groupby(["fund", "yq"])["security"].transform("count")
    df = df[cnt >= mh]
    nq = df.groupby("fund")["yq"].transform("nunique")
    df = df[nq >= cfg.min_years * 4]

    # Category Activity Rates (paper App. A), lagged one quarter, within InvTypeCode.
    lab = df.dropna(subset=["Y"])
    n_cat = df["inv_type"].nunique() if "inv_type" in df.columns else 1
    aggs = dict(peer_buy=("Y", lambda s: (s > 0).mean()),
                peer_sell=("Y", lambda s: (s < 0).mean()),
                peer_hold=("Y", lambda s: (s == 0).mean()))
    if "inv_type" in df.columns and n_cat > 1:
        rate = lab.groupby(["inv_type", "yq"], observed=True).agg(**aggs).sort_index()
        rate = rate.groupby(level=0).shift(1)                   # lag within category
        key = pd.MultiIndex.from_arrays([df["inv_type"].to_numpy(), df["yq"].to_numpy()])
        for col in ("peer_buy", "peer_sell", "peer_hold"):
            df[col] = rate[col].reindex(key).to_numpy(dtype=F32, na_value=np.nan)
        print(f"[data] peer rates computed within {n_cat} InvTypeCode categories")
    else:
        rate = lab.groupby("yq").agg(**aggs).sort_index().shift(1)
        for col in ("peer_buy", "peer_sell", "peer_hold"):
            df[col] = df["yq"].map(rate[col]).astype(F32)
        if "inv_type" in df.columns:
            print("[data] single InvTypeCode -> peer rates are market-wide "
                  "(no cross-sectional variation; pass several codes for category peers)")

    # fund past-quarter return proxy (reindex-map -> no merge/copy)
    contrib = (df["w_lag1"] * df["quarterly_ret"])
    fr = contrib.groupby([df["fund"], df["yq"]]).sum()
    fr_l1 = fr.groupby(level=0).shift(1)
    key = pd.MultiIndex.from_arrays([df["fund"].to_numpy(), df["yq"].to_numpy()])
    df["fund_ret_l1"] = fr_l1.reindex(key).to_numpy(dtype=F32, na_value=np.nan)

    qs = pd.PeriodIndex(sorted(df["yq"].unique()), freq="Q")
    df["qi"] = df["yq"].map({q: i for i, q in enumerate(qs)}).astype("int32")
    df["held"] = np.int8(1)

    feat = [f for f in cfg.features if f in df.columns]
    keep = set(["fund", "security", "yq", "qi", "held", "Y", "fwd_1q", "fwd_2q", "fwd_3q",
                "inv_type", "weight", "rank"] + feat)
    df.drop(columns=[c for c in df.columns if c not in keep], inplace=True)
    if cfg.downcast:
        df["fund"] = df["fund"].astype("category")
        df["security"] = df["security"].astype("category")
        if "inv_type" in df.columns:
            df["inv_type"] = df["inv_type"].astype("category")
    df.attrs["eff_rank"] = eff_rank
    return df


# ============================================================ TENSORS (T, N, F)
def build_window_tensors(fp: pd.DataFrame, feat: List[str], cfg: Config,
                         tr_lo: int, te_hi: int, n_slots: int):
    """The paper's (T, N, F) samples for ONE fund-window.

    For each label quarter t: columns = the fund's top-`n_slots` securities AS RANKED AT t
    (column 0 = largest at t). Each column then follows THAT SAME security back through
    t-7 ... t; quarters where it was not held are padding (sh_past = 0). Within a sample
    the N columns are therefore a FIXED set of N securities.

    Returns X [S,T,N,F], held_seq [S,T,N], step_mask [S,T], feas [S,N],
            y [S,N] ({0,1,2}, -1 = ignore/empty), qi_lab [S],
            sec_lab [S,N] (category codes, -1 = empty column), cats — or None.
    """
    seq, N, F = cfg.seq_len, int(n_slots), len(feat)
    qmin, qmax = tr_lo - seq + 1, te_hi - 1
    sub = fp[(fp["qi"] >= qmin) & (fp["qi"] <= qmax)]
    sub = sub[sub["rank"].notna()]
    if sub.empty:
        return None
    Q = qmax - qmin + 1

    # ---- dense lookup over every security seen in the window: [Q, M, .] ----
    s = sub["security"]
    if isinstance(s.dtype, pd.CategoricalDtype):
        cats, gcodes = s.cat.categories, s.cat.codes.to_numpy()
    else:
        cats = pd.Index(pd.unique(s))
        gcodes = s.map({c: i for i, c in enumerate(cats)}).to_numpy()
    uniq = pd.unique(gcodes)                       # global codes present in this window
    loc = {c: i for i, c in enumerate(uniq)}       # global code -> local column in lookup
    M = len(uniq)
    lidx = np.array([loc[c] for c in gcodes], dtype=int)
    q = (sub["qi"].to_numpy() - qmin).astype(int)

    Gall = np.zeros((Q, M, F), dtype=np.float32)
    Hall = np.zeros((Q, M), dtype=np.float32)
    Yall = np.full((Q, M), np.nan, dtype=np.float32)
    # to_numpy(na_value=...) turns pd.NA / nullable columns into plain floats
    Gall[q, lidx, :] = sub[feat].to_numpy(dtype="float32", na_value=0.0)
    Hall[q, lidx] = 1.0
    Yall[q, lidx] = sub["Y"].to_numpy(dtype="float32", na_value=np.nan)

    qi_arr, rk_arr = sub["qi"].to_numpy(), sub["rank"].to_numpy()
    labs = [t for t in range(tr_lo, te_hi) if (t - qmin) >= seq - 1]
    Xs, Hs, Ys, Cs, keep = [], [], [], [], []
    for t in labs:
        i = t - qmin
        m = qi_arr == t
        if not m.any():
            continue
        # this quarter's top-N securities, ordered by rank at t
        order = np.argsort(rk_arr[m], kind="stable")[:N]
        cols = lidx[m][order]                      # the FIXED security set for this sample
        k = len(cols)
        Xi = np.zeros((seq, N, F), dtype=np.float32)
        Hi = np.zeros((seq, N), dtype=np.float32)
        yi = np.full(N, -1, dtype=np.int64)
        ci = np.full(N, -1, dtype=np.int64)
        # same securities walked back through the 8-quarter horizon; gaps stay 0 (padding)
        Xi[:, :k, :] = Gall[i - seq + 1:i + 1, cols, :]
        Hi[:, :k] = Hall[i - seq + 1:i + 1, cols]
        yv = Yall[i, cols]
        ok = ~np.isnan(yv)
        yi[:k][ok] = (yv[ok] + 1).astype(np.int64)
        ci[:k] = uniq[cols]
        Xs.append(Xi); Hs.append(Hi); Ys.append(yi); Cs.append(ci); keep.append(t)
    if not Xs:
        return None
    X = np.stack(Xs)                                   # [S,T,N,F]
    hs = np.stack(Hs)                                  # [S,T,N]
    step_mask = (hs.sum(axis=2) > 0).astype(np.float32)  # [S,T] all-padding quarter -> 0
    feas = (hs.min(axis=1) > 0)                        # [S,N] THIS security held all T steps
    y = np.stack(Ys)                                   # [S,N]
    sec_lab = np.stack(Cs)                             # [S,N]
    return X, hs, step_mask, feas, y, np.array(keep), sec_lab, cats


# ============================================================ MODEL
def _device(cfg):
    import torch
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device


def _set_threads(n):
    import torch
    try:
        torch.set_num_threads(max(1, int(n)))
    except Exception:
        pass


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


def _train_predict(X, hs, step_mask, feas, y, labs, sec_lab, cats, fund_arr, tr, te, cfg, dev):
    """Fit one PanelLSTM on the train sequences; return test predictions as rows.
    `fund_arr` [S] labels each sample's fund (one fund in per_fund mode, many in global),
    so the naive baseline stays PER FUND in both modes and the comparison is like-for-like."""
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

    numcell = int(np.clip(N, 16, cfg.hidden_cap))       # "scales with the cross-section"
    model = _make_panel_lstm(N, F, numcell, cfg.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-1)   # -1 = padded / unlabelled slot

    Xt = torch.from_numpy(Xf); yt = torch.from_numpy(y); mt = torch.from_numpy(step_mask)
    tr_i = np.where(tr)[0]
    nval = int(len(tr_i) * cfg.val_frac)                 # chronological (never shuffle)
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

    # naive: PER FUND, max(frequency) class across all securities and time steps of the
    # fund's training window ("the naive classifier predicts the max(frequency) class
    # observed across all securities and time steps").
    naive_map = {}
    for f in np.unique(fund_arr[tr]):
        yy = y[tr & (fund_arr == f)]
        yy = yy[yy >= 0]
        naive_map[f] = float(np.bincount(yy, minlength=3).argmax() - 1) if yy.size else 0.0

    rows = []
    for k, s in enumerate(te_i):
        p = P[k].copy()
        p[~feas[s], 0] = 0.0                  # feasibility: can't sell what you don't hold
        p = p / p.sum(1, keepdims=True)
        valid = np.where((y[s] >= 0) & (sec_lab[s] >= 0))[0]
        if valid.size == 0:
            continue
        rows.append(pd.DataFrame({
            "fund": fund_arr[s],
            "security": cats[sec_lab[s][valid]],      # whoever occupies the slot at t
            "qi": int(labs[s]),
            "slot": valid + 1,                        # = rank at the label quarter
            "y_pred": p[valid].argmax(1) - 1,
            "p_sell": p[valid, 0], "p_hold": p[valid, 1], "p_buy": p[valid, 2],
            "feasible": feas[s][valid],
            "y_naive": naive_map.get(fund_arr[s], 0.0),
        }))
    return pd.concat(rows, ignore_index=True) if rows else None


# ============================================================ WALK-FORWARD
def _windows(qi_min, qi_max, cfg):
    for c in range(qi_min + cfg.window_q, qi_max + 2, cfg.step):
        yield c - cfg.window_q, c - cfg.test_q, c - cfg.test_q, c   # tr_lo,tr_hi,te_lo,te_hi


def _iter_funds(panel):
    """(fund, sub-frame), LARGEST fund first (longest-job-first -> no idle cores at the tail)."""
    order = panel["fund"].value_counts().index
    cat = pd.Categorical(panel["fund"], categories=order, ordered=True)
    for f, fp in panel.groupby(cat, sort=True, observed=True):
        yield f, fp


def _fund_index_order(panel):
    """(fund, row-positions), largest first. Only index arrays -> worker threads .take()
    their fund's rows from the SHARED panel on demand (memory-flat)."""
    idx = panel.groupby("fund", observed=True).indices
    for f in sorted(idx, key=lambda k: -len(idx[k])):
        yield f, idx[f]


def _fund_task(fund, fp, feat, cfg, n_slots, dev="cpu"):
    """All rolling windows for ONE fund. Single-threaded (parallelism is across funds)."""
    _set_threads(1)
    ql = np.array(sorted(fp["qi"].unique()))
    if len(ql) < cfg.window_q:
        return []
    outs = []
    for tr_lo, tr_hi, te_lo, te_hi in _windows(int(ql[0]), int(ql[-1]), cfg):
        t = build_window_tensors(fp, feat, cfg, tr_lo, te_hi, n_slots)
        if t is None:
            continue
        X, hs, sm, feas, y, labs, sec_lab, cats = t
        tr = (labs >= tr_lo) & (labs < tr_hi)
        te = (labs >= te_lo) & (labs < te_hi)
        if tr.sum() < cfg.min_samples or te.sum() == 0:
            continue
        fund_arr = np.array([fund] * len(labs), dtype=object)
        out = _train_predict(X, hs, sm, feas, y, labs, sec_lab, cats, fund_arr,
                             tr, te, cfg, dev)
        if out is not None and len(out):
            outs.append(out)
    return outs


def _global_window(panel, feat, cfg, n_slots, tr_lo, tr_hi, te_lo, te_hi, dev):
    """ONE model per window, pooled across ALL funds. Only possible because N is fixed
    at max_rank, so every fund shares the same (N*F) -> (N*3) shape."""
    Xs, hss, sms, fes, ys, lbs, sls, funds = [], [], [], [], [], [], [], []
    cats = None
    for f, fp in _iter_funds(panel):
        ql = fp["qi"].to_numpy()
        if ql.min() > tr_lo or ql.max() < te_lo:      # fund not alive across this window
            continue
        t = build_window_tensors(fp, feat, cfg, tr_lo, te_hi, n_slots)
        if t is None:
            continue
        X, hs, sm, feas, y, labs, sec_lab, c = t
        cats = c
        Xs.append(X); hss.append(hs); sms.append(sm); fes.append(feas)
        ys.append(y); lbs.append(labs); sls.append(sec_lab)
        funds.append(np.array([f] * len(labs), dtype=object))
    if not Xs:
        return None
    X = np.concatenate(Xs); hs = np.concatenate(hss); sm = np.concatenate(sms)
    feas = np.concatenate(fes); y = np.concatenate(ys); labs = np.concatenate(lbs)
    sec_lab = np.concatenate(sls); fund_arr = np.concatenate(funds)
    tr = (labs >= tr_lo) & (labs < tr_hi)
    te = (labs >= te_lo) & (labs < te_hi)
    if tr.sum() < cfg.min_samples or te.sum() == 0:
        return None
    return _train_predict(X, hs, sm, feas, y, labs, sec_lab, cats, fund_arr, tr, te, cfg, dev)


def run_model(panel: pd.DataFrame, cfg: Config, verbose=True):
    """Walk-forward. `per_fund` = one PanelLSTM per
    fund-window (paper); `global` = one PanelLSTM per window pooled across funds
    (possible because N is fixed at the template width). Returns OOS predictions."""
    import torch
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = _device(cfg)
    feat = [f for f in cfg.features if f in panel.columns]
    ncore = os.cpu_count() or 1
    njobs = ncore if cfg.n_jobs in (-1, 0) else cfg.n_jobs
    n_funds = int(panel["fund"].nunique())
    n_slots = int(panel.attrs.get("eff_rank") or panel["rank"].max())
    if verbose:
        print(f"[model] paper (T,N,F)->(N x 3) | mode={cfg.model_mode} "
              f"device={dev} N={n_slots} F={len(feat)} -> LSTM in {n_slots*len(feat)}, "
              f"out {n_slots*3} | funds={n_funds} cores={ncore}")

    preds = []
    if cfg.model_mode == "global":
        _set_threads(cfg.torch_threads or ncore)
        Nq = int(panel["qi"].max()) + 1
        wins = list(_windows(0, Nq - 1, cfg))
        for wi, (tr_lo, tr_hi, te_lo, te_hi) in enumerate(wins, 1):
            out = _global_window(panel, feat, cfg, n_slots, tr_lo, tr_hi, te_lo, te_hi, dev)
            if out is not None and len(out):
                preds.append(out)
                if verbose:
                    a = (out.loc[out.feasible, "y_pred"] == out.loc[out.feasible, "Y"]).mean() \
                        if "Y" in out else float("nan")
                    print(f"  win {wi}/{len(wins)} test qi[{te_lo},{te_hi}) n={len(out):,}")
    elif dev != "cpu" or njobs == 1 or cfg.parallel_backend == "serial":
        _set_threads(cfg.torch_threads or ncore)
        for i, (f, fp) in enumerate(_iter_funds(panel), 1):
            preds += _fund_task(f, fp, feat, cfg, n_slots, dev)
            if verbose and i % 50 == 0:
                print(f"  {i}/{n_funds} funds")
    else:
        # SHARED-MEMORY threads: one panel in RAM; torch releases the GIL during compute
        from concurrent.futures import ThreadPoolExecutor
        _set_threads(1)
        if verbose:
            print(f"  {n_funds} funds on {njobs} THREADS (shared memory, low RAM)")

        def _wrk(a):
            f, ix = a
            return _fund_task(f, panel.take(ix), feat, cfg, n_slots, "cpu")
        done = 0
        with ThreadPoolExecutor(max_workers=njobs) as ex:
            for outs in ex.map(_wrk, _fund_index_order(panel)):
                preds += outs; done += 1
                if verbose and done % 100 == 0:
                    print(f"  {done}/{n_funds} funds done")
    if not preds:
        raise RuntimeError("no predictions -- window too short, or lower min_samples")
    out = pd.concat(preds, ignore_index=True)
    cols = [c for c in ["fund", "security", "qi", "yq", "Y", "fwd_1q", "fwd_2q", "fwd_3q",
                        "inv_type", "weight", "rank"] if c in panel.columns]
    return out.merge(panel[cols], on=["fund", "security", "qi"], how="left")


# ============================================================ EVALUATION
def _t(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan


def _resolve_eval(cfg, cols):
    """eval_mode -> (return column, extra lag on the sort variable)."""
    m = cfg.eval_mode
    if m == "contemporaneous":
        return "fwd_1q", 0
    if m == "lagged":
        return "fwd_1q", 1
    if m == "predictive":
        if cols.get("fwd_2q"):
            return "fwd_2q", 0
        print("[warn] eval_mode='predictive' needs `future_2q_ret`; falling back to 'lagged'")
        return "fwd_1q", 1
    if m == "tradeable":
        if cols.get("fwd_3q"):
            return "fwd_3q", 0
        print("[warn] eval_mode='tradeable' needs `future_3q_ret`; falling back to 'predictive'")
        return ("fwd_2q", 0) if cols.get("fwd_2q") else ("fwd_1q", 1)
    raise ValueError(f"unknown eval_mode: {m!r}")


def evaluate(preds: pd.DataFrame, cfg: Config):
    """Predictability + portfolio sorts (Tables X / XI / XII). Timing = cfg.eval_mode."""
    P = preds.copy()
    P["yq"] = P["yq"].astype("period[Q]")
    has = {c: bool(P[c].notna().any()) if c in P.columns else False
           for c in ("fwd_1q", "fwd_2q", "fwd_3q")}
    ret_col, lag = _resolve_eval(cfg, has)
    P["_ret"] = P[ret_col]
    feas = P[P["feasible"]]
    m = {}
    m["lstm_precision_pooled"] = float((feas.y_pred == feas.Y).mean())
    m["naive_precision_pooled"] = float((feas.y_naive == feas.Y).mean())
    fp = feas.groupby("fund", observed=True).apply(lambda d: pd.Series({
        "lstm": (d.y_pred == d.Y).mean(), "naive": (d.y_naive == d.Y).mean()}))
    m["lstm_precision_fundavg"] = float(fp["lstm"].mean())
    m["naive_precision_fundavg"] = float(fp["naive"].mean())
    m["n_predictions"] = int(len(P)); m["n_feasible"] = int(len(feas))
    m["n_funds"] = int(feas.fund.nunique())
    m["eval_mode"] = cfg.eval_mode; m["eval_return"] = ret_col; m["eval_sort_lag"] = lag

    def xsq(s):
        v = s.dropna()
        if v.nunique() < 5:
            return pd.Series(np.nan, index=s.index)
        return (pd.qcut(v.rank(method="first"), 5, labels=False, duplicates="drop") + 1).reindex(s.index)

    # ---- fund-level: predictability + benchmark-adjusted future return ----
    P["wc"] = P["weight"] * P["_ret"]
    has_cat = "inv_type" in P.columns and P["inv_type"].nunique() > 1

    def _fq(d):
        o = {"fund_ret": d.wc.sum() / d.weight.sum() if d.weight.sum() > 0 else np.nan,
             "prec": (d.loc[d.feasible, "y_pred"] == d.loc[d.feasible, "Y"]).mean()}
        if has_cat:
            o["inv_type"] = d["inv_type"].iloc[0]
        return pd.Series(o)
    fq = P.groupby(["fund", "yq"], observed=True).apply(_fq).reset_index()
    bench = ["inv_type", "yq"] if has_cat else ["yq"]
    m["benchmark"] = "InvTypeCode x quarter" if has_cat else "universe x quarter"
    fq["abn"] = fq["fund_ret"] - fq.groupby(bench, observed=True)["fund_ret"].transform("mean")
    fq = fq.sort_values(["fund", "yq"])
    fq["prec_lag"] = fq.groupby("fund", observed=True)["prec"].shift(lag)
    for h in range(1, 5):
        fq[f"cabn{h}"] = fq.groupby("fund")["abn"].rolling(h).sum().shift(-(h - 1)).reset_index(0, drop=True)
    fq["Q"] = fq.groupby("yq")["prec_lag"].transform(xsq)
    fqq = fq.dropna(subset=["Q"])
    rowsX = []
    for q in [1, 2, 3, 4, 5]:
        r = {"quintile": f"Q{q}"}
        for h in range(1, 5):
            s = fqq[fqq.Q == q].groupby("yq")[f"cabn{h}"].mean()
            r[f"cum_abn_{h}q"] = s.mean(); r[f"t_{h}q"] = _t(s)
        rowsX.append(r)
    r = {"quintile": "Q5-Q1"}
    for h in range(1, 5):
        d = (fqq[fqq.Q == 5].groupby("yq")[f"cabn{h}"].mean()
             - fqq[fqq.Q == 1].groupby("yq")[f"cabn{h}"].mean()).dropna()
        r[f"cum_abn_{h}q"] = d.mean(); r[f"t_{h}q"] = _t(d)
    rowsX.append(r)
    tableX = pd.DataFrame(rowsX)
    m["tableX_Q5mQ1_4q"] = float(tableX.iloc[-1]["cum_abn_4q"])
    m["tableX_Q5mQ1_4q_t"] = float(tableX.iloc[-1]["t_4q"])

    # ---- Table XI: correct vs incorrect positions ----
    # accuracy(t) is only observable at t+1; eval_mode decides whether the return window
    # already starts after that ("predictive"/"tradeable") or the flag must be lagged.
    P = P.sort_values(["fund", "security", "yq"])
    P["correct"] = (P.y_pred == P.Y).astype(float)
    P["correct_s"] = P.groupby(["fund", "security"], observed=True)["correct"].shift(lag) \
        if lag else P["correct"]
    ci = P.dropna(subset=["_ret", "correct_s"])
    corr = ci[ci.correct_s == 1].groupby("yq")["_ret"].mean()
    inco = ci[ci.correct_s == 0].groupby("yq")["_ret"].mean()
    diff = (corr - inco).dropna()
    tableXI = pd.DataFrame({"portfolio": ["Correct", "Incorrect", "Correct-Incorrect"],
                            "mean_qret": [corr.mean(), inco.mean(), diff.mean()],
                            "t": [_t(corr), _t(inco), _t(diff)]})
    m["correct_minus_incorrect"] = float(diff.mean())
    m["correct_minus_incorrect_t"] = float(_t(diff))

    # ---- Table XII: stock quintiles on cross-fund prediction accuracy ----
    stk = P.groupby(["security", "yq"], observed=True).agg(
        acc=("correct", "mean"), fwd=("_ret", "first")).reset_index()
    stk = stk.sort_values(["security", "yq"])
    stk["acc_s"] = stk.groupby("security", observed=True)["acc"].shift(lag) if lag else stk["acc"]
    stk = stk.dropna(subset=["acc_s", "fwd"])
    stk["Q"] = stk.groupby("yq")["acc_s"].transform(xsq)
    stk = stk.dropna(subset=["Q"])
    rowsXII = [{"quintile": f"Q{q}", "mean_qret": stk[stk.Q == q].groupby("yq")["fwd"].mean().mean(),
                "t": _t(stk[stk.Q == q].groupby("yq")["fwd"].mean())} for q in [1, 2, 3, 4, 5]]
    ls = (stk[stk.Q == 1].groupby("yq")["fwd"].mean()
          - stk[stk.Q == 5].groupby("yq")["fwd"].mean()).dropna()
    rowsXII.append({"quintile": "Q1-Q5", "mean_qret": ls.mean(), "t": _t(ls)})
    tableXII = pd.DataFrame(rowsXII)
    m["tableXII_Q1mQ5"] = float(ls.mean()); m["tableXII_Q1mQ5_t"] = float(_t(ls))
    m["_ls_cum"] = ls.sort_index().cumsum()
    m["_fund_prec"] = fp
    return m, {"tableX": tableX, "tableXI": tableXI, "tableXII": tableXII}


def compare_eval_modes(preds: pd.DataFrame, cfg: Config = None,
                       modes=("contemporaneous", "lagged", "predictive", "tradeable")):
    """Re-evaluate the SAME predictions under each timing convention (training is
    independent of it, so this is nearly free). If the spread only shows up under
    'contemporaneous', it is same-quarter co-movement -- not predictive alpha."""
    cfg = cfg or Config()
    rows = []
    for md in modes:
        try:
            mm, _ = evaluate(preds, replace(cfg, eval_mode=md))
            rows.append({"eval_mode": md, "return_used": mm["eval_return"],
                         "sort_lag": mm["eval_sort_lag"],
                         "TableXII_Q1mQ5": mm["tableXII_Q1mQ5"], "XII_t": mm["tableXII_Q1mQ5_t"],
                         "TableXI_corr_minus_inc": mm["correct_minus_incorrect"],
                         "XI_t": mm["correct_minus_incorrect_t"],
                         "TableX_Q5mQ1_4q": mm["tableX_Q5mQ1_4q"], "X_t": mm["tableX_Q5mQ1_4q_t"]})
        except Exception as e:
            rows.append({"eval_mode": md, "error": str(e)[:70]})
    return pd.DataFrame(rows)


# ============================================================ FIGURES
def _figures(metrics, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = {}
    if cfg.save_outputs:
        os.makedirs(cfg.out_dir, exist_ok=True)
    fp = metrics["_fund_prec"]
    fig1, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fp["naive"].dropna(), bins=30, alpha=.5, label=f"Naive ({fp['naive'].mean():.2f})", color="tab:red")
    ax.hist(fp["lstm"].dropna(), bins=30, alpha=.6, label=f"LSTM ({fp['lstm'].mean():.2f})", color="tab:blue")
    ax.axvline(.5, ls="--", c="k", lw=1)
    ax.set_xlabel("per-fund precision"); ax.set_ylabel("# funds")
    ax.set_title("Fund-level trade-direction predictability (paper architecture)"); ax.legend()
    fig1.tight_layout()
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    metrics["_ls_cum"].plot(ax=ax2)
    ax2.set_title("Cumulative Q1-Q5 (least - most predictable stocks)")
    ax2.set_ylabel("cumulative quarterly return"); fig2.tight_layout()
    if cfg.save_outputs:
        p1, p2 = f"{cfg.out_dir}/fig_precision_dist.png", f"{cfg.out_dir}/fig_stock_ls.png"
        fig1.savefig(p1, dpi=130); fig2.savefig(p2, dpi=130)
        paths = {"precision_dist": p1, "stock_ls": p2}
    return {"precision_dist": fig1, "stock_ls": fig2, "paths": paths}


# ============================================================ ORCHESTRATE
def run(cfg: Config = None, verbose=True):
    """Full pipeline. Returns {panel, predictions, metrics, tables, figures, config}."""
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
    figs = _figures(metrics, cfg)
    clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
    clean["architecture"] = "paper_(T,N,F)->(Nx3)"
    if verbose:
        print("\n=== PREDICTABILITY ===")
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
    """Run the FULL pipeline at several rank cutoffs. max_rank changes the panel itself,
    so every cutoff needs its own training run."""
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


if __name__ == "__main__":
    run(Config())
