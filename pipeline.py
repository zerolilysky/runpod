"""
mimicking_pipeline.py
=====================
Self-contained replication of Cohen, Lu & Nguyen (2025), "Mimicking Finance",
driven by a SINGLE holdings panel parquet -- no external data pulls.

It predicts the direction (buy / hold / sell) of each fund's next-quarter trade in
each security with an LSTM, then relates *predictability* to future returns
(Tables X / XI / XII).  A config flag chooses between:
    * one LSTM PER FUND         (cfg.model_mode = "per_fund")   -- faithful to the paper
    * one GLOBAL LSTM across funds (cfg.model_mode = "global")  -- faster

Expected input columns (rename via Config if yours differ):
    fund, date, security, quarter, shares, position_value, shares_change, close,
    market_cap, volume, isUs, security_name, quarterly_ret, past_1q_ret,
    future_1q_ret, year, portfolio_value, weight, rank, n_holdings

NO LOOK-AHEAD
-------------
* Target Y_t = sign of the t -> t+1 share change (a label, never a feature).
* Every feature is observable at quarter t (own past trades via `pdsh`; peer rates
  lagged one quarter; characteristics realized by end of t).
* `future_1q_ret` is used ONLY as the realized next-quarter return for evaluation.
* Rolling windows split chronologically; the scaler is fit on train rows only;
  the model never sees the test block while fitting.
"""
from __future__ import annotations
import os, json, warnings
from dataclasses import dataclass, field, asdict
from typing import List
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================ CONFIG
@dataclass
class Config:
    # ---- data ----
    data_path: str = ("manager_holdings/master_batches_returnfiltered/"
                      "panel_holdings_All_Funds_2002_master.parquet")
    out_dir: str = "outputs"
    # column mapping: {your_column: internal_name}. Adjust only if your names differ.
    col_map: dict = field(default_factory=lambda: {
        "fund": "fund", "date": "date", "security": "security", "shares": "shares",
        "position_value": "position_value", "shares_change": "shares_change",
        "close": "close", "market_cap": "market_cap", "volume": "volume",
        "isUs": "isUs", "quarterly_ret": "quarterly_ret", "past_1q_ret": "past_1q_ret",
        "future_1q_ret": "future_1q_ret", "future_2q_ret": "future_2q_ret",
        "portfolio_value": "portfolio_value",
        "weight": "weight", "rank": "rank", "n_holdings": "n_holdings",
    })
    # ---- evaluation timing --------------------------------------------------
    # accuracy(t) = "did we predict the t->t+1 trade right?" -> only OBSERVABLE at t+1.
    #   "predictive"      : sort on accuracy(t), return over t+1->t+2  (`future_2q_ret`)
    #                       -> no overlap, freshest signal.  RECOMMENDED.
    #   "contemporaneous" : sort on accuracy(t), return over t->t+1    (`future_1q_ret`)
    #                       -> sort variable overlaps its own return window (look-ahead).
    #                          Keep only as the biased benchmark for comparison.
    #   "lagged"          : sort on accuracy(t-1), return over t->t+1  -> clean but staler.
    # Training is independent of this -> use compare_eval_modes() to get all three
    # from a SINGLE training run.
    eval_mode: str = "predictive"
    # ---- sample filters (paper Sec 3.1) ----
    us_only: bool = True
    min_years: int = 7          # >7 calendar years of history
    min_holdings: int = 10      # >=10 securities per quarter (auto-capped at max_rank)
    # Keep only positions with rank <= max_rank (1 = largest holding).
    #   None -> use whatever the file already contains (auto-detect, no filter)
    #   e.g. 25 / 10 -> re-run at a tighter cross-section to compare.
    # NOTE: changing this changes the panel, so each value needs its own training
    # run -- see run_rank_sweep().
    max_rank: int = None
    change_band: float = 0.01   # +-1% dead-band around zero share change
    # ---- rolling design (paper Fig 2) ----
    window_q: int = 28          # observation window
    train_q: int = 20           # train quarters within a window
    test_q: int = 8             # test quarters within a window
    seq_len: int = 8            # LSTM input sequence length (quarters)
    step: int = 8               # quarters to advance each window (1 = fully overlapping)
    # ---- model ----
    model_mode: str = "per_fund"   # "per_fund" or "global"
    hidden: int = 64               # global mode hidden size
    hidden_cap: int = 64           # per-fund: hidden = clip(#securities, 16, hidden_cap)
    dropout: float = 0.25
    max_epochs: int = 25
    patience: int = 6
    lr: float = 3e-3
    batch: int = 4096
    min_seq_per_fund: int = 120    # skip a fund-window with fewer train sequences
    min_train_global: int = 2000
    device: str = "auto"           # "auto" | "cpu" | "cuda"
    seed: int = 42
    # ---- CPU performance / memory ----
    n_jobs: int = -1               # per-fund: parallel funds across cores (-1 = all). 1 = serial.
    torch_threads: int = 0         # intra-op threads. 0 = auto
    parallel_backend: str = "threading"  # "threading" (SHARED memory, low RAM) | "loky" (processes, needs RAM) | "serial"
    downcast: bool = True          # float32 + categoricals -> ~halves panel RAM
    keep_panel: bool = False       # include the (large) panel in run() output (False frees it)
    # ---- misc ----
    save_outputs: bool = True

    @property
    def features(self) -> List[str]:
        return ["weight", "w_lag1", "dw", "rank", "log_posval", "log_pv", "log_mktcap",
                "quarterly_ret", "past_1q_ret", "pdsh", "pdsh_sign", "pdsh_lag1",
                "sh_lag1", "sh_lag2", "sh_lag3",
                "peer_buy", "peer_sell", "peer_hold", "n_holdings", "fund_ret_l1"]


# ============================================================ DATA + FEATURES
def load_and_prepare(cfg: Config) -> pd.DataFrame:
    """Load the panel, build the target and all (lagged) features. Returns a tidy
    fund-security-quarter frame with columns: fund, security, yq, qi, <features>,
    Y (target), fwd_qret, weight, rank."""
    inv = {v: k for k, v in cfg.col_map.items()}          # internal -> your column
    # (a) READ ONLY the raw columns we actually use (huge saving on a 20M-row file)
    want = ["fund", "date", "security", "shares", "position_value", "market_cap",
            "quarterly_ret", "past_1q_ret", "future_1q_ret", "future_2q_ret",
            "portfolio_value", "weight", "rank", "n_holdings", "isUs"]
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
    df = df.sort_values("date").drop_duplicates(["fund", "yq", "security"], keep="last")
    # rank cutoff: honour max_rank if given, else use whatever the file contains
    eff_rank = None
    if "rank" in df.columns:
        if cfg.max_rank is not None:
            df = df[df["rank"] <= cfg.max_rank]
        eff_rank = int(df["rank"].max()) if len(df) else 0
    df.drop(columns=[c for c in ("date", "isUs") if c in df.columns], inplace=True)

    # (b) DOWNCAST raw numerics to float32 UP FRONT -> every derived feature stays
    # float32, pd.NA/nullable become np.nan, and no float64 (n_cols, n_rows) block forms.
    F32 = "float32" if cfg.downcast else "float64"
    raw_num = ["shares", "position_value", "market_cap", "quarterly_ret", "past_1q_ret",
               "future_1q_ret", "future_2q_ret", "portfolio_value", "weight", "rank",
               "n_holdings"]
    for c in raw_num:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(F32) if c in df.columns \
            else np.array(np.nan, dtype=F32)
    df = df.sort_values(["fund", "security", "yq"]).reset_index(drop=True)
    g = df.groupby(["fund", "security"], sort=False)

    # own-position dynamics (lagged; float32). Only lags 1-3 are used as features.
    for k in (1, 2, 3):
        df[f"sh_lag{k}"] = g["shares"].shift(k).astype(F32)
    df["w_lag1"] = g["weight"].shift(1).astype(F32)
    df["dw"] = (df["weight"] - df["w_lag1"]).astype(F32)
    df["pdsh"] = ((df["shares"] - df["sh_lag1"]) / (df["sh_lag1"].abs() + 1.0)).astype(F32)
    df["pdsh_sign"] = np.sign(df["pdsh"]).fillna(0.0).astype(F32)
    df["pdsh_lag1"] = g["pdsh"].shift(1).astype(F32)
    df["log_posval"] = np.log(df["position_value"].abs() + 1.0).astype(F32)
    df["log_pv"] = np.log(df["portfolio_value"].abs() + 1.0).astype(F32)
    df["log_mktcap"] = np.log(df["market_cap"].abs() + 1.0).astype(F32)

    # TARGET: sign of NEXT-quarter fractional share change (+-band). label, not a feature.
    sh_next = g["shares"].shift(-1)
    dsh = (sh_next - df["shares"]) / (df["shares"].abs() + 1.0)
    df["Y"] = np.select([dsh <= -cfg.change_band, dsh >= cfg.change_band], [-1.0, 1.0], default=0.0).astype(F32)
    df.loc[dsh.isna(), "Y"] = np.nan
    # Realised returns -- EVALUATION ONLY, never features. Carry BOTH so all three
    # eval_modes can be produced from one training run.
    df["fwd_1q"] = df["future_1q_ret"]                     # t   -> t+1
    df["fwd_2q"] = df["future_2q_ret"]                     # t+1 -> t+2  (strictly after acc(t))

    # fund-level filters. min_holdings must be capped at the rank cutoff: with
    # max_rank=10 a fund can never hold more than 10 of its top-10, so an
    # uncapped min_holdings=10 would silently empty the panel.
    mh = cfg.min_holdings
    if eff_rank and mh > eff_rank:
        print(f"[data] min_holdings {mh} > max rank {eff_rank} -> capped to {eff_rank}")
        mh = eff_rank
    cnt = df.groupby(["fund", "yq"])["security"].transform("count")
    df = df[cnt >= mh]
    nq = df.groupby("fund")["yq"].transform("nunique")
    df = df[nq >= cfg.min_years * 4]
    df.attrs["eff_rank"] = eff_rank

    # peer activity across the universe (lagged one quarter). map() -> no frame copy.
    lab = df.dropna(subset=["Y"])
    rate = lab.groupby("yq").agg(
        peer_buy=("Y", lambda s: (s > 0).mean()),
        peer_sell=("Y", lambda s: (s < 0).mean()),
        peer_hold=("Y", lambda s: (s == 0).mean())).sort_index().shift(1)
    for col in ("peer_buy", "peer_sell", "peer_hold"):
        df[col] = df["yq"].map(rate[col]).astype(F32)

    # fund past-quarter return proxy (weight-weighted). reindex-map -> no merge/copy.
    contrib = (df["w_lag1"] * df["quarterly_ret"])
    fr = contrib.groupby([df["fund"], df["yq"]]).sum()            # (fund,yq) -> ret
    fr_l1 = fr.groupby(level=0).shift(1)                          # prior quarter within fund
    key = pd.MultiIndex.from_arrays([df["fund"].to_numpy(), df["yq"].to_numpy()])
    df["fund_ret_l1"] = fr_l1.reindex(key).to_numpy(dtype=F32, na_value=np.nan)

    # integer quarter index for windowing
    qs = pd.PeriodIndex(sorted(df["yq"].unique()), freq="Q")
    df["qi"] = df["yq"].map({q: i for i, q in enumerate(qs)}).astype("int32")
    df["held"] = np.int8(1)

    # (c) PRUNE IN PLACE to only what's needed (no big .copy()). Everything already float32.
    feat = [f for f in cfg.features if f in df.columns]
    keep = set(["fund", "security", "yq", "qi", "held", "Y", "fwd_1q", "fwd_2q",
                "weight", "rank"] + feat)
    df.drop(columns=[c for c in df.columns if c not in keep], inplace=True)
    if cfg.downcast:
        df["fund"] = df["fund"].astype("category")
        df["security"] = df["security"].astype("category")
    return df


# ============================================================ SEQUENCES
def build_sequences(sub: pd.DataFrame, feat_cols: List[str], seq_len: int):
    """[N, seq_len, F] tensors for every held, labelled position in `sub`. Steps
    where the security was not held that quarter are zero-filled and flagged in the
    mask. `feasible` = held in all seq_len steps. Label lives at the last step."""
    sub = sub.sort_values(["fund", "security", "qi"])
    g = sub.groupby(["fund", "security"], sort=False)
    valid = (sub["held"].eq(1) & sub["Y"].notna()).values
    N = int(valid.sum())
    if N == 0:
        return None
    F = len(feat_cols)
    qi = sub["qi"].values
    X = np.zeros((N, seq_len, F), dtype=np.float32)
    mask = np.zeros((N, seq_len), dtype=np.float32)
    for k in range(seq_len):                                   # k=0 newest ... last=oldest
        step = seq_len - 1 - k
        # to_numpy(na_value=np.nan) turns pd.NA / nullable / pyarrow columns into
        # plain float np.nan -- `.values` would leave pd.NA and break numpy/torch.
        vals = g[feat_cols].shift(k).to_numpy(dtype="float32", na_value=np.nan)
        qik = g["qi"].shift(k).to_numpy(dtype="float64", na_value=np.nan)
        heldk = g["held"].shift(k).to_numpy(dtype="float64", na_value=0.0)
        present = (qik == (qi - k)) & (heldk > 0)
        m = present[valid].astype(np.float32)
        X[:, step, :] = np.nan_to_num(vals[valid], nan=0.0, posinf=0.0, neginf=0.0) * m[:, None]
        mask[:, step] = m
    feasible = mask.all(axis=1)
    y = (sub["Y"].values[valid] + 1).astype(np.int64)          # {-1,0,1} -> {0,1,2}
    meta = sub.loc[valid, ["fund", "security", "yq", "qi", "Y", "fwd_1q", "fwd_2q",
                           "weight", "rank"]].reset_index(drop=True)
    return X, mask, y, feasible, meta


# ============================================================ MODEL
def _device(cfg):
    import torch
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device


def _make_model(F, hidden, dropout):
    import torch.nn as nn

    class SeqLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(F, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 3)

        def forward(self, x, m=None):
            if m is not None:
                x = x * m.unsqueeze(-1)
            o, _ = self.lstm(x)
            return self.head(self.drop(o[:, -1, :]))
    return SeqLSTM()


def _train_predict(X, mask, y, feas, meta, tr, te, F, hidden, cfg, dev):
    """Fit one LSTM on train rows, return a predictions DataFrame for test rows.
    Training, validation, and prediction are ALL mini-batched so no full split is
    ever materialised on-device -- safe for many cores with limited RAM."""
    import torch
    Xtr, Mtr = X[tr], mask[tr].astype(bool)
    flat, pres = Xtr.reshape(-1, F), Mtr.reshape(-1)
    if pres.sum() < 50:
        return None
    mu, sd = flat[pres].mean(0), flat[pres].std(0) + 1e-6
    Xz = ((X - mu) / sd).astype(np.float32) * mask[..., None]
    Xt, yt, mt = torch.from_numpy(Xz), torch.from_numpy(y), torch.from_numpy(mask)
    model = _make_model(F, hidden, cfg.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = torch.nn.CrossEntropyLoss()
    idx = np.where(tr)[0]
    rng = np.random.default_rng(cfg.seed); rng.shuffle(idx)
    nval = max(1, int(0.15 * len(idx))); val_i, trn_i = idx[:nval], idx[nval:]
    bs = cfg.batch

    def _val_loss():
        model.eval(); tot, n = 0.0, 0
        with torch.inference_mode():
            for b in range(0, len(val_i), bs):
                bi = val_i[b:b + bs]
                l = lossf(model(Xt[bi].to(dev), mt[bi].to(dev)), yt[bi].to(dev))
                tot += float(l) * len(bi); n += len(bi)
        return tot / max(n, 1)

    best, best_state, bad = 1e9, None, 0
    for _ in range(cfg.max_epochs):
        model.train()
        rng.shuffle(trn_i)
        for b in range(0, len(trn_i), bs):
            bi = trn_i[b:b + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[bi].to(dev), mt[bi].to(dev)), yt[bi].to(dev))
            loss.backward(); opt.step()
        vl = _val_loss()
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    te_i = np.where(te)[0]
    chunks = []
    with torch.inference_mode():
        for b in range(0, len(te_i), bs):
            bi = te_i[b:b + bs]
            chunks.append(torch.softmax(model(Xt[bi].to(dev), mt[bi].to(dev)), 1).cpu().numpy())
    P = np.concatenate(chunks) if chunks else np.zeros((0, 3))
    m = meta.iloc[te_i].copy()
    infeas = ~feas[te_i]
    P[infeas, 0] = 0.0; P = P / P.sum(1, keepdims=True)
    m["p_sell"], m["p_hold"], m["p_buy"] = P[:, 0], P[:, 1], P[:, 2]
    m["y_pred"] = P.argmax(1) - 1
    m["feasible"] = feas[te_i]
    maj = meta[tr].groupby("fund")["Y"].agg(lambda s: s.value_counts().idxmax())
    m["y_naive"] = m["fund"].map(maj).fillna(0.0)
    return m


def _windows(qi_min, qi_max, cfg):
    for c in range(qi_min + cfg.window_q, qi_max + 2, cfg.step):
        yield c - cfg.window_q, c - cfg.test_q, c - cfg.test_q, c   # tr_lo, tr_hi, te_lo, te_hi


def _set_threads(n):
    import torch
    try:
        torch.set_num_threads(max(1, int(n)))
    except Exception:
        pass


def _iter_funds(panel):
    """Yield (fund, sub-frame) LARGEST fund first (longest-job-first scheduling ->
    keeps every core busy through the tail of the run). Single groupby pass; never
    materialises all frames at once."""
    order = panel["fund"].value_counts().index                 # most rows first
    cat = pd.Categorical(panel["fund"], categories=order, ordered=True)
    for f, fp in panel.groupby(cat, sort=True, observed=True):
        yield f, fp


def _fund_index_order(panel):
    """(fund, row-positions) pairs, largest fund first. Only integer index arrays
    are produced (a few hundred MB for 20M rows) -- NO frame copies -- so worker
    threads slice the SHARED panel with .take() on demand (memory-flat)."""
    idx = panel.groupby("fund", observed=True).indices          # {fund: positions}
    for f in sorted(idx, key=lambda k: -len(idx[k])):
        yield f, idx[f]


def _fund_task(fund, fp, feat, cfg, dev="cpu"):
    """All rolling windows for ONE fund. Sequences are built ONCE for the whole
    fund and sliced per window (big CPU saving vs rebuilding each window). Runs
    single-threaded so parallelism happens ACROSS funds. Holds only this fund's
    data -> low memory. Returns a list of prediction DataFrames."""
    _set_threads(1)
    ql = np.array(sorted(fp["qi"].unique()))
    if len(ql) < cfg.window_q:
        return []
    seq = build_sequences(fp, feat, cfg.seq_len)
    if seq is None:
        return []
    X, mask, y, feas, meta = seq
    F = len(feat)
    qi = meta["qi"].values
    outs = []
    for tr_lo, tr_hi, te_lo, te_hi in _windows(int(ql[0]), int(ql[-1]), cfg):
        tr = (qi >= tr_lo) & (qi < tr_hi)
        te = (qi >= te_lo) & (qi < te_hi)
        if tr.sum() < cfg.min_seq_per_fund or te.sum() == 0:
            continue
        n_sec = int(meta.loc[te, "security"].nunique())
        hidden = int(np.clip(n_sec, 16, cfg.hidden_cap))
        out = _train_predict(X, mask, y, feas, meta, tr, te, F, hidden, cfg, dev)
        if out is not None:
            outs.append(out)
    return outs


def run_model(panel: pd.DataFrame, cfg: Config, verbose=True):
    """Walk-forward training. `global` = one shared model per window (batched,
    memory-heavier); `per_fund` = one model per fund, funds parallelised across
    cores (CPU-friendly, low memory). Returns pooled OOS predictions."""
    import os as _os
    import torch
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = _device(cfg)
    feat = [f for f in cfg.features if f in panel.columns]
    F = len(feat)
    ncore = _os.cpu_count() or 1
    if verbose:
        print(f"[model] mode={cfg.model_mode} device={dev} F={F} "
              f"quarters={panel.qi.nunique()} cores={ncore}")

    if cfg.model_mode == "global":
        _set_threads(cfg.torch_threads or ncore)      # batched matmuls -> use all cores
        Nq = int(panel["qi"].max()) + 1
        preds = []
        for wi, (tr_lo, tr_hi, te_lo, te_hi) in enumerate(_windows(0, Nq - 1, cfg)):
            sub = panel[(panel["qi"] >= tr_lo - cfg.seq_len) & (panel["qi"] < te_hi)]
            seq = build_sequences(sub, feat, cfg.seq_len)
            if seq is None:
                continue
            X, mask, y, feas, meta = seq
            tr = ((meta["qi"] >= tr_lo) & (meta["qi"] < tr_hi)).values
            te = ((meta["qi"] >= te_lo) & (meta["qi"] < te_hi)).values
            if tr.sum() < cfg.min_train_global or te.sum() == 0:
                continue
            out = _train_predict(X, mask, y, feas, meta, tr, te, F, cfg.hidden, cfg, dev)
            if out is not None:
                preds.append(out)
                if verbose:
                    acc = (out.loc[out.feasible, "y_pred"] == out.loc[out.feasible, "Y"]).mean()
                    print(f"  win {wi+1} test qi[{te_lo},{te_hi}) n_te={int(te.sum()):,} feas_acc={acc:.3f}")
    else:  # per_fund  (funds are independent)
        n_funds = int(panel["fund"].nunique())
        njobs = ncore if cfg.n_jobs in (-1, 0) else cfg.n_jobs
        backend = cfg.parallel_backend if (dev == "cpu" and njobs > 1) else "serial"
        preds = []

        if backend == "serial":
            _set_threads(cfg.torch_threads or ncore)
            acc = []
            for done, (f, fp) in enumerate(_iter_funds(panel), 1):
                outs = _fund_task(f, fp, feat, cfg, dev); preds += outs
                acc += [(o.loc[o.feasible, "y_pred"] == o.loc[o.feasible, "Y"]).mean() for o in outs]
                if verbose and done % 50 == 0:
                    print(f"  [per-fund] {done}/{n_funds} | running feas_acc={np.nanmean(acc):.3f}")

        elif backend == "threading":
            # SHARED-MEMORY threads: one panel in RAM, workers .take() their fund's
            # rows on demand. torch releases the GIL during LSTM compute, so many
            # cores are used with ~no extra memory (fixes the OOM from process pools).
            from concurrent.futures import ThreadPoolExecutor
            _set_threads(1)                     # each thread's torch ops single-threaded
            if verbose:
                print(f"  [per-fund] {n_funds} funds on {njobs} THREADS (shared memory, low RAM)")

            def _wrk(args):
                f, ix = args
                return _fund_task(f, panel.take(ix), feat, cfg, "cpu")

            done = 0
            with ThreadPoolExecutor(max_workers=njobs) as ex:
                for outs in ex.map(_wrk, _fund_index_order(panel)):
                    preds += outs; done += 1
                    if verbose and done % 100 == 0:
                        print(f"  [per-fund] {done}/{n_funds} funds done")

        else:  # "loky" processes -- higher RAM (each worker re-imports torch)
            from joblib import Parallel, delayed
            _set_threads(1)
            if verbose:
                print(f"  [per-fund] {n_funds} funds on {njobs} loky processes (needs RAM)")
            gen = (delayed(_fund_task)(f, fp, feat, cfg, "cpu") for f, fp in _iter_funds(panel))
            results = Parallel(n_jobs=njobs, backend="loky", batch_size=1,
                               pre_dispatch="2*n_jobs", verbose=(5 if verbose else 0))(gen)
            preds = [o for r in results for o in r]
    if not preds:
        raise RuntimeError("no predictions produced -- check filters / window sizes")
    return pd.concat(preds, ignore_index=True)


# ============================================================ EVALUATION
def _t(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan


def _resolve_eval(cfg, cols):
    """eval_mode -> (return column, extra lag on the sort variable).

    accuracy(t) is only observable at t+1, so it must never be paired with the
    t->t+1 return unless it is lagged.
      predictive      -> fwd_2q (t+1->t+2), lag 0   : no overlap, freshest signal
      contemporaneous -> fwd_1q (t  ->t+1), lag 0   : OVERLAPS -> biased benchmark only
      lagged          -> fwd_1q (t  ->t+1), lag 1   : clean but staler
    """
    m = cfg.eval_mode
    if m == "contemporaneous":
        return "fwd_1q", 0
    if m == "lagged":
        return "fwd_1q", 1
    if m == "predictive":
        if "fwd_2q" in cols and pd.notna(cols["fwd_2q"]):
            return "fwd_2q", 0
        print("[warn] eval_mode='predictive' needs `future_2q_ret`; falling back to 'lagged'")
        return "fwd_1q", 1
    raise ValueError(f"unknown eval_mode: {m!r}")


def evaluate(preds: pd.DataFrame, cfg: Config):
    """Predictability + portfolio sorts (Tables X / XI / XII). Returns
    (metrics dict, {table_name: DataFrame}). Timing is set by cfg.eval_mode."""
    P = preds.copy()
    P["yq"] = P["yq"].astype("period[Q]")
    has = {c: (P[c].notna().any() if c in P.columns else np.nan) for c in ("fwd_1q", "fwd_2q")}
    ret_col, lag = _resolve_eval(cfg, has)
    P["_ret"] = P[ret_col]
    feas = P[P["feasible"]]
    m = {}
    m["lstm_precision_pooled"] = float((feas.y_pred == feas.Y).mean())
    m["naive_precision_pooled"] = float((feas.y_naive == feas.Y).mean())
    fp = feas.groupby("fund").apply(lambda d: pd.Series({
        "lstm": (d.y_pred == d.Y).mean(), "naive": (d.y_naive == d.Y).mean()}))
    m["lstm_precision_fundavg"] = float(fp["lstm"].mean())
    m["naive_precision_fundavg"] = float(fp["naive"].mean())
    m["n_predictions"] = int(len(P)); m["n_feasible"] = int(len(feas)); m["n_funds"] = int(feas.fund.nunique())
    m["eval_mode"] = cfg.eval_mode; m["eval_return"] = ret_col; m["eval_sort_lag"] = lag

    def xsq(s):
        v = s.dropna()
        if v.nunique() < 5:
            return pd.Series(np.nan, index=s.index)
        return (pd.qcut(v.rank(method="first"), 5, labels=False, duplicates="drop") + 1).reindex(s.index)

    # ---- fund-level: predictability + benchmark-adjusted future return ----
    P["wc"] = P["weight"] * P["_ret"]
    fq = P.groupby(["fund", "yq"], observed=True).apply(lambda d: pd.Series({
        "fund_ret": d.wc.sum() / d.weight.sum() if d.weight.sum() > 0 else np.nan,
        "prec": (d.loc[d.feasible, "y_pred"] == d.loc[d.feasible, "Y"]).mean()})).reset_index()
    fq["abn"] = fq["fund_ret"] - fq.groupby("yq")["fund_ret"].transform("mean")   # vs universe mean
    fq = fq.sort_values(["fund", "yq"])
    fq["prec_lag"] = fq.groupby("fund", observed=True)["prec"].shift(lag)
    for h in range(1, 5):
        fq[f"cabn{h}"] = fq.groupby("fund")["abn"].rolling(h).sum().shift(-(h - 1)).reset_index(0, drop=True)
    fq["Q"] = fq.groupby("yq")["prec_lag"].transform(xsq)
    rowsX = []
    for q in [1, 2, 3, 4, 5]:
        r = {"quintile": f"Q{q}"}
        for h in range(1, 5):
            s = fq[fq.Q == q].groupby("yq")[f"cabn{h}"].mean()
            r[f"cum_abn_{h}q"] = s.mean(); r[f"t_{h}q"] = _t(s)
        rowsX.append(r)
    r = {"quintile": "Q5-Q1"}
    for h in range(1, 5):
        d = (fq[fq.Q == 5].groupby("yq")[f"cabn{h}"].mean() - fq[fq.Q == 1].groupby("yq")[f"cabn{h}"].mean()).dropna()
        r[f"cum_abn_{h}q"] = d.mean(); r[f"t_{h}q"] = _t(d)
    rowsX.append(r)
    tableX = pd.DataFrame(rowsX)
    m["tableX_Q5mQ1_4q"] = float(tableX.iloc[-1]["cum_abn_4q"]); m["tableX_Q5mQ1_4q_t"] = float(tableX.iloc[-1]["t_4q"])

    # ---- Table XI: correct vs incorrect positions ----
    # TIMING handled by eval_mode: in "predictive" the return (_ret = t+1->t+2) already
    # starts AFTER correct(t) is revealed, so lag=0. In "lagged" we shift correct by 1.
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
    m["correct_minus_incorrect"] = float(diff.mean()); m["correct_minus_incorrect_t"] = float(_t(diff))

    # ---- Table XII: stock quintiles on cross-fund prediction accuracy ----
    stk = P.groupby(["security", "yq"], observed=True).agg(
        acc=("correct", "mean"), fwd=("_ret", "first")).reset_index()
    stk = stk.sort_values(["security", "yq"])
    # same timing rule as above, driven by eval_mode
    stk["acc_s"] = stk.groupby("security", observed=True)["acc"].shift(lag) if lag else stk["acc"]
    stk = stk.dropna(subset=["acc_s", "fwd"])
    stk["Q"] = stk.groupby("yq")["acc_s"].transform(xsq)
    stk = stk.dropna(subset=["Q"])
    rowsXII = [{"quintile": f"Q{q}", "mean_qret": stk[stk.Q == q].groupby("yq")["fwd"].mean().mean(),
                "t": _t(stk[stk.Q == q].groupby("yq")["fwd"].mean())} for q in [1, 2, 3, 4, 5]]
    ls = (stk[stk.Q == 1].groupby("yq")["fwd"].mean() - stk[stk.Q == 5].groupby("yq")["fwd"].mean()).dropna()
    rowsXII.append({"quintile": "Q1-Q5", "mean_qret": ls.mean(), "t": _t(ls)})
    tableXII = pd.DataFrame(rowsXII)
    m["tableXII_Q1mQ5"] = float(ls.mean()); m["tableXII_Q1mQ5_t"] = float(_t(ls))
    m["_ls_cum"] = ls.sort_index().cumsum()          # for the figure
    m["_fund_prec"] = fp                              # for the distribution figure

    return m, {"tableX": tableX, "tableXI": tableXI, "tableXII": tableXII}


def run_rank_sweep(cfg: Config = None, ranks=(10, 25), verbose=True):
    """Run the FULL pipeline at several rank cutoffs and compare.

    Unlike eval_mode, max_rank changes the panel itself (which positions exist),
    so every cutoff needs its own training run. Returns (summary DataFrame,
    {rank: full result dict}).

    Reading it: a tighter cutoff keeps only the manager's largest, highest-conviction
    positions. The paper argues big positions are traded more dynamically and are
    therefore HARDER to predict -- so precision should fall as max_rank shrinks.
    """
    from dataclasses import replace
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


def compare_eval_modes(preds: pd.DataFrame, cfg: Config = None,
                       modes=("contemporaneous", "lagged", "predictive")):
    """Re-evaluate the SAME predictions under each timing convention (training is
    independent of it, so this is nearly free). If the spread only shows up under
    'contemporaneous', it is same-quarter co-movement -- not predictive alpha."""
    from dataclasses import replace
    cfg = cfg or Config()
    rows = []
    for md in modes:
        try:
            mm, _ = evaluate(preds, replace(cfg, eval_mode=md))
            rows.append({"eval_mode": md, "return_used": mm["eval_return"], "sort_lag": mm["eval_sort_lag"],
                         "TableXII_Q1mQ5": mm["tableXII_Q1mQ5"], "XII_t": mm["tableXII_Q1mQ5_t"],
                         "TableXI_corr_minus_inc": mm["correct_minus_incorrect"],
                         "XI_t": mm["correct_minus_incorrect_t"],
                         "TableX_Q5mQ1_4q": mm["tableX_Q5mQ1_4q"], "X_t": mm["tableX_Q5mQ1_4q_t"]})
        except Exception as e:
            rows.append({"eval_mode": md, "error": str(e)[:70]})
    return pd.DataFrame(rows)


# ============================================================ ORCHESTRATE
def run(cfg: Config = None, verbose=True):
    """Full pipeline. Returns dict: panel, predictions, metrics, tables, figures."""
    cfg = cfg or Config()
    if cfg.save_outputs:
        os.makedirs(cfg.out_dir, exist_ok=True)
    panel = load_and_prepare(cfg)
    if verbose:
        bal = panel.dropna(subset=["Y"])["Y"].value_counts(normalize=True).round(3).to_dict()
        er = panel.attrs.get("eff_rank")
        print(f"[data] rows={len(panel):,} funds={panel.fund.nunique()} "
              f"quarters={panel.qi.nunique()} max_rank={er} class_balance={bal}")
    preds = run_model(panel, cfg, verbose=verbose)
    metrics, tables = evaluate(preds, cfg)
    figs = _figures(metrics, cfg)
    clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
    if verbose:
        print("\n=== PREDICTABILITY ===")
        print(f"  LSTM  precision: pooled {clean['lstm_precision_pooled']:.3f} | "
              f"fund-avg {clean['lstm_precision_fundavg']:.3f}   (paper 0.71)")
        print(f"  Naive precision: pooled {clean['naive_precision_pooled']:.3f} | "
              f"fund-avg {clean['naive_precision_fundavg']:.3f}   (paper 0.52)")
        print(f"\n  [eval_mode={clean['eval_mode']} | return={clean['eval_return']} | sort_lag={clean['eval_sort_lag']}]")
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


def _figures(metrics, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = {}
    if cfg.save_outputs:
        os.makedirs(cfg.out_dir, exist_ok=True)
    # fund precision distribution
    fp = metrics["_fund_prec"]
    fig1, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fp["naive"].dropna(), bins=30, alpha=.5, label=f"Naive ({fp['naive'].mean():.2f})", color="tab:red")
    ax.hist(fp["lstm"].dropna(), bins=30, alpha=.6, label=f"LSTM ({fp['lstm'].mean():.2f})", color="tab:blue")
    ax.axvline(.5, ls="--", c="k", lw=1); ax.set_xlabel("per-fund precision"); ax.set_ylabel("# funds")
    ax.set_title("Fund-level trade-direction predictability"); ax.legend()
    fig1.tight_layout()
    # cumulative Q1-Q5 stock long-short
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    metrics["_ls_cum"].plot(ax=ax2)
    ax2.set_title("Cumulative Q1-Q5 (least - most predictable stocks)")
    ax2.set_ylabel("cumulative quarterly return"); fig2.tight_layout()
    if cfg.save_outputs:
        p1, p2 = f"{cfg.out_dir}/fig_precision_dist.png", f"{cfg.out_dir}/fig_stock_ls.png"
        fig1.savefig(p1, dpi=130); fig2.savefig(p2, dpi=130); paths = {"precision_dist": p1, "stock_ls": p2}
    return {"precision_dist": fig1, "stock_ls": fig2, "paths": paths}


if __name__ == "__main__":
    run(Config())
