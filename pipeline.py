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
        "future_1q_ret": "future_1q_ret", "portfolio_value": "portfolio_value",
        "weight": "weight", "rank": "rank", "n_holdings": "n_holdings",
    })
    # ---- sample filters (paper Sec 3.1) ----
    us_only: bool = True
    min_years: int = 7          # >7 calendar years of history
    min_holdings: int = 10      # >=10 securities per quarter
    n_slots: int = 100          # keep top-N ranked positions per fund-quarter
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
    # ---- CPU performance ----
    n_jobs: int = -1               # per-fund: parallel funds across cores (-1 = all). 1 = serial.
    torch_threads: int = 0         # intra-op threads. 0 = auto (1 per worker for per-fund; all cores for global)
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
    df = pd.read_parquet(cfg.data_path)
    inv = {v: k for k, v in cfg.col_map.items()}          # internal -> your column
    df = df.rename(columns={inv[c]: c for c in cfg.col_map.values() if inv[c] in df.columns})
    need = ["fund", "date", "security", "shares"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"missing required columns after mapping: {miss}")

    df["date"] = pd.to_datetime(df["date"])
    df["yq"] = df["date"].dt.to_period("Q")
    if cfg.us_only and "isUs" in df.columns:
        df = df[df["isUs"].astype(bool)]
    # one row per (fund, quarter, security) -- keep the latest report in the quarter
    df = df.sort_values("date").drop_duplicates(["fund", "yq", "security"], keep="last")
    if "rank" in df.columns:
        df = df[df["rank"] <= cfg.n_slots]

    # fill optional columns so features always exist
    for c, d in [("weight", np.nan), ("rank", np.nan), ("position_value", np.nan),
                 ("portfolio_value", np.nan), ("market_cap", np.nan),
                 ("quarterly_ret", np.nan), ("past_1q_ret", np.nan),
                 ("future_1q_ret", np.nan), ("n_holdings", np.nan)]:
        if c not in df.columns:
            df[c] = d

    df = df.sort_values(["fund", "security", "yq"]).reset_index(drop=True)
    g = df.groupby(["fund", "security"], sort=False)

    # own-position dynamics (all lagged / known at t)
    for k in range(1, 7):
        df[f"sh_lag{k}"] = g["shares"].shift(k)
    df["w_lag1"] = g["weight"].shift(1)
    df["dw"] = df["weight"] - df["w_lag1"]
    df["pdsh"] = (df["shares"] - df["sh_lag1"]) / (df["sh_lag1"].abs() + 1.0)   # past trade into t
    df["pdsh_sign"] = np.sign(df["pdsh"]).fillna(0.0)
    df["pdsh_lag1"] = g["pdsh"].shift(1)
    df["log_posval"] = np.log(df["position_value"].abs() + 1.0)
    df["log_pv"] = np.log(df["portfolio_value"].abs() + 1.0)
    df["log_mktcap"] = np.log(df["market_cap"].abs() + 1.0)

    # TARGET: sign of NEXT-quarter fractional share change (+-band). label, not a feature.
    sh_next = g["shares"].shift(-1)
    dsh = (sh_next - df["shares"]) / (df["shares"].abs() + 1.0)
    df["Y"] = np.select([dsh <= -cfg.change_band, dsh >= cfg.change_band], [-1.0, 1.0], default=0.0)
    df.loc[dsh.isna(), "Y"] = np.nan
    df["fwd_qret"] = df["future_1q_ret"]                   # realized next-qtr return (eval only)

    # fund-level filters
    cnt = df.groupby(["fund", "yq"])["security"].transform("count")
    df = df[cnt >= cfg.min_holdings]
    nq = df.groupby("fund")["yq"].transform("nunique")
    df = df[nq >= cfg.min_years * 4]

    # peer activity across the universe (lagged one quarter -> known at t)
    lab = df.dropna(subset=["Y"])
    rate = lab.groupby("yq").agg(
        peer_buy=("Y", lambda s: (s > 0).mean()),
        peer_sell=("Y", lambda s: (s < 0).mean()),
        peer_hold=("Y", lambda s: (s == 0).mean())).sort_index().shift(1)
    df = df.merge(rate, left_on="yq", right_index=True, how="left")

    # fund past-quarter return proxy (weight-weighted realized holding return)
    df["_contrib"] = df["w_lag1"] * df["quarterly_ret"]
    fr = df.groupby(["fund", "yq"])["_contrib"].sum().rename("fund_ret").reset_index()
    fr["fund_ret_l1"] = fr.sort_values("yq").groupby("fund")["fund_ret"].shift(1)
    df = df.merge(fr[["fund", "yq", "fund_ret_l1"]], on=["fund", "yq"], how="left").drop(columns="_contrib")

    # integer quarter index for windowing
    qs = pd.PeriodIndex(sorted(df["yq"].unique()), freq="Q")
    qmap = {q: i for i, q in enumerate(qs)}
    df["qi"] = df["yq"].map(qmap)
    df["held"] = 1
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
        vals = g[feat_cols].shift(k).values
        qik = g["qi"].shift(k).values
        heldk = g["held"].shift(k).fillna(0).values
        present = (qik == (qi - k)) & (heldk > 0)
        m = present[valid].astype(np.float32)
        X[:, step, :] = np.nan_to_num(vals[valid], nan=0.0, posinf=0.0, neginf=0.0) * m[:, None]
        mask[:, step] = m
    feasible = mask.all(axis=1)
    y = (sub["Y"].values[valid] + 1).astype(np.int64)          # {-1,0,1} -> {0,1,2}
    meta = sub.loc[valid, ["fund", "security", "yq", "qi", "Y", "fwd_qret", "weight", "rank"]].reset_index(drop=True)
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
    else:  # per_fund  (funds are independent -> one core per fund, 1 torch thread each)
        n_funds = int(panel["fund"].nunique())
        njobs = ncore if cfg.n_jobs in (-1, 0) else cfg.n_jobs
        # NB: iterate the groupby LAZILY (never build a list of all fund frames) so
        # memory stays bounded on the full "All_Funds" panel.
        if dev != "cpu" or njobs == 1:
            _set_threads(cfg.torch_threads or ncore)
            preds, acc = [], []
            for done, (f, fp) in enumerate(_iter_funds(panel), 1):
                outs = _fund_task(f, fp, feat, cfg, dev)
                preds += outs
                acc += [(o.loc[o.feasible, "y_pred"] == o.loc[o.feasible, "Y"]).mean() for o in outs]
                if verbose and done % 50 == 0:
                    print(f"  [per-fund] {done}/{n_funds} funds | running feas_acc={np.nanmean(acc):.3f}")
        else:
            try:
                from joblib import Parallel, delayed
            except Exception:
                Parallel = None
            if Parallel is None:
                if verbose:
                    print("  [per-fund] joblib not found -> serial. `pip install joblib` for multi-core.")
                preds = []
                for f, fp in _iter_funds(panel):
                    preds += _fund_task(f, fp, feat, cfg, "cpu")
            else:
                if verbose:
                    print(f"  [per-fund] {n_funds} funds on {njobs}/{ncore} cores (loky), lazy dispatch...")
                # generator + bounded pre_dispatch keeps every core fed while holding
                # only ~2*njobs fund frames in flight (memory-safe for all funds).
                gen = (delayed(_fund_task)(f, fp, feat, cfg, "cpu")
                       for f, fp in _iter_funds(panel))
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


def evaluate(preds: pd.DataFrame, cfg: Config):
    """Predictability + portfolio sorts (Tables X / XI / XII). Returns
    (metrics dict, {table_name: DataFrame})."""
    P = preds.copy()
    P["yq"] = P["yq"].astype("period[Q]")
    feas = P[P["feasible"]]
    m = {}
    m["lstm_precision_pooled"] = float((feas.y_pred == feas.Y).mean())
    m["naive_precision_pooled"] = float((feas.y_naive == feas.Y).mean())
    fp = feas.groupby("fund").apply(lambda d: pd.Series({
        "lstm": (d.y_pred == d.Y).mean(), "naive": (d.y_naive == d.Y).mean()}))
    m["lstm_precision_fundavg"] = float(fp["lstm"].mean())
    m["naive_precision_fundavg"] = float(fp["naive"].mean())
    m["n_predictions"] = int(len(P)); m["n_feasible"] = int(len(feas)); m["n_funds"] = int(feas.fund.nunique())

    def xsq(s):
        v = s.dropna()
        if v.nunique() < 5:
            return pd.Series(np.nan, index=s.index)
        return (pd.qcut(v.rank(method="first"), 5, labels=False, duplicates="drop") + 1).reindex(s.index)

    # ---- fund-level: predictability + benchmark-adjusted next-qtr return ----
    P["wc"] = P["weight"] * P["fwd_qret"]
    fq = P.groupby(["fund", "yq"]).apply(lambda d: pd.Series({
        "fund_ret": d.wc.sum() / d.weight.sum() if d.weight.sum() > 0 else np.nan,
        "prec": (d.loc[d.feasible, "y_pred"] == d.loc[d.feasible, "Y"]).mean()})).reset_index()
    fq["abn"] = fq["fund_ret"] - fq.groupby("yq")["fund_ret"].transform("mean")   # vs universe mean
    fq = fq.sort_values(["fund", "yq"])
    fq["prec_lag"] = fq.groupby("fund")["prec"].shift(1)
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
    P["correct"] = (P.y_pred == P.Y).astype(int)
    ci = P.dropna(subset=["fwd_qret"])
    corr = ci[ci.correct == 1].groupby("yq")["fwd_qret"].mean()
    inco = ci[ci.correct == 0].groupby("yq")["fwd_qret"].mean()
    diff = (corr - inco).dropna()
    tableXI = pd.DataFrame({"portfolio": ["Correct", "Incorrect", "Correct-Incorrect"],
                            "mean_qret": [corr.mean(), inco.mean(), diff.mean()],
                            "t": [_t(corr), _t(inco), _t(diff)]})
    m["correct_minus_incorrect"] = float(diff.mean()); m["correct_minus_incorrect_t"] = float(_t(diff))

    # ---- Table XII: stock quintiles on cross-fund prediction accuracy ----
    stk = P.groupby(["security", "yq"]).agg(acc=("correct", "mean"), fwd=("fwd_qret", "first")).reset_index()
    stk = stk.dropna(subset=["acc", "fwd"])
    stk["Q"] = stk.groupby("yq")["acc"].transform(xsq)
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


# ============================================================ ORCHESTRATE
def run(cfg: Config = None, verbose=True):
    """Full pipeline. Returns dict: panel, predictions, metrics, tables, figures."""
    cfg = cfg or Config()
    if cfg.save_outputs:
        os.makedirs(cfg.out_dir, exist_ok=True)
    panel = load_and_prepare(cfg)
    if verbose:
        bal = panel.dropna(subset=["Y"])["Y"].value_counts(normalize=True).round(3).to_dict()
        print(f"[data] rows={len(panel):,} funds={panel.fund.nunique()} "
              f"quarters={panel.qi.nunique()} class_balance={bal}")
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
        print(f"\n  Table XII Q1-Q5 = {clean['tableXII_Q1mQ5']*100:+.2f}%/qtr "
              f"(t={clean['tableXII_Q1mQ5_t']:.2f})   (paper +1.06%, t=5.74)")
        print(f"  Table XI  corr-incorr = {clean['correct_minus_incorrect']*100:+.2f}%/qtr "
              f"(t={clean['correct_minus_incorrect_t']:.2f})   (paper -0.23%, t=-12.4)")
    if cfg.save_outputs:
        json.dump(clean, open(f"{cfg.out_dir}/metrics.json", "w"), indent=2)
        for nm, tb in tables.items():
            tb.to_csv(f"{cfg.out_dir}/{nm}.csv", index=False)
        preds.to_parquet(f"{cfg.out_dir}/predictions.parquet", index=False)
    return {"panel": panel, "predictions": preds, "metrics": clean,
            "tables": tables, "figures": figs, "config": asdict(cfg)}


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
