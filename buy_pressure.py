"""buy_pressure.py -- predict next-quarter BUYING PRESSURE on a security, then test for alpha.

A security-level study, distinct from the fund-security work in company_replication.py.

    unit of observation : (security, quarter)
    target              : buy_frac(s, t+1) = # funds buying s / # funds owning s, next quarter
    features            : latest Barra GEMLT exposures as of quarter end
                          past security returns
                          current buy_frac / sell_frac and their lags, plus ownership breadth
    alpha test          : rank securities by PREDICTED buy_frac, then look at forward returns

TIMING -- the thing most easily got wrong here
----------------------------------------------
buy_frac over the window [t, t+1] compares holdings at t against holdings at t+1, so it is
NOT observable at t. It is known once the t+1 holdings exist, and public only after the
~45-60 day filing delay. Writing q for the quarter whose buy_frac we are predicting:

    features   must be dated at or before the start of q, and buy_frac features must refer
               to windows that CLOSED before q began
    target     buy_frac over q
    returns    "predictive"  starts at the beginning of q      -- no overlap, ignores filing lag
               "tradeable"   starts one quarter after q begins -- also clears the lag
               "contemporaneous" overlaps the target window    -- biased, benchmark only

`Config.eval_timing` selects which one the headline numbers use (default "predictive").

JOINING BARRA TO HOLDINGS
-------------------------
The Barra file carries `stkid`, which is the same security identifier the holdings panel
calls `security`, so the join is direct -- no mapping table. Both sides pass through
`_norm_id` first, since one may be an int and the other a zero-padded string ('0010001',
'10001.0' and 10001 all denote the same security). The loader prints match rates by row,
by security and by quarter, and refuses to continue on a near-empty join.
"""
from __future__ import annotations

__version__ = "2026.08.03.2"

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================ CONFIG
@dataclass
class Config:
    # ---- inputs ----
    holdings_path: str = ("manager_holdings/master_batches_return_filtered/"
                          "master_all_funds_add_filter_ivy_rank_active_rank.parquet")
    barra_path: str = "manager_holdings/barra_GEMLTL_R3000_Prod_new_MSCI_wkly.pickle"

    # ---- how the two datasets join ----
    # The Barra file carries `stkid`, which IS the holdings panel's `security` id, so the
    # join is direct. Both sides are normalised through _norm_id first, because one may be
    # an int and the other a zero-padded string ('0010001', '10001.0' and 10001 all mean
    # the same security). `sedol` is present too but is not needed for the join.
    barra_id_col: str = "stkid"
    min_match_rate: float = 0.05      # abort if fewer than this share of rows join

    # ---- holdings column names (same convention as the other module) ----
    col_map: dict = field(default_factory=lambda: {
        "fund": "fund", "date": "date", "security": "security", "shares": "shares",
        "position_value": "position_value", "market_cap": "market_cap", "isUs": "isUs",
        "quarterly_ret": "quarterly_ret", "past_1q_ret": "past_1q_ret",
        "future_1q_ret": "future_1q_ret", "future_2q_ret": "future_2q_ret",
        "future_3q_ret": "future_3q_ret", "InvTypeCode": "inv_type",
        "future_1q_shares_change_pct": "chg_pct", "weight": "weight", "rank": "rank",
    })
    inv_type_codes: tuple = (401,)
    us_only: bool = True
    max_rank: Optional[int] = None    # None = use every position; buying pressure is a
                                      # breadth measure, so truncating the book distorts it
    change_band: float = 0.01         # +-1% dead band, as in the fund-level work
    drop_missing_position: bool = True

    # ---- sample filters at the SECURITY level ----
    min_owners: int = 5               # a buy_frac from 2 funds is noise
    min_quarters: int = 12            # securities need some history to be sortable

    # ---- evaluation ----
    eval_timing: str = "predictive"   # predictive | tradeable | contemporaneous
    n_quintiles: int = 5

    # ---- rolling design ----
    window_q: int = 28
    test_q: int = 8
    step: int = 8

    # ---- model ----
    # "gbm"   gradient boosting on the current row; history enters through explicit lags
    # "ridge" linear benchmark, useful for checking the signal is not a tree artifact
    # "lstm"  one sample = one security's last `seq_len` quarters, [T, F]; the network sees
    #         the trajectory of exposures and buying pressure rather than flattened lags
    model: str = "gbm"
    max_iter: int = 300               # gbm
    learning_rate: float = 0.06
    max_depth: int = 6
    seed: int = 0
    # ---- lstm ----
    seq_len: int = 8                  # quarters of history per sample
    hidden: int = 64
    dropout: float = 0.25
    lr: float = 3e-3
    max_epochs: int = 40
    patience: int = 6
    batch: int = 4096
    device: str = "auto"              # "auto" | "cuda" | "cpu"
    lstm_max_train: Optional[int] = None   # None = all sequences

    # ---- Barra ----
    barra_prefix: str = "GEMLT_"
    barra_cols: Optional[List[str]] = None   # None = every column with the prefix
    winsorize: float = 0.01                  # clip exposures at these tails, per quarter


# ============================================================ HELPERS
def _t(x, lags=0):
    """t-stat of the mean; lags>0 applies Newey-West (Bartlett)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 2 or x.std() == 0:
        return np.nan
    if lags <= 0:
        return x.mean() / (x.std(ddof=1) / np.sqrt(n))
    e = x - x.mean(); var = float(e @ e) / n
    for L in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - L / (lags + 1.0)) * float(e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(var / n) if var > 0 else np.nan


def _norm_id(s: pd.Series) -> pd.Series:
    """Canonical string id so an int `security` joins a string `stkid`.

    Numeric-looking values collapse to their integer string ('10001.0', 10001 and '0010001'
    all become '10001'); anything else is stripped text. Same convention the wrds_pull
    signal code uses, so ids stay consistent across the two code bases.
    """
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().all():
        return v.astype("int64").astype(str)
    out = s.astype(str).str.strip()
    ok = v.notna()
    if ok.any():
        out.loc[ok] = v[ok].astype("int64").astype(str)
    return out


def _qcut(s, n=5):
    return (pd.qcut(s.rank(method="first"), n, labels=False, duplicates="drop") + 1
            if s.nunique() >= n else pd.Series(np.nan, index=s.index))


def check_version(verbose=True):
    path = os.path.abspath(__file__)
    if verbose:
        print(f"buy_pressure {__version__}  |  {path}")
    return __version__


# ============================================================ 1. BARRA
def inspect_barra(cfg: Config, n=5):
    """Print the pickle's shape before committing to a parse. Run this first."""
    df = pd.read_pickle(cfg.barra_path)
    print(f"type {type(df).__name__}  shape {getattr(df, 'shape', '?')}")
    if isinstance(df, pd.DataFrame):
        print(f"columns ({len(df.columns)}): {list(df.columns)[:40]}")
        gem = [c for c in df.columns if str(c).startswith(cfg.barra_prefix)]
        print(f"\n{len(gem)} columns with prefix {cfg.barra_prefix!r}: {gem[:30]}")
        for idc in (cfg.barra_id_col, "sedol", "day"):
            if idc in df.columns:
                print(f"  {idc:8s} dtype={df[idc].dtype}  n_unique={df[idc].nunique():,}  "
                      f"sample={list(df[idc].dropna().unique()[:4])}")
        print(f"\nhead:\n{df.head(n)}")
    return df


def load_barra(cfg: Config) -> pd.DataFrame:
    """Weekly Barra exposures -> ONE row per (sedol, quarter): the LAST weekly observation
    on or before quarter end. Using the last observation inside the quarter keeps the
    feature dated at the quarter boundary; nothing from the following quarter is used.
    """
    df = pd.read_pickle(cfg.barra_path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a DataFrame in {cfg.barra_path}, got {type(df)}")
    idc = cfg.barra_id_col
    need = {"day", idc}
    df = df.reset_index() if not need.issubset(df.columns) else df
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"barra file is missing {miss}; columns are {list(df.columns)[:30]}")

    cols = cfg.barra_cols or [c for c in df.columns if str(c).startswith(cfg.barra_prefix)]
    if not cols:
        raise ValueError(f"no columns start with {cfg.barra_prefix!r}")
    df = df[["day", idc] + cols].copy()
    df["security_id"] = _norm_id(df[idc])
    df["day"] = pd.to_datetime(df["day"])
    df["yq"] = df["day"].dt.to_period("Q")
    n_weekly = len(df)
    # LAST weekly snapshot inside each quarter -- dated at the quarter boundary, so nothing
    # from the following quarter enters the feature
    df = df.sort_values("day").drop_duplicates(["security_id", "yq"], keep="last")
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    print(f"[barra] {n_weekly:,} weekly rows -> {len(df):,} security-quarters "
          f"(last obs per quarter)")
    print(f"[barra] {df.security_id.nunique():,} securities | {df.yq.nunique()} quarters "
          f"| {len(cols)} exposures")
    print(f"[barra] exposure coverage (non-null share):")
    print(df[cols].notna().mean().round(4).to_string())
    return df.drop(columns=["day", idc])


# ============================================================ 2. HOLDINGS -> BREADTH
def load_buy_pressure(cfg: Config) -> pd.DataFrame:
    """From the fund-security-quarter holdings, build SECURITY-quarter buying pressure.

        n_owning(s,q)  funds holding s with a usable label for the q -> q+1 window
        n_buying(s,q)  of those, how many increased shares by more than the dead band
        buy_frac(s,q)  n_buying / n_owning        <- the target, and its own lag as a feature

    buy_frac(s,q) describes the window q -> q+1 and so is only observable at q+1. Lags of it
    are therefore the only version safe to use as a feature for predicting a later window.
    """
    inv = {v: k for k, v in cfg.col_map.items()}
    want = ["fund", "date", "security", "shares", "position_value", "market_cap",
            "quarterly_ret", "past_1q_ret", "future_1q_ret", "future_2q_ret",
            "future_3q_ret", "chg_pct", "inv_type", "weight", "rank", "isUs"]
    raw = [inv[c] for c in want if c in inv]
    try:
        import pyarrow.parquet as pq
        avail = set(pq.ParquetFile(cfg.holdings_path).schema.names)
        keep = [c for c in raw if c in avail]
        df = pd.read_parquet(cfg.holdings_path, columns=keep or None)
    except Exception:
        df = pd.read_parquet(cfg.holdings_path)
    df = df.rename(columns={inv[c]: c for c in cfg.col_map.values() if inv[c] in df.columns})

    df["date"] = pd.to_datetime(df["date"])
    df["yq"] = df["date"].dt.to_period("Q")
    if cfg.us_only and "isUs" in df.columns:
        df = df[df["isUs"].astype(bool)]
    if cfg.inv_type_codes is not None and "inv_type" in df.columns:
        df = df[df["inv_type"].astype(str).isin({str(c) for c in cfg.inv_type_codes})]
    if cfg.max_rank and "rank" in df.columns:
        df = df[df["rank"] <= cfg.max_rank]
    df = df.sort_values("date").drop_duplicates(["fund", "yq", "security"], keep="last")

    for c in ("shares", "chg_pct", "position_value", "market_cap", "weight",
              "quarterly_ret", "future_1q_ret", "future_2q_ret", "future_3q_ret"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    # ---- trade direction over q -> q+1, same convention as the fund-level module ----
    if "chg_pct" in df.columns and df["chg_pct"].notna().any():
        chg = df["chg_pct"].astype("float64")
        nz = chg[chg.abs() > 1e-9].abs()
        med = float(nz.median()) if len(nz) else np.nan
        dsh = chg / 100.0 if (np.isfinite(med) and med > 1.5) else chg
        if cfg.drop_missing_position:
            bad = (dsh + 1.0).abs() < 1e-6      # -100% = position info missing, not a sale
            print(f"[hold] prefilter: dropped {int(bad.sum()):,} rows ({bad.mean():.1%}) "
                  f"with chg_pct = -100%")
            df, dsh = df[~bad], dsh[~bad]
    else:
        df = df.sort_values(["fund", "security", "yq"])
        g = df.groupby(["fund", "security"], observed=True)
        shn, qn = g["shares"].shift(-1), g["yq"].shift(-1)
        shn = shn.where(qn == df["yq"] + 1)                  # exact next quarter only
        dsh = (shn - df["shares"]) / (df["shares"].abs() + 1.0)
    df["Y"] = np.select([dsh <= -cfg.change_band, dsh >= cfg.change_band],
                        [-1.0, 1.0], default=0.0)
    df.loc[pd.isna(dsh), "Y"] = np.nan

    lab = df[df["Y"].notna()].copy()
    lab["is_buy"] = (lab["Y"] > 0).astype("float32")
    lab["is_sell"] = (lab["Y"] < 0).astype("float32")
    sq = lab.groupby(["security", "yq"], observed=True).agg(
        n_owning=("fund", "size"),
        buy_frac=("is_buy", "mean"),
        sell_frac=("is_sell", "mean"),
        w_mean=("weight", "mean"),
        mktcap=("market_cap", "first"),
        ret_q=("quarterly_ret", "first"),
        fwd_1q=("future_1q_ret", "first"),
        fwd_2q=("future_2q_ret", "first"),
        fwd_3q=("future_3q_ret", "first"),
    ).reset_index()
    sq["n_buying"] = (sq["buy_frac"] * sq["n_owning"]).round()
    sq = sq[sq["n_owning"] >= cfg.min_owners]
    print(f"[hold] {len(sq):,} security-quarters | {sq.security.nunique():,} securities | "
          f"{sq.yq.nunique()} quarters | median owners {sq.n_owning.median():.0f}")
    print(f"[hold] buy_frac  mean {sq.buy_frac.mean():.3f}  sd {sq.buy_frac.std():.3f}")

    return sq


def report_match(sq: pd.DataFrame, ba: pd.DataFrame, cfg: Config) -> None:
    """Coverage of the id join, reported the way the wrds_pull signal code does it:
    by row, by distinct security, and by quarter -- a high row rate can still hide whole
    quarters or a whole class of securities being absent."""
    hold_ids, barra_ids = set(sq["security_id"]), set(ba["security_id"])
    inter = hold_ids & barra_ids
    print("\n--- id join coverage ---")
    print(f"  holdings securities   {len(hold_ids):>7,}")
    print(f"  barra securities      {len(barra_ids):>7,}")
    print(f"  in both               {len(inter):>7,}   "
          f"({len(inter)/max(len(hold_ids),1):.1%} of holdings)")
    on_key = sq.merge(ba[["security_id", "yq"]].assign(_hit=1),
                      on=["security_id", "yq"], how="left")
    print(f"  security-quarters hit {float(on_key['_hit'].notna().mean()):.1%}")
    byq = on_key.groupby("yq")["_hit"].apply(lambda s: float(s.notna().mean()))
    print(f"  per-quarter hit rate  min={byq.min():.1%}  med={byq.median():.1%}  "
          f"max={byq.max():.1%}")
    worst = byq.nsmallest(4)
    if len(worst) and worst.iloc[0] < 0.5:
        print(f"  weakest quarters: {', '.join(f'{q}={v:.0%}' for q, v in worst.items())}")
        print("  (a quarter near 0% usually means the Barra file starts later than the "
              "holdings, or those quarters are missing from the weekly file)")


# ============================================================ 3. PANEL
def build_panel(cfg: Config = None) -> pd.DataFrame:
    """Security-quarter panel: Barra exposures + returns + buying-pressure history + target."""
    cfg = cfg or Config()
    sq = load_buy_pressure(cfg)
    ba = load_barra(cfg)
    sq["security_id"] = _norm_id(sq["security"])       # both sides to one canonical form
    report_match(sq, ba, cfg)

    before = len(sq)
    df = sq.merge(ba, on=["security_id", "yq"], how="left")
    gem = [c for c in df.columns if str(c).startswith(cfg.barra_prefix)]
    rate = float(df[gem].notna().any(axis=1).mean()) if gem else 0.0
    print(f"\n[join] {before:,} security-quarters | Barra exposures present on {rate:.1%}")
    if rate < cfg.min_match_rate:
        raise ValueError(
            f"only {rate:.1%} of security-quarters carry Barra exposures. The ids are "
            f"probably not the same vintage -- inspect a few values of `{cfg.barra_id_col}` "
            "against the holdings `security` (run inspect_barra), and check the two files "
            "cover overlapping dates.")

    # ---- winsorise exposures per quarter, then cross-sectionally standardise ----
    for c in gem:
        lo = df.groupby("yq")[c].transform(lambda s: s.quantile(cfg.winsorize))
        hi = df.groupby("yq")[c].transform(lambda s: s.quantile(1 - cfg.winsorize))
        df[c] = df[c].clip(lo, hi)
        mu = df.groupby("yq")[c].transform("mean")
        sd = df.groupby("yq")[c].transform("std")
        df[c] = ((df[c] - mu) / sd.where(sd > 0)).astype("float32")

    # ---- integer quarter index ----
    qs = pd.PeriodIndex(sorted(df["yq"].unique()), freq="Q")
    df["qi"] = df["yq"].map({q: i for i, q in enumerate(qs)}).astype("int32")
    df = df.sort_values(["security", "qi"]).reset_index(drop=True)
    g = df.groupby("security", observed=True)
    qi = df["qi"].to_numpy()

    # ---- buying-pressure history. buy_frac(q) covers q -> q+1 and is known only at q+1,
    #      so ONLY ITS LAGS may be used to predict a later window. ----
    for k in (1, 2, 3, 4):
        for col in ("buy_frac", "sell_frac", "n_owning"):
            v, qq = g[col].shift(k), g["qi"].shift(k)
            df[f"{col}_lag{k}"] = v.where(qq == qi - k).astype("float32")
    df["buy_frac_ma4"] = df[[f"buy_frac_lag{k}" for k in (1, 2, 3, 4)]].mean(axis=1)
    df["buy_frac_chg"] = df["buy_frac_lag1"] - df["buy_frac_lag2"]
    df["log_owners"] = np.log(df["n_owning"].astype("float64") + 1.0).astype("float32")
    df["log_mktcap"] = np.log(df["mktcap"].abs() + 1.0).astype("float32")

    # ---- past returns, exact-quarter aligned ----
    for k in (1, 2, 4):
        v, qq = g["ret_q"].shift(k), g["qi"].shift(k)
        df[f"ret_lag{k}"] = v.where(qq == qi - k).astype("float32")
    df["ret_ma4"] = df[["ret_lag1", "ret_lag2", "ret_lag4"]].mean(axis=1)

    # ---- TARGET: buy_frac of the NEXT quarter, exact ----
    v, qq = g["buy_frac"].shift(-1), g["qi"].shift(-1)
    df["target_buy_frac"] = v.where(qq == qi + 1).astype("float32")

    nq = g["qi"].transform("size")
    df = df[nq >= cfg.min_quarters]
    print(f"[panel] {len(df):,} rows | {df.security.nunique():,} securities | "
          f"{df.qi.max()+1} quarters | target available {df.target_buy_frac.notna().mean():.1%}")
    return df


def feature_list(df: pd.DataFrame, cfg: Config) -> List[str]:
    """Everything dated at or before the start of the target window."""
    gem = [c for c in df.columns if str(c).startswith(cfg.barra_prefix)]
    hist = [c for c in df.columns
            if c.startswith(("buy_frac_lag", "sell_frac_lag", "n_owning_lag"))]
    other = ["buy_frac_ma4", "buy_frac_chg", "log_owners", "log_mktcap", "w_mean",
             "ret_lag1", "ret_lag2", "ret_lag4", "ret_ma4"]
    return gem + hist + [c for c in other if c in df.columns]


# ============================================================ 4. MODELS
_KEEP = ["security", "qi", "yq", "target_buy_frac", "buy_frac", "n_owning",
         "fwd_1q", "fwd_2q", "fwd_3q"]


def _pick_device(cfg):
    """Choose a device, checking this torch build actually supports the card. A new GPU on
    an old torch crashes inside cuDNN; falling back with a message beats crashing."""
    import torch
    if cfg.device != "auto":
        return cfg.device
    if not torch.cuda.is_available():
        return "cpu"
    try:
        cap = torch.cuda.get_device_capability(0)
        sm = f"sm_{cap[0]}{cap[1]}"
        if sm not in torch.cuda.get_arch_list():
            print(f"  [warn] GPU {sm} unsupported by torch {torch.__version__} -> CPU. "
                  f"For RTX 50-series install a cu128 build.")
            return "cpu"
    except Exception:
        return "cpu"
    return "cuda"


def _build_sequences(df: pd.DataFrame, feats: List[str], seq_len: int):
    """Sequence INDICES, not a materialised [N, T, F] tensor.

    Keeps `Feat` [n_rows, F] plus `hist` [N, T] int32 and assembles each batch on the fly
    via Feat[hist[bi]]. A quarter missing for that security gets mask 0; the label sits at
    the last step. Every step is an exact quarter -- never a stale row pulled across a gap.
    """
    df = df.sort_values(["security", "qi"]).reset_index(drop=True)
    g = df.groupby("security", observed=True, sort=False)
    valid = df["target_buy_frac"].notna().to_numpy()
    N = int(valid.sum())
    if N == 0:
        return None
    qi = df["qi"].to_numpy()
    Feat = np.nan_to_num(df[feats].to_numpy(dtype="float32", na_value=np.nan),
                         nan=0.0, posinf=0.0, neginf=0.0)
    row = pd.Series(np.arange(len(df), dtype="int64"), index=df.index)
    hist = np.zeros((N, seq_len), dtype=np.int32)
    M = np.zeros((N, seq_len), dtype=np.float32)
    for k in range(seq_len):
        step = seq_len - 1 - k                       # k=0 is the current quarter -> last
        rk = row.groupby(df["security"], observed=True, sort=False).shift(k).to_numpy(
            dtype="float64", na_value=np.nan)
        qk = g["qi"].shift(k).to_numpy(dtype="float64", na_value=np.nan)
        pres = (qk == qi - k) & ~np.isnan(rk)
        pv = pres[valid]
        M[:, step] = pv.astype(np.float32)
        hist[:, step] = np.where(pv, np.nan_to_num(rk[valid], nan=0.0), 0).astype(np.int32)
    y = df["target_buy_frac"].to_numpy("float32")[valid]
    meta = df.loc[valid, [c for c in _KEEP if c in df.columns]].reset_index(drop=True)
    return Feat, hist, M, y, meta


def _fit_lstm(Feat, hist, M, y, tr, te, cfg):
    """Sequence LSTM regressor on next-quarter buy_frac. Returns predictions for `te`."""
    import torch, torch.nn as nn
    dev = _pick_device(cfg)
    torch.manual_seed(cfg.seed)
    F = Feat.shape[1]
    tr_i = np.where(tr)[0]
    if len(tr_i) < 100:
        return None, dev
    used = np.unique(hist[tr_i][M[tr_i] > 0])
    if used.size < 50:
        return None, dev
    if used.size > 500_000:
        used = np.random.default_rng(cfg.seed).choice(used, 500_000, replace=False)
    mu = Feat[used].mean(0).astype(np.float32)
    sd = (Feat[used].std(0) + 1e-6).astype(np.float32)

    def batch(bi):
        xb = (Feat[hist[bi]] - mu) / sd
        xb *= M[bi][..., None]
        return torch.from_numpy(xb), torch.from_numpy(M[bi])

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(F, cfg.hidden, batch_first=True)
            self.drop = nn.Dropout(cfg.dropout)
            self.head = nn.Linear(cfg.hidden, 1)

        def forward(self, x, m):
            o, _ = self.lstm(x * m.unsqueeze(-1))
            return self.head(self.drop(o[:, -1, :])).squeeze(-1)

    model = Net().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = nn.MSELoss()
    yt = torch.from_numpy(y)
    idx = tr_i.copy()
    rng = np.random.default_rng(cfg.seed); rng.shuffle(idx)
    if cfg.lstm_max_train and len(idx) > cfg.lstm_max_train:
        idx = idx[:cfg.lstm_max_train]
    nval = max(1, int(0.15 * len(idx)))
    val_i, trn_i = idx[:nval], idx[nval:]
    best, best_state, bad = 1e9, None, 0
    for _ in range(cfg.max_epochs):
        model.train(); rng.shuffle(trn_i)
        for b in range(0, len(trn_i), cfg.batch):
            bi = trn_i[b:b + cfg.batch]
            xb, mb = batch(bi)
            opt.zero_grad()
            lossf(model(xb.to(dev), mb.to(dev)), yt[bi].to(dev)).backward()
            opt.step()
        model.eval(); tot = n = 0
        with torch.inference_mode():
            for b in range(0, len(val_i), cfg.batch):
                bi = val_i[b:b + cfg.batch]
                xb, mb = batch(bi)
                tot += float(lossf(model(xb.to(dev), mb.to(dev)), yt[bi].to(dev))) * len(bi)
                n += len(bi)
        vl = tot / max(n, 1)
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    te_i = np.where(te)[0]
    pr = []
    with torch.inference_mode():
        for b in range(0, len(te_i), cfg.batch):
            bi = te_i[b:b + cfg.batch]
            xb, mb = batch(bi)
            pr.append(model(xb.to(dev), mb.to(dev)).cpu().numpy())
    return (np.concatenate(pr) if pr else None), dev


def run_model(df: pd.DataFrame, cfg: Config, verbose=True) -> pd.DataFrame:
    """Rolling-window out-of-sample regression on next-quarter buy_frac.
    cfg.model selects gbm / ridge / lstm; the split scheme is identical across all three."""
    feats = feature_list(df, cfg)
    d = df[df["target_buy_frac"].notna()]
    out = []

    if cfg.model == "lstm":
        seq = _build_sequences(df, feats, cfg.seq_len)
        if seq is None:
            raise RuntimeError("no usable sequences")
        Feat, hist, M, y, meta = seq
        sqi = meta["qi"].to_numpy()
        if verbose:
            naive = len(y) * cfg.seq_len * len(feats) * 4 / 1e9
            used = (Feat.nbytes + hist.nbytes + M.nbytes) / 1e9
            print(f"  [lstm] {len(y):,} sequences x T={cfg.seq_len} x F={len(feats)}"
                  f"  memory {used:.2f} GB (materialising would need {naive:.1f} GB)")

    for c in range(cfg.window_q, int(d.qi.max()) + 2, cfg.step):
        if cfg.model == "lstm":
            tr = (sqi >= c - cfg.window_q) & (sqi < c - cfg.test_q)
            te = (sqi >= c - cfg.test_q) & (sqi < c)
            if tr.sum() < 500 or te.sum() == 0:
                continue
            yp, dev = _fit_lstm(Feat, hist, M, y, tr, te, cfg)
            if yp is None:
                continue
            p = meta.iloc[np.where(te)[0]].copy()
            p["pred_buy_frac"] = yp
        else:
            tr = d[(d.qi >= c - cfg.window_q) & (d.qi < c - cfg.test_q)]
            te = d[(d.qi >= c - cfg.test_q) & (d.qi < c)]
            if len(tr) < 500 or len(te) == 0:
                continue
            X, yy = tr[feats].to_numpy("float32"), tr["target_buy_frac"].to_numpy("float32")
            if cfg.model == "ridge":
                from sklearn.linear_model import Ridge
                from sklearn.impute import SimpleImputer
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  Ridge(1.0))
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor
                m = HistGradientBoostingRegressor(max_iter=cfg.max_iter,
                                                  learning_rate=cfg.learning_rate,
                                                  max_depth=cfg.max_depth,
                                                  random_state=cfg.seed)
            m.fit(X, yy)
            p = te[[c_ for c_ in _KEEP if c_ in te.columns]].copy()
            p["pred_buy_frac"] = m.predict(te[feats].to_numpy("float32"))
            dev = "cpu"
        out.append(p)
        if verbose:
            ic = float(pd.Series(p.pred_buy_frac).corr(p.target_buy_frac, method="spearman"))
            print(f"  window {c:>3}  n_test={len(p):>7,}  rank-IC={ic:+.3f}  [{cfg.model}/{dev}]")
    if not out:
        raise RuntimeError("no predictions -- check window sizes against the panel length")
    return pd.concat(out, ignore_index=True)


# ============================================================ 5. EVALUATION
def prediction_quality(P: pd.DataFrame) -> pd.DataFrame:
    """How well is next-quarter buying pressure predicted at all?"""
    per_q = P.groupby("qi").apply(lambda d: pd.Series({
        "rank_ic": d["pred_buy_frac"].corr(d["target_buy_frac"], method="spearman"),
        "pearson": d["pred_buy_frac"].corr(d["target_buy_frac"]),
        "n": len(d)}))
    naive = P.groupby("qi").apply(
        lambda d: d["buy_frac"].corr(d["target_buy_frac"], method="spearman"))
    return pd.DataFrame([
        {"metric": "rank IC (model)", "mean": per_q["rank_ic"].mean(),
         "t": _t(per_q["rank_ic"]), "n_quarters": int(per_q["n"].size)},
        {"metric": "rank IC (naive: this quarter's buy_frac)", "mean": naive.mean(),
         "t": _t(naive), "n_quarters": int(naive.size)},
        {"metric": "pearson (model)", "mean": per_q["pearson"].mean(),
         "t": _t(per_q["pearson"]), "n_quarters": int(per_q["n"].size)},
    ])


_TIMING = {"contemporaneous": "fwd_1q", "predictive": "fwd_2q", "tradeable": "fwd_3q"}


def alpha_sort(P: pd.DataFrame, cfg: Config, timing=None, on="pred_buy_frac") -> pd.DataFrame:
    """Rank securities on predicted buying pressure -> forward returns by quintile.

    `on="pred_buy_frac"` is the question asked; `on="buy_frac"` sorts on the CURRENT window
    instead, which is the naive comparison -- it needs the same t+1 holdings the target does,
    so it is not tradeable, only a reference.
    """
    timing = timing or cfg.eval_timing
    col = _TIMING[timing]
    d = P.dropna(subset=[on, col]).copy()
    if d.empty:
        return pd.DataFrame([{"quintile": "n/a", "mean_ret_pct_per_quarter": np.nan, "t": np.nan}])
    d["Q"] = d.groupby("qi")[on].transform(lambda s: _qcut(s, cfg.n_quintiles))
    d = d.dropna(subset=["Q"])
    per = d.groupby(["qi", "Q"])[col].mean().unstack()
    rows = [{"quintile": f"Q{int(q)}", "mean_ret_pct_per_quarter": per[q].mean() * 100,
             "t": _t(per[q]), "n_quarters": int(per[q].notna().sum())}
            for q in sorted(per.columns)]
    hi, lo = per.columns.max(), per.columns.min()
    sp = (per[hi] - per[lo]).dropna()
    rows.append({"quintile": f"Q{int(hi)}-Q{int(lo)}",
                 "mean_ret_pct_per_quarter": sp.mean() * 100, "t": _t(sp),
                 "n_quarters": int(sp.size)})
    return pd.DataFrame(rows)


def run_one(panel: pd.DataFrame, cfg: Config, tag: str = "", verbose=True) -> dict:
    """One model on an already-built panel. The panel is passed in so several models can
    share it rather than rebuilding (the Barra join is the slow part)."""
    print(f"{'='*74}\n{tag or cfg.model}  |  model={cfg.model}  "
          f"eval_timing={cfg.eval_timing}\n{'='*74}")
    P = run_model(panel, cfg, verbose=verbose)
    r = {"tag": tag, "cfg": cfg, "preds": P,
         "quality": prediction_quality(P),
         "alpha_pred": {t: alpha_sort(P, cfg, t, on="pred_buy_frac") for t in _TIMING},
         "alpha_actual": {t: alpha_sort(P, cfg, t, on="buy_frac") for t in _TIMING}}
    hz = cfg.eval_timing
    q = r["quality"]
    sp = r["alpha_pred"][hz].iloc[-1]
    print(f"\n  rank IC (model) = {q.iloc[0]['mean']:+.4f} (t={q.iloc[0]['t']:+.2f})"
          f"   vs naive persistence {q.iloc[1]['mean']:+.4f} (t={q.iloc[1]['t']:+.2f})")
    print(f"  alpha on PREDICTED buying, {hz}: {sp.quintile} = "
          f"{sp.mean_ret_pct_per_quarter:+.3f}%/qtr (t={sp.t:+.2f})")
    return r


def compare(results: dict) -> pd.DataFrame:
    """Side-by-side across models."""
    rows = []
    for tag, r in results.items():
        if not isinstance(r, dict) or "quality" not in r:
            continue
        q, cfg = r["quality"], r["cfg"]
        row = {"config": tag, "model": cfg.model,
               "rank_IC": round(q.iloc[0]["mean"], 4), "IC_t": round(q.iloc[0]["t"], 2),
               "naive_IC": round(q.iloc[1]["mean"], 4)}
        for t in _TIMING:
            sp = r["alpha_pred"][t].iloc[-1]
            row[f"alpha_{t[:5]}"] = round(sp.mean_ret_pct_per_quarter, 3)
            row[f"alpha_t_{t[:5]}"] = round(sp.t, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def free(results: dict = None):
    """Drop prediction detail between models so back-to-back runs do not accumulate."""
    import gc
    if results:
        for r in results.values():
            if isinstance(r, dict) and "preds" in r:
                r["preds"] = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("[free] cleared.")


def run_all(cfg: Config = None, models=("gbm", "lstm"), verbose=True) -> dict:
    """Build the panel once, then run each model on it."""
    cfg = cfg or Config()
    panel = build_panel(cfg)
    out = {"_panel": panel}
    for m in models:
        from dataclasses import replace
        out[m] = run_one(panel, replace(cfg, model=m), m, verbose=verbose)
    print("\n" + compare(out).to_string(index=False))
    return out
