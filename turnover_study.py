"""turnover_study.py -- five security characteristics, two questions, six models.

Deliberately small and independent of the rest of the pipeline. No funds, no holdings, no
Barra: one row is one (security, quarter), and every number comes from price/volume/size.

    features (all known at the close of q)
        turnover     volume / shares outstanding, over quarter q
        log_mktcap   log market cap at q
        ret_q        the security's return DURING q
        vol_ret      dispersion of its last `vol_window` quarterly returns, q included
        log_price    log close at q

    targets (both strictly after every feature)
        turnover_next  turnover over q+1
        ret_next       the return over q -> q+1

    models
        five univariate GBMs, one per feature, plus one on all five together

TIMING. Write I_t for what is knowable once quarter t has closed. Every feature is
I_q-measurable: turnover, mktcap, price and ret_q are all realised within q, and vol_ret uses
q and earlier. Both targets land after q closes, so a forecast made at the q close precedes
the whole window it is graded on. That means a quintile sort on a prediction can be traded at
the q close, and the return spread it earns is not contaminated by overlap.

The models are compared against the RAW feature as well. For a single feature a tree is close
to a monotone transform of it, so `model:x` and `raw:x` should nearly agree -- when they do
not, the tree has fitted noise.

Usage
    import turnover_study as T
    panel = T.build_panel(T.Config(holdings_path=...))
    table = T.run_all(panel, T.Config())
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List

__version__ = "2026.08.07.1"

FEATURES = ["turnover", "log_mktcap", "ret_q", "vol_ret", "log_price"]
TARGETS = ["turnover_next", "ret_next"]


@dataclass
class Config:
    holdings_path: str = ("manager_holdings/master_batches_return_filtered/"
                          "master_all_funds_add_filter_ivy_rank_active_rank.parquet")
    # column names in the raw file
    col_map: dict = field(default_factory=lambda: {
        "security": "security", "date": "date", "close": "close", "volume": "volume",
        "market_cap": "market_cap", "quarterly_ret": "quarterly_ret",
        "future_1q_ret": "future_1q_ret", "InvTypeCode": "inv_type", "isUs": "isUs",
    })
    inv_type_codes = (401,)
    us_only: bool = True

    vol_window: int = 4        # quarters used for vol_ret (current + 3 lags)
    winsorize: float = 0.01    # per-quarter clip on every feature and on turnover targets
    min_quarters: int = 8      # expanding history a security needs before it is usable

    # rolling out-of-sample split, same scheme for every model
    window_q: int = 28
    test_q: int = 8
    step: int = 8

    n_quintiles: int = 5
    max_iter: int = 300
    learning_rate: float = 0.06
    max_depth: int = 4         # shallow: five features do not need more
    seed: int = 0


# ------------------------------------------------------------------ helpers
def _t(x) -> float:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return np.nan
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(n)))


def _qcut(s, n=5):
    return (pd.qcut(s.rank(method="first"), n, labels=False, duplicates="drop") + 1
            if s.nunique() >= n else pd.Series(np.nan, index=s.index))


def _winsor_by_q(df, cols, p):
    for c in cols:
        if c in df.columns and df[c].notna().any():
            lo = df.groupby("yq")[c].transform(lambda s: s.quantile(p))
            hi = df.groupby("yq")[c].transform(lambda s: s.quantile(1 - p))
            df[c] = df[c].clip(lo, hi).astype("float32")
    return df


# ------------------------------------------------------------------ panel
def build_panel(cfg: Config = None) -> pd.DataFrame:
    cfg = cfg or Config()
    inv = {v: k for k, v in cfg.col_map.items()}
    want = ["security", "date", "close", "volume", "market_cap", "quarterly_ret",
            "future_1q_ret", "inv_type", "isUs"]
    raw = [inv[c] for c in want if c in inv]
    try:
        import pyarrow.parquet as pq
        avail = set(pq.ParquetFile(cfg.holdings_path).schema.names)
        df = pd.read_parquet(cfg.holdings_path, columns=[c for c in raw if c in avail] or None)
    except Exception:
        df = pd.read_parquet(cfg.holdings_path)
    df = df.rename(columns={inv[c]: c for c in cfg.col_map.values() if inv[c] in df.columns})

    missing = [c for c in ("close", "volume", "market_cap") if c not in df.columns]
    if missing:
        raise KeyError(f"turnover needs {missing} -- rename them in Config.col_map")

    df["date"] = pd.to_datetime(df["date"])
    df["yq"] = df["date"].dt.to_period("Q")
    if cfg.us_only and "isUs" in df.columns:
        df = df[df["isUs"].astype(bool)]
    if cfg.inv_type_codes is not None and "inv_type" in df.columns:
        df = df[df["inv_type"].astype(str).isin({str(c) for c in cfg.inv_type_codes})]

    for c in ("close", "volume", "market_cap", "quarterly_ret", "future_1q_ret"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # the file is a fund-security-quarter panel; these are all security-level, so one row per
    # (security, quarter) is all that is needed
    sq = df.groupby(["security", "yq"], observed=True).agg(
        close=("close", "first"), volume=("volume", "first"),
        mktcap=("market_cap", "first"), ret_q=("quarterly_ret", "first"),
        ret_next=("future_1q_ret", "first"), n_rows=("security", "size"),
    ).reset_index()
    print(f"[panel] {len(sq):,} security-quarters | {sq.security.nunique():,} securities | "
          f"{sq.yq.nunique()} quarters")

    # ---- turnover. shares_out = mktcap / close, so turnover = volume * close / mktcap.
    # If `volume` is already a DOLLAR volume that formula is off by `close`, so compute both
    # and pick the one whose median lands in a plausible range (a quarter's turnover is
    # normally a few tens of percent). Printed either way so it can be checked.
    cap = sq["mktcap"].abs().where(sq["mktcap"].abs() > 0)
    as_shares = sq["volume"] * sq["close"] / cap
    as_dollars = sq["volume"] / cap
    m_sh, m_dl = float(as_shares.median()), float(as_dollars.median())
    if np.isfinite(m_sh) and 0.005 < m_sh < 5:
        sq["turnover"], pick = as_shares, "volume is SHARE volume: volume*close/mktcap"
    elif np.isfinite(m_dl) and 0.005 < m_dl < 5:
        sq["turnover"], pick = as_dollars, "volume is DOLLAR volume: volume/mktcap"
    else:
        sq["turnover"], pick = as_shares, "NEITHER looks plausible -- CHECK UNITS"
    print(f"[turnover] median if share-volume {m_sh:.4f} | if dollar-volume {m_dl:.4f}")
    print(f"[turnover] using: {pick}")

    sq["log_mktcap"] = np.log(sq["mktcap"].abs() + 1.0)
    sq["log_price"] = np.log(sq["close"].abs() + 1e-6)

    # ---- integer quarter index, then everything exact-quarter aligned ----
    qs = pd.PeriodIndex(sorted(sq["yq"].unique()), freq="Q")
    sq["qi"] = sq["yq"].map({q: i for i, q in enumerate(qs)}).astype("int32")
    sq = sq.sort_values(["security", "qi"]).reset_index(drop=True)
    g = sq.groupby("security", observed=True)
    qi = sq["qi"].to_numpy()

    # vol_ret from the current return plus exact lags -- never a stale row across a gap
    lag_cols = []
    for k in range(1, cfg.vol_window):
        v, qq = g["ret_q"].shift(k), g["qi"].shift(k)
        sq[f"_r{k}"] = v.where(qq == qi - k)
        lag_cols.append(f"_r{k}")
    sq["vol_ret"] = sq[["ret_q"] + lag_cols].std(axis=1, ddof=1)
    sq = sq.drop(columns=lag_cols)

    # target 1: next quarter's turnover, exact
    v, qq = g["turnover"].shift(-1), g["qi"].shift(-1)
    sq["turnover_next"] = v.where(qq == qi + 1)
    # target 2: the q -> q+1 return, already on this row

    sq = _winsor_by_q(sq, FEATURES + ["turnover_next"], cfg.winsorize)

    seen = g.cumcount() + 1          # expanding, not full-sample: no survivorship
    sq = sq[seen >= cfg.min_quarters].reset_index(drop=True)

    for c in FEATURES + TARGETS:
        sq[c] = sq[c].astype("float32")
    print(f"[panel] {len(sq):,} rows after history filter | "
          f"turnover_next available {sq.turnover_next.notna().mean():.1%} | "
          f"ret_next available {sq.ret_next.notna().mean():.1%}")
    print("[panel] feature coverage: " + "  ".join(
        f"{c} {sq[c].notna().mean():.0%}" for c in FEATURES))
    return sq


# ------------------------------------------------------------------ model
def _rolling_predict(panel: pd.DataFrame, feats: List[str], target: str,
                     cfg: Config) -> pd.DataFrame:
    """Same rolling out-of-sample split for every model, so the six are comparable."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    d = panel.dropna(subset=[target])
    out = []
    for c in range(cfg.window_q, int(d.qi.max()) + 2, cfg.step):
        tr = d[(d.qi >= c - cfg.window_q) & (d.qi < c - cfg.test_q)]
        te = d[(d.qi >= c - cfg.test_q) & (d.qi < c)]
        if len(tr) < 500 or len(te) == 0:
            continue
        m = HistGradientBoostingRegressor(max_iter=cfg.max_iter,
                                          learning_rate=cfg.learning_rate,
                                          max_depth=cfg.max_depth,
                                          random_state=cfg.seed)
        m.fit(tr[feats].to_numpy("float32"), tr[target].to_numpy("float32"))
        # dict.fromkeys de-duplicates while keeping order: when target IS "ret_next" a plain
        # list would carry it twice and every later `d[target]` would be a 2-column frame
        keep = list(dict.fromkeys(["security", "qi", target, "ret_next"]))
        p = te[keep].copy()
        p["pred"] = m.predict(te[feats].to_numpy("float32"))
        out.append(p)
    if not out:
        raise RuntimeError("no predictions -- window_q/test_q too large for this panel")
    return pd.concat(out, ignore_index=True)


def _score(d: pd.DataFrame, pred_col: str, target: str, cfg: Config) -> dict:
    """rank-IC against the target, and the quintile spread measured in NEXT-QUARTER RETURN.

    The spread is always in return units, whatever the target is -- that is the only way a
    turnover model and a return model can be put in the same table.
    """
    x = d.dropna(subset=[pred_col, target])
    ic = x.groupby("qi").apply(lambda t: t[pred_col].corr(t[target], method="spearman"))
    r = d.dropna(subset=[pred_col, "ret_next"]).copy()
    r["Q"] = r.groupby("qi")[pred_col].transform(lambda s: _qcut(s, cfg.n_quintiles))
    r = r.dropna(subset=["Q"])
    per = r.groupby(["qi", "Q"])["ret_next"].mean().unstack()
    hi, lo = per.columns.max(), per.columns.min()
    sp = (per[hi] - per[lo]).dropna()
    return {"rank_IC": ic.mean(), "IC_t": _t(ic),
            "Q1_ret_pct": per[lo].mean() * 100, "Q5_ret_pct": per[hi].mean() * 100,
            "Q5_Q1_pct": sp.mean() * 100, "spread_t": _t(sp), "n_quarters": int(sp.size)}


def run_all(panel: pd.DataFrame, cfg: Config = None, verbose: bool = True) -> pd.DataFrame:
    """Six models per target: one per feature, plus one on all five."""
    cfg = cfg or Config()
    specs = [(f"model:{f}", [f]) for f in FEATURES] + [("model:ALL", FEATURES)]
    rows = []
    for target in TARGETS:
        for name, feats in specs:
            P = _rolling_predict(panel, feats, target, cfg)
            rows.append({"target": target, "model": name, **_score(P, "pred", target, cfg)})
        # raw features as references: no model, just sort on the characteristic itself
        base = panel.dropna(subset=[target])
        for f in FEATURES:
            rows.append({"target": target, "model": f"raw:{f}",
                         **_score(base, f, target, cfg)})
    out = pd.DataFrame(rows)
    if verbose:
        for target in TARGETS:
            sub = out[out.target == target].drop(columns="target")
            print(f"\n{'='*92}\nTARGET = {target}\n{'='*92}")
            print(sub.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        print("\n  rank_IC     against that row's target. For ret_next, >0.03 is respectable")
        print("              and >0.05 strong -- returns are near-unforecastable, so a large")
        print("              IC there is a reason to look for leakage.")
        print("  Q5_Q1_pct   quintile spread in NEXT-QUARTER RETURN, %/quarter, for every row")
        print("              -- so a turnover model and a return model sit on one scale.")
        print("  raw:x       sorting on the characteristic itself. model:x should nearly match")
        print("              it; a big gap means the tree fitted noise.")
    return out


def check_version(verbose=True):
    import os
    if verbose:
        print(f"turnover_study {__version__}  |  {os.path.abspath(__file__)}")
    return __version__
