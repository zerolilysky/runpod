"""Self-contained n_buy / n_funds signal builder + performance evaluation.

Reads ONLY the raw deliverables:
  <root>/manager_holdings/panel_holdings_All_Funds_{year}.parquet   (2002..2024)
      cols: IgcyInvestorId, day, security, InvTypeCode, isUs,
            SharesHld, SHsHldChg, ShsHldVal, ShsHldValChg, MARKET_CAP, VOLUME
  <root>/return_data_v2.csv
      cols: date, perm_id, trade_id, signal_name, signal_val
      (trade_id == security; signal_val == the security's quarterly total return)

Everything downstream -- active weights, the change in active weight, forward
returns, the security-quarter counts, the portfolios, and the statistics -- is
built here. No other project file is required; depends on pandas + numpy only.

THE TWO SIGNALS (security x quarter):
  n_funds   how many funds hold the security that quarter          (coverage)
  n_buy     how many of them RAISED their active weight that qtr   (buying)

Also computed for context, because corr(n_buy, n_funds) ~ 0.96 in this data:
  frac_buy      n_buy / n_funds                    (the buying RATE)
  n_buy_resid   n_buy with coverage projected out  (buying, coverage removed)

Usage
-----
    import signals_perf as S
    res = S.run(S.Config(root="/path/to/root"))
    print(res.perf.to_string())         # t, return, Sharpe, hit, ... per signal
    res.spreads["n_buy|h1"]             # per-quarter long-short series for plotting
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

RET_COL = {1: "future_1q_ret", 2: "future_2q_ret", 3: "future_3q_ret"}


# ================================================================= CONFIG
@dataclass
class Config:
    root: str = "."                       # folder holding manager_holdings/ + csv
    panels_subdir: str = "manager_holdings"
    returns_csv: str = "return_data_v2.csv"
    start_year: int = 2002
    end_year: int = 2024

    us_only: bool = True                  # keep isUs == True
    inv_type: int | None = 401            # keep this InvTypeCode; None = all
    min_shares_val: float = 0.0           # drop non-positive position values

    horizons: tuple = (1, 2, 3)           # t->t+1, t+1->t+2, t+2->t+3
    n_bins: int = 10                      # decile sort
    size_neutral: bool = True             # sort within market-cap terciles
    min_names: int = 20                   # skip a quarter with too few names
    split: str = "2014Q1"                 # discovery < split <= validation
    ann: int = 4                          # quarters per year (annualisation)

    signals: tuple = ("n_buy", "n_funds", "frac_buy", "n_buy_resid")

    @property
    def panels_dir(self) -> str:
        return os.path.join(self.root, self.panels_subdir)

    @property
    def returns_path(self) -> str:
        return os.path.join(self.root, self.returns_csv)


@dataclass
class Result:
    stock_q: pd.DataFrame
    perf: pd.DataFrame
    spreads: Dict[str, pd.Series] = field(default_factory=dict)
    corr: pd.DataFrame = None


# ================================================================= LOADING
def load_panels(cfg: Config) -> pd.DataFrame:
    """Read every yearly holdings panel and stack them.

    Only the columns needed for these signals are read: the fund, quarter, security,
    the filters, and the two fields the active weight is built from
    (ShsHldVal = position value, MARKET_CAP).
    """
    want = ["IgcyInvestorId", "day", "security", "InvTypeCode", "isUs",
            "ShsHldVal", "MARKET_CAP"]
    frames = []
    for year in range(cfg.start_year, cfg.end_year + 1):
        p = os.path.join(cfg.panels_dir, f"panel_holdings_All_Funds_{year}.parquet")
        if not os.path.exists(p):
            print(f"[load] missing {p}, skipped")
            continue
        try:
            import pyarrow.parquet as pq
            avail = set(pq.ParquetFile(p).schema.names)
            cols = [c for c in want if c in avail]
            df = pd.read_parquet(p, columns=cols or None)
        except Exception:
            df = pd.read_parquet(p)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no panel files under {cfg.panels_dir}")
    df = pd.concat(frames, ignore_index=True)

    df = df.rename(columns={
        "IgcyInvestorId": "fund", "day": "date", "ShsHldVal": "position_value",
        "MARKET_CAP": "market_cap", "InvTypeCode": "inv_type"})
    df["date"] = pd.to_datetime(df["date"])
    df["yq"] = df["date"].dt.to_period("Q")

    if cfg.us_only and "isUs" in df.columns:
        df = df[df["isUs"].fillna(False).astype(bool)]
    if cfg.inv_type is not None and "inv_type" in df.columns:
        df = df[df["inv_type"] == cfg.inv_type]
    for c in ("position_value", "market_cap"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df = df[df["position_value"] > cfg.min_shares_val]

    # one row per (fund, quarter, security)
    df = df.sort_values("date").drop_duplicates(["fund", "yq", "security"],
                                                keep="last")
    print(f"[load] {len(df):,} fund-security-quarters | "
          f"{df.fund.nunique():,} funds | {df.security.nunique():,} securities | "
          f"{df.yq.min()}..{df.yq.max()}")
    return df


def load_returns(cfg: Config) -> pd.DataFrame:
    """Quarterly returns + forward returns per (security, quarter).

    return_data_v2's trade_id is the security key; signal_val is the quarterly total
    return. future_h(t) is the return realised over quarter t+h, built by shifting the
    security's own return series -- so it is an OUTCOME, never an input to the signal.
    """
    r = pd.read_csv(cfg.returns_path)
    r["date"] = pd.to_datetime(r["date"])
    r["yq"] = r["date"].dt.to_period("Q")
    r = r.rename(columns={"trade_id": "security", "signal_val": "quarterly_ret"})
    r["security"] = pd.to_numeric(r["security"], errors="coerce").astype("Int64")
    r = r.dropna(subset=["security"])
    r["security"] = r["security"].astype("int64")
    r = (r.groupby(["security", "yq"], as_index=False)["quarterly_ret"].mean()
           .sort_values(["security", "yq"]))
    g = r.groupby("security", sort=False)["quarterly_ret"]
    r["future_1q_ret"] = g.shift(-1)
    r["future_2q_ret"] = g.shift(-2)
    r["future_3q_ret"] = g.shift(-3)
    print(f"[load] returns: {len(r):,} security-quarters")
    return r


# ================================================= SIGNAL CONSTRUCTION
def build_stock_quarter(panel: pd.DataFrame, returns: pd.DataFrame,
                        cfg: Config) -> pd.DataFrame:
    """Fund-level active weights -> security-quarter counts, joined to returns.

    active weight  a_i = w_real_i - w_ref_i, both normalised over the SAME set of a
    fund's holdings (those with a positive value and a market cap), so a sums to zero
    within each fund-quarter. buy = the fund raised a in that name vs last quarter.
    """
    df = panel.copy()
    ok = (df["position_value"] > 0) & (df["market_cap"] > 0)
    df["_pv"] = df["position_value"].where(ok)
    df["_mc"] = df["market_cap"].where(ok)
    g = df.groupby(["fund", "yq"], sort=False)
    pv_tot = g["_pv"].transform("sum")
    mc_tot = g["_mc"].transform("sum")
    df["w_real"] = np.where(pv_tot > 0, df["_pv"] / pv_tot, np.nan)
    df["w_ref"] = np.where(mc_tot > 0, df["_mc"] / mc_tot, np.nan)
    df["active_weight"] = df["w_real"] - df["w_ref"]

    # change in the fund's active weight in that name (the trade / tilt)
    df = df.sort_values(["fund", "security", "yq"])
    df["aw_lag1"] = df.groupby(["fund", "security"], sort=False)[
        "active_weight"].shift(1)
    df["d_aw"] = df["active_weight"] - df["aw_lag1"]
    df["buy"] = df["d_aw"] > 0

    sq = df.groupby(["security", "yq"], observed=True).agg(
        n_funds=("fund", "size"),          # coverage
        n_buy=("buy", "sum"),              # buying
        frac_buy=("buy", "mean"),          # rate
        market_cap=("market_cap", "first"),
    ).reset_index()

    sq = sq.merge(
        returns[["security", "yq", "quarterly_ret", "future_1q_ret",
                 "future_2q_ret", "future_3q_ret"]],
        on=["security", "yq"], how="left")

    # n_buy with coverage regressed out, cross-sectionally, within each quarter
    sq["log_nf"] = np.log(sq["n_funds"].clip(lower=1))

    def _resid(gg):
        x, y = gg["log_nf"], gg["n_buy"]
        m = x.notna() & y.notna()
        if m.sum() < 30:
            return pd.Series(np.nan, index=gg.index)
        b0, b1 = np.polyfit(x[m], y[m], 1)
        return y - (b0 * x + b1)

    sq["n_buy_resid"] = sq.groupby("yq", group_keys=False).apply(_resid)
    print(f"[signal] {len(sq):,} security-quarters built")
    return sq


# ==================================================== PORTFOLIO FORMATION
def decile_spread(sq: pd.DataFrame, signal: str, horizon: int,
                  cfg: Config) -> pd.Series:
    """Per-quarter return of (top decile - bottom decile) sorted on `signal`.

    size_neutral: sort inside market-cap terciles and average, so the spread is a bet
    on the signal, not on firm size.
    """
    ret = RET_COL[horizon]
    d = sq[["yq", "security", signal, ret, "market_cap"]].dropna(subset=[signal, ret])

    def _bin(x: pd.DataFrame) -> float:
        if len(x) < cfg.min_names:
            return np.nan
        b = pd.qcut(x[signal].rank(method="first"), cfg.n_bins,
                    labels=False, duplicates="drop")
        if b.nunique() < 2:
            return np.nan
        return x.loc[b == b.max(), ret].mean() - x.loc[b == b.min(), ret].mean()

    def _quarter(q: pd.DataFrame) -> float:
        if not cfg.size_neutral:
            return _bin(q)
        q = q.dropna(subset=["market_cap"])
        if len(q) < cfg.min_names:
            return np.nan
        terc = pd.qcut(q["market_cap"].rank(method="first"), 3,
                       labels=False, duplicates="drop")
        vals = [_bin(sub) for _, sub in q.groupby(terc)]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan

    return d.groupby("yq").apply(_quarter, include_groups=False).dropna()


# ================================================================= INFERENCE
def newey_west_t(x: np.ndarray, lags: int = 0) -> float:
    """t-stat of the mean with a Newey-West HAC variance (lags = horizon-1)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return np.nan
    e = x - x.mean()
    var = (e @ e) / n
    for L in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - L / (lags + 1.0)) * ((e[L:] @ e[:-L]) / n)
    return x.mean() / np.sqrt(var / n) if var > 0 else np.nan


def performance(spread: pd.Series, horizon: int, cfg: Config) -> dict:
    """Full performance record for one per-quarter long-short series."""
    s = spread.dropna()
    n = len(s)
    if n < 8:
        return dict(n_quarters=n)
    mu = float(s.mean())
    vol = float(s.std(ddof=1))
    t_nw = newey_west_t(s.to_numpy(), lags=horizon - 1)
    t_naive = mu / (vol / np.sqrt(n)) if vol > 0 else np.nan
    ann_ret = mu * cfg.ann                                   # arithmetic annualised
    geo_ann = (np.prod(1.0 + s.to_numpy()) ** (cfg.ann / n) - 1.0)  # compounded
    ann_vol = vol * np.sqrt(cfg.ann)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = float(np.prod(1.0 + s.to_numpy()) - 1.0)
    wealth = np.cumprod(1.0 + s.to_numpy())
    dd = float((wealth / np.maximum.accumulate(wealth) - 1.0).min())
    return dict(
        n_quarters=n,
        mean_q=mu,
        t_nw=float(t_nw),
        t_naive=float(t_naive),
        hit=float((s > 0).mean()),
        ann_return=float(ann_ret),
        geo_ann_return=float(geo_ann),
        ann_vol=float(ann_vol),
        sharpe_ann=float(sharpe),
        cum_return=cum,
        max_drawdown=dd,
    )


def _sample_mask(idx: pd.PeriodIndex, sample: str, split: pd.Period):
    if sample == "discovery":
        return idx < split
    if sample == "validation":
        return idx >= split
    return np.ones(len(idx), bool)


# ================================================================= DRIVER
def run(cfg: Config = Config(), verbose: bool = True) -> Result:
    """End to end: load -> build -> evaluate. Returns a Result."""
    panel = load_panels(cfg)
    returns = load_returns(cfg)
    sq = build_stock_quarter(panel, returns, cfg)

    split = pd.Period(cfg.split, freq="Q")
    rows: List[dict] = []
    spreads: Dict[str, pd.Series] = {}
    for sig in cfg.signals:
        for h in cfg.horizons:
            sp = decile_spread(sq, sig, h, cfg)
            spreads[f"{sig}|h{h}"] = sp
            for sample in ("discovery", "validation", "all"):
                mask = _sample_mask(sp.index, sample, split)
                perf = performance(sp[mask], h, cfg)
                perf.update(signal=sig, horizon=h, sample=sample)
                rows.append(perf)

    perf = pd.DataFrame(rows)
    front = ["signal", "horizon", "sample", "n_quarters", "mean_q", "t_nw",
             "t_naive", "hit", "ann_return", "sharpe_ann", "cum_return",
             "max_drawdown"]
    perf = perf[[c for c in front if c in perf.columns] +
                [c for c in perf.columns if c not in front]]

    c = sq[["n_buy", "n_funds", "frac_buy", "market_cap"]].dropna()
    corr = c.assign(log_mktcap=np.log(c.market_cap.where(c.market_cap > 0)))[
        ["n_buy", "n_funds", "frac_buy", "log_mktcap"]].corr()

    if verbose:
        print("\n=== signal correlations (why n_buy ~ coverage) ===")
        print(corr.round(3).to_string())
        print("\n=== performance (full sample) ===")
        show = perf[perf["sample"] == "all"].set_index(["signal", "horizon"])
        print(show[["n_quarters", "mean_q", "t_nw", "sharpe_ann", "hit",
                    "ann_return"]].round(4).to_string())
    return Result(stock_q=sq, perf=perf, spreads=spreads, corr=corr)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="n_buy / n_funds signals + performance")
    ap.add_argument("--root", default=".", help="folder with manager_holdings/ + csv")
    ap.add_argument("--out", default="signals_performance.csv")
    args = ap.parse_args()
    res = run(Config(root=args.root))
    res.perf.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
