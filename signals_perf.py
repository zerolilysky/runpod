"""Self-contained n_buy / n_funds signal builder + performance evaluation.

Reads ONLY the raw deliverables:
  <root>/manager_holdings/panel_holdings_All_Funds_{year}.parquet   (2002..2024)
      cols: LgcyInvestorId, day, security, InvTypeCode, isUs,
            SharesHld, SHsHldChg, ShsHldVal, ShsHldValChg, MARKET_CAP, VOLUME
  <root>/return_data_v2.csv
      cols: date, perm_id, trade_id, signal_name, signal_val
      perm_id  == the SECURITY id (joins to the panel's `security`), a STRING
      signal_val == the security's DAILY total return, rows where
                    signal_name == 'TRET_T1D'

Everything downstream -- active weights, the change in active weight, quarterly and
forward returns, the security-quarter counts, the portfolios, and the statistics --
is built here. Depends on pandas + numpy only.

THE TWO SIGNALS (security x quarter):
  n_funds   how many funds hold the security that quarter          (coverage)
  n_buy     how many of them RAISED their active weight that qtr   (buying)
  n_sell    how many CUT it, including funds that exited entirely  (selling)
  n_active  held at t OR t-1 -- the correct denominator for rates

Also computed, because corr(n_buy, n_funds) is ~0.96 on real data:
  frac_buy      n_buy / n_funds                    (the buying RATE)
  n_buy_resid   n_buy with coverage projected out  (buying, coverage removed)

CORRECTNESS NOTES (these bit earlier versions -- do not "simplify" them away)
  * Daily -> quarterly. signal_val is a DAILY return, so a quarter's return is
    prod(1+r)-1 over that quarter's days, accumulated chunk-wise so a 100M-row file
    does not have to fit in memory.
  * Strict one-quarter lags. Securities jump in and out of a fund's book, so
    groupby(...).shift(1) returns the previous OBSERVATION, which can be years
    earlier. Every lag/lead here is an explicit join on an integer quarter index.
  * ID types. perm_id is a string; panel `security` may be int. Both sides go
    through the same normaliser before joining.
  * InvTypeCode may arrive as a string -> pd.to_numeric before comparing.

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

# the panel's fund id has been seen under both spellings
FUND_ID_CANDIDATES = ("LgcyInvestorId", "IgcyInvestorId", "lgcyinvestorid",
                      "igcyinvestorid")
RETURNS_NAME_CANDIDATES = ("return_data_v2.csv", "returns_data_v2.csv",
                           "return_data.csv")


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
    signal_name: str = "TRET_T1D"         # which rows of the returns file to use
    min_days_per_quarter: int = 0         # drop thin quarters (0 = keep all)

    # A security absent from a fund's book has active weight 0 -- in BOTH
    # directions. Absent last quarter and held now = an initiation (lag 0, so the
    # tilt rose). Held last quarter and absent now = an EXIT, which the raw panel
    # does not record at all, so those rows are reconstructed with weight 0.
    # Without this, sells are invisible and every rate is computed on a denominator
    # that silently drops the funds that sold out.
    #   True  -> union grid {held at t} u {held at t-1}, absent = 0   (default)
    #   False -> only rows the panel actually contains; lag NaN if absent
    absent_is_zero: bool = True

    horizons: tuple = (1, 2, 3)           # t->t+1, t+1->t+2, t+2->t+3
    n_bins: int = 10                      # decile sort
    size_neutral: bool = True             # sort within market-cap terciles
    min_names: int = 20                   # skip a quarter with too few names
    split: str = "2014Q1"                 # discovery < split <= validation
    ann: int = 4                          # quarters per year (annualisation)
    chunksize: int = 5_000_000            # rows per read_csv chunk

    signals: tuple = ("n_buy", "n_funds", "frac_buy", "n_buy_resid",
                      "net_buy", "n_sell")

    @property
    def panels_dir(self) -> str:
        return os.path.join(self.root, self.panels_subdir)


@dataclass
class Result:
    stock_q: pd.DataFrame
    perf: pd.DataFrame
    spreads: Dict[str, pd.Series] = field(default_factory=dict)
    corr: pd.DataFrame = None


# ================================================================= HELPERS
def _norm_id(s: pd.Series) -> pd.Series:
    """Canonical string id, so an int `security` joins a string `perm_id`.

    Numeric-looking values become their integer string ('10001.0' and 10001 and
    '0010001' all -> '10001'); anything else is stripped text.
    """
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().all():
        return v.astype("int64").astype(str)
    out = s.astype(str).str.strip()
    ok = v.notna()
    if ok.any():
        out.loc[ok] = v[ok].astype("int64").astype(str)
    return out


def _to_bool(s: pd.Series) -> pd.Series:
    """isUs may arrive as bool, 0/1, or 'True'/'Y'/'t'."""
    if s.dtype == bool:
        return s
    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0) != 0
    return (s.astype(str).str.strip().str.lower()
             .isin(["true", "t", "1", "y", "yes"]))


def _qi(yq: pd.Series) -> pd.Series:
    """Global integer quarter index; consecutive quarters differ by exactly 1."""
    return (yq.dt.year * 4 + yq.dt.quarter).astype("int64")


def _find_returns(cfg: Config) -> str:
    """Locate the returns csv: as configured, then a few obvious fallbacks."""
    cands = [os.path.join(cfg.root, cfg.returns_csv),
             os.path.join(cfg.panels_dir, cfg.returns_csv)]
    for name in RETURNS_NAME_CANDIDATES:
        cands += [os.path.join(cfg.root, name),
                  os.path.join(cfg.panels_dir, name)]
    for p in cands:
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(cfg.root, "**", "return*data*.csv"),
                     recursive=True)
    if hits:
        return sorted(hits)[0]
    raise FileNotFoundError(
        f"returns csv not found. looked for {cfg.returns_csv} under {cfg.root} "
        f"and {cfg.panels_dir}")


def _strict_join(df: pd.DataFrame, keys: List[str], value_cols: List[str],
                 offset: int, suffix_map: Dict[str, str]) -> pd.DataFrame:
    """Attach values from exactly `offset` quarters away via an explicit join.

    offset=+1 -> the value one quarter EARLIER (a lag); offset=-1 -> one quarter
    LATER (a lead). Never uses shift(), so gaps cannot silently pull a distant
    observation into an adjacent slot.
    """
    src = df[keys + ["qi"] + value_cols].copy()
    src["qi"] = src["qi"] + offset
    src = src.rename(columns=suffix_map)
    return df.merge(src, on=keys + ["qi"], how="left")


# ================================================================= LOADING
def load_panels(cfg: Config) -> pd.DataFrame:
    """Read every yearly holdings panel and stack them."""
    frames = []
    for year in range(cfg.start_year, cfg.end_year + 1):
        p = os.path.join(cfg.panels_dir, f"panel_holdings_All_Funds_{year}.parquet")
        if not os.path.exists(p):
            print(f"[load] missing {p}, skipped")
            continue
        try:
            import pyarrow.parquet as pq
            avail = list(pq.ParquetFile(p).schema.names)
            fund_col = next((c for c in FUND_ID_CANDIDATES if c in avail), None)
            want = [c for c in [fund_col, "day", "security", "InvTypeCode", "isUs",
                                "ShsHldVal", "MARKET_CAP"] if c and c in avail]
            df = pd.read_parquet(p, columns=want or None)
        except Exception:
            df = pd.read_parquet(p)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no panel files under {cfg.panels_dir}")
    df = pd.concat(frames, ignore_index=True)

    fund_col = next((c for c in FUND_ID_CANDIDATES if c in df.columns), None)
    if fund_col is None:
        raise KeyError(f"no fund id column; looked for {FUND_ID_CANDIDATES}, "
                       f"got {list(df.columns)}")
    df = df.rename(columns={fund_col: "fund", "day": "date",
                            "ShsHldVal": "position_value",
                            "MARKET_CAP": "market_cap", "InvTypeCode": "inv_type"})
    df["date"] = pd.to_datetime(df["date"])
    df["yq"] = df["date"].dt.to_period("Q")

    if cfg.us_only and "isUs" in df.columns:
        df = df[_to_bool(df["isUs"])]
    if cfg.inv_type is not None and "inv_type" in df.columns:
        # InvTypeCode can arrive as a string -> coerce before comparing
        df = df[pd.to_numeric(df["inv_type"], errors="coerce") == cfg.inv_type]

    df["security"] = _norm_id(df["security"])
    df["fund"] = _norm_id(df["fund"])
    for c in ("position_value", "market_cap"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df = df[df["position_value"] > 0]

    df = df.sort_values("date").drop_duplicates(["fund", "yq", "security"],
                                                keep="last")
    df["qi"] = _qi(df["yq"])
    print(f"[load] {len(df):,} fund-security-quarters | {df.fund.nunique():,} funds "
          f"| {df.security.nunique():,} securities | {df.yq.min()}..{df.yq.max()}")
    return df


def load_returns(cfg: Config) -> pd.DataFrame:
    """DAILY returns -> quarterly, plus strictly-aligned forward returns.

    signal_val is a daily total return, so the quarter's return is prod(1+r)-1.
    Products are accumulated chunk-by-chunk, so a 100M-row file never has to sit in
    memory at once. (A file that is already quarterly passes through unchanged: the
    product of a single observation is that observation.)
    """
    path = _find_returns(cfg)
    print(f"[load] returns <- {path}")

    prod = None      # (security, yq) -> running product of (1+r)
    ndays = None
    n_raw = 0
    reader = pd.read_csv(path, chunksize=cfg.chunksize,
                         usecols=lambda c: c.strip().lower() in
                         ("date", "perm_id", "signal_name", "signal_val"),
                         dtype={"perm_id": str, "signal_name": str})
    for chunk in reader:
        chunk.columns = [c.strip() for c in chunk.columns]
        n_raw += len(chunk)
        if "signal_name" in chunk.columns:
            chunk = chunk.loc[chunk["signal_name"].astype(str).str.strip()
                              .eq(cfg.signal_name)]
        if chunk.empty:
            continue
        # perm_id IS the security id in this file
        chunk["security"] = _norm_id(chunk["perm_id"])
        chunk["yq"] = pd.to_datetime(chunk["date"]).dt.to_period("Q")
        r = pd.to_numeric(chunk["signal_val"], errors="coerce")
        chunk = chunk.assign(gross=1.0 + r).dropna(subset=["gross"])
        g = chunk.groupby(["security", "yq"], observed=True)["gross"]
        p, n = g.prod(), g.size()
        prod = p if prod is None else prod.mul(p, fill_value=1.0)
        ndays = n if ndays is None else ndays.add(n, fill_value=0)

    if prod is None:
        raise ValueError(f"no rows with signal_name == {cfg.signal_name!r} in {path}")

    r = prod.rename("gross").reset_index()
    r["n_days"] = ndays.reindex(prod.index).to_numpy()
    r["quarterly_ret"] = r["gross"] - 1.0
    if cfg.min_days_per_quarter > 0:
        before = len(r)
        r = r[r["n_days"] >= cfg.min_days_per_quarter]
        print(f"[load] min_days_per_quarter={cfg.min_days_per_quarter}: "
              f"{before:,} -> {len(r):,}")
    r["qi"] = _qi(r["yq"])

    # forward returns by STRICT quarter join (securities have gaps)
    for h in (1, 2, 3):
        r = _strict_join(r, ["security"], ["quarterly_ret"], offset=-h,
                         suffix_map={"quarterly_ret": f"future_{h}q_ret"})

    print(f"[load] returns: {n_raw:,} raw rows -> {len(r):,} security-quarters "
          f"({r.security.nunique():,} securities, median {r.n_days.median():.0f} "
          f"days/quarter)")
    return r[["security", "yq", "qi", "quarterly_ret", "future_1q_ret",
              "future_2q_ret", "future_3q_ret"]]


# ================================================= SIGNAL CONSTRUCTION
def build_stock_quarter(panel: pd.DataFrame, returns: pd.DataFrame,
                        cfg: Config) -> pd.DataFrame:
    """Fund-level active weights -> security-quarter counts, joined to returns.

    active weight  a_i = w_real_i - w_ref_i, both normalised over the SAME set of a
    fund's holdings (positive value AND a market cap), so a sums to zero within each
    fund-quarter. buy = the fund raised a in that name vs the STRICT previous quarter.
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

    # ---- the (fund, security, quarter) grid to measure changes on ----------
    # The panel only contains HELD positions, so an exit just vanishes. Build the
    # union {held at t} u {held at t-1} per fund and fill absent sides with 0.
    cur = df[["fund", "security", "qi", "active_weight"]].copy()
    cur["held_now"] = True
    prev = df[["fund", "security", "qi", "active_weight"]].copy()
    prev["qi"] += 1                                  # STRICT one-quarter lag
    prev = prev.rename(columns={"active_weight": "aw_lag1"})
    prev["held_prev"] = True

    if cfg.absent_is_zero:
        u = cur.merge(prev, on=["fund", "security", "qi"], how="outer")
        # keep only fund-quarters the fund actually reported, so a fund that stops
        # filing does not generate phantom exits forever
        fq = df[["fund", "qi", "yq"]].drop_duplicates()
        u = u.merge(fq, on=["fund", "qi"], how="inner")
        u["held_now"] = u["held_now"].fillna(False)
        u["held_prev"] = u["held_prev"].fillna(False)
        u["active_weight"] = u["active_weight"].fillna(0.0)   # exited  -> 0
        u["aw_lag1"] = u["aw_lag1"].fillna(0.0)               # initiated -> was 0
    else:
        u = cur.merge(prev, on=["fund", "security", "qi"], how="left")
        u = u.merge(df[["fund", "qi", "yq"]].drop_duplicates(),
                    on=["fund", "qi"], how="left")
        u["held_prev"] = u["held_prev"].fillna(False)

    u["d_aw"] = u["active_weight"] - u["aw_lag1"]
    # A fund must still HOLD the name to be buying it. Without this guard, exiting
    # an UNDERweight position (a < 0 -> 0) shows up as d_aw > 0 and would be
    # miscounted as a purchase.
    u["buy"] = (u["d_aw"] > 0) & u["held_now"]
    u["sell"] = (u["d_aw"] < 0) & u["held_prev"]

    n_exit = int((~u["held_now"] & u["held_prev"]).sum())
    n_init = int((u["held_now"] & ~u["held_prev"]).sum())
    print(f"[signal] grid {len(u):,} rows | initiations {n_init:,} "
          f"({n_init / max(len(u), 1):.1%}) | exits {n_exit:,} "
          f"({n_exit / max(len(u), 1):.1%}) | absent_is_zero={cfg.absent_is_zero}")

    sq = u.groupby(["security", "yq"], observed=True).agg(
        n_funds=("held_now", "sum"),       # coverage: funds actually holding at t
        n_active=("fund", "size"),         # held at t OR t-1: the rate denominator
        n_buy=("buy", "sum"),              # buying
        n_sell=("sell", "sum"),            # selling, now including full exits
    ).reset_index()
    sq["n_funds"] = sq["n_funds"].astype("int64")
    sq["frac_buy"] = sq["n_buy"] / sq["n_active"].where(sq["n_active"] > 0)
    sq["net_buy"] = sq["n_buy"] - sq["n_sell"]

    # market cap comes from the HELD rows (an all-exited name has none that quarter)
    mcap = (df.groupby(["security", "yq"], observed=True)["market_cap"]
              .first().rename("market_cap").reset_index())
    sq = sq.merge(mcap, on=["security", "yq"], how="left")

    sq = sq.merge(returns.drop(columns=["qi"]), on=["security", "yq"], how="left")

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
    matched = sq["quarterly_ret"].notna().mean()
    print(f"[signal] {len(sq):,} security-quarters | return match {matched:.1%}")
    if matched < 0.5:
        print("[warn] fewer than half the security-quarters matched a return -- "
              "check that panel `security` and returns `perm_id` are the same id")
    return sq


# ==================================================== PORTFOLIO FORMATION
def decile_spread(sq: pd.DataFrame, signal: str, horizon: int,
                  cfg: Config) -> pd.Series:
    """Per-quarter return of (top decile - bottom decile) sorted on `signal`."""
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
    ann_ret = mu * cfg.ann
    geo_ann = float(np.prod(1.0 + s.to_numpy()) ** (cfg.ann / n) - 1.0)
    ann_vol = vol * np.sqrt(cfg.ann)
    wealth = np.cumprod(1.0 + s.to_numpy())
    dd = float((wealth / np.maximum.accumulate(wealth) - 1.0).min())
    return dict(
        n_quarters=n, mean_q=mu, t_nw=float(t_nw), t_naive=float(t_naive),
        hit=float((s > 0).mean()), ann_return=float(ann_ret),
        geo_ann_return=geo_ann, ann_vol=float(ann_vol),
        sharpe_ann=float(ann_ret / ann_vol) if ann_vol > 0 else np.nan,
        cum_return=float(wealth[-1] - 1.0), max_drawdown=dd,
    )


def _sample_mask(idx, sample: str, split: pd.Period):
    if sample == "discovery":
        return idx < split
    if sample == "validation":
        return idx >= split
    return np.ones(len(idx), bool)


# ================================================================= DRIVER
def run(cfg: Config = Config(), verbose: bool = True) -> Result:
    """End to end: load -> build -> evaluate."""
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
                perf = performance(sp[_sample_mask(sp.index, sample, split)], h, cfg)
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
    ap.add_argument("--no-absent-zero", action="store_true",
                    help="do NOT reconstruct exits as weight 0")
    args = ap.parse_args()
    res = run(Config(root=args.root, absent_is_zero=not args.no_absent_zero))
    res.perf.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
