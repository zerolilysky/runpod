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


# Every signal build_stock_quarter() emits. Kept as an explicit tuple so that
# `for sig in cfg.signals` always works; evaluate() unions it with whatever it
# actually finds in the frame, so this list going stale is not fatal.
ALL_SIGNALS = (
    # counts and rates of trading / holding
    "n_funds", "n_active", "n_holders", "n_buy", "n_sell", "frac_buy", "net_buy",
    "n_buy_resid",
    # what counts as a "buy": active weight / weight / market weight / shares
    "n_buy_aw", "n_buy_w", "n_buy_mw", "n_buy_sh",
    # what counts as "holding": weight > 0 vs active_weight > 0
    "n_funds_w", "n_funds_aw",
    # intensity, z > k : fraction / count / coverage-stripped count
    "frac_zaw_hi", "frac_zw_hi", "frac_ts_aw_hi", "frac_ts_w_hi",
    "n_zaw_hi", "n_zw_hi", "n_ts_aw_hi", "n_ts_w_hi",
    "n_zaw_hi_resid", "n_ts_aw_hi_resid",
    # intensity, |z| > k : conspicuous either way (attention, not direction)
    "frac_zaw_abs_hi", "frac_ts_aw_abs_hi", "n_zaw_abs_hi", "n_ts_aw_abs_hi",
    "n_zaw_abs_hi_resid", "n_ts_aw_abs_hi_resid",
    # |residual| : abnormally MANY OR FEW relative to coverage
    "n_zaw_hi_resid_abs", "n_ts_aw_hi_resid_abs",
    "n_zaw_abs_hi_resid_abs", "n_ts_aw_abs_hi_resid_abs",
    # continuous: no threshold, no ties
    "mean_z_aw", "mean_z_w", "mean_z_ts_aw", "mean_z_ts_w",
    "mean_abs_z_aw", "mean_abs_z_ts_aw", "sum_z_aw", "sum_z_ts_aw",
)


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
    # how to bucket TIED signal values (n_buy/n_funds are integer counts, so ties
    # are pervasive). "average" never splits a tied group; "first" sorts ties by
    # row order == security id == listing age, which is a real bias. See _bucket.
    tie_break: str = "random"
    # Market-cap neutralisation: sort inside `size_groups` cap groups and average.
    # 1 = none. n_buy correlates ~0.58 with log cap, and a tercile still spans a
    # ~60x cap range, so 3 removes only about two thirds of the size leakage.
    size_groups: int = 3
    size_neutral: bool = True             # deprecated alias; False -> size_groups=1
    # Restrict the cross-section to rows where the signal is non-zero. The counts
    # have a large mass at 0, which otherwise dominates the bottom bucket.
    nonzero_only: bool = False
    # What to do with a security-quarter whose FORWARD return is missing (the
    # security vanished from the returns file -- typically a delisting).
    #   "drop" (default) listwise deletion, before the buckets are formed
    #   "zero"           treat the missing return as 0 (survivorship-neutral in the
    #                    sense that the name stays in the sort, but it assumes a
    #                    delisting was a flat outcome, which it rarely is)
    missing_return: str = "drop"
    # "significantly over-bought" thresholds for the z-score signal families
    z_threshold: float = 1.0              # z above this = a conspicuous tilt
    ts_min_history: int = 4               # quarters of own history before a
                                          # time-series z is defined
    min_names: int = 20                   # skip a quarter with too few names
    split: str = "2014Q1"                 # discovery < split <= validation
    ann: int = 4                          # quarters per year (annualisation)
    chunksize: int = 5_000_000            # rows per read_csv chunk

    # Always a real tuple, so `for sig in cfg.signals` works. Consumers skip any
    # name not present in the frame, and evaluate() additionally unions in
    # all_signals(sq), so a signal added to build_stock_quarter is still picked up
    # even if it is missing from this list.
    signals: tuple = ALL_SIGNALS

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


def _bucket(x: pd.Series, n_bins: int, tie_break: str):
    """Cross-sectional buckets, 0 = lowest. Handles TIED signal values honestly.

    n_buy and n_funds are integer COUNTS, so a large share of the cross-section
    shares a value. rank(method="first") then splits a tied group by DataFrame row
    order -- and rows are ordered by security id, which proxies listing age. That
    silently sorts on firm age inside every tied group; measured on synthetic data
    with a modest age/return relation it produced a spurious spread of ~6 sd.

      "random"  (default) ties are ordered by a hash of the row's IDENTITY, so
                buckets are equal-sized and the result is exactly invariant to row
                order. Ties are split arbitrarily -- but arbitrarily is unbiased,
                and it keeps every signal on the same 10%/10% footing.
      "average" tied values share a rank, so a bucket edge can never fall inside a
                tied group. Never splits a tie, but for a heavily discrete signal
                the buckets become wildly uneven: on a Poisson(1.5) draw the
                "bottom decile" swallowed 57% of the cross-section, which is no
                longer a decile sort and is not comparable across signals.
      "first"   the old row-order behaviour. Kept only for comparison.
    """
    if tie_break == "average":
        r = x.rank(method="average")
    elif tie_break == "random":
        # The jitter must be keyed to the row's IDENTITY, not its position -- a
        # position-keyed permutation is still a function of row order and keeps the
        # very bias this is meant to remove.
        j = pd.Series([hash((3407, v)) % 1_000_003 for v in x.index], index=x.index)
        o = pd.DataFrame({"x": x, "j": j}).sort_values(["x", "j"]).index
        r = pd.Series(np.arange(len(o)), index=o).reindex(x.index)
    elif tie_break == "first":
        r = x.rank(method="first")
    else:
        raise ValueError(f"tie_break must be average|random|first, got {tie_break!r}")
    try:
        return pd.qcut(r, n_bins, labels=False, duplicates="drop")
    except (ValueError, IndexError):
        return None


# columns that are keys, outcomes or bookkeeping -- never signals
NON_SIGNAL = {"security", "yq", "qi", "market_cap", "n_days", "sample",
              "log_nh", "log_nf",          # regression helpers, not signals
              "quarterly_ret", "future_1q_ret", "future_2q_ret", "future_3q_ret",
              "n_ts_defined"}


def _nanmean_abs(x) -> float:
    """Mean |x| ignoring NaN; NaN when nothing is defined (no warning)."""
    v = np.abs(np.asarray(x, dtype="float64"))
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else np.nan


def all_signals(sq: pd.DataFrame) -> list:
    """Every numeric column of `sq` that is a candidate signal.

    Used when Config.signals is None so that adding a signal to
    build_stock_quarter() is enough -- no second list to keep in sync.
    """
    return [c for c in sq.columns
            if c not in NON_SIGNAL and pd.api.types.is_numeric_dtype(sq[c])]


def _zscore_within(x: pd.Series, keys: List[pd.Series]) -> pd.Series:
    """Cross-sectional z of `x` within each group -- how conspicuous is this
    position INSIDE the fund's own book this quarter."""
    g = x.groupby(keys, sort=False)
    mu, sd = g.transform("mean"), g.transform("std")
    return (x - mu) / sd.where(sd > 0)


def _zscore_vs_own_past(df: pd.DataFrame, value: str, keys: List[str],
                        min_history: int) -> pd.Series:
    """z of `value` against the group's OWN STRICTLY PRIOR history.

    Answers "does this quarter look unusual for THIS fund in THIS name", which the
    cross-sectional z cannot: a fund that always holds 8% of one stock is not
    over-buying it, it is just concentrated.

    Vectorised with cumulative sums (expanding() would crawl on a 37M-row panel).
    The current row is removed from its own mean and sd, so there is no look-ahead
    and no self-inclusion.

    Caveat: "past" means past OBSERVATIONS of this pair, which for a position the
    fund exited and re-entered can straddle a gap. That is the right notion for a
    long-run average, unlike a one-quarter lag, which must be a strict join.
    """
    d = df.sort_values(keys + ["qi"])
    v = d[value]
    g = v.groupby([d[k] for k in keys], sort=False)
    n_past = g.cumcount()                                  # rows strictly before
    sum_past = g.cumsum() - v
    sq_past = (v ** 2).groupby([d[k] for k in keys], sort=False).cumsum() - v ** 2
    mu = sum_past / n_past.where(n_past > 0)
    var = (sq_past / n_past.where(n_past > 0) - mu ** 2) * (
        n_past / (n_past - 1).where(n_past > 1))
    sd = np.sqrt(var.where(var > 0))
    z = (v - mu) / sd
    return z.where(n_past >= min_history).reindex(df.index)


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
                                "ShsHldVal", "MARKET_CAP", "SharesHld"]
                    if c and c in avail]
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
    if "SharesHld" not in df.columns:
        df["SharesHld"] = np.nan
    df = df.rename(columns={"SharesHld": "shares"})
    for c in ("position_value", "market_cap", "shares"):
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
    return r[["security", "yq", "qi", "n_days", "quarterly_ret", "future_1q_ret",
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
    # Four things a fund's "position" can be measured by. They are NOT equivalent:
    #   active_weight  w_real - w_ref : tilt vs a cap-weighted version of the book
    #   weight         w_real         : portfolio weight -- rises when the stock
    #                                   outperforms even if the fund never traded
    #   market_weight  w_ref          : the stock's share of the book's total cap.
    #                                   Identical for every fund holding it, so as a
    #                                   "buy" rule it is a placebo, included as one
    #   shares         SharesHld      : the actual position size. The only one
    #                                   immune to price moves -- but sensitive to
    #                                   splits if SharesHld is not split-adjusted
    LEVELS = ["active_weight", "weight", "market_weight", "shares"]
    df["weight"] = df["w_real"]
    df["market_weight"] = df["w_ref"]
    keep = ["fund", "security", "qi"] + LEVELS
    cur = df[keep].copy()
    cur["held_now"] = True
    prev = df[keep].copy()
    prev["qi"] += 1                                  # STRICT one-quarter lag
    prev = prev.rename(columns={c: f"{c}_lag1" for c in LEVELS})
    prev["held_prev"] = True

    if cfg.absent_is_zero:
        u = cur.merge(prev, on=["fund", "security", "qi"], how="outer")
        # keep only fund-quarters the fund actually reported, so a fund that stops
        # filing does not generate phantom exits forever
        fq = df[["fund", "qi", "yq"]].drop_duplicates()
        u = u.merge(fq, on=["fund", "qi"], how="inner")
        u["held_now"] = u["held_now"].fillna(False)
        u["held_prev"] = u["held_prev"].fillna(False)
        for c in LEVELS:                       # absent on either side -> 0
            u[c] = u[c].fillna(0.0)
            u[f"{c}_lag1"] = u[f"{c}_lag1"].fillna(0.0)
    else:
        u = cur.merge(prev, on=["fund", "security", "qi"], how="left")
        u = u.merge(df[["fund", "qi", "yq"]].drop_duplicates(),
                    on=["fund", "qi"], how="left")
        u["held_prev"] = u["held_prev"].fillna(False)

    # one buy/sell flag per definition of "position"
    SUF = {"active_weight": "aw", "weight": "w", "market_weight": "mw",
           "shares": "sh"}
    for c in LEVELS:
        d = u[c] - u[f"{c}_lag1"]
        u[f"d_{SUF[c]}"] = d
        # A fund must still HOLD the name to be buying it. Without this guard,
        # exiting an UNDERweight position (a < 0 -> 0) shows up as d > 0 and would
        # be miscounted as a purchase.
        u[f"buy_{SUF[c]}"] = (d > 0) & u["held_now"]
        u[f"sell_{SUF[c]}"] = (d < 0) & u["held_prev"]
    u["d_aw"] = u["d_aw"]                       # the default definition
    u["buy"] = u["buy_aw"]
    u["sell"] = u["sell_aw"]
    # two definitions of "holds it at all"
    u["hold_w"] = u["held_now"] & (u["weight"] > 0)
    u["hold_aw"] = u["held_now"] & (u["active_weight"] > 0)

    n_exit = int((~u["held_now"] & u["held_prev"]).sum())
    n_init = int((u["held_now"] & ~u["held_prev"]).sum())
    print(f"[signal] grid {len(u):,} rows | initiations {n_init:,} "
          f"({n_init / max(len(u), 1):.1%}) | exits {n_exit:,} "
          f"({n_exit / max(len(u), 1):.1%}) | absent_is_zero={cfg.absent_is_zero}")

    # ---------- z-score families: how CONSPICUOUS is the tilt? ----------------
    # n_buy/n_funds are headcounts of direction. These instead measure INTENSITY:
    #   cross-sectional  is this position unusual inside the fund's own book now?
    #   time-series      is it unusual for this fund in this name vs its own past?
    h = u[u["held_now"]].copy()
    k = cfg.z_threshold
    h["z_aw"] = _zscore_within(h["active_weight"], [h["fund"], h["qi"]])
    h["z_w"] = _zscore_within(h["weight"], [h["fund"], h["qi"]])
    h["z_ts_aw"] = _zscore_vs_own_past(h, "active_weight", ["fund", "security"],
                                       cfg.ts_min_history)
    h["z_ts_w"] = _zscore_vs_own_past(h, "weight", ["fund", "security"],
                                      cfg.ts_min_history)
    for c in ("z_aw", "z_w", "z_ts_aw", "z_ts_w"):
        h[f"{c}_hi"] = h[c] > k                 # conspicuously OVER-weight
        h[f"{c}_abs_hi"] = h[c].abs() > k       # conspicuous EITHER WAY -- funds
                                                # hold a strong view on this name,
                                                # over- or under-weight. A different
                                                # hypothesis: attention/disagreement
                                                # rather than direction.

    zq = h.groupby(["security", "yq"], observed=True).agg(
        n_holders=("fund", "size"),
        # ---- FRACTION: share of holders for whom the tilt is conspicuous ------
        frac_zaw_hi=("z_aw_hi", "mean"),
        frac_zw_hi=("z_w_hi", "mean"),
        frac_ts_aw_hi=("z_ts_aw_hi", "mean"),
        frac_ts_w_hi=("z_ts_w_hi", "mean"),
        # ---- COUNT: how MANY funds, not what share --------------------------
        # These inherit the coverage problem that made n_buy ~= n_funds: a widely
        # held name has more funds above any threshold simply because it has more
        # holders. Compare each against n_holders before believing it.
        n_zaw_hi=("z_aw_hi", "sum"),
        n_zw_hi=("z_w_hi", "sum"),
        n_ts_aw_hi=("z_ts_aw_hi", "sum"),
        n_ts_w_hi=("z_ts_w_hi", "sum"),
        # ---- |z| > k : conspicuous in EITHER direction ----------------------
        frac_zaw_abs_hi=("z_aw_abs_hi", "mean"),
        frac_ts_aw_abs_hi=("z_ts_aw_abs_hi", "mean"),
        n_zaw_abs_hi=("z_aw_abs_hi", "sum"),
        n_ts_aw_abs_hi=("z_ts_aw_abs_hi", "sum"),
        # ---- CONTINUOUS: no threshold, no ties, usually better behaved -------
        mean_z_aw=("z_aw", "mean"),
        mean_z_w=("z_w", "mean"),
        mean_z_ts_aw=("z_ts_aw", "mean"),
        mean_z_ts_w=("z_ts_w", "mean"),
        # continuous |z|: average conspicuousness, no threshold
        # nanmean of an all-NaN group is NaN by design (no history yet), so the
        # empty-slice warning is suppressed rather than papered over with a 0
        mean_abs_z_aw=("z_aw", lambda x: _nanmean_abs(x)),
        mean_abs_z_ts_aw=("z_ts_aw", lambda x: _nanmean_abs(x)),
        # summed z: a count-like continuous measure (intensity x breadth)
        sum_z_aw=("z_aw", "sum"),
        sum_z_ts_aw=("z_ts_aw", "sum"),
        n_ts_defined=("z_ts_aw", "count"),
    ).reset_index()
    for c in ("n_zaw_hi", "n_zw_hi", "n_ts_aw_hi", "n_ts_w_hi",
              "n_zaw_abs_hi", "n_ts_aw_abs_hi"):
        zq[c] = zq[c].astype("int64")
    # count signals with coverage projected out, the same control n_buy_resid gets
    zq["log_nh"] = np.log(zq["n_holders"].clip(lower=1))

    def _resid_z(gg, col):
        x, y = gg["log_nh"], gg[col]
        m = x.notna() & y.notna()
        if m.sum() < 30:
            return pd.Series(np.nan, index=gg.index)
        b0, b1 = np.polyfit(x[m], y[m], 1)
        return y - (b0 * x + b1)

    for col in ("n_zaw_hi", "n_ts_aw_hi", "n_zaw_abs_hi", "n_ts_aw_abs_hi"):
        zq[f"{col}_resid"] = zq.groupby("yq", group_keys=False).apply(
            _resid_z, col)
    # |residual|: not "more overweighters than expected" but "ABNORMALLY MANY OR
    # FEW". A stock whose overweighter count is far below what its coverage
    # predicts is just as unusual as one far above -- this scores both.
    for col in ("n_zaw_hi_resid", "n_ts_aw_hi_resid", "n_zaw_abs_hi_resid",
                "n_ts_aw_abs_hi_resid"):
        if col in zq.columns:
            zq[f"{col}_abs"] = zq[col].abs()
    zq = zq.drop(columns=["log_nh"])
    print(f"[signal] z-scores: cross-sectional defined on "
          f"{h['z_aw'].notna().mean():.1%} of held rows, time-series on "
          f"{h['z_ts_aw'].notna().mean():.1%} (needs >{cfg.ts_min_history}q history)")

    sq = u.groupby(["security", "yq"], observed=True).agg(
        n_funds=("held_now", "sum"),       # coverage: funds actually holding at t
        n_active=("fund", "size"),         # held at t OR t-1: the rate denominator
        n_buy=("buy", "sum"),              # buying  (== n_buy_aw)
        n_sell=("sell", "sum"),            # selling, now including full exits
        # --- n_buy under each definition of "position" ---
        n_buy_aw=("buy_aw", "sum"),
        n_buy_w=("buy_w", "sum"),
        n_buy_mw=("buy_mw", "sum"),
        n_buy_sh=("buy_sh", "sum"),
        # --- n_funds under each definition of "holds it" ---
        n_funds_w=("hold_w", "sum"),
        n_funds_aw=("hold_aw", "sum"),
    ).reset_index()
    for c in ("n_buy_aw", "n_buy_w", "n_buy_mw", "n_buy_sh",
              "n_funds_w", "n_funds_aw"):
        sq[c] = sq[c].astype("int64")
    sq["n_funds"] = sq["n_funds"].astype("int64")
    sq["frac_buy"] = sq["n_buy"] / sq["n_active"].where(sq["n_active"] > 0)
    sq["net_buy"] = sq["n_buy"] - sq["n_sell"]

    sq = sq.merge(zq, on=["security", "yq"], how="left")

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
    d = sq[["yq", "security", signal, ret, "market_cap"]].dropna(subset=[signal])
    if cfg.missing_return == "zero":
        d = d.assign(**{ret: d[ret].fillna(0.0)})
    elif cfg.missing_return == "drop":
        d = d.dropna(subset=[ret])
    else:
        raise ValueError(f"missing_return must be drop|zero, "
                         f"got {cfg.missing_return!r}")
    if cfg.nonzero_only:
        d = d[d[signal] != 0]

    def _bin(x: pd.DataFrame) -> float:
        if len(x) < cfg.min_names:
            return np.nan
        b = _bucket(x[signal], cfg.n_bins, cfg.tie_break)
        if b is None or b.nunique() < 2:
            return np.nan
        return x.loc[b == b.max(), ret].mean() - x.loc[b == b.min(), ret].mean()

    ng = cfg.size_groups if cfg.size_neutral else 1

    def _quarter(q: pd.DataFrame) -> float:
        if ng <= 1:
            return _bin(q)
        q = q.dropna(subset=["market_cap"])
        if len(q) < cfg.min_names:
            return np.nan
        grp = _bucket(q["market_cap"], ng, cfg.tie_break)
        if grp is None:
            return _bin(q)
        vals = [_bin(sub) for _, sub in q.groupby(grp)]
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
    del panel
    return evaluate(sq, cfg, verbose=verbose)


def evaluate(sq: pd.DataFrame, cfg: Config = Config(),
             verbose: bool = True) -> Result:
    """Evaluate an ALREADY-BUILT security-quarter frame.

    Split out from run() so a notebook (or the robustness sweep) can build `sq`
    once and re-evaluate it under many configurations without re-reading 23
    parquet files each time.
    """
    split = pd.Period(cfg.split, freq="Q")
    declared = list(cfg.signals) if cfg.signals else []
    if tuple(declared) == ALL_SIGNALS or not declared:
        # the DEFAULT: evaluate everything, and union in anything the frame has
        # that ALL_SIGNALS forgot, so a new signal is never silently dropped
        found = all_signals(sq)
        signals = [c for c in declared if c in sq.columns] + \
                  [c for c in found if c not in declared]
    else:
        # an EXPLICIT restriction -- honour it exactly, add nothing
        signals = [c for c in declared if c in sq.columns]
    rows: List[dict] = []
    spreads: Dict[str, pd.Series] = {}
    for sig in signals:
        if sig not in sq.columns:
            continue
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

    # how tied is each signal? a nearly-degenerate sort is worth seeing.
    bal = []
    last = sq["yq"].max()
    for sig in signals:
        x = sq.loc[sq["yq"].eq(last), sig].dropna()
        if len(x) < cfg.min_names:
            continue
        b = _bucket(x, cfg.n_bins, cfg.tie_break)
        if b is None:
            continue
        vc = b.value_counts().sort_index()
        bal.append(dict(signal=sig, n=len(x), n_distinct=int(x.nunique()),
                        n_buckets=int(vc.size), bottom=int(vc.iloc[0]),
                        top=int(vc.iloc[-1]),
                        max_share=float(vc.max() / len(x))))
    if verbose:
        try:
            print_missing_report(sq, cfg, signals[0])
        except Exception as e:
            print(f"[warn] missing_report failed: {type(e).__name__}: {e}")

    if bal and verbose:
        bdf = pd.DataFrame(bal)
        print(f"\n=== bucket balance in {last} (tie_break={cfg.tie_break!r}) ===")
        print(bdf.round(3).to_string(index=False))
        bad = bdf[bdf.max_share > 0.25]
        if len(bad):
            print(f"[warn] heavily tied: {list(bad.signal)} -- one bucket holds "
                  f">25% of the cross-section, so this is not really a decile sort")

    c = sq[["n_buy", "n_funds", "frac_buy", "market_cap"]].dropna()
    corr = c.assign(log_mktcap=np.log(c.market_cap.where(c.market_cap > 0)))[
        ["n_buy", "n_funds", "frac_buy", "log_mktcap"]].corr()

    if verbose:
        print("\n=== signal correlations (why n_buy ~ coverage) ===")
        print(corr.round(3).to_string())
        n_all = int((perf["sample"] == "all").sum())
        print(f"\n=== performance: sample=='all' ONLY ({n_all} of {len(perf)} rows) "
              f"===")
        print("    the returned .perf also holds 'discovery' and 'validation' rows;"
              "\n    filter it, e.g. perf[perf['sample'] == 'all']")
        show = perf[perf["sample"] == "all"].set_index(["signal", "horizon"])
        print(show[["n_quarters", "mean_q", "t_nw", "sharpe_ann", "hit",
                    "ann_return"]].round(4).to_string())
    return Result(stock_q=sq, perf=perf, spreads=spreads, corr=corr)



def missing_report(sq: pd.DataFrame, cfg: Config = Config(),
                   signal: str = "n_buy") -> Dict[str, pd.DataFrame]:
    """How much data does the sort silently throw away, and is it random?

    decile_spread() does `dropna(subset=[signal, ret])`, so any security-quarter
    without a forward return is dropped listwise, before the buckets are formed. A
    forward return goes missing when the security has NO row in the returns file
    that quarter -- i.e. it delisted (acquired, bankrupt, moved).

    That is not random. Delisting returns are extreme, and delisting is likelier
    for names funds are dumping -- so missingness can correlate with the signal.
    Simulated with a plausible pattern (27% delisting in the lowest n_buy decile vs
    3% in the highest, -40% delisting return), dropping them turned a true +6.8%
    spread into -0.7%: the SIGN FLIPPED.

    Returns three frames:
      by_column  overall NaN rate per return column
      by_bucket  NaN rate per signal decile -- if this slopes, the drop is biased
      partial    quarters built from few trading days (a stub, not a quarter)
    """
    rets = [c for c in ("quarterly_ret", "future_1q_ret", "future_2q_ret",
                        "future_3q_ret") if c in sq.columns]
    by_column = pd.DataFrame([
        dict(column=c, n=len(sq), n_missing=int(sq[c].isna().sum()),
             pct_missing=float(sq[c].isna().mean())) for c in rets])

    rows = []
    if signal in sq.columns:
        for c in rets:
            d = sq[[signal, c, "yq"]].dropna(subset=[signal])
            b = _bucket(d[signal], cfg.n_bins, cfg.tie_break)
            if b is None:
                continue
            g = d.assign(bucket=b, miss=d[c].isna()).groupby("bucket")["miss"].mean()
            for k, v in g.items():
                rows.append(dict(column=c, bucket=int(k), pct_missing=float(v)))
    by_bucket = (pd.DataFrame(rows).pivot(index="bucket", columns="column",
                                          values="pct_missing")
                 if rows else pd.DataFrame())

    partial = pd.DataFrame()
    if "n_days" in sq.columns:
        nd = sq["n_days"].dropna()
        partial = pd.DataFrame([dict(
            median_days=float(nd.median()),
            pct_under_20_days=float((nd < 20).mean()),
            pct_under_40_days=float((nd < 40).mean()),
            n_quarter_stubs=int((nd < 20).sum()))])
    return dict(by_column=by_column, by_bucket=by_bucket, partial=partial)


def print_missing_report(sq: pd.DataFrame, cfg: Config = Config(),
                         signal: str = "n_buy") -> None:
    r = missing_report(sq, cfg, signal)
    print("\n=== missing returns: what the sort drops ===")
    print(r["by_column"].round(4).to_string(index=False))
    if len(r["by_bucket"]):
        print(f"\nNaN rate by {signal} decile (0 = lowest). A SLOPE here means the "
              f"drop is not random:")
        print(r["by_bucket"].round(4).to_string())
        c = "future_1q_ret"
        if c in r["by_bucket"].columns:
            b = r["by_bucket"][c]
            lo, hi = b.iloc[0], b.iloc[-1]
            print(f"  bottom decile {lo:.1%} vs top decile {hi:.1%}"
                  f"  (ratio {lo / max(hi, 1e-9):.1f}x)")
            if abs(lo - hi) > 0.02:
                print("  [warn] missingness is correlated with the signal -- the "
                      "listwise drop is a DELISTING BIAS, not a neutral filter")
    if len(r["partial"]):
        print("\nquarters built from partial data (delisting stubs):")
        print(r["partial"].round(4).to_string(index=False))


# ================================================================= ROBUSTNESS
TIE_BREAKS = ("first", "average", "random")
SIZE_GROUPS = (1, 2, 3, 4, 5)
NONZERO = (False, True)


def sweep(sq: pd.DataFrame, cfg: Config, signals=None, horizons=(1, 2, 3),
          tie_breaks=TIE_BREAKS, size_groups=SIZE_GROUPS, nonzero=NONZERO,
          verbose: bool = True) -> pd.DataFrame:
    """Re-run portfolio formation across every arbitrary construction choice.

    Same signal, same data -- only the choices differ. A result that holds across
    the grid is a result; one that appears in a few cells is a choice. On a
    synthetic signal built to carry ZERO information, |t| >= 2 still showed up in
    83% of the 30 cells, so this grid is not optional for the count signals.

    `sq` is built once by build_stock_quarter(); only formation is repeated.
    """
    from dataclasses import replace
    import itertools

    signals = list(signals or cfg.signals or all_signals(sq))
    split = pd.Period(cfg.split, freq="Q")
    rows = []
    combos = list(itertools.product(tie_breaks, size_groups, nonzero))
    for i, (tb, ng, nz) in enumerate(combos, 1):
        c = replace(cfg, tie_break=tb, size_groups=ng, size_neutral=(ng > 1),
                    nonzero_only=nz)
        for sig in signals:
            if sig not in sq.columns:
                continue
            for h in horizons:
                sp = decile_spread(sq, sig, h, c)
                if len(sp) < 8:
                    continue
                for sample in ("discovery", "validation", "all"):
                    perf = performance(sp[_sample_mask(sp.index, sample, split)],
                                       h, c)
                    if perf.get("n_quarters", 0) < 8:
                        continue
                    perf.update(signal=sig, horizon=h, sample=sample,
                                tie_break=tb, size_groups=ng, nonzero_only=nz)
                    rows.append(perf)
        if verbose:
            print(f"  [{i}/{len(combos)}] tie_break={tb:7s} size_groups={ng} "
                  f"nonzero_only={nz}", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------- missing-return policy comparison
def sweep_missing(sq: pd.DataFrame, cfg: Config, signals=None, horizons=(1, 2, 3),
                  tie_break: str = "average", size_groups: int = 3) -> pd.DataFrame:
    """Drop the missing forward returns, or call them zero?

    "drop" is listwise deletion of securities that vanished from the returns file
    -- overwhelmingly delistings. That is the standard silent choice and it is a
    survivorship filter: delisting is non-random and correlates with the signal.
    "zero" keeps the name in the sort but asserts the delisting outcome was flat,
    which is generous to bankruptcies and stingy to takeovers.

    Neither is right. If they DISAGREE, the result depends on dead companies and
    the honest fix is real delisting returns (CRSP DLRET), not a choice between
    these two.
    """
    from dataclasses import replace
    signals = list(signals or cfg.signals or all_signals(sq))
    split = pd.Period(cfg.split, freq="Q")
    rows = []
    for policy in ("drop", "zero"):
        c = replace(cfg, missing_return=policy, tie_break=tie_break,
                    size_groups=size_groups, size_neutral=(size_groups > 1))
        for sig in signals:
            if sig not in sq.columns:
                continue
            for h in horizons:
                sp = decile_spread(sq, sig, h, c)
                if len(sp) < 8:
                    continue
                for sample in ("discovery", "validation", "all"):
                    perf = performance(sp[_sample_mask(sp.index, sample, split)],
                                       h, c)
                    if perf.get("n_quarters", 0) < 8:
                        continue
                    perf.update(signal=sig, horizon=h, sample=sample,
                                missing_return=policy)
                    rows.append(perf)
    res = pd.DataFrame(rows)
    if len(res):
        piv = res[res["sample"] == "all"].pivot_table(
            index=["signal", "horizon"], columns="missing_return", values="t_nw")
        if {"drop", "zero"}.issubset(piv.columns):
            piv["delta"] = piv["zero"] - piv["drop"]
            piv["sign_flip"] = (piv["drop"] * piv["zero"]) < 0
        res.attrs["summary"] = piv
    return res


# ------------------------------------------- alternative signal definitions
BUY_DEFS = {
    "n_buy_aw": "active_weight rose  (default: tilt vs cap-weighted book)",
    "n_buy_w": "weight rose          (contaminated: rises on price alone)",
    "n_buy_mw": "market_weight rose   (PLACEBO: identical for all funds)",
    "n_buy_sh": "shares rose          (the actual trade; split-sensitive)",
}
HOLD_DEFS = {
    "n_funds_w": "weight > 0         (holds it at all)",
    "n_funds_aw": "active_weight > 0  (OVERweights it vs cap)",
}


def sweep_definitions(sq: pd.DataFrame, cfg: Config, horizons=(1, 2, 3),
                      tie_break: str = "average", size_groups: int = 3,
                      which: str = "buy") -> pd.DataFrame:
    """Same machinery, different definition of 'bought' / 'holds'.

    which='buy'  -> n_buy under active_weight / weight / market_weight / shares
    which='hold' -> n_funds under weight>0 / active_weight>0

    n_buy_mw is a deliberate placebo: market_weight is a property of the SECURITY,
    so every fund holding it gets the same flag and the "count of buyers" collapses
    to a rescaled holder count. If it scores like the others, the signal is not
    about fund behaviour.
    """
    defs = BUY_DEFS if which == "buy" else HOLD_DEFS
    sigs = [k for k in defs if k in sq.columns]
    res = sweep_missing(sq, cfg, sigs, horizons, tie_break, size_groups)
    if len(res):
        res["definition"] = res["signal"].map(defs)
    return res


def sweep_table(res: pd.DataFrame, signal: str, horizon: int = 1,
                sample: str = "all", value: str = "t_nw") -> pd.DataFrame:
    """One readable grid: (nonzero_only, tie_break) x size_groups."""
    d = res[(res.signal == signal) & (res.horizon == horizon)
            & (res["sample"] == sample)]
    if d.empty:
        return pd.DataFrame()
    return d.pivot_table(index=["nonzero_only", "tie_break"],
                         columns="size_groups", values=value)


def sweep_verdict(res: pd.DataFrame, signals=None, horizon: int = 1,
                  sample: str = "all") -> pd.DataFrame:
    """Per signal: does anything survive the grid, or is it a construction choice?"""
    out = []
    for sig in (signals or res.signal.unique()):
        d = res[(res.signal == sig) & (res.horizon == horizon)
                & (res["sample"] == sample)]
        t = d["t_nw"].dropna()
        if t.empty:
            continue
        out.append(dict(signal=sig, horizon=horizon, n_cells=len(t),
                        t_min=t.min(), t_max=t.max(), t_median=t.median(),
                        share_abs_t_ge2=float((t.abs() >= 2).mean()),
                        sign_flips=bool(t.min() < 0 < t.max()),
                        robust=bool((t.abs() >= 2).all()
                                    and not (t.min() < 0 < t.max()))))
    return pd.DataFrame(out)

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
