"""buy_pressure.py -- predict next-quarter BUYING PRESSURE on a security, then test for alpha.

A security-level study, distinct from the fund-security work in company_replication.py.

    unit of observation : (security, quarter)
    target              : buy_frac(s, q) = # funds buying s / # funds owning s over q -> q+1
    features            : latest Barra GEMLT exposures as of quarter end
                          past security returns
                          LAGS of buy_frac / sell_frac, plus ownership breadth
    alpha test          : rank securities by PREDICTED buy_frac, then look at forward returns

TIMING -- the thing most easily got wrong here
----------------------------------------------
Write I_t for everything knowable once the quarter-t holdings snapshot exists. buy_frac over
[q, q+1] compares holdings at q against holdings at q+1, so

    buy_frac(s, q)  is I_{q+1}-measurable, NOT I_q-measurable

even though the column sits physically on the q row (exactly like future_1q_ret does). The
row is dated q, so:

    features   every one I_q-measurable. buy_frac enters ONLY through its lags: buy_frac(q-1)
               closed at q, so it is I_q-measurable; buy_frac(q) is not and is excluded.
    target     buy_frac(s, q), the buying over q -> q+1. Strictly after the features, so this
               is not look-ahead -- and it is the TIGHTEST honest choice. Shifting the target
               one further quarter would skip a whole quarter for no gain in validity and
               would cut the attainable IC from rho_1 to rho_2.
    returns    named by the DECISION POINT, since pred_buy_frac already exists at the q close
               "quarter_end"  fwd_1q -- trade at the q close, hold q -> q+1. The forecast
                              precedes the whole window, so this is unbiased AND the headline.
               "one_q_delay"  fwd_2q -- only if the holdings behind buy_frac_lag1 arrive late
               "two_q_delay"  fwd_3q -- a full ~45-60 day filing delay plus a quarter

CAUTION, and it is the one asymmetry here: alpha_sort(on="buy_frac") ranks on REALISED
buying, which is I_{q+1}-measurable. At "quarter_end" that overlaps fwd_1q and IS biased --
read it as a perfect-foresight ceiling, never as a strategy. alpha_sort(on="pred_buy_frac")
has no such problem at any horizon.

The label matches company_replication.py -- the q -> q+1 share-change direction -- so the two
modules stay comparable. What differs is only which horizon is the honest headline, because
there the sort variable is realised accuracy and here it is a pure forecast.

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
    min_owners: int = 5               # on n_holders, the count known at q -- no selection bias
    # Optional extra floor on n_owning (the LABELLED count). It sharpens the target -- a
    # buy_frac from 2 funds is noise -- but n_owning needs q+1 data, so switching it on builds
    # the evaluation universe partly out of forward information. 0 = off.
    min_labelled_owners: int = 0
    min_quarters: int = 12            # securities need some history to be sortable

    # ---- evaluation ----
    # quarter_end  decide at the close of q and trade immediately -- correct for a forecast,
    #              because pred_buy_frac is already known then and fwd_1q lies entirely after
    # one_q_delay / two_q_delay  only if the holdings feeding buy_frac_lag1 arrive late
    eval_timing: str = "quarter_end"
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

    # ---- SNAPSHOT of what is known at q, taken BEFORE any forward-looking filtering -------
    # Everything below either needs the q -> q+1 change (the label) or drops rows on the basis
    # of it (the -100% prefilter). Both shrink `df` to "positions that survive into q+1", so a
    # count taken afterwards is I_{q+1}-measurable. Since the target is buy_frac(q) and
    # n_owning is its own denominator, using that count as a feature would hand the model the
    # target's denominator for the very same quarter.
    base = df.groupby(["security", "yq"], observed=True).agg(
        n_holders=("fund", "size"),
        w_mean=("weight", "mean"),
        mktcap=("market_cap", "first"),
        ret_q=("quarterly_ret", "first"),      # return DURING q -> I_q-measurable
        ret_past=("past_1q_ret", "first"),     # return during q-1, used to verify that
        fwd_1q=("future_1q_ret", "first"),
        fwd_2q=("future_2q_ret", "first"),
        fwd_3q=("future_3q_ret", "first"),
    ).reset_index()

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
    pres = lab.groupby(["security", "yq"], observed=True).agg(
        n_owning=("fund", "size"),          # denominator of buy_frac; I_{q+1}-measurable
        buy_frac=("is_buy", "mean"),
        sell_frac=("is_sell", "mean"),
    ).reset_index()

    sq = base.merge(pres, on=["security", "yq"], how="inner")
    sq["n_buying"] = (sq["buy_frac"] * sq["n_owning"]).round()

    # Universe filter on the KNOWN-AT-q count. Filtering on `n_owning` instead would build the
    # evaluation universe out of forward information: n_owning only counts holders whose
    # q -> q+1 change is observable, so it silently requires survival into q+1.
    sq = sq[sq["n_holders"] >= cfg.min_owners]
    if cfg.min_labelled_owners:            # opt-in: trades label precision for selection bias
        sq = sq[sq["n_owning"] >= cfg.min_labelled_owners]
    ratio = float((sq["n_owning"] / sq["n_holders"]).mean())
    print(f"[hold] {len(sq):,} security-quarters | {sq.security.nunique():,} securities | "
          f"{sq.yq.nunique()} quarters | median holders {sq.n_holders.median():.0f}")
    print(f"[hold] buy_frac  mean {sq.buy_frac.mean():.3f}  sd {sq.buy_frac.std():.3f}")
    print(f"[hold] labelled/holders ratio {ratio:.3f}"
          + ("" if ratio > 0.9 else
             "  <- well below 1: many positions vanish next quarter, so anything built on "
             "n_owning carries survival information"))

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
    # breadth from the KNOWN-AT-q holder count. `n_owning` is buy_frac's own denominator and
    # is I_{q+1}-measurable, so it may enter only through its lags, never as a level.
    df["log_owners"] = np.log(df["n_holders"].astype("float64") + 1.0).astype("float32")
    df["log_mktcap"] = np.log(df["mktcap"].abs() + 1.0).astype("float32")

    # ---- returns, exact-quarter aligned ----
    for k in (1, 2, 4):
        v, qq = g["ret_q"].shift(k), g["qi"].shift(k)
        df[f"ret_lag{k}"] = v.where(qq == qi - k).astype("float32")
    df["ret_ma4"] = df[["ret_lag1", "ret_lag2", "ret_lag4"]].mean(axis=1)

    # VERIFY the return convention rather than assuming it. If quarterly_ret(q) is the return
    # earned DURING q -- the reading that makes it I_q-measurable and usable as a feature --
    # then past_1q_ret(q) must equal quarterly_ret(q-1) = ret_lag1(q). A correlation near 1
    # confirms it; anything else means ret_q is shifted and must NOT be used as a feature.
    if "ret_past" in df.columns:
        ok = df["ret_past"].notna() & df["ret_lag1"].notna()
        rho = float(df.loc[ok, "ret_past"].corr(df.loc[ok, "ret_lag1"])) if ok.sum() > 100 else np.nan
        rho_f = float(df.loc[df["ret_q"].notna() & df["fwd_1q"].notna(), "ret_q"]
                      .corr(df["fwd_1q"])) if df["fwd_1q"].notna().any() else np.nan
        print(f"[check] corr(past_1q_ret, ret_lag1) = {rho:+.3f}  "
              f"(near +1 => quarterly_ret is the return DURING q, so it is I_q-measurable)")
        print(f"[check] corr(ret_q, fwd_1q)         = {rho_f:+.3f}  "
              f"(should be small; near +1 would mean ret_q is actually a FORWARD return)")
        if np.isfinite(rho) and rho < 0.9:
            print("  [warn] convention NOT confirmed -- drop 'ret_q' from feature_list until "
                  "you know which window quarterly_ret spans")

    # ---- TARGET: buy_frac over q -> q+1, i.e. THIS row's own buy_frac ----
    # It is I_{q+1}-measurable while every feature is I_q-measurable, so the target is
    # strictly in the future of the features -- no look-ahead, and no quarter thrown away.
    # (An extra .shift(-1) here would predict q+1 -> q+2 instead, skipping a full quarter and
    # dropping the attainable IC from rho_1 to rho_2 without buying any extra validity.)
    df["target_buy_frac"] = df["buy_frac"].astype("float32")

    # History requirement, EXPANDING rather than full-sample. `transform("size")` counts a
    # security's entire life including quarters after q, so it keeps only names that turn out
    # to survive -- survivorship bias baked into the evaluation universe. cumcount() asks the
    # I_q-measurable question instead: has this security been observed long enough BY q?
    seen = g.cumcount() + 1
    df = df[seen >= cfg.min_quarters]
    print(f"[panel] {len(df):,} rows | {df.security.nunique():,} securities | "
          f"{df.qi.max()+1} quarters | target available {df.target_buy_frac.notna().mean():.1%}")
    return df


def feature_list(df: pd.DataFrame, cfg: Config) -> List[str]:
    """Everything dated at or before the start of the target window."""
    gem = [c for c in df.columns if str(c).startswith(cfg.barra_prefix)]
    hist = [c for c in df.columns
            if c.startswith(("buy_frac_lag", "sell_frac_lag", "n_owning_lag"))]
    # ret_q is the return earned DURING q, so it is known at the q close and is a legitimate
    # feature -- and given that weight-targeting funds ADD shares to names that fell, it is
    # probably the single most informative variable here. It is also the control you need in
    # order to argue any alpha is NOT just that rebalancing mechanic.
    # The convention is verified in build_panel; if that check fails, drop it.
    other = ["buy_frac_ma4", "buy_frac_chg", "log_owners", "log_mktcap", "w_mean",
             "ret_q", "ret_lag1", "ret_lag2", "ret_lag4", "ret_ma4"]
    return gem + hist + [c for c in other if c in df.columns]


# ============================================================ 4. MODELS
_KEEP = ["security", "qi", "yq", "target_buy_frac", "buy_frac", "buy_frac_lag1", "n_owning",
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
    """How well is buying pressure over q -> q+1 predicted from information known at q?

    The naive benchmark is buy_frac_lag1 = buy_frac(q-1), i.e. "assume this quarter looks
    like last quarter". It is I_q-measurable, so it sits on EXACTLY the information set the
    model is given -- a fair hurdle. Under the old target convention the benchmark used
    buy_frac(q) itself, which is I_{q+1}-measurable and therefore one lag closer to the
    target than anything the model could see; that comparison was rigged against the model
    and is gone.
    """
    per_q = P.groupby("qi").apply(lambda d: pd.Series({
        "rank_ic": d["pred_buy_frac"].corr(d["target_buy_frac"], method="spearman"),
        "pearson": d["pred_buy_frac"].corr(d["target_buy_frac"]),
        "n": len(d)}))
    rows = [{"metric": "rank IC (model)", "mean": per_q["rank_ic"].mean(),
             "t": _t(per_q["rank_ic"]), "n_quarters": int(per_q["n"].size)}]
    if "buy_frac_lag1" in P.columns:
        naive = P.groupby("qi").apply(
            lambda d: d["buy_frac_lag1"].corr(d["target_buy_frac"], method="spearman"))
        rows.append({"metric": "rank IC (naive: buy_frac_lag1, same information as model)",
                     "mean": naive.mean(), "t": _t(naive), "n_quarters": int(naive.size)})
        edge = per_q["rank_ic"] - naive.reindex(per_q.index)
        rows.append({"metric": "  -> model minus naive (the only number that matters)",
                     "mean": edge.mean(), "t": _t(edge.dropna()),
                     "n_quarters": int(edge.notna().sum())})
    rows.append({"metric": "pearson (model)", "mean": per_q["pearson"].mean(),
                 "t": _t(per_q["pearson"]), "n_quarters": int(per_q["n"].size)})
    return pd.DataFrame(rows)


# Timing is named by the DECISION POINT, not by the return window, because what makes a
# horizon honest here is when the sort variable became knowable.
#
#   pred_buy_frac is I_q-measurable -- it exists the moment quarter q closes. Trading on it
#   at the q close and holding through q+1 earns fwd_1q, and the whole return window lies
#   AFTER the forecast. There is no overlap and no bias: "quarter_end" is the headline.
#
#   The old name for this was "contemporaneous", carried over from company_replication.py
#   where the sort variable is realised accuracy (I_{q+1}-measurable) and therefore DOES
#   overlap fwd_1q. That warning applies to alpha_sort(on="buy_frac") here, never to
#   alpha_sort(on="pred_buy_frac"). Old names still work.
_TIMING = {"quarter_end": "fwd_1q",    # decide at the close of q, hold q -> q+1
           "one_q_delay": "fwd_2q",    # decide at the close of q, wait a quarter, hold q+1 -> q+2
           "two_q_delay": "fwd_3q"}    # wait two quarters -- only needed if holdings arrive late
_TIMING_ALIAS = {"contemporaneous": "quarter_end",
                 "predictive": "one_q_delay",
                 "tradeable": "two_q_delay"}


def _canon(timing: str) -> str:
    """Canonical timing name, accepting the legacy ones."""
    t = _TIMING_ALIAS.get(timing, timing)
    if t not in _TIMING:
        raise KeyError(f"unknown timing {timing!r}; use one of {list(_TIMING)} "
                       f"(legacy names {list(_TIMING_ALIAS)} also work)")
    return t


def _ret_col(timing: str) -> str:
    """Return column for a timing name, accepting the legacy names."""
    return _TIMING[_canon(timing)]


class _TimingDict(dict):
    """Results keyed by canonical timing name, but readable through the legacy names too,
    so notebooks and configs written against `predictive` / `tradeable` keep working."""

    def __missing__(self, key):
        alias = _TIMING_ALIAS.get(key)
        if alias is not None and alias in self:
            return self[alias]
        raise KeyError(
            f"{key!r} is not a timing. Available: {list(self)} "
            f"(legacy names {list(_TIMING_ALIAS)} map onto them in order).")


def alpha_sort(P: pd.DataFrame, cfg: Config, timing=None, on="pred_buy_frac") -> pd.DataFrame:
    """Rank securities on predicted buying pressure -> forward returns by quintile.

    `on="pred_buy_frac"` is the question asked.
    `on="buy_frac"` sorts on the REALISED buying, which under this target convention is the
    target itself -- a perfect-foresight sort. Not a strategy (it is I_{q+1}-measurable), but
    the ceiling: if even knowing the answer exactly earns nothing at `tradeable` timing, then
    no forecast of buying pressure can, and the sign on the predicted sort is coming from
    somewhere other than buying pressure.
    """
    timing = timing or cfg.eval_timing
    col = _ret_col(timing)
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
         "alpha_pred": _TimingDict(
             (t, alpha_sort(P, cfg, t, on="pred_buy_frac")) for t in _TIMING),
         "alpha_actual": _TimingDict(
             (t, alpha_sort(P, cfg, t, on="buy_frac")) for t in _TIMING)}
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
