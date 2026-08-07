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
    # ---- what "buying pressure" means -------------------------------------------------
    # breadth        share of holders raising SHARE count. One vote per fund: a $10m fund and
    #                a $10bn fund count the same, and a 1% trim counts like a full exit.
    # dollar_breadth same vote, weighted by position value -- big holders speak louder.
    # net_flow       sum of ACTUAL dollars traded, over market cap:
    #                    sum_f (dS_{f,s,q} * P_{s,q}) / mktcap_{s,q}
    #                This is the variable price impact actually scales with, it is continuous
    #                and signed (no +-1% dead band), and it does not carry n_owning in its
    #                denominator -- so it sidesteps the granularity problem in breadth.
    # NOTE net_flow uses dS * P_q, the dollars traded. It is deliberately NOT
    #      position_value(q+1) - position_value(q), which is dominated by the price move and
    #      would measure returns rather than trading.
    # weight_change  mean over holders of w(q+1) - w(q). What the manager asked for, but it
    #                DRIFTS: a stock that rallies gains weight with no decision behind it.
    # active_weight_change  the same move with that drift removed,
    #                    dw_active = w(q+1) - w(q) * (1 + r_s) / (1 + r_portfolio)
    #                so it is the part of the reallocation the manager actually chose. This is
    #                the honest reading of "how much did its weight change", and it is the same
    #                construction as `sum_abs_aw` in wrds_pull.
    pressure_measure: str = "breadth"
    # breadth | dollar_breadth | net_flow | weight_change | active_weight_change
    flow_winsor: float = 0.01           # net_flow / weight moves have fat tails; clip per quarter

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

    # ---- WEIGHT side: how much did the position's weight move, q -> q+1 -----------------
    # Raw dw mixes two different things. If a stock rallies and the manager does nothing, its
    # weight rises anyway. dw_active removes that drift by comparing next quarter's weight
    # against the weight a do-nothing manager would have ended up with:
    #     w_passive = w_q * (1 + r_s) / (1 + r_portfolio)
    # so dw_active is the part of the move the manager actually chose. Same idea as
    # `sum_abs_aw` in wrds_pull.
    df = df.sort_values(["fund", "security", "yq"])
    if "weight" in df.columns and df["weight"].notna().any():
        w = df["weight"].astype("float64")
        wmed = float(w[w.abs() > 1e-12].abs().median())
        if np.isfinite(wmed) and wmed > 0.02:            # stored as percent -> to fraction
            w = w / 100.0
            print(f"[hold] weight looks like percent (median |w| = {wmed:.3f}) -> /100")
        df["_w"] = w
        gfs = df.groupby(["fund", "security"], observed=True)
        w_next = gfs["_w"].shift(-1).where(gfs["yq"].shift(-1) == df["yq"] + 1)
        df["dw"] = (w_next - df["_w"]).astype("float32")

        if "future_1q_ret" in df.columns and df["future_1q_ret"].notna().any():
            rs = df["future_1q_ret"].astype("float64")           # r_s over q -> q+1
            ok = df["_w"].notna() & rs.notna()
            gfq = df[ok].groupby(["fund", "yq"], observed=True)
            num = gfq.apply(lambda t: float((t["_w"] * rs.loc[t.index]).sum()))
            den = gfq["_w"].sum()
            rp = (num / den.where(den != 0)).rename("rp")
            rp = df.set_index(["fund", "yq"]).index.map(rp).astype("float64")
            w_pass = df["_w"] * (1.0 + rs) / (1.0 + pd.Series(rp, index=df.index))
            df["dw_active"] = (w_next - w_pass).astype("float32")
        else:
            df["dw_active"] = np.nan
        df = df.drop(columns=["_w"])
    else:
        df["dw"] = np.nan; df["dw_active"] = np.nan

    lab = df[df["Y"].notna()].copy()
    lab["is_buy"] = (lab["Y"] > 0).astype("float32")
    lab["is_sell"] = (lab["Y"] < 0).astype("float32")

    # ---- DOLLAR side. dsh is the fractional SHARE change and position_value is S_q * P_q,
    #      so their product is dS * P_q: the dollars actually traded, with the price move
    #      stripped out. (position_value(q+1) - position_value(q) would be mostly return.)
    lab["_dsh"] = pd.to_numeric(dsh, errors="coerce").reindex(lab.index).astype("float64")
    if "position_value" in lab.columns:
        pv = lab["position_value"].astype("float64")
        lab["dollar_chg"] = lab["_dsh"] * pv
        lab["_pv_buy"] = lab["is_buy"] * pv
        lab["_pv"] = pv
    else:
        lab["dollar_chg"] = np.nan; lab["_pv_buy"] = np.nan; lab["_pv"] = np.nan

    pres = lab.groupby(["security", "yq"], observed=True).agg(
        n_owning=("fund", "size"),          # denominator of buy_frac; I_{q+1}-measurable
        buy_frac=("is_buy", "mean"),
        sell_frac=("is_sell", "mean"),
        net_dollar=("dollar_chg", "sum"),   # signed dollars traded by the funds we observe
        gross_dollar=("dollar_chg", lambda s: float(np.abs(s).sum())),
        _pv_buy=("_pv_buy", "sum"),
        _pv=("_pv", "sum"),
        weight_chg=("dw", "mean"),          # mean weight move across holders
        active_weight_chg=("dw_active", "mean"),   # same, price drift removed
    ).reset_index()
    # value-weighted breadth: same vote, weighted by how much each holder actually owns
    pres["dollar_buy_frac"] = (pres["_pv_buy"] / pres["_pv"].where(pres["_pv"] > 0)
                               ).astype("float32")
    pres = pres.drop(columns=["_pv_buy", "_pv"])

    sq = base.merge(pres, on=["security", "yq"], how="inner")
    # dollars traded as a fraction of the company -- the price-impact scaling
    sq["flow_pct_cap"] = (sq["net_dollar"] / sq["mktcap"].abs().where(sq["mktcap"].abs() > 0)
                          ).astype("float32")
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
    print(f"[hold] buy_frac        mean {sq.buy_frac.mean():+.4f}  sd {sq.buy_frac.std():.4f}")
    print(f"[hold] dollar_buy_frac mean {sq.dollar_buy_frac.mean():+.4f}  "
          f"sd {sq.dollar_buy_frac.std():.4f}")
    print(f"[hold] flow_pct_cap    mean {sq.flow_pct_cap.mean():+.5f}  "
          f"sd {sq.flow_pct_cap.std():.5f}  (dollars traded / market cap)")
    print(f"[hold] weight_chg      mean {sq.weight_chg.mean():+.5f}  "
          f"sd {sq.weight_chg.std():.5f}  (raw dw, DRIFTS with price)")
    print(f"[hold] active_wgt_chg  mean {sq.active_weight_chg.mean():+.5f}  "
          f"sd {sq.active_weight_chg.std():.5f}  (drift removed = the decision)")
    _m = ["buy_frac", "dollar_buy_frac", "flow_pct_cap", "weight_chg", "active_weight_chg"]
    _r = sq[[c for c in _m if c in sq.columns]].corr(method="spearman").round(3)
    print("[hold] rank correlations between the measures "
          "(far below 1 => they ask different questions):")
    print(_r.to_string())
    if {"weight_chg", "active_weight_chg"} <= set(sq.columns):
        rho = float(sq["weight_chg"].corr(sq["active_weight_chg"], method="spearman"))
        print(f"[hold] raw dw vs active dw  rank corr {rho:+.3f}"
              + ("" if rho < 0.95 else
                 "  <- near 1: price drift is small here, the two are interchangeable"))
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
    # net_flow has fat tails -- one index reconstitution can dwarf a quarter. Clip per quarter
    # BEFORE lagging, so the feature and the target see the same treatment.
    for _c in ("flow_pct_cap", "weight_chg", "active_weight_chg"):
        if cfg.flow_winsor and _c in df.columns and df[_c].notna().any():
            lo = df.groupby("yq")[_c].transform(lambda s: s.quantile(cfg.flow_winsor))
            hi = df.groupby("yq")[_c].transform(lambda s: s.quantile(1 - cfg.flow_winsor))
            df[_c] = df[_c].clip(lo, hi).astype("float32")

    _PRESSURE = ("buy_frac", "sell_frac", "n_owning", "dollar_buy_frac", "flow_pct_cap",
                 "weight_chg", "active_weight_chg")
    for k in (1, 2, 3, 4):
        for col in _PRESSURE:
            if col not in df.columns:
                continue
            v, qq = g[col].shift(k), g["qi"].shift(k)
            df[f"{col}_lag{k}"] = v.where(qq == qi - k).astype("float32")
    df["buy_frac_ma4"] = df[[f"buy_frac_lag{k}" for k in (1, 2, 3, 4)]].mean(axis=1)
    df["buy_frac_chg"] = df["buy_frac_lag1"] - df["buy_frac_lag2"]
    if "flow_pct_cap_lag1" in df.columns:
        df["flow_ma4"] = df[[f"flow_pct_cap_lag{k}" for k in (1, 2, 3, 4)]].mean(axis=1)
        df["flow_chg"] = df["flow_pct_cap_lag1"] - df["flow_pct_cap_lag2"]
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
    # `target_buy_frac` keeps its name for backwards compatibility; it holds whichever
    # pressure measure `cfg.pressure_measure` selects, alongside the plain alias
    # `pressure_now` and the fair naive benchmark `pressure_lag1`. Every measure and every
    # measure's lags are already on the frame, so `set_pressure(panel, other)` switches the
    # target later without rebuilding -- no second Barra join, no second copy in memory.
    df = set_pressure(df, cfg.pressure_measure)
    print(f"[panel] pressure_measure={cfg.pressure_measure!r} -> target is "
          f"{_TARGET_COL[cfg.pressure_measure]!r}   "
          f"(switch later with B.set_pressure(panel, ...) -- no rebuild needed)")

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
    # Lags of ALL THREE pressure measures are offered regardless of which one is the target.
    # They answer different questions -- breadth is agreement, net_flow is dollars -- so the
    # model (and the SHAP report) can tell you which history actually carries the signal.
    hist = [c for c in df.columns
            if c.startswith(("buy_frac_lag", "sell_frac_lag", "n_owning_lag",
                             "dollar_buy_frac_lag", "flow_pct_cap_lag",
                             "weight_chg_lag", "active_weight_chg_lag"))]
    # ret_q is the return earned DURING q, so it is known at the q close and is a legitimate
    # feature -- and given that weight-targeting funds ADD shares to names that fell, it is
    # probably the single most informative variable here. It is also the control you need in
    # order to argue any alpha is NOT just that rebalancing mechanic.
    # The convention is verified in build_panel; if that check fails, drop it.
    other = ["buy_frac_ma4", "buy_frac_chg", "flow_ma4", "flow_chg",
             "log_owners", "log_mktcap", "w_mean",
             "ret_q", "ret_lag1", "ret_lag2", "ret_lag4", "ret_ma4"]
    return gem + hist + [c for c in other if c in df.columns]


_TARGET_COL = {"breadth": "buy_frac",
               "dollar_breadth": "dollar_buy_frac",
               "net_flow": "flow_pct_cap",
               "weight_change": "weight_chg",
               "active_weight_change": "active_weight_chg"}


def set_pressure(panel: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Point the target at a different pressure measure, IN PLACE.

    build_panel already computes every measure and every measure's lags -- only the three
    target columns depend on `pressure_measure`. So switching measures needs no rebuild, and
    in particular no second Barra join, which is the slow part. Rebuilding the panel five
    times to compare five measures would also hold five full copies in memory.

    Returns the same object, so `set_pressure(panel, "net_flow")` reads naturally inline.
    """
    tcol = _TARGET_COL.get(measure)
    if tcol is None:
        raise ValueError(f"measure must be one of {list(_TARGET_COL)}, got {measure!r}")
    if tcol not in panel.columns:
        raise ValueError(f"{measure!r} needs column {tcol!r}, which the panel does not have")
    panel["target_buy_frac"] = panel[tcol].astype("float32")
    panel["pressure_now"] = panel["target_buy_frac"]
    panel["pressure_lag1"] = panel[f"{tcol}_lag1"].astype("float32")
    return panel


def feature_family(name: str, cfg: Config = None) -> str:
    """Coarse grouping used by the SHAP rollup -- the manager's question is which KIND of
    information drives the forecast, not which individual column."""
    cfg = cfg or Config()
    if str(name).startswith(cfg.barra_prefix):
        return "barra_style"
    if name.startswith(("weight_chg_lag", "active_weight_chg_lag")):
        return "weight_change_history"
    if name.startswith(("flow_pct_cap_lag", "dollar_buy_frac_lag")) or name in (
            "flow_ma4", "flow_chg"):
        return "dollar_flow_history"
    if name.startswith(("buy_frac_lag", "sell_frac_lag")) or name in (
            "buy_frac_ma4", "buy_frac_chg"):
        return "breadth_history"
    if name.startswith("n_owning_lag") or name in ("log_owners", "log_mktcap", "w_mean"):
        return "size_breadth"
    if name.startswith("ret_"):
        return "past_returns"
    return "other"


# ============================================================ 4. MODELS
_KEEP = ["security", "qi", "yq", "target_buy_frac", "pressure_now", "pressure_lag1",
         "buy_frac", "buy_frac_lag1", "dollar_buy_frac", "flow_pct_cap", "n_owning",
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
    lag_col = "pressure_lag1" if "pressure_lag1" in P.columns else "buy_frac_lag1"
    if lag_col in P.columns:
        naive = P.groupby("qi").apply(
            lambda d: d[lag_col].corr(d["target_buy_frac"], method="spearman"))
        rows.append({"metric": f"rank IC (naive: {lag_col}, same information as model)",
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


# ============================================================ 5b. WHAT DRIVES THE FORECAST
def _rank_ic(pred, y):
    s = pd.Series(pred); return float(s.corr(pd.Series(y), method="spearman"))


def _permutation_importance(model, X, y, feats, seed=0, repeats=3):
    """Drop in rank-IC when a column is shuffled. Rank-IC, not MSE, because rank-IC is the
    quantity the study is judged on -- a feature can cut MSE while adding nothing to the sort.
    Shuffling breaks the feature's link to y while preserving its marginal distribution."""
    rng = np.random.default_rng(seed)
    base = _rank_ic(model.predict(X), y)
    out = np.zeros(len(feats))
    Xp = X.copy()
    for j in range(len(feats)):
        col = X[:, j].copy()
        drops = np.empty(repeats)
        for r in range(repeats):
            Xp[:, j] = rng.permutation(col)
            drops[r] = base - _rank_ic(model.predict(Xp), y)
        Xp[:, j] = col
        out[j] = drops.mean()
    return base, out


def explain_model(panel: pd.DataFrame, cfg: Config = None, n_explain: int = 4000,
                  background: int = 200, method: str = "auto", seed: int = 0,
                  verbose: bool = True) -> dict:
    """Which information actually drives the forecast?

    Fits the GBM on the LAST rolling training window and explains its predictions on the
    matching test window -- i.e. explains an out-of-sample model, not one that has seen the
    rows being explained.

    method
      "auto"        use SHAP if importable, otherwise permutation importance
      "shap"        require SHAP (raises if missing)
      "permutation" skip SHAP entirely; fast and dependency-free

    Returns {"per_feature": df, "by_family": df, "method": str, "base_rank_ic": float}.
    `mean_abs` is the magnitude of a feature's contribution; `direction` is the within-sample
    correlation between the feature's value and its own contribution, so +1 means "more of
    this pushes the prediction up" and -1 the reverse.
    """
    cfg = cfg or Config()
    from sklearn.ensemble import HistGradientBoostingRegressor

    feats = feature_list(panel, cfg)
    d = panel[panel["target_buy_frac"].notna()]
    c = int(d.qi.max()) + 1                                  # last window end
    tr = d[(d.qi >= c - cfg.window_q) & (d.qi < c - cfg.test_q)]
    te = d[(d.qi >= c - cfg.test_q) & (d.qi < c)]
    if len(tr) < 500 or len(te) == 0:
        raise RuntimeError("last window too small to explain -- lower window_q/test_q")

    m = HistGradientBoostingRegressor(max_iter=cfg.max_iter, learning_rate=cfg.learning_rate,
                                      max_depth=cfg.max_depth, random_state=cfg.seed)
    m.fit(tr[feats].to_numpy("float32"), tr["target_buy_frac"].to_numpy("float32"))

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(te), min(n_explain, len(te)), replace=False)
    X = te[feats].to_numpy("float32")[idx]
    y = te["target_buy_frac"].to_numpy("float32")[idx]

    used, sv = method, None
    if method in ("auto", "shap"):
        try:
            import shap
            bg = X[rng.choice(len(X), min(background, len(X)), replace=False)]
            # sklearn's HistGradientBoosting is not a TreeExplainer model, so explain the
            # predict function directly; shap picks a permutation-based explainer.
            sv = np.asarray(shap.Explainer(m.predict, bg)(X).values)
            used = "shap"
        except ImportError:
            if method == "shap":
                raise ImportError("method='shap' needs the shap package: pip install shap")
            used = "permutation"
        except Exception as e:                                # explainer failed, not missing
            if method == "shap":
                raise
            print(f"  [warn] SHAP failed ({type(e).__name__}: {e}); using permutation instead")
            used = "permutation"

    if sv is not None:
        mean_abs = np.abs(sv).mean(0)
        direction = np.array([
            float(pd.Series(X[:, j]).corr(pd.Series(sv[:, j]))) for j in range(len(feats))])
        base_ic = _rank_ic(m.predict(X), y)
    else:
        used = "permutation"
        base_ic, mean_abs = _permutation_importance(m, X, y, feats, seed=seed)
        direction = np.array([float(pd.Series(X[:, j]).corr(pd.Series(y)))
                              for j in range(len(feats))])

    per = pd.DataFrame({"feature": feats, "family": [feature_family(f, cfg) for f in feats],
                        "mean_abs": mean_abs, "direction": direction})
    tot = per["mean_abs"].sum()
    per["share_pct"] = 100.0 * per["mean_abs"] / (tot if tot else 1.0)
    per = per.sort_values("mean_abs", ascending=False).reset_index(drop=True)

    fam = (per.groupby("family")
              .agg(share_pct=("share_pct", "sum"), n_features=("feature", "size"),
                   top_feature=("feature", "first"))
              .sort_values("share_pct", ascending=False).reset_index())

    if verbose:
        print(f"\n{'='*70}\nWHAT DRIVES THE FORECAST  [{used}]  "
              f"target={cfg.pressure_measure}\n{'='*70}")
        print(f"  explained on {len(X):,} out-of-sample rows, quarters "
              f"{int(te.qi.min())}-{int(te.qi.max())}, rank-IC there {base_ic:+.4f}\n")
        print(fam.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
        print("\n  top 15 individual features:")
        print(per.head(15).to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        if used == "permutation":
            print("\n  (permutation importance: drop in rank-IC when the column is shuffled."
                  "\n   `pip install shap` for per-observation attributions instead.)")
    return {"per_feature": per, "by_family": fam, "method": used, "base_rank_ic": base_ic}


# ============================================================ 5c. SKIP THE INTERMEDIATE
def predict_returns(panel: pd.DataFrame, cfg: Config = None,
                    horizons=("fwd_1q", "fwd_2q"), models=("gbm", "ridge"),
                    verbose: bool = True) -> pd.DataFrame:
    """Regress the SAME features directly on the security's forward return.

    The buying-pressure study is a two-step bet: features -> buying, buying -> returns. If
    step two is where it breaks, going straight to returns should do no worse. This runs the
    identical rolling split with the return as the target, so the two are comparable.

    Timing is unchanged: every feature is I_q-measurable and fwd_1q spans q -> q+1, so the
    forecast precedes the whole return window. Sorting on the prediction and earning fwd_1q
    is what a manager could actually have done at the q close.

    Two reference rows are added per horizon:
      ret_q          last quarter's return -- pure momentum/reversal, the cheapest rival
      pressure_lag1  lagged buying pressure used raw as a return signal

    Calibration: a cross-sectional return signal with rank-IC above ~0.03 is respectable and
    above ~0.05 is strong. Do not read a 0.30 here the way you would read it for buying
    pressure -- returns are far closer to unforecastable.
    """
    from dataclasses import replace
    cfg = cfg or Config()
    feats = feature_list(panel, cfg)
    rows = []

    def _sort_stats(d, pred_col, ret_col):
        d = d.dropna(subset=[pred_col, ret_col])
        if d.empty:
            return {}
        q = d.groupby("qi")[pred_col].transform(lambda s: _qcut(s, cfg.n_quintiles))
        d = d.assign(Q=q).dropna(subset=["Q"])
        per = d.groupby(["qi", "Q"])[ret_col].mean().unstack()
        hi, lo = per.columns.max(), per.columns.min()
        sp = (per[hi] - per[lo]).dropna()
        ic = d.groupby("qi").apply(
            lambda t: t[pred_col].corr(t[ret_col], method="spearman"))
        return {"rank_IC": ic.mean(), "IC_t": _t(ic),
                "Q1_pct": per[lo].mean() * 100, "Q5_pct": per[hi].mean() * 100,
                "Q5_Q1_pct": sp.mean() * 100, "spread_t": _t(sp),
                "n_quarters": int(sp.size)}

    saved = panel["target_buy_frac"].copy()
    try:
        for h in horizons:
            if h not in panel.columns or panel[h].notna().sum() < 1000:
                continue
            panel["target_buy_frac"] = panel[h]          # reuse the rolling-split machinery
            for m in models:
                P = run_model(panel, replace(cfg, model=m), verbose=False)
                st = _sort_stats(P.rename(columns={"pred_buy_frac": "_p"}), "_p", h)
                if st:
                    rows.append({"target": h, "predictor": f"model:{m}", **st})
            # cheap rivals, scored on exactly the same rows the models were scored on
            base = panel.dropna(subset=[h])
            for ref in ("ret_q", "pressure_lag1"):
                if ref in base.columns:
                    st = _sort_stats(base, ref, h)
                    if st:
                        rows.append({"target": h, "predictor": f"ref:{ref}", **st})
    finally:
        panel["target_buy_frac"] = saved

    out = pd.DataFrame(rows)
    if verbose and len(out):
        print(f"\n{'='*94}\nDIRECT RETURN PREDICTION -- same {len(feats)} features, "
              f"target = the security's forward return\n{'='*94}")
        print(out.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        print("\n  rank_IC   cross-sectional; >0.03 respectable, >0.05 strong FOR RETURNS")
        print("  Q5_Q1_pct quintile spread per quarter, in percent")
        print("  ref:ret_q is momentum/reversal alone -- a model that cannot beat it has")
        print("            added nothing beyond the sign of last quarter's move")
    return out


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
             (t, alpha_sort(P, cfg, t, on=("pressure_now" if "pressure_now" in P.columns
                                           else "buy_frac"))) for t in _TIMING)}
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
