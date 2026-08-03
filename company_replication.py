"""company_replication.py -- Mimicking Finance replication on a single holdings parquet.

No WRDS needed. Produces three sets of results:

  1. PRECISION   accuracy on real positions, AND precision after counting the paper's
                 PADDED template slots (the paper's 0.71 / 0.52 are reproduced only when
                 padded slots are scored).
  2. TABLE X     funds sorted into quintiles on predictability -> cumulative abnormal
                 returns CRET_{0,1..4}.  Paper: Q1 +0.36 / Q5 -0.42 / Q5-Q1 -0.79 (t=-3.05)
                 Reported under TWO holding conventions:
                   actual  -- each quarter uses that quarter's reported holdings
                   frozen  -- weights locked at t, buy-and-hold (isolates stock picking)
  3. TABLE XII   stocks sorted on cross-fund prediction accuracy -> Q1-Q5.
                 Paper: +1.06%/qtr (t=5.74)

Every result is reported under all three timing conventions; `Config.eval_timing` picks
which one the HEADLINE lines use (default "predictive"):
  contemporaneous  acc(t) x t->t+1     OVERLAPS the measurement window; biased benchmark
  predictive       acc(t) x t+1->t+2   no overlap; ignores the 13F filing delay  [DEFAULT]
  tradeable        acc(t) x t+2->t+3   also clears the 45-60 day delay

KEY SWITCH `use_manager_memory`
-------------------------------
Manager-memory features (expanding fund / fund-security trade rates) raise accuracy from
~0.53 to ~0.58, but they quietly redefine "predictable" as "this manager never touches this
position". Low-turnover funds outperform historically, so Table X FLIPS SIGN.

    use_manager_memory=False -> Q5-Q1 = -0.66 (t=-3.20)  ~ paper -0.79 (t=-3.05)   OK
    use_manager_memory=True  -> Q5-Q1 = +0.12 (t=+0.82)  wrong sign                NO

Use False to replicate the paper. True is kept only to demonstrate that raising accuracy
can destroy the economic content.

Verified on real WRDS data (10.9M rows, 12,321 funds, 2010-2024, InvTypeCode 401):
    precision real positions   0.5755 (gbm) / 0.5291 (lstm)
    precision incl. padding    0.7156          [paper 0.71]
    naive     incl. padding    0.5208          [paper 0.52]
    Table X Q5-Q1 tradeable    -0.660 (gbm) / -0.657 (lstm)   [paper -0.79]
                               (measured under "tradeable"; "predictive" is the default here)
    Table XII Q1-Q5 contemp.   +1.213                          [paper +1.06]

Usage
-----
    import company_replication as R
    cfg = R.Config(data_path="your.parquet")
    panel = R.load_and_prepare(cfg)
    res = R.run_config(panel, cfg, "A")
"""
from __future__ import annotations

# Bump when Config gains/loses a field. A stale copy on another machine then fails with a
# clear message instead of "unexpected keyword argument".
__version__ = "2026.08.03.1"
REQUIRED_FIELDS = ("eval_timing", "enforce_sell_feasibility", "feasibility_mode",
                   "feasible_only", "use_manager_memory", "model")

import os
import warnings
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================ CONFIG
@dataclass
class Config:
    data_path: str = "master_all_funds_add_filter_ivy_rank_active_rank.parquet"
    col_map: dict = field(default_factory=lambda: {
        "fund": "fund", "date": "date", "security": "security", "shares": "shares",
        "position_value": "position_value", "market_cap": "market_cap", "isUs": "isUs",
        "quarterly_ret": "quarterly_ret", "past_1q_ret": "past_1q_ret",
        "future_1q_ret": "future_1q_ret", "future_2q_ret": "future_2q_ret",
        "future_3q_ret": "future_3q_ret", "InvTypeCode": "inv_type",
        "future_1q_shares_change_pct": "chg_pct",
        "portfolio_value": "portfolio_value", "weight": "weight", "rank": "rank",
        "n_holdings": "n_holdings", "volume": "volume",
    })
    inv_type_codes: tuple = (401,)     # start with 401
    us_only: bool = True
    max_rank: int = 25                 # keep top-N positions (None = whatever the file has)
    change_band: float = 0.01          # +-1% dead band
    # chg_pct == -100% means the position is ABSENT from the file next quarter (info
    # missing), not a genuine full exit -> drop as a prefilter
    drop_missing_position: bool = True

    # ---- key switch: see module docstring ----
    use_manager_memory: bool = False

    # ---- sell-feasibility constraint (paper 3.3) ----
    # You cannot sell what you do not hold, so the paper zeroes the SELL probability for
    # infeasible securities and renormalises over {hold, buy}. It gives two definitions:
    #   "loose"  (paper's own looser variant, DEFAULT) infeasible = not held in the FINAL
    #            quarter of the input horizon. In this per-position panel every labelled
    #            row IS held at its label quarter, so the mask never binds.
    #   "strict" infeasible = not held in ALL seq_len quarters. Only ~29% of real rows
    #            qualify, so this FORBIDS "sell" on ~71% of rows while sell is ~39% of
    #            labels -- it mechanically wrecks pooled precision (0.57 -> 0.40) and can
    #            flip the sign of the return sorts. Pair it with feasible_only=True, i.e.
    #            evaluate on the same rows you constrained, or do not use it.
    #   "none"   no constraint at all.
    # ---- which timing the HEADLINE numbers use (all three are always computed) ----
    #   "predictive"      acc(t) x t+1->t+2   no overlap; ignores the 13F filing delay
    #   "tradeable"       acc(t) x t+2->t+3   also clears the 45-60 day delay
    #   "contemporaneous" acc(t) x t->t+1     OVERLAPS the measurement window; biased
    eval_timing: str = "predictive"

    enforce_sell_feasibility: bool = True
    feasibility_mode: str = "loose"    # "loose" | "strict" | "none"
    feasible_only: bool = False        # evaluate ONLY on feasible rows (sample restriction)

    # ---- padding: the paper's N-slot template ----
    template_N: int = 75               # template width; padding share follows from the holdings distribution

    # ---- rolling windows (paper Fig 2) ----
    window_q: int = 28
    test_q: int = 8
    step: int = 8
    min_years: int = 7                 # >7 calendar years of history
    min_holdings: int = 10             # >=10 securities/quarter (auto-capped at max_rank)

    # ---- model ----
    # "gbm"        gradient boosting; sequence flattened into y_lag1..4 columns. Fastest.
    # "lstm"       weight-shared per-position sequence LSTM: one sample = one position's
    #              last seq_len quarters, [T, F]
    # "panel_lstm" the paper's own architecture: one sample = one fund-quarter's whole
    #              cross-section, [T, N, F] -> LSTM(N*F -> numcell) -> [N, 3], N = max_rank
    model: str = "gbm"
    max_iter: int = 250                # gbm
    learning_rate: float = 0.08
    max_depth: int = 7
    n_max_train: int = 1_500_000       # cap on train rows per window (runtime control)
    seed: int = 0
    # ---- neural-net only ----
    seq_len: int = 8                   # input sequence length (paper T=8)
    hidden: int = 64
    dropout: float = 0.25
    lr: float = 3e-3
    max_epochs: int = 25
    patience: int = 5
    batch: int = 8192
    device: str = "auto"               # "auto" | "cuda" | "cpu"
    # None = train on everything, no subsampling. Sequences are assembled lazily from
    # indices (see _build_sequences), so 10M rows costs ~2 GB, not ~7 GB.
    # Only set a cap (e.g. 300_000) on a CPU-only box to trade accuracy for time.
    lstm_max_train: int = None         # cap on train sequences per window (None = all)
    lstm_max_rows: int = None          # subsample the panel before building sequences (None = no)

    @property
    def base_features(self) -> List[str]:
        return ["weight", "w_lag1", "dw", "rank", "rank_pct", "log_posval", "log_pv",
                "log_mktcap", "quarterly_ret", "past_1q_ret",
                "pdsh", "pdsh_sign", "pdsh_lag1", "sh_lag1", "sh_lag2", "sh_lag3",
                "peer_buy", "peer_sell", "peer_hold", "n_holdings",
                "n_funds", "log_inst_own", "sum_abs_aw", "own_rank",
                "y_lag1", "y_lag2", "y_lag3", "y_lag4", "pos_age", "d_rank", "w_drift",
                # Volume: whether a position CAN be traded, not merely whether the manager wants to.
                # Absent automatically if the file has no volume column.
                "log_volume", "vol_rank", "pos_to_vol", "d_log_vol", "amihud"]

    @property
    def memory_features(self) -> List[str]:
        return ["fund_buy_rate", "fund_hold_rate", "fund_sell_rate",
                "fs_hold_rate", "fs_buy_rate", "fs_sell_rate", "fs_n_obs",
                "sec_buy_rate", "sec_hold_rate", "sec_sell_rate"]

    @property
    def features(self) -> List[str]:
        return self.base_features + (self.memory_features if self.use_manager_memory else [])


def check_version(verbose=True):
    """Verify this copy has the fields the notebook expects. Call it first thing."""
    import os, time
    missing = [f for f in REQUIRED_FIELDS if f not in Config.__dataclass_fields__]
    path = os.path.abspath(__file__)
    mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
    if verbose:
        print(f"company_replication {__version__}  |  {path}  |  modified {mtime}")
    if missing:
        raise ImportError(
            f"This copy of company_replication.py is STALE -- missing {missing}.\n"
            f"  file    : {path}\n"
            f"  modified: {mtime}\n"
            "Copy the current mimicking_pipeline/ folder across, then RESTART THE KERNEL "
            "(%autoreload does not pick up new dataclass fields).")
    if verbose:
        print(f"  all {len(REQUIRED_FIELDS)} expected Config fields present")
    return __version__


# ============================================================ HELPERS
def _t(x, lags=0):
    """t-stat of the mean. lags>0 applies a Newey-West serial-correlation correction.

    CRET_{0,h} windows OVERLAP (t and t+1 share h-1 quarters), so the quarterly series is
    autocorrelated and a plain OLS t is SYSTEMATICALLY OVERSTATED. Use lags=h-1.
    """
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 2 or x.std() == 0:
        return np.nan
    if lags <= 0:
        return x.mean() / (x.std(ddof=1) / np.sqrt(n))
    e = x - x.mean()
    var = float(e @ e) / n
    for L in range(1, min(lags, n - 1) + 1):     # Bartlett kernel
        cov = float(e[L:] @ e[:-L]) / n
        var += 2.0 * (1.0 - L / (lags + 1.0)) * cov
    if var <= 0:
        return np.nan
    return x.mean() / np.sqrt(var / n)


def _q5(s, n=5):
    return (pd.qcut(s.rank(method="first"), n, labels=False, duplicates="drop") + 1
            if s.nunique() >= n else pd.Series(np.nan, index=s.index))


def _chg_scale(chg, cfg):
    nz = chg[chg.abs() > 1e-9].abs()
    med = float(nz.median()) if len(nz) else np.nan
    return "percent" if (np.isfinite(med) and med > 1.5) else "fraction"


# ============================================================ DATA + FEATURES
def load_and_prepare(cfg: Config) -> pd.DataFrame:
    inv = {v: k for k, v in cfg.col_map.items()}
    want = ["fund", "date", "security", "shares", "position_value", "market_cap",
            "quarterly_ret", "past_1q_ret", "future_1q_ret", "future_2q_ret",
            "future_3q_ret", "chg_pct", "inv_type", "portfolio_value", "weight",
            "rank", "n_holdings", "isUs", "volume"]
    want_raw = [inv[c] for c in want if c in inv]
    try:
        import pyarrow.parquet as pq
        avail = set(pq.ParquetFile(cfg.data_path).schema.names)
        df = pd.read_parquet(cfg.data_path, columns=[c for c in want_raw if c in avail])
    except Exception:
        df = pd.read_parquet(cfg.data_path)
    df = df.rename(columns={inv[c]: c for c in cfg.col_map.values() if inv[c] in df.columns})

    df["date"] = pd.to_datetime(df["date"])
    df["yq"] = df["date"].dt.to_period("Q")
    if cfg.us_only and "isUs" in df.columns:
        df = df[df["isUs"].astype(bool)]
    if cfg.inv_type_codes is not None and "inv_type" in df.columns:
        codes = {str(c) for c in cfg.inv_type_codes}
        df = df[df["inv_type"].astype(str).isin(codes)]
    df = df.sort_values("date").drop_duplicates(["fund", "yq", "security"], keep="last")
    if "rank" in df.columns and cfg.max_rank:
        df = df[df["rank"] <= cfg.max_rank]
    df = df.drop(columns=[c for c in ("date", "isUs") if c in df.columns])

    F32 = "float32"
    for c in ["shares", "position_value", "market_cap", "quarterly_ret", "past_1q_ret",
              "future_1q_ret", "future_2q_ret", "future_3q_ret", "chg_pct",
              "portfolio_value", "weight", "rank", "n_holdings", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(F32) if c in df.columns \
            else np.array(np.nan, dtype=F32)

    # ---- PREFILTER: drop the -100% sentinel (position info missing next quarter),
    # ---- before any feature / weight / peer aggregation touches it ----
    if cfg.drop_missing_position and df["chg_pct"].notna().any():
        chg = df["chg_pct"].astype("float64")
        sc = _chg_scale(chg, cfg)
        frac = chg / 100.0 if sc == "percent" else chg
        bad = (frac + 1.0).abs() < 1e-6
        print(f"[data] prefilter: dropped {int(bad.sum()):,} rows ({bad.mean():.1%}) with chg_pct=-100% "
              f"(units: {sc}) -- position info missing next quarter, not a real sell")
        df = df[~bad]

    df = df.sort_values(["fund", "security", "yq"]).reset_index(drop=True)
    keys = ["fund", "security"]
    g = df.groupby(keys, observed=True)

    # ---- TARGET: strictly the next quarter ----
    if df["chg_pct"].notna().any():
        chg = df["chg_pct"].astype("float64")
        sc = _chg_scale(chg, cfg)
        dsh = chg / 100.0 if sc == "percent" else chg
        print(f"[data] target from future_1q_shares_change_pct (units: {sc})")
    else:
        sh_n = g["shares"].shift(-1); q_n = g["yq"].shift(-1)
        sh_n = sh_n.where(q_n == df["yq"] + 1)                   # exact t+1 only
        dsh = (sh_n - df["shares"]) / (df["shares"].abs() + 1.0)
        print("[data] no chg_pct column -> target from shares[t+1] (exact t+1; dropped on a gap)")
    # keep the raw fractional share change: needed for VALUE-WEIGHTED turnover, which a
    # {-1,0,1} label cannot express (a 1% trim and a full exit both map to -1).
    df["dsh"] = pd.to_numeric(dsh, errors="coerce").astype(F32)
    df["Y"] = np.select([dsh <= -cfg.change_band, dsh >= cfg.change_band],
                        [-1.0, 1.0], default=0.0).astype(F32)
    df.loc[pd.isna(dsh), "Y"] = np.nan
    # Realised returns: EVALUATION ONLY, never features. Carry all three horizons so one
    # training run yields all three timing conventions.
    df["fwd_1q"] = df["future_1q_ret"]      # t   -> t+1
    df["fwd_2q"] = df["future_2q_ret"]      # t+1 -> t+2
    df["fwd_3q"] = df["future_3q_ret"]      # t+2 -> t+3
    bal = pd.Series(df["Y"]).value_counts(normalize=True)
    print(f"[data] class balance  sell {bal.get(-1.,0):.3f} | hold {bal.get(0.,0):.3f} | "
          f"buy {bal.get(1.,0):.3f}  (labelled {int(df['Y'].notna().sum()):,})")

    # ---- FEATURES: levels forward-filled (past only); never fillna(0) ----
    for k in (1, 2, 3):
        df[f"sh_lag{k}"] = g["shares"].shift(k).fillna(df["shares"]).astype(F32)
    df["w_lag1"] = g["weight"].shift(1).fillna(df["weight"]).astype(F32)
    df["dw"] = (df["weight"] - df["w_lag1"]).astype(F32)
    df["pdsh"] = ((df["shares"] - df["sh_lag1"]) / (df["sh_lag1"].abs() + 1.0)).astype(F32)
    df["pdsh_sign"] = np.sign(df["pdsh"]).fillna(0.0).astype(F32)
    df["pdsh_lag1"] = df.groupby(keys, observed=True)["pdsh"].shift(1).fillna(0.0).astype(F32)
    df["log_posval"] = np.log(df["position_value"].abs() + 1.0).astype(F32)
    df["log_pv"] = np.log(df["portfolio_value"].abs() + 1.0).astype(F32)
    df["log_mktcap"] = np.log(df["market_cap"].abs() + 1.0).astype(F32)
    df["rank_pct"] = (df["rank"] / df["n_holdings"].where(df["n_holdings"] > 0)).astype(F32)

    # sample filters (paper 3.1)
    mh = min(cfg.min_holdings, cfg.max_rank or cfg.min_holdings)
    cnt = df.groupby(["fund", "yq"], observed=True)["security"].transform("size")
    df = df[cnt >= mh]
    nq = df.groupby("fund", observed=True)["yq"].transform("nunique")
    df = df[nq >= cfg.min_years * 4]

    # integer quarter index
    qs = pd.PeriodIndex(sorted(df["yq"].unique()), freq="Q")
    df["qi"] = df["yq"].map({q: i for i, q in enumerate(qs)}).astype("int32")

    # ---- peer activity rates, lagged one EXACT quarter ----
    lab = df.dropna(subset=["Y"])
    rate = lab.groupby("yq")["Y"].agg(peer_buy=lambda s: (s > 0).mean(),
                                      peer_sell=lambda s: (s < 0).mean(),
                                      peer_hold=lambda s: (s == 0).mean())
    allq = pd.PeriodIndex(sorted(df["yq"].unique()), freq="Q")
    rate = rate.reindex(allq).ffill()
    prev = df["yq"] - 1
    for c in ("peer_buy", "peer_sell", "peer_hold"):
        df[c] = prev.map(rate[c]).astype(F32)

    # ---- security level: cross-fund ownership (quarter-t holdings only) ----
    gsq = df.groupby(["security", "yq"], observed=True)
    df["n_funds"] = gsq["fund"].transform("size").astype(F32)
    own = df["position_value"] / df["market_cap"].where(df["market_cap"] > 0)
    df["own_frac"] = own.astype(F32)
    inst = df.groupby(["security", "yq"], observed=True)["own_frac"].transform("sum")
    df["log_inst_own"] = np.log(inst.abs() + 1e-6).astype(F32)
    df["own_rank"] = inst.groupby(df["yq"]).rank(pct=True).astype(F32)
    gf = df.groupby(["fund", "yq"], observed=True)
    aw = (df["position_value"] / gf["position_value"].transform("sum")
          - df["market_cap"] / gf["market_cap"].transform("sum")).abs()
    df["sum_abs_aw"] = aw.groupby([df["security"].to_numpy(),
                                   df["yq"].to_numpy()]).transform("sum").astype(F32)

    # ---- VOLUME: the liquidity constraint ----
    # "Wants to sell" and "can sell" are different. A position worth 20 days of volume can
    # only be unwound gradually, so next quarter is more likely a partial trim than an exit;
    # liquid positions are traded far more freely. All use quarter-t or earlier info only.
    # Skipped entirely when the file has no volume column.
    if "volume" in df.columns and df["volume"].notna().any():
        vol = df["volume"].astype("float64").where(df["volume"] > 0)
        df["log_volume"] = np.log(vol.fillna(0) + 1.0).astype(F32)
        # per-quarter cross-sectional percentile: immune to whether volume is shares or dollars
        df["vol_rank"] = vol.groupby(df["yq"]).rank(pct=True).astype(F32)
        # position size in units of volume -- a days-to-liquidate proxy; higher = harder to exit
        df["pos_to_vol"] = (df["shares"] / vol).clip(0, 50).astype(F32)
        # Amihud-style illiquidity: return magnitude per unit of volume (higher = less liquid)
        df["amihud"] = (df["quarterly_ret"].abs() / (vol / 1e6)).clip(0, 100).astype(F32)
        gv = df.groupby(["security", "yq"], observed=True)["log_volume"].first()
        sv = df[["security", "yq"]].copy()
        prev = pd.MultiIndex.from_arrays([sv["security"].to_numpy(),
                                          (sv["yq"] - 1).to_numpy()])
        df["d_log_vol"] = (df["log_volume"] - gv.reindex(prev).to_numpy()).astype(F32)
        n_vol = int(df["log_volume"].notna().sum())
        print(f"[data] volume features added (coverage {n_vol/len(df):.1%}): "
              f"log_volume / vol_rank / pos_to_vol / d_log_vol / amihud")
    else:
        print("[data] no volume column -> volume features skipped")

    # ---- sequence / lifecycle (exact-quarter aligned) ----
    df = df.sort_values(["fund", "security", "qi"]).reset_index(drop=True)
    g = df.groupby(keys, observed=True, sort=False)
    qi = df["qi"].to_numpy()
    for k in (1, 2, 3, 4):
        v, q = g["Y"].shift(k), g["qi"].shift(k)
        df[f"y_lag{k}"] = v.where(q == qi - k).astype(F32)
    newblk = (g["qi"].diff() != 1).to_numpy()
    blk = np.cumsum(newblk)
    df["pos_age"] = df.groupby([df["fund"].to_numpy(), df["security"].to_numpy(), blk],
                               sort=False).cumcount().astype(F32)
    rk, rq = g["rank"].shift(1), g["qi"].shift(1)
    df["d_rank"] = (df["rank"] - rk.where(rq == qi - 1)).astype(F32)
    df["w_drift"] = ((df["weight"] - df["w_lag1"]) /
                     (df["w_lag1"].abs() + 1e-6)).clip(-5, 5).astype(F32)

    # ---- manager memory: expanding rates over STRICTLY PAST quarters ----
    _y = df["Y"].to_numpy(); _lab = ~np.isnan(_y)
    df["_lab"] = _lab.astype(F32)
    for w, v in (("buy", (_y == 1) & _lab), ("hold", (_y == 0) & _lab),
                 ("sell", (_y == -1) & _lab)):
        df[f"_{w}"] = v.astype(F32)

    def past_rates(kk, prefix):
        gg = df.groupby(kk, observed=True, sort=False)
        cnt = gg["_lab"].cumsum().to_numpy() - df["_lab"].to_numpy()
        for w in ("buy", "hold", "sell"):
            cs = gg[f"_{w}"].cumsum().to_numpy() - df[f"_{w}"].to_numpy()
            df[f"{prefix}_{w}_rate"] = np.where(cnt > 0, cs / np.maximum(cnt, 1.0),
                                                np.nan).astype(F32)
        df[f"{prefix}_n_obs"] = gg.cumcount().to_numpy().astype(F32)

    past_rates(["fund"], "fund"); past_rates(keys, "fs"); past_rates(["security"], "sec")
    df.drop(columns=["_lab", "_buy", "_hold", "_sell"], inplace=True)

    df = add_feasible(df, cfg.seq_len)
    print(f"[data] panel {len(df):,} rows | {df.fund.nunique():,} funds | "
          f"{df.security.nunique():,} securities | {df.qi.max()+1} quarters")
    return df


def add_feasible(df: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    """feasible = this (fund, security) was present in EVERY quarter qi-seq_len+1 .. qi.

    Vectorised: an exact-quarter presence check per lag, ANDed together. A gap anywhere in
    the window makes the position infeasible.
    """
    df = df.sort_values(["fund", "security", "qi"]).reset_index(drop=True)
    g = df.groupby(["fund", "security"], observed=True, sort=False)
    qi = df["qi"].to_numpy()
    ok = np.ones(len(df), dtype=bool)
    for k in range(1, seq_len):
        qk = g["qi"].shift(k).to_numpy(dtype="float64", na_value=np.nan)
        ok &= (qk == qi - k)
    df["feasible"] = ok
    print(f"[data] feasible (held all {seq_len} quarters): {ok.mean():.1%} of rows")
    return df


def _apply_sell_feasibility(proba, feasible, classes, cfg):
    """Zero the SELL probability where a sell is infeasible, then renormalise.

    proba    [n, 3] class probabilities
    feasible [n]    bool, from the chosen feasibility_mode
    classes  the model's class order, so we zero the right column
    Returns predicted labels in {-1, 0, 1}.
    """
    mode = getattr(cfg, "feasibility_mode", "loose")
    if not cfg.enforce_sell_feasibility or mode == "none":
        return classes[proba.argmax(1)]
    if mode == "loose":
        # every labelled row is held at its own label quarter -> the loose mask never binds
        return classes[proba.argmax(1)]
    sell_col = int(np.where(classes == -1)[0][0]) if (classes == -1).any() else None
    if sell_col is None:
        return classes[proba.argmax(1)]
    p = proba.copy()
    p[~feasible, sell_col] = 0.0
    s = p.sum(1, keepdims=True); s[s == 0] = 1.0
    return classes[(p / s).argmax(1)]


# ============================================================ MODEL (rolling, OOS)
_KEEP = ["fund", "security", "qi", "Y", "dsh", "weight", "w_lag1", "n_holdings",
         "feasible", "fwd_1q", "fwd_2q", "fwd_3q"]


def _pick_device(cfg):
    """Pick the device, checking the GPU's compute capability against this torch build.
    A new card (RTX 50 series = sm_120) on an older torch (cu124) crashes inside cuDNN;
    falling back to CPU with an actionable message beats crashing."""
    import torch
    if cfg.device != "auto":
        return cfg.device
    if not torch.cuda.is_available():
        return "cpu"
    try:
        cap = torch.cuda.get_device_capability(0)
        sm = f"sm_{cap[0]}{cap[1]}"
        if sm not in torch.cuda.get_arch_list():
            print(f"  [warn] GPU capability {sm} unsupported by torch {torch.__version__} "
                  f"(supports {torch.cuda.get_arch_list()[-3:]}...) -> falling back to CPU.\n"
                  f"         For GPU install a matching build, e.g. for RTX 50 series:\n"
                  f"         pip install --pre torch --index-url "
                  f"https://download.pytorch.org/whl/nightly/cu128")
            return "cpu"
    except Exception:
        return "cpu"
    return "cuda"


def _build_sequences(sub: pd.DataFrame, feats, seq_len):
    """Build sequence INDICES rather than materialising the [N, T, F] tensor.

    Materialising costs 10M rows x 8 steps x 40 features x 4 bytes ~ 13 GB, and 26 GB once
    a standardised copy is made. Instead we keep only:
        Feat      [n_rows, F]  float32  -- the feature matrix (each row stored once)
        hist_idx  [N, T]       int32    -- which Feat row each sample's step t reads
    A batch assembles [batch, T, F] on the fly via Feat[hist_idx[bi]], cutting memory to
    ~(n_rows x F x 4) + (N x T x 4): about 1.6 GB + 0.2 GB for 10M rows.

    When the position is absent in a quarter that step gets mask=0 (the LSTM masks it out)
    and its index is arbitrary. The label sits at the last step.
    Returns (Feat, hist_idx, mask, y, meta).
    """
    sub = sub.sort_values(["fund", "security", "qi"]).reset_index(drop=True)
    g = sub.groupby(["fund", "security"], observed=True, sort=False)
    valid = sub["Y"].notna().to_numpy()
    N = int(valid.sum())
    if N == 0:
        return None
    F = len(feats)
    qi = sub["qi"].to_numpy()
    Feat = np.nan_to_num(sub[feats].to_numpy(dtype="float32", na_value=np.nan),
                         nan=0.0, posinf=0.0, neginf=0.0)
    row = pd.Series(np.arange(len(sub), dtype="int64"), index=sub.index)

    hist_idx = np.zeros((N, seq_len), dtype=np.int32)
    M = np.zeros((N, seq_len), dtype=np.float32)
    for k in range(seq_len):
        step = seq_len - 1 - k                      # k=0 is the current quarter -> last step
        rk = row.groupby([sub["fund"], sub["security"]], observed=True,
                         sort=False).shift(k).to_numpy(dtype="float64", na_value=np.nan)
        qk = g["qi"].shift(k).to_numpy(dtype="float64", na_value=np.nan)
        present = (qk == qi - k) & ~np.isnan(rk)    # must be the exact t-k quarter
        pv = present[valid]
        M[:, step] = pv.astype(np.float32)
        hist_idx[:, step] = np.where(pv, np.nan_to_num(rk[valid], nan=0.0), 0).astype(np.int32)
    y = (sub["Y"].to_numpy()[valid] + 1).astype(np.int64)      # {-1,0,1} -> {0,1,2}
    meta = sub.loc[valid, [c for c in _KEEP if c in sub.columns]].reset_index(drop=True)
    return Feat, hist_idx, M, y, meta


def _fit_lstm(Feat, hist_idx, M, y, tr, te, cfg):
    """Train a weight-shared sequence LSTM on tr; return predictions on te ({-1,0,1}).
    Sequences are assembled per batch via Feat[hist_idx[bi]]; [N,T,F] is never materialised."""
    import torch, torch.nn as nn
    dev = _pick_device(cfg)
    torch.manual_seed(cfg.seed)
    F = Feat.shape[1]
    tr_i_all = np.where(tr)[0]
    if len(tr_i_all) < 100:
        return None, dev
    # standardisation stats estimated directly on the Feat rows the train samples touch
    used = np.unique(hist_idx[tr_i_all][M[tr_i_all] > 0])
    if used.size < 50:
        return None, dev
    if used.size > 500_000:                     # sampling is fine for mu/sd; it does not reduce the training set
        used = np.random.default_rng(cfg.seed).choice(used, 500_000, replace=False)
    mu = Feat[used].mean(0).astype(np.float32)
    sd = (Feat[used].std(0) + 1e-6).astype(np.float32)

    def _batch(bi):
        """Assemble + standardise + re-mask padding on the fly. Peak memory = one batch."""
        xb = (Feat[hist_idx[bi]] - mu) / sd      # [b, T, F]
        xb *= M[bi][..., None]
        return torch.from_numpy(xb), torch.from_numpy(M[bi])

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(F, cfg.hidden, batch_first=True)
            self.drop = nn.Dropout(cfg.dropout)
            self.head = nn.Linear(cfg.hidden, 3)

        def forward(self, x, m):
            o, _ = self.lstm(x * m.unsqueeze(-1))
            return self.head(self.drop(o[:, -1, :]))

    model = Net().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = nn.CrossEntropyLoss()
    yt = torch.from_numpy(y)
    idx = tr_i_all.copy()
    rng = np.random.default_rng(cfg.seed); rng.shuffle(idx)
    # cap training size on CPU, otherwise a single window can take hours
    if cfg.lstm_max_train and len(idx) > cfg.lstm_max_train:
        idx = idx[:cfg.lstm_max_train]
    nval = max(1, int(0.15 * len(idx)))
    val_i, trn_i = idx[:nval], idx[nval:]
    best, best_state, bad = 1e9, None, 0
    for _ in range(cfg.max_epochs):
        model.train(); rng.shuffle(trn_i)
        for b in range(0, len(trn_i), cfg.batch):
            bi = trn_i[b:b + cfg.batch]
            xb, mb = _batch(bi)
            opt.zero_grad()
            lossf(model(xb.to(dev), mb.to(dev)), yt[bi].to(dev)).backward()
            opt.step()
        model.eval(); tot = n = 0
        with torch.inference_mode():
            for b in range(0, len(val_i), cfg.batch):
                bi = val_i[b:b + cfg.batch]
                xb, mb = _batch(bi)
                l = lossf(model(xb.to(dev), mb.to(dev)), yt[bi].to(dev))
                tot += float(l) * len(bi); n += len(bi)
        vl = tot / max(n, 1)
        if vl < best - 1e-4:
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
    preds = []
    with torch.inference_mode():
        for b in range(0, len(te_i), cfg.batch):
            bi = te_i[b:b + cfg.batch]
            xb, mb = _batch(bi)
            preds.append(torch.softmax(model(xb.to(dev), mb.to(dev)), 1).cpu().numpy())
    return (np.concatenate(preds) if preds else None), dev


def _build_panel_tensor(sub: pd.DataFrame, feats, seq_len, N):
    """The paper's own sample: one (fund, label quarter t) = one [T, N, F] tensor.

    Column j = the security ranked j-th by position value in that fund at t; the SAME
    security is then tracked back over t-7..t (paper 3.1's template + 3.3's "N distinct
    security identifiers"). When the fund holds fewer than N names the extra columns are
    padding (mask=0) -- exactly the slots that inflate precision.

    Indices only, again: hist [S, T, N] int32 + mask [S, T, N].
    Returns (Feat, hist, mask, ylab, meta); ylab[s, j] in {0,1,2}, -1 = no label.
    """
    sub = sub.sort_values(["fund", "qi", "rank"]).reset_index(drop=True)
    row = np.arange(len(sub), dtype=np.int64)
    # (fund, security, qi) -> row number, used to track one security back through time
    key = pd.MultiIndex.from_arrays([sub["fund"].to_numpy(), sub["security"].to_numpy(),
                                     sub["qi"].to_numpy()])
    lookup = pd.Series(row, index=key)
    # top-N per fund-quarter (already sorted by rank)
    head = sub.groupby(["fund", "qi"], observed=True).head(N)
    slot = head.groupby(["fund", "qi"], observed=True).cumcount().to_numpy()
    fq = head[["fund", "qi"]].drop_duplicates().reset_index(drop=True)
    fq_id = pd.Series(np.arange(len(fq)), index=pd.MultiIndex.from_arrays(
        [fq["fund"].to_numpy(), fq["qi"].to_numpy()]))
    sid = fq_id.reindex(pd.MultiIndex.from_arrays(
        [head["fund"].to_numpy(), head["qi"].to_numpy()])).to_numpy()
    S = len(fq)
    hist = np.zeros((S, seq_len, N), dtype=np.int32)
    mask = np.zeros((S, seq_len, N), dtype=np.float32)
    ylab = np.full((S, N), -1, dtype=np.int64)
    hrow = head.index.to_numpy()
    yv = sub["Y"].to_numpy()
    ok = ~np.isnan(yv[hrow])
    ylab[sid[ok], slot[ok]] = (yv[hrow][ok] + 1).astype(np.int64)
    for k in range(seq_len):
        step = seq_len - 1 - k
        idx = pd.MultiIndex.from_arrays([head["fund"].to_numpy(),
                                         head["security"].to_numpy(),
                                         head["qi"].to_numpy() - k])
        r = lookup.reindex(idx).to_numpy()          # row of the SAME security at t-k
        pres = ~pd.isna(r)
        hist[sid[pres], step, slot[pres]] = r[pres].astype(np.int32)
        mask[sid[pres], step, slot[pres]] = 1.0
    Feat = np.nan_to_num(sub[feats].to_numpy(dtype="float32", na_value=np.nan),
                         nan=0.0, posinf=0.0, neginf=0.0)
    meta_cols = [c for c in _KEEP if c in sub.columns]
    meta = head[meta_cols].reset_index(drop=True)
    meta["_sid"] = sid; meta["_slot"] = slot
    return Feat, hist, mask, ylab, meta, fq


def _fit_panel_lstm(Feat, hist, mask, ylab, tr, te, cfg):
    """Paper architecture: [T, N*F] -> LSTM -> Linear -> [N, 3]. Padded slots are skipped
    via ignore_index=-1."""
    import torch, torch.nn as nn
    dev = _pick_device(cfg)
    torch.manual_seed(cfg.seed)
    S, T, N = hist.shape
    F = Feat.shape[1]
    tr_i = np.where(tr)[0]
    if len(tr_i) < 20:
        return None, dev
    used = np.unique(hist[tr_i][mask[tr_i] > 0])
    if used.size < 50:
        return None, dev
    if used.size > 500_000:
        used = np.random.default_rng(cfg.seed).choice(used, 500_000, replace=False)
    mu = Feat[used].mean(0).astype(np.float32)
    sd = (Feat[used].std(0) + 1e-6).astype(np.float32)

    def _batch(bi):
        xb = (Feat[hist[bi]] - mu) / sd                 # [b, T, N, F]
        xb *= mask[bi][..., None]
        b = xb.shape[0]
        step_mask = (mask[bi].sum(axis=2) > 0).astype(np.float32)   # a fully padded quarter -> masked out of recurrence
        return (torch.from_numpy(xb.reshape(b, T, N * F)),
                torch.from_numpy(step_mask))

    numcell = int(np.clip(N, 16, cfg.hidden))

    class PanelNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_drop = nn.Dropout(cfg.dropout)
            self.lstm = nn.LSTM(N * F, numcell, batch_first=True)
            self.out_drop = nn.Dropout(cfg.dropout)
            self.head = nn.Linear(numcell, N * 3)

        def forward(self, x, sm):
            o, _ = self.lstm(self.in_drop(x * sm.unsqueeze(-1)))
            return self.head(self.out_drop(o[:, -1, :])).view(-1, N, 3)

    model = PanelNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-1)
    yt = torch.from_numpy(ylab)
    idx = tr_i.copy()
    rng = np.random.default_rng(cfg.seed); rng.shuffle(idx)
    if cfg.lstm_max_train and len(idx) > cfg.lstm_max_train:
        idx = idx[:cfg.lstm_max_train]
    nval = max(1, int(0.15 * len(idx)))
    val_i, trn_i = idx[:nval], idx[nval:]
    bs = max(32, cfg.batch // max(N, 1))                # each sample is a whole cross-section, so batches must be smaller
    best, best_state, bad = 1e9, None, 0
    for _ in range(cfg.max_epochs):
        model.train(); rng.shuffle(trn_i)
        for b in range(0, len(trn_i), bs):
            bi = trn_i[b:b + bs]
            xb, sm = _batch(bi)
            opt.zero_grad()
            lg = model(xb.to(dev), sm.to(dev))
            lossf(lg.reshape(-1, 3), yt[bi].reshape(-1).to(dev)).backward()
            opt.step()
        model.eval(); tot = n = 0
        with torch.inference_mode():
            for b in range(0, len(val_i), bs):
                bi = val_i[b:b + bs]
                xb, sm = _batch(bi)
                lg = model(xb.to(dev), sm.to(dev))
                l = lossf(lg.reshape(-1, 3), yt[bi].reshape(-1).to(dev))
                tot += float(l) * len(bi); n += len(bi)
        vl = tot / max(n, 1)
        if vl < best - 1e-4:
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
    P = np.zeros((len(te_i), N), dtype=np.float32)
    Pp = np.zeros((len(te_i), N, 3), dtype=np.float32)
    with torch.inference_mode():
        for b in range(0, len(te_i), bs):
            bi = te_i[b:b + bs]
            xb, sm = _batch(bi)
            pr = torch.softmax(model(xb.to(dev), sm.to(dev)), dim=2).cpu().numpy()
            P[b:b + len(bi)] = pr.argmax(2) - 1          # slot-level labels
            Pp[b:b + len(bi)] = pr
    return (te_i, P, Pp), dev


def run_model(panel: pd.DataFrame, cfg: Config, verbose=True) -> pd.DataFrame:
    """Rolling-window out-of-sample predictions. cfg.model selects the architecture."""
    feats = [f for f in cfg.features if f in panel.columns]
    d = panel[panel["Y"].notna()] if cfg.model == "gbm" else panel
    out = []

    if cfg.model == "lstm":
        sub = panel
        if cfg.lstm_max_rows and int(panel["Y"].notna().sum()) > cfg.lstm_max_rows:
            keep = panel[panel["Y"].notna()].sample(cfg.lstm_max_rows,
                                                    random_state=cfg.seed).index
            sub = panel.loc[panel.index.isin(keep) | panel["Y"].isna()]
            print(f"  [lstm] panel subsampled -> {cfg.lstm_max_rows:,} labelled rows")
        seq = _build_sequences(sub, feats, cfg.seq_len)
        if seq is None:
            raise RuntimeError("no usable samples")
        Feat, hist_idx, M, y, meta = seq
        qi = meta["qi"].to_numpy()
        if verbose:
            naive_gb = len(y) * cfg.seq_len * len(feats) * 4 / 1e9
            used_gb = (Feat.nbytes + hist_idx.nbytes + M.nbytes) / 1e9
            print(f"  [lstm] {len(y):,} sequences x T={cfg.seq_len} x F={Feat.shape[1]}"
                  f"  memory {used_gb:.2f} GB (lazy indices; materialising would need {naive_gb:.1f} GB)")

    if cfg.model == "panel_lstm":
        N = cfg.max_rank or int(panel["rank"].max())
        Feat, hist, pmask, ylab, pmeta, fq = _build_panel_tensor(panel, feats, cfg.seq_len, N)
        sqi = fq["qi"].to_numpy()
        if verbose:
            naive = hist.shape[0] * cfg.seq_len * N * len(feats) * 4 / 1e9
            used = (Feat.nbytes + hist.nbytes + pmask.nbytes) / 1e9
            print(f"  [panel_lstm] {hist.shape[0]:,} fund-quarter samples x T={cfg.seq_len} "
                  f"x N={N} x F={len(feats)}  memory {used:.2f} GB (materialising needs {naive:.1f} GB)")
            pad = 1.0 - float(pmask[:, -1, :].mean())
            print(f"  [panel_lstm] padding share at the last step = {pad:.1%}"
                  f" -- these are the slots that inflate precision")

    for c in range(cfg.window_q, int(panel.qi.max()) + 2, cfg.step):
        if cfg.model == "panel_lstm":
            tr = (sqi >= c - cfg.window_q) & (sqi < c - cfg.test_q)
            te = (sqi >= c - cfg.test_q) & (sqi < c)
            if tr.sum() < 50 or te.sum() == 0:
                continue
            got, dev = _fit_panel_lstm(Feat, hist, pmask, ylab, tr, te, cfg)
            if got is None:
                continue
            te_i, Pmat, Pproba = got
            sid2pos = {s: i for i, s in enumerate(te_i)}
            sel = pmeta[pmeta["_sid"].isin(sid2pos)].copy()
            rows_ = [sid2pos[s] for s in sel["_sid"]]; cols_ = sel["_slot"].to_numpy()
            feas = (sel["feasible"].to_numpy() if "feasible" in sel.columns
                    else np.ones(len(sel), bool))
            sel["y_pred"] = _apply_sell_feasibility(
                Pproba[rows_, cols_], feas, np.array([-1.0, 0.0, 1.0]), cfg)
            p = sel[sel["Y"].notna()].drop(columns=["_sid", "_slot"]).reset_index(drop=True)
            if len(p) == 0:
                continue
        elif cfg.model == "lstm":
            tr = (qi >= c - cfg.window_q) & (qi < c - cfg.test_q)
            te = (qi >= c - cfg.test_q) & (qi < c)
            if tr.sum() < 5000 or te.sum() == 0:
                continue
            proba, dev = _fit_lstm(Feat, hist_idx, M, y, tr, te, cfg)
            if proba is None:
                continue
            p = meta.iloc[np.where(te)[0]].copy()
            feas = (p["feasible"].to_numpy() if "feasible" in p.columns
                    else np.ones(len(p), bool))
            p["y_pred"] = _apply_sell_feasibility(proba, feas,
                                                  np.array([-1.0, 0.0, 1.0]), cfg)
        else:
            tr = d[(d.qi >= c - cfg.window_q) & (d.qi < c - cfg.test_q)]
            te = d[(d.qi >= c - cfg.test_q) & (d.qi < c)]
            if len(tr) < 5000 or len(te) == 0:
                continue
            from sklearn.ensemble import HistGradientBoostingClassifier
            trs = (tr.sample(cfg.n_max_train, random_state=cfg.seed)
                   if len(tr) > cfg.n_max_train else tr)
            clf = HistGradientBoostingClassifier(max_iter=cfg.max_iter,
                                                 learning_rate=cfg.learning_rate,
                                                 max_depth=cfg.max_depth,
                                                 random_state=cfg.seed)
            clf.fit(trs[feats].to_numpy("float32"), trs["Y"].to_numpy())
            p = te[[k for k in _KEEP if k in te.columns]].copy()
            proba = clf.predict_proba(te[feats].to_numpy("float32"))
            feas = (te["feasible"].to_numpy() if "feasible" in te.columns
                    else np.ones(len(te), bool))
            p["y_pred"] = _apply_sell_feasibility(proba, feas, clf.classes_, cfg)
            dev = "cpu"
        out.append(p)
        if verbose:
            print(f"  window {c:>3}  n_test={len(p):>9,}  "
                  f"acc={float((p.y_pred==p.Y).mean()):.4f}  [{cfg.model}/{dev}]")
    return pd.concat(out, ignore_index=True)


# ============================================================ 1. PRECISION (+padding)
def precision_report(P: pd.DataFrame, cfg: Config):
    y, pr = P["Y"].to_numpy(), P["y_pred"].to_numpy()
    acc = float((pr == y).mean())
    hold = float((y == 0).mean())
    naive_pool = float(pd.Series(y).value_counts(normalize=True).max())
    pf = P.assign(c=(pr == y)).groupby("fund", observed=True)["c"].mean()
    rows = [{"basis": "real positions", "N": "-", "padding": 0.0,
             "precision": acc, "naive": naive_pool}]
    fq = P.drop_duplicates(["fund", "qi"])[["fund", "qi", "n_holdings"]]
    for N in (50, cfg.template_N, 100):
        slots = np.minimum(fq["n_holdings"].to_numpy(), N)
        p = float((N - slots).sum() / (N * len(fq)))
        rows.append({"basis": f"N={N} template (incl. padding)", "N": N, "padding": p,
                     "precision": p + (1 - p) * acc, "naive": p + (1 - p) * hold})
    t = pd.DataFrame(rows)
    return t, {"accuracy": acc, "naive_pooled": naive_pool,
               "precision_per_fund": float(pf.mean()), "hold_share": hold}


# ============================================================ 2. Table X
_TIMING = {"contemporaneous": 0, "predictive": 1, "tradeable": 2}


def table_x_frozen(P: pd.DataFrame, cfg: Config, timing=None, hmax=4,
                   sort_var="precision"):
    """CRET_{0,h} with holdings FROZEN at t (buy and hold), not following the fund's
    actual rebalancing.

    Difference from table_x:
      table_x        every quarter uses that quarter's reported holdings -> includes the
                     contribution of subsequent rebalancing
      table_x_frozen weights locked at t, each stock compounds on its own -> measures only
                     the t-dated portfolio itself

    The gap between them IS the rebalancing contribution. If the paper's "less predictable
    managers outperform" survives under frozen holdings, predictability is linked to STOCK
    PICKING; if it holds only under actual rebalancing, it is linked to TRADING.

        CRET_frozen(i,t,h) = Σ_s w_s(t)·[Π_{j=0}^{h-1}(1+r_s(t+start+j))] / Σ_s w_s(t) − 1

    A stock missing any of the h quarterly returns drops out entirely (never compound
    across a gap).
    """
    timing = timing or getattr(cfg, "eval_timing", "predictive")
    start = _TIMING[timing]
    P = P.copy()
    P["correct"] = (P.y_pred == P.Y).astype(float)
    sret = P.groupby(["security", "qi"], observed=True)["fwd_1q"].mean()   # r(q -> q+1)
    sec = P["security"].to_numpy(); qi = P["qi"].to_numpy(); w = P["weight"].to_numpy()
    prec = fund_sort_var(P, sort_var)      # sorting variable (default: precision)
    rows_out = []
    cum = {}
    for h in range(1, hmax + 1):
        comp = np.ones(len(P)); good = np.ones(len(P), bool)
        for j in range(h):
            r = sret.reindex(pd.MultiIndex.from_arrays([sec, qi + start + j])).to_numpy()
            m = pd.isna(r); good &= ~m
            comp = comp * (1.0 + np.where(m, 0.0, r))
        cum[h] = np.where(good, comp - 1.0, np.nan)
    base = pd.DataFrame({"fund": P["fund"].to_numpy(), "qi": qi, "w": w})
    fq = None
    for h in range(1, hmax + 1):
        d = base.assign(c=cum[h])
        d = d[d["c"].notna() & d["w"].notna()]
        g = d.assign(wc=d["w"] * d["c"]).groupby(["fund", "qi"], observed=True).agg(
            wsum=("w", "sum"), wc=("wc", "sum")).reset_index()
        g[f"cabn{h}"] = g["wc"] / g["wsum"].where(g["wsum"] > 0)
        g[f"cabn{h}"] = g[f"cabn{h}"] - g.groupby("qi")[f"cabn{h}"].transform("mean")
        g = g[["fund", "qi", f"cabn{h}"]]
        fq = g if fq is None else fq.merge(g, on=["fund", "qi"], how="outer")
    fq = fq.merge(prec, on=["fund", "qi"], how="left")
    fq["Q"] = fq.groupby("qi")["prec"].transform(_q5)
    fq = fq.dropna(subset=["Q"])
    for q in (1, 2, 3, 4, 5):
        r = {"quintile": f"Q{q}"}
        for h in range(1, hmax + 1):
            s = fq[fq.Q == q].groupby("qi")[f"cabn{h}"].mean()
            r[f"CRET_0_{h}"] = s.mean() * 100
            r[f"t{h}"] = _t(s, lags=h - 1); r[f"t{h}_ols"] = _t(s)
        rows_out.append(r)
    r = {"quintile": "Q5-Q1"}
    for h in range(1, hmax + 1):
        d = (fq[fq.Q == 5].groupby("qi")[f"cabn{h}"].mean()
             - fq[fq.Q == 1].groupby("qi")[f"cabn{h}"].mean()).dropna()
        r[f"CRET_0_{h}"] = d.mean() * 100
        r[f"t{h}"] = _t(d, lags=h - 1); r[f"t{h}_ols"] = _t(d)
    rows_out.append(r)
    return pd.DataFrame(rows_out)


def table_x(P: pd.DataFrame, cfg: Config, timing=None, hmax=4,
            sort_var="precision"):
    """Funds in precision quintiles -> CRET_{0,1..hmax}. `start` is set by the timing.

    This is the ACTUAL-REBALANCING basis: each quarter uses that quarter's reported
    holdings. For buy-and-hold see table_x_frozen.
    """
    timing = timing or getattr(cfg, "eval_timing", "predictive")
    start = _TIMING[timing]
    P = P.copy()
    P["correct"] = (P.y_pred == P.Y).astype(float)
    # Fund quarterly return = holdings return weighted by BEGINNING-OF-PERIOD weights:
    #     fund_ret(t) = Σ_s w_{s,t} · r_{s, t->t+1} / Σ_s w_{s,t}
    # The weight must be dated at the START of the return window. fwd_1q spans t->t+1, so it
    # pairs with weight at t; pairing it with w_lag1 (t-1) would be off by one quarter.
    # The denominator sums only positions WITH a return; otherwise missing-return positions
    # keep their weight and dilute the fund return toward zero.
    ok = P["fwd_1q"].notna() & P["weight"].notna()
    P["wc"] = P["weight"] * P["fwd_1q"]
    fq = P[ok].groupby(["fund", "qi"], observed=True).agg(
        wsum=("weight", "sum"), wc=("wc", "sum")).reset_index()
    fq["fund_ret"] = fq["wc"] / fq["wsum"].where(fq["wsum"] > 0)
    fq = fq.merge(fund_sort_var(P, sort_var), on=["fund", "qi"], how="left")
    fq["abn"] = fq["fund_ret"] - fq.groupby("qi")["fund_ret"].transform("mean")
    lut = fq.set_index(["fund", "qi"])["abn"]
    for h in range(1, hmax + 1):
        tot = np.zeros(len(fq)); good = np.ones(len(fq), bool)
        for j in range(h):
            i2 = pd.MultiIndex.from_arrays([fq["fund"].to_numpy(),
                                            fq["qi"].to_numpy() + start + j])
            a = lut.reindex(i2).to_numpy()
            m = pd.isna(a); good &= ~m; tot = tot + np.where(m, 0.0, a)
        fq[f"cabn{h}"] = np.where(good, tot, np.nan)
    fq["Q"] = fq.groupby("qi")["prec"].transform(_q5)
    fq = fq.dropna(subset=["Q"])
    # h-quarter cumulation -> windows overlap h-1 periods -> Newey-West with lags=h-1
    rows = []
    for q in (1, 2, 3, 4, 5):
        r = {"quintile": f"Q{q}"}
        for h in range(1, hmax + 1):
            s = fq[fq.Q == q].groupby("qi")[f"cabn{h}"].mean()
            r[f"CRET_0_{h}"] = s.mean() * 100
            r[f"t{h}"] = _t(s, lags=h - 1)
            r[f"t{h}_ols"] = _t(s)
        rows.append(r)
    r = {"quintile": "Q5-Q1"}
    for h in range(1, hmax + 1):
        d = (fq[fq.Q == 5].groupby("qi")[f"cabn{h}"].mean()
             - fq[fq.Q == 1].groupby("qi")[f"cabn{h}"].mean()).dropna()
        r[f"CRET_0_{h}"] = d.mean() * 100
        r[f"t{h}"] = _t(d, lags=h - 1)
        r[f"t{h}_ols"] = _t(d)
    rows.append(r)
    return pd.DataFrame(rows)


# ============================================================ 3. Table XII
def table_xii(P: pd.DataFrame, cfg: Config, timing=None):
    timing = timing or getattr(cfg, "eval_timing", "predictive")
    col = {"contemporaneous": "fwd_1q", "predictive": "fwd_2q", "tradeable": "fwd_3q"}[timing]
    if col not in P.columns or P[col].notna().sum() == 0:
        return pd.DataFrame([{"quintile": "n/a", "mean_qret": np.nan, "t": np.nan}])
    P = P.copy()
    P["correct"] = (P.y_pred == P.Y).astype(float)
    stk = P.groupby(["security", "qi"], observed=True).agg(
        acc=("correct", "mean"), fwd=(col, "mean")).reset_index().dropna(subset=["fwd"])
    stk["Q"] = stk.groupby("qi")["acc"].transform(_q5)
    stk = stk.dropna(subset=["Q"])
    per = stk.groupby(["qi", "Q"])["fwd"].mean().unstack()
    rows = [{"quintile": f"Q{int(q)}", "mean_qret": per[q].mean() * 100, "t": _t(per[q])}
            for q in sorted(per.columns)]
    ls = (per[per.columns.min()] - per[per.columns.max()]).dropna()
    rows.append({"quintile": "Q1-Q5", "mean_qret": ls.mean() * 100, "t": _t(ls)})
    return pd.DataFrame(rows)


def predictability_vs_turnover(P: pd.DataFrame, cfg: Config = None):
    """Is "predictable" just a restatement of "does not trade"?

    Both quantities are measured over the SAME t -> t+1 window, per (fund, quarter):

        prec(i,t)     = share of fund i's t -> t+1 trades predicted correctly
        turnover(i,t) = share of fund i's positions that actually moved, |Y| = 1

    A manager who never trades is trivially predictable ("always hold"), so a strongly
    NEGATIVE relation means precision is largely a turnover proxy rather than a measure of
    behavioural regularity. That matters because low-turnover funds outperform historically:
    if precision is really turnover in disguise, Table X is picking up the turnover premium,
    not predictability. This is exactly the channel that flips Table X's sign when the
    manager-memory features are switched on.

    Correlations are reported two ways:
      pooled         across all fund-quarters (contaminated by time trends in turnover)
      within-quarter both variables demeaned by quarter first -> pure cross-sectional link

    Returns (summary dict, quintile table).
    """
    d = P.copy()
    d["correct"] = (d.y_pred == d.Y).astype(float)
    d["traded"] = (d.Y != 0).astype(float)
    d["pred_hold"] = (d.y_pred == 0).astype(float)
    fq = d.groupby(["fund", "qi"], observed=True).agg(
        prec=("correct", "mean"), turnover=("traded", "mean"),
        pred_hold=("pred_hold", "mean"), n=("Y", "size")).reset_index()
    fq = fq[fq["n"] >= 5]
    dp = fq["prec"] - fq.groupby("qi")["prec"].transform("mean")
    dt = fq["turnover"] - fq.groupby("qi")["turnover"].transform("mean")
    m = {
        "n_fund_quarters": int(len(fq)),
        "corr_pooled": float(fq["prec"].corr(fq["turnover"])),
        "corr_spearman": float(fq["prec"].corr(fq["turnover"], method="spearman")),
        "corr_within_quarter": float(dp.corr(dt)),
        "mean_turnover": float(fq["turnover"].mean()),
        "mean_prec": float(fq["prec"].mean()),
        # how much of precision is explained by turnover alone, cross-sectionally
        "r2_within_quarter": float(dp.corr(dt) ** 2),
    }
    fq["Q"] = fq.groupby("qi")["prec"].transform(_q5)
    tbl = (fq.dropna(subset=["Q"]).groupby("Q")
           .agg(mean_prec=("prec", "mean"), mean_turnover=("turnover", "mean"),
                share_predicted_hold=("pred_hold", "mean"),
                n_fund_quarters=("prec", "size")).reset_index()
           .rename(columns={"Q": "prec_quintile"}))
    tbl["prec_quintile"] = tbl["prec_quintile"].astype(int).map(lambda q: f"Q{q}")
    return m, tbl


def fund_sort_var(P: pd.DataFrame, sort_var="precision") -> pd.Series:
    """Fund-quarter sorting variable, indexed by (fund, qi).

    precision   share of fund i's t -> t+1 trades predicted correctly  [the paper's sort]
    turnover    share of fund i's positions that actually moved, |Y| = 1, over the SAME
                t -> t+1 window. The decisive control: if sorting on turnover reproduces
                the precision-sorted Table X, "predictability" adds nothing beyond "this
                fund barely trades" and the result is the low-turnover premium.
                Sign convention: turnover is the OPPOSITE of predictability, so a
                turnover-sorted Q5-Q1 should flip sign relative to a precision-sorted one.
    sum_abs_aw  mean collective active tilt of the names the fund holds -- "is this fund
                crowded into consensus positions?"
    <any column on P>  averaged over the fund's positions that quarter.
    """
    d = P.copy()
    if sort_var == "precision":
        d["_v"] = (d.y_pred == d.Y).astype(float)
    elif sort_var == "turnover":
        # VALUE-WEIGHTED turnover -- the standard definition:
        #   sum_s |d shares_s| * p_s / sum_s shares_s * p_s  =  sum_s w_s(t) * |dsh_s|
        # since |d shares_s| / shares_s = |dsh_s|, the fractional share change.
        # A count-based version (share of positions that moved at all) is NOT turnover:
        # it treats a 1% trim and a full exit identically and has almost no cross-sectional
        # variation, which mechanically kills any correlation.
        if "dsh" not in d.columns:
            raise ValueError("value-weighted turnover needs the `dsh` column "
                             "(fractional share change) on the predictions")
        wsum = d.groupby(["fund", "qi"], observed=True)["weight"].transform("sum")
        d["_v"] = d["weight"] * d["dsh"].abs() / wsum.where(wsum > 0)
        g = d.groupby(["fund", "qi"], observed=True).agg(v=("_v", "sum"), n=("Y", "size"))
        return g.loc[g["n"] >= 5, "v"].rename("prec")
    elif sort_var == "turnover_count":          # the old count-based measure, for contrast
        d["_v"] = (d.Y != 0).astype(float)
    elif sort_var in d.columns:
        d["_v"] = pd.to_numeric(d[sort_var], errors="coerce")
    else:
        raise ValueError(f"unknown sort_var {sort_var!r}; not a column on the predictions")
    g = d.groupby(["fund", "qi"], observed=True).agg(v=("_v", "mean"), n=("Y", "size"))
    return g.loc[g["n"] >= 5, "v"].rename("prec")      # named `prec` for the table builders


# ============================================================ ONE CONFIGURATION
def _restrict(P, cfg):
    """feasible_only: evaluate on the SAME rows the sell-constraint was applied to.
    Never constrain predictions on one row set and then score a different one."""
    if getattr(cfg, "feasible_only", False) and "feasible" in P.columns:
        return P[P["feasible"]].copy()
    return P


def run_config(panel: pd.DataFrame, cfg: Config, tag: str = "", verbose=True) -> dict:
    """Run one configuration; return every result for it (precision + Table X actual and
    frozen + Table XII, each under all three timings). The panel is passed in so it is
    built once and shared across configurations."""
    print(f"{'='*74}\n{tag or cfg.model}  |  model={cfg.model}  "
          f"manager_memory={cfg.use_manager_memory}  ({len(cfg.features)} features)\n{'='*74}")
    P = run_model(panel, cfg, verbose=verbose)
    P = _restrict(P, cfg)
    prec_tbl, prec_m = precision_report(P, cfg)
    r = {"tag": tag, "cfg": cfg, "preds": P,
         "precision_table": prec_tbl, "precision": prec_m,
         # Table X on BOTH holding conventions: `tableX` follows the fund's actual
         # rebalancing, `tableX_frozen` locks weights at t (buy and hold). Their gap is
         # the contribution of subsequent trading.
         "tableX": {t: table_x(P, cfg, t) for t in _TIMING},
         "tableX_frozen": {t: table_x_frozen(P, cfg, t) for t in _TIMING},
         "tableXII": {t: table_xii(P, cfg, t) for t in _TIMING}}
    r["turnover"], r["turnover_table"] = predictability_vs_turnover(P, cfg)
    hz = getattr(cfg, "eval_timing", "predictive")
    tf = r["tableX_frozen"][hz].iloc[-1]
    tx = r["tableX"][hz].iloc[-1]
    t12 = r["tableXII"]["contemporaneous"].iloc[-1]
    print(f"\n  accuracy (real positions) = {prec_m['accuracy']:.4f}   "
          f"naive = {prec_m['naive_pooled']:.4f}")
    print(f"  incl. padding (N={cfg.template_N}) precision = "
          f"{prec_tbl.iloc[2]['precision']:.4f}   naive = {prec_tbl.iloc[2]['naive']:.4f}"
          f"   [paper 0.71 / 0.52]")
    print(f"  Table X  Q5-Q1 ({hz}, actual) = {tx.CRET_0_4:+.3f}% (t={tx.t4:+.2f})"
          f"   [paper -0.79, t=-3.05]")
    print(f"  Table X  Q5-Q1 ({hz}, frozen) = {tf.CRET_0_4:+.3f}% (t={tf.t4:+.2f})"
          f"   [gap vs actual = rebalancing contribution]")
    print(f"  Table XII Q1-Q5 (contemporaneous)= {t12.mean_qret:+.3f}% (t={t12.t:+.2f})"
          f"   [paper +1.06, t=5.74]")
    tv = r["turnover"]
    print(f"  corr(precision, turnover) within-quarter = {tv['corr_within_quarter']:+.3f}"
          f"  (R2 {tv['r2_within_quarter']:.3f})"
          f"   [strongly negative => precision is largely a turnover proxy]")
    return r


def free(results: dict = None, keep_tables=True):
    """Free caches after a configuration so back-to-back runs do not OOM.

    On a 10M-row panel each configuration's `preds` can be several hundred MB and they
    accumulate across configurations. keep_tables=True drops only `preds` (the bulk) and
    keeps every result table.
    """
    import gc
    if results:
        for tag, r in results.items():
            if isinstance(r, dict) and keep_tables and "preds" in r:
                r["preds"] = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    except Exception:
        pass
    try:                      # report current usage so you can judge whether another run fits
        import resource
        mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if os.name != "posix" else 1024)
        print(f"[free] cleared. peak process memory ~ {mb:.0f} MB")
    except Exception:
        print("[free] cleared.")


def summarize(results: dict) -> pd.DataFrame:
    """Combine several run_config results into one comparison table."""
    rows = []
    for tag, r in results.items():
        if not isinstance(r, dict) or "tableX" not in r:
            continue
        c = r["cfg"]; row = {"config": tag, "model": c.model,
                             "manager_memory": c.use_manager_memory,
                             "accuracy": round(r["precision"]["accuracy"], 4),
                             "precision_with_padding": round(r["precision_table"].iloc[2]["precision"], 4)}
        for tm in _TIMING:
            x = r["tableX"][tm].iloc[-1]
            row[f"X_Q5-Q1_{tm[:5]}"] = round(x.CRET_0_4, 3)
            row[f"X_t_{tm[:5]}"] = round(x.t4, 2)
        for tm in _TIMING:                       # frozen (buy-and-hold) basis
            x = r["tableX_frozen"][tm].iloc[-1]
            row[f"Xfz_Q5-Q1_{tm[:5]}"] = round(x.CRET_0_4, 3)
            row[f"Xfz_t_{tm[:5]}"] = round(x.t4, 2)
        for tm in _TIMING:
            x = r["tableXII"][tm].iloc[-1]
            row[f"XII_Q1-Q5_{tm[:5]}"] = round(x.mean_qret, 3)
            row[f"XII_t_{tm[:5]}"] = round(x.t, 2)
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        hz = results[next(iter(results))]["cfg"]
        hz = getattr(hz, "eval_timing", "predictive")[:5]
        out["headline_timing"] = hz
        out["X_sign_matches_paper"] = np.where(out[f"X_Q5-Q1_{hz}"] < 0, "yes", "no")
    return out


# ============================================================ ONE-SHOT ABLATION
def run_ablation(cfg: Config = None, verbose=True):
    """Run both use_manager_memory = False / True and return all results for comparison."""
    cfg = cfg or Config()
    out = {}
    panel = load_and_prepare(cfg)          # the panel is switch-independent; build it once
    for use_mem in (False, True):
        c = Config(**{**cfg.__dict__, "use_manager_memory": use_mem})
        tag = "with_memory" if use_mem else "no_memory"
        print(f"\n{'='*72}\n{tag}  ({len(c.features)} features)\n{'='*72}")
        P = run_model(panel, c, verbose=verbose)
        prec_tbl, prec_m = precision_report(P, c)
        out[tag] = {
            "preds": P, "precision_table": prec_tbl, "precision": prec_m,
            "tableX": {t: table_x(P, c, t) for t in _TIMING},
            "tableXII": {t: table_xii(P, c, t) for t in _TIMING},
        }
        print(f"  accuracy={prec_m['accuracy']:.4f}  naive={prec_m['naive_pooled']:.4f}")
    out["_panel"] = panel
    return out


def show(res: dict):
    """Print the full comparison."""
    print("\n" + "=" * 78)
    print("1) PRECISION: real positions vs incl. padding  [paper 0.71 / naive 0.52]")
    print("=" * 78)
    for tag in ("no_memory", "with_memory"):
        print(f"\n--- {tag} ---")
        t = res[tag]["precision_table"].copy()
        t["padding"] = (t["padding"] * 100).round(1)
        print(t.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("2) TABLE X (CRET_0,4, %)  paper: Q1 +0.36  Q5 -0.42  Q5-Q1 -0.79 (t=-3.05)")
    print("=" * 78)
    for tag in ("no_memory", "with_memory"):
        print(f"\n--- {tag} ---")
        for tm in ("predictive", "tradeable", "contemporaneous"):
            r = res[tag]["tableX"][tm]
            v = "  ".join(f"{x:+6.3f}" for x in r["CRET_0_4"][:5])
            sp = r.iloc[-1]
            print(f"  {tm:<16} {v}   Q5-Q1 {sp.CRET_0_4:+6.3f} (t={sp.t4:+5.2f})")

    print("\n" + "=" * 78)
    print("3) TABLE XII (Q1-Q5, %/qtr)  paper: +1.06 (t=5.74)")
    print("=" * 78)
    for tag in ("no_memory", "with_memory"):
        print(f"\n--- {tag} ---")
        for tm in ("predictive", "tradeable", "contemporaneous"):
            sp = res[tag]["tableXII"][tm].iloc[-1]
            print(f"  {tm:<16} Q1-Q5 {sp.mean_qret:+6.3f} (t={sp.t:+5.2f})")

    print("\n" + "=" * 78)
    print("Read: no_memory is the paper-replication basis. with_memory scores higher but")
    print("      FLIPS Table X, because fs_hold_rate turns \"predictable\" into \"this")
    print("      manager never touches this position\".")
    print("=" * 78)


if __name__ == "__main__":
    res = run_ablation(Config())
    show(res)
