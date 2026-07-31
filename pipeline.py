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
    data_path: str = ("manager_holdings/master_batches_return_filtered/"
                      "panel_holdings_All_Funds_add_filter_ivy_rank_active_rank.parquet")
    out_dir: str = "outputs"
    # column mapping: {your_column: internal_name}. Adjust only if your names differ.
    col_map: dict = field(default_factory=lambda: {
        "fund": "fund", "date": "date", "security": "security", "shares": "shares",
        "position_value": "position_value", "shares_change": "shares_change",
        "close": "close", "market_cap": "market_cap", "volume": "volume",
        "isUs": "isUs", "quarterly_ret": "quarterly_ret", "past_1q_ret": "past_1q_ret",
        "future_1q_ret": "future_1q_ret", "future_2q_ret": "future_2q_ret",
        "future_3q_ret": "future_3q_ret", "InvTypeCode": "inv_type",
        "future_1q_shares_change_pct": "chg_pct",   # t -> t+1 % change in shares (target)
        "portfolio_value": "portfolio_value",
        "weight": "weight", "rank": "rank", "n_holdings": "n_holdings",
    })
    # Keep only these InvTypeCode values (e.g. [401]). None = all types.
    # When more than one type survives, InvTypeCode also becomes the CATEGORY used for
    # peer activity rates and for benchmark-adjusting fund returns (the paper uses the
    # Morningstar style category for both).
    inv_type_codes: tuple = None
    # ---- evaluation timing --------------------------------------------------
    # accuracy(t) = "did we predict the t->t+1 trade right?" -> needs shares[t+1], so it
    # is OBSERVABLE at t+1, and only PUBLIC ~45-60 days later (filing delay), i.e. partway
    # through t+2.
    #   "tradeable"       : accuracy(t), return over t+2->t+3 (`future_3q_ret`)
    #                       -> also clears the filing delay. STRICTEST / truly tradeable.
    #   "predictive"      : accuracy(t), return over t+1->t+2 (`future_2q_ret`)
    #                       -> no window overlap, but ignores the filing delay. DEFAULT.
    #   "contemporaneous" : accuracy(t), return over t->t+1   (`future_1q_ret`)
    #                       -> sort variable OVERLAPS its own return window (look-ahead).
    #                          Biased; keep only as the comparison benchmark.
    #   "lagged"          : accuracy(t-1), return over t->t+1 -> clean but staler.
    # Training is independent of this -> compare_eval_modes() gives all four from a
    # SINGLE training run.
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
    change_band: float = 0.01   # +-1% dead-band around zero share change (in FRACTION units)
    # Units of `future_1q_shares_change_pct`: "auto" detects fraction (0.05) vs percent
    # (5.0) and normalises to a fraction so change_band is always a fraction. Or force
    # "fraction" / "percent".
    target_pct_scale: str = "auto"
    # Which target to build. "auto"/"chg_pct" = use future_1q_shares_change_pct if present
    # (captures full exits). "shares" = force the old shares[t+1]-shares[t] label (exact
    # t+1, drops exits). Use it to A/B whether the target change moved a result.
    target_source: str = "auto"
    # PREFILTER: `future_1q_shares_change_pct == -100%` (fraction -1.0) does NOT mean a real
    # full exit in this dataset -- it means the position simply is not in the file next
    # quarter, i.e. the position info is MISSING. Labelling those "sell" fabricates the
    # class, so drop them BEFORE any feature / weight / peer-rate / active-weight
    # aggregation, so the removed rows never influence the panel at all.
    # Matched on the NORMALISED fraction, so it is correct whether the column is stored in
    # fraction (-1.0) or percent (-100.0) units.
    drop_missing_position: bool = True
    missing_position_value: float = -1.0    # fraction units: -1.0 == -100%
    # Strict next-quarter target. The chg_pct column is TRUSTED to be a t->t+1 change; a
    # diagnostic always reports whether that holds. With this True, ENFORCE it: any position
    # absent at the exact next quarter is treated as a full exit -> "sell" (a gap-spanning
    # chg_pct is overridden). Sound only when the panel is NOT rank-filtered (a rank drop
    # also looks "absent"); leave False if you filter by max_rank and trust your column.
    target_strict_next: bool = False
    # How to fill a lag feature (sh_lag*, w_lag1, pdsh, pdsh_lag1) when the prior quarter
    # is missing (a gap). "ffill" (default) = forward-fill the last-known level across the
    # gap (a level should persist, not reset); a gap then produces no fake trade. "zero" =
    # old behaviour: exact-quarter -> NaN -> 0 in the tensor, which makes a gap look like a
    # jump from an empty position and distorts the panel. Use "zero" only to A/B the old result.
    lag_fill: str = "ffill"
    # Feasibility = a security held in ALL seq_len quarters, so "sell" is a feasible action.
    # TWO INDEPENDENT switches (they used to be coupled, which was wrong):
    #
    #   enforce_sell_feasibility (default TRUE): _train_predict zeroes the sell probability
    #     for infeasible rows and renormalises over {hold, buy}. This is a PREDICTION
    #     constraint applied on the FULL sample -- it never drops any row. Empirically very
    #     useful (a security you have not held for the whole window cannot be "sold"), so it
    #     is on by default and independent of any sample restriction.
    #
    #   feasible_only (default FALSE): an EVALUATION sample restriction only. False -> every
    #     table (precision, Table X prec, Table XI, Table XII) uses ALL labelled rows. True
    #     -> those tables restrict to feasible rows. It does NOT touch the sell-zeroing.
    enforce_sell_feasibility: bool = True
    feasible_only: bool = False
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
    # ---- per-bucket models (train + test a SEPARATE model per bucket) ----
    # bucket_by: column to split samples on before training. None = one model (default).
    #   "sum_abs_aw" -> collective active tilt on the security that quarter,
    #   sum_abs_aw = sum_funds |w_fund - w_capweight|, aggregated per (security, quarter)
    #   (mirrors wrds_pull/own_disp). Computed here if absent; look-ahead-safe (uses only
    #   quarter-t holdings). Any other PRESENT column is used as-is.
    # Samples are bucketed by PER-QUARTER quantile of bucket_by (n_buckets), then each
    # bucket gets its own LSTM trained + tested only on that bucket -- never pooled training.
    bucket_by: str = None
    n_buckets: int = 5
    # Minimum train sequences for ONE (window, bucket) cell. None (default) = adaptive:
    # the per-model minimum divided by n_buckets, because bucketing splits each window's
    # training rows n_buckets ways. A FIXED value that ignores n_buckets (e.g. 500 while
    # min_seq_per_fund=120) silently starves every bucket -> "no predictions produced".
    min_train_bucket: int = None
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


# ============================================================ helpers
def _shift_exact(df, keys, col, k, qcol="yq"):
    """Shift `col` by EXACTLY k quarters within `keys`, using the calendar quarter
    `qcol`. A plain groupby().shift(k) returns the previous *row*, which after a gap
    can be 3 or 20 quarters back -- silently leaking a stale value. Rows whose true
    t-k quarter is missing are set to NaN."""
    if k == 0:
        return df[col]
    g = df.groupby(keys, observed=True)
    v = g[col].shift(k)
    q = g[qcol].shift(k)
    return v.where(q == df[qcol] - k)


def _at_prev_quarter(df, keys, series_by_key_q, k=1):
    """Look up `series_by_key_q` (indexed by keys+quarter) at (keys, yq - k) for every
    row of df. Exact-quarter, gap-safe -- no row shifting."""
    arrs = [df[c].to_numpy() for c in keys] + [(df["yq"] - k).to_numpy()]
    idx = pd.MultiIndex.from_arrays(arrs)
    return series_by_key_q.reindex(idx).to_numpy()


def _dense_ffill(agg, keycol, allq):
    """Reindex an aggregate (indexed by keycol + 'yq', observed quarters only) onto the
    dense keycol x allq grid and forward-fill within keycol. Then a (keycol, yq-1) lookup
    returns the LAST-OBSERVED value <= yq-1 (forward-fill across a gap, past info only)
    instead of NaN -> 0. `agg` may be a Series or DataFrame; keeps the same type."""
    keys_u = agg.index.get_level_values(keycol).unique()
    idx = pd.MultiIndex.from_product([keys_u, allq], names=[keycol, "yq"])
    return agg.reindex(idx).groupby(level=keycol).ffill()


def _chg_scale(chg, cfg):
    """Units of the raw chg_pct column: 'fraction' (0.05) vs 'percent' (5.0).
    Detected from the median absolute nonzero value unless cfg pins it."""
    scale = getattr(cfg, "target_pct_scale", "auto")
    if scale == "auto":
        nz = chg[chg.abs() > 1e-9].abs()
        med = float(nz.median()) if len(nz) else np.nan
        scale = "percent" if (np.isfinite(med) and med > 1.5) else "fraction"
    return scale


def _sum_abs_aw(df):
    """sum_abs_aw per (security, yq): collective active tilt on a name that quarter
    (mirrors wrds_pull/own_disp). active_weight of fund i in security s =
    w_fund - w_capweight = pv/sum_fund_pv - mc/sum_fund_mc (each normalised over the
    fund's own book, so it sums to zero within a fund-quarter). sum_abs_aw = sum over
    holders of |active_weight|. Uses only quarter-t holdings -> observable at t, no
    look-ahead. Returns a Series aligned to df.index."""
    pv = df["position_value"].astype("float64")
    mc = df["market_cap"].astype("float64")
    gf = df.groupby(["fund", "yq"], observed=True)
    aw = (pv / gf["position_value"].transform("sum")
          - mc / gf["market_cap"].transform("sum")).abs()
    saw = aw.groupby([df["security"].to_numpy(), df["yq"].to_numpy()]).transform("sum")
    return pd.Series(saw.to_numpy(), index=df.index)


def _per_quarter_bucket(val, yq, n):
    """Look-ahead-safe bucket id 0..n-1: within EACH quarter's cross-section only, rank
    `val` and cut into n equal-count groups. Each quarter is bucketed on its own data
    (known at t), so no future information leaks. NaN val -> NaN bucket (excluded)."""
    r = val.groupby(yq).rank(pct=True, method="first")     # per-quarter percentile in (0,1]
    b = np.ceil(r * n) - 1.0
    return b.clip(lower=0, upper=n - 1)


# ============================================================ DATA + FEATURES
def load_and_prepare(cfg: Config) -> pd.DataFrame:
    """Load the panel, build the target and all (lagged) features. Returns a tidy
    fund-security-quarter frame with columns: fund, security, yq, qi, <features>,
    Y (target), fwd_qret, weight, rank."""
    inv = {v: k for k, v in cfg.col_map.items()}          # internal -> your column
    # (a) READ ONLY the raw columns we actually use (huge saving on a 20M-row file)
    want = ["fund", "date", "security", "shares", "position_value", "market_cap",
            "quarterly_ret", "past_1q_ret", "future_1q_ret", "future_2q_ret",
            "future_3q_ret", "chg_pct", "inv_type", "portfolio_value", "weight", "rank",
            "n_holdings", "isUs"]
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
    # InvTypeCode filter (compare as strings so 401 and "401" both work)
    if cfg.inv_type_codes is not None and "inv_type" in df.columns:
        codes = {str(c) for c in cfg.inv_type_codes}
        before = len(df)
        df = df[df["inv_type"].astype(str).isin(codes)]
        print(f"[data] InvTypeCode filter {sorted(codes)}: {before:,} -> {len(df):,} rows")
        if len(df) == 0:
            raise ValueError(f"no rows for InvTypeCode {sorted(codes)}")
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
               "future_1q_ret", "future_2q_ret", "future_3q_ret", "chg_pct", "portfolio_value",
               "weight", "rank", "n_holdings"]
    for c in raw_num:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(F32) if c in df.columns \
            else np.array(np.nan, dtype=F32)
    # ---- PREFILTER: next-quarter position info MISSING (chg_pct == -100%) --------------
    # Not a genuine full exit: the position is simply absent from the file next quarter, so
    # a "sell" label would be fabricated. Dropped HERE -- before sorting, lags, weights,
    # peer rates and active-weight aggregation -- so these rows never touch the panel.
    if cfg.drop_missing_position and "chg_pct" in df.columns and df["chg_pct"].notna().any():
        _chg = pd.to_numeric(df["chg_pct"], errors="coerce").astype("float64")
        _scale = _chg_scale(_chg, cfg)
        _frac = _chg / 100.0 if _scale == "percent" else _chg
        bad = ((_frac - cfg.missing_position_value).abs() < 1e-6).to_numpy()
        before = len(df)
        df = df[~bad]
        print(f"[data] prefilter: dropped {int(bad.sum()):,} rows "
              f"({bad.mean():.1%}) with chg_pct == {cfg.missing_position_value:.0%} "
              f"(units detected: {_scale}) -- position info missing next quarter, not a "
              f"real sell. {before:,} -> {len(df):,} rows")
        if len(df) == 0:
            raise ValueError("prefilter removed every row -- check missing_position_value / units")

    df = df.sort_values(["fund", "security", "yq"]).reset_index(drop=True)
    keys = ["fund", "security"]

    # own-position dynamics. These are LEVEL features fed to the LSTM as context, so a
    # missing prior quarter must be FORWARD-FILLED (carry the last-known level across the
    # gap), NOT reset. df is sorted by (fund, security, yq), and the panel has one row per
    # OBSERVED quarter only, so groupby(keys).shift(k) already returns the previous
    # *observed* quarter -- i.e. forward-fill across gaps, using only past info.
    #
    # This is deliberately different from the TARGET and the peer/fund lookups, which use
    # EXACT-quarter alignment (_shift_exact / _at_prev_quarter): a change/label must never
    # be fabricated across a gap. But a level fed as context should persist, not drop to 0.
    #
    # Why it matters: a lag feature left NaN at a *present* quarter is zero-filled in
    # build_sequences and masked IN as a real 0 -- so a one-quarter gap would look like
    # "shares/weight jumped up from an empty position", a fabricated giant trade that
    # distorts the whole panel. lag_fill="zero" reproduces that old distorted behaviour
    # (exact-quarter -> NaN -> 0) for A/B; "ffill" (default) is correct.
    _ff = getattr(cfg, "lag_fill", "ffill") == "ffill"
    g = df.groupby(keys, observed=True)
    if _ff:
        # forward-fill: previous observed quarter; fillna(current) only for a genuine
        # first-ever appearance (no prior at all) -> "no trade" (change 0), never a spike.
        for k in (1, 2, 3):
            df[f"sh_lag{k}"] = g["shares"].shift(k).fillna(df["shares"]).astype(F32)
        df["w_lag1"] = g["weight"].shift(1).fillna(df["weight"]).astype(F32)
        df["dw"] = (df["weight"] - df["w_lag1"]).astype(F32)
        df["pdsh"] = ((df["shares"] - df["sh_lag1"]) / (df["sh_lag1"].abs() + 1.0)).astype(F32)
        df["pdsh_sign"] = np.sign(df["pdsh"]).fillna(0.0).astype(F32)
        df["pdsh_lag1"] = df.groupby(keys, observed=True)["pdsh"].shift(1).fillna(0.0).astype(F32)
    else:  # "zero": old exact-quarter -> NaN -> 0-spike (distorted; kept only for A/B)
        for k in (1, 2, 3):
            df[f"sh_lag{k}"] = _shift_exact(df, keys, "shares", k).astype(F32)
        df["w_lag1"] = _shift_exact(df, keys, "weight", 1).astype(F32)
        df["dw"] = (df["weight"] - df["w_lag1"]).astype(F32)
        df["pdsh"] = ((df["shares"] - df["sh_lag1"]) / (df["sh_lag1"].abs() + 1.0)).astype(F32)
        df["pdsh_sign"] = np.sign(df["pdsh"]).fillna(0.0).astype(F32)
        df["pdsh_lag1"] = _shift_exact(df, keys, "pdsh", 1).astype(F32)
    df["log_posval"] = np.log(df["position_value"].abs() + 1.0).astype(F32)
    df["log_pv"] = np.log(df["portfolio_value"].abs() + 1.0).astype(F32)
    df["log_mktcap"] = np.log(df["market_cap"].abs() + 1.0).astype(F32)

    # ---------------------------------------------------------------- TARGET
    # Y = sign of the t -> t+1 fractional share change, with a +-change_band dead-band:
    # {-1 sell, 0 hold, +1 buy}. A label, never a feature.
    #
    # Preferred source: the data's own `future_1q_shares_change_pct`. It is already the
    # t->t+1 change and encodes a FULL EXIT as -100%, so genuine sells (incl. positions
    # that leave the panel) are captured -- nothing is silently dropped.
    # Fallback (older files without it): shares[t+1] - shares[t], requiring EXACTLY t+1
    # (a gap -> the row is dropped, never guessed across the gap).
    _use_chg = (getattr(cfg, "target_source", "auto") in ("auto", "chg_pct")
                and "chg_pct" in df.columns and df["chg_pct"].notna().any())
    if _use_chg:
        chg = df["chg_pct"].astype("float64")
        scale = _chg_scale(chg, cfg)
        if scale == "percent":
            chg = chg / 100.0                      # -> fraction, so change_band is a fraction
        print(f"[data] target from future_1q_shares_change_pct (units detected: {scale})")
        dsh = chg
        # strictness check: is chg_pct really a t -> t+1 change (not spanning a gap)?
        sh_next = _shift_exact(df, keys, "shares", -1)       # shares at EXACT t+1 (else NaN)
        has_tp1 = sh_next.notna().to_numpy()
        absent = (~has_tp1) & df["shares"].notna().to_numpy() & df["chg_pct"].notna().to_numpy()
        exit_like = (float(np.mean(np.abs(dsh.to_numpy()[absent] + 1.0) < 0.05))
                     if absent.sum() else 1.0)
        print(f"[data] target strictness: {has_tp1.mean():.1%} of rows have an exact t+1 obs; "
              f"of {int(absent.sum()):,} absent at t+1, {exit_like:.0%} look like full exits "
              f"(chg=-100%). Low % => your chg_pct may span gaps (not strictly 1-quarter).")
        if getattr(cfg, "target_strict_next", False):        # enforce: absent at t+1 -> sell
            dsh = dsh.copy(); dsh[~has_tp1] = -1.0
            print("[data] target_strict_next=True -> forced 'sell' where the exact t+1 obs "
                  "is absent (position exited). Do NOT use with a max_rank filter.")
    else:
        sh_next = _shift_exact(df, keys, "shares", -1)     # exact t+1 only
        dsh = (sh_next - df["shares"]) / (df["shares"].abs() + 1.0)
        drop = int((sh_next.isna() & df["shares"].notna()).sum())
        print(f"[data] no chg_pct column -> target from shares[t+1] (exact t+1); "
              f"{drop:,} rows dropped (position absent at t+1)")
    df["Y"] = np.select([dsh <= -cfg.change_band, dsh >= cfg.change_band],
                        [-1.0, 1.0], default=0.0).astype(F32)
    df.loc[dsh.isna(), "Y"] = np.nan
    bal = pd.Series(df["Y"]).value_counts(normalize=True)
    print(f"[data] class balance  sell {bal.get(-1.0, 0):.3f} | hold {bal.get(0.0, 0):.3f} "
          f"| buy {bal.get(1.0, 0):.3f}  (labelled {int(df['Y'].notna().sum()):,})")
    # Realised returns -- EVALUATION ONLY, never features. Carry BOTH so all three
    # eval_modes can be produced from one training run.
    df["fwd_1q"] = df["future_1q_ret"]                     # t   -> t+1
    df["fwd_2q"] = df["future_2q_ret"]                     # t+1 -> t+2  (after acc(t) is known)
    df["fwd_3q"] = df["future_3q_ret"]                     # t+2 -> t+3  (also after filing delay)

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

    # Category Activity Rates (paper App. A): share-increase / -decrease / no-change
    # rates among PEER managers, lagged one quarter. Grouped WITHIN InvTypeCode when
    # available -- a market-wide rate is identical for every fund in a quarter and so
    # carries no cross-sectional information at all.
    lab = df.dropna(subset=["Y"])
    n_cat = df["inv_type"].nunique() if "inv_type" in df.columns else 1
    use_cat = "inv_type" in df.columns and n_cat > 1
    aggs = dict(peer_buy=("Y", lambda s: (s > 0).mean()),
                peer_sell=("Y", lambda s: (s < 0).mean()),
                peer_hold=("Y", lambda s: (s == 0).mean()))
    allq = pd.PeriodIndex(sorted(df["yq"].unique()), freq="Q")   # dense quarter grid
    if use_cat:
        rate = lab.groupby(["inv_type", "yq"], observed=True).agg(**aggs)   # index (inv_type, yq)
        # forward-fill the RATE across gaps so a missing (inv_type, yq-1) carries the
        # last-observed rate instead of NaN -> 0 ("0% of peers bought", a distortion).
        if _ff:
            rate = _dense_ffill(rate, "inv_type", allq)
        for col in ("peer_buy", "peer_sell", "peer_hold"):                  # look up (inv_type, yq-1)
            df[col] = pd.Series(_at_prev_quarter(df, ["inv_type"], rate[col], 1), index=df.index).astype(F32)
        print(f"[data] peer rates computed within {n_cat} InvTypeCode categories")
    else:
        rate = lab.groupby("yq").agg(**aggs)                    # index yq (no row-shift)
        if _ff:
            rate = rate.reindex(allq).ffill()                   # carry last-known rate across gaps
        prevq = df["yq"] - 1                                    # exact previous quarter
        for col in ("peer_buy", "peer_sell", "peer_hold"):
            df[col] = prevq.map(rate[col]).astype(F32)
        if "inv_type" in df.columns:
            print("[data] single InvTypeCode -> peer rates are market-wide "
                  "(no cross-sectional variation; pass several codes to enable category peers)")

    # fund past-quarter return proxy (weight-weighted). exact-quarter lookup, no row-shift;
    # forward-filled across a full-fund gap so yq-1 missing carries the last-known fund
    # return instead of NaN -> 0.
    contrib = (df["w_lag1"] * df["quarterly_ret"])
    fr = contrib.groupby([df["fund"], df["yq"]]).sum()          # index (fund, yq)
    fr.index = fr.index.set_names(["fund", "yq"])
    if _ff:
        fr = _dense_ffill(fr, "fund", allq)
    df["fund_ret_l1"] = pd.Series(_at_prev_quarter(df, ["fund"], fr, 1), index=df.index).astype(F32)

    # integer quarter index for windowing
    qs = pd.PeriodIndex(sorted(df["yq"].unique()), freq="Q")
    df["qi"] = df["yq"].map({q: i for i, q in enumerate(qs)}).astype("int32")
    df["held"] = np.int8(1)

    # per-bucket support: a sample's bucket = per-quarter quantile of `bucket_by` at its
    # label quarter t (look-ahead-safe). Compute sum_abs_aw here if requested and absent.
    bcol = cfg.bucket_by
    if bcol:
        if bcol in df.columns:
            bval = df[bcol].astype("float64")
        elif bcol == "sum_abs_aw":
            bval = _sum_abs_aw(df)
        else:
            raise ValueError(f"bucket_by={bcol!r} not in data and not computable "
                             "(only 'sum_abs_aw' is auto-computed)")
        df["_bkt"] = _per_quarter_bucket(bval, df["yq"], cfg.n_buckets).astype("float32")
        nb = df.loc[df["Y"].notna(), "_bkt"]
        print(f"[data] bucket_by={bcol} -> {cfg.n_buckets} per-quarter buckets | "
              f"labelled-row counts: {nb.value_counts(dropna=False).sort_index().to_dict()}")

    # (c) PRUNE IN PLACE to only what's needed (no big .copy()). Everything already float32.
    feat = [f for f in cfg.features if f in df.columns]
    df.drop(columns=["qi_tmp"], inplace=True, errors="ignore")
    keep = set(["fund", "security", "yq", "qi", "held", "Y", "fwd_1q", "fwd_2q", "fwd_3q",
                "inv_type", "weight", "rank"] + (["_bkt"] if bcol else []) + feat)
    df.drop(columns=[c for c in df.columns if c not in keep], inplace=True)
    if cfg.downcast:
        df["fund"] = df["fund"].astype("category")
        df["security"] = df["security"].astype("category")
        if "inv_type" in df.columns:
            df["inv_type"] = df["inv_type"].astype("category")
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
    mcols = [c for c in ["fund", "security", "yq", "qi", "Y", "fwd_1q", "fwd_2q", "fwd_3q",
                         "inv_type", "weight", "rank", "_bkt"] if c in sub.columns]
    meta = sub.loc[valid, mcols].reset_index(drop=True)
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
    if cfg.enforce_sell_feasibility:            # prohibit "sell" for infeasible securities
        infeas = ~feas[te_i]                    # (full sample -- a prediction constraint,
        P[infeas, 0] = 0.0                      #  NOT a sample drop)
        s = P.sum(1, keepdims=True); s[s == 0] = 1.0
        P = P / s
    m["p_sell"], m["p_hold"], m["p_buy"] = P[:, 0], P[:, 1], P[:, 2]
    m["y_pred"] = P.argmax(1) - 1
    m["feasible"] = feas[te_i]
    maj = meta[tr].groupby("fund")["Y"].agg(lambda s: s.value_counts().idxmax())
    gmaj = meta.loc[tr, "Y"].value_counts().idxmax()   # global train majority (unbiased fallback
    m["y_naive"] = m["fund"].map(maj).fillna(gmaj)     # for a fund unseen in this train window;
    return m                                           # NOT 0/"hold", which would bias the naive)


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
        outs += _fit_bucketed(X, mask, y, feas, meta, tr, te, F, hidden, cfg, dev)
    return outs


_BSKIP = {"seen": 0, "thin_train": 0, "no_test": 0, "model_none": 0, "max_seen": 0}


def _min_train_bucket(cfg):
    """Train-sequence floor for ONE (window, bucket). Bucketing splits a window's training
    rows n_buckets ways, so the floor must scale with n_buckets -- otherwise every bucket
    is starved and nothing trains."""
    if cfg.min_train_bucket is not None:
        return int(cfg.min_train_bucket)
    base = cfg.min_train_global if cfg.model_mode == "global" else cfg.min_seq_per_fund
    return max(30, int(base) // max(1, int(cfg.n_buckets)))


def _fit_bucketed(X, mask, y, feas, meta, tr, te, F, hidden, cfg, dev):
    """Train ONE model on (tr,te) -- or, when cfg.bucket_by is set, a SEPARATE model per
    bucket: for each bucket b, restrict BOTH train and test to samples whose label-quarter
    bucket == b, so training and testing never mix buckets. Returns a list of prediction
    frames (tagged with `bucket`)."""
    if not cfg.bucket_by or "_bkt" not in meta.columns:
        o = _train_predict(X, mask, y, feas, meta, tr, te, F, hidden, cfg, dev)
        return [o] if o is not None else []
    bk = meta["_bkt"].to_numpy()
    thr = _min_train_bucket(cfg)
    outs = []
    for b in range(cfg.n_buckets):
        mb = bk == b
        trb, teb = tr & mb, te & mb
        _BSKIP["seen"] += 1
        if int(trb.sum()) < thr:
            _BSKIP["thin_train"] += 1
            _BSKIP["max_seen"] = max(_BSKIP["max_seen"], int(trb.sum()))
            continue
        if int(teb.sum()) == 0:
            _BSKIP["no_test"] += 1
            continue
        o = _train_predict(X, mask, y, feas, meta, trb, teb, F, hidden, cfg, dev)
        if o is not None:
            o["bucket"] = int(b)
            outs.append(o)
        else:
            _BSKIP["model_none"] += 1
    return outs


def run_model(panel: pd.DataFrame, cfg: Config, verbose=True):
    """Walk-forward training. `global` = one shared model per window (batched,
    memory-heavier); `per_fund` = one model per fund, funds parallelised across
    cores (CPU-friendly, low memory). Returns pooled OOS predictions.

    When cfg.bucket_by is set, each window trains a SEPARATE model per bucket (predictions
    carry a `bucket` column) -- never one pooled model."""
    import os as _os
    import torch
    for _k in _BSKIP:                      # fresh bucket accounting each run
        _BSKIP[_k] = 0
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
            for out in _fit_bucketed(X, mask, y, feas, meta, tr, te, F, cfg.hidden, cfg, dev):
                preds.append(out)
                if verbose:
                    acc = (out.loc[out.feasible, "y_pred"] == out.loc[out.feasible, "Y"]).mean()
                    btag = f" b{int(out['bucket'].iloc[0])}" if "bucket" in out.columns else ""
                    print(f"  win {wi+1}{btag} test qi[{te_lo},{te_hi}) "
                          f"n_te={int(len(out)):,} feas_acc={acc:.3f}")
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
        if cfg.bucket_by and _BSKIP["seen"]:
            thr = _min_train_bucket(cfg)
            raise RuntimeError(
                f"no predictions produced: every (window, bucket) cell was skipped.\n"
                f"  cells seen           : {_BSKIP['seen']}\n"
                f"  skipped, thin train  : {_BSKIP['thin_train']}  "
                f"(largest cell had {_BSKIP['max_seen']} train seqs, need >= {thr})\n"
                f"  skipped, empty test  : {_BSKIP['no_test']}\n"
                f"  model returned None  : {_BSKIP['model_none']}\n"
                f"Bucketing splits each window's training rows {cfg.n_buckets} ways, so each "
                f"cell holds ~1/{cfg.n_buckets} of them. Fix by LOWERING min_train_bucket "
                f"(currently {'auto=' if cfg.min_train_bucket is None else ''}{thr}), lowering "
                f"n_buckets, or using model_mode='global' (pools funds -> far more rows per cell).")
        raise RuntimeError("no predictions produced -- check filters / window sizes")
    if cfg.bucket_by and verbose:
        print(f"[model] bucket cells: {_BSKIP['seen']} seen, "
              f"{_BSKIP['thin_train']} skipped (thin train, need >= {_min_train_bucket(cfg)}), "
              f"{_BSKIP['no_test']} skipped (empty test)")
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
        if cols.get("fwd_2q") is True:
            return "fwd_2q", 0
        print("[warn] eval_mode='predictive' needs `future_2q_ret`; falling back to 'lagged'")
        return "fwd_1q", 1
    if m == "tradeable":
        if cols.get("fwd_3q") is True:
            return "fwd_3q", 0
        print("[warn] eval_mode='tradeable' needs `future_3q_ret`; falling back to 'predictive'")
        return ("fwd_2q", 0) if cols.get("fwd_2q") is True else ("fwd_1q", 1)
    raise ValueError(f"unknown eval_mode: {m!r}")


def evaluate(preds: pd.DataFrame, cfg: Config):
    """Predictability + portfolio sorts (Tables X / XI / XII). Returns
    (metrics dict, {table_name: DataFrame}). Timing is set by cfg.eval_mode."""
    P = preds.copy()
    P["yq"] = P["yq"].astype("period[Q]")
    has = {c: bool(P[c].notna().any()) if c in P.columns else False
           for c in ("fwd_1q", "fwd_2q", "fwd_3q")}
    ret_col, lag = _resolve_eval(cfg, has)
    P["_ret"] = P[ret_col]
    # feasible_only is an EVALUATION sample restriction ONLY (the sell-zeroing prediction
    # constraint is separate; see enforce_sell_feasibility). False (default) -> every table
    # uses all labelled rows; True -> restrict to feasible rows. Applied to ALL tables alike.
    E = P[P["feasible"]] if cfg.feasible_only else P
    m = {}
    m["lstm_precision_pooled"] = float((E.y_pred == E.Y).mean())
    m["naive_precision_pooled"] = float((E.y_naive == E.Y).mean())
    fp = E.groupby("fund", observed=True).apply(lambda d: pd.Series({
        "lstm": (d.y_pred == d.Y).mean(), "naive": (d.y_naive == d.Y).mean()}))
    m["lstm_precision_fundavg"] = float(fp["lstm"].mean())
    m["naive_precision_fundavg"] = float(fp["naive"].mean())
    m["feasible_only"] = cfg.feasible_only
    m["n_predictions"] = int(len(P)); m["n_used"] = int(len(E))
    m["n_feasible"] = int(P["feasible"].sum()); m["n_funds"] = int(E.fund.nunique())
    m["eval_mode"] = cfg.eval_mode; m["eval_return"] = ret_col; m["eval_sort_lag"] = lag
    if "bucket" in E.columns:                    # per-bucket precision (samples were trained
        m["bucket_precision"] = {int(b): float((g.y_pred == g.Y).mean())   # separately)
                                 for b, g in E.groupby("bucket")}
        m["enforce_sell_feasibility"] = cfg.enforce_sell_feasibility

    def xsq(s):
        v = s.dropna()
        if v.nunique() < 5:
            return pd.Series(np.nan, index=s.index)
        return (pd.qcut(v.rank(method="first"), 5, labels=False, duplicates="drop") + 1).reindex(s.index)

    def lag_q(d, keys, col, k):
        """Lag `col` by EXACTLY k quarters within `keys`. A plain groupby().shift(k)
        would return the previous *row*, which after gaps can be 3 or 20 quarters back
        -- silently sorting on a stale signal. Rows without a true t-k are set to NaN."""
        if k == 0:
            return d[col]
        v = d.groupby(keys, observed=True)[col].shift(k)
        q = d.groupby(keys, observed=True)["yq"].shift(k)
        return v.where(q == d["yq"] - k, np.nan)

    # ---- fund-level: predictability + benchmark-adjusted future return ----
    # Benchmark = mean of PEER funds in the same InvTypeCode that quarter (the paper
    # benchmarks against the fund's own Morningstar category). Falls back to the whole
    # universe if there is only one category.
    P["wc"] = P["weight"] * P["_ret"]
    has_cat = "inv_type" in P.columns and P["inv_type"].nunique() > 1

    def _fq(d):
        sub = d[d.feasible] if cfg.feasible_only else d       # prec respects the switch
        rm = d["_ret"].notna()                                # weighted mean over AVAILABLE returns:
        wsum = float(d.loc[rm, "weight"].sum())               # a missing-return position must be out
        o = {"fund_ret": d.loc[rm, "wc"].sum() / wsum if wsum > 0 else np.nan,  # of BOTH num AND denom
             "prec": (sub.y_pred == sub.Y).mean()}            # (else it dilutes fund_ret toward 0)
        if has_cat:
            o["inv_type"] = d["inv_type"].iloc[0]
        return pd.Series(o)
    fq = P.groupby(["fund", "yq"], observed=True).apply(_fq).reset_index()
    bench = ["inv_type", "yq"] if has_cat else ["yq"]
    m["benchmark"] = "InvTypeCode x quarter" if has_cat else "universe x quarter"
    fq["abn"] = fq["fund_ret"] - fq.groupby(bench, observed=True)["fund_ret"].transform("mean")
    fq = fq.sort_values(["fund", "yq"])
    fq["prec_lag"] = lag_q(fq, "fund", "prec", lag)
    # forward cumulative abnormal return over [t, t+h-1], QUARTER-AWARE: NaN if any of
    # those quarters is missing for the fund (never sum across a gap).
    abn_lut = fq.set_index(["fund", "yq"])["abn"]
    for h in range(1, 5):
        tot = np.zeros(len(fq)); ok = np.ones(len(fq), dtype=bool)
        for j in range(h):
            a = _at_prev_quarter(fq, ["fund"], abn_lut, -j)   # abn at t+j
            miss = pd.isna(a); ok &= ~miss
            tot = tot + np.where(miss, 0.0, a)
        fq[f"cabn{h}"] = np.where(ok, tot, np.nan)
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

    # ---- Table XI: correct vs incorrect positions ----  (same E subset as precision)
    # TIMING handled by eval_mode: in "predictive" the return (_ret = t+1->t+2) already
    # starts AFTER correct(t) is revealed, so lag=0. In "lagged" we shift correct by 1.
    Pe = E.sort_values(["fund", "security", "yq"]).copy()
    Pe["correct"] = (Pe.y_pred == Pe.Y).astype(float)
    Pe["correct_s"] = lag_q(Pe, ["fund", "security"], "correct", lag)
    ci = Pe.dropna(subset=["_ret", "correct_s"])
    corr = ci[ci.correct_s == 1].groupby("yq")["_ret"].mean()
    inco = ci[ci.correct_s == 0].groupby("yq")["_ret"].mean()
    diff = (corr - inco).dropna()
    tableXI = pd.DataFrame({"portfolio": ["Correct", "Incorrect", "Correct-Incorrect"],
                            "mean_qret": [corr.mean(), inco.mean(), diff.mean()],
                            "t": [_t(corr), _t(inco), _t(diff)]})
    m["correct_minus_incorrect"] = float(diff.mean()); m["correct_minus_incorrect_t"] = float(_t(diff))

    # ---- Table XII: stock quintiles on cross-fund prediction accuracy ----
    stk = Pe.groupby(["security", "yq"], observed=True).agg(
        acc=("correct", "mean"), fwd=("_ret", "mean")).reset_index()   # mean skips per-fund NaN

    stk = stk.sort_values(["security", "yq"])
    # same timing rule as above, driven by eval_mode (exact-quarter lag)
    stk["acc_s"] = lag_q(stk, "security", "acc", lag)
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


def evaluate_by_bucket(preds: pd.DataFrame, cfg: Config):
    """Run the full evaluation SEPARATELY on each bucket's predictions (each bucket was
    trained + tested by its own model), plus the pooled result. Returns
    {'pooled': metrics, 0: metrics_b0, 1: ...}. Use after run() when bucket_by is set."""
    out = {"pooled": evaluate(preds, cfg)[0]}
    if "bucket" in preds.columns:
        for b, g in preds.groupby("bucket"):
            try:
                out[int(b)] = evaluate(g, cfg)[0]
            except Exception as e:
                out[int(b)] = {"error": str(e), "n": int(len(g))}
    return out


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
                       modes=("contemporaneous", "lagged", "predictive", "tradeable")):
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
