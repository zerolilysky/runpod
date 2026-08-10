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
TARGETS = ["turnover_next", "ret_next"]          # the default pair

# Longer return horizons, compounded from the forward-return columns already in the file:
#     ret_next_2q = (1+f1)(1+f2) - 1     ret_next_3q = (1+f1)(1+f2)(1+f3) - 1
# Quarterly-rebalanced h-quarter returns OVERLAP h-1 times, so their t-stats need a
# Newey-West correction -- handled in _score, not something the caller has to remember.
RETURN_TARGETS = {"ret_next": 1, "ret_next_2q": 2, "ret_next_3q": 3}


@dataclass
class Config:
    holdings_path: str = ("manager_holdings/master_batches_return_filtered/"
                          "master_all_funds_add_filter_ivy_rank_active_rank.parquet")
    # column names in the raw file
    col_map: dict = field(default_factory=lambda: {
        "security": "security", "date": "date", "close": "close", "volume": "volume",
        "market_cap": "market_cap", "quarterly_ret": "quarterly_ret",
        "future_1q_ret": "future_1q_ret", "future_2q_ret": "future_2q_ret",
        "future_3q_ret": "future_3q_ret", "InvTypeCode": "inv_type", "isUs": "isUs",
    })
    # NOTE the annotation. A dataclass only turns ANNOTATED class attributes into fields, so
    # `inv_type_codes = (401,)` would be a plain class attribute and Config(inv_type_codes=...)
    # would raise TypeError: unexpected keyword argument.
    inv_type_codes: tuple = (401,)
    us_only: bool = True

    # Which return horizons to test as targets, in quarters. (1,) is the q -> q+1 return.
    # (1, 2, 3) adds the compounded 2- and 3-quarter returns; they need future_2q_ret /
    # future_3q_ret in the file and their spreads get Newey-West t-stats.
    return_horizons: tuple = (1, 2)

    vol_window: int = 4        # quarters used for vol_ret (current + 3 lags)
    # Per-quarter clip on the FEATURES. Cross-sectional, so a quarter is clipped using only
    # its own securities -- no information travels backwards in time. 0.01 clips at the 1st
    # and 99th percentile. Set 0.0 to disable.
    winsorize: float = 0.01
    # Applies to the five features and to turnover_next. The RETURN targets are never
    # clipped: a quintile mean is meant to include the tail.
    # History a security needs before it enters the universe. Expanding, never full-sample,
    # so the requirement is answerable at q and carries no survivorship.
    min_quarters: int = 8
    # How to count that history.
    #   False  any `min_quarters` OBSERVED quarters, gaps allowed. A name that vanishes for
    #          four years and reappears qualifies immediately, on stale history, while its
    #          lagged features are all NaN.
    #   True   `min_quarters` CONSECUTIVE quarters ending at q. Stricter and the honest
    #          reading of "has enough history", but it drops gappy names for longer.
    require_consecutive: bool = False

    # rolling out-of-sample split, same scheme for every model
    window_q: int = 28
    test_q: int = 8
    step: int = 8

    # Score every row of the table on ONE common sample: the rows where every target and
    # every feature is available, restricted to the quarters the models could score. Without
    # it, turnover_next and ret_next_2q are graded on different quarters, and a raw sort is
    # graded on fewer rows than its own model. Training is never restricted by this.
    align_eval_sample: bool = True

    n_quintiles: int = 5

    # "hgb"     HistGradientBoostingRegressor -- a step function of the features, can find a
    #           non-monotone shape (mid turnover best, both tails bad) but can also fit noise
    # "linear"  median-imputed, standardised OLS. Monotone by construction, so for a SINGLE
    #           feature `model:x` becomes an exact monotone map of x and can only differ from
    #           `raw:x` in SIGN -- which makes it the clean way to read the direction the
    #           training window taught. If hgb barely beats it, the relationship is linear and
    #           the trees are adding nothing.
    model: str = "hgb"
    max_iter: int = 300
    learning_rate: float = 0.06
    max_depth: int = 4         # shallow: five features do not need more
    seed: int = 0


# ------------------------------------------------------------------ helpers
def _t(x, lags: int = 0) -> float:
    """t-stat of a mean. `lags > 0` applies a Newey-West (Bartlett) correction, which is
    required whenever the series overlaps -- an h-quarter return sampled every quarter shares
    h-1 quarters with its neighbour, so the naive t-stat is inflated by roughly sqrt(h)."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return np.nan
    if lags <= 0:
        return float(x.mean() / (x.std(ddof=1) / np.sqrt(n)))
    e = x - x.mean()
    var = float(e @ e) / n
    for L in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - L / (lags + 1.0)) * float(e[L:] @ e[:-L]) / n
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else np.nan


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
            "future_1q_ret", "future_2q_ret", "future_3q_ret", "inv_type", "isUs"]
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

    for c in ("close", "volume", "market_cap", "quarterly_ret",
              "future_1q_ret", "future_2q_ret", "future_3q_ret"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("future_2q_ret", "future_3q_ret"):
        if c not in df.columns:
            df[c] = np.nan

    # the file is a fund-security-quarter panel; these are all security-level, so one row per
    # (security, quarter) is all that is needed
    sq = df.groupby(["security", "yq"], observed=True).agg(
        close=("close", "first"), volume=("volume", "first"),
        mktcap=("market_cap", "first"), ret_q=("quarterly_ret", "first"),
        ret_next=("future_1q_ret", "first"), _f2=("future_2q_ret", "first"),
        _f3=("future_3q_ret", "first"), n_rows=("security", "size"),
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

    # target 1: next quarter's turnover, exact quarter only.
    # NOTE `g` was built before `sq = sq.drop(columns=lag_cols)` above, and .drop() returns a
    # NEW frame, so `g` still refers to the pre-drop object. That is harmless here only
    # because nothing has modified `turnover` in between -- the two frames hold the same
    # values. Do not insert a column-modifying step above this line without rebuilding `g`.
    v, qq = g["turnover"].shift(-1), g["qi"].shift(-1)
    sq["turnover_next"] = v.where(qq == qi + 1)

    # target 2+: returns. ret_next (q -> q+1) is already on the row; longer horizons compound
    # the forward columns. Compounding, not summing -- (1+r1)(1+r2)-1 is the return actually
    # earned by holding through both quarters.
    sq["ret_next_2q"] = (1 + sq["ret_next"]) * (1 + sq["_f2"]) - 1
    sq["ret_next_3q"] = (1 + sq["ret_next"]) * (1 + sq["_f2"]) * (1 + sq["_f3"]) - 1
    sq = sq.drop(columns=["_f2", "_f3"])

    sq = _winsor_by_q(sq, FEATURES + ["turnover_next"], cfg.winsorize)

    # History filter. cumcount() counts OBSERVED rows for the security in (security, qi)
    # order, so gaps are simply skipped -- it is "the n-th quarter we have seen", not "n
    # quarters in a row". require_consecutive restarts the count wherever the chain breaks.
    if cfg.require_consecutive:
        brk = g["qi"].diff() != 1                      # True at a gap and at the first row
        run = brk.groupby(sq["security"]).cumsum()
        seen = sq.groupby(["security", run]).cumcount() + 1
    else:
        seen = g.cumcount() + 1
    n_before = len(sq)
    sq = sq[seen >= cfg.min_quarters].reset_index(drop=True)
    print(f"[panel] history filter: {n_before:,} -> {len(sq):,} rows "
          f"(min_quarters={cfg.min_quarters}, "
          f"{'consecutive' if cfg.require_consecutive else 'observed, gaps allowed'})")

    for c in FEATURES + ["turnover_next"] + list(RETURN_TARGETS):
        sq[c] = sq[c].astype("float32")
    print(f"[panel] {len(sq):,} rows after history filter")
    print("[panel] target coverage:  " + "  ".join(
        f"{c} {sq[c].notna().mean():.0%}" for c in ["turnover_next"] + list(RETURN_TARGETS)))
    print("[panel] feature coverage: " + "  ".join(
        f"{c} {sq[c].notna().mean():.0%}" for c in FEATURES))
    return sq


def target_list(cfg: Config) -> List[str]:
    """turnover plus whichever return horizons the config asks for and the panel can supply."""
    inv = {v: k for k, v in RETURN_TARGETS.items()}
    return ["turnover_next"] + [inv[h] for h in cfg.return_horizons if h in inv]


# ------------------------------------------------------------------ model
def _make_estimator(cfg: Config):
    """`cfg.model` picks the learner; everything else about the run is unchanged, so the two
    are directly comparable on the same folds and the same evaluation sample.

    linear needs the imputer and the scaler that HGB does not: HGB routes missing values down
    a branch of their own and is scale-free, OLS is neither.
    """
    kind = str(cfg.model).lower()
    if kind in ("hgb", "gbm", "tree"):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=cfg.max_iter,
                                             learning_rate=cfg.learning_rate,
                                             max_depth=cfg.max_depth,
                                             random_state=cfg.seed)
    if kind in ("linear", "ols", "lm"):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LinearRegression())
    raise ValueError(f"cfg.model must be 'hgb' or 'linear', got {cfg.model!r}")


def fold_schedule(panel: pd.DataFrame, targets: List[str], cfg: Config) -> List[int]:
    """The fold end-points every target can run, so all targets share one schedule.

    Left to itself, `range(window_q, d.qi.max() + 2, step)` is computed per target -- and
    d.qi.max() differs between them. turnover_next is NaN in the final quarter (there is no
    q+1), so its panel ends one quarter earlier and the WHOLE schedule shifts, not just the
    last fold. A fold can also drop out for one target alone if that target leaves it with
    too little training data. Intersecting the surviving folds makes every target land on the
    same test quarters.
    """
    ends = None
    for t in targets:
        d = panel.dropna(subset=[t])
        ok = {c for c in range(cfg.window_q, int(d.qi.max()) + 2, cfg.step)
              if ((d.qi >= c - cfg.window_q) & (d.qi < c - cfg.test_q)).sum() >= 500
              and ((d.qi >= c - cfg.test_q) & (d.qi < c)).sum() > 0}
        ends = ok if ends is None else (ends & ok)
    return sorted(ends or [])


def _rolling_predict(panel: pd.DataFrame, feats: List[str], target: str,
                     cfg: Config, folds: List[int] = None) -> pd.DataFrame:
    """Same rolling out-of-sample split for every model, so the six are comparable."""
    d = panel.dropna(subset=[target])
    if folds is None:
        folds = list(range(cfg.window_q, int(d.qi.max()) + 2, cfg.step))
    out = []
    for c in folds:
        tr = d[(d.qi >= c - cfg.window_q) & (d.qi < c - cfg.test_q)]
        te = d[(d.qi >= c - cfg.test_q) & (d.qi < c)]
        if len(tr) < 500 or len(te) == 0:
            continue
        m = _make_estimator(cfg)
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
    """rank-IC against the target, plus the quintile spread measured in RETURN.

    Which return? The one that matches the target's own horizon: a 3-quarter target is graded
    on the 3-quarter return. A turnover target has no horizon of its own, so it is graded on
    the 1-quarter return. `Q5_Q1_per_q` divides by the horizon so every row is comparable.

    Overlapping horizons get a Newey-West t-stat with h-1 lags: quarterly-rebalanced
    h-quarter returns share h-1 quarters with their neighbour, and the naive t is inflated by
    roughly sqrt(h).
    """
    h = RETURN_TARGETS.get(target, 1)
    ret_col = target if target in RETURN_TARGETS else "ret_next"

    blank = {"rank_IC": np.nan, "IC_t": np.nan, "ret_h_q": h, "Q1_ret_pct": np.nan,
             "Q5_ret_pct": np.nan, "Q5_Q1_pct": np.nan, "Q5_Q1_per_q": np.nan,
             "spread_t": np.nan, "n_quarters": 0, "n_rows": 0, "sample_id": 0}

    x = d.dropna(subset=[pred_col, target])
    ic = (x.groupby("qi").apply(lambda t: t[pred_col].corr(t[target], method="spearman"))
          if len(x) else pd.Series(dtype=float))
    r = d.dropna(subset=[pred_col, ret_col]).copy()
    r["Q"] = r.groupby("qi")[pred_col].transform(lambda s: _qcut(s, cfg.n_quintiles))
    r = r.dropna(subset=["Q"])
    # A constant or heavily tied predictor cannot be cut into quintiles, so every quarter
    # drops and `per` comes back empty. Return NaNs for that row rather than taking down the
    # whole table -- the reader still needs the other eleven rows.
    if r.empty:
        return blank
    per = r.groupby(["qi", "Q"])[ret_col].mean().unstack()
    if per.shape[1] < 2:
        return blank
    hi, lo = per.columns.max(), per.columns.min()
    sp = (per[hi] - per[lo]).dropna()
    if sp.empty:
        return blank
    # n_rows is reported because rows can still be lost AFTER the quarters are matched: the
    # tree predicts happily on a row whose feature is NaN (HGB routes missing values down a
    # branch of their own), while a raw sort cannot rank a NaN and drops it. A raw reference
    # with visibly fewer rows than its model is being scored on an easier sub-sample.
    # `sample_id` fingerprints the exact (security, quarter) rows this row of the table was
    # scored on. Equal counts are not proof of equal samples -- two models can lose different
    # rows and still tie. Comparing the fingerprint is the actual guarantee.
    sid = int(pd.util.hash_pandas_object(
        r[["security", "qi"]].sort_values(["security", "qi"]), index=False).sum() & 0xFFFFFFFF)
    return {"rank_IC": ic.mean(), "IC_t": _t(ic),
            "ret_h_q": h, "Q1_ret_pct": per[lo].mean() * 100,
            "Q5_ret_pct": per[hi].mean() * 100, "Q5_Q1_pct": sp.mean() * 100,
            "Q5_Q1_per_q": sp.mean() * 100 / h,
            "spread_t": _t(sp, lags=h - 1), "n_quarters": int(sp.size),
            "n_rows": int(len(r)), "sample_id": sid}


def run_all(panel: pd.DataFrame, cfg: Config = None, verbose: bool = True) -> pd.DataFrame:
    """Six models per target: one per feature, plus one on all five.

    Targets are turnover plus every return horizon in `cfg.return_horizons`.
    """
    cfg = cfg or Config()
    specs = [(f"model:{f}", [f]) for f in FEATURES] + [("model:ALL", FEATURES)]
    targets = [t for t in target_list(cfg)
               if t in panel.columns and panel[t].notna().sum() > 500]

    # ---- ONE evaluation sample for the whole table ------------------------------------
    # Three different things would otherwise make the rows incomparable:
    #   (a) targets differ in availability -- turnover_next is NaN in the last quarter and at
    #       every gap, ret_next_2q needs one more forward quarter still
    #   (b) a raw sort cannot rank a NaN feature, while HGB predicts on those rows anyway
    #   (c) the models cannot score the first `window_q` quarters at all
    # Intersecting over every target AND every feature fixes (a) and (b); (c) is handled by
    # restricting to the quarters the models actually scored, below. Training is NOT
    # restricted -- each model still fits on all the data its own target allows.
    if cfg.align_eval_sample:
        ok = np.ones(len(panel), bool)
        for c in targets + FEATURES:
            ok &= panel[c].notna().to_numpy()
        eval_keys = panel.loc[ok, ["security", "qi"]]
        print(f"[eval] common sample: {int(ok.sum()):,} of {len(panel):,} rows carry every "
              f"target {targets} and every feature")
    else:
        eval_keys = None

    def _restrict(d):
        return d if eval_keys is None else d.merge(eval_keys, on=["security", "qi"])

    # One fold schedule for every target, so the test QUARTERS match as well as the rows.
    folds = fold_schedule(panel, targets, cfg) if cfg.align_eval_sample else None
    if folds is not None:
        print(f"[eval] shared fold schedule: {len(folds)} folds ending at {folds}")

    rows = []
    for target in targets:
        test_qi = None
        for name, feats in specs:
            P = _restrict(_rolling_predict(panel, feats, target, cfg, folds))
            if test_qi is None:
                test_qi = set(P["qi"].unique())     # same split for every model
            r = {"target": target, "model": name, **_score(P, "pred", target, cfg)}
            # Which way did the model learn? For a univariate model the prediction is some
            # step function of its one input, so the rank correlation between prediction and
            # input says whether the fit came out increasing or decreasing. Near +1 with a
            # NEGATIVE rank_IC means the training window taught the opposite sign to the one
            # that held out of sample -- the model did not "fail to flip the sign", it flipped
            # it the way the training data asked.
            if len(feats) == 1 and feats[0] in panel.columns:
                j = P.merge(panel[["security", "qi", feats[0]]], on=["security", "qi"],
                            how="left")
                r["dir_vs_x"] = float(j["pred"].corr(j[feats[0]], method="spearman"))
            rows.append(r)
        # Raw features as references: no model, just sort on the characteristic itself --
        # scored on THE SAME test quarters, otherwise the comparison mixes a model difference
        # with a sample difference (the models cannot score the first `window_q` quarters).
        base = _restrict(panel[panel.qi.isin(test_qi)]).dropna(subset=[target])
        for f in FEATURES:
            rows.append({"target": target, "model": f"raw:{f}", "dir_vs_x": 1.0,
                         **_score(base, f, target, cfg)})
    out = pd.DataFrame(rows)
    out.insert(1, "learner", cfg.model)

    # ---- ALIGNMENT AUDIT -----------------------------------------------------------------
    # Not a formality. `_qcut` returns NaN for a quarter with fewer distinct values than
    # quintiles, and a quarter missing either extreme quintile drops from the spread -- both
    # can hit one predictor and not another, leaving the table quietly incomparable even after
    # the fold schedule and the row set have been matched. The fingerprint catches that.
    ids = out.groupby("target")["sample_id"].nunique()
    if cfg.align_eval_sample:
        ok_within = (ids == 1).all()
        ok_across = out["sample_id"].nunique() == 1
        if verbose or not (ok_within and ok_across):
            print(f"\n[align] identical sample within each target: {bool(ok_within)}"
                  f" | identical across targets: {bool(ok_across)}")
        if not ok_within:
            bad = ids[ids > 1]
            print("  [warn] these targets scored their predictors on DIFFERENT rows:")
            for t in bad.index:
                sub = out[out.target == t]
                print(sub.groupby("sample_id")["model"].apply(list).to_string())
            print("  Usually a predictor with heavy ties: a quarter where it has fewer than")
            print("  n_quintiles distinct values is dropped for that predictor alone.")
        elif not ok_across and verbose:
            print("  (targets differ from each other -- expected only if a target is graded")
            print("   on a horizon that runs past the end of the sample)")

    if verbose:
        for target in targets:
            sub = out[out.target == target].drop(columns=["target", "learner"])
            h = RETURN_TARGETS.get(target, 1)
            grade = target if target in RETURN_TARGETS else "ret_next"
            print(f"\n{'='*100}\nTARGET = {target}   learner = {cfg.model}   "
                  f"(spread graded on {grade}, {h} quarter{'s' if h > 1 else ''})\n{'='*100}")
            print(sub.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        print("\n  rank_IC      against that row's target. For a RETURN target, >0.03 is")
        print("               respectable and >0.05 strong -- returns are near-unforecastable,")
        print("               so a large IC there is a reason to hunt for leakage.")
        print("  dir_vs_x     rank corr between a univariate model's prediction and its own")
        print("               input. +1 = the fit came out increasing in x, -1 = decreasing.")
        print("               dir_vs_x near +1 with rank_IC NEGATIVE means the training window")
        print("               taught the OPPOSITE sign to the one that held out of sample --")
        print("               the relation is unstable in time, not a bug in the model. Then")
        print("               model:x and raw:x agree almost exactly, since the prediction is")
        print("               then just a monotone relabelling of x.")
        print("  Q5_Q1_pct    quintile spread over the whole horizon, in percent")
        print("  Q5_Q1_per_q  the same divided by the horizon -- compare ACROSS targets here")
        print("  spread_t     Newey-West with h-1 lags once the horizon overlaps")
        print("  raw:x        sorting on the characteristic itself. model:x should nearly")
        print("               match it; a big gap means the tree fitted noise.")
    return out


def check_version(verbose=True):
    import os
    if verbose:
        print(f"turnover_study {__version__}  |  {os.path.abspath(__file__)}")
    return __version__
