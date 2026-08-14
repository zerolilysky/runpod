"""stratified_study.py -- the turnover and buying-pressure questions, split by security size.

New file. `turnover_study.py` and `buy_pressure.py` are untouched; this one is self-contained
so it cannot be broken by edits to either.

One row is one (security, quarter). Everything is run SEPARATELY inside each `security_size`
bucket, because a pooled result is dominated by whichever bucket has the most names and can
hide a sign that flips between them. Quintile sorts are formed inside the bucket too, so a
large-cap sort never ranks a large cap against a micro cap.

TIMING -- the one thing to get right
------------------------------------
Write I_t for what is knowable once quarter t has closed.

    security_size, fund_size        a classification at q                       -> I_q
    active_weight                   a state at q                                -> I_q
    security_*_fund_turnover        this quarter's weight vs LAST quarter's,
                                    i.e. the window q-1 -> q                    -> I_q
    chg_pct (future_1q_shares_...)  the window q -> q+1                         -> I_{q+1}
    weight / dw                     dw spans q -> q+1                           -> I_{q+1}

So the three `*_fund_turnover` columns and `active_weight` may be used as features AS THEY
ARE -- they look backwards. Anything derived from `chg_pct` or from dw looks forwards and may
only appear as a TARGET, or lagged by one quarter.

    `assume_fund_turnover_backward = False` flips that assumption and lags them, in case the
    columns actually describe q -> q+1 in your build.

PRESSURE MEASURES
-----------------
    breadth              # holders raising shares / # holders
    dollar_breadth       the same vote weighted by position value
    net_flow             sum of dollars traded / market cap
    weight_change        mean dw across holders
    active_weight_change mean dw with the passive price drift removed
    buy_weight_ratio     sum max(0, dw) / sum |dw|          <- NEW
    buy_dollar_ratio     sum max(0, d$) / sum |d$|          <- NEW

The two ratios are bounded in [0, 1] and are normalised by TOTAL activity, which separates
two things every other measure conflates: how much trading happened, and which way it went.
0.5 means buying and selling balanced; 1 means every dollar of reallocation was a purchase.

Usage
    import stratified_study as S
    panel = S.build_panel(S.Config(holdings_path=...))
    table = S.run_stratified(panel, S.Config())
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict

__version__ = "2026.08.08.1"

SIZE_LABEL = {0: "small", 1: "mid", 2: "large"}

# I_q-measurable, usable as features exactly as they stand
FEATURES = [
    "turnover", "log_mktcap", "ret_q", "vol_ret", "log_price",
    "active_weight_mean", "active_weight_absmean",
    "turn_small", "turn_mid", "turn_large", "turn_large_share",
    "weight_chg_lag1", "buy_frac_lag1", "buy_weight_ratio_lag1",
]

# I_{q+1}-measurable: targets, or features only when lagged
PRESSURE = {
    "breadth": "buy_frac",
    "dollar_breadth": "dollar_buy_frac",
    "net_flow": "flow_pct_cap",
    "weight_change": "weight_chg",
    "active_weight_change": "active_weight_chg",
    "buy_weight_ratio": "buy_weight_ratio",
    "buy_dollar_ratio": "buy_dollar_ratio",
}
RETURN_TARGETS = {"ret_next": 1, "ret_next_2q": 2, "ret_next_3q": 3}


@dataclass
class Config:
    holdings_path: str = ("manager_holdings/master_batches_return_filtered/"
                          "master_all_funds_add_filter_ivy_rank_active_rank.parquet")
    col_map: dict = field(default_factory=lambda: {
        "security": "security", "fund": "fund", "date": "date",
        "close": "close", "volume": "volume", "market_cap": "market_cap",
        "position_value": "position_value", "weight": "weight",
        "quarterly_ret": "quarterly_ret", "future_1q_ret": "future_1q_ret",
        "future_2q_ret": "future_2q_ret", "future_3q_ret": "future_3q_ret",
        "future_1q_shares_change_pct": "chg_pct",
        "InvTypeCode": "inv_type", "isUs": "isUs",
        # the new columns
        "security_size": "security_size", "fund_size": "fund_size",
        "active_weight": "active_weight",
        "security_small_fund_turnover": "turn_small",
        "security_mid_fund_turnover": "turn_mid",
        "security_large_fund_turnover": "turn_large",
    })
    inv_type_codes: tuple = (401,)
    us_only: bool = True

    # See the module docstring. True = the *_fund_turnover columns describe q-1 -> q and are
    # already I_q-measurable. False = they describe q -> q+1 and get lagged one quarter.
    assume_fund_turnover_backward: bool = True

    change_band: float = 0.01
    drop_missing_position: bool = True
    vol_window: int = 4
    winsorize: float = 0.01
    min_quarters: int = 8
    require_consecutive: bool = False

    strata: tuple = (0, 1, 2)          # which security_size buckets to run
    min_stratum_rows: int = 2000       # skip a bucket too small to model

    window_q: int = 28
    test_q: int = 8
    step: int = 8
    align_eval_sample: bool = True
    n_quintiles: int = 5

    model: str = "hgb"                 # hgb | linear
    train_target_transform: str = "rank"   # none | winsor | rank
    train_winsor: float = 0.01
    max_iter: int = 300
    learning_rate: float = 0.06
    max_depth: int = 4
    seed: int = 0


# ------------------------------------------------------------------ helpers
def _t(x, lags: int = 0) -> float:
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
    """Per-quarter cross-sectional clip. Each quarter uses only its own securities, so no
    information travels backwards in time."""
    for c in cols:
        if c in df.columns and df[c].notna().any():
            lo = df.groupby("yq")[c].transform(lambda s: s.quantile(p))
            hi = df.groupby("yq")[c].transform(lambda s: s.quantile(1 - p))
            df[c] = df[c].clip(lo, hi).astype("float32")
    return df


def _autoscale(s: pd.Series, name: str, thresh: float = 1.5) -> pd.Series:
    """Percent or fraction? Decide from the median absolute non-zero value and say so."""
    v = pd.to_numeric(s, errors="coerce").astype("float64")
    nz = v[v.abs() > 1e-12].abs()
    med = float(nz.median()) if len(nz) else np.nan
    if np.isfinite(med) and med > thresh:
        print(f"[scale] {name}: median |x| = {med:.4f} -> treated as PERCENT, divided by 100")
        return v / 100.0
    print(f"[scale] {name}: median |x| = {med:.4f} -> treated as a FRACTION")
    return v


# ------------------------------------------------------------------ panel
def build_panel(cfg: Config = None) -> pd.DataFrame:
    cfg = cfg or Config()
    inv = {v: k for k, v in cfg.col_map.items()}
    want = list(cfg.col_map.values())
    raw = [inv[c] for c in want if c in inv]
    try:
        import pyarrow.parquet as pq
        avail = set(pq.ParquetFile(cfg.holdings_path).schema.names)
        df = pd.read_parquet(cfg.holdings_path, columns=[c for c in raw if c in avail] or None)
    except Exception:
        df = pd.read_parquet(cfg.holdings_path)
    df = df.rename(columns={inv[c]: c for c in want if inv[c] in df.columns})

    need = ["security", "date", "close", "volume", "market_cap", "security_size"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns {missing} -- remap them in Config.col_map")

    df["date"] = pd.to_datetime(df["date"])
    df["yq"] = df["date"].dt.to_period("Q")
    if cfg.us_only and "isUs" in df.columns:
        df = df[df["isUs"].astype(bool)]
    if cfg.inv_type_codes is not None and "inv_type" in df.columns:
        df = df[df["inv_type"].astype(str).isin({str(c) for c in cfg.inv_type_codes})]

    for c in ("close", "volume", "market_cap", "position_value", "quarterly_ret",
              "future_1q_ret", "future_2q_ret", "future_3q_ret",
              "turn_small", "turn_mid", "turn_large", "active_weight"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("future_2q_ret", "future_3q_ret"):
        if c not in df.columns:
            df[c] = np.nan

    have_w = {"fund", "weight"} <= set(df.columns) and df["weight"].notna().any()
    have_b = "chg_pct" in df.columns and df["chg_pct"].notna().any()
    have_pv = "position_value" in df.columns and df["position_value"].notna().any()

    # ---- share change over q -> q+1, and the dollars it moves --------------------------
    if have_b:
        dsh = _autoscale(df["chg_pct"], "chg_pct")
        keep = dsh.notna()
        if cfg.drop_missing_position:
            sentinel = (dsh + 1.0).abs() < 1e-6
            print(f"[hold] {int(sentinel.sum()):,} rows ({sentinel.mean():.1%}) have "
                  "chg_pct = -100% (missing next-quarter position, not a sale) -- excluded "
                  "from the pressure measures only, the rows stay in the panel")
            keep &= ~sentinel
        df["_dsh"] = dsh.where(keep)
        df["_buy"] = (df["_dsh"] >= cfg.change_band).astype("float32").where(keep)
        df["_dollar"] = (df["_dsh"] * df["position_value"]) if have_pv else np.nan

    # ---- weight change over q -> q+1, raw and drift-free -------------------------------
    if have_w:
        w = _autoscale(df["weight"], "weight", thresh=0.02)
        df["_w"] = w
        df = df.sort_values(["fund", "security", "yq"])
        gfs = df.groupby(["fund", "security"], observed=True)
        w_next = gfs["_w"].shift(-1).where(gfs["yq"].shift(-1) == df["yq"] + 1)
        df["_dw"] = w_next - df["_w"]
        if "future_1q_ret" in df.columns:
            rs = df["future_1q_ret"].astype("float64")
            ok = df["_w"].notna() & rs.notna()
            gfq = df[ok].groupby(["fund", "yq"], observed=True)
            num = gfq.apply(lambda t: float((t["_w"] * rs.loc[t.index]).sum()))
            den = gfq["_w"].sum()
            rp = pd.Series(df.set_index(["fund", "yq"]).index.map(
                (num / den.where(den != 0)).rename("rp")), index=df.index).astype("float64")
            # what the weight becomes if the manager does nothing
            df["_dw_active"] = w_next - df["_w"] * (1.0 + rs) / (1.0 + rp)
        else:
            df["_dw_active"] = np.nan

    # ---- security-quarter aggregation --------------------------------------------------
    agg: Dict[str, tuple] = dict(
        close=("close", "first"), volume=("volume", "first"),
        mktcap=("market_cap", "first"), ret_q=("quarterly_ret", "first"),
        ret_next=("future_1q_ret", "first"), _f2=("future_2q_ret", "first"),
        _f3=("future_3q_ret", "first"), security_size=("security_size", "first"),
        n_holders=("security", "size"),
    )
    for c in ("turn_small", "turn_mid", "turn_large"):
        if c in df.columns:
            agg[c] = (c, "first")          # already a security-level total
    if "active_weight" in df.columns:
        agg["active_weight_mean"] = ("active_weight", "mean")
        df["_aw_abs"] = df["active_weight"].abs()
        agg["active_weight_absmean"] = ("_aw_abs", "mean")
    sq = df.groupby(["security", "yq"], observed=True).agg(**agg).reset_index()

    def _merge(sub):
        nonlocal sq
        sq = sq.merge(sub, on=["security", "yq"], how="left")

    if have_b:
        lab = df[df["_dsh"].notna()]
        b = lab.groupby(["security", "yq"], observed=True).agg(
            buy_frac=("_buy", "mean"), n_labelled=("_buy", "size")).reset_index()
        _merge(b)
        if have_pv:
            dd = lab.groupby(["security", "yq"], observed=True)["_dollar"]
            d = pd.DataFrame({"net_dollar": dd.sum(),
                              "gross_dollar": dd.apply(lambda s: float(np.abs(s).sum())),
                              "buy_dollar": dd.apply(lambda s: float(s.clip(lower=0).sum())),
                              }).reset_index()
            _merge(d)
            pv = lab.assign(_pvb=lab["_buy"] * lab["position_value"]).groupby(
                ["security", "yq"], observed=True).agg(
                _pvb=("_pvb", "sum"), _pv=("position_value", "sum")).reset_index()
            pv["dollar_buy_frac"] = pv["_pvb"] / pv["_pv"].where(pv["_pv"] > 0)
            _merge(pv[["security", "yq", "dollar_buy_frac"]])
    if have_w:
        ww = df[df["_dw"].notna()].groupby(["security", "yq"], observed=True)["_dw"]
        w1 = pd.DataFrame({"weight_chg": ww.mean(),
                           "gross_dw": ww.apply(lambda s: float(np.abs(s).sum())),
                           "buy_dw": ww.apply(lambda s: float(s.clip(lower=0).sum())),
                           }).reset_index()
        _merge(w1)
        aw = df[df["_dw_active"].notna()].groupby(
            ["security", "yq"], observed=True)["_dw_active"].mean().rename(
            "active_weight_chg").reset_index()
        _merge(aw)

    # ---- the derived pressure measures -------------------------------------------------
    if "net_dollar" in sq.columns:
        cap = sq["mktcap"].abs().where(sq["mktcap"].abs() > 0)
        sq["flow_pct_cap"] = sq["net_dollar"] / cap
        # NEW: of every dollar of reallocation in this name, what share was buying?
        sq["buy_dollar_ratio"] = sq["buy_dollar"] / sq["gross_dollar"].where(
            sq["gross_dollar"] > 0)
    if "gross_dw" in sq.columns:
        # NEW: same question in weight space
        sq["buy_weight_ratio"] = sq["buy_dw"] / sq["gross_dw"].where(sq["gross_dw"] > 0)

    # ---- turnover, from close/volume/market cap ----------------------------------------
    cap = sq["mktcap"].abs().where(sq["mktcap"].abs() > 0)
    as_shares, as_dollars = sq["volume"] * sq["close"] / cap, sq["volume"] / cap
    m_sh, m_dl = float(as_shares.median()), float(as_dollars.median())
    if np.isfinite(m_sh) and 0.005 < m_sh < 5:
        sq["turnover"], pick = as_shares, "volume is SHARE volume: volume*close/mktcap"
    elif np.isfinite(m_dl) and 0.005 < m_dl < 5:
        sq["turnover"], pick = as_dollars, "volume is DOLLAR volume: volume/mktcap"
    else:
        sq["turnover"], pick = as_shares, "NEITHER plausible -- CHECK UNITS"
    print(f"[turnover] median if share-volume {m_sh:.4f} | if dollar-volume {m_dl:.4f}")
    print(f"[turnover] using: {pick}")

    sq["log_mktcap"] = np.log(sq["mktcap"].abs() + 1.0)
    sq["log_price"] = np.log(sq["close"].abs() + 1e-6)
    tt = [c for c in ("turn_small", "turn_mid", "turn_large") if c in sq.columns]
    if len(tt) == 3:
        tot = sq[tt].sum(axis=1)
        sq["turn_large_share"] = sq["turn_large"] / tot.where(tot > 0)

    # ---- quarter index, lags, targets --------------------------------------------------
    qs = pd.PeriodIndex(sorted(sq["yq"].unique()), freq="Q")
    sq["qi"] = sq["yq"].map({q: i for i, q in enumerate(qs)}).astype("int32")
    sq = sq.sort_values(["security", "qi"]).reset_index(drop=True)
    g = sq.groupby("security", observed=True)
    qi = sq["qi"].to_numpy()

    lag_cols = []
    for k in range(1, cfg.vol_window):
        v, qq = g["ret_q"].shift(k), g["qi"].shift(k)
        sq[f"_r{k}"] = v.where(qq == qi - k)
        lag_cols.append(f"_r{k}")
    sq["vol_ret"] = sq[["ret_q"] + lag_cols].std(axis=1, ddof=1)

    # I_{q+1} quantities enter only through their lag
    for src, dst in (("weight_chg", "weight_chg_lag1"), ("buy_frac", "buy_frac_lag1"),
                     ("buy_weight_ratio", "buy_weight_ratio_lag1")):
        if src in sq.columns:
            v, qq = g[src].shift(1), g["qi"].shift(1)
            sq[dst] = v.where(qq == qi - 1)
    if not cfg.assume_fund_turnover_backward:
        for c in tt:
            v, qq = g[c].shift(1), g["qi"].shift(1)
            sq[c] = v.where(qq == qi - 1)
        if "turn_large_share" in sq.columns:
            v, qq = g["turn_large_share"].shift(1), g["qi"].shift(1)
            sq["turn_large_share"] = v.where(qq == qi - 1)
        print("[panel] *_fund_turnover lagged one quarter (assume_fund_turnover_backward=False)")

    v, qq = g["turnover"].shift(-1), g["qi"].shift(-1)
    sq["turnover_next"] = v.where(qq == qi + 1)
    sq["ret_next_2q"] = (1 + sq["ret_next"]) * (1 + sq["_f2"]) - 1
    sq["ret_next_3q"] = (1 + sq["ret_next"]) * (1 + sq["_f2"]) * (1 + sq["_f3"]) - 1
    sq = sq.drop(columns=lag_cols + ["_f2", "_f3"])

    feats_here = [f for f in FEATURES if f in sq.columns]
    sq = _winsor_by_q(sq, feats_here + ["turnover_next"], cfg.winsorize)

    if cfg.require_consecutive:
        brk = g["qi"].diff() != 1
        run = brk.groupby(sq["security"]).cumsum()
        seen = sq.groupby(["security", run]).cumcount() + 1
    else:
        seen = g.cumcount() + 1
    n0 = len(sq)
    sq = sq[seen >= cfg.min_quarters].reset_index(drop=True)
    sq["size_label"] = sq["security_size"].map(SIZE_LABEL).fillna("?")

    print(f"[panel] {n0:,} -> {len(sq):,} rows after the history filter | "
          f"{sq.security.nunique():,} securities | {sq.qi.max()+1} quarters")
    print("[panel] rows per security_size: "
          + "  ".join(f"{SIZE_LABEL.get(k, k)} {v:,}"
                      for k, v in sq["security_size"].value_counts().sort_index().items()))
    print("[panel] features present: " + "  ".join(
        f"{c} {sq[c].notna().mean():.0%}" for c in feats_here))
    print("[panel] pressure measures: " + "  ".join(
        f"{k} {sq[v].notna().mean():.0%}" for k, v in PRESSURE.items() if v in sq.columns))
    return sq


def feature_list(panel: pd.DataFrame) -> List[str]:
    return [f for f in FEATURES if f in panel.columns and panel[f].notna().any()]


def pressure_list(panel: pd.DataFrame) -> List[str]:
    return [k for k, v in PRESSURE.items() if v in panel.columns and panel[v].notna().any()]


# ------------------------------------------------------------------ model
def _make_estimator(cfg: Config):
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


def _train_y(tr: pd.DataFrame, target: str, cfg: Config) -> np.ndarray:
    """Training label, optionally made robust. Computed within each TRAINING quarter and from
    training rows alone, so no test-period value reaches it. Both options preserve within-
    quarter ordering, which is all rank-IC and a quintile sort ever use."""
    how = str(cfg.train_target_transform).lower()
    y = tr[target]
    if how in ("none", "", "raw"):
        pass
    elif how == "rank":
        y = tr.groupby("qi")[target].rank(pct=True)
    elif how == "winsor":
        p = cfg.train_winsor
        lo = tr.groupby("qi")[target].transform(lambda s: s.quantile(p))
        hi = tr.groupby("qi")[target].transform(lambda s: s.quantile(1 - p))
        y = y.clip(lo, hi)
    else:
        raise ValueError(f"train_target_transform must be none|winsor|rank, got {how!r}")
    return y.to_numpy("float32")


def fold_schedule(panel: pd.DataFrame, targets: List[str], cfg: Config) -> List[int]:
    """Fold end-points every target can run. Targets differ in availability -- turnover_next
    is NaN in the last quarter, ret_next_2q needs one more forward quarter -- and because the
    loop bound comes from `d.qi.max()`, one missing quarter can drop a whole END-POINT, i.e.
    `test_q` quarters at once. Intersecting keeps every target on the same test quarters."""
    ends = None
    for t in targets:
        d = panel.dropna(subset=[t])
        if d.empty:
            return []
        ok = {c for c in range(cfg.window_q, int(d.qi.max()) + 2, cfg.step)
              if ((d.qi >= c - cfg.window_q) & (d.qi < c - cfg.test_q)).sum() >= 300
              and ((d.qi >= c - cfg.test_q) & (d.qi < c)).sum() > 0}
        ends = ok if ends is None else (ends & ok)
    return sorted(ends or [])


def _rolling_predict(panel, feats, target, cfg, folds=None) -> pd.DataFrame:
    d = panel.dropna(subset=[target])
    if folds is None:
        folds = list(range(cfg.window_q, int(d.qi.max()) + 2, cfg.step))
    out = []
    for c in folds:
        tr = d[(d.qi >= c - cfg.window_q) & (d.qi < c - cfg.test_q)]
        te = d[(d.qi >= c - cfg.test_q) & (d.qi < c)]
        if len(tr) < 300 or len(te) == 0:
            continue
        m = _make_estimator(cfg)
        m.fit(tr[feats].to_numpy("float32"), _train_y(tr, target, cfg))
        keep = list(dict.fromkeys(["security", "qi", target, "ret_next"]))
        p = te[keep].copy()
        p["pred"] = m.predict(te[feats].to_numpy("float32"))
        out.append(p)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _score(d: pd.DataFrame, pred_col: str, target: str, cfg: Config) -> dict:
    """rank-IC against the target, and the quintile spread in RETURN over the horizon that
    matches the target. Overlapping horizons get a Newey-West t with h-1 lags."""
    h = RETURN_TARGETS.get(target, 1)
    ret_col = target if target in RETURN_TARGETS else "ret_next"
    blank = {"rank_IC": np.nan, "IC_t": np.nan, "ret_h_q": h, "Q1_ret_pct": np.nan,
             "Q5_ret_pct": np.nan, "Q5_Q1_pct": np.nan, "Q5_Q1_per_q": np.nan,
             "spread_t": np.nan, "n_quarters": 0, "n_rows": 0}
    if d.empty or pred_col not in d.columns:
        return blank
    x = d.dropna(subset=[pred_col, target])
    ic = (x.groupby("qi").apply(lambda t: t[pred_col].corr(t[target], method="spearman"))
          if len(x) else pd.Series(dtype=float))
    r = d.dropna(subset=[pred_col, ret_col]).copy()
    r["Q"] = r.groupby("qi")[pred_col].transform(lambda s: _qcut(s, cfg.n_quintiles))
    r = r.dropna(subset=["Q"])
    if r.empty:
        return blank
    per = r.groupby(["qi", "Q"])[ret_col].mean().unstack()
    if per.shape[1] < 2:
        return blank
    hi, lo = per.columns.max(), per.columns.min()
    sp = (per[hi] - per[lo]).dropna()
    if sp.empty:
        return blank
    return {"rank_IC": ic.mean(), "IC_t": _t(ic), "ret_h_q": h,
            "Q1_ret_pct": per[lo].mean() * 100, "Q5_ret_pct": per[hi].mean() * 100,
            "Q5_Q1_pct": sp.mean() * 100, "Q5_Q1_per_q": sp.mean() * 100 / h,
            "spread_t": _t(sp, lags=h - 1), "n_quarters": int(sp.size),
            "n_rows": int(len(r))}


def run_stratum(panel: pd.DataFrame, cfg: Config, targets: List[str],
                label: str = "", verbose: bool = True) -> pd.DataFrame:
    """Every target, one model on all features plus one per feature, inside ONE stratum.

    Quintiles are formed within the stratum, so a large cap is never ranked against a micro
    cap -- which is the whole point of stratifying.
    """
    feats = feature_list(panel)
    if not feats:
        return pd.DataFrame()
    specs = [("model:ALL", feats)] + [(f"model:{f}", [f]) for f in feats]

    if cfg.align_eval_sample:
        ok = np.ones(len(panel), bool)
        for c in targets + feats:
            if c in panel.columns:
                ok &= panel[c].notna().to_numpy()
        eval_keys = panel.loc[ok, ["security", "qi"]]
    else:
        eval_keys = None

    def _restrict(d):
        return d if (eval_keys is None or d.empty) else d.merge(eval_keys,
                                                               on=["security", "qi"])

    folds = fold_schedule(panel, targets, cfg) if cfg.align_eval_sample else None
    rows = []
    for target in targets:
        test_qi = None
        for name, fs in specs:
            P = _restrict(_rolling_predict(panel, fs, target, cfg, folds))
            if test_qi is None and not P.empty:
                test_qi = set(P["qi"].unique())
            rows.append({"stratum": label, "target": target, "model": name,
                         **_score(P, "pred", target, cfg)})
        if test_qi:
            base = _restrict(panel[panel.qi.isin(test_qi)]).dropna(subset=[target])
            for f in feats:
                rows.append({"stratum": label, "target": target, "model": f"raw:{f}",
                             **_score(base, f, target, cfg)})
    out = pd.DataFrame(rows)
    if verbose and len(out):
        print(f"  {label:<8} {len(panel):>8,} rows | {len(feats)} features | "
              f"{len(folds or []) if folds is not None else '?'} folds | "
              f"{out.n_quarters.max()} test quarters")
    return out


def run_stratified(panel: pd.DataFrame, cfg: Config = None, targets: List[str] = None,
                   verbose: bool = True) -> pd.DataFrame:
    """Run every stratum, plus the pooled sample for comparison.

    `targets` defaults to the return targets plus turnover_next plus every pressure measure
    the panel can supply. Pass a shorter list to keep the run small.
    """
    cfg = cfg or Config()
    if targets is None:
        targets = (["turnover_next", "ret_next"]
                   + [PRESSURE[k] for k in pressure_list(panel)])
    targets = [t for t in targets if t in panel.columns and panel[t].notna().sum() > 500]
    if verbose:
        print(f"targets: {targets}\nstrata: {[SIZE_LABEL.get(s, s) for s in cfg.strata]}"
              " (+ pooled)\n")
    parts = []
    for s in cfg.strata:
        sub = panel[panel.security_size == s]
        if len(sub) < cfg.min_stratum_rows:
            print(f"  {SIZE_LABEL.get(s, s):<8} only {len(sub):,} rows -- skipped "
                  f"(min_stratum_rows={cfg.min_stratum_rows})")
            continue
        parts.append(run_stratum(sub, cfg, targets, SIZE_LABEL.get(s, str(s)), verbose))
    parts.append(run_stratum(panel, cfg, targets, "pooled", verbose))
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    out.insert(0, "learner", cfg.model)
    out.insert(1, "train_y", cfg.train_target_transform)
    return out


def summary(table: pd.DataFrame, model: str = "model:ALL") -> pd.DataFrame:
    """One row per (stratum, target): the headline IC and spread for a single model."""
    d = table[table.model == model]
    return d.pivot_table(index="target", columns="stratum",
                         values=["rank_IC", "Q5_Q1_per_q", "spread_t"]).round(4)


def beats_naive(table: pd.DataFrame) -> pd.DataFrame:
    """model:ALL against the best single characteristic, per (stratum, target).

    Compared on |rank_IC|, not the signed value. A raw sort's SIGN is arbitrary -- it is
    fixed by the direction of the characteristic, not learned -- so a rival with IC = -0.47
    is exactly as good as one with +0.47: rank the other way and you have it. Subtracting the
    signed numbers would credit the model with 0.94 of edge it does not have.
    """
    rows = []
    for (st, tg), d in table.groupby(["stratum", "target"]):
        m = d[d.model == "model:ALL"]
        raw = d[d.model.str.startswith("raw:")].dropna(subset=["rank_IC"])
        if m.empty or raw.empty or pd.isna(m.rank_IC.iloc[0]):
            continue
        best = raw.loc[raw.rank_IC.abs().idxmax()]
        ic_m, ic_r = float(m.rank_IC.iloc[0]), float(best.rank_IC)
        rows.append({"stratum": st, "target": tg,
                     "IC_model": round(ic_m, 4), "absIC_model": round(abs(ic_m), 4),
                     "best_raw": best.model, "absIC_best_raw": round(abs(ic_r), 4),
                     "edge": round(abs(ic_m) - abs(ic_r), 4)})
    out = pd.DataFrame(rows)
    return (out.sort_values(["target", "stratum"]).reset_index(drop=True)
            if len(out) else out)


def check_version(verbose=True):
    import os
    if verbose:
        print(f"stratified_study {__version__}  |  {os.path.abspath(__file__)}")
    return __version__
