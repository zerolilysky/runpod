"""size_block_study.py -- turnover and buying pressure in nine size blocks.

The nine blocks are the Cartesian product

    security_size in {small, mid, large}
    fund_size     in {small, mid, large}

One row is one (security, quarter, fund_size).  Market variables belong to the security and
quarter; holder-derived variables are recomputed using only funds in that row's fund-size
bucket.  Every model, training-target rank, prediction sort, and score is then computed
separately inside one security-size x fund-size block.

The feature set is exactly the same 14-characteristic set as ``stratified_study.py``.
Only the panel grouping changes from security size alone to security size x fund size.

This file owns the different panel construction.  It deliberately reuses the estimator,
rolling split, and scoring functions from ``stratified_study.py`` so the old three-bucket
study and this nine-block study are methodologically comparable.

Timing follows ``stratified_study.py``.  Quantities measured over q -> q+1 (share change and
portfolio-weight change) are targets or enter only through a one-quarter lag.  The three
``security_*_fund_turnover`` columns are used at q when
``assume_fund_turnover_backward=True``; set it to False if they actually describe q -> q+1.

Usage
    import size_block_study as S
    cfg = S.Config(holdings_path=...)
    panel = S.build_panel(cfg)
    table = S.run_blocks(panel, cfg, targets=["turnover_next", "ret_next"])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd

import stratified_study as base

__version__ = "2026.08.17.1"

SIZE_LABEL = {0: "small", 1: "mid", 2: "large"}

# Identical to stratified_study.FEATURES.  The nine-block study changes grouping, not features.
FEATURES = [
    "turnover", "log_mktcap", "ret_q", "vol_ret", "log_price",
    "active_weight_mean", "active_weight_absmean",
    "turn_small", "turn_mid", "turn_large", "turn_large_share",
    "weight_chg_lag1", "buy_frac_lag1", "buy_weight_ratio_lag1",
]

PRESSURE = base.PRESSURE
RETURN_TARGETS = base.RETURN_TARGETS


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
        "security_size": "security_size", "fund_size": "fund_size",
        "active_weight": "active_weight",
        "security_small_fund_turnover": "turn_small",
        "security_mid_fund_turnover": "turn_mid",
        "security_large_fund_turnover": "turn_large",
    })
    inv_type_codes: tuple = (401,)
    us_only: bool = True

    # True means the three fund-turnover columns describe q-1 -> q and are known at q.
    assume_fund_turnover_backward: bool = True

    change_band: float = 0.01
    drop_missing_position: bool = True
    vol_window: int = 4
    winsorize: float = 0.01
    min_quarters: int = 8
    require_consecutive: bool = False

    security_strata: tuple = (0, 1, 2)
    fund_strata: tuple = (0, 1, 2)
    min_block_rows: int = 2000

    window_q: int = 28
    test_q: int = 8
    step: int = 8
    align_eval_sample: bool = True
    # True gives all non-thin blocks the same test quarters, making the 3x3 cells comparable.
    align_block_folds: bool = True
    n_quintiles: int = 5

    model: str = "hgb"                 # hgb | linear
    train_target_transform: str = "rank"   # none | winsor | rank
    train_winsor: float = 0.01
    max_iter: int = 300
    learning_rate: float = 0.06
    max_depth: int = 4
    seed: int = 0


# ------------------------------------------------------------------ input and helpers
def _read_holdings(cfg: Config) -> pd.DataFrame:
    """Read only mapped columns that actually exist, then rename them to internal names."""
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(cfg.holdings_path).schema.names)
        raw_columns = [c for c in cfg.col_map if c in available]
        df = pd.read_parquet(cfg.holdings_path, columns=raw_columns or None)
    except Exception:
        df = pd.read_parquet(cfg.holdings_path)
    return df.rename(columns={c: cfg.col_map[c] for c in df.columns if c in cfg.col_map})


def _winsor_by_block_q(df: pd.DataFrame, cols: List[str], p: float) -> pd.DataFrame:
    """Cross-sectional clipping inside each quarter and each of the nine blocks."""
    if not p:
        return df
    keys = ["security_size", "fund_size", "yq"]
    for c in cols:
        if c in df.columns and df[c].notna().any():
            lo = df.groupby(keys, observed=True)[c].transform(lambda s: s.quantile(p))
            hi = df.groupby(keys, observed=True)[c].transform(lambda s: s.quantile(1 - p))
            df[c] = df[c].clip(lo, hi).astype("float32")
    return df


def _size_name(value) -> str:
    return SIZE_LABEL.get(value, str(value))


def _block_name(security_size, fund_size) -> str:
    return f"security_{_size_name(security_size)}__fund_{_size_name(fund_size)}"


# ------------------------------------------------------------------ panel
def build_panel(cfg: Config = None) -> pd.DataFrame:
    """Build one row per (security, quarter, fund_size), ready for nine separate runs."""
    cfg = cfg or Config()
    df = _read_holdings(cfg)

    required = ["security", "fund", "date", "close", "volume", "market_cap",
                "quarterly_ret", "future_1q_ret", "security_size", "fund_size"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns {missing} -- remap them in Config.col_map")

    df["date"] = pd.to_datetime(df["date"])
    df["yq"] = df["date"].dt.to_period("Q")
    if cfg.us_only and "isUs" in df.columns:
        df = df[df["isUs"].astype(bool)]
    if cfg.inv_type_codes is not None and "inv_type" in df.columns:
        df = df[df["inv_type"].astype(str).isin({str(c) for c in cfg.inv_type_codes})]

    numeric = (
        "close", "volume", "market_cap", "position_value", "quarterly_ret",
        "future_1q_ret", "future_2q_ret", "future_3q_ret", "security_size", "fund_size",
        "turn_small", "turn_mid", "turn_large", "active_weight",
    )
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("future_2q_ret", "future_3q_ret"):
        if c not in df.columns:
            df[c] = np.nan

    df = df[df["security_size"].isin(cfg.security_strata)
            & df["fund_size"].isin(cfg.fund_strata)]
    if df.empty:
        raise ValueError("no rows remain after the security_size and fund_size filters")
    duplicate = df.duplicated(["fund", "security", "yq"], keep=False)
    if duplicate.any():
        raise ValueError(f"found {int(duplicate.sum()):,} duplicate fund-security-quarter rows")

    # A fund should have one size classification in a quarter.  Failing loudly prevents one
    # holding row from being counted in the wrong block while the rest of its fund is elsewhere.
    fund_q_sizes = df.groupby(["fund", "yq"], observed=True)["fund_size"].nunique()
    if (fund_q_sizes > 1).any():
        n_bad = int((fund_q_sizes > 1).sum())
        raise ValueError(f"fund_size is not unique within {n_bad:,} fund-quarter(s)")
    security_q_sizes = df.groupby(["security", "yq"], observed=True)[
        "security_size"].nunique()
    if (security_q_sizes > 1).any():
        n_bad = int((security_q_sizes > 1).sum())
        raise ValueError(f"security_size is not unique within {n_bad:,} security-quarter(s)")

    have_w = {"weight", "fund"} <= set(df.columns) and df["weight"].notna().any()
    have_b = "chg_pct" in df.columns and df["chg_pct"].notna().any()
    have_pv = "position_value" in df.columns and df["position_value"].notna().any()

    # Share change and dollar flow over q -> q+1: block-specific targets, never raw features.
    if have_b:
        dsh = base._autoscale(df["chg_pct"], "chg_pct")
        keep = dsh.notna()
        if cfg.drop_missing_position:
            sentinel = (dsh + 1.0).abs() < 1e-6
            print(f"[hold] {int(sentinel.sum()):,} rows ({sentinel.mean():.1%}) have "
                  "chg_pct = -100% -- excluded from pressure measures only")
            keep &= ~sentinel
        df["_dsh"] = dsh.where(keep)
        df["_buy"] = (df["_dsh"] >= cfg.change_band).astype("float32").where(keep)
        df["_dollar"] = df["_dsh"] * df["position_value"] if have_pv else np.nan

    # Weight change is computed before grouping so a fund that changes size bucket is assigned
    # to its known bucket at q, the start of the q -> q+1 decision window.
    if have_w:
        df["_w"] = base._autoscale(df["weight"], "weight", thresh=0.02)
        df = df.sort_values(["fund", "security", "yq"])
        gfs = df.groupby(["fund", "security"], observed=True)
        w_next = gfs["_w"].shift(-1).where(gfs["yq"].shift(-1) == df["yq"] + 1)
        df["_dw"] = w_next - df["_w"]
        if "future_1q_ret" in df.columns:
            rs = df["future_1q_ret"].astype("float64")
            ok = df["_w"].notna() & rs.notna()
            df["_wret"] = df["_w"] * rs
            gfq = df[ok].groupby(["fund", "yq"], observed=True)
            num = gfq["_wret"].sum()
            den = gfq["_w"].sum()
            fund_ret = (num / den.where(den != 0)).rename("fund_ret")
            keys = pd.MultiIndex.from_arrays([df["fund"], df["yq"]])
            rp = pd.Series(keys.map(fund_ret), index=df.index, dtype="float64")
            df["_dw_active"] = w_next - df["_w"] * (1.0 + rs) / (1.0 + rp)
        else:
            df["_dw_active"] = np.nan

    # Market variables are built once at security-quarter level.  Their lags and forward
    # targets must not disappear merely because one fund-size bucket had no holder that quarter.
    market_agg: Dict[str, tuple] = dict(
        close=("close", "first"), volume=("volume", "first"),
        mktcap=("market_cap", "first"), ret_q=("quarterly_ret", "first"),
        ret_next=("future_1q_ret", "first"), _f2=("future_2q_ret", "first"),
        _f3=("future_3q_ret", "first"), security_size=("security_size", "first"),
    )
    for c in ("turn_small", "turn_mid", "turn_large"):
        if c in df.columns:
            market_agg[c] = (c, "first")
    market = df.groupby(["security", "yq"], observed=True).agg(**market_agg).reset_index()

    cap = market["mktcap"].abs().where(market["mktcap"].abs() > 0)
    as_shares = market["volume"] * market["close"] / cap
    as_dollars = market["volume"] / cap
    m_sh, m_dl = float(as_shares.median()), float(as_dollars.median())
    if np.isfinite(m_sh) and 0.005 < m_sh < 5:
        market["turnover"], pick = as_shares, "volume is SHARE volume: volume*close/mktcap"
    elif np.isfinite(m_dl) and 0.005 < m_dl < 5:
        market["turnover"], pick = as_dollars, "volume is DOLLAR volume: volume/mktcap"
    else:
        market["turnover"], pick = as_shares, "NEITHER plausible -- CHECK UNITS"
    print(f"[turnover] median if share-volume {m_sh:.4f} | if dollar-volume {m_dl:.4f}")
    print(f"[turnover] using: {pick}")

    market["log_mktcap"] = np.log(market["mktcap"].abs() + 1.0)
    market["log_price"] = np.log(market["close"].abs() + 1e-6)
    turnover_features = [c for c in ("turn_small", "turn_mid", "turn_large")
                         if c in market.columns]
    if len(turnover_features) == 3:
        total = market[turnover_features].sum(axis=1)
        market["turn_large_share"] = market["turn_large"] / total.where(total > 0)
    quarters = pd.PeriodIndex(sorted(market["yq"].unique()), freq="Q")
    market["qi"] = market["yq"].map({q: i for i, q in enumerate(quarters)}).astype("int32")
    market = market.sort_values(["security", "qi"]).reset_index(drop=True)
    gm = market.groupby("security", observed=True)

    lag_cols = []
    for k in range(1, cfg.vol_window):
        value, q_lag = gm["ret_q"].shift(k), gm["qi"].shift(k)
        name = f"_r{k}"
        market[name] = value.where(q_lag == market["qi"] - k)
        lag_cols.append(name)
    market["vol_ret"] = market[["ret_q"] + lag_cols].std(axis=1, ddof=1)
    value, q_next = gm["turnover"].shift(-1), gm["qi"].shift(-1)
    market["turnover_next"] = value.where(q_next == market["qi"] + 1)
    market["ret_next_2q"] = (1 + market["ret_next"]) * (1 + market["_f2"]) - 1
    market["ret_next_3q"] = ((1 + market["ret_next"]) * (1 + market["_f2"])
                              * (1 + market["_f3"]) - 1)
    market = market.drop(columns=lag_cols + ["_f2", "_f3"])

    # Holder-derived quantities are aggregated inside (security, quarter, fund_size).
    block_keys = ["security", "yq", "fund_size"]
    block_agg: Dict[str, tuple] = {"n_holders": ("fund", "size")}
    if "active_weight" in df.columns:
        df["_aw_abs"] = df["active_weight"].abs()
        block_agg.update(active_weight_mean=("active_weight", "mean"),
                         active_weight_absmean=("_aw_abs", "mean"))
    block = df.groupby(block_keys, observed=True).agg(**block_agg).reset_index()

    def _merge(values: pd.DataFrame):
        nonlocal block
        block = block.merge(values, on=block_keys, how="left", validate="one_to_one")

    if have_b:
        labelled = df[df["_dsh"].notna()]
        breadth = labelled.groupby(block_keys, observed=True).agg(
            buy_frac=("_buy", "mean"), n_labelled=("_buy", "size")).reset_index()
        _merge(breadth)
        if have_pv:
            dollar = labelled.groupby(block_keys, observed=True)["_dollar"]
            flows = pd.DataFrame({
                "net_dollar": dollar.sum(),
                "gross_dollar": dollar.apply(lambda s: float(np.abs(s).sum())),
                "buy_dollar": dollar.apply(lambda s: float(s.clip(lower=0).sum())),
            }).reset_index()
            _merge(flows)
            weighted = labelled.assign(_pvb=labelled["_buy"] * labelled["position_value"])
            weighted = weighted.groupby(block_keys, observed=True).agg(
                _pvb=("_pvb", "sum"), _pv=("position_value", "sum")).reset_index()
            weighted["dollar_buy_frac"] = weighted["_pvb"] / weighted["_pv"].where(
                weighted["_pv"] > 0)
            _merge(weighted[block_keys + ["dollar_buy_frac"]])
    if have_w:
        weight_group = df[df["_dw"].notna()].groupby(block_keys, observed=True)["_dw"]
        weight_flow = pd.DataFrame({
            "weight_chg": weight_group.mean(),
            "gross_dw": weight_group.apply(lambda s: float(np.abs(s).sum())),
            "buy_dw": weight_group.apply(lambda s: float(s.clip(lower=0).sum())),
        }).reset_index()
        _merge(weight_flow)
        active = df[df["_dw_active"].notna()].groupby(
            block_keys, observed=True)["_dw_active"].mean().rename(
            "active_weight_chg").reset_index()
        _merge(active)

    panel = block.merge(market, on=["security", "yq"], how="left", validate="many_to_one")
    if "net_dollar" in panel.columns:
        cap = panel["mktcap"].abs().where(panel["mktcap"].abs() > 0)
        panel["flow_pct_cap"] = panel["net_dollar"] / cap
        panel["buy_dollar_ratio"] = panel["buy_dollar"] / panel["gross_dollar"].where(
            panel["gross_dollar"] > 0)
    if "gross_dw" in panel.columns:
        panel["buy_weight_ratio"] = panel["buy_dw"] / panel["gross_dw"].where(
            panel["gross_dw"] > 0)

    panel = panel.sort_values(["security", "fund_size", "qi"]).reset_index(drop=True)
    gb = panel.groupby(["security", "fund_size"], observed=True)
    qi = panel["qi"]
    for source, lagged in (("weight_chg", "weight_chg_lag1"),
                           ("buy_frac", "buy_frac_lag1"),
                           ("buy_weight_ratio", "buy_weight_ratio_lag1")):
        if source in panel.columns:
            value, q_lag = gb[source].shift(1), gb["qi"].shift(1)
            panel[lagged] = value.where(q_lag == qi - 1)

    if not cfg.assume_fund_turnover_backward:
        for source in turnover_features + (["turn_large_share"]
                                           if "turn_large_share" in panel.columns else []):
            value, q_lag = gb[source].shift(1), gb["qi"].shift(1)
            panel[source] = value.where(q_lag == qi - 1)
        print("[panel] *_fund_turnover lagged one quarter "
              "(assume_fund_turnover_backward=False)")

    present_features = [f for f in FEATURES if f in panel.columns]
    panel = _winsor_by_block_q(panel, present_features + ["turnover_next"], cfg.winsorize)

    if cfg.require_consecutive:
        broken = gb["qi"].diff() != 1
        run = broken.groupby([panel["security"], panel["fund_size"]]).cumsum()
        seen = panel.groupby([panel["security"], panel["fund_size"], run]).cumcount() + 1
    else:
        seen = gb.cumcount() + 1
    n_before = len(panel)
    panel = panel[seen >= cfg.min_quarters].reset_index(drop=True)

    panel["security_label"] = panel["security_size"].map(SIZE_LABEL).fillna("?")
    panel["fund_label"] = panel["fund_size"].map(SIZE_LABEL).fillna("?")
    panel["block"] = [_block_name(s, f)
                      for s, f in zip(panel["security_size"], panel["fund_size"])]

    print(f"[panel] {n_before:,} -> {len(panel):,} rows after the history filter | "
          f"{panel.security.nunique():,} securities | {panel.qi.max() + 1} quarters")
    print("[panel] rows per security_size x fund_size:")
    print(block_counts(panel).to_string())
    print("[panel] features present: " + "  ".join(
        f"{c} {panel[c].notna().mean():.0%}" for c in feature_list(panel)))
    print("[panel] pressure measures: " + "  ".join(
        f"{k} {panel[v].notna().mean():.0%}" for k, v in PRESSURE.items()
        if v in panel.columns))
    return panel


def feature_list(panel: pd.DataFrame) -> List[str]:
    return [f for f in FEATURES if f in panel.columns and panel[f].notna().any()]


def pressure_list(panel: pd.DataFrame) -> List[str]:
    return [k for k, v in PRESSURE.items() if v in panel.columns and panel[v].notna().any()]


def block_counts(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows in each security-size x fund-size cell."""
    return (panel.groupby(["security_label", "fund_label"], observed=True).size()
            .unstack(fill_value=0).reindex(index=list(SIZE_LABEL.values()),
                                          columns=list(SIZE_LABEL.values()), fill_value=0))


# ------------------------------------------------------------------ models and scores
def _score(values: pd.DataFrame, pred_col: str, target: str, cfg: Config) -> dict:
    """Score as in the source study; a constant cross-section has undefined IC, recorded NaN."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="An input array is constant;.*")
        return base._score(values, pred_col, target, cfg)


def _run_block(panel: pd.DataFrame, cfg: Config, targets: List[str], security_size,
               fund_size, folds=None, verbose: bool = True) -> pd.DataFrame:
    """Run all-feature, one-feature, and raw-feature specifications in one block."""
    features = feature_list(panel)
    if not features:
        return pd.DataFrame()
    specs = [("model:ALL", features)] + [(f"model:{f}", [f]) for f in features]

    if cfg.align_eval_sample:
        complete = np.ones(len(panel), dtype=bool)
        for c in targets + features:
            if c in panel.columns:
                complete &= panel[c].notna().to_numpy()
        eval_keys = panel.loc[complete, ["security", "qi"]]
    else:
        eval_keys = None

    def _restrict(values: pd.DataFrame) -> pd.DataFrame:
        if eval_keys is None or values.empty:
            return values
        return values.merge(eval_keys, on=["security", "qi"], validate="one_to_one")

    if folds is None and cfg.align_eval_sample:
        folds = base.fold_schedule(panel, targets, cfg)

    rows = []
    for target in targets:
        test_qi = None
        for model_name, model_features in specs:
            pred = _restrict(base._rolling_predict(panel, model_features, target, cfg, folds))
            if test_qi is None and not pred.empty:
                test_qi = set(pred["qi"].unique())
            rows.append({"security_size": security_size, "fund_size": fund_size,
                         "security_label": _size_name(security_size),
                         "fund_label": _size_name(fund_size),
                         "block": _block_name(security_size, fund_size),
                         "target": target, "model": model_name,
                         **_score(pred, "pred", target, cfg)})
        if test_qi:
            raw = _restrict(panel[panel["qi"].isin(test_qi)]).dropna(subset=[target])
            for feature in features:
                rows.append({"security_size": security_size, "fund_size": fund_size,
                             "security_label": _size_name(security_size),
                             "fund_label": _size_name(fund_size),
                             "block": _block_name(security_size, fund_size),
                             "target": target, "model": f"raw:{feature}",
                             **_score(raw, feature, target, cfg)})

    out = pd.DataFrame(rows)
    if verbose and not out.empty:
        print(f"  sec={_size_name(security_size):<5} fund={_size_name(fund_size):<5} "
              f"{len(panel):>8,} rows | {len(features)} features | "
              f"{len(folds or []) if folds is not None else '?'} folds | "
              f"{out.n_quarters.max()} test quarters")
    return out


def run_blocks(panel: pd.DataFrame, cfg: Config = None, targets: List[str] = None,
               verbose: bool = True) -> pd.DataFrame:
    """Run the Cartesian product of configured security and fund size strata."""
    cfg = cfg or Config()
    if targets is None:
        targets = (["turnover_next", "ret_next"]
                   + [PRESSURE[k] for k in pressure_list(panel)])
    targets = [t for t in targets if t in panel.columns and panel[t].notna().sum() > 500]
    if not targets:
        raise ValueError("none of the requested targets has at least 500 observations")

    active = []
    for security_size in cfg.security_strata:
        for fund_size in cfg.fund_strata:
            sub = panel[(panel["security_size"] == security_size)
                        & (panel["fund_size"] == fund_size)]
            if len(sub) < cfg.min_block_rows:
                if verbose:
                    print(f"  sec={_size_name(security_size):<5} "
                          f"fund={_size_name(fund_size):<5} only {len(sub):,} rows -- skipped "
                          f"(min_block_rows={cfg.min_block_rows})")
                continue
            active.append((security_size, fund_size, sub))

    if not active:
        raise ValueError("all nine blocks are thinner than min_block_rows")

    common_folds = None
    if cfg.align_block_folds:
        schedules = [set(base.fold_schedule(sub, targets, cfg)) for _, _, sub in active]
        common_folds = sorted(set.intersection(*schedules)) if schedules else []
        if not common_folds:
            raise ValueError("no rolling fold is available in every non-thin block; set "
                             "align_block_folds=False or inspect block coverage")

    if verbose:
        print(f"targets: {targets}")
        print(f"blocks: {len(active)} of "
              f"{len(cfg.security_strata) * len(cfg.fund_strata)}")
        if common_folds is not None:
            print(f"common fold endpoints: {common_folds}\n")

    parts = [_run_block(sub, cfg, targets, security_size, fund_size,
                        folds=common_folds, verbose=verbose)
             for security_size, fund_size, sub in active]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "learner", cfg.model)
    out.insert(1, "train_y", cfg.train_target_transform)
    return out


def block_matrix(table: pd.DataFrame, target: str, value: str = "rank_IC",
                 model: str = "model:ALL") -> pd.DataFrame:
    """A readable 3x3 matrix for one target, metric, and model specification."""
    values = table[(table["target"] == target) & (table["model"] == model)]
    return (values.pivot_table(index="security_label", columns="fund_label", values=value)
            .reindex(index=list(SIZE_LABEL.values()),
                     columns=list(SIZE_LABEL.values())).round(4))


def summary(table: pd.DataFrame, model: str = "model:ALL") -> pd.DataFrame:
    """One tidy row per target and size block for the headline model."""
    cols = ["target", "security_label", "fund_label", "rank_IC", "IC_t",
            "Q5_Q1_per_q", "spread_t", "n_quarters", "n_rows"]
    return (table.loc[table["model"] == model, cols]
            .sort_values(["target", "security_label", "fund_label"]).reset_index(drop=True)
            .round(4))


def beats_naive(table: pd.DataFrame) -> pd.DataFrame:
    """All-feature model minus the best raw characteristic in absolute rank-IC."""
    rows = []
    keys = ["security_size", "fund_size", "security_label", "fund_label", "target"]
    for key, values in table.groupby(keys, observed=True):
        model = values[values["model"] == "model:ALL"]
        raw = values[values["model"].str.startswith("raw:")].dropna(subset=["rank_IC"])
        if model.empty or raw.empty or pd.isna(model["rank_IC"].iloc[0]):
            continue
        best = raw.loc[raw["rank_IC"].abs().idxmax()]
        ic_model = float(model["rank_IC"].iloc[0])
        ic_raw = float(best["rank_IC"])
        row = dict(zip(keys, key))
        row.update(IC_model=round(ic_model, 4), absIC_model=round(abs(ic_model), 4),
                   best_raw=best["model"], absIC_best_raw=round(abs(ic_raw), 4),
                   edge=round(abs(ic_model) - abs(ic_raw), 4))
        rows.append(row)
    return (pd.DataFrame(rows).sort_values(["target", "security_size", "fund_size"])
            .reset_index(drop=True) if rows else pd.DataFrame())


def check_version(verbose: bool = True) -> str:
    import os

    if verbose:
        print(f"size_block_study {__version__}  |  {os.path.abspath(__file__)}")
        print(f"reusing stratified_study {base.__version__}  |  {os.path.abspath(base.__file__)}")
    return __version__
