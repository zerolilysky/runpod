"""Ownership level and disagreement signals -- a small, fast, standalone module.

Split out of signals_perf because these two families need NONE of its expensive
machinery. inst_own and disp_* are pure cross-sectional measures at t: no lagged
active weight, no strict quarter joins, no expanding time-series z. One groupby
over the held rows and it is done, so this runs in a fraction of the time.

OWNERSHIP LEVEL -- how much of the company these institutions hold
  inst_own    sum(position_value / market_cap)   the collective stake
  max_own     the largest single holder's stake
  mean_own    average stake
  top5_own    the five largest stakes combined   (concentration)
  hhi_own     sum of squared stakes              (concentration, Herfindahl)

DISAGREEMENT -- how differently funds position in the name
  disp_aw       sd of active_weight across holders
  sum_abs_aw    sum |active_weight|      total tilt in either direction
  mean_abs_aw   mean |active_weight|     the same, per holder
  disp_z_aw     sd of the within-fund z  (removes each fund's own scale)
  disp_aw_cv    disp_aw / mean|active_weight|   relative disagreement

The two are conceptually independent: `own` is about WEIGHT of conviction,
`disp` about AGREEMENT. The interesting question is the interaction -- does the
ownership signal work better where funds disagree? -- which conditional_sort()
answers by evaluating `own` inside disagreement buckets.

Usage
-----
    import own_disp as OD
    sq  = OD.build(OD.Config(root="/path/to/root"))
    perf = OD.evaluate(sq)                       # every signal, every horizon
    cond = OD.conditional_sort(sq, "inst_own", "disp_aw", n_buckets=5)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

# tested loaders and inference from the big module -- no need to duplicate them
from signals_perf import (Config as _BaseConfig, load_panels, load_returns,
                          decile_spread, performance, newey_west_t, _bucket,
                          _sample_mask, RET_COL)

OWN_SIGNALS = ["inst_own", "max_own", "mean_own", "top5_own", "hhi_own"]
DISP_SIGNALS = ["disp_aw", "sum_abs_aw", "mean_abs_aw", "disp_z_aw", "disp_aw_cv"]
ALL = OWN_SIGNALS + DISP_SIGNALS


@dataclass
class Config(_BaseConfig):
    """Same knobs as signals_perf.Config; only the signal list differs."""
    signals: tuple = tuple(ALL)


# ================================================================= BUILD
def build(cfg: Config = None, panel: pd.DataFrame = None,
          returns: pd.DataFrame = None, verbose: bool = True) -> pd.DataFrame:
    """security x quarter frame with the ownership and disagreement signals.

    Pass an already-loaded `panel` / `returns` to skip the file reads.
    """
    cfg = cfg or Config()
    panel = load_panels(cfg) if panel is None else panel
    returns = load_returns(cfg) if returns is None else returns

    df = panel
    ok = (df["position_value"] > 0) & (df["market_cap"] > 0)
    df = df.loc[ok, ["fund", "security", "yq", "position_value", "market_cap"]].copy()

    # active weight: real minus a cap-weighted version of the fund's own book,
    # both normalised over the same holdings so the difference sums to zero
    g = df.groupby(["fund", "yq"], sort=False)
    df["active_weight"] = (df["position_value"] / g["position_value"].transform("sum")
                           - df["market_cap"] / g["market_cap"].transform("sum"))
    # each fund's tilt on its own scale, so a concentrated fund does not dominate
    gz = df.groupby(["fund", "yq"], sort=False)["active_weight"]
    mu, sd = gz.transform("mean"), gz.transform("std")
    df["z_aw"] = (df["active_weight"] - mu) / sd.where(sd > 0)

    df["own_frac"] = df["position_value"] / df["market_cap"]
    df["abs_aw"] = df["active_weight"].abs()

    def _top5(x):
        return float(np.sort(x.to_numpy())[-5:].sum())

    sq = df.groupby(["security", "yq"], observed=True).agg(
        n_funds=("fund", "size"),
        # ---- ownership level ----
        inst_own=("own_frac", "sum"),
        max_own=("own_frac", "max"),
        mean_own=("own_frac", "mean"),
        top5_own=("own_frac", _top5),
        hhi_own=("own_frac", lambda x: float(np.square(x.to_numpy()).sum())),
        # ---- disagreement ----
        disp_aw=("active_weight", "std"),
        sum_abs_aw=("abs_aw", "sum"),
        mean_abs_aw=("abs_aw", "mean"),
        disp_z_aw=("z_aw", "std"),
        market_cap=("market_cap", "first"),
    ).reset_index()

    # relative disagreement: raw sd grows mechanically with the size of the tilts
    sq["disp_aw_cv"] = sq["disp_aw"] / sq["mean_abs_aw"].where(sq["mean_abs_aw"] > 0)

    sq = sq.merge(returns.drop(columns=["qi"], errors="ignore"),
                  on=["security", "yq"], how="left")
    if verbose:
        print(f"[own_disp] {len(sq):,} security-quarters | "
              f"{sq.security.nunique():,} securities x {sq.yq.nunique()} quarters | "
              f"return match {sq['quarterly_ret'].notna().mean():.1%}")
    return sq


# ================================================================= EVALUATE
def evaluate(sq: pd.DataFrame, cfg: Config = None, signals: List[str] = None,
             horizons=(1, 2, 3), verbose: bool = True) -> pd.DataFrame:
    """Long-short decile performance for each signal, horizon and sample."""
    cfg = cfg or Config()
    signals = [c for c in (signals or ALL) if c in sq.columns]
    split = pd.Period(cfg.split, freq="Q")
    rows = []
    for sig in signals:
        for h in horizons:
            sp = decile_spread(sq, sig, h, cfg)
            for sample in ("discovery", "validation", "all"):
                r = performance(sp[_sample_mask(sp.index, sample, split)], h, cfg)
                r.update(signal=sig, horizon=h, sample=sample)
                rows.append(r)
    perf = pd.DataFrame(rows)
    front = ["signal", "horizon", "sample", "n_quarters", "mean_q", "t_nw",
             "hit", "sharpe_ann", "ann_return", "max_drawdown"]
    perf = perf[[c for c in front if c in perf.columns] +
                [c for c in perf.columns if c not in front]]
    if verbose and len(perf):
        show = perf[perf["sample"] == "all"].set_index(["signal", "horizon"])
        have = [c for c in ["n_quarters", "mean_q", "t_nw", "sharpe_ann", "hit"]
                if c in show.columns]
        print("\n=== own / disp, full sample ===")
        print(show[have].round(4).to_string() if len(have) > 1
              else "  no cell has >= 8 quarters")
    return perf


# ============================================== CONDITIONAL (DOUBLE) SORT
def conditional_sort(sq: pd.DataFrame, signal: str, condition: str,
                     n_buckets: int = 5, cfg: Config = None, horizon: int = 1,
                     size_neutral: bool = False) -> pd.DataFrame:
    """Does `signal` work better in some `condition` buckets than others?

    Each quarter the cross-section is split into `n_buckets` groups on
    `condition` (bucket 0 = lowest), then the usual decile long-short on `signal`
    is run INSIDE each bucket. A monotone pattern across buckets is far more
    convincing than one bucket being significant -- with 5 buckets, one clearing
    |t| > 2 by chance is unremarkable.

    size_neutral=False by default: conditioning on disagreement AND on market cap
    at once fragments the cross-section badly.
    """
    cfg = cfg or Config()
    ret = RET_COL[horizon]
    d = sq[["yq", "security", signal, condition, ret, "market_cap"]].dropna(
        subset=[signal, condition, ret])
    split = pd.Period(cfg.split, freq="Q")

    def _bin(x):
        if len(x) < cfg.min_names:
            return np.nan
        b = _bucket(x[signal], cfg.n_bins, cfg.tie_break)
        if b is None or b.nunique() < 2:
            return np.nan
        return x.loc[b == b.max(), ret].mean() - x.loc[b == b.min(), ret].mean()

    def _one(q, k):
        cb = _bucket(q[condition], n_buckets, cfg.tie_break)
        if cb is None:
            return np.nan
        sub = q[cb == k]
        if not size_neutral:
            return _bin(sub)
        sg = _bucket(sub["market_cap"], cfg.size_groups, cfg.tie_break)
        if sg is None:
            return _bin(sub)
        v = [_bin(s2) for _, s2 in sub.groupby(sg)]
        v = [x for x in v if np.isfinite(x)]
        return float(np.mean(v)) if v else np.nan

    rows = []
    for k in range(n_buckets):
        sp = d.groupby("yq").apply(lambda q: _one(q, k),
                                   include_groups=False).dropna()
        for sample in ("discovery", "validation", "all"):
            s = sp[_sample_mask(sp.index, sample, split)]
            r = performance(s, horizon, cfg)
            r.update(signal=signal, condition=condition, bucket=k,
                     horizon=horizon, sample=sample)
            rows.append(r)
    return pd.DataFrame(rows)


def bucket_table(res: pd.DataFrame, sample: str = "all") -> pd.DataFrame:
    """The readable view of conditional_sort: one row per condition bucket.

    Kept as a function rather than stashed in .attrs -- pandas compares attrs
    element-wise when concatenating, and a DataFrame in there makes concat raise.
    """
    d = res[res["sample"] == sample]
    cols = [c for c in ("n_quarters", "mean_q", "t_nw", "sharpe_ann", "hit")
            if c in d.columns]
    return d.set_index("bucket")[cols]


def conditional_grid(sq: pd.DataFrame, signals=None, conditions=None,
                     n_buckets: int = 5, cfg: Config = None,
                     horizons=(1, 2)) -> pd.DataFrame:
    """conditional_sort over several (signal, condition, horizon) combinations."""
    cfg = cfg or Config()
    signals = signals or OWN_SIGNALS
    conditions = conditions or ["disp_aw", "sum_abs_aw", "disp_z_aw"]
    out = []
    for sig in signals:
        for cond in conditions:
            if sig not in sq.columns or cond not in sq.columns:
                continue
            for h in horizons:
                out.append(conditional_sort(sq, sig, cond, n_buckets, cfg, h))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def monotonicity(res: pd.DataFrame, sample: str = "all") -> pd.DataFrame:
    """Per (signal, condition, horizon): is the effect monotone across buckets?

    Reports the Spearman correlation between bucket index and t, plus the
    high-minus-low gap. A real interaction shows a gradient, not one lucky cell.
    """
    d = res[res["sample"] == sample]
    rows = []
    for (sig, cond, h), g in d.groupby(["signal", "condition", "horizon"]):
        g = g.sort_values("bucket").dropna(subset=["t_nw"])
        if len(g) < 3:
            continue
        rows.append(dict(
            signal=sig, condition=cond, horizon=h, n_buckets=len(g),
            t_lowest=g.t_nw.iloc[0], t_highest=g.t_nw.iloc[-1],
            gap=g.t_nw.iloc[-1] - g.t_nw.iloc[0],
            spearman=g.bucket.corr(g.t_nw, method="spearman"),
            n_cells_sig=int((g.t_nw.abs() >= 2).sum())))
    return pd.DataFrame(rows).sort_values("gap", key=abs, ascending=False)


# ================================================== 5 x 5 INDEPENDENT DOUBLE SORT
def double_sort(sq: pd.DataFrame, own: str = "inst_own", disp: str = "disp_aw",
                n_own: int = 5, n_disp: int = 5, cfg: Config = None,
                horizon: int = 1, sample: str = "all") -> Dict[str, pd.DataFrame]:
    """Mean forward return in every (own bucket x disp bucket) cell.

    INDEPENDENT sorts: each quarter the cross-section is bucketed on `own` and,
    separately, on `disp`; a stock lands in the cell given by both. Cell counts
    are therefore uneven whenever the two are correlated -- `n` reports that, and
    a near-empty corner cell is not interpretable.

    Returns
      mean   [n_own x n_disp]  average forward return per cell
      t      [n_own x n_disp]  Newey-West t of that cell's quarterly mean series
      n      [n_own x n_disp]  average number of stocks per cell per quarter
      own_ls per disp column: top own bucket - bottom own bucket (the own signal
             evaluated at that level of disagreement)
      disp_ls per own row: top disp - bottom disp
    """
    cfg = cfg or Config()
    ret = RET_COL[horizon]
    d = sq[["yq", "security", own, disp, ret]].dropna()
    split = pd.Period(cfg.split, freq="Q")

    def _cells(q):
        bo = _bucket(q[own], n_own, cfg.tie_break)
        bd = _bucket(q[disp], n_disp, cfg.tie_break)
        if bo is None or bd is None:
            return None
        return q.assign(_bo=bo, _bd=bd)

    per_q = []
    for yq, q in d.groupby("yq"):
        c = _cells(q)
        if c is None:
            continue
        g = c.groupby(["_bo", "_bd"])[ret].agg(["mean", "size"])
        g["yq"] = yq
        per_q.append(g.reset_index())
    if not per_q:
        return {}
    P = pd.concat(per_q, ignore_index=True)
    P = P[_sample_mask(pd.PeriodIndex(P["yq"]), sample, split)]

    mean = P.pivot_table(index="_bo", columns="_bd", values="mean", aggfunc="mean")
    n = P.pivot_table(index="_bo", columns="_bd", values="size", aggfunc="mean")
    t = P.pivot_table(index="_bo", columns="_bd", values="mean",
                      aggfunc=lambda x: newey_west_t(x.to_numpy(),
                                                     lags=max(0, horizon - 1)))

    # margins: the own signal at each level of disagreement, and vice versa
    def _ls(pivot_col, hi, lo):
        w = P.pivot_table(index="yq", columns=pivot_col, values="mean")
        if hi not in w.columns or lo not in w.columns:
            return {}
        sp = (w[hi] - w[lo]).dropna()
        r = performance(sp, horizon, cfg)
        return {k: r.get(k) for k in ("n_quarters", "mean_q", "t_nw", "sharpe_ann",
                                      "hit")}

    own_ls, disp_ls = [], []
    for k in sorted(P["_bd"].unique()):
        sub = P[P["_bd"] == k]
        w = sub.pivot_table(index="yq", columns="_bo", values="mean")
        if w.shape[1] >= 2:
            sp = (w[w.columns.max()] - w[w.columns.min()]).dropna()
            r = performance(sp, horizon, cfg)
            own_ls.append(dict(disp_bucket=k, **{c: r.get(c) for c in
                          ("n_quarters", "mean_q", "t_nw", "sharpe_ann", "hit")}))
    for k in sorted(P["_bo"].unique()):
        sub = P[P["_bo"] == k]
        w = sub.pivot_table(index="yq", columns="_bd", values="mean")
        if w.shape[1] >= 2:
            sp = (w[w.columns.max()] - w[w.columns.min()]).dropna()
            r = performance(sp, horizon, cfg)
            disp_ls.append(dict(own_bucket=k, **{c: r.get(c) for c in
                           ("n_quarters", "mean_q", "t_nw", "sharpe_ann", "hit")}))

    for m in (mean, t, n):
        m.index.name = f"{own} bucket"
        m.columns.name = f"{disp} bucket"
    return dict(mean=mean, t=t, n=n,
                own_ls=pd.DataFrame(own_ls), disp_ls=pd.DataFrame(disp_ls))


def print_double_sort(res: Dict[str, pd.DataFrame], own="inst_own",
                      disp="disp_aw") -> None:
    if not res:
        print("no cells")
        return
    print(f"=== mean forward return per cell (rows {own}, cols {disp}) ===")
    print(res["mean"].round(4).to_string())
    print("\n=== t-stat per cell ===")
    print(res["t"].round(2).to_string())
    print("\n=== avg stocks per cell per quarter ===")
    print(res["n"].round(0).to_string())
    if len(res["own_ls"]):
        print(f"\n=== {own} high-minus-low, WITHIN each {disp} bucket ===")
        print("   (does the ownership signal work better where funds disagree?)")
        print(res["own_ls"].round(4).to_string(index=False))
    if len(res["disp_ls"]):
        print(f"\n=== {disp} high-minus-low, WITHIN each {own} bucket ===")
        print(res["disp_ls"].round(4).to_string(index=False))


def strategy_spread(sq: pd.DataFrame, signal: str = "inst_own",
                    condition: str = "sum_abs_aw", bucket: int = 0,
                    n_buckets: int = 5, cfg: Config = None, horizon: int = 1,
                    n_bins: int = None) -> pd.Series:
    """Per-quarter long-short return of `signal` INSIDE one `condition` bucket.

    bucket=0 is the LOWEST bucket of `condition` (buckets run 0..n_buckets-1).
    Long the top `n_bins` bucket of `signal`, short the bottom, equal weighted.

    Returns the raw quarterly series so it can be compounded, plotted, or
    combined -- conditional_sort() reports the same thing already summarised.
    """
    cfg = cfg or Config()
    nb = n_bins or cfg.n_bins
    ret = RET_COL[horizon]
    d = sq[["yq", "security", signal, condition, ret]].dropna()

    def _one(q):
        cb = _bucket(q[condition], n_buckets, cfg.tie_break)
        if cb is None:
            return np.nan
        sub = q[cb == bucket]
        if len(sub) < cfg.min_names:
            return np.nan
        b = _bucket(sub[signal], nb, cfg.tie_break)
        if b is None or b.nunique() < 2:
            return np.nan
        return sub.loc[b == b.max(), ret].mean() - sub.loc[b == b.min(), ret].mean()

    return d.groupby("yq").apply(_one, include_groups=False).dropna()


def report_strategy(sp: pd.Series, cfg: Config = None, horizon: int = 1,
                    label: str = "strategy") -> pd.DataFrame:
    """discovery / validation / full-sample performance of one spread series."""
    cfg = cfg or Config()
    split = pd.Period(cfg.split, freq="Q")
    rows = []
    for sample in ("discovery", "validation", "all"):
        s = sp[_sample_mask(sp.index, sample, split)]
        r = performance(s, horizon, cfg)
        r.update(strategy=label, sample=sample,
                 first=str(s.index.min()) if len(s) else None,
                 last=str(s.index.max()) if len(s) else None)
        rows.append(r)
    out = pd.DataFrame(rows)
    front = ["strategy", "sample", "first", "last", "n_quarters", "mean_q",
             "t_nw", "hit", "ann_return", "sharpe_ann", "cum_return",
             "max_drawdown"]
    return out[[c for c in front if c in out.columns]]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ownership + disagreement signals")
    ap.add_argument("--root", default=".")
    ap.add_argument("--buckets", type=int, default=5)
    ap.add_argument("--out", default="own_disp")
    a = ap.parse_args()

    cfg = Config(root=a.root)
    sq = build(cfg)
    perf = evaluate(sq, cfg)
    perf.to_csv(f"{a.out}_performance.csv", index=False)

    grid = conditional_grid(sq, cfg=cfg, n_buckets=a.buckets)
    grid.to_csv(f"{a.out}_conditional.csv", index=False)

    ds = double_sort(sq, "inst_own", "disp_aw", a.buckets, a.buckets, cfg)
    print()
    print_double_sort(ds)
    if ds:
        ds["mean"].to_csv(f"{a.out}_matrix_mean.csv")
        ds["t"].to_csv(f"{a.out}_matrix_t.csv")
    print(f"\n=== is `own` better in some disagreement buckets? "
          f"({a.buckets} buckets) ===")
    print(monotonicity(grid).round(3).to_string(index=False))
    print(f"\nwrote {a.out}_performance.csv, {a.out}_conditional.csv")
