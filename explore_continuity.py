"""Does 'continuous presence over the last 8 quarters' drive the CONTEMPORANEOUS
result? Pure panel exploration -- no model, no LSTM, no training.

The pipeline's feasibility mask forbids predicting SELL for any position lacking a
full 8-quarter history. That is not really feasibility (every ranked position IS
held at t); it is a filter on POSITION AGE. The hypothesis to test:

    age is mechanically correlated with the SAME-QUARTER return, because a young
    position is one the fund recently built -- typically into a name that was
    rising -- so conditioning on age imports that co-movement into any sort,
    and buys nothing once the return window stops overlapping.

If true you should see, with no model at all:
  A. young/intermittent positions earn much MORE contemporaneously and no more
     (or less) one quarter later;
  B. the active-weight sort's contemporaneous spread shrinks a lot when you
     restrict to continuous positions only -- while the predictive spread barely
     moves either way;
  C. continuity by itself, used as a signal, "predicts" contemporaneously and
     dies forward.

Inputs are the raw deliverables (see signals_perf.py for the schema notes); this
module reuses that loader so ids, daily->quarterly returns and strict lags are
handled identically.

Usage
-----
    import explore_continuity as E
    out = E.run(E.Config(root="/path/to/root"))
    out["age_buckets"]; out["filter_effect"]; out["continuity_signal"]
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

import signals_perf as S


@dataclass
class Config:
    root: str = "."
    seq_len: int = 8              # the pipeline's 8-quarter window
    max_rank: int = 25            # top-N by active weight, as the pipeline uses
    n_bins: int = 5               # quintiles for the active-weight sort
    min_names: int = 20
    split: str = "2014Q1"
    ann: int = 4


# ------------------------------------------------------------------ build
def build(cfg: Config) -> pd.DataFrame:
    """Panel with active weight, rank, and continuity flags. No model involved."""
    base = S.Config(root=cfg.root)
    panel = S.load_panels(base)
    rets = S.load_returns(base)

    df = panel.copy()
    ok = (df["position_value"] > 0) & (df["market_cap"] > 0)
    df["_pv"] = df["position_value"].where(ok)
    df["_mc"] = df["market_cap"].where(ok)
    g = df.groupby(["fund", "yq"], sort=False)
    df["active_weight"] = (df["_pv"] / g["_pv"].transform("sum")
                           - df["_mc"] / g["_mc"].transform("sum"))
    df = df.dropna(subset=["active_weight"])

    # rank 1 = most overweight, within fund-quarter
    df["rank"] = (df.groupby(["fund", "yq"], sort=False)["active_weight"]
                    .rank(method="first", ascending=False))
    if cfg.max_rank:
        df = df[df["rank"] <= cfg.max_rank]

    # ---- CONTINUITY: how many of the last seq_len quarters was it held? -----
    # STRICT quarter arithmetic -- a fund's holdings jump in and out, so counting
    # observations rather than quarters would be wrong.
    held = panel[["fund", "security", "qi"]].drop_duplicates()
    held["held"] = 1
    n_held = []
    for k in range(cfg.seq_len):
        h = held.copy()
        h["qi"] = h["qi"] + k                    # was it held k quarters ago?
        n_held.append(h.rename(columns={"held": f"h{k}"}))
    for k, h in enumerate(n_held):
        df = df.merge(h, on=["fund", "security", "qi"], how="left")
    hcols = [f"h{k}" for k in range(cfg.seq_len)]
    df["n_held_8q"] = df[hcols].fillna(0).sum(axis=1).astype(int)
    df = df.drop(columns=hcols)

    # feasible == the pipeline's mask: held in EVERY one of the last 8 quarters
    df["feasible"] = df["n_held_8q"] >= cfg.seq_len
    # age = consecutive quarters held up to and including t
    df["is_new"] = df["n_held_8q"] <= 1

    df = df.merge(rets.drop(columns=["qi"]), on=["security", "yq"], how="left")
    print(f"[build] {len(df):,} ranked positions | feasible {df.feasible.mean():.1%} "
          f"| new {df.is_new.mean():.1%}")
    return df


# ------------------------------------------------------------------ helpers
def _t(x, lags=0):
    return S.newey_west_t(np.asarray(x, float), lags=lags)


def _per_quarter(d: pd.DataFrame, by: str, col: str) -> pd.DataFrame:
    """Mean of `col` per (bucket, quarter), then averaged over quarters + t-stat."""
    pq = d.groupby([by, "yq"], observed=True)[col].mean()
    out = pq.groupby(by, observed=True).agg(mean="mean", n_q="size")
    out["t"] = pq.groupby(by, observed=True).apply(lambda s: _t(s.to_numpy()))
    return out


# ------------------------------------------------ A. age vs return window
def age_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Return by position age, contemporaneously vs one and two quarters ahead.

    The mechanism claim: young positions earn a lot in the SAME quarter (the fund
    bought into strength) and nothing extra afterwards.
    """
    d = df.copy()
    d["age"] = pd.cut(d["n_held_8q"], [-1, 1, 3, 5, 7, 8],
                      labels=["1 (new)", "2-3", "4-5", "6-7", "8 (continuous)"])
    rows = []
    for col, lab in (("quarterly_ret", "contemporaneous t"),
                     ("future_1q_ret", "next quarter t+1"),
                     ("future_2q_ret", "t+2 (predictive)")):
        r = _per_quarter(d.dropna(subset=[col]), "age", col)
        r["window"] = lab
        rows.append(r.reset_index())
    out = pd.concat(rows, ignore_index=True)
    return out.pivot(index="age", columns="window", values="mean")


# ------------------------------- B. does the filter change the sort's spread?
def filter_effect(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Active-weight quintile spread, computed on ALL positions vs CONTINUOUS only.

    Same sort, same funds, only the eligible row set differs. If the contemporaneous
    spread depends on including young positions, restricting to continuous ones
    should shrink it -- while the predictive spread should be unaffected (it is
    ~zero either way).
    """
    rows = []
    for subset, sub in (("all positions", df),
                        ("continuous only (8/8)", df[df["feasible"]]),
                        ("non-continuous only", df[~df["feasible"]])):
        for col, lab, lag in (("quarterly_ret", "contemporaneous", 0),
                              ("future_1q_ret", "t+1", 0),
                              ("future_2q_ret", "t+2 predictive", 1)):
            d = sub.dropna(subset=["active_weight", col])

            def one(q):
                if len(q) < cfg.min_names:
                    return np.nan
                b = pd.qcut(q["active_weight"].rank(method="first"), cfg.n_bins,
                            labels=False, duplicates="drop")
                if b.nunique() < 2:
                    return np.nan
                return q.loc[b == b.max(), col].mean() - q.loc[b == b.min(), col].mean()

            sp = d.groupby("yq").apply(one, include_groups=False).dropna()
            if len(sp) < 8:
                continue
            rows.append(dict(subset=subset, window=lab, n_quarters=len(sp),
                             spread=sp.mean(), t=_t(sp.to_numpy(), lags=lag),
                             hit=(sp > 0).mean()))
    return pd.DataFrame(rows)


# ------------------------------------------- C. continuity as its own signal
def continuity_signal(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Sort on continuity itself: continuous minus non-continuous, per quarter.

    No active weight involved. If continuity 'works' contemporaneously and dies
    forward, it is the same co-movement, isolated.
    """
    rows = []
    for col, lab, lag in (("quarterly_ret", "contemporaneous", 0),
                          ("future_1q_ret", "t+1", 0),
                          ("future_2q_ret", "t+2 predictive", 1),
                          ("future_3q_ret", "t+3 tradeable", 2)):
        d = df.dropna(subset=[col])
        pq = d.groupby(["feasible", "yq"], observed=True)[col].mean()
        try:
            sp = (pq.xs(True, level="feasible") - pq.xs(False, level="feasible")).dropna()
        except KeyError:
            continue
        if len(sp) < 8:
            continue
        rows.append(dict(window=lab, n_quarters=len(sp), spread=sp.mean(),
                         t=_t(sp.to_numpy(), lags=lag), hit=(sp > 0).mean()))
    return pd.DataFrame(rows)


# --------------------------------- D. FEASIBLE vs INFEASIBLE *SELLING*
def add_trade_label(df: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Realised trade t -> t+1 per position: sell / hold / buy.

    An EXIT (held at t, gone at t+1) is a sell -- the panel has no row for it, so it
    must be reconstructed, exactly as in signals_perf. Without that, sells are
    systematically undercounted, and undercounted MORE among young positions (which
    are likelier to be exited), which is precisely the asymmetry under test.
    """
    nxt = panel[["fund", "security", "qi", "position_value"]].copy()
    nxt["qi"] -= 1                                  # value one quarter LATER
    nxt = nxt.rename(columns={"position_value": "pv_next"})
    d = df.merge(nxt, on=["fund", "security", "qi"], how="left")

    # a fund that still reports at t+1 but not this name => exited (value 0)
    reports = panel[["fund", "qi"]].drop_duplicates().assign(rep=1)
    reports["qi"] -= 1
    d = d.merge(reports, on=["fund", "qi"], how="left")
    exited = d["pv_next"].isna() & d["rep"].eq(1)
    d["pv_next"] = d["pv_next"].where(~exited, 0.0)
    d["exit"] = exited

    chg = (d["pv_next"] - d["position_value"]) / d["position_value"].abs()
    d["trade"] = np.select([chg < -0.01, chg > 0.01], ["sell", "buy"],
                           default="hold")
    d.loc[d["pv_next"].isna(), "trade"] = None      # fund stopped reporting
    return d


def selling_asymmetry(df: pd.DataFrame, panel: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Is SELLING different for feasible vs infeasible positions?

    The mask forbids predicting 'sell' for infeasible positions. That is only
    harmless if infeasible sells are rare and unremarkable. Three things to check:
      1. base rate  -- how often does each group actually sell?
      2. exits      -- how much of that selling is a full exit?
      3. the return -- what do sells earn in each group, in each window?
    """
    d = add_trade_label(df, panel).dropna(subset=["trade"])
    grp = d.groupby("feasible", observed=True)

    rates = pd.DataFrame({
        "n_positions": grp.size(),
        "P(sell)": grp["trade"].apply(lambda s: (s == "sell").mean()),
        "P(hold)": grp["trade"].apply(lambda s: (s == "hold").mean()),
        "P(buy)": grp["trade"].apply(lambda s: (s == "buy").mean()),
        "P(exit | sell)": grp.apply(
            lambda g: g.loc[g.trade == "sell", "exit"].mean(), include_groups=False),
    })

    # returns of SELLS only, feasible vs not, per window
    rows = []
    sells = d[d["trade"] == "sell"]
    for col, lab, lag in (("quarterly_ret", "contemporaneous", 0),
                          ("future_1q_ret", "t+1", 0),
                          ("future_2q_ret", "t+2 predictive", 1)):
        for feas, sub in sells.groupby("feasible", observed=True):
            s = sub.dropna(subset=[col])
            pq = s.groupby("yq")[col].mean().dropna()
            if len(pq) < 8:
                continue
            rows.append(dict(window=lab, feasible=bool(feas), n_quarters=len(pq),
                             mean_ret=pq.mean(), t=_t(pq.to_numpy(), lags=lag)))
        # the difference that the mask effectively assumes away
        pq_t = sells[sells.feasible].dropna(subset=[col]).groupby("yq")[col].mean()
        pq_f = sells[~sells.feasible].dropna(subset=[col]).groupby("yq")[col].mean()
        sp = (pq_f - pq_t).dropna()
        if len(sp) >= 8:
            rows.append(dict(window=lab, feasible=None, n_quarters=len(sp),
                             mean_ret=sp.mean(), t=_t(sp.to_numpy(), lags=lag)))
    sell_ret = pd.DataFrame(rows)

    # what the mask actually costs: infeasible positions that DID sell
    forced = d[(~d["feasible"]) & (d["trade"] == "sell")]
    cost = pd.DataFrame([dict(
        infeasible_sells=len(forced),
        share_of_all_sells=len(forced) / max((d["trade"] == "sell").sum(), 1),
        share_of_all_positions=len(forced) / max(len(d), 1),
        mean_contemporaneous_ret=forced["quarterly_ret"].mean(),
        mean_next_q_ret=forced["future_1q_ret"].mean())])
    return {"sell_base_rates": rates, "sell_returns": sell_ret,
            "mask_cost": cost}


# ------------------------------------------------------------------ driver
def run(cfg: Config = Config(), verbose: bool = True) -> Dict[str, pd.DataFrame]:
    df = build(cfg)
    panel = S.load_panels(S.Config(root=cfg.root))
    out = {
        "age_buckets": age_buckets(df),
        "filter_effect": filter_effect(df, cfg),
        "continuity_signal": continuity_signal(df, cfg),
    }
    out.update(selling_asymmetry(df, panel))
    if verbose:
        print("\n=== A. mean return by position age ===")
        print("   (young positions should earn MUCH more contemporaneously,")
        print("    and no more afterwards, if the effect is co-movement)")
        print(out["age_buckets"].round(4).to_string())

        print("\n=== B. active-weight spread, with vs without the age filter ===")
        print(out["filter_effect"].pivot_table(
            index="subset", columns="window", values=["spread", "t"]
        ).round(4).to_string())

        print("\n=== C. continuity used as a signal by itself ===")
        print("   (continuous minus non-continuous, equal weighted)")
        print(out["continuity_signal"].round(4).to_string(index=False))

        print("\n=== D1. SELL base rates: feasible vs infeasible ===")
        print("   the mask forbids 'sell' for infeasible -- how often do they sell?")
        print(out["sell_base_rates"].round(4).to_string())

        print("\n=== D2. what SELLS earn, by feasibility ===")
        print("   feasible=None rows are (infeasible - feasible), the difference")
        print("   the mask assumes away")
        print(out["sell_returns"].round(4).to_string(index=False))

        print("\n=== D3. what the mask actually suppresses ===")
        print(out["mask_cost"].round(4).to_string(index=False))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--max-rank", type=int, default=25)
    ap.add_argument("--out-prefix", default="explore_continuity")
    a = ap.parse_args()
    res = run(Config(root=a.root, seq_len=a.seq_len, max_rank=a.max_rank))
    for k, v in res.items():
        p = f"{a.out_prefix}_{k}.csv"
        v.to_csv(p)
        print(f"wrote {p}")
