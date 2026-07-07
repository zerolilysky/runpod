"""
VIX-gated wrapper around price_trend_cnn.py.

Import usage:
    import price_trend_vix_cnn as m
    res = m.run(
        data_dir=".",
        pattern="sp00_5MA",
        horizon=5,
        test_start="2015",
        test_end="2019-12-31",
        gate="gate_16_21",
        cpu_threads="auto",
    )

Gate semantics:
    gate_{lower}_{upper} trades outside the middle VIX band:
      - gate_16_21      -> active when VIX < 16 or VIX > 21
      - gate_nan_21     -> active when VIX > 21
      - gate_16_nan     -> active when VIX < 16
      - gate_nan_nan    -> no VIX gate

The base CNN is always trained by price_trend_cnn.run from the chart-image
pickle in this call. This module does not read a pre-existing predictions.csv,
so it cannot silently pick up a cache from another model run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import price_trend_cnn as base


def configure_cpu(cpu_threads="auto"):
    """Set CPU math-library threads before the base model imports torch."""
    if cpu_threads in (None, "", "none", "None"):
        return None
    if str(cpu_threads).lower() == "auto":
        n = max(1, (os.cpu_count() or 2) - 1)
    else:
        n = max(1, int(cpu_threads))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(key, str(n))
    try:
        import torch
        torch.set_num_threads(n)
        torch.set_num_interop_threads(max(1, min(4, n // 2)))
    except Exception:
        pass
    return n


def resolve_pattern(pattern, data_dir=None):
    """Accept local shorthand/typo patterns such as sp00_5MA."""
    pattern = pattern or "sp00_5MA"
    data_dir = Path(data_dir) if data_dir else Path.cwd()
    try:
        if base._find_files(data_dir, pattern):
            return pattern
    except Exception:
        pass
    if str(pattern).lower() in {"sp00_5ma", "sp500_5ma", "s&p500_5ma"}:
        return "SP500_5MA"
    return pattern


def _target_col(horizon):
    return f"rF{int(horizon)}D"


def _train_cut_from_test_start(test_start):
    if test_start in (None, "", "none", "None"):
        raise ValueError("training filters need test_start so train/test labels stay separated")
    return pd.Timestamp(str(test_start)) - pd.Timedelta(days=1)


def _format_filter_value(x):
    x = float(x)
    pct = x * 100.0 if abs(x) < 1.0 else x
    return str(int(round(pct))) if abs(pct - round(pct)) < 1e-9 else ("%g" % pct)


def _find_source_pickle(data_dir, pattern):
    files = base._find_files(data_dir, pattern)
    if not files:
        raise FileNotFoundError(f"no chart-image pickle matching {pattern!r} in {data_dir}")
    return Path(files[0])


def _label_counts(df, target, train_mask):
    return {
        "train_label_count": int(df.loc[train_mask, target].notna().sum()),
        "test_label_count": int(df.loc[~train_mask, target].notna().sum()),
    }


def _write_filtered_training_input(
    source_pickle,
    out_dir,
    horizon=5,
    train_cut="2014-12-31",
    drop_train_abs_ret=None,
    train_vix_lt=None,
    vix_series=None,
    data_dir=None,
    vix_symbol="^VIX",
    vix_cache="vix.csv",
    vix_download=True,
):
    """Write a chart pickle with selected training labels masked to NaN."""
    source_pickle = Path(source_pickle)
    out_dir = Path(out_dir)
    target = _target_col(horizon)
    df = pd.read_pickle(source_pickle).copy()
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not found in {source_pickle}")
    if "date" not in df.columns:
        raise KeyError(f"date column not found in {source_pickle}")

    df["date"] = pd.to_datetime(df["date"])
    train_cut = pd.Timestamp(train_cut)
    train_mask = df["date"] <= train_cut
    before = _label_counts(df, target, train_mask)
    suffixes = []
    meta = {
        "source_pickle": str(source_pickle),
        "target": target,
        "train_cut": str(train_cut.date()),
        "train_label_count_before": before["train_label_count"],
        "test_label_count_before": before["test_label_count"],
    }

    if drop_train_abs_ret is not None:
        threshold = float(drop_train_abs_ret)
        bad = train_mask & df[target].notna() & pd.to_numeric(df[target], errors="coerce").abs().gt(threshold)
        dropped = int(bad.sum())
        df.loc[bad, target] = np.nan
        suffixes.append(f"drop_train_abs{_format_filter_value(threshold)}")
        meta.update(
            {
                "drop_train_abs_ret": threshold,
                "dropped_train_extreme_count": dropped,
            }
        )

    if train_vix_lt is not None:
        threshold = float(train_vix_lt)
        if vix_series is None:
            vix_series = base.load_vix(
                start=(df["date"].min() - pd.Timedelta(days=10)).date(),
                end=(train_cut + pd.Timedelta(days=5)).date(),
                data_dir=Path(data_dir) if data_dir else source_pickle.parent,
                symbol=vix_symbol,
                cache=vix_cache,
                download=vix_download,
            )
        aligned_vix = _align_series(vix_series, df["date"])
        bad = train_mask & df[target].notna() & ~(pd.Series(aligned_vix, index=df.index).astype(float) < threshold)
        dropped = int(bad.sum())
        df.loc[bad, target] = np.nan
        suffixes.append(f"train_vix_lt{_format_filter_value(threshold)}")
        meta.update(
            {
                "train_vix_lt": threshold,
                "dropped_train_vix_count": dropped,
            }
        )

    if not suffixes:
        raise ValueError("at least one training filter must be provided")

    after = _label_counts(df, target, train_mask)
    meta.update(
        {
            "train_label_count_after": after["train_label_count"],
            "test_label_count_after": after["test_label_count"],
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_pickle.stem}_{'_'.join(suffixes)}.pickle"
    df.to_pickle(out_path)
    meta["filtered_input"] = str(out_path)
    (out_dir / f"{out_path.stem}.json").write_text(json.dumps(meta, indent=2, default=str))
    return out_path, meta


def write_drop_train_abs_ret_input(source_pickle, out_dir, threshold, horizon=5,
                                   train_cut="2014-12-31"):
    """Mask training labels whose absolute forward return exceeds threshold."""
    return _write_filtered_training_input(
        source_pickle,
        out_dir,
        horizon=horizon,
        train_cut=train_cut,
        drop_train_abs_ret=threshold,
    )


def write_train_vix_lt_input(source_pickle, out_dir, threshold, horizon=5,
                             train_cut="2014-12-31", vix_series=None,
                             data_dir=None, vix_symbol="^VIX",
                             vix_cache="vix.csv", vix_download=True):
    """Mask training labels unless aligned VIX is below threshold."""
    return _write_filtered_training_input(
        source_pickle,
        out_dir,
        horizon=horizon,
        train_cut=train_cut,
        train_vix_lt=threshold,
        vix_series=vix_series,
        data_dir=data_dir,
        vix_symbol=vix_symbol,
        vix_cache=vix_cache,
        vix_download=vix_download,
    )


def _is_nan(x) -> bool:
    try:
        return bool(math.isnan(float(x)))
    except (TypeError, ValueError):
        return False


def _parse_bound(text):
    s = str(text).strip().lower()
    if s in {"", "nan", "none", "null", "na"}:
        return float("nan")
    return float(s)


def _format_bound(x):
    if _is_nan(x):
        return "nan"
    x = float(x)
    return str(int(x)) if x.is_integer() else ("%g" % x)


def parse_gate(gate="gate_nan_nan", lower=None, upper=None):
    """Return (lower, upper, canonical_name) for gate_{lower}_{upper}."""
    if gate is None:
        gate = "gate_nan_nan"
    if isinstance(gate, (tuple, list)) and len(gate) == 2:
        lo, hi = map(_parse_bound, gate)
    else:
        m = re.fullmatch(r"gate_([^_]+)_([^_]+)", str(gate).strip().lower())
        if not m:
            raise ValueError("gate must look like gate_16_21, gate_nan_21, or gate_16_nan")
        lo, hi = _parse_bound(m.group(1)), _parse_bound(m.group(2))
    if lower is not None:
        lo = _parse_bound(lower)
    if upper is not None:
        hi = _parse_bound(upper)
    if not _is_nan(lo) and not _is_nan(hi) and float(lo) > float(hi):
        raise ValueError("gate lower must be <= upper")
    return lo, hi, f"gate_{_format_bound(lo)}_{_format_bound(hi)}"


def vix_gate_mask(vix, lower, upper):
    """Boolean active mask. Finite lower+upper means trade outside the band."""
    arr = pd.Series(vix).astype(float)
    lo_nan, hi_nan = _is_nan(lower), _is_nan(upper)
    if lo_nan and hi_nan:
        return pd.Series(True, index=arr.index)
    if lo_nan:
        return arr > float(upper)
    if hi_nan:
        return arr < float(lower)
    return (arr < float(lower)) | (arr > float(upper))


def _ppy(freq):
    return {"W": 52.0, "M": 12.0, "Q": 4.0}.get(str(freq).upper(), 52.0)


def _sharpe(x, periods_per_year):
    r = pd.Series(x).dropna().astype(float)
    if len(r) == 0:
        return float("nan")
    return float(r.mean() / (r.std() + 1e-12) * np.sqrt(periods_per_year))


def _cagr(x, periods_per_year):
    r = pd.Series(x).dropna().astype(float)
    if len(r) == 0:
        return float("nan")
    return float(np.cumprod(1.0 + r.values)[-1] ** (periods_per_year / len(r)) - 1.0)


def _align_series(series, dates):
    s = pd.Series(series).dropna().sort_index()
    idx = s.index.values.astype("datetime64[ns]")
    vals = s.to_numpy(float)
    d = pd.to_datetime(dates).values.astype("datetime64[ns]")
    loc = np.searchsorted(idx, d, side="right") - 1
    out = np.full(len(d), np.nan, dtype=np.float32)
    good = loc >= 0
    out[good] = vals[loc[good]].astype(np.float32)
    return out


def period_diagnostics(predictions, freq="W", min_names=20):
    """Build period-level top/bottom return diagnostics from raw predictions."""
    pred = predictions.dropna(subset=["score", "ret"]).copy()
    pred["per"] = pred["date"].dt.to_period(freq)
    rows = []
    for per, g in pred.groupby("per"):
        g = g.dropna(subset=["score", "ret"])
        if len(g) < int(min_names) or g["score"].std() < 1e-12:
            continue
        n_tail = max(1, int(np.ceil(0.1 * len(g))))
        top = g.nlargest(n_tail, "score")
        bot = g.nsmallest(n_tail, "score")
        if len(top) == 0 or len(bot) == 0:
            continue
        ic_s = g[["score", "ret"]].corr(method="spearman").iloc[0, 1]
        ic_p = g[["score", "ret"]].corr(method="pearson").iloc[0, 1]
        rows.append(
            {
                "per": str(per),
                "rebalance_date": g["date"].max(),
                "n_names": int(len(g)),
                "score_mean": float(g["score"].mean()),
                "score_std": float(g["score"].std()),
                "ic_spearman": float(ic_s),
                "ic_pearson": float(ic_p),
                "long_only": float(top["ret"].mean()),
                "bottom": float(bot["ret"].mean()),
                "long_short": float(top["ret"].mean() - bot["ret"].mean()),
                "top_hit": float((top["ret"] > 0).mean()),
                "bottom_hit": float((bot["ret"] < 0).mean()),
                "wrong_tail": bool(top["ret"].mean() < bot["ret"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("rebalance_date").reset_index(drop=True)


def add_vix_and_gate(periods, data_dir=None, gate="gate_nan_nan", lower=None,
                     upper=None, vix_symbol="^VIX", vix_cache="vix.csv",
                     vix_download=True):
    """Attach VIX and active mask to period diagnostics."""
    lo, hi, gate_name = parse_gate(gate, lower=lower, upper=upper)
    data_dir = Path(data_dir) if data_dir else Path.cwd()
    vix = base.load_vix(
        start=(pd.to_datetime(periods["rebalance_date"]).min() - pd.Timedelta(days=10)).date(),
        end=(pd.to_datetime(periods["rebalance_date"]).max() + pd.Timedelta(days=5)).date(),
        data_dir=data_dir,
        symbol=vix_symbol,
        cache=vix_cache,
        download=vix_download,
    )
    out = periods.copy()
    out["vix"] = _align_series(vix, out["rebalance_date"])
    out["gate"] = gate_name
    out["active"] = vix_gate_mask(out["vix"], lo, hi).to_numpy(bool)
    if _is_nan(lo) and _is_nan(hi):
        out["vix_regime"] = "all"
    elif _is_nan(lo):
        out["vix_regime"] = np.where(out["active"], f"vix_gt_{_format_bound(hi)}", "inactive")
    elif _is_nan(hi):
        out["vix_regime"] = np.where(out["active"], f"vix_lt_{_format_bound(lo)}", "inactive")
    else:
        out["vix_regime"] = np.select(
            [out["vix"] < float(lo), out["vix"] > float(hi)],
            [f"vix_lt_{_format_bound(lo)}", f"vix_gt_{_format_bound(hi)}"],
            default=f"mid_{_format_bound(lo)}_{_format_bound(hi)}",
        )
    out["gated_long_only"] = out["long_only"].where(out["active"], 0.0)
    out["gated_long_short"] = out["long_short"].where(out["active"], 0.0)
    return out, lo, hi, gate_name


def summarize_gate(periods, lower, upper, gate_name, freq="W"):
    ppy = _ppy(freq)
    active = periods["active"].astype(bool)
    return {
        "gate": gate_name,
        "gate_lower": None if _is_nan(lower) else float(lower),
        "gate_upper": None if _is_nan(upper) else float(upper),
        "n_periods": int(len(periods)),
        "active_periods": int(active.sum()),
        "inactive_periods": int((~active).sum()),
        "active_share": float(active.mean()) if len(periods) else float("nan"),
        "ls_sharpe": _sharpe(periods["gated_long_short"], ppy),
        "ls_cagr": _cagr(periods["gated_long_short"], ppy),
        "lo_sharpe": _sharpe(periods["gated_long_only"], ppy),
        "lo_cagr": _cagr(periods["gated_long_only"], ppy),
        "raw_ls_sharpe": _sharpe(periods["long_short"], ppy),
        "raw_lo_sharpe": _sharpe(periods["long_only"], ppy),
        "active_ls_mean": float(periods.loc[active, "long_short"].mean()) if active.any() else float("nan"),
        "inactive_ls_mean": float(periods.loc[~active, "long_short"].mean()) if (~active).any() else float("nan"),
        "active_ic_mean": float(periods.loc[active, "ic_spearman"].mean()) if active.any() else float("nan"),
        "inactive_ic_mean": float(periods.loc[~active, "ic_spearman"].mean()) if (~active).any() else float("nan"),
    }


def vix_bucket_summary(periods, bins=None):
    if bins is None:
        bins = [-np.inf, 12, 14, 16, 18, 20, 22, 25, np.inf]
    out = periods.copy()
    out["vix_bucket"] = pd.cut(out["vix"], bins=bins)
    rows = []
    for bucket, g in out.groupby("vix_bucket", observed=False):
        if len(g) == 0:
            continue
        rows.append(
            {
                "vix_bucket": str(bucket),
                "n_periods": int(len(g)),
                "active_share": float(g["active"].mean()),
                "ls_mean": float(g["long_short"].mean()),
                "ls_sharpe_active_bucket": _sharpe(g["long_short"], 52.0),
                "ic_mean": float(g["ic_spearman"].mean()),
                "wrong_tail_rate": float(g["wrong_tail"].mean()),
                "top_hit": float(g["top_hit"].mean()),
                "bottom_hit": float(g["bottom_hit"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _safe_import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_diagnostics(periods, bucket, out_dir, gate_name):
    """Write VIX gate visual diagnostics to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        plt = _safe_import_matplotlib()
    except Exception as exc:
        (out_dir / "plot_error.txt").write_text(str(exc))
        return []

    paths = []
    dates = pd.to_datetime(periods["rebalance_date"])

    fig, ax1 = plt.subplots(figsize=(12, 5))
    raw = (1.0 + periods["long_short"].fillna(0.0)).cumprod()
    gated = (1.0 + periods["gated_long_short"].fillna(0.0)).cumprod()
    ax1.plot(dates, raw, lw=1.6, label="raw long-short")
    ax1.plot(dates, gated, lw=1.9, label=f"{gate_name} long-short")
    ax1.set_ylabel("cumulative growth")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(dates, periods["vix"], color="tab:gray", lw=0.9, alpha=0.55, label="VIX")
    ax2.set_ylabel("VIX")
    inactive = ~periods["active"].astype(bool)
    if inactive.any():
        ax1.scatter(dates[inactive], raw[inactive], s=9, color="tab:red", alpha=0.45, label="inactive weeks")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    ax1.set_title(f"Cumulative performance with {gate_name}")
    fig.tight_layout()
    path = out_dir / "vix_gate_cumulative.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = np.where(periods["active"], "tab:blue", "tab:red")
    ax.scatter(periods["vix"], periods["long_short"], c=colors, s=28, alpha=0.75)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("VIX at rebalance")
    ax.set_ylabel("weekly long-short return")
    ax.set_title("Where the cross-section works vs fails")
    ax.grid(alpha=0.25)
    path = out_dir / "vix_vs_long_short.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    if len(bucket):
        fig, ax1 = plt.subplots(figsize=(12, 5))
        x = np.arange(len(bucket))
        colors = np.where(bucket["ls_mean"] >= 0, "tab:green", "tab:red")
        ax1.bar(x, bucket["ls_mean"], color=colors, alpha=0.8)
        ax1.axhline(0, color="black", lw=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(bucket["vix_bucket"], rotation=35, ha="right")
        ax1.set_ylabel("mean weekly long-short")
        ax2 = ax1.twinx()
        ax2.plot(x, bucket["ic_mean"], color="tab:blue", marker="o", label="mean IC")
        ax2.set_ylabel("mean Spearman IC")
        ax1.set_title("VIX bucket performance")
        fig.tight_layout()
        path = out_dir / "vix_bucket_performance.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    return paths


def display_results(res, max_bucket_rows=20):
    """Pretty-display a run() result inside Jupyter."""
    try:
        from IPython.display import Image, display
    except Exception as exc:
        print("IPython display is unavailable:", exc)
        print(json.dumps(res.get("summary", {}), indent=2, default=str))
        return

    summary = res.get("summary", {})
    display(pd.DataFrame([summary]).T.rename(columns={0: "value"}))
    bucket = res.get("vix_bucket_summary")
    if isinstance(bucket, pd.DataFrame) and len(bucket):
        display(bucket.head(max_bucket_rows))
    for path in summary.get("plots", []):
        if Path(path).exists():
            display(Image(filename=str(path)))


def run(data_dir=None, pattern="sp00_5MA",
        gate="gate_nan_nan", gate_lower=None, gate_upper=None,
        task="clf", protocol="faithful", horizon=5,
        test_start="2015", test_end="2019-12-31",
        seeds=5, epochs=50, pixel_norm="global",
        vix_symbol="^VIX", vix_cache="vix.csv", vix_download=True,
        drop_train_abs_ret=None, train_vix_lt=None,
        out_dir=None, source_out_dir=None, tag=None,
        save_model=False, make_plots=True, cpu_threads="auto",
        verbose=True, **kwargs):
    """Run base CNN, apply a VIX gate, save diagnostics, and return results."""
    data_dir = Path(data_dir) if data_dir else Path.cwd()
    used_threads = configure_cpu(cpu_threads)
    pattern = resolve_pattern(pattern, data_dir)
    lower, upper, gate_name = parse_gate(gate, lower=gate_lower, upper=gate_upper)
    tag = tag or f"i5_vix_{gate_name}"
    source_out_dir = Path(source_out_dir) if source_out_dir else data_dir / f"results_{tag}_raw"
    out_dir = Path(out_dir) if out_dir else data_dir / f"results_{tag}"
    run_data_dir = data_dir
    run_pattern = pattern
    filter_meta = {}

    if drop_train_abs_ret is not None or train_vix_lt is not None:
        train_cut = _train_cut_from_test_start(test_start)
        source_pickle = _find_source_pickle(data_dir, pattern)
        filtered_path, filter_meta = _write_filtered_training_input(
            source_pickle,
            source_out_dir / "filtered_input",
            horizon=horizon,
            train_cut=train_cut,
            drop_train_abs_ret=drop_train_abs_ret,
            train_vix_lt=train_vix_lt,
            data_dir=data_dir,
            vix_symbol=vix_symbol,
            vix_cache=vix_cache,
            vix_download=vix_download,
        )
        run_data_dir = filtered_path.parent
        run_pattern = filtered_path.name

    res = base.run(
        data_dir=str(run_data_dir),
        pattern=run_pattern,
        task=task,
        protocol=protocol,
        horizon=horizon,
        test_start=test_start,
        test_end=test_end,
        seeds=seeds,
        epochs=epochs,
        pixel_norm=pixel_norm,
        save_csv=True,
        save_model=save_model,
        out_dir=str(source_out_dir),
        tag=f"{tag}_raw",
        verbose=verbose,
        **kwargs,
    )
    predictions = res["predictions"].drop(columns=["per"], errors="ignore").copy()
    source_summary = res["summary"]

    freq = str(source_summary.get("rebalance") or {5: "W", 20: "M", 60: "Q"}.get(int(horizon), "W"))
    periods = period_diagnostics(predictions, freq=freq)
    periods, lower, upper, gate_name = add_vix_and_gate(
        periods,
        data_dir=data_dir,
        gate=gate_name,
        lower=lower,
        upper=upper,
        vix_symbol=vix_symbol,
        vix_cache=vix_cache,
        vix_download=vix_download,
    )
    summary = summarize_gate(periods, lower, upper, gate_name, freq=freq)
    summary.update(
        {
            "tag": tag,
            "raw_result_dir": str(source_out_dir),
            "pattern": pattern,
            "base_pattern": run_pattern,
            "cpu_threads": used_threads,
            "task": task,
            "protocol": protocol,
            "horizon": int(horizon),
            "rebalance": freq,
            "test_start": test_start,
            "test_end": test_end,
            "threshold_note": "If selected using the test window, treat this as diagnostic, not clean OOS.",
        }
    )
    summary.update(filter_meta)
    bucket = vix_bucket_summary(periods)

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    periods.to_csv(out_dir / "vix_period_diagnostics.csv", index=False)
    periods[[
        "per", "rebalance_date", "vix", "active", "long_only", "long_short",
        "gated_long_only", "gated_long_short", "n_names",
    ]].to_csv(out_dir / "portfolio_period_returns.csv", index=False)
    bucket.to_csv(out_dir / "vix_bucket_summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    plot_paths = plot_diagnostics(periods, bucket, out_dir, gate_name) if make_plots else []
    summary["plots"] = plot_paths
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    if verbose:
        print(json.dumps(summary, indent=2, default=str))
    return {
        "summary": summary,
        "predictions": predictions,
        "period_diagnostics": periods,
        "vix_bucket_summary": bucket,
        "out_dir": str(out_dir),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--pattern", default="sp00_5MA")
    ap.add_argument("--gate", default="gate_nan_nan")
    ap.add_argument("--gate-lower", default=None)
    ap.add_argument("--gate-upper", default=None)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--test-start", default="2015")
    ap.add_argument("--test-end", default="2019-12-31")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--pixel-norm", default="global")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--source-out-dir", default=None)
    ap.add_argument("--drop-train-abs-ret", type=float, default=None,
                    help="mask training labels with abs(rF{h}D) above this value, e.g. 0.10")
    ap.add_argument("--train-vix-lt", type=float, default=None,
                    help="mask training labels unless aligned VIX is below this value, e.g. 20")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--cpu-threads", default="auto",
                    help="'auto' uses os.cpu_count()-1; set an integer or 'none'")
    args = ap.parse_args(argv)
    run(
        data_dir=args.data_dir,
        pattern=args.pattern,
        gate=args.gate,
        gate_lower=args.gate_lower,
        gate_upper=args.gate_upper,
        horizon=args.horizon,
        test_start=args.test_start,
        test_end=args.test_end,
        seeds=args.seeds,
        epochs=args.epochs,
        pixel_norm=args.pixel_norm,
        drop_train_abs_ret=args.drop_train_abs_ret,
        train_vix_lt=args.train_vix_lt,
        out_dir=args.out_dir,
        source_out_dir=args.source_out_dir,
        save_model=args.save_model,
        make_plots=not args.no_plots,
        cpu_threads=args.cpu_threads,
    )


if __name__ == "__main__":
    main()
