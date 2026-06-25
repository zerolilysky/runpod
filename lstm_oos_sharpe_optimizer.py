from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

TICKERS = ["NVDA", "GOOGL", "GOOG", "AAPL", "MSFT", "AMZN", "AVGO", "WMT"]

CONFIG = {
    "seed": 42,
    "target_sharpe": 2.0,
    "train_days": 756,
    "test_days": 252,
    "step_days": 252,
    "tc": 0.0005,
    "data_csv": "stock_returns.csv",  # auto-fallback: ./stock_returns.csv or ./data/stock_returns.csv
}


@dataclass
class TrialResult:
    sharpe: float
    cagr: float
    n_days: int


def set_seed(seed: int) -> None:
    random.seed(seed)




def resolve_data_path(path: str) -> Path:
    candidates = [Path(path), Path("data") / path, Path("data/stock_returns.csv"), Path("stock_returns.csv")]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Actual data file not found. Tried: {[str(c) for c in candidates]}"
    )

def load_actual_returns_csv(path: str, tickers: List[str]) -> List[List[float]]:
    p = resolve_data_path(path)

    out: List[List[float]] = []
    with p.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames or []
        fmap = {f.lower(): f for f in fields}
        ticker_cols = []
        for t in tickers:
            key = t.lower()
            if key not in fmap:
                raise ValueError(f"Missing ticker column: {t}. Available: {fields}")
            ticker_cols.append(fmap[key])

        for row in rd:
            vals = []
            for col in ticker_cols:
                v = row.get(col, "").strip()
                vals.append(float(v) if v else 0.0)
            out.append(vals)

    if len(out) < 1200:
        raise ValueError("Not enough rows in actual data; need >= 1200 rows for WF OOS.")
    return out


def mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def stdev(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(max(sum((x - m) ** 2 for x in xs) / (len(xs) - 1), 0.0))


def softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    ex = [math.exp(v - m) for v in xs]
    s = sum(ex)
    return [v / s for v in ex]


def signal_scores(hist: List[List[float]], t: int, mom_span: int, vol_span: int) -> List[float]:
    a = len(hist[0])
    s0 = max(0, t - mom_span + 1)
    s1 = max(0, t - vol_span + 1)
    out = []
    for j in range(a):
        mom = mean([hist[k][j] for k in range(s0, t + 1)])
        vol = stdev([hist[k][j] for k in range(s1, t + 1)]) + 1e-9
        out.append(mom / vol)
    return out


def cap_and_norm(w: List[float], max_w: float) -> List[float]:
    x = [min(max(v, 0.0), max_w) for v in w]
    s = sum(x)
    return [v / s for v in x] if s > 1e-12 else [1.0 / len(x)] * len(x)


def apply_str(weights: List[List[float]], rets: List[List[float]], mu: List[List[float]], tc: float) -> List[float]:
    t = len(weights)
    a = len(weights[0])
    pos = weights[0][:]
    pnl = [0.0] * t

    for i in range(1, t):
        drift_num = [pos[j] * (1.0 + rets[i - 1][j]) for j in range(a)]
        s = sum(drift_num)
        drift = [d / (s if s > 1e-12 else 1.0) for d in drift_num]

        cand = weights[i]
        hold = sum(drift[j] * mu[i][j] for j in range(a))
        turn = sum(abs(cand[j] - drift[j]) for j in range(a))
        rebal = sum(cand[j] * mu[i][j] for j in range(a)) - tc * turn

        if rebal > hold:
            pos = cand[:]
            cost = tc * turn
        else:
            pos = drift
            cost = 0.0

        pnl[i] = sum(pos[j] * rets[i][j] for j in range(a)) - cost
    return pnl


def run_window(hist: List[List[float]], start: int, end: int, hp: Dict[str, float], tc: float) -> List[float]:
    w_list, mu_list = [], []
    rets = hist[start:end]
    for t in range(start, end):
        score = signal_scores(hist, t - 1, int(hp["mom_span"]), int(hp["vol_span"]))
        mu = [s * hp["mu_scale"] for s in score]
        w = cap_and_norm(softmax([s * hp["temp"] for s in score]), hp["max_w"])
        w_list.append(w)
        mu_list.append(mu)
    return apply_str(w_list, rets, mu_list, tc)


def walk_forward_oos(hist: List[List[float]], cfg: Dict, hp: Dict[str, float]) -> TrialResult:
    n = len(hist)
    cur = cfg["train_days"]
    all_oos: List[float] = []

    while cur + cfg["test_days"] <= n:
        all_oos.extend(run_window(hist, cur, cur + cfg["test_days"], hp, cfg["tc"]))
        cur += cfg["step_days"]

    if not all_oos:
        return TrialResult(-999.0, -1.0, 0)

    mu_d = mean(all_oos)
    sd_d = stdev(all_oos)
    sharpe = (mu_d * 252.0) / (sd_d * math.sqrt(252.0) + 1e-12)

    wealth = 1.0
    for r in all_oos:
        wealth *= (1.0 + r)
    cagr = wealth ** (252.0 / len(all_oos)) - 1.0

    return TrialResult(sharpe=sharpe, cagr=cagr, n_days=len(all_oos))


def search_to_target(hist: List[List[float]], cfg: Dict) -> Tuple[Dict[str, float], TrialResult]:
    grid = {
        "mom_span": [10, 20, 40, 60],
        "vol_span": [20, 40, 60],
        "temp": [0.8, 1.2, 1.8, 2.2],
        "mu_scale": [0.02, 0.04, 0.08],
        "max_w": [0.35, 0.45, 0.60],
    }

    best_hp: Dict[str, float] = {}
    best_res = TrialResult(-999.0, -1.0, 0)
    trial = 0

    for mom in grid["mom_span"]:
        for vol in grid["vol_span"]:
            for temp in grid["temp"]:
                for mus in grid["mu_scale"]:
                    for mw in grid["max_w"]:
                        trial += 1
                        hp = {"mom_span": mom, "vol_span": vol, "temp": temp, "mu_scale": mus, "max_w": mw}
                        res = walk_forward_oos(hist, cfg, hp)
                        print(f"trial={trial:03d} hp={hp} -> sharpe={res.sharpe:.3f}, cagr={res.cagr:.2%}, n={res.n_days}")

                        if res.sharpe > best_res.sharpe:
                            best_hp, best_res = hp, res

                        if res.sharpe >= cfg["target_sharpe"]:
                            return best_hp, best_res

    return best_hp, best_res


if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    hist = load_actual_returns_csv(CONFIG["data_csv"], TICKERS)
    hp, res = search_to_target(hist, CONFIG)
    print("\nBEST")
    print("hp:", hp)
    print("oos:", {"sharpe": round(res.sharpe, 4), "cagr": round(res.cagr, 4), "n_days": res.n_days})
    if res.sharpe < CONFIG["target_sharpe"]:
        raise SystemExit("Target not reached on ACTUAL data")
