# ======================================================================
# vix_multimodel_forecast_pipeline.py
# ----------------------------------------------------------------------
# Daily VIX multi-horizon level forecasting (h = 1, 5, 10 trading days).
#
# Data        : yfinance daily Close of ^VIX and SPY (S&P 500 ETF).
# Targets     : VIX close level at t+h for each horizon h.
# Models      : linear (OLS), lstm, cnn, transformer, inception, informer.
# Feature sets: notebook | vix_level | vix_ret | vix_spyret | vix_vixret_spyret
# Split       : train  1996-01-01 .. 2019-01-01   (single split, NO rolling)
#               test   2019-01-01 .. 2022-01-01
# Metrics     : per horizon, ranked by R^2 -> MAE, RMSE, R^2, AE-std.
#
# No look-ahead bias:
#   * feature engineering is strictly causal (ewm / trailing rolling only);
#   * train/test are split by TARGET date, so no test-period VIX value is
#     ever used as a training label;
#   * feature & target standardisers are fit on the training sub-set only
#     and merely applied to validation/test;
#   * the validation split used for early stopping is the chronological tail
#     of the training period (never future data).
#
# Windows / Dell friendly:
#   * CUDA can be disabled via CONFIG["use_cuda"] = False (or --no-cuda);
#   * AMP / fused-AdamW are only used on CUDA;
#   * multiprocessing uses the "spawn" context and is guarded by __main__;
#   * everything runs on CPU out of the box.
#
# Efficiency:
#   * features are fully vectorised (pandas ewm / rolling);
#   * sequence windows are built with numpy sliding_window_view (no loops);
#   * the (feature_set x model x horizon) grid can be run in parallel
#     across processes (CONFIG["n_workers"]).
#
# Model references (for documentation only):
#   LSTM           : Hochreiter & Schmidhuber (1997).
#   InceptionTime  : Ismail Fawaz et al. (2020), Data Min. Knowl. Disc.
#   Transformer    : Vaswani et al. (2017), NeurIPS.
#   Informer       : Zhou et al. (2021), AAAI (ProbSparse attention + distil).
# ======================================================================
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import sys
import time
import warnings
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# torch is imported lazily inside functions/workers so that the "download"
# stage and simple imports do not require torch to be present.

# ======================================================================
# Section 0 — Global constants
# ======================================================================
DAILY_ANN = 252.0          # daily annualisation factor for vol features
EPS = 1e-10

WORK = "vix_forecast_po"
os.makedirs(WORK, exist_ok=True)

CONFIG: dict = {
    # ----- data ------------------------------------------------------
    "vix_ticker": "^VIX",
    # "gsicspy" -> S&P 500 proxy. SPY is used because the feature sets
    # reference "spy ret". Switch to "^GSPC" here to use the index itself.
    "spy_ticker": "SPY",
    "data_start": "1996-01-01",     # training period begins here
    "data_end":   "2022-01-01",     # download end (== test_end, exclusive in yfinance)
    "return_type": "log",           # "log" or "pct" returns for the *_ret features

    # ----- split (single split, no rolling) --------------------------
    "train_start": "1996-01-01",
    "test_start":  "2019-01-01",
    "test_end":    "2022-01-01",
    "val_frac":    0.15,            # chronological tail of TRAIN used for early stopping

    # ----- experiment grid -------------------------------------------
    "horizons":     [1, 5, 10],
    "feature_sets": ["notebook", "vix_level", "vix_ret", "vix_spyret", "vix_vixret_spyret"],
    "models":       ["linear", "lstm", "cnn", "transformer", "inception", "informer"],

    # ----- sequence / common training --------------------------------
    "lookback":    20,
    "batch_size":  128,
    "max_epochs":  80,
    "patience":    10,
    "lr":          1e-3,
    "weight_decay": 1e-4,
    "seed":        42,

    # linear baseline: "flatten" (whole lookback window) or "last" (last step)
    "linreg_mode": "flatten",

    # ----- per-model hyper-parameters --------------------------------
    "lstm_hidden": 32, "lstm_layers": 1, "lstm_dropout": 0.10,

    "cnn_channels": 32, "cnn_kernel": 3, "cnn_layers": 2, "cnn_dropout": 0.10,

    "tr_d_model": 32, "tr_heads": 4, "tr_layers": 2, "tr_ff": 64, "tr_dropout": 0.10,

    "inc_filters": 32, "inc_depth": 3, "inc_bottleneck": 32,
    "inc_kernels": (9, 19, 39),

    "inf_d_model": 32, "inf_heads": 4, "inf_layers": 2, "inf_ff": 64,
    "inf_dropout": 0.10, "inf_factor": 5, "inf_distil": True,

    # ----- compute ---------------------------------------------------
    "use_cuda": True,               # set False (or --no-cuda) for CPU-only Dell box
    "use_amp": True,                # only honoured on CUDA
    "n_workers": 1,                 # >1 -> parallel process pool over the grid (CPU)
    "torch_threads": None,          # None -> os.cpu_count()

    # ----- io --------------------------------------------------------
    "cache_dir":  os.path.join(WORK, "cache"),
    "output_dir": os.path.join(WORK, "results"),
    "log_file":   os.path.join(WORK, "pipeline.log"),
    "force_download": False,
    "save_predictions": True,
}

os.makedirs(CONFIG["cache_dir"], exist_ok=True)
os.makedirs(CONFIG["output_dir"], exist_ok=True)

PREPARED_CACHE = os.path.join(CONFIG["cache_dir"], "vix_spy_daily.parquet")
PREPARED_CACHE_PKL = os.path.join(CONFIG["cache_dir"], "vix_spy_daily.pkl")


def _save_cache(df: pd.DataFrame) -> str:
    """Save the prepared frame, preferring parquet but falling back to pickle."""
    try:
        df.to_parquet(PREPARED_CACHE)
        return PREPARED_CACHE
    except Exception as exc:
        logging.warning("parquet unavailable (%s) -> using pickle cache instead", exc)
        df.to_pickle(PREPARED_CACHE_PKL)
        return PREPARED_CACHE_PKL


def _cache_exists() -> bool:
    return os.path.exists(PREPARED_CACHE) or os.path.exists(PREPARED_CACHE_PKL)


def _read_cache() -> pd.DataFrame:
    if os.path.exists(PREPARED_CACHE):
        try:
            return pd.read_parquet(PREPARED_CACHE)
        except Exception:
            pass
    return pd.read_pickle(PREPARED_CACHE_PKL)


def _configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(isinstance(h, logging.FileHandler) for h in root.handlers):
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] [%(processName)s] %(message)s",
        handlers=[logging.FileHandler(CONFIG["log_file"]), logging.StreamHandler()],
    )


_configure_logging()


# ======================================================================
# Section 1 — Reproducibility / device helpers
# ======================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def resolve_device(config: dict):
    import torch
    if config.get("use_cuda", True) and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def configure_threads(config: dict, divisor: int = 1) -> None:
    import torch
    n = config.get("torch_threads") or (os.cpu_count() or 1)
    n = max(1, int(n) // max(1, divisor))
    try:
        torch.set_num_threads(n)
    except Exception:
        pass


# ======================================================================
# Section 2 — yfinance daily download
# ======================================================================
def _extract_close(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Robustly pull the Close column whether yfinance returns flat or MultiIndex columns."""
    if isinstance(df.columns, pd.MultiIndex):
        # try ('Close', ticker) then any 'Close'
        if ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        else:
            close_block = df["Close"]
            s = close_block.iloc[:, 0] if isinstance(close_block, pd.DataFrame) else close_block
    else:
        s = df["Close"]
    return pd.Series(s).astype(float)


def download_daily(config: dict = CONFIG) -> pd.DataFrame:
    """
    Download daily Close of VIX and the S&P 500 proxy, align on common dates,
    compute returns, and cache to parquet. Returns a tidy DataFrame indexed
    by date with columns [vix, spy, vix_ret, spy_ret].
    """
    if not config.get("force_download", False) and _cache_exists():
        logging.info("Loading cached daily data")
        return _read_cache()

    import yfinance as yf

    vix_tkr = config["vix_ticker"]
    spy_tkr = config["spy_ticker"]
    start = config["data_start"]
    end = config["data_end"]

    logging.info("Downloading %s and %s from yfinance %s -> %s ...", vix_tkr, spy_tkr, start, end)
    vix_raw = yf.download(vix_tkr, start=start, end=end, auto_adjust=True, progress=False)
    spy_raw = yf.download(spy_tkr, start=start, end=end, auto_adjust=True, progress=False)
    if vix_raw is None or len(vix_raw) == 0:
        raise RuntimeError(f"No data returned for {vix_tkr}")
    if spy_raw is None or len(spy_raw) == 0:
        raise RuntimeError(f"No data returned for {spy_tkr}")

    vix = _extract_close(vix_raw, vix_tkr).rename("vix")
    spy = _extract_close(spy_raw, spy_tkr).rename("spy")

    df = pd.concat([vix, spy], axis=1).dropna()
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    if config["return_type"] == "log":
        df["vix_ret"] = np.log(df["vix"]).diff()
        df["spy_ret"] = np.log(df["spy"]).diff()
    else:
        df["vix_ret"] = df["vix"].pct_change()
        df["spy_ret"] = df["spy"].pct_change()
    df[["vix_ret", "spy_ret"]] = df[["vix_ret", "spy_ret"]].fillna(0.0)

    saved = _save_cache(df)
    logging.info("Saved %s: %d rows  %s -> %s",
                 saved, len(df), df.index[0].date(), df.index[-1].date())
    return df


def load_prepared(config: dict = CONFIG) -> pd.DataFrame:
    if not _cache_exists():
        return download_daily(config)
    return _read_cache()


# ======================================================================
# Section 3 — Causal, vectorised feature engineering
# ======================================================================
MOM_SPANS = [5, 15, 60]
VOL_SPANS = [15, 60]
BB_PERIOD = 20
DD_SPAN = 20
CORR_WINDOW = 60


def _ewm_mean(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=1).mean()


def _ewm_std(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=1).std().fillna(0.0) + EPS


def _per_asset_features(price: pd.Series, ret: pd.Series, prefix: str) -> pd.DataFrame:
    """Causal per-asset feature block: momentum z-scores, vols, %B, drawdown."""
    cols = {}
    for sp in MOM_SPANS:
        cols[f"{prefix}_mom{sp}"] = (_ewm_mean(ret, sp) / _ewm_std(ret, sp)).clip(-5, 5)
    for sp in VOL_SPANS:
        cols[f"{prefix}_vol{sp}"] = _ewm_std(ret, sp) * math.sqrt(DAILY_ANN)

    rm = price.rolling(BB_PERIOD, min_periods=1).mean()
    rs = price.rolling(BB_PERIOD, min_periods=1).std().fillna(0.0)
    upper = rm + 2 * rs
    lower = rm - 2 * rs
    bw = (upper - lower).replace(0.0, EPS)
    cols[f"{prefix}_bb"] = ((price - lower) / bw).clip(0, 1)

    peak = price.rolling(DD_SPAN, min_periods=1).max()
    cols[f"{prefix}_dd{DD_SPAN}"] = (price - peak) / peak.replace(0.0, EPS)

    return pd.DataFrame(cols, index=price.index)


def _pair_features(vix_ret: pd.Series, spy_ret: pd.Series) -> pd.DataFrame:
    spread = (vix_ret - spy_ret).rename("pair_spread")
    corr = vix_ret.rolling(CORR_WINDOW, min_periods=5).corr(spy_ret).fillna(0.0).rename("pair_corr60")
    return pd.concat([spread, corr], axis=1)


def build_feature_matrix(df: pd.DataFrame, feature_set: str) -> Tuple[np.ndarray, List[str]]:
    """
    Build the (T, n_features) causal feature matrix for the requested set.
    All transforms use only information up to and including time t.
    """
    vix, spy = df["vix"], df["spy"]
    vret, sret = df["vix_ret"], df["spy_ret"]

    if feature_set == "vix_level":
        F = pd.DataFrame({"vix": vix})
    elif feature_set == "vix_ret":
        F = pd.DataFrame({"vix": vix, "vix_ret": vret})
    elif feature_set == "vix_spyret":
        F = pd.DataFrame({"vix": vix, "spy_ret": sret})
    elif feature_set == "vix_vixret_spyret":
        F = pd.DataFrame({"vix": vix, "vix_ret": vret, "spy_ret": sret})
    elif feature_set == "notebook":
        F = pd.concat(
            [
                _per_asset_features(vix, vret, "vix"),
                _per_asset_features(spy, sret, "spy"),
                _pair_features(vret, sret),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    F = F.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return F.values.astype(np.float32), list(F.columns)


# ======================================================================
# Section 4 — Vectorised sequence building + leak-free split
# ======================================================================
def build_sequences(F: np.ndarray, target_level: np.ndarray, lookback: int, horizon: int):
    """
    X[j] = F[i-lookback+1 .. i]     (origin index i = lookback-1+j)
    y[j] = target_level[i + horizon]
    Only origins whose target index is in range are kept (no padding, fully causal).
    Returns X (N,L,nf), y (N,), origin_idx (N,), target_idx (N,).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    T, nf = F.shape
    if T < lookback + horizon:
        return (np.empty((0, lookback, nf), np.float32), np.empty((0,), np.float32),
                np.empty((0,), int), np.empty((0,), int))

    W = sliding_window_view(F, window_shape=lookback, axis=0)   # (T-L+1, nf, L)
    W = np.moveaxis(W, 2, 1).astype(np.float32)                 # (T-L+1, L, nf)

    origin_idx = np.arange(lookback - 1, T)                     # one per window
    target_idx = origin_idx + horizon
    valid = target_idx <= (T - 1)

    X = np.ascontiguousarray(W[valid])
    oi = origin_idx[valid]
    ti = target_idx[valid]
    y = target_level[ti].astype(np.float32)
    return X, y, oi, ti


def build_dataset_for_task(df: pd.DataFrame, feature_set: str, horizon: int, config: dict):
    """
    Build train/val/test arrays for one (feature_set, horizon) task with no
    look-ahead. Split is by TARGET date.
    """
    F, feat_names = build_feature_matrix(df, feature_set)
    target_level = df["vix"].values.astype(np.float32)
    dates = df.index

    X, y, oi, ti = build_sequences(F, target_level, config["lookback"], horizon)
    target_dates = dates[ti]

    train_start = pd.Timestamp(config["train_start"])
    test_start = pd.Timestamp(config["test_start"])
    test_end = pd.Timestamp(config["test_end"])

    train_mask = (target_dates >= train_start) & (target_dates < test_start)
    test_mask = (target_dates >= test_start) & (target_dates < test_end)

    Xtr_all, ytr_all = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    test_dates = target_dates[test_mask]

    # chronological validation tail (the data is already in date order)
    n_tr = len(Xtr_all)
    n_val = max(1, int(round(n_tr * config["val_frac"]))) if n_tr > 1 else 0
    n_sub = n_tr - n_val
    Xtr, ytr = Xtr_all[:n_sub], ytr_all[:n_sub]
    Xvl, yvl = Xtr_all[n_sub:], ytr_all[n_sub:]

    return {
        "Xtr": Xtr, "ytr": ytr, "Xvl": Xvl, "yvl": yvl,
        "Xtr_all": Xtr_all, "ytr_all": ytr_all,
        "Xte": Xte, "yte": yte, "test_dates": test_dates,
        "feat_names": feat_names, "n_features": F.shape[1],
    }


# ======================================================================
# Section 5 — Standardisers (fit on TRAIN sub-set only -> no leakage)
# ======================================================================
class SeqScaler:
    """Standardise (N,L,F) sequences per feature using train statistics."""

    def fit(self, X: np.ndarray):
        flat = X.reshape(-1, X.shape[-1])
        self.mu_ = flat.mean(axis=0)
        self.sd_ = flat.std(axis=0)
        self.sd_[self.sd_ < EPS] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu_) / self.sd_).astype(np.float32)


class TargetScaler:
    def fit(self, y: np.ndarray):
        self.mu_ = float(np.mean(y))
        self.sd_ = float(np.std(y))
        if self.sd_ < EPS:
            self.sd_ = 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.mu_) / self.sd_).astype(np.float32)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y) * self.sd_ + self.mu_).astype(np.float32)


# ======================================================================
# Section 6 — Torch models
# ======================================================================
def _build_torch_modules():
    """Define and return the torch model classes (lazy import of torch)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class LSTMRegressor(nn.Module):
        def __init__(self, input_size, cfg, lookback):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, cfg["lstm_hidden"], cfg["lstm_layers"],
                batch_first=True,
                dropout=cfg["lstm_dropout"] if cfg["lstm_layers"] > 1 else 0.0,
            )
            self.head = nn.Linear(cfg["lstm_hidden"], 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    class CNNRegressor(nn.Module):
        def __init__(self, input_size, cfg, lookback):
            super().__init__()
            layers, c_in = [], input_size
            k = cfg["cnn_kernel"]
            for _ in range(cfg["cnn_layers"]):
                layers += [
                    nn.Conv1d(c_in, cfg["cnn_channels"], k, padding=k // 2),
                    nn.BatchNorm1d(cfg["cnn_channels"]),
                    nn.ReLU(),
                    nn.Dropout(cfg["cnn_dropout"]),
                ]
                c_in = cfg["cnn_channels"]
            self.net = nn.Sequential(*layers)
            self.head = nn.Linear(cfg["cnn_channels"], 1)

        def forward(self, x):
            x = x.transpose(1, 2)        # (B, F, L)
            x = self.net(x)
            x = x.mean(dim=2)            # global average pool
            return self.head(x).squeeze(-1)

    class TransformerRegressor(nn.Module):
        def __init__(self, input_size, cfg, lookback):
            super().__init__()
            d = cfg["tr_d_model"]
            self.embed = nn.Linear(input_size, d)
            self.pos = PositionalEncoding(d, max_len=lookback + 16)
            enc = nn.TransformerEncoderLayer(
                d_model=d, nhead=cfg["tr_heads"], dim_feedforward=cfg["tr_ff"],
                dropout=cfg["tr_dropout"], batch_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc, cfg["tr_layers"])
            self.head = nn.Linear(d, 1)

        def forward(self, x):
            x = self.pos(self.embed(x))
            x = self.encoder(x)
            x = x.mean(dim=1)
            return self.head(x).squeeze(-1)

    class InceptionModule(nn.Module):
        def __init__(self, in_ch, n_filters, kernel_sizes, bottleneck):
            super().__init__()
            self.use_bottleneck = in_ch > 1
            bch = bottleneck if self.use_bottleneck else in_ch
            self.bottleneck = (nn.Conv1d(in_ch, bottleneck, 1, bias=False)
                               if self.use_bottleneck else nn.Identity())
            self.convs = nn.ModuleList(
                [nn.Conv1d(bch, n_filters, k, padding=k // 2, bias=False) for k in kernel_sizes]
            )
            self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
            self.convpool = nn.Conv1d(in_ch, n_filters, 1, bias=False)
            self.bn = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))
            self.act = nn.ReLU()

        def forward(self, x):
            inp = x
            xb = self.bottleneck(x)
            outs = [c(xb) for c in self.convs]
            outs.append(self.convpool(self.maxpool(inp)))
            x = torch.cat(outs, dim=1)
            return self.act(self.bn(x))

    class InceptionTimeRegressor(nn.Module):
        def __init__(self, input_size, cfg, lookback):
            super().__init__()
            nf = cfg["inc_filters"]
            ks = tuple(cfg["inc_kernels"])
            bn = cfg["inc_bottleneck"]
            self.depth = cfg["inc_depth"]
            out_ch = nf * (len(ks) + 1)
            self.blocks = nn.ModuleList()
            self.shortcuts = nn.ModuleList()
            ch = input_size
            res_in = input_size
            for d in range(self.depth):
                self.blocks.append(InceptionModule(ch, nf, ks, bn))
                ch = out_ch
                if d % 3 == 2:
                    self.shortcuts.append(
                        nn.Sequential(nn.Conv1d(res_in, out_ch, 1, bias=False),
                                      nn.BatchNorm1d(out_ch))
                    )
                    res_in = out_ch
            self.act = nn.ReLU()
            self.head = nn.Linear(out_ch, 1)

        def forward(self, x):
            x = x.transpose(1, 2)        # (B, F, L)
            res, si = x, 0
            for d in range(self.depth):
                x = self.blocks[d](x)
                if d % 3 == 2:
                    x = self.act(x + self.shortcuts[si](res))
                    res, si = x, si + 1
            x = x.mean(dim=2)
            return self.head(x).squeeze(-1)

    # -------- Informer (ProbSparse attention + distilling) -----------
    class ProbAttention(nn.Module):
        def __init__(self, factor=5, scale=None, dropout=0.1):
            super().__init__()
            self.factor = factor
            self.scale = scale
            self.dropout = nn.Dropout(dropout)

        def _prob_QK(self, Q, K, sample_k, n_top):
            B, H, L_K, E = K.shape
            _, _, L_Q, _ = Q.shape
            K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
            idx = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
            K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), idx, :]
            Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
            M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
            M_top = M.topk(n_top, sorted=False)[1]
            Q_reduce = Q[torch.arange(B)[:, None, None],
                         torch.arange(H)[None, :, None],
                         M_top, :]
            Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
            return Q_K, M_top

        def _get_initial_context(self, V, L_Q):
            B, H, L_V, D = V.shape
            V_sum = V.mean(dim=-2)
            return V_sum.unsqueeze(-2).expand(B, H, L_Q, D).clone()

        def _update_context(self, context, V, scores, index):
            B, H, L_V, D = V.shape
            attn = torch.softmax(scores, dim=-1)
            context[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :] = torch.matmul(attn, V).type_as(context)
            return context

        def forward(self, q, k, v):
            B, L_Q, H, D = q.shape
            _, L_K, _, _ = k.shape
            q = q.transpose(2, 1); k = k.transpose(2, 1); v = v.transpose(2, 1)
            U_part = min(L_K, max(1, self.factor * int(np.ceil(np.log(max(L_K, 2))))))
            u = min(L_Q, max(1, self.factor * int(np.ceil(np.log(max(L_Q, 2))))))
            scores_top, index = self._prob_QK(q, k, sample_k=U_part, n_top=u)
            scores_top = scores_top * (self.scale or 1.0 / math.sqrt(D))
            context = self._get_initial_context(v, L_Q)
            context = self._update_context(context, v, scores_top, index)
            return context.transpose(2, 1).contiguous()

    class AttentionLayer(nn.Module):
        def __init__(self, attention, d_model, n_heads):
            super().__init__()
            dk = d_model // n_heads
            self.inner = attention
            self.q_proj = nn.Linear(d_model, dk * n_heads)
            self.k_proj = nn.Linear(d_model, dk * n_heads)
            self.v_proj = nn.Linear(d_model, dk * n_heads)
            self.out_proj = nn.Linear(dk * n_heads, d_model)
            self.n_heads = n_heads

        def forward(self, x):
            B, L, _ = x.shape
            H = self.n_heads
            q = self.q_proj(x).view(B, L, H, -1)
            k = self.k_proj(x).view(B, L, H, -1)
            v = self.v_proj(x).view(B, L, H, -1)
            out = self.inner(q, k, v).view(B, L, -1)
            return self.out_proj(out)

    class ConvLayer(nn.Module):       # distilling
        def __init__(self, c_in):
            super().__init__()
            self.down = nn.Conv1d(c_in, c_in, 3, padding=1, padding_mode="circular")
            self.norm = nn.BatchNorm1d(c_in)
            self.act = nn.ELU()
            self.pool = nn.MaxPool1d(3, stride=2, padding=1)

        def forward(self, x):
            x = self.down(x.transpose(1, 2))
            x = self.act(self.norm(x))
            x = self.pool(x)
            return x.transpose(1, 2)

    class InformerEncoderLayer(nn.Module):
        def __init__(self, attn, d_model, d_ff, dropout):
            super().__init__()
            self.attn = attn
            self.conv1 = nn.Conv1d(d_model, d_ff, 1)
            self.conv2 = nn.Conv1d(d_ff, d_model, 1)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = x + self.dropout(self.attn(x))
            x = self.norm1(x)
            y = self.dropout(F.relu(self.conv1(x.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
            return self.norm2(x + y)

    class InformerRegressor(nn.Module):
        def __init__(self, input_size, cfg, lookback):
            super().__init__()
            d = cfg["inf_d_model"]
            self.embed = nn.Linear(input_size, d)
            self.pos = PositionalEncoding(d, max_len=lookback + 16)
            e = cfg["inf_layers"]
            self.attn_layers = nn.ModuleList([
                InformerEncoderLayer(
                    AttentionLayer(ProbAttention(cfg["inf_factor"], dropout=cfg["inf_dropout"]),
                                   d, cfg["inf_heads"]),
                    d, cfg["inf_ff"], cfg["inf_dropout"])
                for _ in range(e)
            ])
            self.conv_layers = (nn.ModuleList([ConvLayer(d) for _ in range(e - 1)])
                                if (cfg["inf_distil"] and e > 1) else None)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)

        def forward(self, x):
            x = self.pos(self.embed(x))
            if self.conv_layers is not None:
                for a, c in zip(self.attn_layers[:-1], self.conv_layers):
                    x = c(a(x))
                x = self.attn_layers[-1](x)
            else:
                for a in self.attn_layers:
                    x = a(x)
            x = self.norm(x).mean(dim=1)
            return self.head(x).squeeze(-1)

    return {
        "lstm": LSTMRegressor,
        "cnn": CNNRegressor,
        "transformer": TransformerRegressor,
        "inception": InceptionTimeRegressor,
        "informer": InformerRegressor,
    }


# ======================================================================
# Section 7 — Training utilities (torch)
# ======================================================================
def _iter_batches(n: int, bs: int, shuffle: bool, generator=None):
    import torch
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    batches, s = [], 0
    while s < n:
        e = min(s + bs, n)
        batches.append(idx[s:e]); s = e
    # avoid a trailing singleton batch (breaks BatchNorm during training)
    if len(batches) >= 2 and len(batches[-1]) == 1:
        last = batches.pop()
        batches[-1] = torch.cat([batches[-1], last])
    return batches


def train_torch_model(model, Xtr, ytr, Xvl, yvl, config, device, seed):
    import torch
    import torch.nn.functional as F
    import torch.optim as optim

    set_seed(seed)
    model.to(device)

    amp = bool(config.get("use_amp", True)) and device.type == "cuda"
    bs = config["batch_size"]

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    has_val = len(Xvl) > 0
    if has_val:
        Xvl_t = torch.from_numpy(Xvl).to(device)
        yvl_t = torch.from_numpy(yvl).to(device)

    try:
        opt = optim.AdamW(model.parameters(), lr=config["lr"],
                          weight_decay=config["weight_decay"],
                          fused=(device.type == "cuda"))
    except Exception:
        opt = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    rng = torch.Generator(device="cpu").manual_seed(seed)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    wait, patience = 0, config["patience"]
    n = len(Xtr_t)

    for _ in range(config["max_epochs"]):
        model.train()
        for idx in _iter_batches(n, bs, shuffle=True, generator=rng):
            idx = idx.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                pred = model(Xtr_t[idx])
                loss = F.mse_loss(pred, ytr_t[idx])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        if has_val:
            model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
                vloss = F.mse_loss(model(Xvl_t), yvl_t).item()
        else:
            vloss = loss.item()

        if np.isfinite(vloss) and vloss < best_val - 1e-9:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    model.load_state_dict(best_state)
    return model


def predict_torch(model, X, device, bs=4096):
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            xb = torch.from_numpy(X[s:s + bs]).to(device)
            outs.append(model(xb).detach().cpu().numpy().reshape(-1))
    return np.concatenate(outs) if outs else np.empty((0,), np.float32)


# ======================================================================
# Section 8 — Metrics
# ======================================================================
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_true - y_pred
    ae = np.abs(err)
    mae = float(ae.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    ae_std = float(ae.std(ddof=1)) if len(ae) > 1 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "AE_std": ae_std, "n_test": int(len(y_true))}


# ======================================================================
# Section 9 — One task = (feature_set, model, horizon)
# ======================================================================
def run_task(task: dict, config: dict) -> Optional[dict]:
    """
    Train + evaluate a single (feature_set, model, horizon) combination.
    Picklable / process-pool safe: loads data from the parquet cache itself.
    """
    feature_set = task["feature_set"]
    model_name = task["model"]
    horizon = task["horizon"]

    df = load_prepared(config)
    data = build_dataset_for_task(df, feature_set, horizon, config)

    if len(data["Xte"]) == 0 or len(data["Xtr_all"]) == 0:
        logging.warning("Empty split for %s/%s/h%d", feature_set, model_name, horizon)
        return None

    seed = config["seed"]
    t0 = time.time()

    if model_name == "linear":
        # ---- OLS baseline: full pre-2019 training, no early stopping ----
        xs = SeqScaler().fit(data["Xtr_all"])
        ts = TargetScaler().fit(data["ytr_all"])

        def _flat(X):
            if config["linreg_mode"] == "last":
                return X[:, -1, :]
            return X.reshape(X.shape[0], -1)

        Xtr = _flat(xs.transform(data["Xtr_all"]))
        Xte = _flat(xs.transform(data["Xte"]))
        ytr = ts.transform(data["ytr_all"])

        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(Xtr, ytr)
        pred_std = reg.predict(Xte)
        y_pred = ts.inverse(pred_std)

    else:
        # ---- neural models: train sub-set + chronological val early stop ----
        import torch
        device = torch.device("cpu") if task.get("force_cpu", False) else resolve_device(config)
        configure_threads(config, divisor=(config["n_workers"] if task.get("force_cpu") else 1))
        models = _build_torch_modules()
        model_cls = models[model_name]

        xs = SeqScaler().fit(data["Xtr"])            # fit on TRAIN-sub only
        ts = TargetScaler().fit(data["ytr"])

        Xtr = xs.transform(data["Xtr"]); ytr = ts.transform(data["ytr"])
        Xvl = xs.transform(data["Xvl"]); yvl = ts.transform(data["yvl"])
        Xte = xs.transform(data["Xte"])

        set_seed(seed)
        model = model_cls(data["n_features"], config, config["lookback"])
        model = train_torch_model(model, Xtr, ytr, Xvl, yvl, config, device, seed)
        pred_std = predict_torch(model, Xte, device)
        y_pred = ts.inverse(pred_std)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    y_true = data["yte"]
    metrics = regression_metrics(y_true, y_pred)
    elapsed = time.time() - t0

    row = {
        "horizon": horizon,
        "feature_set": feature_set,
        "model": model_name,
        **metrics,
        "train_secs": round(elapsed, 2),
    }

    if config.get("save_predictions", True):
        pred_df = pd.DataFrame(
            {"date": data["test_dates"], "y_true": y_true, "y_pred": y_pred}
        )
        fn = os.path.join(config["output_dir"],
                          f"pred_h{horizon}_{feature_set}_{model_name}.csv")
        pred_df.to_csv(fn, index=False)

    logging.info("done %s/%s/h%-2d | R2=%.4f RMSE=%.3f MAE=%.3f (%.1fs)",
                 feature_set, model_name, horizon, metrics["R2"],
                 metrics["RMSE"], metrics["MAE"], elapsed)
    return row


# ======================================================================
# Section 10 — Orchestration (sequential or parallel grid)
# ======================================================================
def build_task_list(config: dict) -> List[dict]:
    tasks = []
    for h, fs, m in product(config["horizons"], config["feature_sets"], config["models"]):
        tasks.append({"feature_set": fs, "model": m, "horizon": h})
    return tasks


def _task_worker(args):
    task, config = args
    task = dict(task); task["force_cpu"] = True
    try:
        return run_task(task, config)
    except Exception as exc:  # never kill the pool
        logging.error("task %s failed: %s", task, exc, exc_info=True)
        return {"horizon": task["horizon"], "feature_set": task["feature_set"],
                "model": task["model"], "error": str(exc)}


def run_grid(config: dict = CONFIG) -> pd.DataFrame:
    download_daily(config)            # ensure cache exists before workers load it
    tasks = build_task_list(config)
    logging.info("Total tasks: %d (%d horizons x %d feature sets x %d models)",
                 len(tasks), len(config["horizons"]),
                 len(config["feature_sets"]), len(config["models"]))

    n_workers = max(1, int(config.get("n_workers", 1)))
    rows: List[dict] = []

    if n_workers == 1:
        configure_threads(config, divisor=1)
        try:
            from tqdm import tqdm
            it = tqdm(tasks, desc="tasks")
        except Exception:
            it = tasks
        for task in it:
            r = run_task(task, config)
            if r is not None:
                rows.append(r)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_task_worker, (t, config)) for t in tasks]
            for fut in as_completed(futs):
                r = fut.result()
                if r is not None:
                    rows.append(r)

    df = pd.DataFrame(rows)
    if df.empty:
        logging.error("No results produced.")
        return df

    return finalize_and_report(df, config)


def finalize_and_report(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out_dir = config["output_dir"]
    full_csv = os.path.join(out_dir, "results_all.csv")
    df.to_csv(full_csv, index=False)

    ok = df[~df.get("error", pd.Series([np.nan] * len(df))).notna()] if "error" in df else df

    print("\n" + "=" * 78)
    print("  VIX FORECAST RESULTS  (test 2019-01-01 .. 2022-01-01, ranked by R^2)")
    print("=" * 78)
    cols = ["model", "feature_set", "R2", "RMSE", "MAE", "AE_std", "n_test", "train_secs"]
    for h in sorted(ok["horizon"].unique()):
        sub = ok[ok["horizon"] == h].copy()
        sub = sub.sort_values("R2", ascending=False)
        sub_csv = os.path.join(out_dir, f"results_h{h}.csv")
        sub.to_csv(sub_csv, index=False)
        print(f"\n--- Horizon h = {h} trading day(s) ---")
        present = [c for c in cols if c in sub.columns]
        with pd.option_context("display.float_format", lambda v: f"{v:,.4f}"):
            print(sub[present].to_string(index=False))

    print("\nSaved:")
    print(f"  {full_csv}")
    for h in sorted(ok["horizon"].unique()):
        print(f"  {os.path.join(out_dir, f'results_h{h}.csv')}")
    print("=" * 78 + "\n")
    return df


# ======================================================================
# Section 11 — CLI
# ======================================================================
def _in_jupyter() -> bool:
    if "ipykernel" in sys.modules or "IPython" in sys.modules:
        return True
    a0 = (sys.argv[0] or "").lower()
    return any(k in a0 for k in ["ipykernel", "jupyter", "colab"])


def main():
    ap = argparse.ArgumentParser(description="Daily VIX multi-horizon multi-model forecasting")
    ap.add_argument("stage", nargs="?", default="all", choices=["download", "run", "all"])
    ap.add_argument("--no-cuda", action="store_true", help="force CPU even if a GPU is present")
    ap.add_argument("--workers", type=int, default=None, help="parallel processes over the grid")
    ap.add_argument("--force-download", action="store_true")
    args, _ = ap.parse_known_args()

    if args.no_cuda:
        CONFIG["use_cuda"] = False
    if args.workers is not None:
        CONFIG["n_workers"] = args.workers
    if args.force_download:
        CONFIG["force_download"] = True

    if args.stage in ("download", "all"):
        download_daily(CONFIG)
    if args.stage in ("run", "all"):
        run_grid(CONFIG)


if __name__ == "__main__":
    if _in_jupyter():
        logging.info("Detected Jupyter — call run_grid(CONFIG) directly.")
    else:
        main()
