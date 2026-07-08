"""
Self-contained VIX-gated price-trend CNN.

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

The embedded CNN runner follows the compact/fast reference form from the original
price-trend script. The module trains from the chart-image pickle in each call
and does not require price_trend_cnn.py to be present. It also does not read a
pre-existing predictions.csv, so it cannot silently pick up a cache from another
model run.
"""
# ======================================================================
# price_trend_cnn.py
# ----------------------------------------------------------------------
# Return forecasting from OHLC chart IMAGES, re-implementing the CNN of
# Jiang, Kelly & Xiu (2023, "(Re-)Imag(in)ing Price Trends", RFS) on top
# of the engineering scaffold of the IV-surface runner (walk-forward,
# leak control, seed-ensembling, portfolio evaluation, CLI). Everything
# lives in this single importable module.
#
# Data: one pickle of stock-day charts, e.g.
#   R1000_picMap20MA_price_chart_image_df.pickle   (also sp500_5MA / sp600_5MA)
# columns: sedol, date, days, ph, vh, MA, idx, rF5D, rF20D, rF60D,
#          img_height, img_width, price_chart_image
# where price_chart_image is a FLATTENED binary image (0/1) reshaped
# C-order to (img_height, img_width); `ph`/`vh` are the price/volume
# sub-heights, `MA` flags the moving-average line, `days` the look-back
# (5/20/60). The file is read from the CURRENT WORKING DIRECTORY by
# default, so the notebook just sits next to it.
#
# Two design switches:
#   task     : "clf" -> Jiang-Kelly-Xiu classification (label = 1{rF>0},
#              softmax + cross-entropy, rank by P(up)); "reg" -> regression
#              on the forward return (tanh-bounded MSE, rank by prediction).
#   protocol : "faithful" -> train+validate ONCE on the initial window
#              (70/30 random split), freeze for the whole OOS test (the
#              paper's protocol); "walkforward" -> expanding/rolling refit
#              with an embargo. Long-short decile (H-L) and long-only
#              top-decile portfolios are always both reported.
#
# ----- EXTRA EXPERIMENTS (all controlled from the notebook by kwargs /
#       the EXPERIMENTS presets; everything still lives in THIS one file) -
#   (1) CROSS-UNIVERSE  : train on sp500, test on sp600 (test_pattern=...).
#       Test formation is forced STRICTLY AFTER the train label window
#       (+embargo) so the two universes never overlap in time.
#   (2) PERSISTENCE      : predictions AND the trained model (ensemble
#       state-dicts + scalers) are saved every run; reload with
#       load_predictor().
#   (3) HALF-YEAR BETA   : beta_over_time() regresses realised return on the
#       (standardised) predicted score in each half-year -> beta-vs-time plot,
#       i.e. "did the score separate stocks that period?".
#   (4) PORTFOLIO TAPE   : long-short and long-only PERIOD returns are saved
#       WITH the exact rebalance date; decile_profile() saves the average
#       realised return of each score-decile over the whole OOS.
#   (5) VIX CHANNEL      : add_vix=True appends `vix_rows` rows of a
#       column-aligned, SEPARATELY-scaled VIX strip under the volume block
#       (VIX from yfinance, cached to vix.csv; scaler fit on TRAIN only;
#       the VIX window length aligns to the image look-back).
#   (6) PER-IMAGE NORM   : pixel_norm="per_image" normalises every chart by
#       its own mean/std instead of the global (train) pixel mean/std.
#   (7) 15-YEAR TRAIN    : train_years=15.
#   (8) sp500_20MA h=5   : 20-day images, horizon=5 (weekly rebalance).
#
# No-look-ahead contract (preserved for every experiment):
#   * the label is a STRICTLY FORWARD return rF{h}D over (t, t+h]; the image
#     ends at t, so today's return never predicts itself;
#   * training images are embargoed by the horizon so no training label
#     window reaches into the test period; cross-universe tests additionally
#     force the test start strictly after the train cut;
#   * the pixel normaliser, the y-scaler, the VIX scaler and any QC threshold
#     are fit on the TRAINING window ONLY; early stopping uses a validation
#     split carved out of training, never the test set.
#
# Quick start (import):
#   from price_trend_cnn import run, run_experiment, EXPERIMENTS, make_synth_suite
#   make_synth_suite()                            # optional offline fixtures
#   res = run()                                   # 20D, clf, faithful
#   res = run_experiment("cross_sp500_to_sp600")  # one of the presets
#   res["portfolio"].head(); res["summary"]; res["beta"]; res["decile_profile"]
#
# CLI:
#   python price_trend_cnn.py --task clf --protocol faithful --horizon 20
#   python price_trend_cnn.py --experiment vix_sp500_5MA
# ======================================================================
from __future__ import annotations
import argparse, fnmatch, glob, json, logging, time
from pathlib import Path
import numpy as np
import pandas as pd

log = logging.getLogger("price_trend_cnn")

# ---- canonical Jiang-Kelly-Xiu geometries (days -> H, W, ph, vh) ----
GEOMETRY = {5: (32, 15, 25, 6), 20: (64, 60, 51, 12), 60: (96, 180, 76, 19)}
# Only the first conv block differs by horizon; later blocks use stride (1,1),
# dilation (1,1), padding (2,1). This reproduces the paper's flattened FC sizes
# 15,360 / 46,080 / 184,320 for the 5/20/60-day models.
FIRST_BLOCK = {
    5:  dict(stride=(1, 1), dilation=(1, 1), padding=(2, 1)),
    20: dict(stride=(3, 1), dilation=(2, 1), padding=(12, 1)),
    60: dict(stride=(3, 1), dilation=(3, 1), padding=(12, 1)),
}
N_BLOCKS = {5: 2, 20: 3, 60: 4}
FILTERS = [64, 128, 256, 512]


def _setup_log(data_dir=None, level=logging.DEBUG):
    if log.handlers:
        return
    log.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); sh.setLevel(logging.INFO)
    log.addHandler(sh)
    try:
        d = Path(data_dir) if data_dir else Path.cwd()
        fh = logging.FileHandler(d / "price_trend_cnn.log"); fh.setFormatter(fmt)
        log.addHandler(fh)
    except Exception:
        pass


def _hint(msg, suggestion):
    log.error(msg); log.error("SUGGESTION: " + suggestion)
    raise RuntimeError(f"{msg} | {suggestion}")


def _ensure_dir(p):
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p


# ======================================================================
# PART 1 -- image reshape + look-ahead-free quality control (vectorized)
# ======================================================================
def to_images(flat, H, W, order="C"):
    """Stack/reshape flattened chart vectors into an (N, H, W) float32 cube.
    `flat` may be 2-D (N,H*W), 1-D (H*W,), or a Series/list of per-row arrays."""
    A = np.asarray(flat)
    if A.dtype == object:
        A = np.stack([np.ravel(np.asarray(v)) for v in A])
    A = np.asarray(A, dtype=np.float32)
    if A.ndim == 1:
        A = A[None, :]
    if A.shape[1] != H * W:
        raise ValueError(f"flat length {A.shape[1]} != H*W = {H*W}")
    return A.reshape(-1, H, W) if order == "C" else \
        np.stack([a.reshape(W, H).T for a in A]).astype(np.float32)


def _as_cube(G):
    G = np.asarray(G, dtype=np.float32)
    return G[None] if G.ndim == 2 else G


def price_region(G, ph):  return _as_cube(G)[:, :ph, :]      # OHLC + MA sub-image
def volume_region(G, vh): return _as_cube(G)[:, -vh:, :]     # volume-bar sub-image


# ---- per-image QC metrics (pure function of a single image -> safe at test) --
def m_white(G):
    G = _as_cube(G); return G.reshape(len(G), -1).mean(1)


def m_price_white(G, ph):
    P = price_region(G, ph); return P.reshape(len(P), -1).mean(1)


def m_blank_cols(G, ph):
    """Fraction of fully-black columns in the price region. High => missing
    days (JKX leave missing days black). Pure per-image incompleteness signal."""
    P = price_region(G, ph)
    return 1.0 - (P.sum(1) > 0).mean(1)


def m_vextent(G, ph):
    """Fraction of price rows that contain any object (low => flat/degenerate)."""
    P = price_region(G, ph)
    return (P.sum(2) > 0).mean(1)


def m_region_balance(G, ph, vh):
    """white(volume)/white(price): unusually large => volume artefact."""
    Gc = _as_cube(G)
    pw = price_region(Gc, ph).reshape(len(Gc), -1).sum(1)
    vw = volume_region(Gc, vh).reshape(len(Gc), -1).sum(1)
    return vw / (pw + 1e-9)


METRICS = {
    "white": lambda G, ph, vh: m_white(G),
    "price_white": lambda G, ph, vh: m_price_white(G, ph),
    "blank_cols": lambda G, ph, vh: m_blank_cols(G, ph),
    "vextent": lambda G, ph, vh: m_vextent(G, ph),
    "region_balance": lambda G, ph, vh: m_region_balance(G, ph, vh),
}
METRIC_NAMES = list(METRICS) + ["composite"]


def all_metrics(G, ph, vh):
    return {k: f(G, ph, vh) for k, f in METRICS.items()}


def qc_score(G, ph, vh, metric="composite", train=None, metrics=None):
    """Per-image QC value (higher = more suspect). 'composite' calibrates robust
    z-scores on `train` images only; single metrics are intrinsic to each image."""
    if metric == "composite":
        sc = ImageQC(metrics=metrics) if metrics else ImageQC()
        sc.fit(G if train is None else train, ph, vh)
        return sc.score(G, ph, vh)
    if metric in METRICS:
        return METRICS[metric](G, ph, vh)
    raise ValueError(f"unknown metric {metric!r}; choose from {METRIC_NAMES}")


class ImageQC:
    """Train-calibrated composite image-quality score (fit on TRAIN only)."""
    _FLIP = {"vextent"}                      # low extent is the bad direction

    def __init__(self, metrics=("blank_cols", "vextent", "region_balance")):
        self.metrics = tuple(metrics)
        self.med_, self.mad_, self.train_score_ = {}, {}, None

    def fit(self, G_train, ph, vh):
        for k in self.metrics:
            v = METRICS[k](G_train, ph, vh)
            self.med_[k] = float(np.median(v))
            self.mad_[k] = float(np.median(np.abs(v - self.med_[k])) + 1e-9)
        self._ph, self._vh = ph, vh
        self.train_score_ = self.score(G_train, ph, vh)
        return self

    def score(self, G, ph, vh):
        Z = []
        for k in self.metrics:
            z = (METRICS[k](G, ph, vh) - self.med_[k]) / (1.4826 * self.mad_[k])
            Z.append(-z if k in self._FLIP else z)
        return np.clip(np.stack(Z, 0), 0, None).mean(0)

    def threshold(self, q):
        return float(np.quantile(self.train_score_, q))


def despeckle(G, H, W):
    """Remove isolated single white pixels (no 4-neighbour lit) -> (N, H*W).
    Aggressive for thin chart lines, so OFF by default; provided for parity."""
    Gc = _as_cube(G).copy()
    up = np.zeros_like(Gc); up[:, 1:, :] = Gc[:, :-1, :]
    dn = np.zeros_like(Gc); dn[:, :-1, :] = Gc[:, 1:, :]
    lf = np.zeros_like(Gc); lf[:, :, 1:] = Gc[:, :, :-1]
    rt = np.zeros_like(Gc); rt[:, :, :-1] = Gc[:, :, 1:]
    Gc[(Gc > 0) & (up + dn + lf + rt == 0)] = 0.0
    return Gc.reshape(len(Gc), -1)


# ======================================================================
# PART 2 -- the Jiang-Kelly-Xiu CNN + trainable, savable Predictor
# ======================================================================
def build_cnn(days, H, W, task="clf"):
    """Stacked [Conv(5x3) -> BatchNorm -> LeakyReLU -> MaxPool(2x1)] blocks
    (2/3/4 for 5/20/60-day; filters 64,128,256,512), flatten -> Dropout(0.5)
    -> Linear. task='clf' -> 2-logit head; 'reg' -> single output. Flatten size
    inferred by a dummy forward pass so any H,W works (incl. the +VIX rows)."""
    import torch, torch.nn as nn
    nb = N_BLOCKS.get(days, 3)
    first = FIRST_BLOCK.get(days, dict(stride=(1, 1), dilation=(1, 1), padding=(2, 1)))

    class CNN(nn.Module):
        def __init__(s):
            super().__init__()
            blocks, c = [], 1
            for b in range(nb):
                cfg = first if b == 0 else dict(stride=(1, 1), dilation=(1, 1),
                                                padding=(2, 1))
                blocks += [nn.Conv2d(c, FILTERS[b], (5, 3), **cfg),
                           nn.BatchNorm2d(FILTERS[b]),
                           nn.LeakyReLU(0.01, inplace=True),
                           nn.MaxPool2d((2, 1))]
                c = FILTERS[b]
            s.conv = nn.Sequential(*blocks)
            with torch.no_grad():
                s.flat = s.conv(torch.zeros(1, 1, H, W)).flatten(1).shape[1]
            s.drop = nn.Dropout(0.5)
            s.head = nn.Linear(s.flat, 2 if task == "clf" else 1)
            for m in s.modules():                # Xavier (Glorot) init
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(s, x):
            return s.head(s.drop(s.conv(x).flatten(1)))

    return CNN()


class Predictor:
    """Callable ensemble wrapper. `predictor(Xte) -> score` (P(up) for clf,
    return units for reg). Carries the pixel normaliser (global mean/std OR the
    per-image flag), the y-scaler, and the trained nets so the whole thing can be
    saved to disk and reloaded. Everything it holds was fit on TRAIN only."""

    def __init__(self, nets, task, days, H, W, pixel_norm,
                 mu, sd, ymu, ysd, cap, device="cpu"):
        self.nets = nets                 # list[nn.Module] (already .eval())
        self.task, self.days = task, int(days)
        self.H, self.W = int(H), int(W)
        self.pixel_norm = pixel_norm      # "global" | "per_image"
        self.mu, self.sd = mu, sd         # floats (global) or None (per_image)
        self.ymu, self.ysd, self.cap = ymu, ysd, cap
        self.device = device

    # -- pixel normalisation (vectorized; works on any (n,H,W) slice) --
    def _norm(self, A):
        A = np.asarray(A, dtype=np.float32)
        if self.pixel_norm == "per_image":
            flat = A.reshape(len(A), -1)
            m = flat.mean(1)[:, None, None]
            s = flat.std(1)[:, None, None] + 1e-8
            return (A - m) / s
        return (A - self.mu) / self.sd

    def __call__(self, Xte, batch=4096):
        import torch
        Xte = np.ascontiguousarray(Xte, dtype=np.float32)
        out = np.zeros(len(Xte), np.float64)
        with torch.no_grad():
            for s0 in range(0, len(Xte), batch):
                xb = self._norm(Xte[s0:s0 + batch])
                xb = torch.from_numpy(np.ascontiguousarray(xb)) \
                    .view(-1, 1, self.H, self.W).to(self.device)
                for net in self.nets:
                    o = net(xb)
                    if self.task == "clf":
                        out[s0:s0 + batch] += torch.softmax(o, 1)[:, 1].cpu().numpy()
                    else:
                        out[s0:s0 + batch] += (torch.tanh(o.squeeze(-1)) * self.cap).cpu().numpy()
        out /= len(self.nets)
        return out if self.task == "clf" else out * self.ysd + self.ymu

    predict = __call__

    # -- persistence (feature 2) --
    def save(self, out_dir, tag="model"):
        import torch
        d = _ensure_dir(out_dir)
        meta = dict(task=self.task, days=self.days, H=self.H, W=self.W,
                    pixel_norm=self.pixel_norm, mu=self.mu, sd=self.sd,
                    ymu=self.ymu, ysd=self.ysd, cap=self.cap, n_nets=len(self.nets))
        for k, net in enumerate(self.nets):
            torch.save(net.state_dict(), d / f"{tag}_seed{k}.pt")
        (d / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))
        log.info("saved model: %d nets + meta -> %s", len(self.nets), d / f"{tag}_meta.json")
        return d


def load_predictor(model_dir, tag="model", device=None):
    """Rebuild a saved Predictor: reads <tag>_meta.json + <tag>_seed*.pt."""
    import torch
    d = Path(model_dir)
    meta = json.loads((d / f"{tag}_meta.json").read_text())
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    nets = []
    for k in range(int(meta["n_nets"])):
        net = build_cnn(meta["days"], meta["H"], meta["W"], meta["task"]).to(dev)
        net.load_state_dict(torch.load(d / f"{tag}_seed{k}.pt", map_location=dev))
        net.eval(); nets.append(net)
    return Predictor(nets, meta["task"], meta["days"], meta["H"], meta["W"],
                     meta["pixel_norm"], meta["mu"], meta["sd"],
                     meta["ymu"], meta["ysd"], meta["cap"], device=dev)


def train_cnn(Xtr, ytr, days, H, W, task="clf", seeds=5, epochs=50,
              patience=2, lr=1e-5, batch=128, val_frac=0.30, winsorize=True,
              device=None, seed0=42, pixel_norm="global"):
    """Train the JKX CNN ensemble on TRAINING images only; return a `Predictor`.
    Pixel normaliser and y-scaler are fit on train only; a random `val_frac`
    split drives early stopping. No test data is ever seen during fitting.
    pixel_norm='global' (train mean/std) or 'per_image' (each image self-norms)."""
    import torch, torch.nn.functional as F
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = np.ascontiguousarray(Xtr, dtype=np.float32)

    # ---- pixel normalisation (feature 6) : fit on TRAIN only ----
    if pixel_norm == "per_image":
        mu = sd = None
        def _norm(A):
            A = np.asarray(A, dtype=np.float32); flat = A.reshape(len(A), -1)
            return (A - flat.mean(1)[:, None, None]) / (flat.std(1)[:, None, None] + 1e-8)
    elif pixel_norm == "global":
        mu, sd = float(Xtr.mean()), float(Xtr.std() + 1e-8)
        def _norm(A):
            return (np.asarray(A, dtype=np.float32) - mu) / sd
    else:
        raise ValueError(f"pixel_norm must be 'global' or 'per_image', got {pixel_norm!r}")

    if task == "clf":
        lab = (ytr > 0).astype(np.int64)                    # 1{forward return > 0}
        ymu = ysd = cap = None
    else:
        y = ytr.astype(np.float32)
        if winsorize:
            lo, hi = np.quantile(y, [0.01, 0.99]); y = np.clip(y, lo, hi)
        ymu, ysd = float(y.mean()), float(y.std() + 1e-8)
        lab = ((y - ymu) / ysd).astype(np.float32)
        cap = float(np.quantile(np.abs(lab), 0.99))

    rng = np.random.default_rng(seed0)                      # random 70/30 split
    perm = rng.permutation(len(Xtr))
    nval = max(50, int(val_frac * len(Xtr)))
    vi, ti = perm[:nval], perm[nval:]

    def t_img(A, idx):
        a = _norm(A[idx])
        return torch.from_numpy(np.ascontiguousarray(a)).view(-1, 1, H, W).to(dev)

    Xt, Xv = t_img(Xtr, ti), t_img(Xtr, vi)
    yt = torch.from_numpy(lab[ti]).to(dev)
    yv = torch.from_numpy(lab[vi]).to(dev)
    if task == "clf":
        loss_fn = lambda o, y_: F.cross_entropy(o, y_)
    else:
        loss_fn = lambda o, y_: F.mse_loss(torch.tanh(o.squeeze(-1)) * cap, y_)

    nets, idx_t = [], torch.arange(len(ti))
    for k in range(seeds):
        torch.manual_seed(seed0 + k)
        net = build_cnn(days, H, W, task).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=lr)     # Adam, lr=1e-5 (JKX)
        best, bst, wait = 1e18, None, 0
        for _ in range(epochs):
            net.train()
            pm = idx_t[torch.randperm(len(idx_t))]
            for s0 in range(0, len(pm), batch):
                b = pm[s0:s0 + batch]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss_fn(net(Xt[b]), yt[b]).backward()
                opt.step()
            net.eval()
            total, nobs = 0.0, 0
            with torch.no_grad():
                for s0 in range(0, len(Xv), batch):
                    xb = Xv[s0:s0 + batch]
                    yb = yv[s0:s0 + batch]
                    if len(yb) == 0:
                        continue
                    loss = loss_fn(net(xb), yb)
                    total += float(loss.item()) * len(yb)
                    nobs += len(yb)
            v = total / max(nobs, 1)
            if v < best - 1e-7:
                best, wait, bst = v, 0, {a: b_.clone()
                                        for a, b_ in net.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        if bst:
            net.load_state_dict(bst)
        net.eval(); nets.append(net)

    return Predictor(nets, task, days, H, W, pixel_norm, mu, sd, ymu, ysd, cap, dev)


# ======================================================================
# PART 3 -- data loading (chart panels + VIX channel)
# ======================================================================
def _ci(cols, names):
    low = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def _find_files(data_dir, pattern):
    """Resolve a chart pickle spec robustly and CASE-INSENSITIVELY. Accepts a full
    filename, a glob, or a bare universe tag like 'SP500_5MA' -- which also matches
    'SP500_pic_Map5MA_price_chart_image_df.pickle' (underscores act as wildcards).
    A '5MA' tag never grabs a '20MA' file (and vice-versa)."""
    dirs = [Path(data_dir)] if data_dir else [Path.cwd(), Path(__file__).parent]
    stem = pattern
    for ext in (".pickle", ".pkl"):
        stem = stem.replace(ext, "")
    loose = stem.replace("_", "*")        # 'SP500_5MA' -> 'SP500*5MA'
    img = "price_chart_image"             # canonical chart-image df signature
    # Candidates are tried IN ORDER; the first that matches wins. The tag+chart-image
    # patterns come first so an unrelated dict pickle that merely shares the tag is
    # never picked up. Only if nothing chart-image-like matches do we fall back to a
    # broader tag glob.
    cands = [pattern, pattern + ".pickle", pattern + ".pkl",
             f"*{stem}*{img}*.pickle", f"*{stem}*{img}*.pkl",
             f"*{loose}*{img}*.pickle", f"*{loose}*{img}*.pkl",
             f"*{stem}*.pickle", f"*{stem}*.pkl",
             f"*{loose}*.pickle", f"*{loose}*.pkl",
             f"*{stem}*", f"*{loose}*"]
    seen, uniq = set(), []
    for c in cands:                       # de-dup while preserving order
        if c not in seen:
            seen.add(c); uniq.append(c)
    cands = uniq
    for d in dirs:
        try:
            entries = sorted(p for p in Path(d).iterdir() if p.is_file())
        except (FileNotFoundError, NotADirectoryError):
            continue
        for pat in cands:
            patl = pat.lower()
            wild = any(ch in pat for ch in "*?[")
            hits = []
            for p in entries:
                if not fnmatch.fnmatchcase(p.name.lower(), patl):
                    continue
                # wildcard candidates must resolve to a pickle (never a dir/log/csv);
                # an exact filename is accepted with any extension.
                if wild and p.suffix.lower() not in (".pickle", ".pkl"):
                    continue
                hits.append(str(p))
            if hits:
                return hits
    return []


def _coerce_to_df(obj, src=""):
    """Coerce a loaded pickle payload into a tidy DataFrame. Real picMap pickles
    are sometimes stored as a *dict* (of columns, or wrapping/holding DataFrames)
    rather than a DataFrame -- handle the common shapes and fail loudly otherwise."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    if isinstance(obj, dict):
        # 1) dict that HOLDS DataFrame(s): {'df': df} or {sedol: df, ...}
        df_vals = [v for v in obj.values() if isinstance(v, pd.DataFrame)]
        if df_vals:
            log.info("%s: dict holding %d DataFrame(s) -> concat", src, len(df_vals))
            return pd.concat(df_vals, ignore_index=True) if len(df_vals) > 1 else df_vals[0]
        # 2) COLUMN-oriented dict: {'date':[...], 'price_chart_image':[...], ...}
        arr_len = {}
        for k, v in obj.items():
            if isinstance(v, (str, bytes)) or not hasattr(v, "__len__"):
                continue
            try:
                arr_len[k] = len(v)
            except TypeError:
                pass
        if arr_len:
            vals = list(arr_len.values())
            L = max(set(vals), key=vals.count)          # dominant column length
            cols = {}
            for k, v in obj.items():
                if arr_len.get(k) == L:
                    cols[k] = list(v)                    # array/2D -> per-row objects
                elif isinstance(v, (str, bytes)) or not hasattr(v, "__len__"):
                    cols[k] = v                          # scalar metadata -> broadcast
            if cols:
                log.info("%s: column-dict -> DataFrame (%d cols, %d rows)",
                         src, len(cols), L)
                return pd.DataFrame(cols)
        # 3) dict of RECORDS keyed by id: {key: {col: val, ...}, ...}
        if obj and all(isinstance(v, dict) for v in obj.values()):
            return pd.DataFrame.from_dict(obj, orient="index").reset_index(drop=True)
        info = {k: type(v).__name__ + (f"[{len(v)}]" if hasattr(v, "__len__")
                and not isinstance(v, (str, bytes)) else "")
                for k, v in list(obj.items())[:25]}
        _hint(f"{src}: pickle is a dict I couldn't coerce to a DataFrame; keys->types: {info}",
              "share this mapping, or pre-convert to a DataFrame with columns "
              "sedol,date,days,ph,vh,MA,rF5D,rF20D,rF60D,img_height,img_width,price_chart_image")
    if isinstance(obj, (list, tuple)) and len(obj):
        if all(isinstance(v, pd.DataFrame) for v in obj):
            return pd.concat(list(obj), ignore_index=True)
        if all(isinstance(v, dict) for v in obj):
            return pd.DataFrame(list(obj))
    _hint(f"{src}: unsupported pickle payload of type {type(obj).__name__}",
          "expected a DataFrame or a dict of columns; pre-convert and re-save")


def load_panel(data_dir=None, pattern="R1000_picMap*_price_chart_image_df.pickle"):
    """Load the chart-image pickle(s) matching `pattern` in `data_dir` (default:
    cwd then this file's folder). Returns (df, X) with X an (N, H, W) float32
    cube. `pattern` may be a glob, a filename, or a bare tag like 'SP500_5MA'.
    Each pickle may be a DataFrame OR a dict (of columns / of DataFrames)."""
    files = _find_files(data_dir, pattern)
    if not files:
        _hint(f"no files matching {pattern!r} in "
              f"{data_dir or [str(Path.cwd()), str(Path(__file__).parent)]}",
              "put the pickle next to the notebook, or pass data_dir=/ pattern=")
    df = pd.concat([_coerce_to_df(pd.read_pickle(f), Path(f).name) for f in files],
                   ignore_index=True)
    idc = _ci(df.columns, ["sedol", "perm_ID", "permno", "id"])
    dtc = _ci(df.columns, ["date", "Date", "DATE"])
    imc = _ci(df.columns, ["price_chart_image", "image", "img"])
    if not (idc and dtc and imc):
        _hint(f"required columns not found; have {list(df.columns)[:15]}",
              "need an id (sedol), a date, and a flattened image column")
    df = df.rename(columns={idc: "sedol", dtc: "date", imc: "price_chart_image"})
    df["date"] = pd.to_datetime(df["date"])
    df = (df.dropna(subset=["price_chart_image"])
            .drop_duplicates(["sedol", "date"])
            .sort_values(["sedol", "date"]).reset_index(drop=True))
    H, W = int(df["img_height"].iloc[0]), int(df["img_width"].iloc[0])
    ph, vh = int(df["ph"].iloc[0]), int(df["vh"].iloc[0])
    days = int(df["days"].iloc[0])
    X = to_images(df["price_chart_image"].values, H, W)      # vectorized reshape
    finite = np.isfinite(X.reshape(len(X), -1)).all(1)
    if (~finite).any():
        log.warning("dropped %d rows with malformed images", int((~finite).sum()))
        df, X = df[finite].reset_index(drop=True), X[finite]
    df["_m"] = df["date"].dt.to_period("M")
    df["is_me"] = (df.groupby(["sedol", "_m"])["date"].transform("max") == df["date"])
    print(f"loaded {len(files)} file(s) -> {len(df):,} images {H}x{W} "
          f"(days={days}, ph={ph}, vh={vh}, MA={bool(df['MA'].iloc[0])}), "
          f"{df['sedol'].nunique()} stocks, "
          f"{df['date'].min().date()}..{df['date'].max().date()}")
    df.attrs.update(H=H, W=W, ph=ph, vh=vh, days=days)
    return df, X


# ---- VIX channel (feature 5) -----------------------------------------
def load_vix(start=None, end=None, data_dir=None, symbol="^VIX",
             cache="vix.csv", download=True):
    """Return a daily VIX close Series (index=date). Reads/writes a CSV cache so
    the run is reproducible and network-free after the first fetch. If yfinance
    is unavailable and no cache exists, raises with a clear hint."""
    d = Path(data_dir) if data_dir else Path.cwd()
    fp = d / cache
    if fp.exists():
        s = pd.read_csv(fp, parse_dates=["date"]).dropna().set_index("date")["vix"]
        return s.sort_index()
    if not download:
        _hint(f"no VIX cache at {fp} and download=False",
              "place a vix.csv with columns (date,vix), or set download=True")
    try:
        import yfinance as yf
    except Exception:
        _hint("yfinance not installed and no vix.csv cache",
              "pip install yfinance, OR drop a vix.csv (date,vix) next to the data")
    raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if raw is None or len(raw) == 0:
        _hint(f"yfinance returned no data for {symbol}", "check symbol / date range")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):          # newer yfinance MultiIndex cols
        close = close.iloc[:, 0]
    s = pd.Series(np.asarray(close, float), index=pd.to_datetime(raw.index), name="vix")
    s.index.name = "date"
    s.sort_index().to_frame().to_csv(fp)
    log.info("downloaded VIX %s (%d obs) -> cached %s", symbol, len(s), fp)
    return s.sort_index()


class VixAugmenter:
    """Append `vix_rows` rows of a column-aligned VIX strip under each chart.

    For an image ending at date t with a `days`-day look-back, we take the `days`
    most recent VIX closes up to and INCLUDING t (known at t -> no look-ahead),
    scale them with a scaler fit on TRAIN dates only, repeat each day across its
    pixel-columns so the strip spans W, and tile it over `vix_rows` rows. VIX has
    its own scaler because it is not 0/1 like the chart pixels."""

    def __init__(self, vix_series, days, W, vix_rows=3,
                 train_lo=None, train_hi=None, mode="minmax"):
        self.days, self.W, self.vix_rows, self.mode = int(days), int(W), int(vix_rows), mode
        vs = pd.Series(vix_series).dropna().sort_index()
        self.vdates = vs.index.values.astype("datetime64[ns]")
        self.vvals = vs.values.astype(np.float32)
        m = np.ones(len(vs), bool)
        if train_lo is not None:
            m &= (self.vdates >= np.datetime64(pd.Timestamp(train_lo)))
        if train_hi is not None:
            m &= (self.vdates <= np.datetime64(pd.Timestamp(train_hi)))
        tr = self.vvals[m] if m.any() else self.vvals
        if mode == "minmax":                      # -> [0,1] like the chart pixels
            self.lo = float(np.nanmin(tr)); self.hi = float(np.nanmax(tr))
            self.den = (self.hi - self.lo) or 1.0
            self._scale = lambda a: np.clip((a - self.lo) / self.den, 0.0, 1.0)
        elif mode == "zscore":                    # standardise then squash to (0,1)
            self.mu = float(np.nanmean(tr)); self.sd = float(np.nanstd(tr) + 1e-8)
            self._scale = lambda a: 1.0 / (1.0 + np.exp(-(a - self.mu) / self.sd))
        else:
            raise ValueError("vix mode must be 'minmax' or 'zscore'")
        self.pxw = max(1, int(round(self.W / self.days)))

    def strip_for_dates(self, dates):
        """(n,) end-dates -> (n, W) scaled VIX strip aligned to the look-back."""
        d = np.asarray(pd.to_datetime(dates).values, dtype="datetime64[ns]")
        j = np.searchsorted(self.vdates, d, side="right") - 1     # last VIX <= t
        offs = np.arange(-self.days + 1, 1)                       # length == days
        idx = np.clip(j[:, None] + offs[None, :], 0, len(self.vvals) - 1)
        win = self._scale(self.vvals[idx])                        # (n, days) in [0,1]
        rep = np.repeat(win, self.pxw, axis=1)                    # (n, days*pxw)
        if rep.shape[1] != self.W:                                # safety: fit to W
            xp = np.linspace(0, 1, rep.shape[1]); xq = np.linspace(0, 1, self.W)
            rep = np.stack([np.interp(xq, xp, r) for r in rep])
        return rep.astype(np.float32)

    def augment(self, X, dates):
        """(n,H,W) chart cube + (n,) dates -> (n, H+vix_rows, W) enriched cube."""
        X = _as_cube(X)
        strip = self.strip_for_dates(dates)                       # (n, W)
        band = np.repeat(strip[:, None, :], self.vix_rows, axis=1)  # (n, vix_rows, W)
        return np.concatenate([X, band.astype(np.float32)], axis=1)

    @property
    def extra_rows(self):
        return self.vix_rows


# ======================================================================
# PART 4 -- optional QC filter, portfolio evaluation, diagnostics, driver
# ======================================================================
def _make_filter(Xtr, ph, vh, denoise=None, drop_train_noise_q=None,
                 drop_metric="composite"):
    """Build (transform, keep_mask) from TRAINING images only. The transform is
    applied identically to train and test; keep_mask only drops noisy TRAIN
    rows -> no look-ahead."""
    ident = (lambda A: A)
    if denoise in (None, "", "none") and drop_train_noise_q is None:
        return ident, np.ones(len(Xtr), bool)
    H, W = Xtr.shape[1], Xtr.shape[2]
    tf = ident
    if denoise == "despeckle":
        tf = lambda A: despeckle(A, H, W).reshape(-1, H, W)
    elif denoise not in (None, "", "none"):
        log.warning("unknown denoise='%s'; using identity", denoise)
    keep = np.ones(len(Xtr), bool)
    if drop_train_noise_q is not None:
        v = qc_score(Xtr, ph, vh, drop_metric, train=Xtr)
        keep = v <= np.quantile(v, drop_train_noise_q)
        log.info("filter: denoise=%s drop %s>q%.2f -> keep %d/%d train images",
                 denoise, drop_metric, drop_train_noise_q, int(keep.sum()), len(keep))
    return tf, keep


def _cap_train(tr, max_train, seed=0):
    """Randomly subsample a boolean TRAIN mask to at most `max_train` rows
    (train-only, look-ahead-free) -- a speed/memory knob, off by default."""
    if not max_train or int(tr.sum()) <= int(max_train):
        return tr
    idx = np.where(tr)[0]
    keep = np.random.default_rng(seed).choice(idx, int(max_train), replace=False)
    m = np.zeros_like(tr); m[keep] = True
    return m


def _evaluate(out, grp, min_names=20):
    """out: [sedol, date, score, ret, per]. Returns (portfolio_df, summary) for
    BOTH the long-short decile (H-L) and long-only top-decile portfolios. Each
    portfolio row carries the EXACT rebalance date (feature 4) = the formation
    date (latest image date used in that period)."""
    sizes = out.groupby("per").size()
    min_n = min(int(min_names), max(2, int(sizes.median() * 0.5)))
    rows = []
    for per_, g in out.groupby("per"):
        g = g.dropna(subset=["score", "ret"])
        if len(g) < min_n or g["score"].std() < 1e-12:
            continue
        q = g["score"].rank(pct=True)
        top, bot = g[q >= 0.9], g[q <= 0.1]
        rows.append({"per": per_, "rebalance_date": g["date"].max(),
                     "long_only": top["ret"].mean(),
                     "long_short": top["ret"].mean() - bot["ret"].mean(),
                     "n_top": len(top), "n_bot": len(bot), "n_names": len(g)})
    pf = pd.DataFrame(rows).sort_values("per").reset_index(drop=True)
    if len(pf) < 6:
        _hint(f"only {len(pf)} evaluable periods",
              "widen the test window or check that predictions were produced")
    ppy = {"M": 12.0, "W": 52.0, "Q": 4.0}[grp]
    sh = lambda r: float(r.mean() / (r.std() + 1e-12) * np.sqrt(ppy))
    cg = lambda r: float(np.cumprod(1.0 + r.values)[-1] ** (ppy / len(r)) - 1.0)
    summary = {"n_periods": int(len(pf)), "avg_names_long": float(pf["n_top"].mean()),
               "ls_sharpe": sh(pf["long_short"]), "ls_cagr": cg(pf["long_short"]),
               "lo_sharpe": sh(pf["long_only"]), "lo_cagr": cg(pf["long_only"])}
    return pf, summary


def beta_over_time(out, freq_months=6, min_obs=30, standardize=True):
    """Half-year (default) cross-sectional OLS of realised return on the predicted
    score (feature 3). Positive, significant beta => the score separated stocks
    that period. Returns a tidy DataFrame [bucket, date, beta, se, tstat, r2, n].

    Diagnostic only -- computed on OUT-OF-SAMPLE predictions; nothing is fit here
    that touches the model. `standardize` puts beta in 'return per 1 sd of score'."""
    o = out.dropna(subset=["score", "ret"]).copy()
    step = max(1, int(freq_months))
    half = ((o["date"].dt.month - 1) // step)                 # 0..(12/step-1)
    o["bucket"] = o["date"].dt.year.astype(str) + "-P" + (half + 1).astype(str)
    rows = []
    for b, g in o.groupby("bucket"):
        x = g["score"].to_numpy(float); y = g["ret"].to_numpy(float)
        if len(g) < min_obs or x.std() < 1e-12:
            continue
        if standardize:
            x = (x - x.mean()) / (x.std() + 1e-12)
        Xd = np.column_stack([np.ones_like(x), x])            # OLS y = a + b x
        coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        resid = y - Xd @ coef
        dof = max(1, len(y) - 2)
        s2 = float(resid @ resid) / dof
        se = float(np.sqrt(np.diag(s2 * np.linalg.inv(Xd.T @ Xd))[1]))
        sst = float(((y - y.mean()) ** 2).sum()) + 1e-12
        rows.append(dict(bucket=b, date=g["date"].max(), beta=float(coef[1]),
                         se=se, tstat=float(coef[1] / (se + 1e-12)),
                         r2=1.0 - float(resid @ resid) / sst, n=int(len(g))))
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def decile_profile(out, n_bins=10):
    """Average realised return of each score-decile over the WHOLE OOS (feature 4).
    Deciles are formed cross-sectionally within each rebalance period, then pooled
    -> shows monotone separation. Returns [decile, mean, std, count]."""
    o = out.dropna(subset=["score", "ret"]).copy()
    # rank cross-sectionally WITHIN each rebalance period, then bin (vectorized)
    q = o.groupby("per")["score"].rank(pct=True, method="first")
    o["decile"] = np.clip((q * n_bins).astype(int) + 1, 1, n_bins)
    prof = (o.groupby("decile")["ret"].agg(["mean", "std", "count"])
              .reset_index().sort_values("decile").reset_index(drop=True))
    return prof


# ---- plotting helpers (each guarded; PNGs saved next to the results) ----
def _plot_cumulative(pf, out_png, title):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts = pf["per"].dt.to_timestamp()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, col, ttl in [(axes[0], "long_short", "long-short H-L decile"),
                             (axes[1], "long_only", "long-only top decile")]:
            ax.plot(ts, np.cumprod(1.0 + pf[col].values) - 1.0, lw=1.6)
            ax.set_title(f"{ttl}\n{title}"); ax.grid(alpha=0.3)
            ax.set_ylabel("cumulative return")
        fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    except Exception as exc:
        log.warning("cumulative plot skipped (%s)", exc)


def _plot_beta(beta_df, out_png, title):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if len(beta_df) == 0:
            return
        x = beta_df["date"]; b = beta_df["beta"].values; se = beta_df["se"].values
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.axhline(0, color="k", lw=0.8)
        ax.fill_between(x, b - 1.96 * se, b + 1.96 * se, alpha=0.20, label="95% CI")
        ax.plot(x, b, "-o", lw=1.6, ms=4, label="beta (ret ~ standardized score)")
        sig = np.abs(beta_df["tstat"].values) >= 1.96
        ax.plot(x[sig], b[sig], "o", ms=7, mfc="none", mec="crimson", label="|t|>=1.96")
        ax.set_title(f"Half-year score->return beta\n{title}")
        ax.set_ylabel("beta"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    except Exception as exc:
        log.warning("beta plot skipped (%s)", exc)


def _plot_decile(prof, out_png, title):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(prof["decile"], prof["mean"])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("score decile (1=low ... 10=high)")
        ax.set_ylabel("avg realised return (OOS)")
        ax.set_title(f"Decile average return\n{title}"); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    except Exception as exc:
        log.warning("decile plot skipped (%s)", exc)


def run(data_dir=None, pattern="R1000_picMap*_price_chart_image_df.pickle",
        test_pattern=None,
        task="clf", protocol="faithful", horizon=None,
        train_years=8, refit_months=12, window="exp",
        test_start=None, test_end=None, seeds=5, epochs=50,
        denoise=None, drop_train_noise_q=None, drop_metric="composite",
        max_train=None, min_names=20,
        pixel_norm="global",
        add_vix=False, vix_rows=3, vix_symbol="^VIX", vix_scale="minmax",
        vix_csv="vix.csv", vix_download=True,
        beta_freq_months=6, n_decile_bins=10,
        save_csv=True, save_model=True, out_dir=None, tag=None,
        return_predictor=False, verbose=True, log_level=None):
    """Forecast forward returns from chart images with the JKX CNN, plus the eight
    extra experiments (see the module header). Returns a dict with
    {'predictions','portfolio','summary','beta','decile_profile','out_dir'} and
    writes CSVs + PNGs into a per-run results folder.

    Key knobs (all notebook-settable):
      pattern / test_pattern : train / (cross-universe) test files, e.g.
                               pattern='sp500_5MA', test_pattern='sp600_5MA'.
      task='clf'|'reg'; protocol='faithful'|'walkforward'; horizon=5/20/60.
      pixel_norm='global'|'per_image'      (feature 6)
      add_vix=True, vix_rows, vix_scale    (feature 5)
      train_years                          (feature 7: set 15)
      save_csv / save_model                (feature 2: predictions + model)
    """
    _setup_log(data_dir)
    if log_level:
        lvl = getattr(logging, str(log_level).upper(), logging.INFO)
        for h_ in log.handlers:
            if isinstance(h_, logging.StreamHandler) and not isinstance(h_, logging.FileHandler):
                h_.setLevel(lvl)
    assert task in ("clf", "reg") and protocol in ("faithful", "walkforward")
    cross = test_pattern is not None
    if cross and protocol != "faithful":
        log.warning("cross-universe test -> forcing protocol='faithful'")
        protocol = "faithful"

    log.info("=" * 70)
    log.info("NEW RUN | task=%s protocol=%s horizon=%s train_years=%s seeds=%s "
             "pixel_norm=%s add_vix=%s cross=%s", task, protocol, horizon,
             train_years, seeds, pixel_norm, add_vix, cross)
    log.info("=" * 70)

    # ---- load train (and optional separate test) universe ----
    df, X = load_panel(data_dir, pattern)
    days, H, W = df.attrs["days"], df.attrs["H"], df.attrs["W"]
    ph, vh = df.attrs["ph"], df.attrs["vh"]
    if cross:
        df_te, X_te = load_panel(data_dir, test_pattern)
        if (df_te.attrs["days"], df_te.attrs["H"], df_te.attrs["W"]) != (days, H, W):
            _hint("train/test geometries differ "
                  f"({(days, H, W)} vs {(df_te.attrs['days'], df_te.attrs['H'], df_te.attrs['W'])})",
                  "use the same image size (e.g. both *_5MA) for cross-universe tests")
    else:
        df_te, X_te = df, X

    h = int(horizon or days)
    grp = {5: "W", 20: "M", 60: "Q"}.get(h, "M")

    def _target(frame):
        t = _ci(frame.columns, [f"rF{h}D", f"rf{h}D"])
        if t is None:
            _hint(f"target rF{h}D not found",
                  "available: " + ", ".join(c for c in frame.columns if "rf" in c.lower()))
        return t

    target, target_te = _target(df), _target(df_te)
    y = pd.to_numeric(df[target], errors="coerce").values.astype(np.float32)
    y_te = pd.to_numeric(df_te[target_te], errors="coerce").values.astype(np.float32)
    ok, ok_te = np.isfinite(y), np.isfinite(y_te)

    def _grid(frame):
        if grp == "M":
            return frame["is_me"].values
        pcode = frame["date"].dt.to_period(grp)
        return (frame.groupby([frame["sedol"], pcode])["date"].transform("max")
                == frame["date"]).values

    test_grid_te = _grid(df_te)
    embargo = pd.DateOffset(days=int(h * 1.6))       # trading -> calendar embargo

    uni_tr = _tag_from_pattern(pattern)
    uni_te = _tag_from_pattern(test_pattern) if cross else uni_tr
    tag = tag or _auto_tag(uni_tr, uni_te, days, h, task, protocol,
                           pixel_norm, add_vix, train_years, cross)
    base = Path(data_dir) if data_dir else Path.cwd()
    out = _ensure_dir(out_dir or (base / f"results_{tag}"))
    log.info("results -> %s", out)

    preds = np.full(len(df_te), np.nan)
    Hn = H + (int(vix_rows) if add_vix else 0)
    t0 = time.time()

    # ------------------------------------------------------------------
    # FAITHFUL protocol (paper's, and the one all extra experiments use)
    # ------------------------------------------------------------------
    if protocol == "faithful":
        yr0 = int(df["date"].dt.year.min())
        if test_start:
            te_lo = pd.Timestamp(str(test_start))
            tr_cut = te_lo - pd.Timedelta(days=1)
        else:
            tr_end_year = yr0 + int(train_years) - 1
            tr_cut = pd.Timestamp(f"{tr_end_year}-12-31")
            te_lo = pd.Timestamp(f"{tr_end_year + 1}-01-01")
        train_end = tr_cut - embargo
        tr = ok & (df["date"] <= train_end).values       # embargo right edge
        train_lo_limit = None
        if train_years is not None and int(train_years) > 0:
            train_lo_limit = te_lo - pd.DateOffset(years=int(train_years))
            tr &= (df["date"] >= train_lo_limit).values
        te_hi = pd.Timestamp(str(test_end)) if test_end else df_te["date"].max() + pd.Timedelta(days=1)
        # cross-universe: guarantee no temporal overlap with the train labels
        if cross and te_lo <= tr_cut:
            te_lo = tr_cut + pd.Timedelta(days=1)
            log.info("cross-universe: nudged test_start to %s (strictly after train)", te_lo.date())
        te = ok_te & test_grid_te & ((df_te["date"] >= te_lo) & (df_te["date"] <= te_hi)).values
        if tr.sum() < 500:
            _hint(f"only {int(tr.sum())} training images", "lower train_years / widen data")
        if int(te.sum()) == 0:
            _hint(f"no test images in {te_lo.date()}..{te_hi.date()} "
                  f"(train ends {tr_cut.date()}; data ends {df_te['date'].max().date()})",
                  "lower train_years, set test_end later, or check the test universe covers "
                  "dates after the training window")
        tr = _cap_train(tr, max_train)
        train_lo = pd.Timestamp(df["date"].values[tr].min())

        vix_aug = _maybe_vix(add_vix, data_dir, base, df, df_te, days, W, vix_rows,
                             vix_symbol, vix_scale, vix_csv, vix_download,
                             train_lo, tr_cut)
        log.info("faithful: train %s..%s (cut %s, embargo %dd) ntr=%d | test %s..%s nte=%d",
                 train_lo.date(), train_end.date(), tr_cut.date(), embargo.days, int(tr.sum()), te_lo.date(),
                 te_hi.date(), int(te.sum()))

        predictor = _fit_predict(
            X[tr], y[tr], df["date"].values[tr],
            X_te[te], df_te["date"].values[te], preds, te,
            ph, vh, denoise, drop_train_noise_q, drop_metric,
            days, Hn, W, task, seeds, epochs, pixel_norm, vix_aug)
        if save_model:
            predictor.save(out / "model", "model")
        if verbose:
            print(f"  trained once on {int(tr.sum()):,} images ({time.time()-t0:.0f}s)",
                  flush=True)

    # ------------------------------------------------------------------
    # WALK-FORWARD protocol (single universe); diagnostics computed the same way
    # ------------------------------------------------------------------
    else:
        yr0 = int(df["date"].dt.year.min())
        start = pd.Period(f"{test_start or (yr0 + int(train_years))}-01", "M")
        end = pd.Period(str(test_end), "M") if test_end else pd.Period(df["date"].max(), "M")
        months = pd.period_range(start, end, freq="M")
        first_lo = months[0].to_timestamp()
        vix_aug = _maybe_vix(add_vix, data_dir, base, df, df_te, days, W, vix_rows,
                             vix_symbol, vix_scale, vix_csv, vix_download,
                             pd.Timestamp(df["date"].min()), first_lo - embargo)
        predictor = None
        for i, mper in enumerate(months):
            te_lo, te_hi = mper.to_timestamp(), (mper + 1).to_timestamp()
            te = ok & test_grid_te & ((df["date"] >= te_lo) & (df["date"] < te_hi)).values
            if te.sum() == 0:
                continue
            if predictor is None or i % max(1, int(refit_months)) == 0:
                tr = ok & (df["date"] < te_lo - embargo).values
                if window.startswith("roll"):
                    tr &= (df["date"] >= te_lo - pd.DateOffset(years=int(window[4:]))).values
                if tr.sum() < 500:
                    log.warning("refit %s skipped: only %d train images", mper, int(tr.sum()))
                    continue
                tr = _cap_train(tr, max_train, seed=i)
                predictor = _fit_predict(
                    X[tr], y[tr], df["date"].values[tr], None, None, None, None,
                    ph, vh, denoise, drop_train_noise_q, drop_metric,
                    days, Hn, W, task, seeds, epochs, pixel_norm, vix_aug,
                    predict_only=False)
                if verbose:
                    print(f"  refit {mper}: ntr={int(tr.sum()):,} ({time.time()-t0:.0f}s)",
                          flush=True)
            Xte = X[te]
            if vix_aug is not None:
                Xte = vix_aug.augment(Xte, df["date"].values[te])
            preds[te] = predictor(Xte)
        if save_model and predictor is not None:
            predictor.save(out / "model", "model")

    # ------------------------------------------------------------------
    # assemble OOS predictions + evaluation + diagnostics
    # ------------------------------------------------------------------
    fin = np.isfinite(preds)
    out_df = df_te.loc[fin, ["sedol", "date"]].copy()
    out_df["score"], out_df["ret"] = preds[fin], y_te[fin]
    out_df["per"] = out_df["date"].dt.to_period(grp)

    pf, summ = _evaluate(out_df, grp, min_names)
    beta_df = beta_over_time(out_df, beta_freq_months)
    prof = decile_profile(out_df, n_decile_bins)

    summary = {"tag": tag, "task": task, "protocol": protocol,
               "train_universe": uni_tr, "test_universe": uni_te, "cross": cross,
               "target": target_te, "horizon": h, "rebalance": grp,
               "pixel_norm": pixel_norm, "add_vix": bool(add_vix),
               "vix_rows": int(vix_rows) if add_vix else 0,
               "train_years": int(train_years), "image_H": Hn, "image_W": W, **summ}
    log.info("%s | LONG-SHORT sharpe=%.2f cagr=%.1f%% | LONG-ONLY sharpe=%.2f "
             "cagr=%.1f%% (%d periods)", tag, summary["ls_sharpe"],
             100 * summary["ls_cagr"], summary["lo_sharpe"], 100 * summary["lo_cagr"],
             summary["n_periods"])
    if verbose:
        print("\n".join(f"  {k}: {v}" for k, v in summary.items()))

    # ---- plots ----
    ttl = f"{tag} ({task},rF{h}D)"
    _plot_cumulative(pf, out / "portfolio_cumulative.png", ttl)
    _plot_beta(beta_df, out / "beta_vs_time.png", ttl)
    _plot_decile(prof, out / "decile_profile.png", ttl)

    # ---- CSVs (predictions, portfolio tape w/ rebalance dates, diagnostics) ----
    if save_csv:
        out_df.drop(columns="per").to_csv(out / "predictions.csv", index=False)
        pf.assign(per=pf["per"].astype(str)).to_csv(out / "portfolio_period_returns.csv", index=False)
        beta_df.to_csv(out / "beta_vs_time.csv", index=False)
        prof.to_csv(out / "decile_profile.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        log.info("wrote predictions/portfolio/beta/decile CSVs -> %s", out)

    res = {"predictions": out_df, "portfolio": pf, "summary": summary,
           "beta": beta_df, "decile_profile": prof, "out_dir": str(out)}
    if return_predictor:
        res["predictor"] = predictor
    return res


def _fit_predict(Xtr_raw, ytr, dtr, Xte_raw, dte, preds, te_mask,
                 ph, vh, denoise, drop_train_noise_q, drop_metric,
                 days, Hn, W, task, seeds, epochs, pixel_norm, vix_aug,
                 predict_only=False):
    """QC-filter (train-only) -> optional VIX augment (both) -> train ensemble ->
    predict test. Returns the trained Predictor. Writes into `preds[te_mask]`
    when a test set is supplied."""
    tf, keep = _make_filter(Xtr_raw, ph, vh, denoise, drop_train_noise_q, drop_metric)
    Xtr = tf(Xtr_raw)[keep]
    ytr = ytr[keep]
    dtr = np.asarray(dtr)[keep]
    if vix_aug is not None:
        Xtr = vix_aug.augment(Xtr, dtr)
    predictor = train_cnn(Xtr, ytr, days, Hn, W, task=task, seeds=seeds,
                          epochs=epochs, pixel_norm=pixel_norm)
    if Xte_raw is not None and preds is not None:
        Xte = tf(Xte_raw)
        if vix_aug is not None:
            Xte = vix_aug.augment(Xte, dte)
        preds[te_mask] = predictor(Xte)
    return predictor


def _maybe_vix(add_vix, data_dir, base, df, df_te, days, W, vix_rows,
               symbol, scale, csv, download, train_lo, train_hi):
    """Build a VixAugmenter whose scaler is fit on the TRAIN window only, or None."""
    if not add_vix:
        return None
    lo = min(pd.Timestamp(df["date"].min()), pd.Timestamp(df_te["date"].min()))
    hi = max(pd.Timestamp(df["date"].max()), pd.Timestamp(df_te["date"].max()))
    vser = load_vix(start=(lo - pd.Timedelta(days=10)).date(),
                    end=(hi + pd.Timedelta(days=5)).date(),
                    data_dir=(data_dir or base), symbol=symbol, cache=csv, download=download)
    aug = VixAugmenter(vser, days=days, W=W, vix_rows=vix_rows,
                       train_lo=train_lo, train_hi=train_hi, mode=scale)
    log.info("VIX channel: +%d rows, scaler=%s fit on train %s..%s (%d VIX obs)",
             vix_rows, scale, pd.Timestamp(train_lo).date(), pd.Timestamp(train_hi).date(),
             len(aug.vvals))
    return aug


# ---- small tag helpers ------------------------------------------------
def _tag_from_pattern(pattern):
    if not pattern:
        return "data"
    s = Path(str(pattern)).stem
    for junk in ("_picMap", "_price_chart_image_df", "price_chart_image_df"):
        s = s.replace(junk, "")
    s = s.strip("*_.").replace("*", "")
    return s or "data"


def _auto_tag(uni_tr, uni_te, days, h, task, protocol, pixel_norm, add_vix,
              train_years, cross):
    uni = f"{uni_tr}_to_{uni_te}" if cross else uni_tr
    bits = [uni, f"{days}MA", f"h{h}", task]
    if protocol != "faithful":
        bits.append(protocol)
    if pixel_norm != "global":
        bits.append("perimg")
    if add_vix:
        bits.append("vix")
    if int(train_years) != 8:
        bits.append(f"{int(train_years)}y")
    return "_".join(bits)


# ======================================================================
# PART 5 -- synthetic fixtures (schema-faithful; used for offline testing)
# ======================================================================
def make_synth_panel(n_stocks=80, start="2004-01-01", years=12, beta=0.9,
                     seed=0, days=20, universe="R1000", drift=0.0003, out=None):
    """Write a pickle with the EXACT schema of the real data so the runner can be
    exercised without it. `days` in {5,20,60} picks the JKX geometry; a faint
    momentum signal is baked into the forward returns. `universe` names the file
    (e.g. 'sp500' -> sp500_picMap5MA_price_chart_image_df.pickle). TEST USE ONLY."""
    H, W, PH, VH = GEOMETRY[days]; DAYS, PXW = days, 3
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=years * 252)

    def draw(o, hi, lo, c, ma, vol):
        img = np.zeros((H, W), np.uint8)
        a, b = float(np.nanmin(lo)), float(np.nanmax(hi)); r = (b - a) or 1.0
        yp = lambda v: PH - 1 - np.clip(((v - a) / r) * (PH - 1), 0, PH - 1)
        yo, yh, yl, yc, yma = yp(o), yp(hi), yp(lo), yp(c), yp(ma)
        vmax = float(np.nanmax(vol)) or 1.0
        for d in range(DAYS):
            x = d * PXW
            t1, t2 = sorted((int(round(yh[d])), int(round(yl[d]))))
            img[t1:t2 + 1, x + 1] = 1
            img[int(round(yo[d])), x] = 1; img[int(round(yc[d])), x + 2] = 1
            if np.isfinite(yma[d]):
                img[int(round(yma[d])), x + 1] = 1
            vb = int(round((vol[d] / vmax) * (VH - 1)))
            img[H - 1 - vb:H, x + 1] = 1
        return img.ravel()

    rows = []
    for s in range(n_stocks):
        ret = rng.normal(drift, 0.02, len(dates))
        p = 100 * np.exp(np.cumsum(ret))
        op = p * (1 + rng.normal(0, 0.005, len(dates)))
        spread = np.abs(rng.normal(0, 0.01, len(dates))) * p
        hi = np.maximum(p, op) + spread; lo = np.minimum(p, op) - spread
        vol = np.abs(rng.normal(1.0, 0.4, len(dates))) + 0.2
        ma = pd.Series(p).rolling(DAYS).mean().values
        wk = pd.Series(dates).dt.to_period("W")
        for dt in pd.Series(dates).groupby(wk).max():
            t = int(np.searchsorted(dates.values, np.datetime64(dt)))
            if t < DAYS or t + 60 >= len(dates):
                continue
            sl = slice(t - DAYS + 1, t + 1)
            trend = (p[t] - p[t - DAYS + 1]) / p[t - DAYS + 1]
            rows.append(dict(
                sedol=f"{universe[:3].upper()}{s:05d}", date=dt, days=DAYS,
                ph=PH, vh=VH, MA=True, idx=len(rows),
                rF5D=np.float32(p[t + 5] / p[t] - 1 + beta * .15 * trend + rng.normal(0, .03)),
                rF20D=np.float32(p[t + 20] / p[t] - 1 + beta * .30 * trend + rng.normal(0, .05)),
                rF60D=np.float32(p[t + 60] / p[t] - 1 + beta * .20 * trend + rng.normal(0, .09)),
                img_height=H, img_width=W,
                price_chart_image=draw(op[sl], hi[sl], lo[sl], p[sl], ma[sl], vol[sl])))
    dfr = pd.DataFrame(rows)
    out = Path(out or Path.cwd() / f"{universe}_picMap{days}MA_price_chart_image_df.pickle")
    dfr.to_pickle(out)
    print(f"wrote {len(dfr):,} synthetic images -> {out}")
    return out


def make_synth_vix(start="2004-01-01", years=13, seed=7, data_dir=None, cache="vix.csv"):
    """Write a schema-simple vix.csv (columns date,vix) so add_vix works offline
    without yfinance. Mean-reverting ~[10,45]. TEST USE ONLY."""
    d = Path(data_dir) if data_dir else Path.cwd()
    dates = pd.bdate_range(start, periods=int(years * 252))
    rng = np.random.default_rng(seed)
    v = np.empty(len(dates)); v[0] = 18.0
    for i in range(1, len(dates)):                 # AR(1) mean-reversion + shocks
        v[i] = max(9.0, 0.94 * v[i - 1] + 0.06 * 19.0 + rng.normal(0, 1.1)
                   + (rng.random() < 0.01) * rng.normal(0, 8))
    s = pd.Series(np.clip(v, 9, 80), index=dates, name="vix"); s.index.name = "date"
    fp = d / cache; s.to_frame().to_csv(fp)
    print(f"wrote {len(s):,} synthetic VIX obs -> {fp}")
    return fp


def make_synth_suite(data_dir=None, years=22, n_stocks=50):
    """Create sp500_5MA, sp600_5MA, sp500_20MA panels + vix.csv for a full offline
    dry-run of every experiment (incl. the 15-year-train case). TEST USE ONLY."""
    d = Path(data_dir) if data_dir else Path.cwd()
    make_synth_panel(n_stocks=n_stocks, days=5, universe="sp500", seed=1, beta=1.0,
                     years=years, out=d / "SP500_pic_Map5MA_price_chart_image_df.pickle")
    make_synth_panel(n_stocks=n_stocks, days=5, universe="sp600", seed=2, beta=0.7,
                     drift=0.0002, years=years,
                     out=d / "SP600_pic_Map5MA_price_chart_image_df.pickle")
    make_synth_panel(n_stocks=n_stocks, days=20, universe="sp500", seed=1, beta=1.0,
                     years=years, out=d / "SP500_pic_Map20MA_price_chart_image_df.pickle")
    make_synth_vix(years=years, data_dir=d)
    print("synthetic suite ready.")


# ======================================================================
# PART 6 -- experiment presets (one-line control from the notebook)
# ======================================================================
# Each preset is just kwargs for run(). Predictions + model + half-year beta +
# portfolio-with-dates + decile profile are produced for EVERY preset.
# Patterns are the bare universe tags; they resolve case-insensitively to the real
# files, e.g. 'SP500_5MA' -> SP500_pic_Map5MA_price_chart_image_df.pickle. Here
# '5MA'/'20MA' denote the image LOOK-BACK length (5-day / 20-day), read from the
# file's own `days` field; `horizon` is the separate forward-return holding period.
EXPERIMENTS = {
    # baseline: SP500, 5-day look-back image, horizon 5, faithful clf
    "baseline_sp500_5MA": dict(
        pattern="SP500_5MA", horizon=5, task="clf", protocol="faithful"),

    # (1) train on SP500, test on SP600 (time-aligned, strictly OOS)
    "cross_sp500_to_sp600": dict(
        pattern="SP500_5MA", test_pattern="SP600_5MA", horizon=5,
        task="clf", protocol="faithful"),

    # (5) VIX channel appended under the volume block
    "vix_sp500_5MA": dict(
        pattern="SP500_5MA", horizon=5, add_vix=True, vix_rows=3, vix_scale="minmax"),

    # (6) per-image pixel normalisation instead of the global one
    "perimg_sp500_5MA": dict(
        pattern="SP500_5MA", horizon=5, pixel_norm="per_image"),

    # (7) 15 years of training
    "train15y_sp500_5MA": dict(
        pattern="SP500_5MA", horizon=5, train_years=15),

    # (8) 20-day look-back image with a 5-day horizon (weekly rebalance)
    "sp500_20MA_h5": dict(
        pattern="SP500_20MA", horizon=5),
}


def run_experiment(name, data_dir=None, **overrides):
    """Run one preset by name: run_experiment('cross_sp500_to_sp600').
    Any run() kwarg can be overridden, e.g. run_experiment('vix_sp500_5MA', seeds=3)."""
    if name not in EXPERIMENTS:
        _hint(f"unknown experiment {name!r}",
              "choose from: " + ", ".join(EXPERIMENTS))
    cfg = {**EXPERIMENTS[name], **overrides}
    if data_dir is not None:
        cfg["data_dir"] = data_dir
    log.info("EXPERIMENT %s | %s", name, cfg)
    return run(**cfg)


def run_all(data_dir=None, **overrides):
    """Run every preset; returns {name: summary}. Heavy -- start with a subset."""
    res = {}
    for name in EXPERIMENTS:
        res[name] = run_experiment(name, data_dir=data_dir, **overrides)["summary"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--pattern", default="R1000_picMap*_price_chart_image_df.pickle")
    ap.add_argument("--test-pattern", default=None, help="cross-universe test file/tag")
    ap.add_argument("--task", default="clf", choices=["clf", "reg"])
    ap.add_argument("--protocol", default="faithful", choices=["faithful", "walkforward"])
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--train-years", type=int, default=8)
    ap.add_argument("--refit-months", type=int, default=12)
    ap.add_argument("--window", default="exp")
    ap.add_argument("--test-start", default=None)
    ap.add_argument("--test-end", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--denoise", default=None, help="despeckle | none")
    ap.add_argument("--drop-train-noise-q", type=float, default=None)
    ap.add_argument("--drop-metric", default="composite")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--min-names", type=int, default=20)
    ap.add_argument("--pixel-norm", default="global", choices=["global", "per_image"])
    ap.add_argument("--add-vix", action="store_true")
    ap.add_argument("--vix-rows", type=int, default=3)
    ap.add_argument("--vix-scale", default="minmax", choices=["minmax", "zscore"])
    ap.add_argument("--no-save-model", action="store_true")
    ap.add_argument("--experiment", default=None, help="run a named EXPERIMENTS preset")
    a = ap.parse_args()
    if a.experiment:
        run_experiment(a.experiment, data_dir=a.data_dir)
        return
    run(a.data_dir, a.pattern, a.test_pattern, a.task, a.protocol, a.horizon,
        a.train_years, a.refit_months, a.window, a.test_start, a.test_end,
        a.seeds, a.epochs, a.denoise, a.drop_train_noise_q, a.drop_metric,
        max_train=a.max_train, min_names=a.min_names, pixel_norm=a.pixel_norm,
        add_vix=a.add_vix, vix_rows=a.vix_rows, vix_scale=a.vix_scale,
        save_model=not a.no_save_model)


if __name__ == "__main__":
    main()

# ---- Embedded base CNN runner captured before the VIX wrapper overrides run/main ----
from types import SimpleNamespace
import inspect
import math
import os
import re

_embedded_base_run = run
base = SimpleNamespace(run=_embedded_base_run, _find_files=_find_files, load_vix=load_vix)


# ======================================================================
# VIX-gated single-file wrapper API
# ======================================================================
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


def _call_base_run(run_kwargs):
    """Call price_trend_cnn.run while tolerating older local signatures."""
    try:
        sig = inspect.signature(base.run)
    except (TypeError, ValueError):
        return base.run(**run_kwargs)
    accepts_extra = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_extra:
        return base.run(**run_kwargs)
    accepted = {k: v for k, v in run_kwargs.items() if k in sig.parameters}
    return base.run(**accepted)


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
                "lo_mean": float(g["long_only"].mean()),
                "ls_sharpe_active_bucket": _sharpe(g["long_short"], 52.0),
                "lo_sharpe_active_bucket": _sharpe(g["long_only"], 52.0),
                "gated_ls_mean": float(g["gated_long_short"].mean()),
                "gated_lo_mean": float(g["gated_long_only"].mean()),
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

    def cumulative_plot(raw_col, gated_col, label, filename):
        fig, ax1 = plt.subplots(figsize=(12, 5))
        raw = (1.0 + periods[raw_col].fillna(0.0)).cumprod()
        gated = (1.0 + periods[gated_col].fillna(0.0)).cumprod()
        ax1.plot(dates, raw, lw=1.6, label=f"raw {label}")
        ax1.plot(dates, gated, lw=1.9, label=f"{gate_name} {label}")
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
        ax1.set_title(f"Cumulative {label} performance with {gate_name}")
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    cumulative_plot("long_short", "gated_long_short", "long-short", "vix_gate_cumulative_long_short.png")
    cumulative_plot("long_only", "gated_long_only", "long-only", "vix_gate_cumulative_long_only.png")

    def scatter_plot(return_col, label, filename):
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = np.where(periods["active"], "tab:blue", "tab:red")
        ax.scatter(periods["vix"], periods[return_col], c=colors, s=28, alpha=0.75)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("VIX at rebalance")
        ax.set_ylabel(f"weekly {label} return")
        ax.set_title(f"Where {label} works vs fails")
        ax.grid(alpha=0.25)
        path = out_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    scatter_plot("long_short", "long-short", "vix_vs_long_short.png")
    scatter_plot("long_only", "long-only", "vix_vs_long_only.png")

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

        fig, ax1 = plt.subplots(figsize=(12, 5))
        x = np.arange(len(bucket))
        colors = np.where(bucket["lo_mean"] >= 0, "tab:green", "tab:red")
        ax1.bar(x, bucket["lo_mean"], color=colors, alpha=0.8)
        ax1.axhline(0, color="black", lw=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(bucket["vix_bucket"], rotation=35, ha="right")
        ax1.set_ylabel("mean weekly long-only")
        ax2 = ax1.twinx()
        ax2.plot(x, bucket["ic_mean"], color="tab:blue", marker="o", label="mean IC")
        ax2.set_ylabel("mean Spearman IC")
        ax1.set_title("VIX bucket long-only performance")
        fig.tight_layout()
        path = out_dir / "vix_bucket_long_only_performance.png"
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
        train_years=8, seeds=5, epochs=50, pixel_norm="global",
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

    base_kwargs = dict(
        data_dir=str(run_data_dir),
        pattern=run_pattern,
        task=task,
        protocol=protocol,
        horizon=horizon,
        train_years=train_years,
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
    )
    base_kwargs.update(kwargs)
    res = _call_base_run(base_kwargs)
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
            "train_years": None if train_years is None else int(train_years),
            "rebalance": freq,
            "test_start": test_start,
            "test_end": test_end,
            "threshold_note": "If selected using the test window, treat this as diagnostic, not clean OOS.",
        }
    )
    summary.update(filter_meta)
    bucket = vix_bucket_summary(periods)

    out_dir.mkdir(parents=True, exist_ok=True)
    long_short_csv = out_dir / "long_short_period_returns.csv"
    long_only_csv = out_dir / "long_only_period_returns.csv"
    combined_returns_csv = out_dir / "portfolio_period_returns.csv"
    summary.update(
        {
            "long_short_csv": str(long_short_csv),
            "long_only_csv": str(long_only_csv),
            "portfolio_period_returns_csv": str(combined_returns_csv),
        }
    )
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    periods.to_csv(out_dir / "vix_period_diagnostics.csv", index=False)
    periods[[
        "per", "rebalance_date", "vix", "active", "long_only", "long_short",
        "gated_long_only", "gated_long_short", "n_names",
    ]].to_csv(combined_returns_csv, index=False)
    periods[[
        "per", "rebalance_date", "vix", "active", "vix_regime",
        "long_short", "gated_long_short", "n_names",
        "ic_spearman", "wrong_tail", "top_hit", "bottom_hit",
    ]].to_csv(long_short_csv, index=False)
    periods[[
        "per", "rebalance_date", "vix", "active", "vix_regime",
        "long_only", "gated_long_only", "n_names",
        "ic_spearman", "wrong_tail", "top_hit", "bottom_hit",
    ]].to_csv(long_only_csv, index=False)
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
    ap.add_argument("--train-years", type=int, default=8,
                    help="limit faithful training to this many years before test_start")
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
        train_years=args.train_years,
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
