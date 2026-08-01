"""company_replication.py — Mimicking Finance 复现实验（公司数据版，无需 WRDS）

只依赖你自己那一个 holdings parquet。跑完给出三组结果：

  1. PRECISION      真实持仓上的精度  +  按 N 格模板把 padding 计入后的精度
                    （论文的 0.71 / 0.52 很可能是把空格子算进去的结果）
  2. TABLE X        基金按可预测性分五组 -> 未来 1~4 季度累计超额收益 CRET_{0,1..4}
                    论文: Q1 +0.36 / Q5 -0.42 / Q5-Q1 -0.79 (t=-3.05)
  3. TABLE XII      股票按跨基金预测准确率分五组 -> Q1-Q5
                    论文: +1.06%/季 (t=5.74)

每个结果都在三种时间口径下报告：
  contemporaneous  acc(t) x t->t+1     有重叠，有偏，只作基准
  predictive       acc(t) x t+1->t+2   无重叠，但忽略 13F 披露延迟
  tradeable        acc(t) x t+2->t+3   还扣掉 45-60 天披露延迟，真正可交易

关键开关 `use_manager_memory`
---------------------------
manager-memory 特征（fund/fund-security 的历史买卖持比例）能把精度从 0.53 提到 0.58，
但它会把"可预测"悄悄变成"这个经理根本不动这个仓位"。低换手基金历史上跑赢，于是
Table X 的符号会被**翻转**（实测 Q5-Q1 从 -0.66 变成 +0.45）。

    use_manager_memory=False  -> Q5-Q1 = -0.66 (t=-3.20)   ≈ 论文 -0.79 (t=-3.05)  ✅
    use_manager_memory=True   -> Q5-Q1 = +0.45 (t=+3.57)   与论文反号             ❌

所以复现论文请用 False；True 只用来演示"提高精度反而破坏经济含义"这个对照。

用法
----
    import company_replication as R
    cfg = R.Config(data_path="你的.parquet")
    res = R.run_ablation(cfg)          # 一次跑完 False/True 两版并对比
    R.show(res)
"""
from __future__ import annotations
import os
import warnings
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================ CONFIG
@dataclass
class Config:
    data_path: str = "master_all_funds_add_filter_ivy_rank_active_rank.parquet"
    col_map: dict = field(default_factory=lambda: {
        "fund": "fund", "date": "date", "security": "security", "shares": "shares",
        "position_value": "position_value", "market_cap": "market_cap", "isUs": "isUs",
        "quarterly_ret": "quarterly_ret", "past_1q_ret": "past_1q_ret",
        "future_1q_ret": "future_1q_ret", "future_2q_ret": "future_2q_ret",
        "future_3q_ret": "future_3q_ret", "InvTypeCode": "inv_type",
        "future_1q_shares_change_pct": "chg_pct",
        "portfolio_value": "portfolio_value", "weight": "weight", "rank": "rank",
        "n_holdings": "n_holdings", "volume": "volume",
    })
    inv_type_codes: tuple = (401,)     # 先从 401 开始
    us_only: bool = True
    max_rank: int = 25                 # 只保留前 N 大持仓（None = 用文件里现有的）
    change_band: float = 0.01          # ±1% 死区
    # chg_pct == -100% 表示"下季度没有持仓信息"，不是真卖出 -> 预处理阶段丢掉
    drop_missing_position: bool = True

    # ---- 关键开关：见模块 docstring ----
    use_manager_memory: bool = False

    # ---- padding：论文的 N 格模板 ----
    template_N: int = 75               # 模板宽度；padding 比例由持仓数分布决定

    # ---- 滚动窗口（论文 Fig 2）----
    window_q: int = 28
    test_q: int = 8
    step: int = 8
    min_years: int = 7                 # >7 年历史
    min_holdings: int = 10             # 每季 >=10 只（自动被 max_rank 上限截断）

    # ---- 模型 ----
    # "gbm"        梯度提升树，CPU 几分钟，把序列拍平成 y_lag1..4 等列
    # "lstm"       权重共享的单仓位序列 LSTM：一个样本 = 一个仓位最近 seq_len 季 [T, F]
    # "panel_lstm" 论文原架构：一个样本 = 一个 fund-quarter 的整个截面
    #              [T, N, F] -> LSTM(N*F -> numcell) -> Linear -> [N, 3]，N = max_rank
    model: str = "gbm"
    max_iter: int = 250                # gbm
    learning_rate: float = 0.08
    max_depth: int = 7
    n_max_train: int = 1_500_000       # 每窗口训练样本上限（控制耗时）
    seed: int = 0
    # ---- lstm 专用 ----
    seq_len: int = 8                   # 输入序列长度（论文 T=8）
    hidden: int = 64
    dropout: float = 0.25
    lr: float = 3e-3
    max_epochs: int = 25
    patience: int = 5
    batch: int = 8192
    device: str = "auto"               # "auto" | "cuda" | "cpu"
    # 默认 None = 用全部样本，不抽样。序列是按索引惰性拼的（见 _build_sequences），
    # 内存 ≈ 特征矩阵 + 索引，1000 万行约 2 GB，所以全量训练是可行的。
    # 只有在纯 CPU 且实在等不及时才设个上限（例如 300_000）来换速度。
    lstm_max_train: int = None         # 每窗口训练序列上限（None = 全部）
    lstm_max_rows: int = None          # 建序列前先对面板抽样（None = 不抽）

    @property
    def base_features(self) -> List[str]:
        return ["weight", "w_lag1", "dw", "rank", "rank_pct", "log_posval", "log_pv",
                "log_mktcap", "quarterly_ret", "past_1q_ret",
                "pdsh", "pdsh_sign", "pdsh_lag1", "sh_lag1", "sh_lag2", "sh_lag3",
                "peer_buy", "peer_sell", "peer_hold", "n_holdings",
                "n_funds", "log_inst_own", "sum_abs_aw", "own_rank",
                "y_lag1", "y_lag2", "y_lag3", "y_lag4", "pos_age", "d_rank", "w_drift",
                # 成交量：决定这个仓位「能不能」被交易掉，而不只是想不想。
                # 文件里没有 volume 列时这几个会自动缺席，不影响运行。
                "log_volume", "vol_rank", "pos_to_vol", "d_log_vol", "amihud"]

    @property
    def memory_features(self) -> List[str]:
        return ["fund_buy_rate", "fund_hold_rate", "fund_sell_rate",
                "fs_hold_rate", "fs_buy_rate", "fs_sell_rate", "fs_n_obs",
                "sec_buy_rate", "sec_hold_rate", "sec_sell_rate"]

    @property
    def features(self) -> List[str]:
        return self.base_features + (self.memory_features if self.use_manager_memory else [])


# ============================================================ 工具
def _t(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan


def _q5(s, n=5):
    return (pd.qcut(s.rank(method="first"), n, labels=False, duplicates="drop") + 1
            if s.nunique() >= n else pd.Series(np.nan, index=s.index))


def _chg_scale(chg, cfg):
    nz = chg[chg.abs() > 1e-9].abs()
    med = float(nz.median()) if len(nz) else np.nan
    return "percent" if (np.isfinite(med) and med > 1.5) else "fraction"


# ============================================================ 数据 + 特征
def load_and_prepare(cfg: Config) -> pd.DataFrame:
    inv = {v: k for k, v in cfg.col_map.items()}
    want = ["fund", "date", "security", "shares", "position_value", "market_cap",
            "quarterly_ret", "past_1q_ret", "future_1q_ret", "future_2q_ret",
            "future_3q_ret", "chg_pct", "inv_type", "portfolio_value", "weight",
            "rank", "n_holdings", "isUs", "volume"]
    want_raw = [inv[c] for c in want if c in inv]
    try:
        import pyarrow.parquet as pq
        avail = set(pq.ParquetFile(cfg.data_path).schema.names)
        df = pd.read_parquet(cfg.data_path, columns=[c for c in want_raw if c in avail])
    except Exception:
        df = pd.read_parquet(cfg.data_path)
    df = df.rename(columns={inv[c]: c for c in cfg.col_map.values() if inv[c] in df.columns})

    df["date"] = pd.to_datetime(df["date"])
    df["yq"] = df["date"].dt.to_period("Q")
    if cfg.us_only and "isUs" in df.columns:
        df = df[df["isUs"].astype(bool)]
    if cfg.inv_type_codes is not None and "inv_type" in df.columns:
        codes = {str(c) for c in cfg.inv_type_codes}
        df = df[df["inv_type"].astype(str).isin(codes)]
    df = df.sort_values("date").drop_duplicates(["fund", "yq", "security"], keep="last")
    if "rank" in df.columns and cfg.max_rank:
        df = df[df["rank"] <= cfg.max_rank]
    df = df.drop(columns=[c for c in ("date", "isUs") if c in df.columns])

    F32 = "float32"
    for c in ["shares", "position_value", "market_cap", "quarterly_ret", "past_1q_ret",
              "future_1q_ret", "future_2q_ret", "future_3q_ret", "chg_pct",
              "portfolio_value", "weight", "rank", "n_holdings", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(F32) if c in df.columns \
            else np.array(np.nan, dtype=F32)

    # ---- 预处理：丢掉"下季无持仓信息"的哨兵行（-100%），在任何特征之前 ----
    if cfg.drop_missing_position and df["chg_pct"].notna().any():
        chg = df["chg_pct"].astype("float64")
        sc = _chg_scale(chg, cfg)
        frac = chg / 100.0 if sc == "percent" else chg
        bad = (frac + 1.0).abs() < 1e-6
        print(f"[data] 预过滤: 丢弃 {int(bad.sum()):,} 行 ({bad.mean():.1%}) chg_pct=-100% "
              f"（单位: {sc}）—— 下季无持仓信息，非真实卖出")
        df = df[~bad]

    df = df.sort_values(["fund", "security", "yq"]).reset_index(drop=True)
    keys = ["fund", "security"]
    g = df.groupby(keys, observed=True)

    # ---- 目标：严格下一季度 ----
    if df["chg_pct"].notna().any():
        chg = df["chg_pct"].astype("float64")
        sc = _chg_scale(chg, cfg)
        dsh = chg / 100.0 if sc == "percent" else chg
        print(f"[data] 目标来自 future_1q_shares_change_pct（单位: {sc}）")
    else:
        sh_n = g["shares"].shift(-1); q_n = g["yq"].shift(-1)
        sh_n = sh_n.where(q_n == df["yq"] + 1)                   # 严格 t+1
        dsh = (sh_n - df["shares"]) / (df["shares"].abs() + 1.0)
        print("[data] 无 chg_pct 列 -> 目标用 shares[t+1]（严格 t+1，跨缺口丢弃）")
    df["Y"] = np.select([dsh <= -cfg.change_band, dsh >= cfg.change_band],
                        [-1.0, 1.0], default=0.0).astype(F32)
    df.loc[pd.isna(dsh), "Y"] = np.nan
    # 已实现收益：只用于评估，绝不做特征。三个horizon都带上，一次训练即可出三种口径。
    df["fwd_1q"] = df["future_1q_ret"]      # t   -> t+1
    df["fwd_2q"] = df["future_2q_ret"]      # t+1 -> t+2
    df["fwd_3q"] = df["future_3q_ret"]      # t+2 -> t+3
    bal = pd.Series(df["Y"]).value_counts(normalize=True)
    print(f"[data] 类别分布  卖 {bal.get(-1.,0):.3f} | 持 {bal.get(0.,0):.3f} | "
          f"买 {bal.get(1.,0):.3f}  （有标签 {int(df['Y'].notna().sum()):,}）")

    # ---- 特征：水平量用 forward-fill（只用过去），绝不 fillna(0) ----
    for k in (1, 2, 3):
        df[f"sh_lag{k}"] = g["shares"].shift(k).fillna(df["shares"]).astype(F32)
    df["w_lag1"] = g["weight"].shift(1).fillna(df["weight"]).astype(F32)
    df["dw"] = (df["weight"] - df["w_lag1"]).astype(F32)
    df["pdsh"] = ((df["shares"] - df["sh_lag1"]) / (df["sh_lag1"].abs() + 1.0)).astype(F32)
    df["pdsh_sign"] = np.sign(df["pdsh"]).fillna(0.0).astype(F32)
    df["pdsh_lag1"] = df.groupby(keys, observed=True)["pdsh"].shift(1).fillna(0.0).astype(F32)
    df["log_posval"] = np.log(df["position_value"].abs() + 1.0).astype(F32)
    df["log_pv"] = np.log(df["portfolio_value"].abs() + 1.0).astype(F32)
    df["log_mktcap"] = np.log(df["market_cap"].abs() + 1.0).astype(F32)
    df["rank_pct"] = (df["rank"] / df["n_holdings"].where(df["n_holdings"] > 0)).astype(F32)

    # 样本过滤（论文 §3.1）
    mh = min(cfg.min_holdings, cfg.max_rank or cfg.min_holdings)
    cnt = df.groupby(["fund", "yq"], observed=True)["security"].transform("size")
    df = df[cnt >= mh]
    nq = df.groupby("fund", observed=True)["yq"].transform("nunique")
    df = df[nq >= cfg.min_years * 4]

    # 季度整数索引
    qs = pd.PeriodIndex(sorted(df["yq"].unique()), freq="Q")
    df["qi"] = df["yq"].map({q: i for i, q in enumerate(qs)}).astype("int32")

    # ---- 同行活跃度（滞后一个精确季度）----
    lab = df.dropna(subset=["Y"])
    rate = lab.groupby("yq")["Y"].agg(peer_buy=lambda s: (s > 0).mean(),
                                      peer_sell=lambda s: (s < 0).mean(),
                                      peer_hold=lambda s: (s == 0).mean())
    allq = pd.PeriodIndex(sorted(df["yq"].unique()), freq="Q")
    rate = rate.reindex(allq).ffill()
    prev = df["yq"] - 1
    for c in ("peer_buy", "peer_sell", "peer_hold"):
        df[c] = prev.map(rate[c]).astype(F32)

    # ---- 证券层面：跨基金持有情况（只用 t 期持仓）----
    gsq = df.groupby(["security", "yq"], observed=True)
    df["n_funds"] = gsq["fund"].transform("size").astype(F32)
    own = df["position_value"] / df["market_cap"].where(df["market_cap"] > 0)
    df["own_frac"] = own.astype(F32)
    inst = df.groupby(["security", "yq"], observed=True)["own_frac"].transform("sum")
    df["log_inst_own"] = np.log(inst.abs() + 1e-6).astype(F32)
    df["own_rank"] = inst.groupby(df["yq"]).rank(pct=True).astype(F32)
    gf = df.groupby(["fund", "yq"], observed=True)
    aw = (df["position_value"] / gf["position_value"].transform("sum")
          - df["market_cap"] / gf["market_cap"].transform("sum")).abs()
    df["sum_abs_aw"] = aw.groupby([df["security"].to_numpy(),
                                   df["yq"].to_numpy()]).transform("sum").astype(F32)

    # ---- 成交量：流动性约束 ----
    # 「想卖」和「卖得掉」是两回事。一个占了 20 天成交量的仓位，经理即使想清仓也只能
    # 慢慢减，所以下季更可能是"小幅卖"而不是"清仓"；反过来流动性好的仓位交易更随意。
    # 这几个特征只用 t 期及以前的信息。文件里没有 volume 列时全部跳过。
    if "volume" in df.columns and df["volume"].notna().any():
        vol = df["volume"].astype("float64").where(df["volume"] > 0)
        df["log_volume"] = np.log(vol.fillna(0) + 1.0).astype(F32)
        # 每季度横截面分位数：对「volume 是股数还是金额」这个单位问题免疫
        df["vol_rank"] = vol.groupby(df["yq"]).rank(pct=True).astype(F32)
        # 仓位相当于多少个单位的成交量 —— 「几天能卖完」的代理，越大越难脱手
        df["pos_to_vol"] = (df["shares"] / vol).clip(0, 50).astype(F32)
        # Amihud 式非流动性：单位成交量对应多少收益波动（越大越不流动）
        df["amihud"] = (df["quarterly_ret"].abs() / (vol / 1e6)).clip(0, 100).astype(F32)
        gv = df.groupby(["security", "yq"], observed=True)["log_volume"].first()
        sv = df[["security", "yq"]].copy()
        prev = pd.MultiIndex.from_arrays([sv["security"].to_numpy(),
                                          (sv["yq"] - 1).to_numpy()])
        df["d_log_vol"] = (df["log_volume"] - gv.reindex(prev).to_numpy()).astype(F32)
        n_vol = int(df["log_volume"].notna().sum())
        print(f"[data] 成交量特征已加（覆盖 {n_vol/len(df):.1%}）："
              f"log_volume / vol_rank / pos_to_vol / d_log_vol / amihud")
    else:
        print("[data] 文件里没有 volume 列 -> 跳过成交量特征")

    # ---- 序列/生命周期（精确季度对齐）----
    df = df.sort_values(["fund", "security", "qi"]).reset_index(drop=True)
    g = df.groupby(keys, observed=True, sort=False)
    qi = df["qi"].to_numpy()
    for k in (1, 2, 3, 4):
        v, q = g["Y"].shift(k), g["qi"].shift(k)
        df[f"y_lag{k}"] = v.where(q == qi - k).astype(F32)
    newblk = (g["qi"].diff() != 1).to_numpy()
    blk = np.cumsum(newblk)
    df["pos_age"] = df.groupby([df["fund"].to_numpy(), df["security"].to_numpy(), blk],
                               sort=False).cumcount().astype(F32)
    rk, rq = g["rank"].shift(1), g["qi"].shift(1)
    df["d_rank"] = (df["rank"] - rk.where(rq == qi - 1)).astype(F32)
    df["w_drift"] = ((df["weight"] - df["w_lag1"]) /
                     (df["w_lag1"].abs() + 1e-6)).clip(-5, 5).astype(F32)

    # ---- manager memory（扩张窗口、严格过去）----
    _y = df["Y"].to_numpy(); _lab = ~np.isnan(_y)
    df["_lab"] = _lab.astype(F32)
    for w, v in (("buy", (_y == 1) & _lab), ("hold", (_y == 0) & _lab),
                 ("sell", (_y == -1) & _lab)):
        df[f"_{w}"] = v.astype(F32)

    def past_rates(kk, prefix):
        gg = df.groupby(kk, observed=True, sort=False)
        cnt = gg["_lab"].cumsum().to_numpy() - df["_lab"].to_numpy()
        for w in ("buy", "hold", "sell"):
            cs = gg[f"_{w}"].cumsum().to_numpy() - df[f"_{w}"].to_numpy()
            df[f"{prefix}_{w}_rate"] = np.where(cnt > 0, cs / np.maximum(cnt, 1.0),
                                                np.nan).astype(F32)
        df[f"{prefix}_n_obs"] = gg.cumcount().to_numpy().astype(F32)

    past_rates(["fund"], "fund"); past_rates(keys, "fs"); past_rates(["security"], "sec")
    df.drop(columns=["_lab", "_buy", "_hold", "_sell"], inplace=True)

    print(f"[data] 面板 {len(df):,} 行 | {df.fund.nunique():,} 基金 | "
          f"{df.security.nunique():,} 证券 | {df.qi.max()+1} 季度")
    return df


# ============================================================ 模型（滚动，样本外）
_KEEP = ["fund", "security", "qi", "Y", "weight", "w_lag1", "n_holdings",
         "fwd_1q", "fwd_2q", "fwd_3q"]


def _pick_device(cfg):
    """选设备，并检查这块 GPU 的算力是否被当前 torch 支持。
    新卡（如 RTX 50 系 = sm_120）配旧 torch(cu124) 会在 cuDNN 里直接崩，
    与其崩不如自动退到 CPU 并给出可执行的修复建议。"""
    import torch
    if cfg.device != "auto":
        return cfg.device
    if not torch.cuda.is_available():
        return "cpu"
    try:
        cap = torch.cuda.get_device_capability(0)
        sm = f"sm_{cap[0]}{cap[1]}"
        if sm not in torch.cuda.get_arch_list():
            print(f"  [warn] GPU 算力 {sm} 不被当前 torch({torch.__version__}) 支持 "
                  f"(支持: {torch.cuda.get_arch_list()[-3:]}...) -> 退回 CPU。\n"
                  f"         想用 GPU 请装匹配的版本，例如 RTX 50 系:\n"
                  f"         pip install --pre torch --index-url "
                  f"https://download.pytorch.org/whl/nightly/cu128")
            return "cpu"
    except Exception:
        return "cpu"
    return "cuda"


def _build_sequences(sub: pd.DataFrame, feats, seq_len):
    """构建序列的**索引**，而不是把 [N, T, F] 张量整个物化。

    物化的代价：1000 万行 x 8 步 x 40 特征 x 4 字节 ≈ 13 GB，标准化再复制一份就 26 GB。
    这里只存两样东西：
        Feat      [n_rows, F]  float32  —— 特征矩阵本身（每行只存一次，不随窗口重复）
        hist_idx  [N, T]       int32    —— 每个样本第 t 步该取 Feat 的哪一行
    batch 时用 Feat[hist_idx[bi]] 现场拼出 [batch, T, F]。内存从 ~13 GB 降到
    ~(n_rows x F x 4) + (N x T x 4)，1000 万行约 1.6 GB + 0.2 GB。

    该仓位在某季度不存在时，那一步的 mask=0（LSTM 会屏蔽），索引随便填 0。
    标签在最后一步。返回 (Feat, hist_idx, mask, y, meta)。
    """
    sub = sub.sort_values(["fund", "security", "qi"]).reset_index(drop=True)
    g = sub.groupby(["fund", "security"], observed=True, sort=False)
    valid = sub["Y"].notna().to_numpy()
    N = int(valid.sum())
    if N == 0:
        return None
    F = len(feats)
    qi = sub["qi"].to_numpy()
    Feat = np.nan_to_num(sub[feats].to_numpy(dtype="float32", na_value=np.nan),
                         nan=0.0, posinf=0.0, neginf=0.0)
    row = pd.Series(np.arange(len(sub), dtype="int64"), index=sub.index)

    hist_idx = np.zeros((N, seq_len), dtype=np.int32)
    M = np.zeros((N, seq_len), dtype=np.float32)
    for k in range(seq_len):
        step = seq_len - 1 - k                      # k=0 是当期，放最后一步
        rk = row.groupby([sub["fund"], sub["security"]], observed=True,
                         sort=False).shift(k).to_numpy(dtype="float64", na_value=np.nan)
        qk = g["qi"].shift(k).to_numpy(dtype="float64", na_value=np.nan)
        present = (qk == qi - k) & ~np.isnan(rk)    # 必须是精确的 t-k 季度
        pv = present[valid]
        M[:, step] = pv.astype(np.float32)
        hist_idx[:, step] = np.where(pv, np.nan_to_num(rk[valid], nan=0.0), 0).astype(np.int32)
    y = (sub["Y"].to_numpy()[valid] + 1).astype(np.int64)      # {-1,0,1} -> {0,1,2}
    meta = sub.loc[valid, [c for c in _KEEP if c in sub.columns]].reset_index(drop=True)
    return Feat, hist_idx, M, y, meta


def _fit_lstm(Feat, hist_idx, M, y, tr, te, cfg):
    """在 tr 上训练一个共享权重的序列 LSTM，返回 te 上的预测（{-1,0,1}）。
    序列在 batch 里用 Feat[hist_idx[bi]] 现场拼，全程不物化 [N,T,F]。"""
    import torch, torch.nn as nn
    dev = _pick_device(cfg)
    torch.manual_seed(cfg.seed)
    F = Feat.shape[1]
    tr_i_all = np.where(tr)[0]
    if len(tr_i_all) < 100:
        return None, dev
    # 标准化统计量：直接在 Feat 上按"被训练样本用到的行"估计，不需要展开序列
    used = np.unique(hist_idx[tr_i_all][M[tr_i_all] > 0])
    if used.size < 50:
        return None, dev
    if used.size > 500_000:                     # 估计 mu/sd 用抽样即可，不影响训练样本量
        used = np.random.default_rng(cfg.seed).choice(used, 500_000, replace=False)
    mu = Feat[used].mean(0).astype(np.float32)
    sd = (Feat[used].std(0) + 1e-6).astype(np.float32)

    def _batch(bi):
        """现场拼序列 + 标准化 + 屏蔽 padding。峰值内存只有一个 batch。"""
        xb = (Feat[hist_idx[bi]] - mu) / sd      # [b, T, F]
        xb *= M[bi][..., None]
        return torch.from_numpy(xb), torch.from_numpy(M[bi])

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(F, cfg.hidden, batch_first=True)
            self.drop = nn.Dropout(cfg.dropout)
            self.head = nn.Linear(cfg.hidden, 3)

        def forward(self, x, m):
            o, _ = self.lstm(x * m.unsqueeze(-1))
            return self.head(self.drop(o[:, -1, :]))

    model = Net().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = nn.CrossEntropyLoss()
    yt = torch.from_numpy(y)
    idx = tr_i_all.copy()
    rng = np.random.default_rng(cfg.seed); rng.shuffle(idx)
    # CPU 上训练量必须封顶，否则一个窗口可能要跑几小时
    if cfg.lstm_max_train and len(idx) > cfg.lstm_max_train:
        idx = idx[:cfg.lstm_max_train]
    nval = max(1, int(0.15 * len(idx)))
    val_i, trn_i = idx[:nval], idx[nval:]
    best, best_state, bad = 1e9, None, 0
    for _ in range(cfg.max_epochs):
        model.train(); rng.shuffle(trn_i)
        for b in range(0, len(trn_i), cfg.batch):
            bi = trn_i[b:b + cfg.batch]
            xb, mb = _batch(bi)
            opt.zero_grad()
            lossf(model(xb.to(dev), mb.to(dev)), yt[bi].to(dev)).backward()
            opt.step()
        model.eval(); tot = n = 0
        with torch.inference_mode():
            for b in range(0, len(val_i), cfg.batch):
                bi = val_i[b:b + cfg.batch]
                xb, mb = _batch(bi)
                l = lossf(model(xb.to(dev), mb.to(dev)), yt[bi].to(dev))
                tot += float(l) * len(bi); n += len(bi)
        vl = tot / max(n, 1)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    te_i = np.where(te)[0]
    preds = []
    with torch.inference_mode():
        for b in range(0, len(te_i), cfg.batch):
            bi = te_i[b:b + cfg.batch]
            xb, mb = _batch(bi)
            preds.append(model(xb.to(dev), mb.to(dev)).argmax(1).cpu().numpy())
    return (np.concatenate(preds) - 1).astype(float) if preds else None, dev


def _build_panel_tensor(sub: pd.DataFrame, feats, seq_len, N):
    """论文原架构的样本：一个 (fund, 标签季度 t) = 一个 [T, N, F] 张量。

    第 j 列 = 该基金在 t 期按持仓市值排名第 j 的证券，然后把**同一只证券**沿
    t-7..t 回溯（论文 §3.1 的模板 + §3.3 的"N 个不同证券标识"）。基金持仓不足 N
    时，多出来的列是 padding（mask=0），这正是把 precision 垫高的那部分。

    同样只存索引不物化：hist [S, T, N] int32 + mask [S, T, N]。
    返回 (Feat, hist, mask, ylab, meta)，ylab[s, j] ∈ {0,1,2}，-1 表示该格无标签。
    """
    sub = sub.sort_values(["fund", "qi", "rank"]).reset_index(drop=True)
    row = np.arange(len(sub), dtype=np.int64)
    # (fund, security, qi) -> 行号，用于沿时间回溯同一只证券
    key = pd.MultiIndex.from_arrays([sub["fund"].to_numpy(), sub["security"].to_numpy(),
                                     sub["qi"].to_numpy()])
    lookup = pd.Series(row, index=key)
    # 每个 fund-quarter 取前 N 名（rank 已排序）
    head = sub.groupby(["fund", "qi"], observed=True).head(N)
    slot = head.groupby(["fund", "qi"], observed=True).cumcount().to_numpy()
    fq = head[["fund", "qi"]].drop_duplicates().reset_index(drop=True)
    fq_id = pd.Series(np.arange(len(fq)), index=pd.MultiIndex.from_arrays(
        [fq["fund"].to_numpy(), fq["qi"].to_numpy()]))
    sid = fq_id.reindex(pd.MultiIndex.from_arrays(
        [head["fund"].to_numpy(), head["qi"].to_numpy()])).to_numpy()
    S = len(fq)
    hist = np.zeros((S, seq_len, N), dtype=np.int32)
    mask = np.zeros((S, seq_len, N), dtype=np.float32)
    ylab = np.full((S, N), -1, dtype=np.int64)
    hrow = head.index.to_numpy()
    yv = sub["Y"].to_numpy()
    ok = ~np.isnan(yv[hrow])
    ylab[sid[ok], slot[ok]] = (yv[hrow][ok] + 1).astype(np.int64)
    for k in range(seq_len):
        step = seq_len - 1 - k
        idx = pd.MultiIndex.from_arrays([head["fund"].to_numpy(),
                                         head["security"].to_numpy(),
                                         head["qi"].to_numpy() - k])
        r = lookup.reindex(idx).to_numpy()          # 同一只证券在 t-k 的行号
        pres = ~pd.isna(r)
        hist[sid[pres], step, slot[pres]] = r[pres].astype(np.int32)
        mask[sid[pres], step, slot[pres]] = 1.0
    Feat = np.nan_to_num(sub[feats].to_numpy(dtype="float32", na_value=np.nan),
                         nan=0.0, posinf=0.0, neginf=0.0)
    meta_cols = [c for c in _KEEP if c in sub.columns]
    meta = head[meta_cols].reset_index(drop=True)
    meta["_sid"] = sid; meta["_slot"] = slot
    return Feat, hist, mask, ylab, meta, fq


def _fit_panel_lstm(Feat, hist, mask, ylab, tr, te, cfg):
    """论文架构：[T, N*F] -> LSTM -> Linear -> [N, 3]。padding 格用 ignore_index=-1 跳过。"""
    import torch, torch.nn as nn
    dev = _pick_device(cfg)
    torch.manual_seed(cfg.seed)
    S, T, N = hist.shape
    F = Feat.shape[1]
    tr_i = np.where(tr)[0]
    if len(tr_i) < 20:
        return None, dev
    used = np.unique(hist[tr_i][mask[tr_i] > 0])
    if used.size < 50:
        return None, dev
    if used.size > 500_000:
        used = np.random.default_rng(cfg.seed).choice(used, 500_000, replace=False)
    mu = Feat[used].mean(0).astype(np.float32)
    sd = (Feat[used].std(0) + 1e-6).astype(np.float32)

    def _batch(bi):
        xb = (Feat[hist[bi]] - mu) / sd                 # [b, T, N, F]
        xb *= mask[bi][..., None]
        b = xb.shape[0]
        step_mask = (mask[bi].sum(axis=2) > 0).astype(np.float32)   # 整季全 padding -> 屏蔽
        return (torch.from_numpy(xb.reshape(b, T, N * F)),
                torch.from_numpy(step_mask))

    numcell = int(np.clip(N, 16, cfg.hidden))

    class PanelNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_drop = nn.Dropout(cfg.dropout)
            self.lstm = nn.LSTM(N * F, numcell, batch_first=True)
            self.out_drop = nn.Dropout(cfg.dropout)
            self.head = nn.Linear(numcell, N * 3)

        def forward(self, x, sm):
            o, _ = self.lstm(self.in_drop(x * sm.unsqueeze(-1)))
            return self.head(self.out_drop(o[:, -1, :])).view(-1, N, 3)

    model = PanelNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-1)
    yt = torch.from_numpy(ylab)
    idx = tr_i.copy()
    rng = np.random.default_rng(cfg.seed); rng.shuffle(idx)
    if cfg.lstm_max_train and len(idx) > cfg.lstm_max_train:
        idx = idx[:cfg.lstm_max_train]
    nval = max(1, int(0.15 * len(idx)))
    val_i, trn_i = idx[:nval], idx[nval:]
    bs = max(32, cfg.batch // max(N, 1))                # 每个样本已是整截面，batch 要小
    best, best_state, bad = 1e9, None, 0
    for _ in range(cfg.max_epochs):
        model.train(); rng.shuffle(trn_i)
        for b in range(0, len(trn_i), bs):
            bi = trn_i[b:b + bs]
            xb, sm = _batch(bi)
            opt.zero_grad()
            lg = model(xb.to(dev), sm.to(dev))
            lossf(lg.reshape(-1, 3), yt[bi].reshape(-1).to(dev)).backward()
            opt.step()
        model.eval(); tot = n = 0
        with torch.inference_mode():
            for b in range(0, len(val_i), bs):
                bi = val_i[b:b + bs]
                xb, sm = _batch(bi)
                lg = model(xb.to(dev), sm.to(dev))
                l = lossf(lg.reshape(-1, 3), yt[bi].reshape(-1).to(dev))
                tot += float(l) * len(bi); n += len(bi)
        vl = tot / max(n, 1)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    te_i = np.where(te)[0]
    P = np.zeros((len(te_i), N), dtype=np.float32)
    with torch.inference_mode():
        for b in range(0, len(te_i), bs):
            bi = te_i[b:b + bs]
            xb, sm = _batch(bi)
            P[b:b + len(bi)] = (model(xb.to(dev), sm.to(dev)).argmax(2).cpu().numpy() - 1)
    return (te_i, P), dev


def run_model(panel: pd.DataFrame, cfg: Config, verbose=True) -> pd.DataFrame:
    """滚动窗口、样本外预测。cfg.model 决定用 gbm 还是 lstm。"""
    feats = [f for f in cfg.features if f in panel.columns]
    d = panel[panel["Y"].notna()] if cfg.model == "gbm" else panel
    out = []

    if cfg.model == "lstm":
        sub = panel
        if cfg.lstm_max_rows and int(panel["Y"].notna().sum()) > cfg.lstm_max_rows:
            keep = panel[panel["Y"].notna()].sample(cfg.lstm_max_rows,
                                                    random_state=cfg.seed).index
            sub = panel.loc[panel.index.isin(keep) | panel["Y"].isna()]
            print(f"  [lstm] 面板抽样 -> {cfg.lstm_max_rows:,} 条有标签样本")
        seq = _build_sequences(sub, feats, cfg.seq_len)
        if seq is None:
            raise RuntimeError("没有可用样本")
        Feat, hist_idx, M, y, meta = seq
        qi = meta["qi"].to_numpy()
        if verbose:
            naive_gb = len(y) * cfg.seq_len * len(feats) * 4 / 1e9
            used_gb = (Feat.nbytes + hist_idx.nbytes + M.nbytes) / 1e9
            print(f"  [lstm] 序列 {len(y):,} 条 x T={cfg.seq_len} x F={Feat.shape[1]}"
                  f"  内存 {used_gb:.2f} GB（惰性索引；物化整份需 {naive_gb:.1f} GB）")

    if cfg.model == "panel_lstm":
        N = cfg.max_rank or int(panel["rank"].max())
        Feat, hist, pmask, ylab, pmeta, fq = _build_panel_tensor(panel, feats, cfg.seq_len, N)
        sqi = fq["qi"].to_numpy()
        if verbose:
            naive = hist.shape[0] * cfg.seq_len * N * len(feats) * 4 / 1e9
            used = (Feat.nbytes + hist.nbytes + pmask.nbytes) / 1e9
            print(f"  [panel_lstm] {hist.shape[0]:,} 个 fund-quarter 样本 x T={cfg.seq_len} "
                  f"x N={N} x F={len(feats)}  内存 {used:.2f} GB（物化整份需 {naive:.1f} GB）")
            pad = 1.0 - float(pmask[:, -1, :].mean())
            print(f"  [panel_lstm] 最后一步的 padding 占比 = {pad:.1%}"
                  f"（这就是把 precision 垫高的那部分）")

    for c in range(cfg.window_q, int(panel.qi.max()) + 2, cfg.step):
        if cfg.model == "panel_lstm":
            tr = (sqi >= c - cfg.window_q) & (sqi < c - cfg.test_q)
            te = (sqi >= c - cfg.test_q) & (sqi < c)
            if tr.sum() < 50 or te.sum() == 0:
                continue
            got, dev = _fit_panel_lstm(Feat, hist, pmask, ylab, tr, te, cfg)
            if got is None:
                continue
            te_i, Pmat = got
            sid2pos = {s: i for i, s in enumerate(te_i)}
            sel = pmeta[pmeta["_sid"].isin(sid2pos)].copy()
            sel["y_pred"] = Pmat[[sid2pos[s] for s in sel["_sid"]],
                                 sel["_slot"].to_numpy()]
            p = sel[sel["Y"].notna()].drop(columns=["_sid", "_slot"]).reset_index(drop=True)
            if len(p) == 0:
                continue
        elif cfg.model == "lstm":
            tr = (qi >= c - cfg.window_q) & (qi < c - cfg.test_q)
            te = (qi >= c - cfg.test_q) & (qi < c)
            if tr.sum() < 5000 or te.sum() == 0:
                continue
            yp, dev = _fit_lstm(Feat, hist_idx, M, y, tr, te, cfg)
            if yp is None:
                continue
            p = meta.iloc[np.where(te)[0]].copy()
            p["y_pred"] = yp
        else:
            tr = d[(d.qi >= c - cfg.window_q) & (d.qi < c - cfg.test_q)]
            te = d[(d.qi >= c - cfg.test_q) & (d.qi < c)]
            if len(tr) < 5000 or len(te) == 0:
                continue
            from sklearn.ensemble import HistGradientBoostingClassifier
            trs = (tr.sample(cfg.n_max_train, random_state=cfg.seed)
                   if len(tr) > cfg.n_max_train else tr)
            clf = HistGradientBoostingClassifier(max_iter=cfg.max_iter,
                                                 learning_rate=cfg.learning_rate,
                                                 max_depth=cfg.max_depth,
                                                 random_state=cfg.seed)
            clf.fit(trs[feats].to_numpy("float32"), trs["Y"].to_numpy())
            p = te[[k for k in _KEEP if k in te.columns]].copy()
            p["y_pred"] = clf.predict(te[feats].to_numpy("float32"))
            dev = "cpu"
        out.append(p)
        if verbose:
            print(f"  窗口 {c:>3}  n_test={len(p):>9,}  "
                  f"acc={float((p.y_pred==p.Y).mean()):.4f}  [{cfg.model}/{dev}]")
    return pd.concat(out, ignore_index=True)


# ============================================================ 1. 精度 (+padding)
def precision_report(P: pd.DataFrame, cfg: Config):
    y, pr = P["Y"].to_numpy(), P["y_pred"].to_numpy()
    acc = float((pr == y).mean())
    hold = float((y == 0).mean())
    naive_pool = float(pd.Series(y).value_counts(normalize=True).max())
    pf = P.assign(c=(pr == y)).groupby("fund", observed=True)["c"].mean()
    rows = [{"口径": "真实持仓", "N": "-", "padding": 0.0,
             "precision": acc, "naive": naive_pool}]
    fq = P.drop_duplicates(["fund", "qi"])[["fund", "qi", "n_holdings"]]
    for N in (50, cfg.template_N, 100):
        slots = np.minimum(fq["n_holdings"].to_numpy(), N)
        p = float((N - slots).sum() / (N * len(fq)))
        rows.append({"口径": f"N={N} 模板(含padding)", "N": N, "padding": p,
                     "precision": p + (1 - p) * acc, "naive": p + (1 - p) * hold})
    t = pd.DataFrame(rows)
    return t, {"accuracy": acc, "naive_pooled": naive_pool,
               "precision_per_fund": float(pf.mean()), "hold_share": hold}


# ============================================================ 2. Table X
_TIMING = {"contemporaneous": 0, "predictive": 1, "tradeable": 2}


def table_x(P: pd.DataFrame, cfg: Config, timing="tradeable", hmax=4):
    """基金按 precision 分五组 -> CRET_{0,1..hmax}。start = 时间口径决定的滞后。"""
    start = _TIMING[timing]
    P = P.copy()
    P["correct"] = (P.y_pred == P.Y).astype(float)
    P["wc"] = P["w_lag1"] * P["fwd_1q"]
    ok = P["fwd_1q"].notna() & P["w_lag1"].notna()
    fq = P[ok].groupby(["fund", "qi"], observed=True).agg(
        wsum=("w_lag1", "sum"), wc=("wc", "sum")).reset_index()
    fq["fund_ret"] = fq["wc"] / fq["wsum"].where(fq["wsum"] > 0)
    fq = fq.merge(P.groupby(["fund", "qi"], observed=True)["correct"].mean().rename("prec"),
                  on=["fund", "qi"], how="left")
    fq["abn"] = fq["fund_ret"] - fq.groupby("qi")["fund_ret"].transform("mean")
    lut = fq.set_index(["fund", "qi"])["abn"]
    for h in range(1, hmax + 1):
        tot = np.zeros(len(fq)); good = np.ones(len(fq), bool)
        for j in range(h):
            i2 = pd.MultiIndex.from_arrays([fq["fund"].to_numpy(),
                                            fq["qi"].to_numpy() + start + j])
            a = lut.reindex(i2).to_numpy()
            m = pd.isna(a); good &= ~m; tot = tot + np.where(m, 0.0, a)
        fq[f"cabn{h}"] = np.where(good, tot, np.nan)
    fq["Q"] = fq.groupby("qi")["prec"].transform(_q5)
    fq = fq.dropna(subset=["Q"])
    rows = []
    for q in (1, 2, 3, 4, 5):
        r = {"quintile": f"Q{q}"}
        for h in range(1, hmax + 1):
            s = fq[fq.Q == q].groupby("qi")[f"cabn{h}"].mean()
            r[f"CRET_0_{h}"] = s.mean() * 100; r[f"t{h}"] = _t(s)
        rows.append(r)
    r = {"quintile": "Q5-Q1"}
    for h in range(1, hmax + 1):
        d = (fq[fq.Q == 5].groupby("qi")[f"cabn{h}"].mean()
             - fq[fq.Q == 1].groupby("qi")[f"cabn{h}"].mean()).dropna()
        r[f"CRET_0_{h}"] = d.mean() * 100; r[f"t{h}"] = _t(d)
    rows.append(r)
    return pd.DataFrame(rows)


# ============================================================ 3. Table XII
def table_xii(P: pd.DataFrame, cfg: Config, timing="tradeable"):
    col = {"contemporaneous": "fwd_1q", "predictive": "fwd_2q", "tradeable": "fwd_3q"}[timing]
    if col not in P.columns or P[col].notna().sum() == 0:
        return pd.DataFrame([{"quintile": "n/a", "mean_qret": np.nan, "t": np.nan}])
    P = P.copy()
    P["correct"] = (P.y_pred == P.Y).astype(float)
    stk = P.groupby(["security", "qi"], observed=True).agg(
        acc=("correct", "mean"), fwd=(col, "mean")).reset_index().dropna(subset=["fwd"])
    stk["Q"] = stk.groupby("qi")["acc"].transform(_q5)
    stk = stk.dropna(subset=["Q"])
    per = stk.groupby(["qi", "Q"])["fwd"].mean().unstack()
    rows = [{"quintile": f"Q{int(q)}", "mean_qret": per[q].mean() * 100, "t": _t(per[q])}
            for q in sorted(per.columns)]
    ls = (per[per.columns.min()] - per[per.columns.max()]).dropna()
    rows.append({"quintile": "Q1-Q5", "mean_qret": ls.mean() * 100, "t": _t(ls)})
    return pd.DataFrame(rows)


# ============================================================ 单个配置
def run_config(panel: pd.DataFrame, cfg: Config, tag: str = "", verbose=True) -> dict:
    """跑一个配置，返回该配置的全部结果（精度 + Table X/XII x 三种口径）。
    面板从外面传进来，多个配置之间只建一次。"""
    print(f"{'='*74}\n{tag or cfg.model}  |  model={cfg.model}  "
          f"manager_memory={cfg.use_manager_memory}  ({len(cfg.features)} 特征)\n{'='*74}")
    P = run_model(panel, cfg, verbose=verbose)
    prec_tbl, prec_m = precision_report(P, cfg)
    r = {"tag": tag, "cfg": cfg, "preds": P,
         "precision_table": prec_tbl, "precision": prec_m,
         "tableX": {t: table_x(P, cfg, t) for t in _TIMING},
         "tableXII": {t: table_xii(P, cfg, t) for t in _TIMING}}
    tx = r["tableX"]["tradeable"].iloc[-1]
    t12 = r["tableXII"]["contemporaneous"].iloc[-1]
    print(f"\n  accuracy(真实持仓) = {prec_m['accuracy']:.4f}   "
          f"naive = {prec_m['naive_pooled']:.4f}")
    print(f"  含 padding (N={cfg.template_N}) precision = "
          f"{prec_tbl.iloc[2]['precision']:.4f}   naive = {prec_tbl.iloc[2]['naive']:.4f}"
          f"   [论文 0.71 / 0.52]")
    print(f"  Table X  Q5-Q1 (tradeable)      = {tx.CRET_0_4:+.3f}% (t={tx.t4:+.2f})"
          f"   [论文 -0.79, t=-3.05]")
    print(f"  Table XII Q1-Q5 (contemporaneous)= {t12.mean_qret:+.3f}% (t={t12.t:+.2f})"
          f"   [论文 +1.06, t=5.74]")
    return r


def free(results: dict = None, keep_tables=True):
    """跑完一个配置后清缓存，避免多实验串跑时 OOM。

    每个配置的 preds 在千万行面板上可能几百 MB；连着跑四个配置就会堆起来。
    keep_tables=True 只丢掉 preds（大头），保留所有结果表。
    """
    import gc
    if results:
        for tag, r in results.items():
            if isinstance(r, dict) and keep_tables and "preds" in r:
                r["preds"] = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    except Exception:
        pass
    try:                      # 打印当前进程占用，便于判断还能不能再跑一个
        import resource
        mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if os.name != "posix" else 1024)
        print(f"[free] 已清理。进程峰值内存 ≈ {mb:.0f} MB")
    except Exception:
        print("[free] 已清理。")


def summarize(results: dict) -> pd.DataFrame:
    """把若干个 run_config 的结果并成一张对照表。"""
    rows = []
    for tag, r in results.items():
        if not isinstance(r, dict) or "tableX" not in r:
            continue
        c = r["cfg"]; row = {"配置": tag, "模型": c.model,
                             "manager_memory": c.use_manager_memory,
                             "accuracy": round(r["precision"]["accuracy"], 4),
                             "含padding_precision": round(r["precision_table"].iloc[2]["precision"], 4)}
        for tm in _TIMING:
            x = r["tableX"][tm].iloc[-1]
            row[f"X_Q5-Q1_{tm[:5]}"] = round(x.CRET_0_4, 3)
            row[f"X_t_{tm[:5]}"] = round(x.t4, 2)
        for tm in _TIMING:
            x = r["tableXII"][tm].iloc[-1]
            row[f"XII_Q1-Q5_{tm[:5]}"] = round(x.mean_qret, 3)
            row[f"XII_t_{tm[:5]}"] = round(x.t, 2)
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["X与论文同号"] = np.where(out["X_Q5-Q1_trade"] < 0, "是 ✓", "否 ✗")
    return out


# ============================================================ 一键对照实验
def run_ablation(cfg: Config = None, verbose=True):
    """跑 use_manager_memory = False / True 两版，返回全部结果供对比。"""
    cfg = cfg or Config()
    out = {}
    panel = load_and_prepare(cfg)          # 面板与开关无关，只建一次
    for use_mem in (False, True):
        c = Config(**{**cfg.__dict__, "use_manager_memory": use_mem})
        tag = "with_memory" if use_mem else "no_memory"
        print(f"\n{'='*72}\n{tag}  ({len(c.features)} 个特征)\n{'='*72}")
        P = run_model(panel, c, verbose=verbose)
        prec_tbl, prec_m = precision_report(P, c)
        out[tag] = {
            "preds": P, "precision_table": prec_tbl, "precision": prec_m,
            "tableX": {t: table_x(P, c, t) for t in _TIMING},
            "tableXII": {t: table_xii(P, c, t) for t in _TIMING},
        }
        print(f"  accuracy={prec_m['accuracy']:.4f}  naive={prec_m['naive_pooled']:.4f}")
    out["_panel"] = panel
    return out


def show(res: dict):
    """打印对照总表。"""
    print("\n" + "=" * 78)
    print("1) 精度：真实持仓 vs 计入 padding（论文 0.71 / naive 0.52）")
    print("=" * 78)
    for tag in ("no_memory", "with_memory"):
        print(f"\n--- {tag} ---")
        t = res[tag]["precision_table"].copy()
        t["padding"] = (t["padding"] * 100).round(1)
        print(t.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("2) TABLE X（CRET_0,4，%）  论文: Q1 +0.36 Q5 -0.42 Q5-Q1 -0.79 (t=-3.05)")
    print("=" * 78)
    for tag in ("no_memory", "with_memory"):
        print(f"\n--- {tag} ---")
        for tm in ("tradeable", "predictive", "contemporaneous"):
            r = res[tag]["tableX"][tm]
            v = "  ".join(f"{x:+6.3f}" for x in r["CRET_0_4"][:5])
            sp = r.iloc[-1]
            print(f"  {tm:<16} {v}   Q5-Q1 {sp.CRET_0_4:+6.3f} (t={sp.t4:+5.2f})")

    print("\n" + "=" * 78)
    print("3) TABLE XII（Q1-Q5，%/季）  论文: +1.06 (t=5.74)")
    print("=" * 78)
    for tag in ("no_memory", "with_memory"):
        print(f"\n--- {tag} ---")
        for tm in ("tradeable", "predictive", "contemporaneous"):
            sp = res[tag]["tableXII"][tm].iloc[-1]
            print(f"  {tm:<16} Q1-Q5 {sp.mean_qret:+6.3f} (t={sp.t:+5.2f})")

    print("\n" + "=" * 78)
    print("解读：no_memory 是论文的复现口径；with_memory 精度更高但 Table X 会反号，")
    print("      因为 fs_hold_rate 把『可预测』变成了『这个经理不动这个仓位』。")
    print("=" * 78)


if __name__ == "__main__":
    res = run_ablation(Config())
    show(res)
