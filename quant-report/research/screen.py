"""Exploratory, single-day factor diagnostics. No strategy return backtest."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from l2sim.data import parse_time
from .pipeline import HORIZONS, dump

FACTORS = {
    "obi1": "一档挂单量不平衡", "obi5": "五档挂单量不平衡", "obi10": "十档挂单量不平衡",
    "weighted_obi": "按档位距离加权的挂单不平衡", "weighted_mid_bias": "挂单量加权中点偏移",
    "ladder_gap_bias": "十档价格空隙不对称", "concentration_bias": "最优档挂单集中度差",
    "trade_imb_5": "近5秒主动成交量不平衡", "trade_imb_30": "近30秒主动成交量不平衡",
    "trade_imb_60": "近60秒主动成交量不平衡", "trade_count_imb": "近30秒主动成交笔数不平衡",
    "large_trade_imb": "近30秒大额逐笔成交不平衡", "new_order_imb": "近30秒新增剩余挂单不平衡",
    "cancel_pressure": "近30秒卖侧减买侧撤单压力", "fast_cancel_pressure": "1秒内撤单方向差",
    "net_resting_flow": "近30秒净挂单流方向", "ofi_5": "近5秒最优档订单流不平衡",
    "ofi_30": "近30秒最优档订单流不平衡", "ofi_60": "近60秒最优档订单流不平衡",
    "momentum_5": "近5秒中间价动量", "momentum_30": "近30秒中间价动量",
    "momentum_60": "近60秒中间价动量", "vwap_gap": "中间价相对近30秒成交均价偏离",
    "peer_momentum_30": "其余样本股票近30秒等权动量", "relative_momentum_30": "相对其余股票的近30秒动量",
}
RISKS = {"spread_bps": "买卖价差", "log_depth10": "十档深度对数", "event_rate": "每秒事件数",
         "realized_vol_60": "近60秒已实现波动", "mean_trade_value": "平均逐笔成交金额",
         "cancel_share": "撤单量占新增和撤单总量比例", "quote_age_ms": "最优价量未更新时长"}


def study_rows(panel):
    # Fixed windows leave warm-up and a full 300s label inside each session.
    am = panel.time_ms.between(parse_time("09:35:00"), parse_time("11:25:00"), inclusive="left")
    pm = panel.time_ms.between(parse_time("13:05:00"), parse_time("14:50:00"), inclusive="left")
    return panel.loc[am | pm].copy()


def per_stock_correlations(frame, factors, targets, split):
    rows = []
    for symbol, x in frame.groupby("symbol"):
        c = x[list(factors) + targets].corr(method="spearman", min_periods=60)
        for factor in factors:
            for target in targets:
                rows.append({"split": split, "symbol": symbol, "factor": factor, "target": target,
                             "ic": c.loc[factor, target], "n": int(x[[factor, target]].dropna().shape[0])})
    return pd.DataFrame(rows)


def cross_ic(frame, factor, target):
    x = frame[["time_ms", factor, target]].dropna()
    ranks = x.groupby("time_ms")[[factor, target]].rank(method="average")
    ranks["time_ms"] = x.time_ms
    ranks["xy"] = ranks[factor] * ranks[target]
    ranks["xx"] = ranks[factor] ** 2
    ranks["yy"] = ranks[target] ** 2
    g = ranks.groupby("time_ms")
    means = g.mean()
    cov = means.xy - means[factor] * means[target]
    denom = np.sqrt((means.xx - means[factor] ** 2).clip(lower=0) * (means.yy - means[target] ** 2).clip(lower=0))
    result = cov / denom.replace(0, np.nan)
    result[g.size() < 20] = np.nan
    return result


def training_parameters(am, stock_ic):
    rank = stock_ic[(stock_ic.split == "AM") & (stock_ic.target == "ret_30s")].groupby("factor").ic.mean()
    ordered = rank.abs().sort_values(ascending=False).index.tolist()
    signs = {f: 1 if rank[f] >= 0 else -1 for f in rank.index}
    thresholds = am.groupby("symbol")[list(FACTORS)].quantile([.2, .8])
    return rank, ordered, signs, thresholds


def tail_direction(frame, factor, thresholds, sign):
    lo = frame.symbol.map(thresholds.xs(.2, level=1)[factor]).to_numpy()
    hi = frame.symbol.map(thresholds.xs(.8, level=1)[factor]).to_numpy()
    value = frame[factor].to_numpy()
    # Strict thresholds prevent huge zero/tie populations being called strong signals.
    return np.where(value > hi, sign, np.where(value < lo, -sign, 0))


def tail_metrics(frame, factor, sign, thresholds, split):
    direction = tail_direction(frame, factor, thresholds, sign)
    rows = []
    for h in HORIZONS:
        ret = frame[f"ret_{h}s"].to_numpy()
        quote = np.where(direction > 0,
                         np.log(frame[f"bid_fwd_{h}"] / frame.ask_cent) * 10000,
                         np.log(frame.bid_cent / frame[f"ask_fwd_{h}"]) * 10000)
        valid = (direction != 0) & np.isfinite(ret) & np.isfinite(quote)
        signed = direction[valid] * ret[valid]
        row = {"split": split, "factor": factor, "horizon": h, "n": int(valid.sum()),
               "bullish_n": int(((direction > 0) & valid).sum()), "bearish_n": int(((direction < 0) & valid).sum()),
               "signed_mid_bps": float(np.mean(signed)), "quoted_roundtrip_bps": float(np.mean(quote[valid])),
               "mid_positive_fraction": float(np.mean(signed > 0)), "mid_zero_fraction": float(np.mean(signed == 0))}
        if h == 30:
            for delay in [1, 5]:
                delayed = frame[f"delayed_{delay}s_30"].to_numpy()
                mask = (direction != 0) & np.isfinite(delayed)
                row[f"delayed_{delay}s_mid_bps"] = float(np.mean(direction[mask] * delayed[mask]))
        rows.append(row)
    return rows


def safe_records(frame):
    return json.loads(frame.to_json(orient="records"))


def analyze(panel, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x = study_rows(panel)
    targets = [f"ret_{h}s" for h in HORIZONS]
    print(f"SCREEN: {len(x):,} rows, {len(FACTORS)} directional factors", flush=True)
    stock = pd.concat([per_stock_correlations(z, FACTORS, targets, split) for split, z in x.groupby("session")], ignore_index=True)
    am = x[x.session == "AM"]
    rank, ordered, signs, thresholds = training_parameters(am, stock)
    selected = ordered[:8]
    stock["oriented_ic"] = stock.ic * stock.factor.map(signs)
    stock.to_parquet(output / "per_stock_ic.parquet", index=False)
    metrics = stock.groupby(["split", "factor", "target"]).agg(mean_ic=("ic", "mean"), median_ic=("ic", "median"),
                    oriented_ic=("oriented_ic", "mean"), positive_stock_fraction=("oriented_ic", lambda s: (s.dropna() > 0).mean()),
                    valid_stocks=("ic", "count"), observations=("n", "sum")).reset_index()
    metrics["horizon"] = metrics.target.str.extract(r"(\d+)").astype(int)
    metrics.to_parquet(output / "factor_horizon_metrics.parquet", index=False)
    print("SCREEN: direction/horizon correlations complete", flush=True)
    tail_rows, cross_rows, block_rows = [], [], []
    for split, z in x.groupby("session"):
        for f in FACTORS:
            tail_rows.extend(tail_metrics(z, f, signs[f], thresholds, split))
        for f in selected:
            for h in HORIZONS:
                ic = cross_ic(z, f, f"ret_{h}s") * signs[f]
                blocks = ic.groupby(ic.index // 300000).mean()
                cross_rows.append({"split": split, "factor": f, "horizon": h, "cross_section_ic": ic.mean(),
                                   "valid_times": int(ic.count()), "positive_5min_block_fraction": (blocks.dropna() > 0).mean()})
                for block, value in blocks.items():
                    block_rows.append({"split": split, "factor": f, "horizon": h, "block": block, "ic": value})
    tails = pd.DataFrame(tail_rows)
    tails.to_parquet(output / "tail_markouts.parquet", index=False)
    increments = []
    for f in selected:
        z = tails[(tails.split == "PM") & (tails.factor == f)].set_index("horizon")
        for left, right in [(1, 5), (5, 10), (10, 30), (30, 60), (60, 120), (120, 300)]:
            increments.append({"factor": f, "start_second": left, "end_second": right,
                               "signed_mid_increment_bps": z.loc[right, "signed_mid_bps"] - z.loc[left, "signed_mid_bps"]})
    pd.DataFrame(increments).to_parquet(output / "incremental_markouts.parquet", index=False)
    pd.DataFrame(cross_rows).to_parquet(output / "cross_section_ic.parquet", index=False)
    pd.DataFrame(block_rows).to_parquet(output / "time_block_ic.parquet", index=False)
    thresholds.reset_index().to_parquet(output / "am_only_thresholds.parquet", index=False)
    # Probe risk forecasting separately: direction-free future absolute price change.
    for h in [30, 60, 300]: x[f"abs_{h}s"] = x[f"ret_{h}s"].abs()
    risk = pd.concat([per_stock_correlations(z, RISKS, ["abs_30s", "abs_60s", "abs_300s"], split) for split, z in x.groupby("session")], ignore_index=True)
    risk_summary = risk.groupby(["split", "factor", "target"]).ic.mean().unstack("split").reset_index()
    risk_summary.to_parquet(output / "risk_metrics.parquet", index=False)
    # Rank-correlation redundancy, normalized inside each symbol using AM only.
    ranked_am = am.groupby("symbol")[list(FACTORS)].rank(pct=True)
    redundancy = ranked_am.corr()
    redundancy.to_parquet(output / "am_factor_redundancy.parquet")
    # Less-overlapping labels: diagnostic only, thresholds/signs remain AM-selected.
    sparse_rows = []
    pm = x[x.session == "PM"]
    for h in [30, 60, 300]:
        sparse = pm[(pm.time_ms - parse_time("13:05:00")) % (h * 1000) == 0]
        c = per_stock_correlations(sparse, selected, [f"ret_{h}s"], "PM_sparse") if h <= 60 else None
        # 21 observations per stock at 300s: report tail markout, no tiny-sample IC.
        for f in selected:
            row = next(r for r in tail_metrics(sparse, f, signs[f], thresholds, "PM_sparse") if r["horizon"] == h)
            if c is not None: row["mean_ic"] = c[c.factor == f].ic.mean() * signs[f]
            sparse_rows.append(row)
    pd.DataFrame(sparse_rows).to_parquet(output / "nonoverlap_diagnostics.parquet", index=False)
    subgroup_rows = []
    spread_median = am.groupby("symbol").spread_bps.median()
    for group, symbols in {"narrow_spread": spread_median[spread_median <= spread_median.median()].index,
                           "wide_spread": spread_median[spread_median > spread_median.median()].index,
                           "STAR": [s for s in pm.symbol.unique() if str(s).startswith("688")],
                           "main_board": [s for s in pm.symbol.unique() if not str(s).startswith("688")]}.items():
        for f in selected:
            rows = stock[(stock.split == "PM") & (stock.factor == f) & (stock.target == "ret_30s") & stock.symbol.isin(symbols)]
            subgroup_rows.append({"group": group, "factor": f, "stocks": len(rows), "oriented_ic": rows.oriented_ic.mean()})
    pd.DataFrame(subgroup_rows).to_parquet(output / "subgroup_diagnostics.parquet", index=False)
    signal_summary = []
    for f in ordered:
        def metric(split, h, col):
            return float(metrics[(metrics.split == split) & (metrics.factor == f) & (metrics.horizon == h)][col].iloc[0])
        t30 = tails[(tails.split == "PM") & (tails.factor == f) & (tails.horizon == 30)].iloc[0]
        curve = metrics[(metrics.split == "PM") & (metrics.factor == f)].sort_values("horizon")
        peak = curve.loc[curve.oriented_ic.idxmax()]
        signal_summary.append({"factor": f, "description": FACTORS[f], "am_sign": signs[f], "selected_in_am_top8": f in selected,
                               "am_ic_30": metric("AM", 30, "oriented_ic"), "pm_ic_30": metric("PM", 30, "oriented_ic"),
                               "pm_positive_stock_fraction": metric("PM", 30, "positive_stock_fraction"),
                               "pm_mid_30_bps": t30.signed_mid_bps, "pm_quote_30_bps": t30.quoted_roundtrip_bps,
                               "pm_delay1_30_bps": t30.delayed_1s_mid_bps, "pm_delay5_30_bps": t30.delayed_5s_mid_bps,
                               "pm_peak_horizon_exploratory": int(peak.horizon), "pm_peak_ic": peak.oriented_ic})
    summary = {"rows": len(x), "am_rows": len(am), "pm_rows": len(pm), "stocks": int(x.symbol.nunique()),
               "directional_factors": len(FACTORS), "risk_factors": len(RISKS), "am_selected_top8": selected,
               "median_spread_bps": float(x.spread_bps.median()), "mean_spread_bps": float(x.spread_bps.mean()),
               "signals": signal_summary, "risk": safe_records(risk_summary), "cross_section": safe_records(pd.DataFrame(cross_rows)),
               "selection": "AM per-stock mean Spearman IC at 30s; signs and stockwise 20/80% cutoffs fixed in AM; PM is a same-day regime holdout, not independent day validation",
               "execution": "Quote markout includes spread only; no fees, market impact, inventory, queue or T+1 feasibility; overlapping signals are not trades"}
    dump(output / "summary.json", summary)
    dump(output / "factor_registry.json", {"directional": FACTORS, "risk": RISKS,
                                           "units": "Price in cents, returns in basis points; 1bp=0.01%"})
    plot_results(metrics, tails, selected, ordered, output)
    from .report import write_report
    write_report(summary, output)
    print(json.dumps({"top8": selected, "samples": len(x), "summary": str(output / "summary.json")}), flush=True)
    return summary


def plot_results(metrics, tails, selected, ordered, output):
    pm = metrics[metrics.split == "PM"].pivot(index="factor", columns="horizon", values="oriented_ic").loc[ordered, HORIZONS]
    fig, ax = plt.subplots(figsize=(10, 10))
    limit = max(.05, np.nanmax(np.abs(pm.to_numpy())))
    im = ax.imshow(pm, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(HORIZONS)), [str(h) for h in HORIZONS])
    ax.set_yticks(range(len(ordered)), ordered, fontsize=9)
    ax.set_xlabel("Forward horizon (seconds)")
    ax.set_title("Afternoon mean per-stock Rank IC\nDirection and row order fixed using morning 30s IC")
    for i in range(len(ordered)):
        for j in range(len(HORIZONS)): ax.text(j, i, f"{pm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Rank IC")
    fig.tight_layout(); fig.savefig(output / "factor_horizon_heatmap.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    for ax, f in zip(axes.flat, selected):
        for split, style in [("AM", "-o"), ("PM", "--s")]:
            s = metrics[(metrics.factor == f) & (metrics.split == split)].sort_values("horizon")
            ax.plot(s.horizon, s.oriented_ic, style, label=split, markersize=3)
        ax.axhline(0, color="grey", linewidth=.6); ax.set_xscale("log"); ax.set_title(f, fontsize=10)
        ax.set_xticks(HORIZONS, [str(h) for h in HORIZONS], fontsize=8)
        ax.set_xlabel("Seconds"); ax.set_ylabel("Rank IC"); ax.legend(fontsize=8)
    fig.suptitle("AM-selected candidates: cumulative horizon curves (not signal half-lives)")
    fig.tight_layout(); fig.savefig(output / "selected_horizon_curves.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    z = tails[(tails.split == "PM") & (tails.horizon == 30)].set_index("factor").loc[selected]
    y = np.arange(len(selected))
    ax.barh(y-.18, z.signed_mid_bps, .35, label="Signed mid-price change")
    ax.barh(y+.18, z.quoted_roundtrip_bps, .35, label="Quote-to-quote markout, before fees")
    ax.set_yticks(y, selected); ax.axvline(0, color="black", linewidth=.7)
    ax.set_xlabel("Basis points per tail observation; overlapping observations are not trades")
    ax.set_title("Afternoon 30s signal vs spread cost, AM-fixed tail thresholds", pad=36)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.13), ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(output / "spread_cost_comparison.png", dpi=160); plt.close(fig)
