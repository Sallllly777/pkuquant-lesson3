"""Single-day execution experiments and AM-fitted book/flow combinations.

All scenarios are small-order, independent counterfactual tasks, not portfolio PnL.
The existing Replay engine is reused without changing matching semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from l2sim.data import ArchiveData, cents, parse_time
from l2sim.engine import Replay
from .pipeline import CLEAN, ROOT, ResearchBook, dump
from .screen import study_rows, per_stock_correlations, cross_ic, tail_direction

OUT = ROOT / "reports" / "execution_research_20260923"
INPUTS = ["obi1", "trade_imb_5", "weighted_obi", "ofi_5"]
MODELS = {"book": ["obi1"], "flow": ["trade_imb_5"],
          "book_flow": ["obi1", "trade_imb_5"],
          "book_flow_depth": ["obi1", "trade_imb_5", "weighted_obi"],
          "book_flow_depth_ofi": INPUTS}


def fit_scores(panel):
    x = study_rows(panel)
    am = x[x.session == "AM"]
    means = am.groupby("symbol")[INPUTS].mean()
    stds = am.groupby("symbol")[INPUTS].std().clip(lower=1e-8)
    target_mean = am.groupby("symbol").ret_30s.mean()
    target_std = am.groupby("symbol").ret_30s.std().clip(lower=1e-8)
    z = pd.DataFrame(index=x.index)
    for f in INPUTS:
        z[f] = ((x[f] - x.symbol.map(means[f])) / x.symbol.map(stds[f])).clip(-5, 5)
    train = x.session == "AM"
    y = ((x.ret_30s - x.symbol.map(target_mean)) / x.symbol.map(target_std))[train].to_numpy()
    coefficients = {}
    for name, features in MODELS.items():
        a = z.loc[train, features].to_numpy()
        # Fixed ridge penalty, no PM parameter selection or hyperparameter search.
        weights = np.linalg.solve(a.T @ a / len(a) + .001 * np.eye(a.shape[1]), a.T @ y / len(a))
        x["score_" + name] = z[features].to_numpy() @ weights
        coefficients[name] = dict(zip(features, weights.tolist()))
    dump(OUT / "am_model_parameters.json", {"target": "AM stockwise standardized future30s mid log return",
         "regularization": .001, "coefficients": coefficients,
         "feature_mean": means.to_dict(), "feature_std": stds.to_dict(),
         "target_mean": target_mean.to_dict(), "target_std": target_std.to_dict(),
         "note": "All feature scales and model coefficients fit on AM only. PM already explored previously; not a fresh holdout."})
    x.to_parquet(OUT / "scored_panel.parquet", index=False)
    return x


def combination_analysis(x):
    scores = ["score_" + name for name in MODELS]
    targets = ["ret_5s", "ret_10s", "ret_30s", "ret_60s", "ret_120s"]
    stock = pd.concat([per_stock_correlations(z, scores, targets, split) for split, z in x.groupby("session")], ignore_index=True)
    stock.to_parquet(OUT / "combination_stock_ic.parquet", index=False)
    metrics = stock.groupby(["split", "factor", "target"]).ic.mean().reset_index()
    metrics.to_parquet(OUT / "combination_ic.parquet", index=False)
    cross = []
    for split, z in x.groupby("session"):
        for f in scores:
            ic = cross_ic(z, f, "ret_30s")
            cross.append({"split": split, "model": f, "cross_ic_30": ic.mean()})
    pd.DataFrame(cross).to_parquet(OUT / "combination_cross_ic.parquet", index=False)
    # Mechanistic cells fixed at +/-0.5: a >3:1 side-volume ratio.
    cells = x[x.session == "PM"].copy()
    for f, name in [("obi1", "book_state"), ("trade_imb_5", "flow_state")]:
        cells[name] = np.select([cells[f] < -.5, cells[f] > .5], ["negative", "positive"], default="neutral")
    cell_stats = cells.groupby(["book_state", "flow_state"]).agg(n=("symbol", "size"), stocks=("symbol", "nunique"),
              mid_5_bps=("ret_5s", "mean"), mid_30_bps=("ret_30s", "mean"), mid_60_bps=("ret_60s", "mean"),
              spread_bps=("spread_bps", "mean"), up_30_fraction=("ret_30s", lambda s: (s > 0).mean()),
              flat_30_fraction=("ret_30s", lambda s: (s == 0).mean())).reset_index()
    stock_equal = cells.groupby(["book_state", "flow_state", "symbol"]).ret_30s.mean().groupby(level=[0, 1]).mean()
    cell_stats = cell_stats.merge(stock_equal.rename("stock_equal_mid_30_bps").reset_index(), on=["book_state", "flow_state"])
    cell_stats.to_parquet(OUT / "book_flow_cells.parquet", index=False)
    # Compare same nominal tail budget with thresholds fixed in AM; ties remain neutral.
    am = x[x.session == "AM"]
    thresholds = am.groupby("symbol")[scores + ["obi1", "trade_imb_5"]].quantile([.2, .8])
    latency = []
    pm = x[x.session == "PM"]
    for f in scores + ["obi1", "trade_imb_5"]:
        direction = tail_direction(pm, f, thresholds, 1)
        keep = direction != 0
        for delay in [0, 1, 5]:
            entry_b = pm.bid_cent if delay == 0 else pm[f"bid_fwd_{delay}"]
            entry_a = pm.ask_cent if delay == 0 else pm[f"ask_fwd_{delay}"]
            end = 30 + delay
            exit_b, exit_a = pm[f"bid_fwd_{end}"], pm[f"ask_fwd_{end}"]
            mid = direction * np.log(((exit_b + exit_a) / 2) / ((entry_b + entry_a) / 2)) * 10000
            quote = np.where(direction > 0, np.log(exit_b / entry_a), np.log(entry_b / exit_a)) * 10000
            valid = keep & np.isfinite(mid) & np.isfinite(quote)
            fixed_mid = direction * np.log(((pm.bid_fwd_30 + pm.ask_fwd_30) / 2) / ((entry_b + entry_a) / 2)) * 10000
            fixed_quote = np.where(direction > 0, np.log(pm.bid_fwd_30 / entry_a), np.log(entry_b / pm.ask_fwd_30)) * 10000
            entry_slippage = np.where(direction > 0, np.log(entry_a / pm.ask_cent), np.log(pm.bid_cent / entry_b)) * 10000
            latency.append({"factor": f, "delay_seconds": delay, "hold_seconds": 30, "n": int(valid.sum()),
                            "mid_bps": np.mean(mid[valid]), "quote_bps": np.mean(quote[valid]),
                            "spread_drag_bps": np.mean(mid[valid] - quote[valid]),
                            "fixed_exit30_mid_bps": np.mean(fixed_mid[valid]),
                            "fixed_exit30_quote_bps": np.mean(fixed_quote[valid]),
                            "entry_slippage_bps": np.mean(entry_slippage[valid]),
                            "net_with_1bp_per_leg": np.mean(quote[valid]) - 2,
                            "net_with_3bp_per_leg": np.mean(quote[valid]) - 6})
    pd.DataFrame(latency).to_parquet(OUT / "latency_cost_decomposition.parquet", index=False)


class ExperimentReplay(Replay):
    def __init__(self, data):
        super().__init__(data, cash=1_000_000_000, initial_position=1_000_000)
        self.book = ResearchBook()
        self.quote_times, self.bid_prices, self.ask_prices = [], [], []

    def _process(self, event):
        if event.kind == "snp":
            # Only the already-public previous close is used from raw snapshots.
            self.time = event.time
            self.cursor += 1
            self.preclose = cents(event.row["PreClose"])
            return
        super()._process(event)
        bid, ask = self.book.best("BUY") or 0, self.book.best("SELL") or 0
        if not self.quote_times or (bid, ask) != (self.bid_prices[-1], self.ask_prices[-1]):
            self.quote_times.append(event.time)
            self.bid_prices.append(bid)
            self.ask_prices.append(ask)

    def reset_task_account(self):
        if self.active(): raise ValueError("Close prior experiment orders before resetting")
        self.orders.clear(); self.fills.clear(); self.consumed.clear()
        self.cash = self.initial_cash
        self.sellable = self.initial_position
        self.bought_today = self.frozen_cash = self.frozen_shares = 0

    def mids_at(self, times):
        ix = np.searchsorted(self.quote_times, times, side="right") - 1
        b = np.asarray(self.bid_prices, dtype=float)[np.maximum(ix, 0)]
        a = np.asarray(self.ask_prices, dtype=float)[np.maximum(ix, 0)]
        valid = (ix >= 0) & (b > 0) & (a > b)
        return np.where(valid, (a + b) / 2, np.nan)


def visible_vwap(book, side, quantity, consumed=None):
    if quantity == 0: return np.nan, 0
    opposite = "SELL" if side == "BUY" else "BUY"
    reserved = {}
    for key, qty in (consumed or {}).items():
        q = book.orders.get(key)
        if q is not None and q.side == opposite:
            reserved[q.price] = reserved.get(q.price, 0) + qty
    remain, notional = quantity, 0
    for price in sorted(book.totals[opposite], reverse=opposite == "BUY"):
        available = max(0, book.totals[opposite][price] - reserved.get(price, 0))
        taken = min(remain, available)
        notional += taken * price; remain -= taken
        if not remain: break
    return notional / (quantity - remain) if quantity != remain else np.nan, quantity - remain


def valid_resubmission(symbol, side, quantity):
    if quantity == 0: return True
    if str(symbol).startswith("688"): return quantity >= 200
    return quantity % 100 == 0  # Initial old inventory has no odd-lot remainder.


def simulate_symbol(data, scored, arrival_delay_seconds=0, sessions=("PM",), state_callback=None):
    sim = ExperimentReplay(data)
    starts = scored[scored.session.isin(sessions) & (scored.time_ms % 60000 == 0)].sort_values("time_ms")
    outcomes, all_fills = [], []
    qty = 200 if str(data.symbol).startswith("688") else 100
    for signal in starts.itertuples():
        t = int(signal.time_ms)
        sim.advance_to(t)
        sim.reset_task_account()
        mid = (sim.book.best("BUY") + sim.book.best("SELL")) / 2
        if abs(mid - signal.mid_cent) > 1e-8: raise AssertionError("Replay/panel price disagreement")
        records = {}
        for side, sign in [("BUY", 1), ("SELL", -1)]:
            immediate, available = visible_vwap(sim.book, side, qty)
            if available != qty: raise ValueError("Insufficient immediate visible depth")
            best = sim.book.best(side)
            records[side] = {"symbol": data.symbol, "time_ms": t, "side": side, "sign": sign,
                  "quantity": qty, "mid_cent": mid, "immediate_cent": immediate, "limit_cent": best,
                  "arrival_delay_seconds": arrival_delay_seconds, "spread_bps": signal.spread_bps,
                  "obi1": signal.obi1, "trade_imb_5": signal.trade_imb_5, "score": signal.score_book_flow,
                  "aligned_obi": sign * signal.obi1, "aligned_flow": sign * signal.trade_imb_5,
                  "aligned_score": sign * signal.score_book_flow}
        deadline = 30 + arrival_delay_seconds
        observation_times = [10, 15, 20, 25] if state_callback else []
        for delay in sorted(set([0, 1, 5, 30, arrival_delay_seconds, deadline] + observation_times)):
            sim.advance_to(t + delay * 1000)
            if delay == arrival_delay_seconds:
                for side, r in records.items():
                    # Original signal-time limit price is preserved during transit.
                    # A delayed passive intention can become marketable at arrival.
                    result = sim.submit(side, r["limit_cent"] / 100, qty)
                    r["order_id"] = result["id"]
                    r["queue_ahead"] = result["queue_ahead"]
            for side, r in records.items():
                if delay not in [1, 5, 30]: continue
                p, available = visible_vwap(sim.book, side, qty)
                if available != qty: raise ValueError("Insufficient delayed visible depth")
                r[f"wait{delay}_cent"] = p
                o = sim.orders[r["order_id"] - 1] if "order_id" in r else None
                r[f"filled_by_{delay}"] = o.filled / qty if o else 0.
            if state_callback is not None and delay in [0, 5, 10, 15, 20, 25, 30]:
                state_callback(sim, signal, records, delay)
        for side, r in records.items():
            o = sim.orders[r["order_id"] - 1]
            r["filled_qty"] = o.filled
            r["filled_fraction"] = o.filled / qty
            r["any_fill"] = o.filled > 0
            r["full_fill"] = o.filled == qty
            r["partial_fill"] = 0 < o.filled < qty
            r["remaining_qty"] = o.remaining
            r["deadline_seconds_from_signal"] = deadline
            r["taker_filled_qty"] = sum(f["quantity"] for f in sim.fills if f["order_id"] == o.id and f["model"] == "TAKER_VISIBLE_BOOK")
            r["remainder_order_lot_valid"] = valid_resubmission(data.symbol, side, o.remaining)
            r["maker_vwap_cent"] = o.notional / o.filled if o.filled else np.nan
            if o.remaining:
                sim.cancel(o.id)
                fallback, available = visible_vwap(sim.book, side, o.remaining, sim.consumed)
                if available != o.remaining: raise ValueError("Insufficient fallback visible depth")
                r["maker_then_cross_cent"] = (o.notional + fallback * o.remaining) / qty
            else:
                r["maker_then_cross_cent"] = r["maker_vwap_cent"]
            for fill in sim.fills:
                if fill["order_id"] == o.id:
                    all_fills.append({"symbol": data.symbol, "time_ms": t, "side": side, "sign": r["sign"],
                                      "fill_time": parse_time(fill["time"]), "fill_cent": cents(fill["price"]),
                                      "fill_qty": fill["quantity"], "model": fill["model"]})
            outcomes.append(r)
    # Labels need real quotes after the last fill; no new simulated tasks here.
    sim.advance_to(parse_time("14:51:00"))
    fills = pd.DataFrame(all_fills)
    if len(fills):
        mid_fill = sim.mids_at(fills.fill_time.to_numpy())
        fills["mid_at_fill"] = mid_fill
        fills["capture_at_fill_bps"] = fills.sign * np.log(mid_fill / fills.fill_cent) * 10000
        for h in [1, 5, 30]:
            future = sim.mids_at(fills.fill_time.to_numpy() + h * 1000)
            fills[f"postfill_drift_{h}_bps"] = fills.sign * np.log(future / mid_fill) * 10000
            fills[f"markout_{h}_bps"] = fills.sign * np.log(future / fills.fill_cent) * 10000
        fills["wait_seconds"] = (fills.fill_time - fills.time_ms) / 1000
    if not sim.book.healthy: raise ValueError(f"Book errors: {sim.book.stats}")
    return pd.DataFrame(outcomes), fills


def passive_experiments(x, symbols=None, arrival_delay_seconds=0):
    source = ArchiveData(ROOT / "sse50_20260923.zip")
    outcomes, fills = [], []
    for symbol in (symbols or source.symbols):
        tasks, executions = simulate_symbol(source.load(symbol), x[x.symbol == symbol], arrival_delay_seconds)
        outcomes.append(tasks); fills.append(executions)
        print(f"EXECUTION delay={arrival_delay_seconds}s {symbol}: tasks={len(tasks)}, any_fill={tasks.any_fill.mean():.3f}, full_fill={tasks.full_fill.mean():.3f}", flush=True)
    tasks = pd.concat(outcomes, ignore_index=True)
    executions = pd.concat(fills, ignore_index=True)
    output = OUT if not arrival_delay_seconds else OUT / f"arrival_{arrival_delay_seconds}s"
    output.mkdir(parents=True, exist_ok=True)
    tasks.to_parquet(output / "passive_tasks.parquet", index=False)
    executions.to_parquet(output / "passive_fills.parquet", index=False)
    return tasks, executions


def execution_analysis(tasks, fills, output=OUT):
    keys = ["symbol", "time_ms", "side"]
    metrics = ["capture_at_fill_bps", "postfill_drift_1_bps", "postfill_drift_5_bps", "postfill_drift_30_bps",
               "markout_1_bps", "markout_5_bps", "markout_30_bps", "wait_seconds"]
    weighted = fills[keys].copy()
    for f in metrics:
        weighted[f] = fills[f] * fills.fill_qty
        weighted[f + "_weight"] = np.where(fills[f].notna(), fills.fill_qty, 0)
    weighted = weighted.groupby(keys).sum()
    for f in metrics: weighted[f] /= weighted[f + "_weight"].replace(0, np.nan)
    x = tasks.merge(weighted[metrics].reset_index(), on=keys, how="left")
    x["obi_regime"] = np.select([x.aligned_obi < -.5, x.aligned_obi > .5], ["adverse", "favorable"], default="neutral")
    x["flow_regime"] = np.select([x.aligned_flow < -.5, x.aligned_flow > .5], ["adverse", "favorable"], default="neutral")
    for delay in [1, 5, 30]:
        x[f"wait{delay}_saving_bps"] = x.sign * (x.immediate_cent - x[f"wait{delay}_cent"]) / x.mid_cent * 10000
    x["maker_then_cross_saving_bps"] = x.sign * (x.immediate_cent - x.maker_then_cross_cent) / x.mid_cent * 10000
    x["min5_commission_oneway_bps"] = 5 / (x.quantity * x.immediate_cent / 100) * 10000
    # Paired mandatory-trade policies. Deadline remainder valuation is NOT an
    # actual resubmission when partial fills violate minimum order size.
    policies = {
        "immediate": x.immediate_cent,
        "wait_1s": x.wait1_cent, "wait_5s": x.wait5_cent, "wait_30s": x.wait30_cent,
        "obi_timing_5s": np.where(x.aligned_obi > 0, x.immediate_cent, x.wait5_cent),
        "obi_timing_30s": np.where(x.aligned_obi > 0, x.immediate_cent, x.wait30_cent),
        "combo_timing_30s": np.where(x.aligned_score > 0, x.immediate_cent, x.wait30_cent),
        "passive30_then_quote_remainder": x.maker_then_cross_cent,
        "combo_aggressive_else_passive": np.where(x.aligned_score > 0, x.immediate_cent, x.maker_then_cross_cent),
    }
    policy_rows = []
    for name, price in policies.items():
        saving = x.sign * (x.immediate_cent - price) / x.mid_cent * 10000
        x["policy_" + name] = saving
        by_stock = x.assign(saving=saving).groupby("symbol").saving.mean()
        policy_rows.append({"policy": name, "tasks": len(x), "mean_saving_bps": saving.mean(),
                           "median_saving_bps": np.median(saving), "positive_task_fraction": (saving > 0).mean(),
                           "positive_stock_fraction": (by_stock > 0).mean(),
                           "p05_saving_bps": np.quantile(saving, .05), "p95_saving_bps": np.quantile(saving, .95)})
    pd.DataFrame(policy_rows).to_parquet(output / "timing_policies.parquet", index=False)
    group_rows = []
    groups = [("ALL", "ALL", x)]
    for grouping in ["obi_regime", "flow_regime", "side"]:
        groups.extend((grouping, name, z) for name, z in x.groupby(grouping))
    for a, b in [("obi_regime", "flow_regime")]:
        groups.extend(("obi_flow", str(name), z) for name, z in x.groupby([a, b]))
    for grouping, name, z in groups:
        row = {"grouping": grouping, "group": name, "tasks": len(z), "any_fill": z.any_fill.mean(),
               "full_fill": z.full_fill.mean(), "partial_fill": z.partial_fill.mean(), "fill_fraction": z.filled_fraction.mean(),
               "median_queue_ahead": z.queue_ahead.median(), "median_spread_bps": z.spread_bps.median(),
               "wait30_saving_bps": z.wait30_saving_bps.mean(), "passive_saving_bps": z.maker_then_cross_saving_bps.mean(),
               "lot_invalid_fraction": (~z.remainder_order_lot_valid).mean(),
               "postfill30_negative_fraction": (z.loc[z.postfill_drift_30_bps.notna(), "postfill_drift_30_bps"] < 0).mean()}
        for f in metrics: row[f] = z[f].mean()
        group_rows.append(row)
    pd.DataFrame(group_rows).to_parquet(output / "passive_groups.parquet", index=False)
    x.to_parquet(output / "execution_enriched.parquet", index=False)
    stocks = x.groupby("symbol").agg(tasks=("side", "size"), any_fill=("any_fill", "mean"), full_fill=("full_fill", "mean"),
           drift30=("postfill_drift_30_bps", "mean"), passive_saving=("maker_then_cross_saving_bps", "mean"),
           timing_saving=("policy_combo_timing_30s", "mean"), min5_fee=("min5_commission_oneway_bps", "median"))
    stocks.to_parquet(output / "execution_by_stock.parquet")
    dump(output / "execution_summary.json", {"tasks": len(x), "stocks": int(x.symbol.nunique()),
         "fill_event_count": len(fills), "fill_model_counts": fills.model.value_counts().to_dict(),
         "missing_fill_markouts": {f: int(fills[f].isna().sum()) for f in metrics},
         "policies": json.loads(pd.DataFrame(policy_rows).to_json(orient="records")),
         "groups": json.loads(pd.DataFrame(group_rows).to_json(orient="records")),
         "min5_commission_median_bps": x.min5_commission_oneway_bps.median(),
         "arrival_delay_seconds": int(x.arrival_delay_seconds.iloc[0]) if "arrival_delay_seconds" in x else 0,
         "taker_task_fraction": float((x.taker_filled_qty > 0).mean()) if "taker_filled_qty" in x else 0.,
         "method": "105 PM minute starts per stock; independent buy and old-inventory sell tasks; 100 main-board or 200 STAR shares; fixed signal-time best limit, arrives after specified delay, stays for 30s after arrival, then immediate cancellation. Original passive intentions can become takers on arrival. Accounts reset between tasks; all residuals valued at deadline opposite visible book (not actual fallback submissions)."})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["all", "signals", "simulate", "analyze", "latency"])
    parser.add_argument("--symbols", nargs="+", type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    source_path = ROOT / "sse50_20260923.zip"
    before = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if args.stage in ("all", "signals"):
        x = fit_scores(pd.read_parquet(CLEAN / "factor_panel_5s.parquet"))
        combination_analysis(x)
        print("SIGNALS complete", flush=True)
    if args.stage in ("all", "simulate"):
        if args.stage == "simulate": x = pd.read_parquet(OUT / "scored_panel.parquet")
        tasks, fills = passive_experiments(x, args.symbols)
        execution_analysis(tasks, fills)
    if args.stage == "analyze":
        tasks = pd.read_parquet(OUT / "passive_tasks.parquet")
        fills = pd.read_parquet(OUT / "passive_fills.parquet")
        execution_analysis(tasks, fills)
    if args.stage == "latency":
        x = pd.read_parquet(OUT / "scored_panel.parquet")
        for delay in [1, 5]:
            tasks, fills = passive_experiments(x, args.symbols, delay)
            execution_analysis(tasks, fills, OUT / f"arrival_{delay}s")
    after = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if before != after: raise RuntimeError("Original archive changed")
    manifest = {"stage": args.stage, "source_unchanged": True, "sha256": after,
         "date": "20260923", "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(OUT / "manifest.json", manifest)
    dump(OUT / f"manifest_{args.stage}.json", manifest)


if __name__ == "__main__": main()
