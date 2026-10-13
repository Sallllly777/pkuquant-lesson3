"""Landmark fill/risk models, purged temporal validation and cancellation trials."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from l2sim.data import ArchiveData, parse_time
from .pipeline import ROOT, CLEAN, dump
from .execution_study import OUT as PREVIOUS, simulate_symbol, visible_vwap

OUT = ROOT / "reports" / "final_research_20260923"
KEYS = ["symbol", "time_ms", "side"]
RAW = ["obi1", "trade_imb_5", "trade_imb_30", "weighted_obi", "ofi_5", "momentum_5", "vwap_gap",
       "spread_bps", "realized_vol_60", "log_depth10", "event_rate", "raw_buy_qty_30", "raw_sell_qty_30", "ret_30s"]
SIGNALS = ["aligned_obi", "aligned_flow5", "aligned_flow30", "aligned_depth", "aligned_ofi", "aligned_momentum", "aligned_vwap"]
CONTEXT = ["spread_bps", "vol60", "log_events", "tick_bps", "is_sell"]
QUEUE = ["age_s", "log_queue", "queue_to_flow", "log_remaining", "limit_distance_ticks"]
FEATURES = SIGNALS + CONTEXT + QUEUE
SPEC = {
    "training": "09:35 <= task start < 10:50", "embargo": "10:50 <= task start < 10:55",
    "validation": "10:55 <= task start < 11:25", "afternoon_audit": "13:05 <= task start < 14:50",
    "observations": "Open orders only, at task ages 0,5,10,15,20,25 seconds; deadline age30",
    "models": "Fixed logistic L2=.01; ridge L2=.1; training-only medians/scales and z clipping +/-5",
    "weights": "Equal weight per observed state. Entry and surviving-order models fit separately; no weights based on future survival length",
    "policy": "Price>=0: immediate; else maker if predicted full fill>=.5 and postfill drift>=-.5bp; else wait30",
    "dynamic_cancel": "Post initial best quote, first age>=5 landmark predicting postfill drift<-.5bp cancels; cancellation delays0/1/5s; no re-entry",
    "warning": "Same afternoon already observed in earlier research; temporal separation is not a new independent day test",
}


def split_for_time(times):
    times = np.asarray(times)
    return np.select([times < parse_time("10:50:00"), times < parse_time("10:55:00"), times < parse_time("11:30:00")],
                     ["train", "embargo", "validation"], default="PM_audit")


def collect():
    panel = pd.read_parquet(PREVIOUS / "scored_panel.parquet")
    original = pd.read_parquet(CLEAN / "factor_panel_5s.parquet", columns=["symbol", "time_ms"] + RAW)
    source = ArchiveData(ROOT / "sse50_20260923.zip")
    all_tasks, all_fills, all_states = [], [], []
    for symbol in source.symbols:
        states = []
        def observe(sim, signal, records, age):
            if age == 30:
                for r in states:
                    if r["time_ms"] != signal.time_ms: continue
                    price, available = visible_vwap(sim.book, r["side"], r["remaining_qty"])
                    if available != r["remaining_qty"]: raise ValueError("Insufficient terminal depth")
                    r["deadline_quote_cent"] = price
                return
            for side, info in records.items():
                order = sim.orders[info["order_id"] - 1]
                if not order.remaining: continue
                price, available = visible_vwap(sim.book, side, order.remaining)
                if available != order.remaining: raise ValueError("Insufficient current depth")
                b, a = sim.book.best("BUY"), sim.book.best("SELL")
                if not b or not a or b >= a: raise ValueError("Invalid state at model landmark")
                sign = info["sign"]
                states.append({"symbol": symbol, "time_ms": int(signal.time_ms), "side": side, "sign": sign,
                     "state_time": sim.time, "age_s": age, "remaining_qty": order.remaining,
                     "filled_so_far": order.filled, "notional_so_far": order.notional,
                     "queue_ahead": sim.queue_ahead(order), "limit_cent": order.price,
                     "limit_distance_ticks": sign * (order.price - sim.book.best(side)),
                     "state_mid_cent": (a + b) / 2, "current_taker_cent": price})
        tasks, fills = simulate_symbol(source.load(symbol), panel[panel.symbol == symbol], sessions=("AM", "PM"), state_callback=observe)
        all_tasks.append(tasks); all_fills.append(fills); all_states.extend(states)
        print(f"LANDMARKS {symbol}: {len(tasks)} tasks, {len(states)} open-order states", flush=True)
    tasks = pd.concat(all_tasks, ignore_index=True)
    fills = pd.concat(all_fills, ignore_index=True)
    states = pd.DataFrame(all_states)
    states = states.merge(original.rename(columns={"time_ms": "state_time"}), on=["symbol", "state_time"], how="left", validate="many_to_one")
    tasks.to_parquet(OUT / "tasks.parquet", index=False)
    fills.to_parquet(OUT / "fills.parquet", index=False)
    states.to_parquet(OUT / "observed_states.parquet", index=False)
    return tasks, fills, states


def make_labels(tasks, fills, states):
    x = states.copy()
    fill_groups = {key: g for key, g in fills.groupby(KEYS)}
    task_values = tasks.set_index(KEYS)
    labels = []
    for r in x.itertuples():
        key = (r.symbol, r.time_ms, r.side)
        g = fill_groups.get(key)
        g = g[g.fill_time > r.state_time] if g is not None else None
        quantity = g.fill_qty.sum() if g is not None else 0
        risk = np.average(g.postfill_drift_30_bps, weights=g.fill_qty) if quantity else np.nan
        mark = np.average(g.markout_30_bps, weights=g.fill_qty) if quantity else np.nan
        task = task_values.loc[key]
        remainder_cost = (task.maker_then_cross_cent * task.quantity - r.notional_so_far) / r.remaining_qty
        labels.append({"y_full": int(quantity >= r.remaining_qty), "y_any": int(quantity > 0),
                       "y_postfill": risk, "y_markout": mark,
                       "y_price30": r.sign * r.ret_30s,
                       "y_passive_saving": r.sign * (r.current_taker_cent - remainder_cost) / r.state_mid_cent * 10000,
                       "y_wait_saving": r.sign * (r.current_taker_cent - r.deadline_quote_cent) / r.state_mid_cent * 10000})
    x = pd.concat([x.reset_index(drop=True), pd.DataFrame(labels)], axis=1)
    mapping = dict(zip(SIGNALS, ["obi1", "trade_imb_5", "trade_imb_30", "weighted_obi", "ofi_5", "momentum_5", "vwap_gap"]))
    for output, raw in mapping.items(): x[output] = x.sign * x[raw]
    x["vol60"] = x.realized_vol_60
    x["log_events"] = np.log1p(x.event_rate)
    x["tick_bps"] = 10000 / x.state_mid_cent
    x["is_sell"] = (x.side == "SELL").astype(int)
    x["log_queue"] = np.log1p(x.queue_ahead)
    contra = np.where(x.side == "BUY", x.raw_sell_qty_30, x.raw_buy_qty_30)
    x["queue_to_flow"] = np.log1p(x.queue_ahead / (contra + 100))
    x["log_remaining"] = np.log1p(x.remaining_qty)
    x["split"] = split_for_time(x.time_ms)
    # An inverse number-of-observations weight would depend on FUTURE fill time:
    # quickly filled orders have fewer observations and would be overweighted.
    x["weight"] = 1.
    x.to_parquet(OUT / "labeled_states.parquet", index=False)
    return x


class LinearModel:
    def __init__(self, columns, kind="ridge"):
        if any(c.startswith(("y_", "ret_", "pred_")) or "fwd" in c for c in columns):
            raise ValueError("Future labels or predictions cannot enter the raw feature model")
        self.columns, self.kind = list(columns), kind

    def matrix(self, frame, fit=False):
        a = frame[self.columns].to_numpy(dtype=float)
        a[~np.isfinite(a)] = np.nan
        if fit: self.median = np.nanmedian(a, axis=0)
        a = np.where(np.isnan(a), self.median, a)
        if fit:
            self.mean = a.mean(axis=0)
            self.std = np.maximum(a.std(axis=0), 1e-8)
        return np.column_stack([np.ones(len(a)), np.clip((a - self.mean) / self.std, -5, 5)])

    def fit(self, frame, target):
        valid = frame[target].notna()
        t = frame[valid]
        a = self.matrix(t, True)
        y = t[target].to_numpy(dtype=float)
        w = t.weight.to_numpy(dtype=float); w /= w.sum()
        penalty = np.eye(a.shape[1]); penalty[0, 0] = 0
        if self.kind == "ridge":
            self.coef = np.linalg.solve(a.T @ (w[:, None] * a) + .1 * penalty, a.T @ (w * y))
        else:
            def loss(beta):
                logits = a @ beta
                value = np.sum(w * (np.logaddexp(0, logits) - y * logits)) + .005 * beta @ penalty @ beta
                grad = a.T @ (w * (expit(logits) - y)) + .01 * penalty @ beta
                return value, grad
            result = minimize(loss, np.zeros(a.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-11})
            if not result.success: raise RuntimeError(result.message)
            self.coef = result.x
        return self

    def predict(self, frame):
        values = self.matrix(frame) @ self.coef
        return expit(values) if self.kind == "logistic" else values

    def config(self):
        return {"features": self.columns, "kind": self.kind, "median": self.median.tolist(),
                "mean": self.mean.tolist(), "std": self.std.tolist(), "coefficient": self.coef.tolist()}


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    positive = y.sum(); negative = len(y) - positive
    if min(positive, negative) <= 0: return np.nan
    return float((rankdata(p)[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def fit_and_evaluate(x):
    train = x[x.split == "train"]
    specifications = {
        "price": (SIGNALS + CONTEXT, "ridge", "y_price30"),
        "fill_age": (["age_s"], "logistic", "y_full"),
        "fill_signal": (["age_s", "aligned_obi", "aligned_flow5"], "logistic", "y_full"),
        "fill_context": (CONTEXT + ["age_s"], "logistic", "y_full"),
        "fill_context_signal": (SIGNALS + CONTEXT + ["age_s"], "logistic", "y_full"),
        "fill_queue": (CONTEXT + QUEUE, "logistic", "y_full"),
        "fill_full": (FEATURES, "logistic", "y_full"),
        "risk_signal": (SIGNALS + CONTEXT, "ridge", "y_postfill"),
        "risk_full": (FEATURES, "ridge", "y_postfill"),
    }
    models, metrics, calibration, stock_metrics = {}, [], [], []
    for name, (cols, kind, target) in specifications.items():
        # New orders and surviving orders have different selection distributions.
        # Fit their models independently, using only the training time interval.
        models[name] = {}
        for cohort, mask in [("entry", x.age_s == 0), ("surviving", x.age_s > 0)]:
            model = LinearModel(cols, kind).fit(x[mask & (x.split == "train")], target)
            x.loc[mask, "pred_" + name] = model.predict(x[mask])
            models[name][cohort] = model.config()
        for split in ["train", "validation", "PM_audit"]:
            for cohort in ["entry", "all_open_states"]:
                z = x[(x.split == split) & x[target].notna()]
                b = train[train[target].notna()]
                if cohort == "entry":
                    z = z[z.age_s == 0]; b = b[b.age_s == 0]
                y, p = z[target].to_numpy(), z["pred_" + name].to_numpy()
                row = {"model": name, "target": target, "split": split, "cohort": cohort, "n": len(z),
                       "observed_mean": float(np.mean(y)), "predicted_mean": float(np.mean(p))}
                if kind == "logistic":
                    baseline = b[target].mean()
                    row.update(auc=auc(y, p), brier=float(np.mean((p-y)**2)), baseline_brier=float(np.mean((baseline-y)**2)))
                    row["brier_skill"] = 1 - row["brier"] / row["baseline_brier"]
                    if cohort == "entry":
                        bins = np.minimum((p * 5).astype(int), 4)
                        for bin_ in range(5):
                            mask = bins == bin_
                            if mask.any(): calibration.append({"model": name, "split": split, "bin": bin_, "n": int(mask.sum()),
                                                               "predicted": p[mask].mean(), "observed": y[mask].mean()})
                else:
                    baseline = b[target].mean()
                    mse = np.mean((y-p)**2); base = np.mean((y-baseline)**2)
                    row.update(mae=float(np.mean(np.abs(y-p))), mse_skill=float(1-mse/base),
                               rank_ic=pd.Series(y).corr(pd.Series(p), method="spearman"))
                metrics.append(row)
                if cohort == "entry" and split != "train":
                    for symbol, g in z.groupby("symbol"):
                        actual, forecast = g[target], g["pred_" + name]
                        stock_metrics.append({"model": name, "split": split, "symbol": symbol, "n": len(g),
                            "auc": auc(actual, forecast) if kind == "logistic" else np.nan,
                            "rank_ic": actual.corr(forecast, method="spearman") if kind == "ridge" and actual.nunique() > 1 and forecast.nunique() > 1 else np.nan})
    dump(OUT / "model_parameters.json", models)
    pd.DataFrame(metrics).to_parquet(OUT / "model_metrics.parquet", index=False)
    pd.DataFrame(calibration).to_parquet(OUT / "fill_calibration.parquet", index=False)
    pd.DataFrame(stock_metrics).to_parquet(OUT / "model_stock_metrics.parquet", index=False)
    x.to_parquet(OUT / "predicted_states.parquet", index=False)
    return x


def cancellation_time(states, delay=0):
    candidates = states[(states.age_s >= 5) & (states.pred_risk_full < -.5)].sort_values("age_s")
    return min(30, int(candidates.age_s.iloc[0]) + delay) if len(candidates) else 30


def policies(tasks, fills, x):
    initial = x[x.age_s == 0].set_index(KEYS)
    by_task = {key: group for key, group in x.groupby(KEYS)}
    by_fill = {key: group for key, group in fills.groupby(KEYS)}
    rows = []
    for r in tasks.itertuples():
        key = (r.symbol, r.time_ms, r.side)
        start = initial.loc[key]
        predictions = (start.pred_price, start.pred_fill_full, start.pred_risk_full)
        if start.split == "embargo": continue
        s0 = r.sign * (r.immediate_cent - r.wait30_cent) / r.mid_cent * 10000
        sm = r.sign * (r.immediate_cent - r.maker_then_cross_cent) / r.mid_cent * 10000
        price, chance, risk = predictions
        chosen = "immediate" if price >= 0 else ("passive" if chance >= .5 and risk >= -.5 else "wait")
        chance_only = "immediate" if price >= 0 else ("passive" if chance >= .5 else "wait")
        risk_only = "immediate" if price >= 0 else ("passive" if risk >= -.5 else "wait")
        action_saving = {"immediate": 0., "wait": s0, "passive": sm}
        cases = {
            "immediate": (0., "immediate"), "wait30": (s0, "wait"), "passive30": (sm, "passive"),
            "price_timing": (0. if price >= 0 else s0, "immediate" if price >= 0 else "wait"),
            "price_then_passive": (0. if price >= 0 else sm, "immediate" if price >= 0 else "passive"),
            "price_fill": (action_saving[chance_only], chance_only),
            "price_risk": (action_saving[risk_only], risk_only),
            "three_head": (action_saving[chosen], chosen),
        }
        fg = by_fill.get(key)
        for name, (saving, action) in cases.items():
            rows.append({"symbol": r.symbol, "time_ms": r.time_ms, "side": r.side, "split": start.split,
                         "policy": name, "action": action, "saving_bps": saving,
                         "passive_fraction": r.filled_fraction if action == "passive" else 0.,
                         "late_fill_qty": 0, "lot_invalid": not r.remainder_order_lot_valid if action == "passive" else False})
        states = by_task[key]
        trigger = cancellation_time(states, 0)
        for delay in [0, 1, 5]:
            stop = cancellation_time(states, delay)
            kept = fg[fg.fill_time <= r.time_ms + stop * 1000] if fg is not None else None
            quantity = int(kept.fill_qty.sum()) if kept is not None else 0
            notional = (kept.fill_cent * kept.fill_qty).sum() if quantity else 0
            # Quotes for remaining quantities are stored at the triggering open
            # state. If delay permits extra fills, using the full remaining VWAP
            # slightly approximates depth impact; these are small-order valuations.
            observed = states[states.age_s == trigger]
            fallback = observed.deadline_quote_cent.iloc[0] if len(observed) else r.wait30_cent
            value = (notional + (r.quantity-quantity) * fallback) / r.quantity
            if trigger == 30:
                value = r.maker_then_cross_cent
            saving = r.sign * (r.immediate_cent-value) / r.mid_cent * 10000
            late = int(kept.loc[kept.fill_time > r.time_ms + trigger * 1000, "fill_qty"].sum()) if quantity and trigger < 30 else 0
            from .execution_study import valid_resubmission
            rows.append({"symbol": r.symbol, "time_ms": r.time_ms, "side": r.side, "split": start.split,
                         "policy": f"risk_cancel_delay{delay}", "action": "cancel" if trigger < 30 else "passive",
                         "saving_bps": saving, "passive_fraction": quantity/r.quantity, "late_fill_qty": late,
                         "lot_invalid": not valid_resubmission(r.symbol, r.side, r.quantity-quantity)})
    result = pd.DataFrame(rows)
    result.to_parquet(OUT / "policy_tasks.parquet", index=False)
    summary = result.groupby(["split", "policy"]).agg(n=("side", "size"), mean_saving=("saving_bps", "mean"),
             median_saving=("saving_bps", "median"), downside_05=("saving_bps", lambda s: s.quantile(.05)),
             passive_fraction=("passive_fraction", "mean"), late_fill_qty=("late_fill_qty", "sum"),
             lot_invalid_fraction=("lot_invalid", "mean")).reset_index()
    summary.to_parquet(OUT / "policy_summary.parquet", index=False)
    result.groupby(["split", "policy", "action"]).size().rename("n").reset_index().to_parquet(OUT / "policy_choices.parquet", index=False)
    # Time-block and stock dispersion, not iid confidence intervals.
    result["block"] = result.time_ms // 300000
    result.groupby(["split", "policy", "block"]).saving_bps.mean().reset_index().to_parquet(OUT / "policy_time_blocks.parquet", index=False)
    result.groupby(["split", "policy", "symbol"]).saving_bps.mean().reset_index().to_parquet(OUT / "policy_stocks.parquet", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["all", "collect", "analyze"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT / "experiment_spec.json", SPEC)
    before = hashlib.sha256((ROOT / "sse50_20260923.zip").read_bytes()).hexdigest()
    if args.stage in ("all", "collect"): tasks, fills, states = collect()
    if args.stage in ("all", "analyze"):
        if args.stage == "analyze":
            tasks = pd.read_parquet(OUT / "tasks.parquet")
            fills = pd.read_parquet(OUT / "fills.parquet")
            states = pd.read_parquet(OUT / "observed_states.parquet")
        x = fit_and_evaluate(make_labels(tasks, fills, states))
        policies(tasks, fills, x)
        print("MODELS AND POLICIES complete", flush=True)
    after = hashlib.sha256((ROOT / "sse50_20260923.zip").read_bytes()).hexdigest()
    if before != after: raise RuntimeError("Source changed")
    dump(OUT / "manifest.json", {"source_unchanged": True, "sha256": after, "stage": args.stage,
         "code_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}})


if __name__ == "__main__": main()
