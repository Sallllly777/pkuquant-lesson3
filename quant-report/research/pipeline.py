from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from l2sim.book import OrderBook
from l2sim.data import ArchiveData, cents, parse_time
from l2sim.rules import AM_START, AM_END, PM_START, CLOSE_START

ROOT = Path(__file__).resolve().parents[1]
DAY = "20260923"
CLEAN = ROOT / "data" / "cleaned" / DAY
OUTPUT = ROOT / "reports" / f"factor_research_{DAY}"
HORIZONS = [1, 5, 10, 30, 60, 120, 300]
SESSIONS = [(AM_START, AM_END), (PM_START, CLOSE_START)]


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                    default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def time_array(raw):
    raw = np.asarray(raw, dtype=np.int64)
    h, m, s, ms = raw // 10000000, raw // 100000 % 100, raw // 1000 % 100, raw % 1000
    valid = (raw >= 0) & (h < 24) & (m < 60) & (s < 60)
    return ((h * 3600 + m * 60 + s) * 1000 + ms).astype(np.int32), valid


def session_array(times):
    result = np.full(len(times), "other", dtype="U18")
    result[(times >= parse_time("09:15:00")) & (times <= parse_time("09:25:00"))] = "open_auction"
    result[(times >= AM_START) & (times < AM_END)] = "continuous_am"
    result[(times >= PM_START) & (times < CLOSE_START)] = "continuous_pm"
    result[(times >= CLOSE_START) & (times <= parse_time("15:00:00"))] = "close_auction"
    result[(times >= parse_time("15:05:00")) & (times <= parse_time("15:30:33"))] = "after_hours"
    return result


def clean(archive):
    """Preserve original fields; add typed analysis fields and partition mechanisms."""
    CLEAN.mkdir(parents=True, exist_ok=True)
    summary = {}
    for kind, original in list(archive.tables.items()):
        n = len(original)
        t = original
        valid = np.ones(n, dtype=bool)
        nulls, nonfinite = {}, {}
        for name in t.column_names:
            col = t[name]
            if col.null_count:
                nulls[name] = col.null_count
                valid &= np.asarray(pc.is_valid(col))
            if pa.types.is_floating(col.type):
                bad = ~np.isfinite(col.to_numpy())
                if bad.any():
                    nonfinite[name] = int(bad.sum())
                    valid &= ~bad
        times, valid_time = time_array(t["Time"].to_numpy())
        valid &= valid_time
        price, volume = t["Price"].to_numpy(), t["Volume"].to_numpy()
        if kind == "ord":
            k, f = t["OrderKind"].to_numpy(), t["FunctionCode"].to_numpy()
            status = k == 83
            valid &= (status & np.isin(f, [73, 79, 74, 67, 86]) & (price == 0) & (volume == 0)) | (np.isin(k, [65, 68]) & np.isin(f, [66, 83]) & (price > 0) & (volume > 0))
        elif kind == "exe":
            valid &= (price > 0) & (volume > 0) & np.isin(t["BSFlag"].to_numpy(), [0, 66, 83])
        else:
            valid &= (price >= 0) & (volume >= 0)
        keys = ["Symbol", "Time"] if kind == "snp" else ["Channel", "BizIndex"]
        keyframe = pd.DataFrame({k: t[k].to_numpy() for k in keys})
        duplicated = keyframe.duplicated(keep=False).to_numpy()
        exact_removed = np.zeros(n, dtype=bool)
        conflicting = np.zeros(n, dtype=bool)
        if duplicated.any():
            rows = t.filter(pa.array(duplicated)).to_pylist()
            positions = np.flatnonzero(duplicated)
            groups = {}
            for i, row in zip(positions, rows):
                groups.setdefault(tuple(row[k] for k in keys), []).append((i, row))
            for group in groups.values():
                if all(r == group[0][1] for _, r in group):
                    for i, _ in group[1:]: exact_removed[i] = True
                else:
                    for i, _ in group: conflicting[i] = True
        # Conflicting tick business IDs cannot be safely reconstructed.
        # Distinct snapshots sharing a nominal timestamp are retained and flagged.
        if kind != "snp": valid &= ~conflicting
        rejected = ~valid
        keep = valid & ~exact_removed
        normalized_prices = {}
        price_fields = [name for name in t.column_names if name in ["Price", "Open", "High", "Low", "PreClose", "AfterPrice"] or name.startswith(("AskPrice", "BidPrice"))]
        for name in price_fields:
            a = t[name].to_numpy().astype(float)
            finite = np.isfinite(a)
            distance = np.abs(a[finite] * 100 - np.rint(a[finite] * 100))
            normalized_prices[name] = {"max_rounding_distance_cent": float(distance.max(initial=0)), "above_0_1_cent": int((distance > .1).sum())}
            if (distance > .1).any():
                # Do not silently round genuine non-cent prices.
                bad = np.zeros(n, dtype=bool)
                bad[finite] = distance > .1
                rejected |= bad; keep &= ~bad
            t = t.append_column(name + "_Cent", pa.array(np.rint(np.where(finite, a, 0) * 100).astype(np.int32)))
        t = t.append_column("Time_Ms", pa.array(times)).append_column("TradingDate", pa.array(np.full(n, int(DAY), dtype=np.int32)))
        t = t.append_column("Session", pa.array(session_array(times)))
        files = {}
        if kind == "ord":
            masks = {"orders": keep & ~status, "status_events": keep & status}
        elif kind == "exe":
            channel = original["Channel"].to_numpy()
            flags = original["BSFlag"].to_numpy()
            after = channel == 2301
            continuous = ((times >= AM_START) & (times < AM_END)) | ((times >= PM_START) & (times < CLOSE_START))
            auction = flags == 0
            masks = {"trades_continuous": keep & ~after & continuous & ~auction,
                     "trades_auction": keep & ~after & auction,
                     "trades_after_hours": keep & after,
                     "trades_other": keep & ~after & ~auction & ~continuous}
        else:
            bid, ask = original["BidPrice1"].to_numpy(), original["AskPrice1"].to_numpy()
            t = t.append_column("TwoSidedValid", pa.array((bid > 0) & (ask > bid)))
            t = t.append_column("NominalTimeConflict", pa.array(conflicting))
            masks = {"snapshots": keep}
        for name, mask in masks.items():
            part = t.filter(pa.array(mask))
            if len(part):
                sort_keys = [("Symbol", "ascending"), ("Time_Ms", "ascending")]
                if "BizIndex" in part.column_names: sort_keys += [("Channel", "ascending"), ("BizIndex", "ascending")]
                part = part.take(pc.sort_indices(part, sort_keys=sort_keys))
            path = CLEAN / (name + ".parquet")
            pq.write_table(part, path, compression="zstd")
            files[path.name] = len(part)
        if rejected.any():
            pq.write_table(original.filter(pa.array(rejected)), CLEAN / f"quarantine_{kind}.parquet", compression="zstd")
        # The research replay uses only retained rows, with the same original schema.
        archive.tables[kind] = original.filter(pa.array(keep))
        summary[kind] = {"input_rows": n, "retained_rows": int(keep.sum()), "exact_duplicates_removed": int(exact_removed.sum()),
                         "conflicting_key_rows": int(conflicting.sum()), "quarantined_rows": int(rejected.sum()),
                         "nulls": nulls, "nonfinite": nonfinite, "invalid_time": int((~valid_time).sum()),
                         "zero_price_rows": int((price == 0).sum()), "zero_volume_rows": int((volume == 0).sum()),
                         "price_rounding": normalized_prices, "files": files}
        print(f"CLEAN {kind}: {n:,} -> {keep.sum():,}; quarantined={rejected.sum()}", flush=True)
    dump(OUTPUT / "quality.json", summary)
    if any(s["quarantined_rows"] for s in summary.values()):
        raise ValueError("Quarantined records require review before full-book factor extraction")
    return summary


class ResearchBook(OrderBook):
    """Same queue semantics as the simulator, with O(log n) best-price lookup."""
    def __init__(self):
        super().__init__()
        self.totals = {"BUY": {}, "SELL": {}}
        self.heaps = {"BUY": [], "SELL": []}

    def _change(self, side, price, delta):
        levels = self.totals[side]
        if price not in levels and delta > 0:
            heapq.heappush(self.heaps[side], -price if side == "BUY" else price)
        levels[price] = levels.get(price, 0) + delta
        if levels[price] <= 0: del levels[price]

    def remove(self, key):
        q = self.orders.get(key)
        if q: self._change(q.side, q.price, -q.quantity)
        super().remove(key)

    def reduce(self, key, quantity, required=True):
        q = self.orders.get(key)
        if q is None:
            self.stats["missing_resting_order" if required else "unpublished_aggressor"] += 1
            return
        if quantity > q.quantity: self.stats["overdraw"] += 1
        self._change(q.side, q.price, -min(quantity, q.quantity))
        q.quantity -= quantity
        if q.quantity <= 0: super().remove(key)

    def apply(self, e, priority):
        super().apply(e, priority)
        if e.kind == "ord" and e.row["OrderKind"] == 65:
            q = self.orders.get((e.channel, e.row["OrderOriNo"]))
            if q: self._change(q.side, q.price, q.quantity)

    def best(self, side):
        heap = self.heaps[side]
        while heap:
            p = -heap[0] if side == "BUY" else heap[0]
            if p in self.totals[side]: return p
            heapq.heappop(heap)
        return None

    def top(self):
        b, a = self.best("BUY"), self.best("SELL")
        return b or 0, self.totals["BUY"].get(b, 0), a or 0, self.totals["SELL"].get(a, 0)

    def ladder(self, side, n=10):
        prices = heapq.nlargest(n, self.totals[side]) if side == "BUY" else heapq.nsmallest(n, self.totals[side])
        return prices, [self.totals[side][p] for p in prices]


FLOW = ["buy_qty", "sell_qty", "buy_count", "sell_count", "buy_large", "sell_large", "notional_cent",
        "add_buy", "add_sell", "cancel_buy", "cancel_sell", "fast_cancel_buy", "fast_cancel_sell", "ofi", "events"]


def ofi_change(old, new):
    b0, bv0, a0, av0 = old
    b1, bv1, a1, av1 = new
    if min(b0, a0, b1, a1) <= 0: return 0
    return int(b1 >= b0) * bv1 - int(b1 <= b0) * bv0 - int(a1 <= a0) * av1 + int(a1 >= a0) * av0


def ratio(a, b):
    return (a - b) / (a + b) if a + b > 0 else 0.0


def build_symbol(data):
    book = ResearchBook()
    added_at = {}
    rows = []
    seconds = {start: np.zeros(((end - start) // 1000 + 1, len(FLOW))) for start, end in SESSIONS}
    fi = {name: i for i, name in enumerate(FLOW)}
    grids = [t for start, end in SESSIONS for t in range(start, end, 5000)]
    grid_index = 0
    last_event_time = None
    previous_top = (0, 0, 0, 0)
    quote_times, bids, asks = [], [], []
    bad_grids = 0

    def sample(t):
        nonlocal bad_grids
        bp, bv = book.ladder("BUY")
        ap, av = book.ladder("SELL")
        if not bp or not ap or bp[0] >= ap[0] or not book.healthy:
            bad_grids += 1
            return
        b, a = bp[0], ap[0]
        mid = (b + a) / 2
        weight_b = sum(v / (i + 1) for i, v in enumerate(bv))
        weight_a = sum(v / (i + 1) for i, v in enumerate(av))
        rows.append({"symbol": data.symbol, "time_ms": t, "session": "AM" if t < AM_END else "PM",
                     "mid_cent": mid, "bid_cent": b, "ask_cent": a,
                     "depth1": bv[0] + av[0], "depth10": sum(bv) + sum(av),
                     "obi1": ratio(bv[0], av[0]), "obi5": ratio(sum(bv[:5]), sum(av[:5])),
                     "obi10": ratio(sum(bv), sum(av)), "weighted_obi": ratio(weight_b, weight_a),
                     "weighted_mid_bias": ((a * bv[0] + b * av[0]) / (bv[0] + av[0]) - mid) / mid * 10000,
                     "ladder_gap_bias": ((ap[-1] - a) - (b - bp[-1])) / mid * 10000,
                     "concentration_bias": bv[0] / sum(bv) - av[0] / sum(av),
                     "spread_bps": (a - b) / mid * 10000,
                     "log_depth10": np.log1p(sum(bv) + sum(av)),
                     "quote_age_ms": t - last_event_time if last_event_time is not None else np.nan})

    tick_events = [e for e in data.events if e.kind in ("ord", "exe")]
    for i, e in enumerate(tick_events):
        while grid_index < len(grids) and grids[grid_index] < e.time:
            sample(grids[grid_index]); grid_index += 1
        old = book.top()
        r = e.row
        in_session = next(((start, end) for start, end in SESSIONS if start <= e.time < end), None)
        flow = None
        if in_session:
            start, end = in_session
            # ceil-to-second bin: each bin contains (t-1s, t], not future events.
            sec = (e.time - start + 999) // 1000
            flow = seconds[start][sec]
            flow[fi["events"]] += 1
        if e.kind == "ord":
            key = (e.channel, r["OrderOriNo"])
            if r["OrderKind"] == 65:
                added_at[key] = e.time
                if flow is not None:
                    flow[fi["add_buy" if r["FunctionCode"] == 66 else "add_sell"]] += r["Volume"]
            elif r["OrderKind"] == 68:
                birth = added_at.get(key)
                if flow is not None:
                    side = "buy" if r["FunctionCode"] == 66 else "sell"
                    flow[fi["cancel_" + side]] += r["Volume"]
                    if birth is not None and 0 <= e.time - birth <= 1000:
                        flow[fi["fast_cancel_" + side]] += r["Volume"]
        elif flow is not None and e.channel != 2301 and r["BSFlag"] in (66, 83):
            side = "buy" if r["BSFlag"] == 66 else "sell"
            flow[fi[side + "_qty"]] += r["Volume"]
            flow[fi[side + "_count"]] += 1
            amount = cents(r["Price"]) * r["Volume"]
            flow[fi["notional_cent"]] += amount
            if amount >= 100000 * 100:
                flow[fi[side + "_large"]] += r["Volume"]
        book.apply(e, i + 1)
        if e.kind == "ord" and r["OrderKind"] == 68:
            added_at.pop((e.channel, r["OrderOriNo"]), None)
        elif e.kind == "exe" and e.channel != 2301:
            for field in ("BidOrder", "AskOrder"):
                key = (e.channel, r[field])
                if key not in book.orders: added_at.pop(key, None)
        new = book.top()
        if flow is not None: flow[fi["ofi"]] += ofi_change(old, new)
        if new != previous_top:
            last_event_time = e.time
            previous_top = new
        # Save prices, including invalid states, so as-of lookup cannot skip an
        # invalid state and accidentally resurrect an older valid quote.
        if not quote_times or (new[0], new[2]) != (bids[-1], asks[-1]):
            quote_times.append(e.time); bids.append(new[0]); asks.append(new[2])
        if i + 1 == len(tick_events) or tick_events[i + 1].time != e.time:
            if grid_index < len(grids) and grids[grid_index] == e.time:
                sample(grids[grid_index]); grid_index += 1
    while grid_index < len(grids):
        sample(grids[grid_index]); grid_index += 1
    if not book.healthy:
        raise ValueError(f"Book integrity failed for {data.symbol}: {book.stats}")
    df = pd.DataFrame(rows)
    times = np.asarray(quote_times)
    bid_values, ask_values = np.asarray(bids, dtype=float), np.asarray(asks, dtype=float)

    def quotes_at(target):
        ix = np.searchsorted(times, np.asarray(target), side="right") - 1
        b, a = bid_values[np.maximum(ix, 0)].copy(), ask_values[np.maximum(ix, 0)].copy()
        invalid = (ix < 0) | (b <= 0) | (a <= b)
        b[invalid] = np.nan; a[invalid] = np.nan
        return b, a

    for start, end in SESSIONS:
        mask = (df.time_ms >= start) & (df.time_ms < end)
        x = df.loc[mask].copy()
        ts = x.time_ms.to_numpy()
        sec = (ts - start) // 1000
        prefix = np.vstack([np.zeros((1, len(FLOW))), np.cumsum(seconds[start], axis=0)])
        for window in [5, 30, 60]:
            # Exclude events exactly at the left endpoint of (t-window,t].
            f = prefix[sec + 1] - prefix[np.maximum(0, sec - window + 1)]
            totals = {name: f[:, j] for j, name in enumerate(FLOW)}
            for name in FLOW: x[f"raw_{name}_{window}"] = totals[name]
            tq = totals["buy_qty"] + totals["sell_qty"]
            x[f"trade_imb_{window}"] = np.divide(totals["buy_qty"] - totals["sell_qty"], tq, out=np.zeros(len(x)), where=tq > 0)
            x[f"ofi_{window}"] = totals["ofi"] / x.depth1.clip(lower=1)
            pb, pa_ = quotes_at(ts - window * 1000)
            pm = (pb + pa_) / 2
            x[f"momentum_{window}"] = np.log(x.mid_cent / pm) * 10000
            x.loc[ts - window * 1000 < start, f"momentum_{window}"] = np.nan
        def imbalance(b, a):
            denominator = x[b] + x[a]
            return (x[b] - x[a]).div(denominator.where(denominator != 0)).fillna(0)
        x["trade_count_imb"] = imbalance("raw_buy_count_30", "raw_sell_count_30")
        x["large_trade_imb"] = imbalance("raw_buy_large_30", "raw_sell_large_30")
        x["new_order_imb"] = imbalance("raw_add_buy_30", "raw_add_sell_30")
        x["cancel_pressure"] = imbalance("raw_cancel_sell_30", "raw_cancel_buy_30")
        x["fast_cancel_pressure"] = imbalance("raw_fast_cancel_sell_30", "raw_fast_cancel_buy_30")
        gross = x.raw_add_buy_30 + x.raw_add_sell_30 + x.raw_cancel_buy_30 + x.raw_cancel_sell_30
        net = x.raw_add_buy_30 - x.raw_add_sell_30 + x.raw_cancel_sell_30 - x.raw_cancel_buy_30
        x["net_resting_flow"] = net.div(gross.where(gross != 0)).fillna(0)
        qty = x.raw_buy_qty_30 + x.raw_sell_qty_30
        count = x.raw_buy_count_30 + x.raw_sell_count_30
        vwap = x.raw_notional_cent_30.div(qty.where(qty != 0))
        x["vwap_gap"] = (x.mid_cent - vwap) / x.mid_cent * 10000
        x["event_rate"] = x.raw_events_30 / 30
        x["mean_trade_value"] = (x.raw_notional_cent_30 / 100).div(count.where(count != 0))
        x["cancel_share"] = (x.raw_cancel_buy_30 + x.raw_cancel_sell_30).div(gross.where(gross != 0)).fillna(0)
        returns = np.log(x.mid_cent).diff()
        x["realized_vol_60"] = np.sqrt(returns.pow(2).rolling(12, min_periods=12).sum()) * 10000
        for h in HORIZONS + [31, 35]:
            fb, fa = quotes_at(ts + h * 1000)
            valid = ts + h * 1000 < end
            fb[~valid] = np.nan; fa[~valid] = np.nan
            x[f"ret_{h}s"] = np.log(((fb + fa) / 2) / x.mid_cent) * 10000
            x[f"bid_fwd_{h}"] = fb
            x[f"ask_fwd_{h}"] = fa
        # Fixed 30-second holding interval after a 1s/5s execution delay.
        x["delayed_1s_30"] = x.ret_31s - x.ret_1s
        x["delayed_5s_30"] = x.ret_35s - x.ret_5s
        for col in x.columns: df.loc[mask, col] = x[col].to_numpy()
    stats = {"symbol": data.symbol, "sample_rows": len(df), "invalid_grid_rows": bad_grids,
             "book_diagnostics": dict(book.stats)}
    return df, stats


def features(archive):
    results, diagnostics = [], []
    for symbol in archive.symbols:
        df, stats = build_symbol(archive.load(symbol))
        results.append(df); diagnostics.append(stats)
        print(f"FEATURES {symbol}: {len(df):,} samples, invalid_grid={stats['invalid_grid_rows']}", flush=True)
    all_rows = pd.concat(results, ignore_index=True).copy()
    # Cross-sectional features use only already-observed past returns; not the
    # actual SSE50 weights (which are absent from this data).
    grouped = all_rows.groupby("time_ms")["momentum_30"]
    n = grouped.transform("count")
    all_rows["peer_momentum_30"] = (grouped.transform("sum") - all_rows.momentum_30) / (n - 1)
    all_rows["relative_momentum_30"] = all_rows.momentum_30 - all_rows.peer_momentum_30
    all_rows = all_rows.replace([np.inf, -np.inf], np.nan)
    all_rows.to_parquet(CLEAN / "factor_panel_5s.parquet", index=False, compression="zstd")
    dump(OUTPUT / "feature_quality.json", {"rows": len(all_rows), "symbols": diagnostics,
                                          "sampling_ms": 5000, "horizons_seconds": HORIZONS})
    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["all", "clean", "features", "analyze"])
    parser.add_argument("--archive", default=str(ROOT / "sse50_20260923.zip"))
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive_path = Path(args.archive)
    before = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if args.stage in ("all", "clean", "features"):
        source = ArchiveData(archive_path)
        if source.date != DAY: raise ValueError("This first study is pinned to 20260923")
        clean(source)
        if args.stage != "clean": panel = features(source)
    if args.stage in ("all", "analyze"):
        from .screen import analyze
        if args.stage == "analyze": panel = pd.read_parquet(CLEAN / "factor_panel_5s.parquet")
        analyze(panel, OUTPUT)
    after = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if before != after: raise RuntimeError("Original archive changed during research")
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    dump(OUTPUT / "manifest.json", {"date": DAY, "stage": args.stage, "source_sha256_before": before,
                                   "source_sha256_after": after, "source_unchanged": True,
                                   "python": platform.python_version(), "pandas": pd.__version__,
                                   "numpy": np.__version__, "pyarrow": pa.__version__, "research_source_sha256": code_hashes})


if __name__ == "__main__":
    main()
