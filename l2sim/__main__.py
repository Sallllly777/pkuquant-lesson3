import argparse
import hashlib
import json
from pathlib import Path

from .data import ArchiveData, parse_time
from .engine import Replay


def save_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {path.resolve()}")


def audit(archive, symbols):
    reports = []
    for symbol in symbols:
        data = archive.load(symbol)
        replay = Replay(data)
        replay.advance_to(parse_time("15:30:33"))
        result = {"symbol": symbol, "date": data.date, "counts": data.counts,
                  "healthy": replay.book.healthy, "diagnostics": dict(replay.book.stats),
                  "snapshot_audit": dict(replay.audit), "snapshot_mismatch_examples": replay.snapshot_examples,
                  "remaining_orders": len(replay.book.orders)}
        reports.append(result)
        print(f"{symbol}: healthy={result['healthy']}, events={sum(data.counts.values()):,}, snapshots={dict(replay.audit)}", flush=True)
    return reports


def demo(archive, symbol):
    replay = Replay(archive.load(symbol), initial_position=1000)
    replay.advance_to(parse_time("09:30:05"))
    min_qty = 200 if str(symbol).startswith("688") else 100
    ask, bid = replay.book.best("SELL"), replay.book.best("BUY")
    if not ask or not bid:
        raise ValueError("No two-sided book at demo time")
    replay.submit("BUY", ask / 100, min_qty)
    replay.submit("SELL", bid / 100, min_qty)
    # Join the historical queue and allow the tape to advance it.
    replay.submit("BUY", bid / 100, min_qty)
    replay.advance_to(parse_time("09:35:00"))
    for o in list(replay.active()):
        replay.cancel(o.id)
    replay.submit("BUY", replay.state()["limits"][0], min_qty)
    replay.cancel(replay.orders[-1].id)
    return replay.state()


def main():
    parser = argparse.ArgumentParser(description="SSE L2 historical replay and paper trading")
    parser.add_argument("command", choices=["serve", "demo", "audit"])
    parser.add_argument("--archive", default="sse50_20260923.zip")
    parser.add_argument("--symbol", type=int, default=600036)
    parser.add_argument("--all", action="store_true", help="Audit all symbols")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    print("Reading ZIP in memory; original data will not be modified...", flush=True)
    archive = ArchiveData(args.archive)
    if args.command == "serve":
        from .server import serve
        serve(archive, args.symbol, args.port)
    elif args.command == "demo":
        result = demo(archive, args.symbol)
        save_report(args.output or "reports/demo.json", result)
        print(json.dumps({k: result[k] for k in ["time", "cash", "position", "sellable", "fills"]}, ensure_ascii=False, indent=2))
    else:
        before = hashlib.sha256(archive.path.read_bytes()).hexdigest()
        results = audit(archive, archive.symbols if args.all else [args.symbol])
        after = hashlib.sha256(archive.path.read_bytes()).hexdigest()
        report = {"archive": archive.path.name, "sha256_before": before, "sha256_after": after,
                  "original_unchanged": before == after, "symbols": results}
        save_report(args.output or "reports/audit.json", report)
        if not all(r["healthy"] for r in results) or before != after:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
