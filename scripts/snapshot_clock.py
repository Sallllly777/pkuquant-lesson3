"""Offline clock diagnosis. Never used to adjust the trading simulator's clock."""
import bisect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l2sim.data import ArchiveData, clock
from l2sim.rules import phase


def main():
    archive = ArchiveData(Path(__file__).resolve().parents[1] / "sse50_20260923.zip")
    results = []
    for symbol in [600036, 600183, 688008]:
        data = archive.load(symbol)
        trades = [e for e in data.events if e.kind == "exe" and e.channel != 2301]
        cumulative = []
        total = 0
        for e in trades:
            total += e.row["Volume"]
            cumulative.append(total)
        deltas, examples = [], []
        for e in data.events:
            if e.kind != "snp" or phase(e.time) != "CONTINUOUS":
                continue
            i = bisect.bisect_left(cumulative, e.row["AccVolume"])
            if i < len(trades) and cumulative[i] == e.row["AccVolume"]:
                dt = trades[i].time - e.time
                deltas.append(dt)
                if dt > 0 and len(examples) < 5:
                    examples.append({"snapshot_time": clock(e.time), "last_included_trade_time": clock(trades[i].time), "ahead_ms": dt})
        deltas.sort()
        results.append({"symbol": symbol, "matched_snapshots": len(deltas),
                        "snapshots_including_later_trade": sum(d > 0 for d in deltas),
                        "max_last_trade_minus_snapshot_ms": max(deltas), "examples": examples})
    report = {"interpretation": "A positive delta proves the nominal snapshot timestamp precedes a trade already included in AccVolume. This is NOT a universal clock correction. Replay matching uses tick events only.", "symbols": results}
    output = Path(__file__).resolve().parents[1] / "reports" / "snapshot_clock.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
