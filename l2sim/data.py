from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


def milliseconds(value: int) -> int:
    h, rest = divmod(int(value), 10000000)
    m, rest = divmod(rest, 100000)
    s, ms = divmod(rest, 1000)
    if h > 23 or m > 59 or s > 59:
        raise ValueError(f"Invalid feed time: {value}")
    return ((h * 60 + m) * 60 + s) * 1000 + ms


def parse_time(value: str) -> int:
    h, m, sec = value.split(":")
    from decimal import Decimal
    result = (int(h) * 3600 + int(m) * 60) * 1000 + int(Decimal(sec) * 1000)
    if not (0 <= int(h) < 24 and 0 <= int(m) < 60 and 0 <= Decimal(sec) < 60):
        raise ValueError("Invalid clock time")
    return result


def clock(value: int) -> str:
    s, ms = divmod(value, 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def cents(value: float) -> int:
    # Feed float32 prices encode cent ticks; normalize ONLY in the in-memory adapter.
    return int(round(float(value) * 100))


@dataclass(slots=True)
class Event:
    time: int
    channel: int
    seq: int
    kind: str
    row: dict


@dataclass
class Dataset:
    symbol: int
    date: str
    events: list[Event]
    counts: dict[str, int]


class ArchiveData:
    """ZIP is opened in r mode. No extraction or data cache is written."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.tables = {}
        with zipfile.ZipFile(self.path, "r") as z:
            members = {}
            dates = set()
            for name in z.namelist():
                if name.startswith("__MACOSX/"):
                    continue
                match = re.search(r"(?:^|/)(ord|exe|snp)_(\d{8})\.parquet$", name)
                if match:
                    kind, date = match.groups()
                    if kind in members:
                        raise ValueError("Expected exactly one trading day per ZIP")
                    members[kind] = name
                    dates.add(date)
            if set(members) != {"ord", "exe", "snp"} or len(dates) != 1:
                raise ValueError("ZIP must contain ord/exe/snp for one date")
            self.date = dates.pop()
            for kind, name in members.items():
                self.tables[kind] = pq.read_table(io.BytesIO(z.read(name)))
        symbol_sets = [set(pc.unique(t["Symbol"]).to_pylist()) for t in self.tables.values()]
        if any(s != symbol_sets[0] for s in symbol_sets[1:]):
            raise ValueError("Symbol coverage differs across files")
        self.symbols = sorted(symbol_sets[0])

    def load(self, symbol: int) -> Dataset:
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} is not in this archive")
        events, counts = [], {}
        for kind, table in self.tables.items():
            rows = table.filter(pc.equal(table["Symbol"], symbol)).to_pylist()
            counts[kind] = len(rows)
            for r in rows:
                events.append(Event(milliseconds(r["Time"]), int(r.get("Channel", 9999)),
                                    int(r.get("BizIndex", 0)), kind, r))
        # Single-symbol continuous feed uses one channel. Snapshots have no BizIndex,
        # so display them AFTER all tick events with the same reported timestamp.
        events.sort(key=lambda e: (e.time, e.channel, e.seq))
        last_seq = {}
        for e in events:
            if e.kind == "snp":
                continue
            if e.seq <= last_seq.get(e.channel, -1):
                raise ValueError(f"Duplicate/reversed BizIndex at {clock(e.time)}, channel {e.channel}")
            last_seq[e.channel] = e.seq
        regular = {e.channel for e in events if e.kind != "snp" and e.channel != 2301}
        if len(regular) != 1:
            raise ValueError("Multiple regular channels for one symbol: ordering needs a feed specification")
        return Dataset(symbol, self.date, events, counts)
