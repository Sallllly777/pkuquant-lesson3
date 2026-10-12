from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .data import Event, cents


@dataclass(slots=True)
class Quote:
    key: tuple[int, int]
    side: str
    price: int
    quantity: int
    priority: int


class OrderBook:
    def __init__(self):
        self.orders: dict[tuple[int, int], Quote] = {}
        self.levels: dict[str, dict[int, dict[tuple[int, int], Quote]]] = {"BUY": {}, "SELL": {}}
        self.stats = Counter()

    def remove(self, key):
        q = self.orders.pop(key, None)
        if q is not None:
            level = self.levels[q.side][q.price]
            del level[key]
            if not level:
                del self.levels[q.side][q.price]

    def reduce(self, key, quantity, required=True):
        q = self.orders.get(key)
        if q is None:
            self.stats["missing_resting_order" if required else "unpublished_aggressor"] += 1
            return
        if quantity > q.quantity:
            self.stats["overdraw"] += 1
        q.quantity -= quantity
        if q.quantity <= 0:
            self.remove(key)

    def apply(self, e: Event, priority: int):
        r = e.row
        if e.kind == "ord":
            if r["OrderKind"] == 83:
                self.stats["status_events"] += 1
                if r["FunctionCode"] not in (73, 79, 74, 67, 86):
                    self.stats["unknown_event"] += 1
                return
            key = (e.channel, r["OrderOriNo"])
            if r["OrderKind"] == 65 and r["FunctionCode"] in (66, 83):
                if key in self.orders:
                    self.stats["duplicate_add"] += 1
                    self.remove(key)
                q = Quote(key, "BUY" if r["FunctionCode"] == 66 else "SELL",
                          cents(r["Price"]), r["Volume"], priority)
                if q.price <= 0 or q.quantity <= 0:
                    self.stats["invalid_add"] += 1
                    return
                self.orders[key] = q
                self.levels[q.side].setdefault(q.price, {})[key] = q
                self.stats["adds"] += 1
            elif r["OrderKind"] == 68:
                q = self.orders.get(key)
                if q is None:
                    self.stats["missing_cancel"] += 1
                elif q.quantity != r["Volume"]:
                    self.stats["cancel_quantity_mismatch"] += 1
                self.remove(key)
                self.stats["cancels"] += 1
            else:
                self.stats["unknown_event"] += 1
        elif e.kind == "exe":
            if e.channel == 2301:
                self.stats["after_hours_trades"] += 1
                return  # Separate mechanism; never reduce the auction/continuous book.
            for field, side in [("BidOrder", "BUY"), ("AskOrder", "SELL")]:
                required = r["BSFlag"] == 0 or (side == "BUY" and r["BSFlag"] == 83) or (side == "SELL" and r["BSFlag"] == 66)
                self.reduce((e.channel, r[field]), r["Volume"], required)
            self.stats["trades"] += 1

    @property
    def healthy(self):
        return not any(self.stats[k] for k in ["missing_resting_order", "overdraw", "duplicate_add",
                                              "missing_cancel", "cancel_quantity_mismatch", "unknown_event", "invalid_add"])

    def best(self, side):
        levels = self.levels[side]
        return (max(levels) if side == "BUY" else min(levels)) if levels else None

    def quotes(self, side):
        for price in sorted(self.levels[side], reverse=side == "BUY"):
            yield from self.levels[side][price].values()

    def depth(self, side, count=10):
        return [{"price": p / 100, "quantity": sum(q.quantity for q in self.levels[side][p].values()),
                 "orders": len(self.levels[side][p])}
                for p in sorted(self.levels[side], reverse=side == "BUY")[:count]]

    def ahead(self, side, price, priority):
        return sum(q.quantity for q in self.levels[side].get(price, {}).values() if q.priority <= priority)
