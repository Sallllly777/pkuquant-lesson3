from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from .book import OrderBook
from .data import Dataset, cents, clock
from .rules import CLOSE_END, OPEN_END, can_cancel, daily_limits, input_price, phase, validate_order


@dataclass
class PaperOrder:
    id: int
    side: str
    price: int
    quantity: int
    remaining: int
    time: int
    priority: int
    status: str = "OPEN"
    filled: int = 0
    notional: int = 0


class Replay:
    """One symbol/day. Deterministic, zero-latency, no-market-impact paper account.

    The reference book is driven by the feed, never re-matched. Synthetic orders
    share a per-event execution budget; taking liquidity reserves its order ID so
    a second synthetic order cannot re-use it. Passive tape fills require proof
    that historical orders with higher priority have left the queue.
    """

    def __init__(self, data: Dataset, cash=1_000_000, initial_position=0):
        if not isinstance(initial_position, int) or initial_position < 0:
            raise ValueError("Initial position must be a nonnegative integer")
        self.data = data
        self.book = OrderBook()
        self.cursor = 0
        self.time = data.events[0].time if data.events else 0
        self.preclose = 0
        self.last_price = 0
        self.snapshot = None
        self.cash = input_price(cash)
        self.initial_cash = self.cash
        self.initial_position = initial_position
        self.sellable = initial_position
        self.bought_today = 0
        self.frozen_cash = 0
        self.frozen_shares = 0
        self.orders: list[PaperOrder] = []
        self.fills: list[dict] = []
        self.consumed = Counter()
        self.audit = Counter()
        self.snapshot_examples = []
        self.recent_trades = []

    def active(self):
        return [o for o in self.orders if o.remaining and o.status in ("OPEN", "PARTIAL")]

    def queue_ahead(self, o):
        # All better prices, then earlier real and synthetic orders at the same price.
        quantity = 0
        for p, level in self.book.levels[o.side].items():
            better = p > o.price if o.side == "BUY" else p < o.price
            if better:
                quantity += sum(q.quantity for q in level.values())
            elif p == o.price:
                quantity += sum(q.quantity for q in level.values() if q.priority <= o.priority)
        quantity += sum(x.remaining for x in self.active() if x.side == o.side and x.id < o.id and x.price == o.price)
        return quantity

    def _fill(self, o, quantity, price, model, seq=None):
        quantity = min(quantity, o.remaining)
        if quantity <= 0:
            return
        if o.side == "BUY":
            self.frozen_cash -= o.price * quantity
            self.cash -= price * quantity
            self.bought_today += quantity
        else:
            self.frozen_shares -= quantity
            self.sellable -= quantity
            self.cash += price * quantity
        o.remaining -= quantity
        o.filled += quantity
        o.notional += price * quantity
        o.status = "FILLED" if not o.remaining else "PARTIAL"
        self.fills.append({"order_id": o.id, "time": clock(self.time), "side": o.side,
                           "price": price / 100, "quantity": quantity, "model": model,
                           "source_biz_index": seq})

    def _release(self, o, status):
        if o.side == "BUY":
            self.frozen_cash -= o.price * o.remaining
        else:
            self.frozen_shares -= o.remaining
        o.status = status

    def submit(self, side, price, quantity):
        side = str(side).upper()
        current_phase = phase(self.time)
        if current_phase not in ("CONTINUOUS", "OPEN_AUCTION", "CLOSE_AUCTION"):
            raise ValueError("This session accepts auction/continuous limit orders only; no submissions in breaks or after hours")
        if not self.book.healthy:
            raise ValueError("Book integrity failed; paper trading disabled")
        if current_phase == "CONTINUOUS" and self.book.best("BUY") and self.book.best("SELL") and self.book.best("BUY") >= self.book.best("SELL"):
            raise ValueError("Reference book is crossed; wait for a consistent feed boundary")
        p = input_price(price)
        available_shares = self.sellable - self.frozen_shares
        validate_order(self.data.symbol, side, p, quantity, self.preclose, self.book,
                       self.last_price, available_shares, current_phase == "CONTINUOUS")
        for old in self.active():
            if old.side != side and (p >= old.price if side == "BUY" else p <= old.price):
                raise ValueError("Crossing your own outstanding order is disabled in this simulator")
        if side == "BUY":
            if p * quantity > self.cash - self.frozen_cash:
                raise ValueError("Insufficient available cash")
            self.frozen_cash += p * quantity
        else:
            if quantity > available_shares:
                raise ValueError("Insufficient sellable shares (T+1: today's purchases cannot be sold)")
            self.frozen_shares += quantity
        o = PaperOrder(len(self.orders) + 1, side, p, quantity, quantity, self.time, self.cursor)
        self.orders.append(o)
        if current_phase == "CONTINUOUS":
            opposite = "SELL" if side == "BUY" else "BUY"
            for q in self.book.quotes(opposite):
                if not o.remaining or (q.price > p if side == "BUY" else q.price < p):
                    break
                available = max(0, q.quantity - self.consumed[q.key])
                taken = min(o.remaining, available)
                self._fill(o, taken, q.price, "TAKER_VISIBLE_BOOK")
                self.consumed[q.key] += taken
        return self.order_view(o)

    def cancel(self, order_id):
        if not can_cancel(self.time):
            raise ValueError("Cancellation is unavailable in this trading phase")
        o = next((o for o in self.active() if o.id == order_id), None)
        if o is None:
            raise ValueError("No cancellable order with this ID")
        self._release(o, "CANCELLED")
        return self.order_view(o)

    def _tape_candidates(self, e):
        r = e.row
        if e.channel == 2301:
            return []
        auction = r["BSFlag"] == 0 and (e.time == OPEN_END or e.time == CLOSE_END)
        if not auction and (phase(e.time) != "CONTINUOUS" or r["BSFlag"] not in (66, 83)):
            return []
        trade_price = cents(r["Price"])
        candidates = []
        for o in self.active():
            passive_side = "SELL" if r["BSFlag"] == 66 else "BUY"
            if not auction and o.side != passive_side:
                continue
            if (o.price < trade_price if o.side == "BUY" else o.price > trade_price):
                continue
            ref_key = (e.channel, r["BidOrder" if o.side == "BUY" else "AskOrder"])
            resting = self.book.orders.get(ref_key)
            if resting is None:
                continue
            better = o.price > resting.price if o.side == "BUY" else o.price < resting.price
            older = o.price == resting.price and o.priority < resting.priority
            # Use the queue BEFORE subtracting this trade. The last trade consuming
            # the last share ahead does not also fill the synthetic order.
            # Exclude synthetic predecessors here: shared budgets below enforce FIFO.
            ahead = self.queue_ahead(o) - sum(x.remaining for x in self.active() if x.side == o.side and x.id < o.id and x.price == o.price)
            if (better or older) and ahead == 0:
                candidates.append((o, trade_price if auction else o.price, "AUCTION_REFERENCE_PRICE" if auction else "MAKER_TAPE_QUEUE"))
        return sorted(candidates, key=lambda item: (item[0].side, -item[0].price if item[0].side == "BUY" else item[0].price, item[0].id))

    def _process(self, e):
        self.time = e.time
        self.cursor += 1
        if e.kind == "snp":
            self.snapshot = e.row
            self.preclose = cents(e.row["PreClose"])
            if phase(e.time) == "CONTINUOUS":
                actual = (self.book.best("BUY") or 0, self.book.best("SELL") or 0)
                expected = (cents(e.row["BidPrice1"]), cents(e.row["AskPrice1"]))
                self.audit["continuous_snapshots"] += 1
                if actual == expected:
                    self.audit["best_price_matches"] += 1
                    depths = (self.book.depth("BUY", 10), self.book.depth("SELL", 10))
                    exact = all(len(depth) == 10 and all(cents(e.row[f"{prefix}Price{i + 1}"]) == cents(level["price"]) and int(e.row[f"{prefix}Volume{i + 1}"]) == level["quantity"] for i, level in enumerate(depth)) for prefix, depth in zip(("Bid", "Ask"), depths))
                    if exact:
                        self.audit["ten_level_matches"] += 1
                elif len(self.snapshot_examples) < 5:
                    self.snapshot_examples.append({"time": clock(e.time), "book": actual, "snapshot": expected})
            return
        candidates = self._tape_candidates(e) if e.kind == "exe" and self.book.healthy else []
        # Drop shadow reservations when their historical order disappears.
        relevant_keys = []
        if e.kind == "exe":
            relevant_keys = [(e.channel, e.row[f]) for f in ("BidOrder", "AskOrder")]
        elif e.kind == "ord" and e.row["OrderKind"] == 68:
            relevant_keys = [(e.channel, e.row["OrderOriNo"])]
        self.book.apply(e, self.cursor)
        for key in relevant_keys:
            if key not in self.book.orders:
                self.consumed.pop(key, None)
        if e.kind == "exe":
            self.last_price = cents(e.row["Price"])
            self.recent_trades.append({"time": clock(e.time), "price": self.last_price / 100, "quantity": e.row["Volume"], "flag": e.row["BSFlag"], "channel": e.channel})
            self.recent_trades = self.recent_trades[-30:]
            if self.book.healthy:
                budgets = {"BUY": e.row["Volume"], "SELL": e.row["Volume"]}
                for o, price, model in candidates:
                    qty = min(o.remaining, budgets[o.side])
                    self._fill(o, qty, price, model, e.seq)
                    budgets[o.side] -= qty
        elif e.kind == "ord" and e.row["OrderKind"] == 65 and self.book.healthy and phase(e.time) == "CONTINUOUS":
            q = self.book.orders.get((e.channel, e.row["OrderOriNo"]))
            if q:
                candidates = sorted([o for o in self.active() if o.side != q.side], key=lambda o: (-o.price if o.side == "BUY" else o.price, o.id))
                for o in candidates:
                    crosses = o.price >= q.price if o.side == "BUY" else o.price <= q.price
                    if crosses:
                        qty = min(o.remaining, max(0, q.quantity - self.consumed[q.key]))
                        self._fill(o, qty, o.price, "MAKER_NEW_RESTING_ORDER", e.seq)
                        self.consumed[q.key] += qty
        if not self.book.healthy:
            for o in self.active():
                self._release(o, "DATA_ERROR")

    def _expire(self):
        if self.time >= CLOSE_END:
            for o in self.active():
                self._release(o, "EXPIRED")

    def advance_to(self, target):
        if target < self.time:
            raise ValueError("Replay cannot rewind an account; reset the session first")
        while self.cursor < len(self.data.events) and self.data.events[self.cursor].time <= target:
            self._process(self.data.events[self.cursor])
        self.time = target
        self._expire()

    def step(self, groups=1):
        if not isinstance(groups, int) or not 1 <= groups <= 100000:
            raise ValueError("Step count must be 1..100000 timestamp groups")
        for _ in range(groups):
            if self.cursor == len(self.data.events):
                break
            self.advance_to(self.data.events[self.cursor].time)

    def order_view(self, o):
        return {**asdict(o), "price": o.price / 100, "time": clock(o.time),
                "average_price": round(o.notional / o.filled / 100, 6) if o.filled else None,
                "queue_ahead": self.queue_ahead(o) if o in self.active() else None}

    def state(self):
        mark = self.last_price or self.preclose
        position = self.sellable + self.bought_today
        low, high = daily_limits(self.preclose, str(self.data.symbol).startswith("688")) if self.preclose else (0, 0)
        return {"symbol": self.data.symbol, "date": self.data.date, "time": clock(self.time), "time_ms": self.time,
                "phase": phase(self.time), "cursor": self.cursor, "event_count": len(self.data.events),
                "done": self.cursor == len(self.data.events), "healthy": self.book.healthy,
                "last_price": mark / 100, "preclose": self.preclose / 100, "limits": [low / 100, high / 100],
                "bids": self.book.depth("BUY"), "asks": self.book.depth("SELL"),
                "cash": self.cash / 100, "available_cash": (self.cash - self.frozen_cash) / 100,
                "frozen_cash": self.frozen_cash / 100, "position": position,
                "sellable": self.sellable - self.frozen_shares, "bought_today": self.bought_today,
                "frozen_shares": self.frozen_shares, "equity": (self.cash + position * mark) / 100,
                "pnl_before_fees": (self.cash + position * mark - self.initial_cash - self.initial_position * self.preclose) / 100,
                "orders": [self.order_view(o) for o in self.orders], "fills": self.fills,
                "recent_trades": self.recent_trades, "diagnostics": dict(self.book.stats),
                "snapshot_audit": dict(self.audit), "snapshot_mismatch_examples": self.snapshot_examples}
