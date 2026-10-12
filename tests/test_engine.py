import unittest

from l2sim.data import Dataset, Event, parse_time
from l2sim.engine import Replay
from l2sim.rules import can_cancel, daily_limits, input_price, validate_order


def add(time, seq, oid, side, price, qty):
    return Event(parse_time(time), 1001, seq, "ord", {"OrderKind": 65, "FunctionCode": 66 if side == "BUY" else 83,
                 "OrderOriNo": oid, "Price": price, "Volume": qty})


def trade(time, seq, bid, ask, flag, price, qty):
    return Event(parse_time(time), 1001, seq, "exe", {"BidOrder": bid, "AskOrder": ask, "BSFlag": flag, "Price": price, "Volume": qty})


def delete(time, seq, oid, qty):
    return Event(parse_time(time), 1001, seq, "ord", {"OrderKind": 68, "FunctionCode": 66, "OrderOriNo": oid, "Volume": qty})


def session(extra=(), cash=100000, position=0, symbol=600036):
    events = [add("09:30:00", 1, 11, "BUY", 10, 100), add("09:30:00", 2, 22, "SELL", 10.01, 100)] + list(extra)
    e = Replay(Dataset(symbol, "20260923", events, {}), cash=cash, initial_position=position)
    e.preclose = 1000
    e.advance_to(parse_time("09:30:00"))
    return e


class MatchingTests(unittest.TestCase):
    def test_partial_taker_no_reuse_and_cancel_release(self):
        e = session()
        first = e.submit("BUY", 10.01, 200)
        self.assertEqual((first["filled"], first["remaining"]), (100, 100))
        second = e.submit("BUY", 10.01, 100)
        self.assertEqual(second["filled"], 0)
        self.assertEqual(e.book.orders[(1001, 22)].quantity, 100)
        e.cancel(1); e.cancel(2)
        self.assertEqual(e.frozen_cash, 0)
        self.assertEqual(e.cash, 10000000 - 100100)

    def test_passive_fifo_shared_trade_budget(self):
        e = session([
            trade("09:30:01", 3, 11, 99, 83, 10, 100),
            add("09:30:02", 4, 33, "BUY", 10, 1000),
            trade("09:30:03", 5, 33, 98, 83, 10, 150),
        ])
        e.submit("BUY", 10, 100); e.submit("BUY", 10, 100)
        e.advance_to(parse_time("09:30:01"))
        self.assertEqual(len(e.fills), 0)  # Last lot ahead cannot also fill us.
        e.advance_to(parse_time("09:30:03"))
        self.assertEqual([o.filled for o in e.orders], [100, 50])
        self.assertTrue(e.book.healthy)

    def test_cancel_advances_queue(self):
        e = session([delete("09:30:01", 3, 11, 100), add("09:30:02", 4, 33, "BUY", 10, 500), trade("09:30:03", 5, 33, 98, 83, 10, 100)])
        e.submit("BUY", 10, 100)
        e.advance_to(parse_time("09:30:03"))
        self.assertEqual(e.orders[0].filled, 100)

    def test_new_order_crosses_improved_quote(self):
        e = session([delete("09:30:00.001", 3, 22, 100), add("09:30:00.002", 4, 44, "SELL", 10.02, 500), add("09:30:01", 5, 33, "SELL", 10.01, 500)])
        e.advance_to(parse_time("09:30:00.002"))
        e.submit("BUY", 10.01, 100)
        e.advance_to(parse_time("09:30:01"))
        self.assertEqual(e.orders[0].filled, 100)
        self.assertEqual(e.fills[0]["model"], "MAKER_NEW_RESTING_ORDER")

    def test_t_plus_one_and_frozen_stock(self):
        e = session()
        e.submit("BUY", 10.01, 100)
        with self.assertRaisesRegex(ValueError, r"T\+1"):
            e.submit("SELL", 10, 100)
        e = session(position=100)
        e.submit("SELL", 10.02, 100)
        with self.assertRaises(ValueError): e.submit("SELL", 10.02, 100)
        e.cancel(1)
        self.assertEqual(e.frozen_shares, 0)

    def test_lunch_and_daily_expiry(self):
        e = session()
        e.submit("BUY", 9, 100)
        e.advance_to(parse_time("12:00:00"))
        with self.assertRaises(ValueError): e.submit("BUY", 9, 100)
        with self.assertRaises(ValueError): e.cancel(1)
        e.advance_to(parse_time("15:00:00"))
        self.assertEqual(e.orders[0].status, "EXPIRED")
        self.assertEqual(e.frozen_cash, 0)

    def test_invalid_tick_cash_cage_and_limits(self):
        e = session(cash=1000)
        for price in [10.001, float("nan"), float("inf"), 0]:
            with self.assertRaises(ValueError): e.submit("BUY", price, 100)
        with self.assertRaises(ValueError): e.submit("BUY", 10.5, 100)
        with self.assertRaises(ValueError): e.submit("BUY", 10.01, 100)
        with self.assertRaises(ValueError): e.submit("BUY", 8.99, 100)
        with self.assertRaises(ValueError): e.submit("BUY", 10, 101)

    def test_star_lot_and_price_cage(self):
        e = session(symbol=688981)
        with self.assertRaises(ValueError): e.submit("BUY", 10, 100)
        e.submit("BUY", 10, 201)
        self.assertEqual(daily_limits(1000, True), (800, 1200))

    def test_no_lookahead(self):
        e = session([add("09:31:00", 3, 33, "SELL", 10.01, 10000)])
        e.submit("BUY", 10.01, 500)
        self.assertEqual(e.orders[0].filled, 100)
        self.assertEqual(e.cursor, 2)
        with self.assertRaises(ValueError): e.advance_to(parse_time("09:29:59"))

    def test_missing_passive_fails_closed(self):
        e = session([trade("09:30:01", 3, 999, 888, 83, 10, 100)])
        e.submit("BUY", 10, 100)
        e.advance_to(parse_time("09:30:01"))
        self.assertFalse(e.book.healthy)
        self.assertEqual(e.orders[0].status, "DATA_ERROR")
        self.assertEqual(e.frozen_cash, 0)

    def test_auction_priority_and_cancel_lock(self):
        events = [add("09:19:00", 1, 11, "BUY", 10, 100), add("09:19:00", 2, 22, "SELL", 10, 500),
                  trade("09:25:00", 3, 11, 22, 0, 10, 100)]
        e = Replay(Dataset(600036, "20260923", events, {}))
        e.preclose = 1000
        e.advance_to(parse_time("09:19:00"))
        e.submit("BUY", 10.1, 100)
        e.advance_to(parse_time("09:20:00"))
        with self.assertRaises(ValueError): e.cancel(1)
        e.advance_to(parse_time("09:25:00"))
        self.assertEqual(e.orders[0].filled, 100)
        self.assertEqual(e.fills[0]["price"], 10)

    def test_after_hours_does_not_touch_book_or_fill(self):
        event = trade("15:05:01", 1, 11, 22, 83, 10, 100)
        event.channel = 2301
        e = session([event])
        e.submit("BUY", 10, 100)
        e.advance_to(parse_time("15:05:01"))
        self.assertEqual(e.book.orders[(1001, 11)].quantity, 100)
        self.assertEqual(e.orders[0].filled, 0)

    def test_boundaries_and_half_up(self):
        self.assertTrue(can_cancel(parse_time("09:19:59.999")))
        self.assertFalse(can_cancel(parse_time("09:20:00")))
        self.assertFalse(can_cancel(parse_time("14:57:00")))
        self.assertEqual(daily_limits(1005, False), (905, 1106))

    def test_odd_lot_can_be_sold_with_part_of_round_lots(self):
        e = session(position=299)
        e.submit("SELL", 10.02, 199)
        self.assertEqual(e.frozen_shares, 199)
        with self.assertRaises(ValueError): e.submit("SELL", 10.02, 99)
        e.submit("SELL", 10.02, 100)

    def test_price_priority_over_submission_time(self):
        e = session([delete("09:30:00.001", 3, 22, 100), add("09:30:00.002", 4, 44, "SELL", 10.03, 1000),
                     add("09:30:01", 5, 33, "SELL", 10.01, 100)])
        e.advance_to(parse_time("09:30:00.002"))
        e.submit("BUY", 10.01, 100)
        e.submit("BUY", 10.02, 100)
        e.advance_to(parse_time("09:30:01"))
        self.assertEqual([o.filled for o in e.orders], [0, 100])
        self.assertEqual(e.fills[0]["price"], 10.02)

    def test_auction_queue_does_not_fill_on_last_trade_ahead(self):
        events = [add("09:19:00", 1, 11, "BUY", 10, 100), add("09:19:00", 2, 22, "SELL", 10, 500),
                  trade("09:25:00", 3, 11, 22, 0, 10, 100)]
        e = Replay(Dataset(600036, "20260923", events, {}))
        e.preclose = 1000
        e.advance_to(parse_time("09:19:00"))
        e.submit("BUY", 10, 100)
        e.advance_to(parse_time("09:25:00"))
        self.assertEqual(e.orders[0].filled, 0)

    def test_close_auction_fills_before_day_expiry(self):
        e = session([add("14:58:00", 3, 33, "BUY", 10, 200), add("14:58:00", 4, 44, "SELL", 10, 200),
                     trade("15:00:00", 5, 33, 44, 0, 10, 200)])
        e.advance_to(parse_time("14:57:00"))
        e.submit("BUY", 10.1, 300)
        e.advance_to(parse_time("15:00:00"))
        self.assertEqual(e.orders[0].filled, 200)
        self.assertEqual(e.orders[0].status, "EXPIRED")
        self.assertEqual(e.frozen_cash, 0)

    def test_cash_and_share_conservation(self):
        e = session(position=500)
        e.submit("BUY", 10.01, 100)
        e.submit("SELL", 10, 100)
        self.assertEqual(e.sellable + e.bought_today, 500)
        self.assertEqual(e.cash, e.initial_cash - 100)
        self.assertEqual(e.frozen_cash, 0)
        self.assertEqual(e.frozen_shares, 0)

    def test_price_cage_rounds_half_up_not_floor_or_ceil(self):
        e = session()
        e.book.remove((1001, 11)); e.book.remove((1001, 22))
        e.book.apply(add("09:30:00", 3, 33, "BUY", 10.01, 100), 3)
        e.book.apply(add("09:30:00", 4, 44, "SELL", 10.29, 100), 4)
        # 10.29 * 1.02 = 10.4958 -> 10.50; 10.01 * .98 = 9.8098 -> 9.81.
        validate_order(600036, "BUY", 1050, 100, 1000, e.book, 1000, 100)
        with self.assertRaises(ValueError): validate_order(600036, "BUY", 1051, 100, 1000, e.book, 1000, 100)
        validate_order(600036, "SELL", 981, 100, 1000, e.book, 1000, 100)


if __name__ == "__main__":
    unittest.main()
