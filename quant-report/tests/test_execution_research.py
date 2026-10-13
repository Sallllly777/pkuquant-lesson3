import unittest
from importlib.util import find_spec

if any(find_spec(name) is None for name in ["numpy", "pandas", "scipy", "matplotlib"]):
    raise unittest.SkipTest("Install requirements-research.txt for execution research tests")

import numpy as np
import pandas as pd

from l2sim.data import Dataset, Event, parse_time
from l2sim.engine import Replay
from research.execution_study import ExperimentReplay, simulate_symbol, visible_vwap, valid_resubmission


def add(t, seq, oid, side, price, qty):
    return Event(parse_time(t), 1001, seq, "ord", {"OrderKind": 65, "FunctionCode": side,
                 "OrderOriNo": oid, "Price": price, "Volume": qty})


def data():
    events = [Event(parse_time("09:15:00"), 9999, 0, "snp", {"PreClose": 10}),
              add("13:04:59", 1, 1, 66, 10, 100), add("13:04:59", 2, 2, 83, 10.01, 1000),
              add("13:04:59", 3, 3, 66, 9.99, 1000),
              Event(parse_time("13:05:01"), 1001, 4, "exe", {"BidOrder": 1, "AskOrder": 999, "BSFlag": 83, "Price": 10., "Volume": 100}),
              add("13:05:02", 5, 4, 66, 10, 200),
              Event(parse_time("13:05:03"), 1001, 6, "exe", {"BidOrder": 4, "AskOrder": 998, "BSFlag": 83, "Price": 10., "Volume": 50}),
              Event(parse_time("13:05:04"), 1001, 7, "ord", {"OrderKind": 68, "FunctionCode": 66, "OrderOriNo": 4, "Price": 10., "Volume": 150})]
    return Dataset(600036, "20260923", events, {})


class ExecutionResearchTests(unittest.TestCase):
    def test_research_wrapper_preserves_engine_matching(self):
        standard, research = Replay(data()), ExperimentReplay(data())
        for sim in [standard, research]:
            sim.advance_to(parse_time("13:05:00"))
            sim.submit("BUY", 10., 100)
            sim.advance_to(parse_time("13:05:30"))
        self.assertEqual(standard.fills, research.fills)
        self.assertEqual(standard.orders[0].filled, 50)
        self.assertEqual(research.orders[0].filled, 50)

    def test_partial_fill_remainder_cost_and_adverse_selection(self):
        signal = pd.DataFrame([{"symbol": 600036, "session": "PM", "time_ms": parse_time("13:05:00"),
                  "mid_cent": 1000.5, "spread_bps": 10, "obi1": -.8, "trade_imb_5": -.8, "score_book_flow": -.2}])
        tasks, fills = simulate_symbol(data(), signal)
        buy = tasks[tasks.side == "BUY"].iloc[0]
        self.assertTrue(buy.partial_fill)
        self.assertEqual(buy.filled_qty, 50)
        self.assertEqual(buy.maker_then_cross_cent, 1000.5)
        self.assertFalse(buy.remainder_order_lot_valid)
        self.assertTrue((fills.postfill_drift_30_bps < 0).all())
        np.testing.assert_allclose(fills.capture_at_fill_bps + fills.postfill_drift_30_bps, fills.markout_30_bps, atol=1e-10)

    def test_visible_vwap_respects_reserved_quantity(self):
        sim = ExperimentReplay(data())
        sim.advance_to(parse_time("13:05:00"))
        price, quantity = visible_vwap(sim.book, "SELL", 200)
        self.assertEqual(quantity, 200)
        self.assertEqual(price, 999.5)
        price, quantity = visible_vwap(sim.book, "SELL", 200, {(1001, 1): 100})
        self.assertEqual(price, 999)

    def test_mids_are_asof_and_reset_requires_no_live_orders(self):
        sim = ExperimentReplay(data())
        sim.advance_to(parse_time("13:05:00"))
        sim.submit("BUY", 10, 100)
        with self.assertRaises(ValueError): sim.reset_task_account()
        sim.advance_to(parse_time("13:05:30"))
        np.testing.assert_allclose(sim.mids_at([parse_time("13:05:00"), parse_time("13:05:04")]), [1000.5, 1000])
        sim.cancel(1); sim.reset_task_account()
        self.assertEqual(sim.sellable, sim.initial_position)
        self.assertEqual(len(sim.orders), 0)

    def test_remaining_quantity_not_assumed_resubmittable(self):
        self.assertFalse(valid_resubmission(688008, "BUY", 100))
        self.assertFalse(valid_resubmission(600036, "BUY", 50))
        self.assertTrue(valid_resubmission(600036, "BUY", 100))

    def test_delayed_passive_intention_can_arrive_as_taker(self):
        events = data().events[:4] + [
            Event(parse_time("13:05:00.500"), 1001, 4, "exe", {"BidOrder": 1, "AskOrder": 999, "BSFlag": 83, "Price": 10., "Volume": 100}),
            Event(parse_time("13:05:00.600"), 1001, 5, "ord", {"OrderKind": 68, "FunctionCode": 83, "OrderOriNo": 2, "Price": 10.01, "Volume": 1000}),
            add("13:05:00.600", 6, 5, 83, 10., 1000)]
        signal = pd.DataFrame([{"symbol": 600036, "session": "PM", "time_ms": parse_time("13:05:00"),
                  "mid_cent": 1000.5, "spread_bps": 10, "obi1": -.8, "trade_imb_5": -.8, "score_book_flow": -.2}])
        tasks, fills = simulate_symbol(Dataset(600036, "20260923", events, {}), signal, arrival_delay_seconds=1)
        buy = tasks[tasks.side == "BUY"].iloc[0]
        self.assertEqual(buy.limit_cent, 1000)
        self.assertEqual(buy.taker_filled_qty, 100)
        self.assertEqual(buy.deadline_seconds_from_signal, 31)
        self.assertEqual(fills.iloc[0].model, "TAKER_VISIBLE_BOOK")
        self.assertEqual(fills.iloc[0].fill_time, parse_time("13:05:01"))


if __name__ == "__main__": unittest.main()
