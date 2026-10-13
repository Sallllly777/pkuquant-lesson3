import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if any(find_spec(name) is None for name in ["numpy", "pandas", "scipy", "matplotlib"]):
    raise unittest.SkipTest("Install requirements-research.txt to run factor research tests")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from l2sim.book import OrderBook
from l2sim.data import Dataset, Event, parse_time
from research.pipeline import ResearchBook, build_symbol, clean, ofi_change
from research.screen import tail_direction, training_parameters, study_rows


def add(t, seq, oid, side, price, qty):
    return Event(parse_time(t), 1001, seq, "ord", {"OrderKind": 65, "FunctionCode": side,
                 "OrderOriNo": oid, "Price": price, "Volume": qty})


def trade(t, seq, ask, qty):
    return Event(parse_time(t), 1001, seq, "exe", {"BidOrder": 999, "AskOrder": ask,
                 "BSFlag": 66, "Price": 10.01, "Volume": qty})


class ResearchIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = [add("09:29:59", 1, 1, 66, 10, 500), add("09:29:59", 2, 2, 83, 10.01, 400),
                      add("09:29:59", 3, 3, 83, 10.02, 400), trade("09:30:00", 4, 2, 100),
                      trade("09:30:04.999", 5, 2, 100), trade("09:30:05", 6, 2, 50),
                      add("09:30:05", 7, 4, 83, 10.01, 50), trade("09:30:05.001", 8, 2, 150),
                      trade("09:30:05.001", 9, 4, 50), add("09:30:10", 10, 5, 66, 10, 100)]
        cls.panel, cls.stats = build_symbol(Dataset(600036, "20260923", cls.events, {}))

    def test_cached_book_matches_reference_after_every_event(self):
        reference, research = OrderBook(), ResearchBook()
        events = self.events + [Event(parse_time("09:30:11"), 1001, 11, "ord", {
            "OrderKind": 68, "FunctionCode": 66, "OrderOriNo": 1, "Volume": 500})]
        for i, e in enumerate(events):
            reference.apply(e, i); research.apply(e, i)
            for side in ["BUY", "SELL"]:
                self.assertEqual(reference.depth(side), research.depth(side))
                self.assertEqual(reference.best(side), research.best(side))
                self.assertEqual(research.totals[side], {p: sum(q.quantity for q in quotes.values())
                                                       for p, quotes in reference.levels[side].items()})
            self.assertEqual(reference.stats, research.stats)

    def test_no_future_events_in_boundary_features(self):
        r = self.panel.set_index("time_ms").loc[parse_time("09:30:05")]
        self.assertEqual(r.ask_cent, 1001)
        self.assertEqual(r.depth1, 700)  # Includes BOTH events at exactly 05.000.
        self.assertEqual(r.raw_buy_qty_5, 150)  # Excludes 00.000 and 05.001.
        self.assertEqual(r.ask_fwd_1, 1002)
        self.assertGreater(r.ret_1s, 0)
        later = self.panel.set_index("time_ms").loc[parse_time("09:30:10")]
        self.assertEqual(later.raw_buy_qty_5, 200)
        self.assertEqual(later.raw_add_buy_5, 100)

    def test_labels_and_momentum_do_not_cross_session_breaks(self):
        r = self.panel.set_index("time_ms").loc[parse_time("11:29:55")]
        self.assertTrue(np.isnan(r.ret_10s))
        r = self.panel.set_index("time_ms").loc[parse_time("13:00:05")]
        self.assertTrue(np.isnan(r.momentum_30))
        x = study_rows(self.panel)
        self.assertTrue(x.ret_300s.notna().all())

    def test_standard_ofi_direction(self):
        self.assertEqual(ofi_change((1000, 100, 1001, 200), (1000, 150, 1001, 170)), 80)
        self.assertEqual(ofi_change((1000, 100, 1001, 200), (1001, 50, 1002, 80)), 250)

    def test_clean_preserves_status_and_removes_only_exact_duplicate(self):
        rows = [{"Symbol": 600036, "Time": 93000000, "Channel": 1001, "BizIndex": 1,
                 "OrderKind": 65, "FunctionCode": 66, "OrderOriNo": 1, "Price": 10., "Volume": 100},
                {"Symbol": 600036, "Time": 92500000, "Channel": 1001, "BizIndex": 2,
                 "OrderKind": 83, "FunctionCode": 79, "OrderOriNo": 0, "Price": 0., "Volume": 0}]
        archive = SimpleNamespace(tables={"ord": pa.Table.from_pylist(rows + [rows[0]])})
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp, patch("research.pipeline.CLEAN", Path(tmp)), patch("research.pipeline.OUTPUT", Path(tmp)):
            quality = clean(archive)
            self.assertEqual(quality["ord"]["exact_duplicates_removed"], 1)
            self.assertEqual(len(pq.read_table(Path(tmp) / "status_events.parquet")), 1)
            self.assertEqual(len(pq.read_table(Path(tmp) / "orders.parquet")), 1)
            self.assertEqual(quality["ord"]["quarantined_rows"], 0)

    def test_conflicting_business_id_is_quarantined(self):
        base = {"Symbol": 600036, "Time": 93000000, "Channel": 1001, "BizIndex": 1,
                "OrderKind": 65, "FunctionCode": 66, "OrderOriNo": 1, "Price": 10., "Volume": 100}
        archive = SimpleNamespace(tables={"ord": pa.Table.from_pylist([base, dict(base, Volume=200)])})
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp, patch("research.pipeline.CLEAN", Path(tmp)), patch("research.pipeline.OUTPUT", Path(tmp)):
            with self.assertRaisesRegex(ValueError, "Quarantined"):
                clean(archive)
            self.assertEqual(len(pq.read_table(Path(tmp) / "quarantine_ord.parquet")), 2)

    def test_tied_tail_thresholds_do_not_create_signals(self):
        thresholds = pd.DataFrame({"f": [0., 0.]}, index=pd.MultiIndex.from_tuples([(1, .2), (1, .8)]))
        frame = pd.DataFrame({"symbol": [1, 1, 1], "f": [-1, 0, 1]})
        np.testing.assert_array_equal(tail_direction(frame, "f", thresholds, 1), [-1, 0, 1])

    def test_selection_and_direction_use_am_only(self):
        am = pd.DataFrame({"symbol": [1] * 5, "f": [1, 2, 3, 4, 5]})
        stock = pd.DataFrame({"split": ["AM", "PM"], "target": ["ret_30s"] * 2,
                              "factor": ["f"] * 2, "ic": [.1, -.9]})
        with patch("research.screen.FACTORS", {"f": "test"}):
            _, ordered, signs, cuts = training_parameters(am, stock)
        self.assertEqual(ordered, ["f"])
        self.assertEqual(signs, {"f": 1})
        self.assertAlmostEqual(cuts.loc[(1, .2), "f"], 1.8)


if __name__ == "__main__":
    unittest.main()
