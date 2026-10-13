import unittest
from importlib.util import find_spec
from unittest.mock import patch

if any(find_spec(name) is None for name in ["numpy", "pandas", "scipy"]):
    raise unittest.SkipTest("Install requirements-research.txt for final study tests")

import numpy as np
import pandas as pd

from l2sim.data import parse_time
from research.final_study import LinearModel, auc, cancellation_time, make_labels, split_for_time


class FinalStudyTests(unittest.TestCase):
    def test_time_partition_has_embargo(self):
        times = [parse_time(t) for t in ["10:49:00", "10:50:00", "10:54:00", "10:55:00", "11:24:00", "13:05:00"]]
        self.assertEqual(split_for_time(times).tolist(), ["train", "embargo", "embargo", "validation", "validation", "PM_audit"])
        self.assertLess(times[0] + 60000, times[3])  # latest fill + 30s markout

    def test_scaling_is_fitted_only_on_given_training_rows(self):
        train = pd.DataFrame({"f": [-1., 0., 1., np.nan], "target": [-1., 0., 1., 0.], "weight": 1.})
        model = LinearModel(["f"]).fit(train, "target")
        mean, std = model.mean.copy(), model.std.copy()
        predictions = model.predict(pd.DataFrame({"f": [1e9, np.nan]}))
        np.testing.assert_array_equal(model.mean, mean)
        np.testing.assert_array_equal(model.std, std)
        self.assertTrue(np.isfinite(predictions).all())
        self.assertEqual(model.median[0], 0)

    def test_labels_cannot_be_raw_features(self):
        for name in ["y_full", "ret_30s", "pred_fill", "bid_fwd_30"]:
            with self.assertRaises(ValueError): LinearModel([name])

    def test_cancellation_ignores_entry_and_uses_first_observed_trigger(self):
        states = pd.DataFrame({"age_s": [0, 5, 10, 15, 20], "pred_risk_full": [-5., 0., -.6, -9., 0.]})
        self.assertEqual(cancellation_time(states), 10)
        self.assertEqual(cancellation_time(states, 5), 15)
        self.assertEqual(cancellation_time(states, 25), 30)
        states.loc[states.age_s > 0, "pred_risk_full"] = 0
        self.assertEqual(cancellation_time(states), 30)

    def test_auc_accounts_for_ties(self):
        self.assertEqual(auc([0, 1], [.5, .5]), .5)
        self.assertEqual(auc([0, 1], [.1, .9]), 1.)
        self.assertTrue(np.isnan(auc([1, 1], [.1, .9])))

    def test_remaining_labels_exclude_already_observed_fills_and_weights_are_causal(self):
        t = parse_time("10:00:00")
        task = pd.DataFrame([{"symbol": 600036, "time_ms": t, "side": "BUY", "quantity": 100, "maker_then_cross_cent": 1000.5}])
        fills = pd.DataFrame([{"symbol": 600036, "time_ms": t, "side": "BUY", "fill_time": t + age * 1000,
                              "fill_qty": 50, "postfill_drift_30_bps": risk, "markout_30_bps": risk + 1}
                             for age, risk in [(5, 99.), (10, -2.)]])
        common = {"symbol": 600036, "time_ms": t, "side": "BUY", "sign": 1,
                  "state_mid_cent": 1000.5, "current_taker_cent": 1001., "deadline_quote_cent": 1002.,
                  "ret_30s": .1, "obi1": .1, "trade_imb_5": 0., "trade_imb_30": 0., "weighted_obi": 0.,
                  "ofi_5": 0., "momentum_5": 0., "vwap_gap": 0., "realized_vol_60": .1, "event_rate": 1.,
                  "queue_ahead": 100, "raw_buy_qty_30": 200, "raw_sell_qty_30": 100}
        states = pd.DataFrame([dict(common, state_time=t, age_s=0, remaining_qty=100, notional_so_far=0),
                               dict(common, state_time=t+5000, age_s=5, remaining_qty=50, notional_so_far=50000)])
        with patch.object(pd.DataFrame, "to_parquet"):
            result = make_labels(task, fills, states)
            one = make_labels(task, fills, states.iloc[:1])
        self.assertEqual(result.iloc[1].y_postfill, -2.)
        self.assertEqual(result.iloc[1].y_full, 1)
        self.assertEqual(result.iloc[1].y_passive_saving, 0.)
        self.assertEqual(result.iloc[0].weight, one.iloc[0].weight)
        self.assertEqual(result.weight.tolist(), [1., 1.])

    def test_entry_intercept_only_fill_model_is_empirical_probability(self):
        frame = pd.DataFrame({"age_s": [0.] * 10, "target": [1.] * 3 + [0.] * 7, "weight": 1.})
        model = LinearModel(["age_s"], "logistic").fit(frame, "target")
        np.testing.assert_allclose(model.predict(frame), .3, atol=1e-4)


if __name__ == "__main__": unittest.main()
