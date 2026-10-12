import unittest
from l2sim.data import clock, milliseconds, parse_time, cents


class TimeTests(unittest.TestCase):
    def test_feed_clock_is_not_decimal_seconds(self):
        self.assertEqual(milliseconds(93006440), parse_time("09:30:06.440"))
        self.assertEqual(clock(milliseconds(152959070)), "15:29:59.070")
        self.assertEqual(parse_time("13:00:00") - parse_time("11:30:00"), 5400000)

    def test_bad_times_and_float32_cent_normalization(self):
        with self.assertRaises(ValueError): milliseconds(96000000)
        with self.assertRaises(ValueError): parse_time("25:00:00")
        self.assertEqual(cents(18.479999542236328), 1848)
