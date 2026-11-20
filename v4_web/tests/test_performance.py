import unittest

from performance import record_request_timing


class PerformanceTests(unittest.TestCase):
    def test_request_timing_records_totals_maximum_and_slow_requests(self):
        metrics = {
            "duration_ms_total": 0.0,
            "duration_ms_max": 0.0,
            "slow_requests": 0,
        }
        record_request_timing(metrics, 0.25)
        record_request_timing(metrics, 1.1)
        self.assertAlmostEqual(metrics["duration_ms_total"], 1350.0)
        self.assertEqual(metrics["duration_ms_max"], 1100.0)
        self.assertEqual(metrics["slow_requests"], 1)

    def test_request_timing_does_not_record_negative_elapsed_time(self):
        metrics = {
            "duration_ms_total": 10.0,
            "duration_ms_max": 10.0,
            "slow_requests": 0,
        }
        record_request_timing(metrics, -1)
        self.assertEqual(metrics["duration_ms_total"], 10.0)
        self.assertEqual(metrics["duration_ms_max"], 10.0)


if __name__ == "__main__":
    unittest.main()
