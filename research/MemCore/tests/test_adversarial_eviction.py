import unittest
from pathlib import Path

from memcore.simulator import run_workload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKLOAD = (
    PROJECT_ROOT
    / "workloads"
    / "oracle_adversarial.json"
)


class TestAdversarialEviction(unittest.TestCase):

    def test_lru_retains_recently_accessed_context(self):
        result = run_workload(
            WORKLOAD,
            policy="LRU",
            verbose=False,
        )

        self.assertEqual(result["hbm_hits"], 3)
        self.assertEqual(result["host_hits"], 0)
        self.assertEqual(result["cold_misses"], 0)

    def test_fifo_evicts_oldest_context(self):
        result = run_workload(
            WORKLOAD,
            policy="FIFO",
            verbose=False,
        )

        self.assertEqual(result["hbm_hits"], 2)
        self.assertEqual(result["host_hits"], 1)
        self.assertEqual(result["cold_misses"], 0)

    def test_slo_evicts_low_reuse_context(self):
        result = run_workload(
            WORKLOAD,
            policy="SLO",
            verbose=False,
        )

        self.assertEqual(result["hbm_hits"], 2)
        self.assertEqual(result["host_hits"], 1)

    def test_slo_margin_evicts_low_reuse_context(self):
        result = run_workload(
            WORKLOAD,
            policy="SLO_MARGIN",
            verbose=False,
        )

        self.assertEqual(result["hbm_hits"], 2)
        self.assertEqual(result["host_hits"], 1)

    def test_adaptive_slo_under_initial_uncertainty(self):
        result = run_workload(
            WORKLOAD,
            policy="ADAPTIVE_SLO",
            verbose=False,
        )

        self.assertEqual(result["hbm_hits"], 2)
        self.assertEqual(result["host_hits"], 1)

    def test_adaptive_margin_retains_recent_context(self):
        result = run_workload(
            WORKLOAD,
            policy="ADAPTIVE_MARGIN",
            verbose=False,
        )

        self.assertEqual(result["hbm_hits"], 3)
        self.assertEqual(result["host_hits"], 0)


if __name__ == "__main__":
    unittest.main()