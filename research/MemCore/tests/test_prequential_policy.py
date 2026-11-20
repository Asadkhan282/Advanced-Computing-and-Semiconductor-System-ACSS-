
import unittest
from copy import deepcopy

from memcore.memory import MemoryManager


class TestPrequentialPolicy(unittest.TestCase):

    BAD = (0.9, 0.2, 0.0)
    GOOD = (0.1, 0.9, 0.0)

    def make_manager(self):
        return MemoryManager(
            hbm_capacity_bytes=128,
            host_capacity_bytes=256,
            policy="PREQUENTIAL_SLO_MARGIN",
        )

    def seed_history(self, manager, records):
        """
        Seed consistent synthetic completed histories for
        isolated guard-decision tests.
        """
        for i, (predicted, baseline, observed) in enumerate(records):
            manager.reuse_prediction_observations.append(
                (predicted, observed)
            )

            manager.reuse_prediction_context_observations.append(
                (f"TEST_{i}", predicted, observed)
            )

            manager.reuse_prediction_prequential_observations.append(
                (predicted, baseline, observed)
            )

        manager._reuse_prediction_metrics_cache = None
        manager.validate()

    def populate_hbm(self, manager):
        manager.insert("A", 64, reuse_probability=0.9)
        manager.insert("B", 64, reuse_probability=0.1)

    def test_policy_is_registered(self):
        manager = self.make_manager()

        self.assertEqual(
            manager.policy,
            "PREQUENTIAL_SLO_MARGIN",
        )
        self.assertIn(
            manager.policy,
            MemoryManager.SUPPORTED_POLICIES,
        )

    def test_cold_start_guard_is_inactive(self):
        manager = self.make_manager()

        status = manager.get_prequential_guard_status()

        self.assertFalse(status["guard_active"])
        self.assertFalse(status["failure_detected"])
        self.assertFalse(status["recovery_detected"])
        self.assertEqual(status["slow_observations"], 0)

    def test_insufficient_evidence_keeps_guard_inactive(self):
        manager = self.make_manager()

        self.seed_history(manager, [self.BAD] * 19)

        status = manager.get_prequential_guard_status()

        self.assertFalse(status["guard_active"])
        self.assertFalse(status["failure_detected"])
        self.assertEqual(status["slow_observations"], 19)

    def test_poor_predictions_activate_guard(self):
        manager = self.make_manager()

        self.seed_history(manager, [self.BAD] * 20)

        status = manager.get_prequential_guard_status()

        self.assertTrue(status["failure_detected"])
        self.assertTrue(status["guard_active"])
        self.assertFalse(status["recovery_detected"])
        self.assertGreater(
            status["slow_excess_brier"],
            manager.PREQUENTIAL_FAILURE_THRESHOLD,
        )

    def test_recent_recovery_releases_guard(self):
        manager = self.make_manager()

        # Slow window remains worse than the baseline,
        # while the fast window shows predictive recovery.
        self.seed_history(
            manager,
            [self.BAD] * 12 + [self.GOOD] * 8,
        )

        status = manager.get_prequential_guard_status()

        self.assertTrue(status["failure_detected"])
        self.assertTrue(status["recovery_detected"])
        self.assertFalse(status["guard_active"])

        self.assertGreater(
            status["slow_excess_brier"],
            manager.PREQUENTIAL_FAILURE_THRESHOLD,
        )

        self.assertLessEqual(
            status["fast_excess_brier"],
            manager.PREQUENTIAL_RECOVERY_THRESHOLD,
        )

    def test_good_predictions_do_not_activate_guard(self):
        manager = self.make_manager()

        self.seed_history(manager, [self.GOOD] * 20)

        status = manager.get_prequential_guard_status()

        self.assertFalse(status["failure_detected"])
        self.assertFalse(status["recovery_detected"])
        self.assertFalse(status["guard_active"])

    def test_inactive_victim_matches_slo_margin(self):
        manager = self.make_manager()
        self.populate_hbm(manager)

        contexts = [
            manager.get_block_metadata(block_id)
            for block_id in manager.hbm
        ]

        original = deepcopy(contexts)

        expected = manager.scheduler.select_victim(
            contexts,
            now_ms=manager.current_time_ms,
            margin=manager.slo_margin,
        )

        actual = manager._select_prequential_victim(
            contexts
        )

        self.assertEqual(actual, expected)
        self.assertEqual(contexts, original)

        manager.validate()

    def test_active_victim_uses_prevalence_fallback(self):
        manager = self.make_manager()
        self.populate_hbm(manager)

        self.seed_history(manager, [self.BAD] * 20)

        self.assertTrue(
            manager.get_prequential_guard_status()[
                "guard_active"
            ]
        )

        contexts = [
            manager.get_block_metadata(block_id)
            for block_id in manager.hbm
        ]

        original = deepcopy(contexts)
        prevalence = manager.get_observed_reuse_rate()

        adjusted = []

        for context in contexts:
            item = dict(context)
            item["reuse_probability"] = prevalence
            adjusted.append(item)

        expected = manager.scheduler.select_victim(
            adjusted,
            now_ms=manager.current_time_ms,
            margin=manager.slo_margin,
        )

        actual = manager._select_prequential_victim(
            contexts
        )

        self.assertEqual(actual, expected)
        self.assertEqual(contexts, original)

        manager.validate()

    def test_guarded_policy_executes_hbm_eviction(self):
        manager = self.make_manager()
        self.populate_hbm(manager)

        self.seed_history(manager, [self.BAD] * 20)

        self.assertTrue(
            manager.get_prequential_guard_status()[
                "guard_active"
            ]
        )

        expected_victim = manager._select_hbm_victim()

        manager.insert(
            "C",
            64,
            reuse_probability=0.8,
        )

        self.assertNotIn(expected_victim, manager.hbm)
        self.assertIn("C", manager.hbm)

        self.assertEqual(
            manager.report()["hbm_evictions"],
            1,
        )

        manager.validate()


if __name__ == "__main__":
    unittest.main()
