
import unittest

from memcore.memory import MemoryManager


class TestPrequentialReuse(unittest.TestCase):

    def make_manager(self):
        return MemoryManager(
            hbm_capacity_bytes=256,
            host_capacity_bytes=256,
        )

    def advance_requests(self, manager, count, block_id="OTHER"):
        for _ in range(count):
            manager._record_reuse_access(block_id)

    def test_empty_history(self):
        manager = self.make_manager()

        metrics = manager.get_prequential_reuse_metrics()

        self.assertEqual(metrics["observations"], 0)
        self.assertIsNone(metrics["model_brier"])
        self.assertIsNone(metrics["baseline_brier"])
        self.assertIsNone(metrics["excess_brier"])
        self.assertIsNone(metrics["skill_score"])

    def test_invalid_window(self):
        manager = self.make_manager()

        for value in (0, -1, True, 1.5, "10"):
            with self.subTest(window=value):
                with self.assertRaises(ValueError):
                    manager.get_prequential_reuse_metrics(
                        window=value
                    )

    def test_prequential_brier_calculation(self):
        manager = self.make_manager()

        manager._register_reuse_prediction("A", 0.9)

        self.assertEqual(
            manager.pending_reuse_predictions["A"][
                "baseline_probability"
            ],
            0.5,
        )

        manager._record_reuse_access("A")
        self.advance_requests(manager, 19)

        manager._register_reuse_prediction("B", 0.1)

        self.assertEqual(
            manager.pending_reuse_predictions["B"][
                "baseline_probability"
            ],
            1.0,
        )

        self.advance_requests(manager, 20)

        metrics = manager.get_prequential_reuse_metrics()

        self.assertEqual(metrics["observations"], 2)
        self.assertAlmostEqual(metrics["model_brier"], 0.01)
        self.assertAlmostEqual(metrics["baseline_brier"], 0.625)
        self.assertAlmostEqual(metrics["excess_brier"], -0.615)
        self.assertAlmostEqual(metrics["skill_score"], 0.984)

        recent = manager.get_prequential_reuse_metrics(
            window=1
        )

        self.assertEqual(recent["observations"], 1)
        self.assertAlmostEqual(recent["baseline_brier"], 1.0)

        manager.validate()

    def test_baseline_is_frozen_at_registration(self):
        manager = self.make_manager()

        manager._register_reuse_prediction("A", 0.7)

        manager._record_reuse_access("A")
        self.advance_requests(manager, 9)

        # B is registered before A's outcome is finalized.
        manager._register_reuse_prediction("B", 0.8)

        self.assertEqual(
            manager.pending_reuse_predictions["B"][
                "baseline_probability"
            ],
            0.5,
        )

        # A finalizes with an observed reuse outcome of 1.
        self.advance_requests(manager, 10)

        self.assertEqual(manager.get_observed_reuse_rate(), 1.0)

        # B's already-registered baseline must remain unchanged.
        self.assertEqual(
            manager.pending_reuse_predictions["B"][
                "baseline_probability"
            ],
            0.5,
        )

        self.advance_requests(manager, 10)

        history = list(
            manager.reuse_prediction_prequential_observations
        )

        self.assertEqual(len(history), 2)

        predicted, baseline, observed = history[-1]

        self.assertAlmostEqual(predicted, 0.8)
        self.assertAlmostEqual(baseline, 0.5)
        self.assertEqual(observed, 0.0)

        manager.validate()

    def test_reuse_on_final_request(self):
        manager = self.make_manager()

        manager._register_reuse_prediction("A", 0.8)

        self.advance_requests(manager, 19)
        manager._record_reuse_access("A")

        history = list(
            manager.reuse_prediction_prequential_observations
        )

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0][2], 1.0)

        metrics = manager.get_prequential_reuse_metrics()

        self.assertAlmostEqual(metrics["model_brier"], 0.04)
        self.assertAlmostEqual(metrics["baseline_brier"], 0.25)

        manager.validate()

    def test_censored_prediction_is_not_scored(self):
        manager = self.make_manager()

        manager.insert(
            "A",
            64,
            reuse_probability=0.9,
        )

        # Reallocation cancels the previous pending forecast.
        manager.insert(
            "A",
            64,
            reuse_probability=0.1,
        )

        self.advance_requests(manager, 20)

        metrics = manager.get_prequential_reuse_metrics()

        self.assertEqual(metrics["observations"], 1)
        self.assertEqual(manager.reuse_prediction_censored, 1)

        self.assertEqual(
            len(manager.reuse_prediction_observations),
            1,
        )

        manager.validate()

    def test_history_remains_bounded(self):
        manager = self.make_manager()

        for i in range(105):
            manager._register_reuse_prediction(
                f"CTX_{i}",
                0.3,
            )

            self.advance_requests(manager, 20)

        expected = manager.REUSE_OBSERVATION_WINDOW

        self.assertEqual(
            len(manager.reuse_prediction_prequential_observations),
            expected,
        )

        self.assertEqual(
            len(manager.reuse_prediction_observations),
            expected,
        )

        self.assertEqual(
            len(manager.reuse_prediction_context_observations),
            expected,
        )

        self.assertEqual(
            manager.get_prequential_reuse_metrics()[
                "observations"
            ],
            expected,
        )

        manager.validate()


if __name__ == "__main__":
    unittest.main()
