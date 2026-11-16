import unittest

from memcore.memory import MemoryManager


class TestContextPredictionHistory(unittest.TestCase):

    def setUp(self):
        self.manager = MemoryManager(
            hbm_capacity_bytes=100,
            host_capacity_bytes=100,
            policy="ADAPTIVE_SLO",
        )

    def test_completed_predictions_preserve_context_identity(self):
        self.manager.insert(
            "context_A",
            10,
            reuse_probability=0.8,
        )

        self.manager.insert(
            "context_B",
            10,
            reuse_probability=0.2,
        )

        # Both predictions have a 20-request observation window.
        # Access A during that window; B remains unused.
        for _ in range(20):
            self.manager.access("context_A")

        observations = list(
            self.manager.reuse_prediction_context_observations
        )

        self.assertEqual(len(observations), 2)

        self.assertEqual(
            observations[0],
            ("context_A", 0.8, 1.0),
        )

        self.assertEqual(
            observations[1],
            ("context_B", 0.2, 0.0),
        )

    def test_global_and_context_histories_stay_consistent(self):
        self.manager.insert(
            "context_A",
            10,
            reuse_probability=0.7,
        )

        for _ in range(20):
            self.manager.access("context_A")

        global_observations = list(
            self.manager.reuse_prediction_observations
        )

        context_observations = list(
            self.manager.reuse_prediction_context_observations
        )

        self.assertEqual(
            len(global_observations),
            len(context_observations),
        )

        self.assertEqual(
            global_observations[0],
            (
                context_observations[0][1],
                context_observations[0][2],
            ),
        )

    def test_context_history_is_bounded(self):
        self.assertEqual(
            self.manager.reuse_prediction_context_observations.maxlen,
            self.manager.REUSE_OBSERVATION_WINDOW,
        )

    def test_pending_prediction_contains_context_identity(self):
        self.manager.insert(
            "context_A",
            10,
            reuse_probability=0.6,
        )

        prediction = (
            self.manager.pending_reuse_predictions["context_A"]
        )

        self.assertEqual(
            prediction["block_id"],
            "context_A",
        )

    def test_incomplete_context_prediction_is_not_recorded(self):
        self.manager.insert(
            "context_A",
            10,
            reuse_probability=0.6,
        )

        for _ in range(10):
            self.manager.access("context_A")

        self.assertEqual(
            len(
                self.manager.reuse_prediction_context_observations
            ),
            0,
        )

        self.assertIn(
            "context_A",
            self.manager.pending_reuse_predictions,
        )


if __name__ == "__main__":
    unittest.main()