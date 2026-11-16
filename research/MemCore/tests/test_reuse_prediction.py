import unittest

from memcore.memory import MemoryManager


class TestReusePredictionTracking(unittest.TestCase):

    def test_access_on_twentieth_request_counts_as_reuse(self):
        manager = MemoryManager(100, 100)
        manager.insert("A", 10, reuse_probability=1.0)

        for _ in range(19):
            manager.access("OTHER")

        self.assertEqual(len(manager.reuse_prediction_observations), 0)
        self.assertEqual(len(manager.pending_reuse_predictions), 1)

        self.assertEqual(manager.access("A"), "HBM_HIT")

        self.assertEqual(
            list(manager.reuse_prediction_observations),
            [(1.0, 1.0)],
        )

    def test_cold_miss_still_counts_as_reuse(self):
        manager = MemoryManager(10, 0)
        manager.insert("B", 10, reuse_probability=1.0)
        manager.insert("C", 10, reuse_probability=0.0)

        for _ in range(19):
            manager.access("OTHER")

        self.assertEqual(manager.access("B"), "COLD_MISS")

        observations = list(manager.reuse_prediction_observations)

        self.assertIn((1.0, 1.0), observations)
        self.assertIn((0.0, 0.0), observations)

    def test_no_reuse_produces_negative_observation(self):
        manager = MemoryManager(100, 100)
        manager.insert("C", 10, reuse_probability=0.0)

        for _ in range(20):
            manager.access("OTHER")

        self.assertIn(
            (0.0, 0.0),
            manager.reuse_prediction_observations,
        )

    def test_incomplete_window_remains_pending(self):
        manager = MemoryManager(100, 100)
        manager.insert("D", 10, reuse_probability=0.9)

        for _ in range(19):
            manager.access("OTHER")

        self.assertEqual(len(manager.reuse_prediction_observations), 0)
        self.assertEqual(len(manager.pending_reuse_predictions), 1)
        self.assertIn("D", manager.pending_reuse_predictions)

    def test_reallocation_censors_previous_prediction(self):
        manager = MemoryManager(100, 100)

        manager.insert("E", 10, reuse_probability=0.9)
        manager.insert("E", 10, reuse_probability=0.1)

        self.assertEqual(manager.reuse_prediction_censored, 1)

        self.assertEqual(
            manager.pending_reuse_predictions["E"]["predicted_probability"],
            0.1,
        )


if __name__ == "__main__":
    unittest.main()