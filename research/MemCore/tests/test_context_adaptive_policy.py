
import unittest

from memcore.memory import MemoryManager


class TestContextAdaptivePolicy(unittest.TestCase):

    def setUp(self):
        self.manager = MemoryManager(
            hbm_capacity_bytes=300,
            host_capacity_bytes=300,
            policy="ADAPTIVE_SLO",
        )

        for block_id in ("context_A", "context_B", "context_C"):
            self.manager.insert(block_id, size_bytes=100)

    def add_observations(self, block_id, accurate=True):
        """
        Add ten completed observations:
        five reuse outcomes and five non-reuse outcomes.

        Accurate predictions:
            reuse     -> 0.9
            no reuse  -> 0.1

        Inverted predictions:
            reuse     -> 0.1
            no reuse  -> 0.9
        """
        for observed in (1, 0) * 5:
            if accurate:
                predicted = 0.9 if observed else 0.1
            else:
                predicted = 0.1 if observed else 0.9

            self.manager.reuse_prediction_context_observations.append(
                (block_id, predicted, observed)
            )

    def test_context_reliability_is_independent(self):
        self.add_observations("context_A", accurate=True)
        self.add_observations("context_B", accurate=False)

        reliability_a = (
            self.manager.get_context_reuse_reliability("context_A")
        )
        reliability_b = (
            self.manager.get_context_reuse_reliability("context_B")
        )

        self.assertGreater(reliability_a, 0.0)
        self.assertEqual(reliability_b, 0.0)

    def test_context_without_history_has_zero_reliability(self):
        reliability = (
            self.manager.get_context_reuse_reliability("context_C")
        )

        self.assertEqual(reliability, 0.0)

    def test_adaptive_ranking_is_deterministic(self):
        self.add_observations("context_A", accurate=True)
        self.add_observations("context_B", accurate=False)

        contexts = [
            vars(context).copy()
            for context in self.manager.hbm.values()
        ]

        first = self.manager._rank_contexts_adaptively(contexts)
        second = self.manager._rank_contexts_adaptively(contexts)

        self.assertEqual(first, second)

        ranked_ids = [item["block_id"] for item in first]

        self.assertEqual(
            set(ranked_ids),
            {"context_A", "context_B", "context_C"},
        )


if __name__ == "__main__":
    unittest.main()
