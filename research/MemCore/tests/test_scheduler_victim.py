import unittest

from memcore.scheduler import SLOAwareScheduler


class TestSchedulerVictimSelection(unittest.TestCase):

    def setUp(self):
        self.scheduler = SLOAwareScheduler()

    def make_context(
        self,
        block_id,
        access_count,
        last_access_tick,
        reuse_probability,
        deadline_ms=1000.0,
        estimated_transfer_ms=1.0,
        quality_risk=0.1,
    ):
        return {
            "block_id": block_id,
            "access_count": access_count,
            "last_access_tick": last_access_tick,
            "reuse_probability": reuse_probability,
            "deadline_ms": deadline_ms,
            "estimated_transfer_ms": estimated_transfer_ms,
            "quality_risk": quality_risk,
        }

    def test_zero_margin_selects_lowest_score(self):
        contexts = [
            self.make_context("CTX_A", 10, 100, 0.9),
            self.make_context("CTX_B", 0, 10, 0.1),
            self.make_context("CTX_C", 5, 50, 0.5),
        ]

        ranked = self.scheduler.rank(contexts, now_ms=0.0)

        expected_victim = ranked[-1]["block_id"]

        actual_victim = self.scheduler.select_victim(
            contexts,
            now_ms=0.0,
            margin=0.0,
        )

        self.assertEqual(actual_victim, expected_victim)

    def test_margin_uses_recency_for_near_ties(self):
        contexts = [
            self.make_context(
                "CTX_A", 0, 100, 0.10,
                deadline_ms=10000.0,
                estimated_transfer_ms=0.1,
                quality_risk=0.0,
            ),
            self.make_context(
                "CTX_B", 0, 10, 0.11,
                deadline_ms=10000.0,
                estimated_transfer_ms=0.1,
                quality_risk=0.0,
            ),
        ]

        victim = self.scheduler.select_victim(
            contexts,
            now_ms=0.0,
            margin=0.03,
        )

        self.assertEqual(victim, "CTX_B")

    def test_large_margin_uses_least_recent_candidate(self):
        contexts = [
            self.make_context("CTX_A", 10, 100, 0.9),
            self.make_context("CTX_B", 5, 50, 0.5),
            self.make_context("CTX_C", 0, 10, 0.1),
        ]

        victim = self.scheduler.select_victim(
            contexts,
            now_ms=0.0,
            margin=1.0,
        )

        self.assertEqual(victim, "CTX_C")

    def test_empty_contexts_are_rejected(self):
        with self.assertRaises(ValueError):
            self.scheduler.select_victim(
                [],
                now_ms=0.0,
            )

    def test_negative_margin_is_rejected(self):
        contexts = [
            self.make_context("CTX_A", 1, 10, 0.5),
        ]

        with self.assertRaises(ValueError):
            self.scheduler.select_victim(
                contexts,
                now_ms=0.0,
                margin=-0.01,
            )

    def test_boolean_margin_is_rejected(self):
        contexts = [
            self.make_context("CTX_A", 1, 10, 0.5),
        ]

        with self.assertRaises(ValueError):
            self.scheduler.select_victim(
                contexts,
                now_ms=0.0,
                margin=True,
            )


if __name__ == "__main__":
    unittest.main()