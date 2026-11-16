import unittest

from memcore.scheduler import (
    SLOAwareScheduler,
    SchedulerWeights,
)


class TestSLOAwareScheduler(unittest.TestCase):

    def setUp(self):
        self.scheduler = SLOAwareScheduler()

    def make_context(self, **overrides):
        metadata = {
            "block_id": "CTX_A",
            "access_count": 3,
            "deadline_ms": 100.0,
            "reuse_probability": 0.8,
            "estimated_transfer_ms": 5.0,
            "quality_risk": 0.2,
        }

        metadata.update(overrides)
        return metadata

    def test_score_is_between_zero_and_one(self):
        context = self.make_context()

        score = self.scheduler.score(context, now_ms=0.0)

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_urgent_deadline_increases_priority(self):
        context = self.make_context()

        urgent_score = self.scheduler.score(context, now_ms=99.0)
        distant_score = self.scheduler.score(context, now_ms=0.0)

        self.assertGreater(urgent_score, distant_score)

    def test_higher_reuse_increases_priority(self):
        low_reuse = self.make_context(reuse_probability=0.1)
        high_reuse = self.make_context(reuse_probability=0.9)

        self.assertGreater(
            self.scheduler.score(high_reuse, 0.0),
            self.scheduler.score(low_reuse, 0.0),
        )

    def test_higher_hotness_increases_priority(self):
        cold = self.make_context(access_count=0)
        hot = self.make_context(access_count=20)

        self.assertGreater(
            self.scheduler.score(hot, 0.0),
            self.scheduler.score(cold, 0.0),
        )

    def test_higher_transfer_cost_increases_priority(self):
        cheap = self.make_context(estimated_transfer_ms=0.1)
        expensive = self.make_context(estimated_transfer_ms=100.0)

        self.assertGreater(
            self.scheduler.score(expensive, 0.0),
            self.scheduler.score(cheap, 0.0),
        )

    def test_higher_quality_risk_increases_priority(self):
        low_risk = self.make_context(quality_risk=0.1)
        high_risk = self.make_context(quality_risk=0.9)

        self.assertGreater(
            self.scheduler.score(high_risk, 0.0),
            self.scheduler.score(low_risk, 0.0),
        )

    def test_missing_deadline_is_supported(self):
        context = self.make_context(deadline_ms=None)

        components = self.scheduler.score_components(
            context,
            now_ms=0.0,
        )

        self.assertEqual(components["deadline_urgency"], 0.0)

    def test_ranking_is_deterministic(self):
        contexts = [
            self.make_context(block_id="CTX_B"),
            self.make_context(block_id="CTX_A"),
        ]

        first = self.scheduler.rank(contexts, now_ms=0.0)
        second = self.scheduler.rank(contexts, now_ms=0.0)

        self.assertEqual(first, second)

    def test_invalid_metadata_is_rejected(self):
        context = self.make_context(reuse_probability=1.5)

        with self.assertRaises(ValueError):
            self.scheduler.score(context, now_ms=0.0)

    def test_invalid_weights_are_rejected(self):
        weights = SchedulerWeights(
            deadline=0.5,
            reuse=0.5,
            hotness=0.5,
            transfer_cost=0.0,
            quality_risk=0.0,
        )

        with self.assertRaises(ValueError):
            SLOAwareScheduler(weights=weights)


if __name__ == "__main__":
    unittest.main()