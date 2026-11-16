import unittest

from memcore.memory import MemoryManager


class TestSLOEviction(unittest.TestCase):

    def test_lowest_priority_context_is_evicted(self):
        memory = MemoryManager(
            1024,
            4096,
            policy="SLO",
        )

        memory.insert(
            "LOW_PRIORITY",
            512,
            deadline_ms=10000.0,
            reuse_probability=0.0,
            estimated_transfer_ms=0.0,
            quality_risk=0.0,
        )

        memory.insert(
            "HIGH_PRIORITY",
            512,
            deadline_ms=1.0,
            reuse_probability=1.0,
            estimated_transfer_ms=10.0,
            quality_risk=1.0,
        )

        memory.insert(
            "NEW_CONTEXT",
            512,
            deadline_ms=10.0,
            reuse_probability=0.9,
            estimated_transfer_ms=5.0,
            quality_risk=0.5,
        )

        self.assertIn("LOW_PRIORITY", memory.host)
        self.assertIn("HIGH_PRIORITY", memory.hbm)
        self.assertIn("NEW_CONTEXT", memory.hbm)

        self.assertTrue(memory.validate())

    def test_tie_breaking_is_deterministic(self):
        memory = MemoryManager(
            1024,
            4096,
            policy="SLO",
        )

        memory.insert("Z_CONTEXT", 512)
        memory.insert("A_CONTEXT", 512)

        memory.insert("NEW_CONTEXT", 512)

        # Identical scores: lexicographically smallest ID is evicted.
        self.assertIn("A_CONTEXT", memory.host)
        self.assertIn("Z_CONTEXT", memory.hbm)
        self.assertIn("NEW_CONTEXT", memory.hbm)

        self.assertTrue(memory.validate())

    def test_metadata_survives_slo_eviction_and_restoration(self):
        memory = MemoryManager(
            1024,
            4096,
            policy="SLO",
        )

        memory.insert(
            "RESTORE_ME",
            512,
            deadline_ms=500.0,
            reuse_probability=0.85,
            estimated_transfer_ms=3.5,
            quality_risk=0.25,
        )

        memory.insert(
            "HIGH_A",
            512,
            deadline_ms=1.0,
            reuse_probability=1.0,
            estimated_transfer_ms=10.0,
            quality_risk=1.0,
        )

        memory.insert(
            "HIGH_B",
            512,
            deadline_ms=2.0,
            reuse_probability=1.0,
            estimated_transfer_ms=10.0,
            quality_risk=1.0,
        )

        self.assertIn("RESTORE_ME", memory.host)

        result = memory.access("RESTORE_ME")

        self.assertEqual(result, "HOST_HIT")
        self.assertIn("RESTORE_ME", memory.hbm)

        metadata = memory.get_block_metadata("RESTORE_ME")

        self.assertEqual(metadata["tier"], "HBM")
        self.assertEqual(metadata["deadline_ms"], 500.0)
        self.assertEqual(metadata["reuse_probability"], 0.85)
        self.assertEqual(metadata["estimated_transfer_ms"], 3.5)
        self.assertEqual(metadata["quality_risk"], 0.25)
        self.assertEqual(metadata["access_count"], 1)

        self.assertTrue(memory.validate())

    def test_simulated_time_is_independent_of_logical_clock(self):
        memory = MemoryManager(
            1024,
            4096,
            policy="SLO",
        )

        memory.insert(
            "CTX_A",
            512,
            deadline_ms=100.0,
        )

        logical_clock_before = memory.logical_clock

        memory.advance_time(25.0)

        self.assertEqual(memory.current_time_ms, 25.0)
        self.assertEqual(
            memory.logical_clock,
            logical_clock_before,
        )

        self.assertTrue(memory.validate())

    def test_invalid_time_increment_is_rejected(self):
        memory = MemoryManager(
            1024,
            4096,
            policy="SLO",
        )

        with self.assertRaises(ValueError):
            memory.advance_time(-1.0)

        with self.assertRaises(ValueError):
            memory.advance_time(float("inf"))

        self.assertEqual(memory.current_time_ms, 0.0)
        self.assertTrue(memory.validate())


if __name__ == "__main__":
    unittest.main()