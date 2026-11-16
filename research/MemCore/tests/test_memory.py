
import unittest

from memcore.memory import MemoryManager


class TestMemoryManager(unittest.TestCase):

    def test_hbm_hit(self):
        memory = MemoryManager(1024, 4096)

        memory.insert("A", 512)

        result = memory.access("A")

        self.assertEqual(result, "HBM_HIT")
        self.assertEqual(memory.metrics["hbm_hits"], 1)
        self.assertTrue(memory.validate())

    def test_lru_eviction(self):
        memory = MemoryManager(1024, 4096, policy="LRU")

        memory.insert("A", 512)
        memory.insert("B", 512)

        memory.access("A")
        memory.insert("C", 512)

        self.assertIn("A", memory.hbm)
        self.assertIn("B", memory.host)
        self.assertIn("C", memory.hbm)

        self.assertTrue(memory.validate())

    def test_host_restoration(self):
        memory = MemoryManager(1024, 4096)

        memory.insert("A", 512)
        memory.insert("B", 512)
        memory.insert("C", 512)

        self.assertIn("A", memory.host)

        result = memory.access("A")

        self.assertEqual(result, "HOST_HIT")
        self.assertIn("A", memory.hbm)
        self.assertNotIn("A", memory.host)

        self.assertTrue(memory.validate())

    def test_cold_miss(self):
        memory = MemoryManager(1024, 4096)

        result = memory.access("UNKNOWN")

        self.assertEqual(result, "COLD_MISS")
        self.assertEqual(memory.metrics["cold_misses"], 1)

        self.assertTrue(memory.validate())

    def test_lru_policy(self):
        memory = MemoryManager(1024, 4096, policy="LRU")

        memory.insert("A", 512)
        memory.insert("B", 512)

        memory.access("A")
        memory.insert("C", 512)

        self.assertIn("A", memory.hbm)
        self.assertIn("B", memory.host)
        self.assertIn("C", memory.hbm)

        self.assertTrue(memory.validate())

    def test_fifo_policy(self):
        memory = MemoryManager(1024, 4096, policy="FIFO")

        memory.insert("A", 512)
        memory.insert("B", 512)

        memory.access("A")
        memory.insert("C", 512)

        self.assertIn("A", memory.host)
        self.assertIn("B", memory.hbm)
        self.assertIn("C", memory.hbm)

        self.assertTrue(memory.validate())

    # ------------------------------------------
    # CONTEXT METADATA TESTS
    # ------------------------------------------

    def test_context_metadata(self):
        memory = MemoryManager(1024, 4096)

        memory.insert(
            "CTX_A",
            512,
            deadline_ms=100.0,
            reuse_probability=0.9,
            estimated_transfer_ms=2.5,
            quality_risk=0.1
        )

        metadata = memory.get_block_metadata("CTX_A")

        self.assertEqual(metadata["deadline_ms"], 100.0)
        self.assertEqual(metadata["reuse_probability"], 0.9)
        self.assertEqual(metadata["estimated_transfer_ms"], 2.5)
        self.assertEqual(metadata["quality_risk"], 0.1)

        self.assertEqual(metadata["tier"], "HBM")
        self.assertTrue(memory.validate())

    def test_metadata_survives_eviction_and_restoration(self):
        memory = MemoryManager(1024, 4096, policy="LRU")

        memory.insert(
            "CTX_A",
            512,
            deadline_ms=200.0,
            reuse_probability=0.95,
            estimated_transfer_ms=3.0,
            quality_risk=0.05
        )

        memory.insert("CTX_B", 512)
        memory.access("CTX_B")
        memory.insert("CTX_C", 512)

        self.assertIn("CTX_A", memory.host)

        metadata_before = memory.get_block_metadata("CTX_A")

        self.assertEqual(metadata_before["tier"], "HOST")
        self.assertEqual(metadata_before["reuse_probability"], 0.95)

        result = memory.access("CTX_A")

        self.assertEqual(result, "HOST_HIT")

        metadata_after = memory.get_block_metadata("CTX_A")

        self.assertEqual(metadata_after["tier"], "HBM")
        self.assertEqual(metadata_after["deadline_ms"], 200.0)
        self.assertEqual(metadata_after["reuse_probability"], 0.95)
        self.assertEqual(metadata_after["quality_risk"], 0.05)
        self.assertEqual(metadata_after["access_count"], 1)

        self.assertTrue(memory.validate())

    def test_invalid_reuse_probability(self):
        memory = MemoryManager(1024, 4096)

        with self.assertRaises(ValueError):
            memory.insert(
                "INVALID_CTX",
                512,
                reuse_probability=1.5
            )

        self.assertNotIn("INVALID_CTX", memory.hbm)
        self.assertNotIn("INVALID_CTX", memory.host)

        self.assertTrue(memory.validate())


if __name__ == "__main__":
    unittest.main()
