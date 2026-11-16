import unittest

from memcore.memory import MemoryManager


class TestAdaptivePolicies(unittest.TestCase):

    def make_manager(self, policy):
        return MemoryManager(
            hbm_capacity_bytes=100,
            host_capacity_bytes=200,
            policy=policy,
        )

    def test_adaptive_slo_policy_is_supported(self):
        manager = self.make_manager("ADAPTIVE_SLO")
        self.assertEqual(manager.policy, "ADAPTIVE_SLO")

    def test_adaptive_margin_policy_is_supported(self):
        manager = self.make_manager("ADAPTIVE_MARGIN")
        self.assertEqual(manager.policy, "ADAPTIVE_MARGIN")

    def assert_eviction_occurred(self, manager):
        # Newly inserted block C must be resident in HBM.
        self.assertIn("C", manager.hbm)

        # Exactly one block should have been evicted.
        self.assertEqual(manager.metrics["hbm_evictions"], 1)

        # Two blocks fit in HBM: 40 + 50 = 90 bytes.
        self.assertEqual(len(manager.hbm), 2)

        # The evicted block should be in host memory.
        self.assertEqual(len(manager.host), 1)

        # Verify memory accounting.
        self.assertEqual(manager.hbm_used, 90)
        self.assertEqual(manager.host_used, 30)

    def test_adaptive_slo_can_evict_under_initial_uncertainty(self):
        manager = self.make_manager("ADAPTIVE_SLO")

        manager.insert("A", 40)
        manager.insert("B", 30)

        manager.access("A")

        manager.insert("C", 50)

        self.assert_eviction_occurred(manager)

    def test_adaptive_margin_can_evict_under_initial_uncertainty(self):
        manager = self.make_manager("ADAPTIVE_MARGIN")

        manager.insert("A", 40)
        manager.insert("B", 30)

        manager.access("A")

        manager.insert("C", 50)

        self.assert_eviction_occurred(manager)

    def test_original_policies_remain_supported(self):
        for policy in ("FIFO", "LRU", "SLO", "SLO_MARGIN"):
            with self.subTest(policy=policy):
                manager = self.make_manager(policy)
                self.assertEqual(manager.policy, policy)


if __name__ == "__main__":
    unittest.main()