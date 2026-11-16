import unittest

from memcore.memory import MemoryManager
from memcore.scheduler import SLOAwareScheduler


class TestSchedulerIntegration(unittest.TestCase):

    def test_scheduler_ranks_memory_manager_contexts(self):
        memory = MemoryManager(2048, 4096, policy="LRU")
        scheduler = SLOAwareScheduler()

        memory.insert(
            "AGENT_A",
            512,
            deadline_ms=20.0,
            reuse_probability=0.95,
            estimated_transfer_ms=8.0,
            quality_risk=0.1
        )

        memory.insert(
            "AGENT_B",
            512,
            deadline_ms=500.0,
            reuse_probability=0.2,
            estimated_transfer_ms=1.0,
            quality_risk=0.0
        )

        memory.insert(
            "AGENT_C",
            512,
            deadline_ms=80.0,
            reuse_probability=0.7,
            estimated_transfer_ms=4.0,
            quality_risk=0.4
        )

        for _ in range(12):
            memory.access("AGENT_A")

        for _ in range(1):
            memory.access("AGENT_B")

        for _ in range(5):
            memory.access("AGENT_C")

        contexts = [
            memory.get_block_metadata(block_id)
            for block_id in memory.hbm
        ]

        ranked = scheduler.rank(contexts, now_ms=0.0)

        ranked_ids = [
            context["block_id"]
            for context in ranked
        ]

        self.assertEqual(
            ranked_ids,
            ["AGENT_A", "AGENT_C", "AGENT_B"]
        )

        self.assertTrue(memory.validate())

    def test_scheduler_does_not_modify_memory_state(self):
        memory = MemoryManager(2048, 4096)
        scheduler = SLOAwareScheduler()

        memory.insert(
            "CTX_1",
            512,
            deadline_ms=100.0,
            reuse_probability=0.9
        )

        memory.insert(
            "CTX_2",
            512,
            deadline_ms=500.0,
            reuse_probability=0.2
        )

        before = memory.report()

        contexts = [
            memory.get_block_metadata(block_id)
            for block_id in memory.hbm
        ]

        scheduler.rank(contexts, now_ms=0.0)

        after = memory.report()

        self.assertEqual(
            before["hbm_used_bytes"],
            after["hbm_used_bytes"]
        )

        self.assertEqual(
            before["host_used_bytes"],
            after["host_used_bytes"]
        )

        self.assertEqual(
            before["hbm_blocks"],
            after["hbm_blocks"]
        )

        self.assertTrue(memory.validate())


if __name__ == "__main__":
    unittest.main()