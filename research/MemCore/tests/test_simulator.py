
import json
import tempfile
import unittest

from pathlib import Path

from memcore.simulator import run_workload


class TestSimulator(unittest.TestCase):

    def run_json_workload(self, workload):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "workload.json"

            path.write_text(
                json.dumps(workload),
                encoding="utf-8"
            )

            return run_workload(
                str(path),
                policy="SLO",
                verbose=False
            )

    def base_workload(self, events):
        return {
            "workload_name": "Simulator Test",
            "memory": {
                "hbm_capacity_bytes": 1024,
                "host_capacity_bytes": 4096
            },
            "events": events
        }

    def test_legacy_workload_without_metadata(self):
        workload = self.base_workload([
            {
                "operation": "allocate",
                "block_id": "A",
                "size_bytes": 512
            },
            {
                "operation": "access",
                "block_id": "A"
            }
        ])

        result = self.run_json_workload(workload)

        self.assertEqual(result["access_requests"], 1)
        self.assertEqual(result["hbm_hits"], 1)
        self.assertEqual(result["metadata_allocations"], 0)

    def test_metadata_is_passed_to_memory_manager(self):
        workload = self.base_workload([
            {
                "operation": "allocate",
                "block_id": "A",
                "size_bytes": 512,
                "deadline_ms": 100.0,
                "reuse_probability": 0.9,
                "estimated_transfer_ms": 2.0,
                "quality_risk": 0.7
            }
        ])

        result = self.run_json_workload(workload)

        self.assertEqual(result["metadata_allocations"], 1)

    def test_advance_time_event(self):
        workload = self.base_workload([
            {
                "operation": "advance_time",
                "delta_ms": 25.0
            }
        ])

        result = self.run_json_workload(workload)

        self.assertAlmostEqual(
            result["memory_metrics"]["current_time_ms"],
            25.0
        )

    def test_multiple_time_advances(self):
        workload = self.base_workload([
            {
                "operation": "advance_time",
                "delta_ms": 10.0
            },
            {
                "operation": "advance_time",
                "delta_ms": 15.0
            }
        ])

        result = self.run_json_workload(workload)

        self.assertAlmostEqual(
            result["memory_metrics"]["current_time_ms"],
            25.0
        )

    def test_unknown_operation_is_rejected(self):
        workload = self.base_workload([
            {
                "operation": "unknown_operation"
            }
        ])

        with self.assertRaises(ValueError):
            self.run_json_workload(workload)

    def test_invalid_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "workload.json"

            path.write_text(
                json.dumps(self.base_workload([])),
                encoding="utf-8"
            )

            with self.assertRaises(ValueError):
                run_workload(
                    str(path),
                    policy="INVALID",
                    verbose=False
                )


if __name__ == "__main__":
    unittest.main()
