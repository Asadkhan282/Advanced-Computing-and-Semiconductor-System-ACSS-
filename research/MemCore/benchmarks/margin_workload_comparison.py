import argparse
import json
from pathlib import Path

from memcore.memory import MemoryManager
from memcore.simulator import METADATA_FIELDS


MARGINS = [0.00, 0.01, 0.03, 0.05, 0.10]

DEFAULT_WORKLOADS = [
    "workloads/agent_workload.json",
    "workloads/lru_friendly_workload.json",
    "workloads/host_pressure_workload.json",
    "workloads/host_pressure_prediction_mismatch.json",
    "workloads/near_tie_workload.json",
]


def execute_workload(workload_path, policy, margin):
    """Execute a workload with a specified memory policy and margin."""

    workload_path = Path(workload_path)

    with workload_path.open("r", encoding="utf-8") as file:
        workload = json.load(file)

    if "memory" not in workload or "events" not in workload:
        raise ValueError(
            f"Invalid workload structure: {workload_path}"
        )

    memory_config = workload["memory"]

    manager = MemoryManager(
        hbm_capacity_bytes=memory_config["hbm_capacity_bytes"],
        host_capacity_bytes=memory_config["host_capacity_bytes"],
        policy=policy,
        slo_margin=margin,
    )

    access_requests = 0

    for event_number, event in enumerate(workload["events"], start=1):
        if not isinstance(event, dict):
            raise ValueError(
                f"Event {event_number} must be a JSON object."
            )

        operation = event.get("operation")

        if not isinstance(operation, str):
            raise ValueError(
                f"Invalid operation in event {event_number}."
            )

        operation = operation.lower()

        if operation == "allocate":
            if "block_id" not in event or "size_bytes" not in event:
                raise ValueError(
                    f"Invalid allocation at event {event_number}."
                )

            metadata = {
                field: event[field]
                for field in METADATA_FIELDS
                if field in event
            }

            manager.insert(
                event["block_id"],
                event["size_bytes"],
                **metadata,
            )

        elif operation == "access":
            if "block_id" not in event:
                raise ValueError(
                    f"Missing block_id at event {event_number}."
                )

            manager.access(event["block_id"])
            access_requests += 1

        elif operation == "advance_time":
            if "delta_ms" not in event:
                raise ValueError(
                    f"Missing delta_ms at event {event_number}."
                )

            manager.advance_time(event["delta_ms"])

        else:
            raise ValueError(
                f"Unsupported operation '{operation}' "
                f"at event {event_number}."
            )

        manager.validate()

    report = manager.report()

    return {
        "workload": workload.get("workload_name", workload_path.name),
        "policy": policy,
        "margin": margin,
        "access_requests": access_requests,
        **report,
    }


def print_results(workload_name, results):
    print("\n" + "=" * 115)
    print(f"WORKLOAD: {workload_name}")
    print("=" * 115)

    headers = [
        "Configuration",
        "HBM hit %",
        "Cached hit %",
        "Cold misses",
        "HBM evict",
        "Host evict",
        "Moved bytes",
        "Restored bytes",
        "Dropped bytes",
    ]

    print(
        f"{headers[0]:<19}"
        f"{headers[1]:>11}"
        f"{headers[2]:>13}"
        f"{headers[3]:>13}"
        f"{headers[4]:>12}"
        f"{headers[5]:>12}"
        f"{headers[6]:>15}"
        f"{headers[7]:>16}"
        f"{headers[8]:>15}"
    )

    print("-" * 115)

    for result in results:
        if result["policy"] == "SLO":
            label = "SLO baseline"
        else:
            label = f"Margin {result['margin']:.2f}"

        print(
            f"{label:<19}"
            f"{result['hbm_hit_rate'] * 100:>10.2f}%"
            f"{result['cached_hit_rate'] * 100:>12.2f}%"
            f"{result['cold_misses']:>13}"
            f"{result['hbm_evictions']:>12}"
            f"{result['host_evictions']:>12}"
            f"{result.get('bytes_moved_to_host', 0):>15,}"
            f"{result.get('bytes_restored_to_hbm', 0):>16,}"
            f"{result.get('bytes_dropped', 0):>15,}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare SLO eviction margins across workloads."
    )

    parser.add_argument(
        "--workloads",
        nargs="+",
        default=DEFAULT_WORKLOADS,
        help="One or more JSON workload paths.",
    )

    args = parser.parse_args()

    print("=" * 115)
    print("ACSS MEMCORE — WORKLOAD-LEVEL MARGIN COMPARISON")
    print("=" * 115)
    print("Reference: standard SLO policy")
    print("Experimental margins:", MARGINS)
    print("All configurations receive identical workload events.")

    for workload_path in args.workloads:
        baseline = execute_workload(
            workload_path,
            policy="SLO",
            margin=0.0,
        )

        results = [baseline]

        for margin in MARGINS:
            result = execute_workload(
                workload_path,
                policy="SLO_MARGIN",
                margin=margin,
            )
            results.append(result)

        print_results(baseline["workload"], results)

    print("\n" + "=" * 115)
    print("EXPERIMENT LIMITATIONS")
    print("=" * 115)
    print("All workloads are synthetic and deterministic.")
    print("These are simulator outcomes, not real GPU benchmarks.")
    print("A higher HBM hit rate does not automatically mean lower")
    print("end-to-end inference latency or better SLO compliance.")
    print("No real transfer latency or recomputation cost is modeled.")


if __name__ == "__main__":
    main()