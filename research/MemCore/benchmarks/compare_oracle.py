
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memcore.simulator import SUPPORTED_POLICIES, run_workload
from benchmarks.offline_oracle import solve_workload


POLICY_ORDER = (
    "LRU",
    "FIFO",
    "SLO",
    "SLO_MARGIN",
    "ADAPTIVE_SLO",
    "ADAPTIVE_MARGIN",
)


def compare_workload(workload_path: Path) -> dict:
    """
    Compare every online policy against the exact offline oracle
    using the same workload and cold-miss objective.
    """

    oracle_result = solve_workload(workload_path)
    oracle_cold_misses = oracle_result["optimal_cold_misses"]

    policy_results = {}

    for policy in POLICY_ORDER:

        if policy not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy {policy} is not supported by the simulator."
            )

        online_result = run_workload(
            workload_path,
            policy=policy,
            verbose=False,
        )

        online_cold_misses = online_result["cold_misses"]

        gap = online_cold_misses - oracle_cold_misses

        if gap < 0:
            raise RuntimeError(
                f"Oracle consistency failure for {workload_path.name}, "
                f"{policy}: online cold misses "
                f"({online_cold_misses}) are below the oracle "
                f"minimum ({oracle_cold_misses})."
            )

        policy_results[policy] = {
            "cold_misses": online_cold_misses,
            "gap": gap,
            "hbm_hits": online_result["hbm_hits"],
            "host_hits": online_result["host_hits"],
            "access_requests": online_result["access_requests"],
        }

    return {
        "workload": oracle_result["workload"],
        "file": workload_path.name,
        "events": oracle_result["event_count"],
        "oracle_cold_misses": oracle_cold_misses,
        "policies": policy_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare MemCore online eviction policies "
            "against the exact offline cold-miss oracle."
        )
    )

    parser.add_argument(
        "--workloads",
        nargs="*",
        help="Optional workload JSON paths.",
    )

    args = parser.parse_args()

    if args.workloads:
        workload_paths = [
            Path(path).resolve()
            for path in args.workloads
        ]
    else:
        workload_paths = sorted(
            (PROJECT_ROOT / "workloads").glob("*.json")
        )

    if not workload_paths:
        raise FileNotFoundError(
            "No workload JSON files were found."
        )

    print("=" * 100)
    print("ACSS MEMCORE — ONLINE POLICY VS EXACT OFFLINE ORACLE")
    print("=" * 100)
    print("Objective: minimize total cold misses")
    print("Online policies: six supported MemCore eviction policies")
    print("Oracle: exact future-aware HBM victim selection")
    print("Host behavior: FIFO in both models")
    print("")

    results = []

    for workload_path in workload_paths:

        print(f"Evaluating: {workload_path.name}")

        result = compare_workload(workload_path)
        results.append(result)

        print(f"  Oracle minimum cold misses: {result['oracle_cold_misses']}")

        for policy in POLICY_ORDER:

            metrics = result["policies"][policy]

            print(
                f"  {policy:18} "
                f"cold misses={metrics['cold_misses']:4} "
                f"oracle gap={metrics['gap']:4}"
            )

        print("")

    # Aggregate online-policy performance across workloads.
    aggregate = {
        policy: {
            "cold_misses": 0,
            "oracle_gap": 0,
            "hbm_hits": 0,
            "host_hits": 0,
            "access_requests": 0,
        }
        for policy in POLICY_ORDER
    }

    total_oracle_cold_misses = 0

    for result in results:

        total_oracle_cold_misses += result["oracle_cold_misses"]

        for policy in POLICY_ORDER:

            metrics = result["policies"][policy]

            for key in aggregate[policy]:
                aggregate[policy][key] += (
                    metrics[
                        "gap" if key == "oracle_gap"
                        else key
                    ]
                )

    print("=" * 100)
    print("AGGREGATE ORACLE COMPARISON")
    print("=" * 100)

    print(f"Workloads evaluated: {len(results)}")
    print(f"Total oracle cold-miss minimum: {total_oracle_cold_misses}")
    print("")

    print(
        f"{'Policy':20} "
        f"{'Cold misses':>14} "
        f"{'Oracle gap':>14} "
        f"{'HBM hits':>12} "
        f"{'Host hits':>12}"
    )

    print("-" * 78)

    for policy in sorted(
        POLICY_ORDER,
        key=lambda item: (
            aggregate[item]["oracle_gap"],
            aggregate[item]["cold_misses"],
            item,
        ),
    ):

        metrics = aggregate[policy]

        print(
            f"{policy:20} "
            f"{metrics['cold_misses']:>14} "
            f"{metrics['oracle_gap']:>14} "
            f"{metrics['hbm_hits']:>12} "
            f"{metrics['host_hits']:>12}"
        )

    best_gap = min(
        aggregate[policy]["oracle_gap"]
        for policy in POLICY_ORDER
    )

    best_policies = [
        policy
        for policy in POLICY_ORDER
        if aggregate[policy]["oracle_gap"] == best_gap
    ]

    print("")
    print(f"Smallest aggregate oracle gap: {best_gap}")
    print(f"Closest online policy/policies: {', '.join(best_policies)}")

    print("")
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print(
        "Oracle gap = online cold misses - minimum possible cold misses."
    )
    print(
        "A zero gap means the online policy matches the oracle "
        "on the cold-miss objective for that workload."
    )
    print(
        "A positive gap represents additional cold misses "
        "relative to the offline lower bound."
    )
    print("")
    print(
        "IMPORTANT: These results are limited to the supplied "
        "synthetic workloads and the modeled two-tier memory system."
    )
    print(
        "They are not GPU benchmarks, latency measurements, "
        "or real-inference performance claims."
    )


if __name__ == "__main__":
    main()
