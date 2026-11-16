
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memcore.simulator import SUPPORTED_POLICIES, run_workload
from benchmarks.offline_hbm_oracle import solve_workload


POLICY_ORDER = (
    "LRU",
    "FIFO",
    "SLO",
    "SLO_MARGIN",
    "ADAPTIVE_SLO",
    "ADAPTIVE_MARGIN",
)


def compare_workload(workload_path: Path) -> dict:

    oracle_result = solve_workload(workload_path)
    oracle_minimum = oracle_result["optimal_hbm_misses"]

    policy_results = {}

    for policy in POLICY_ORDER:

        if policy not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Unsupported simulator policy: {policy}"
            )

        result = run_workload(
            workload_path,
            policy=policy,
            verbose=False,
        )

        # Each host hit requires restoration to HBM.
        # Each cold miss also counts as an HBM miss.
        online_hbm_misses = (
            result["host_hits"] + result["cold_misses"]
        )

        gap = online_hbm_misses - oracle_minimum

        if gap < 0:
            raise RuntimeError(
                f"Oracle consistency failure: {workload_path.name}, "
                f"{policy}: online={online_hbm_misses}, "
                f"oracle={oracle_minimum}"
            )

        policy_results[policy] = {
            "hbm_misses": online_hbm_misses,
            "gap": gap,
            "hbm_hits": result["hbm_hits"],
            "host_hits": result["host_hits"],
            "cold_misses": result["cold_misses"],
            "access_requests": result["access_requests"],
        }

    return {
        "file": workload_path.name,
        "oracle_minimum": oracle_minimum,
        "policies": policy_results,
    }


def main():

    workload_paths = sorted(
        (PROJECT_ROOT / "workloads").glob("*.json")
    )

    if not workload_paths:
        raise FileNotFoundError("No workload JSON files found.")

    results = []

    print("=" * 105)
    print("ACSS MEMCORE — ONLINE POLICIES VS HBM-MISS ORACLE")
    print("=" * 105)
    print("Objective: minimize host restorations + cold misses")
    print("")

    for workload_path in workload_paths:

        result = compare_workload(workload_path)
        results.append(result)

        print(f"Workload: {result['file']}")
        print(
            f"  Oracle minimum HBM misses: "
            f"{result['oracle_minimum']}"
        )

        for policy in POLICY_ORDER:

            metrics = result["policies"][policy]

            print(
                f"  {policy:18} "
                f"HBM misses={metrics['hbm_misses']:4} "
                f"oracle gap={metrics['gap']:4} "
                f"HBM hits={metrics['hbm_hits']:4}"
            )

        print("")

    aggregate = {
        policy: {
            "hbm_misses": 0,
            "oracle_gap": 0,
            "hbm_hits": 0,
            "host_hits": 0,
            "cold_misses": 0,
        }
        for policy in POLICY_ORDER
    }

    total_oracle_minimum = 0

    for result in results:

        total_oracle_minimum += result["oracle_minimum"]

        for policy in POLICY_ORDER:

            metrics = result["policies"][policy]

            aggregate[policy]["hbm_misses"] += metrics["hbm_misses"]
            aggregate[policy]["oracle_gap"] += metrics["gap"]
            aggregate[policy]["hbm_hits"] += metrics["hbm_hits"]
            aggregate[policy]["host_hits"] += metrics["host_hits"]
            aggregate[policy]["cold_misses"] += metrics["cold_misses"]

    print("=" * 105)
    print("AGGREGATE HBM ORACLE COMPARISON")
    print("=" * 105)

    print(f"Workloads evaluated: {len(results)}")
    print(f"Total oracle minimum: {total_oracle_minimum}")
    print("")

    print(
        f"{'Policy':20} "
        f"{'HBM misses':>12} "
        f"{'Oracle gap':>12} "
        f"{'HBM hits':>12} "
        f"{'Host hits':>12} "
        f"{'Cold misses':>12}"
    )

    print("-" * 95)

    for policy in sorted(
        POLICY_ORDER,
        key=lambda item: (
            aggregate[item]["oracle_gap"],
            aggregate[item]["hbm_misses"],
            item,
        ),
    ):

        metrics = aggregate[policy]

        print(
            f"{policy:20} "
            f"{metrics['hbm_misses']:>12} "
            f"{metrics['oracle_gap']:>12} "
            f"{metrics['hbm_hits']:>12} "
            f"{metrics['host_hits']:>12} "
            f"{metrics['cold_misses']:>12}"
        )

    print("")
    print("=" * 105)
    print("INTERPRETATION")
    print("=" * 105)
    print("HBM misses = host hits + cold misses.")
    print("Oracle gap = online HBM misses - minimum possible HBM misses.")
    print("A zero gap means the online policy matches the oracle")
    print("on the modeled total-HBM-miss objective.")
    print("")
    print("These results concern synthetic workloads and a modeled")
    print("two-tier memory system only.")
    print("They are not GPU benchmarks, latency measurements,")
    print("or real-inference performance claims.")


if __name__ == "__main__":
    main()
