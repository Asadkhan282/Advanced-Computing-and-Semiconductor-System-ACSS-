
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memcore.simulator import run_workload
from memcore.memory import MemoryManager


# Automatically discover all robustness workloads.
WORKLOAD_DIR = PROJECT_ROOT / "workloads" / "robustness"
WORKLOADS = sorted(WORKLOAD_DIR.glob("*.json"))

POLICIES = [
    "FIFO",
    "LRU",
    "SLO",
    "SLO_MARGIN",
    "ADAPTIVE_SLO",
    "ADAPTIVE_MARGIN",
]


def percentage(value):
    return f"{value * 100:.2f}%"


def decimal_or_na(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def print_comparison(results, metrics_to_display):

    column_width = 20

    print(
        f"{'METRIC':<32}"
        + "".join(
            f"{policy:>{column_width}}"
            for policy in POLICIES
        )
    )

    print("-" * (32 + column_width * len(POLICIES)))

    for label, key, formatter in metrics_to_display:

        values = []

        for policy in POLICIES:

            value = results[policy][key]

            if formatter:
                value = formatter(value)

            values.append(str(value))

        print(
            f"{label:<32}"
            + "".join(
                f"{value:>{column_width}}"
                for value in values
            )
        )


def flatten_results(workload_results):

    flattened = {}

    for policy in POLICIES:

        result = workload_results[policy]
        memory = result["memory_metrics"]

        flattened[policy] = {
            **result,

            "hbm_evictions": memory["hbm_evictions"],
            "host_evictions": memory["host_evictions"],
            "bytes_moved_to_host": memory["bytes_moved_to_host"],
            "bytes_restored_to_hbm": memory["bytes_restored_to_hbm"],
            "bytes_dropped": memory["bytes_dropped"],

            "reuse_prediction_observations":
                memory["reuse_prediction_observations"],

            "reuse_prediction_pending":
                memory["reuse_prediction_pending"],

            "reuse_prediction_censored":
                memory["reuse_prediction_censored"],

            "reuse_prediction_mae":
                memory["reuse_prediction_mae"],

            "reuse_reliability":
                memory["reuse_reliability"],
        }

    return flattened


def aggregate_results(results):

    aggregate = {
        policy: {
            "access_requests": 0,
            "hbm_hits": 0,
            "host_hits": 0,
            "cold_misses": 0,

            "hbm_evictions": 0,
            "host_evictions": 0,
            "bytes_moved_to_host": 0,
            "bytes_restored_to_hbm": 0,
            "bytes_dropped": 0,

            "reuse_prediction_observations": 0,
            "reuse_prediction_pending": 0,
            "reuse_prediction_censored": 0,

            # Internal accumulator for observation-weighted MAE.
            "reuse_prediction_error_sum": 0.0,
        }
        for policy in POLICIES
    }

    for workload_results in results.values():

        for policy in POLICIES:

            result = workload_results[policy]
            memory = result["memory_metrics"]
            total = aggregate[policy]

            total["access_requests"] += result["access_requests"]
            total["hbm_hits"] += result["hbm_hits"]
            total["host_hits"] += result["host_hits"]
            total["cold_misses"] += result["cold_misses"]

            for key in (
                "hbm_evictions",
                "host_evictions",
                "bytes_moved_to_host",
                "bytes_restored_to_hbm",
                "bytes_dropped",
                "reuse_prediction_pending",
                "reuse_prediction_censored",
            ):
                total[key] += memory[key]

            observations = memory["reuse_prediction_observations"]
            mae = memory["reuse_prediction_mae"]

            total["reuse_prediction_observations"] += observations

            if observations and mae is not None:
                total["reuse_prediction_error_sum"] += (
                    mae * observations
                )

    for policy in POLICIES:

        data = aggregate[policy]
        accesses = data["access_requests"]

        data["hbm_hit_rate"] = (
            data["hbm_hits"] / accesses
            if accesses else 0.0
        )

        data["total_cached_hit_rate"] = (
            (data["hbm_hits"] + data["host_hits"]) / accesses
            if accesses else 0.0
        )

        observations = data["reuse_prediction_observations"]

        data["reuse_prediction_mae"] = (
            data["reuse_prediction_error_sum"] / observations
            if observations else None
        )

        # Descriptive pooled reliability, not a runtime policy score.
        if observations < MemoryManager.REUSE_MIN_OBSERVATIONS:
            data["reuse_reliability"] = 0.0
        else:
            data["reuse_reliability"] = max(
                0.0,
                1.0 - 2.0 * data["reuse_prediction_mae"],
            )

        del data["reuse_prediction_error_sum"]

    return aggregate


def run_benchmarks():

    if not WORKLOADS:
        raise FileNotFoundError(
            f"No JSON workloads found in {WORKLOAD_DIR}"
        )

    results = {}

    print("=" * 152)
    print("                         ACSS MEMCORE MULTI-WORKLOAD BENCHMARK")
    print("=" * 152)

    print(f"\nDiscovered workloads: {len(WORKLOADS)}")
    print(f"Policies: {', '.join(POLICIES)}")

    # Run every workload under every policy.
    for workload_path in WORKLOADS:

        workload_key = str(workload_path.relative_to(PROJECT_ROOT))
        results[workload_key] = {}

        print(f"\nRunning workload: {workload_path.name}")

        for policy in POLICIES:

            print(f"  Running {policy}...")

            results[workload_key][policy] = run_workload(
                workload_path=str(workload_path),
                policy=policy,
                verbose=False,
            )

    # Per-workload comparisons.
    for workload_key, workload_results in results.items():

        print("\n" + "=" * 152)
        print(
            f"WORKLOAD: "
            f"{workload_results['FIFO']['workload_name']}"
        )
        print("=" * 152)

        flattened = flatten_results(workload_results)

        rows = [
            ("Access requests", "access_requests", None),
            ("HBM hits", "hbm_hits", None),
            ("Host hits", "host_hits", None),
            ("Cold misses", "cold_misses", None),
            ("HBM hit rate", "hbm_hit_rate", percentage),
            ("Cached hit rate", "total_cached_hit_rate", percentage),

            ("HBM evictions", "hbm_evictions", None),
            ("Host evictions", "host_evictions", None),
            ("Bytes moved to host", "bytes_moved_to_host", None),
            ("Bytes restored to HBM", "bytes_restored_to_hbm", None),
            ("Bytes dropped", "bytes_dropped", None),

            ("Reuse observations", "reuse_prediction_observations", None),
            ("Reuse predictions pending", "reuse_prediction_pending", None),
            ("Reuse predictions censored", "reuse_prediction_censored", None),
            ("Reuse prediction MAE", "reuse_prediction_mae", decimal_or_na),
            ("Reuse reliability", "reuse_reliability", decimal_or_na),
        ]

        print_comparison(flattened, rows)

        best_rate = max(
            workload_results[policy]["hbm_hit_rate"]
            for policy in POLICIES
        )

        winners = [
            policy
            for policy in POLICIES
            if workload_results[policy]["hbm_hit_rate"] == best_rate
        ]

        print(
            f"\nHBM hit-rate leader: {', '.join(winners)} "
            f"({percentage(best_rate)})"
        )

    # Aggregate counters across all workloads.
    aggregate = aggregate_results(results)

    print("\n" + "=" * 152)
    print("                         AGGREGATE BENCHMARK RESULTS")
    print("=" * 152)

    aggregate_rows = [
        ("Total access requests", "access_requests", None),
        ("Total HBM hits", "hbm_hits", None),
        ("Total host hits", "host_hits", None),
        ("Total cold misses", "cold_misses", None),
        ("Aggregate HBM hit rate", "hbm_hit_rate", percentage),
        ("Aggregate cached hit rate", "total_cached_hit_rate", percentage),

        ("Total HBM evictions", "hbm_evictions", None),
        ("Total host evictions", "host_evictions", None),
        ("Bytes moved to host", "bytes_moved_to_host", None),
        ("Bytes restored to HBM", "bytes_restored_to_hbm", None),
        ("Bytes dropped", "bytes_dropped", None),

        ("Total reuse observations", "reuse_prediction_observations", None),
        ("Total predictions pending", "reuse_prediction_pending", None),
        ("Total predictions censored", "reuse_prediction_censored", None),
        ("Pooled reuse prediction MAE", "reuse_prediction_mae", decimal_or_na),
        ("Pooled descriptive reliability", "reuse_reliability", decimal_or_na),
    ]

    print_comparison(aggregate, aggregate_rows)

    print("\nAGGREGATE HBM HIT-RATE RANKING")

    ranking = sorted(
        POLICIES,
        key=lambda policy: aggregate[policy]["hbm_hit_rate"],
        reverse=True,
    )

    for position, policy in enumerate(ranking, start=1):

        rate = aggregate[policy]["hbm_hit_rate"]

        print(
            f"{position}. {policy:<18} {percentage(rate)}"
        )

    print("\nINTERPRETATION")

    print(
        "Results are aggregated using total HBM hits divided by "
        "total access requests."
    )

    print(
        "Pooled MAE is weighted by the number of completed observations."
    )

    print(
        "Pooled descriptive reliability is calculated from pooled MAE. "
        "It is not the runtime reliability used by any individual workload."
    )

    print(
        "A policy that performs well on one workload may perform "
        "worse on another."
    )

    print(
        "\nThese are deterministic synthetic simulator results, "
        "not real GPU inference benchmarks."
    )

    print("=" * 152)


if __name__ == "__main__":
    run_benchmarks()
