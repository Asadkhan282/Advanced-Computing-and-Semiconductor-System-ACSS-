import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memcore.simulator import run_workload


POLICIES = [
    "FIFO",
    "LRU",
    "SLO",
    "SLO_MARGIN",
    "ADAPTIVE_SLO",
    "ADAPTIVE_MARGIN",
]
DEFAULT_WORKLOAD = PROJECT_ROOT / "workloads" / "agent_workload.json"


def percentage(value):
    return f"{value * 100:.2f}%"


def format_bytes(value):
    return f"{value:,}"


def compare_policies(workload_path):

    results = {}

    workload_path = Path(workload_path)

    if not workload_path.is_absolute():
        workload_path = PROJECT_ROOT / workload_path

    if not workload_path.exists():
        raise FileNotFoundError(
            f"Workload file not found: {workload_path}"
        )

    print("=" * 110)
    print("                         ACSS MEMCORE POLICY BENCHMARK")
    print("=" * 110)

    print(f"\nWorkload file: {workload_path.name}")

    for policy in POLICIES:
        print(f"\nRunning {policy} policy...")

        results[policy] = run_workload(
            workload_path=str(workload_path),
            policy=policy,
            verbose=False,
        )

    metric_width = 32
    column_width = max(
        16,
        max(len(policy) for policy in POLICIES) + 2,
    )

    table_width = metric_width + column_width * len(POLICIES)

    print("\n" + "=" * table_width)
    print("                         COMPARISON RESULTS")
    print("=" * table_width)

    header = f"{'METRIC':<{metric_width}}"

    for policy in POLICIES:
        header += f"{policy:>{column_width}}"

    print(header)
    print("-" * table_width)

    rows = [
        ("Access requests", "access_requests", None),
        ("HBM hits", "hbm_hits", None),
        ("Host hits", "host_hits", None),
        ("Cold misses", "cold_misses", None),
        ("HBM hit rate", "hbm_hit_rate", percentage),
        ("Cached hit rate", "total_cached_hit_rate", percentage),
        ("HBM evictions", "hbm_evictions", None),
        ("Host evictions", "host_evictions", None),
        ("Bytes moved to host", "bytes_moved_to_host", format_bytes),
        ("Bytes restored to HBM", "bytes_restored_to_hbm", format_bytes),
        ("Bytes dropped", "bytes_dropped", format_bytes),
    ]

    for label, key, formatter in rows:

        values = []

        for policy in POLICIES:
            result = results[policy]

            if key in result:
                value = result[key]
            else:
                value = result["memory_metrics"][key]

            if formatter:
                value = formatter(value)

            values.append(str(value))

        line = f"{label:<{metric_width}}"

        for value in values:
            line += f"{value:>{column_width}}"

        print(line)

    print("=" * table_width)

    reference = results[POLICIES[0]]

    print("\nBENCHMARK CONFIGURATION")
    print(f"Workload: {reference['workload_name']}")
    print(f"Total events: {reference['total_events']}")
    print(f"Allocations: {reference['allocations']}")
    print(f"Access requests: {reference['access_requests']}")

    print("\nHBM HIT-RATE RESULT")

    best_rate = max(
        results[policy]["hbm_hit_rate"]
        for policy in POLICIES
    )

    winners = [
        policy
        for policy in POLICIES
        if results[policy]["hbm_hit_rate"] == best_rate
    ]

    print(f"Best HBM hit rate: {percentage(best_rate)}")
    print(f"Policy: {', '.join(winners)}")

    print("\nPOLICY INTERPRETATION")

    for policy in POLICIES:
        rate = results[policy]["hbm_hit_rate"]
        print(f"{policy:<12} HBM hit rate: {percentage(rate)}")

    print("\nIMPORTANT LIMITATIONS")
    print(
        "These are deterministic synthetic simulator results. "
        "They do not establish real GPU inference performance."
    )
    print(
        "SLO_MARGIN is an experimental policy. Its results must be "
        "evaluated across multiple workload patterns before drawing conclusions."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compare MemCore memory eviction policies."
    )

    parser.add_argument(
        "--workload",
        default=str(DEFAULT_WORKLOAD),
        help="Path to a workload JSON file.",
    )

    args = parser.parse_args()

    compare_policies(args.workload)