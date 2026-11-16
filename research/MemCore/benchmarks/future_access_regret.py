
import json
import statistics
from pathlib import Path

from memcore.memory import MemoryManager
from memcore.simulator import METADATA_FIELDS


POLICIES = [
    ("FIFO", 0.0),
    ("LRU", 0.0),
    ("SLO", 0.0),
    ("SLO_MARGIN", 0.03),
]

WORKLOADS = [
    "workloads/agent_workload.json",
    "workloads/lru_friendly_workload.json",
    "workloads/host_pressure_workload.json",
    "workloads/host_pressure_prediction_mismatch.json",
    "workloads/near_tie_workload.json",
]


class TracingMemoryManager(MemoryManager):
    """Record HBM evictions and host-tier drops."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_event_number = 0
        self.eviction_log = []
        self.host_drop_log = []

    def _place_in_host(self, block):
        record = {
            "event_number": self.current_event_number,
            "block_id": block.block_id,
            "size_bytes": block.size_bytes,
            "access_count": block.access_count,
            "last_access_tick": block.last_access_tick,
        }

        host_before = set(self.host.keys())
        super()._place_in_host(block)
        host_after = set(self.host.keys())

        dropped_ids = host_before - host_after

        for dropped_id in sorted(dropped_ids):
            self.host_drop_log.append({
                "event_number": self.current_event_number,
                "block_id": dropped_id,
                "reason": "HOST_CAPACITY_EVICTION",
            })

        if block.block_id not in host_after:
            if block.block_id not in dropped_ids:
                self.host_drop_log.append({
                    "event_number": self.current_event_number,
                    "block_id": block.block_id,
                    "reason": "NOT_RETAINED_IN_HOST",
                })

        record["retained_in_host_at_eviction"] = (
            block.block_id in host_after
        )
        self.eviction_log.append(record)


def execute_workload(workload_path, policy, margin):
    workload_path = Path(workload_path)

    with workload_path.open("r", encoding="utf-8") as file:
        workload = json.load(file)

    if "memory" not in workload or "events" not in workload:
        raise ValueError(f"Invalid workload structure: {workload_path}")

    memory_config = workload["memory"]
    events = workload["events"]

    manager = TracingMemoryManager(
        hbm_capacity_bytes=memory_config["hbm_capacity_bytes"],
        host_capacity_bytes=memory_config["host_capacity_bytes"],
        policy=policy,
        slo_margin=margin,
    )

    access_results = {}

    for event_number, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            raise ValueError(f"Event {event_number} must be an object.")

        manager.current_event_number = event_number
        operation = event.get("operation", "").lower()

        if operation == "allocate":
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
            block_id = event["block_id"]
            outcome = manager.access(block_id)
            access_results[(event_number, block_id)] = outcome

        elif operation == "advance_time":
            manager.advance_time(event["delta_ms"])

        else:
            raise ValueError(
                f"Unsupported operation at event {event_number}: "
                f"{operation}"
            )

        manager.validate()

    # Offline annotation: inspect future events for each eviction.
    for eviction in manager.eviction_log:
        eviction_event = eviction["event_number"]
        block_id = eviction["block_id"]

        eviction.update({
            "next_access_event": None,
            "event_distance": None,
            "next_access_outcome": "NEVER_ACCESSED",
            "reallocated_before_access": False,
            "host_drop_before_access": None,
        })

        for future_event_number in range(
            eviction_event + 1,
            len(events) + 1,
        ):
            future_event = events[future_event_number - 1]
            future_operation = future_event.get("operation", "").lower()

            if future_event.get("block_id") != block_id:
                continue

            if future_operation == "allocate":
                eviction["reallocated_before_access"] = True
                eviction["next_access_outcome"] = (
                    "REALLOCATED_BEFORE_ACCESS"
                )
                break

            if future_operation == "access":
                eviction["next_access_event"] = future_event_number
                eviction["event_distance"] = (
                    future_event_number - eviction_event
                )
                eviction["next_access_outcome"] = access_results.get(
                    (future_event_number, block_id),
                    "UNKNOWN",
                )

                prior_drops = [
                    drop
                    for drop in manager.host_drop_log
                    if (
                        drop["block_id"] == block_id
                        and eviction_event <= drop["event_number"]
                        < future_event_number
                    )
                ]

                if prior_drops:
                    eviction["host_drop_before_access"] = (
                        prior_drops[0]["event_number"]
                    )
                break

    return {
        "workload": workload.get("workload_name", workload_path.name),
        "policy": policy,
        "margin": margin,
        "evictions": manager.eviction_log,
        "host_drops": manager.host_drop_log,
        "report": manager.report(),
    }


def summarize(result):
    evictions = result["evictions"]
    total = len(evictions)

    accessed = [
        item for item in evictions
        if item["next_access_event"] is not None
    ]

    never_accessed = [
        item for item in evictions
        if item["next_access_outcome"] == "NEVER_ACCESSED"
    ]

    reallocated = [
        item for item in evictions
        if item["reallocated_before_access"]
    ]

    outcomes = {
        "HBM_HIT": 0,
        "HOST_HIT": 0,
        "COLD_MISS": 0,
        "UNKNOWN": 0,
    }

    for item in accessed:
        outcome = item["next_access_outcome"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    distances = [
        item["event_distance"]
        for item in accessed
        if item["event_distance"] is not None
    ]

    reused_bytes = sum(item["size_bytes"] for item in accessed)
    evicted_bytes = sum(item["size_bytes"] for item in evictions)

    cold_miss_bytes = sum(
        item["size_bytes"]
        for item in accessed
        if item["next_access_outcome"] == "COLD_MISS"
    )

    return {
        "total": total,
        "accessed": len(accessed),
        "never": len(never_accessed),
        "reallocated": len(reallocated),
        "hbm": outcomes["HBM_HIT"],
        "host": outcomes["HOST_HIT"],
        "cold": outcomes["COLD_MISS"],
        "mean_distance": statistics.mean(distances) if distances else 0.0,
        "median_distance": statistics.median(distances) if distances else 0.0,
        "evicted_bytes": evicted_bytes,
        "reused_bytes": reused_bytes,
        "cold_miss_bytes": cold_miss_bytes,
        "host_drops": len(result["host_drops"]),
    }


def print_workload_results(workload_name, results):
    print("\n" + "=" * 145)
    print(f"WORKLOAD: {workload_name}")
    print("=" * 145)

    print(
        f"{'Policy':<16}{'Evictions':>10}{'Accessed':>10}"
        f"{'Never':>9}{'Realloc.':>10}{'HBM hit':>10}"
        f"{'Host hit':>10}{'Cold miss':>11}"
        f"{'Mean dist.':>12}{'Median':>9}{'Host drops':>12}"
    )
    print("-" * 145)

    for result in results:
        stats = summarize(result)
        label = result["policy"]
        if label == "SLO_MARGIN":
            label += " 0.03"

        print(
            f"{label:<16}{stats['total']:>10}"
            f"{stats['accessed']:>10}{stats['never']:>9}"
            f"{stats['reallocated']:>10}{stats['hbm']:>10}"
            f"{stats['host']:>10}{stats['cold']:>11}"
            f"{stats['mean_distance']:>12.2f}"
            f"{stats['median_distance']:>9.2f}"
            f"{stats['host_drops']:>12}"
        )

    print("\nBYTE-LEVEL ANALYSIS")
    print("-" * 100)
    print(
        f"{'Policy':<16}{'Evicted bytes':>18}"
        f"{'Reused bytes':>18}{'Cold-miss bytes':>20}"
    )

    for result in results:
        stats = summarize(result)
        label = result["policy"]
        if label == "SLO_MARGIN":
            label += " 0.03"

        print(
            f"{label:<16}{stats['evicted_bytes']:>18,}"
            f"{stats['reused_bytes']:>18,}"
            f"{stats['cold_miss_bytes']:>20,}"
        )

    print("\nEVICTION DETAILS")
    print("-" * 110)

    for result in results:
        label = result["policy"]
        if label == "SLO_MARGIN":
            label += " 0.03"

        print(f"\n{label}")

        if not result["evictions"]:
            print("  No HBM evictions.")
            continue

        for item in result["evictions"]:
            print(
                f"  Event {item['event_number']:>2}: "
                f"{item['block_id']:<10} "
                f"size={item['size_bytes']:>5} "
                f"next_access={str(item['next_access_event']):<5} "
                f"distance={str(item['event_distance']):<5} "
                f"outcome={item['next_access_outcome']:<28} "
                f"host_at_eviction="
                f"{item['retained_in_host_at_eviction']}"
            )

            if item["host_drop_before_access"] is not None:
                print(
                    f"             Host drop occurred at event "
                    f"{item['host_drop_before_access']}."
                )


def main():
    print("=" * 145)
    print("ACSS MEMCORE — FUTURE-ACCESS REGRET ANALYSIS")
    print("=" * 145)
    print("Policies:", POLICIES)
    print("Future access is evaluated using the remaining workload trace.")
    print("Event distance is a trace metric, not elapsed time.")
    print("A cold miss means the requested block was unavailable in both tiers.")

    for workload_path in WORKLOADS:
        results = [
            execute_workload(workload_path, policy, margin)
            for policy, margin in POLICIES
        ]
        print_workload_results(results[0]["workload"], results)

    print("\n" + "=" * 145)
    print("EXPERIMENT LIMITATIONS")
    print("=" * 145)
    print("All workloads are synthetic and deterministic.")
    print("Future accesses are used only for offline evaluation.")
    print("The live scheduler does not know future accesses.")
    print("Event distance does not represent transfer latency.")
    print("The experiment does not model GPU execution or recomputation.")
    print("No real inference performance improvement is established.")


if __name__ == "__main__":
    main()
