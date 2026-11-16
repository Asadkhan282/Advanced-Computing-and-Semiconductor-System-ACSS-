
import json

from memcore.memory import MemoryManager


SUPPORTED_POLICIES = {
    "FIFO",
    "LRU",
    "SLO",
    "SLO_MARGIN",
    "ADAPTIVE_SLO",
    "ADAPTIVE_MARGIN",
}

METADATA_FIELDS = (
    "deadline_ms",
    "reuse_probability",
    "estimated_transfer_ms",
    "quality_risk",
)


def _format_memory_state(manager, memory_config):
    """Format the current HBM and host memory state."""

    hbm_blocks = list(manager.hbm.keys())
    host_blocks = list(manager.host.keys())

    metrics = manager.metrics

    hbm_capacity = memory_config["hbm_capacity_bytes"]
    host_capacity = memory_config["host_capacity_bytes"]

    lines = [
        (
            f"    HBM  : {hbm_blocks} "
            f"({manager.hbm_used}/{hbm_capacity} bytes)"
        ),
        (
            f"    HOST : {host_blocks} "
            f"({manager.host_used}/{host_capacity} bytes)"
        ),
        (
            f"    Evictions: "
            f"HBM={metrics['hbm_evictions']}, "
            f"HOST={metrics['host_evictions']}, "
            f"Dropped={metrics['bytes_dropped']} bytes"
        ),
    ]

    return lines


def run_workload(
    workload_path,
    policy="LRU",
    verbose=True,
    trace_memory=False,
):
    """
    Execute a JSON memory workload using the selected eviction policy.

    Supported operations:
        allocate:
            block_id, size_bytes
            Optional: deadline_ms, reuse_probability,
                      estimated_transfer_ms, quality_risk

        access:
            block_id

        advance_time:
            delta_ms

        update_metadata:
            block_id
            At least one of: deadline_ms, reuse_probability,
                             estimated_transfer_ms, quality_risk

    Optional:
        trace_memory=True prints HBM and host state after each event.

    Returns:
        Dictionary containing workload information and memory metrics.
    """

    if not isinstance(policy, str):
        raise ValueError("Policy must be a string.")

    policy = policy.upper()

    if policy not in SUPPORTED_POLICIES:
        raise ValueError(
            f"Unsupported policy '{policy}'. "
            f"Choose one of {sorted(SUPPORTED_POLICIES)}."
        )

    with open(workload_path, "r", encoding="utf-8") as file:
        workload = json.load(file)

    if "memory" not in workload or "events" not in workload:
        raise ValueError(
            "Workload must contain 'memory' and 'events' fields."
        )

    memory_config = workload["memory"]

    manager = MemoryManager(
        hbm_capacity_bytes=memory_config["hbm_capacity_bytes"],
        host_capacity_bytes=memory_config["host_capacity_bytes"],
        policy=policy,
    )

    allocations = 0
    access_requests = 0
    time_advances = 0
    metadata_allocations = 0
    metadata_updates = 0

    if verbose:
        print(f"\nRunning workload: {workload['workload_name']}")
        print(f"Eviction policy: {policy}")
        print("-" * 75)

    for event_number, event in enumerate(
        workload["events"],
        start=1,
    ):

        if not isinstance(event, dict):
            raise ValueError(
                f"Event {event_number} must be a JSON object."
            )

        if "operation" not in event:
            raise ValueError(
                f"Event {event_number} is missing 'operation'."
            )

        operation = event["operation"]

        if not isinstance(operation, str):
            raise ValueError(
                f"Operation in event {event_number} must be a string."
            )

        operation = operation.lower()

        block_id = "-"

        if operation == "allocate":

            required_fields = ("block_id", "size_bytes")

            for field in required_fields:
                if field not in event:
                    raise ValueError(
                        f"Allocation event {event_number} "
                        f"is missing '{field}'."
                    )

            block_id = event["block_id"]
            size_bytes = event["size_bytes"]

            metadata = {
                field: event[field]
                for field in METADATA_FIELDS
                if field in event
            }

            result = manager.insert(
                block_id,
                size_bytes,
                **metadata,
            )

            allocations += 1

            if metadata:
                metadata_allocations += 1

        elif operation == "access":

            if "block_id" not in event:
                raise ValueError(
                    f"Access event {event_number} "
                    "is missing 'block_id'."
                )

            block_id = event["block_id"]

            result = manager.access(block_id)
            access_requests += 1

        elif operation == "advance_time":

            if "delta_ms" not in event:
                raise ValueError(
                    f"Time event {event_number} "
                    "is missing 'delta_ms'."
                )

            delta_ms = event["delta_ms"]

            manager.advance_time(delta_ms)

            result = (
                f"Advanced simulated time by {delta_ms} ms"
            )

            time_advances += 1

        elif operation == "update_metadata":

            if "block_id" not in event:
                raise ValueError(
                    f"Metadata update event {event_number} "
                    "is missing 'block_id'."
                )

            block_id = event["block_id"]

            metadata = {
                field: event[field]
                for field in METADATA_FIELDS
                if field in event
            }

            if not metadata:
                raise ValueError(
                    f"Metadata update event {event_number} "
                    "must contain at least one metadata field."
                )

            result = manager.update_metadata(
                block_id,
                **metadata,
            )

            metadata_updates += 1

        else:
            raise ValueError(
                f"Unsupported operation '{operation}' "
                f"in event {event_number}."
            )

        manager.validate()

        if verbose:
            print(
                f"Event {event_number:02d} | "
                f"{operation.upper():16} | "
                f"{str(block_id):12} | "
                f"{result}"
            )

            if trace_memory:
                for line in _format_memory_state(
                    manager,
                    memory_config,
                ):
                    print(line)

    metrics = manager.report()

    hbm_hit_rate = (
        metrics["hbm_hits"] / access_requests
        if access_requests else 0.0
    )

    total_cached_hit_rate = (
        (
            metrics["hbm_hits"]
            + metrics["host_hits"]
        ) / access_requests
        if access_requests else 0.0
    )

    results = {
        "workload_name": workload["workload_name"],
        "policy": policy,
        "total_events": len(workload["events"]),
        "allocations": allocations,
        "access_requests": access_requests,
        "time_advances": time_advances,
        "metadata_allocations": metadata_allocations,
        "metadata_updates": metadata_updates,
        "hbm_hits": metrics["hbm_hits"],
        "host_hits": metrics["host_hits"],
        "cold_misses": metrics["cold_misses"],
        "hbm_hit_rate": hbm_hit_rate,
        "total_cached_hit_rate": total_cached_hit_rate,
        "memory_metrics": metrics,
    }

    if verbose:
        print("-" * 75)
        print(f"Completed workload using {policy}.")
        print(f"Access requests: {access_requests}")
        print(f"Metadata allocations: {metadata_allocations}")
        print(f"Metadata updates: {metadata_updates}")
        print(f"HBM hit rate: {hbm_hit_rate * 100:.2f}%")
        print(
            f"Cached hit rate: "
            f"{total_cached_hit_rate * 100:.2f}%"
        )
        print(
            f"Simulated time: "
            f"{metrics['current_time_ms']} ms"
        )
        print(
            f"Reuse prediction observations: "
            f"{metrics['reuse_prediction_observations']}"
        )
        print(
            f"Reuse prediction reliability: "
            f"{metrics['reuse_reliability']:.4f}"
        )

    return results
