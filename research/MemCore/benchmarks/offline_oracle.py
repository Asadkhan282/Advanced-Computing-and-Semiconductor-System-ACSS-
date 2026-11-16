
import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple


Block = Tuple[str, int]
HBMState = Tuple[Block, ...]
HostState = Tuple[Block, ...]
MemoryState = Tuple[HBMState, HostState]


def used_bytes(blocks: Tuple[Block, ...]) -> int:
    return sum(size for _, size in blocks)


def canonical_hbm(blocks: List[Block]) -> HBMState:
    """HBM order does not matter to the offline oracle."""
    return tuple(sorted(blocks))


def remove_block(
    blocks: Tuple[Block, ...],
    block_id: str,
) -> Tuple[Block, ...]:
    return tuple(
        block for block in blocks
        if block[0] != block_id
    )


def place_in_host_fifo(
    host: HostState,
    incoming: Block,
    host_capacity: int,
) -> HostState:
    """
    Reproduce the project's current host-tier behavior:
    - Drop an incoming block if it exceeds host capacity.
    - Otherwise evict the oldest host entries until it fits.
    - Append the incoming block at the newest position.
    """

    block_id, size_bytes = incoming

    if size_bytes > host_capacity:
        return host

    remaining = list(host)
    host_used = used_bytes(tuple(remaining))

    while host_used + size_bytes > host_capacity:
        if not remaining:
            break

        _, evicted_size = remaining.pop(0)
        host_used -= evicted_size

    if host_used + size_bytes <= host_capacity:
        remaining.append((block_id, size_bytes))

    return tuple(remaining)


def solve_workload(workload_path: Path) -> Dict:
    with workload_path.open("r", encoding="utf-8") as file:
        workload = json.load(file)

    memory = workload["memory"]
    events = workload["events"]

    hbm_capacity = int(memory["hbm_capacity_bytes"])
    host_capacity = int(memory["host_capacity_bytes"])

    if hbm_capacity <= 0 or host_capacity < 0:
        raise ValueError("Invalid memory capacities.")

    # Enumerate every possible sequence of future-aware HBM
    # victim decisions. Host placement remains FIFO.
    @lru_cache(maxsize=None)
    def ensure_hbm_space(
        hbm: HBMState,
        host: HostState,
        required_bytes: int,
    ) -> Tuple[MemoryState, ...]:

        if used_bytes(hbm) + required_bytes <= hbm_capacity:
            return ((hbm, host),)

        if not hbm:
            raise RuntimeError(
                "Oracle could not free the required HBM capacity."
            )

        possible_states = set()

        # The offline oracle may choose any HBM victim.
        for index, victim in enumerate(hbm):

            next_hbm = (
                hbm[:index] + hbm[index + 1:]
            )

            next_host = place_in_host_fifo(
                host,
                victim,
                host_capacity,
            )

            resulting_states = ensure_hbm_space(
                next_hbm,
                next_host,
                required_bytes,
            )

            possible_states.update(resulting_states)

        return tuple(
            sorted(possible_states)
        )

    @lru_cache(maxsize=None)
    def minimum_future_cold_misses(
        event_index: int,
        hbm: HBMState,
        host: HostState,
    ) -> int:

        if event_index >= len(events):
            return 0

        event = events[event_index]
        operation = event["operation"]

        # Allocation: remove any previous copy, then insert
        # the newly allocated block into HBM.
        if operation == "allocate":

            block_id = event["block_id"]
            size_bytes = int(event["size_bytes"])

            if size_bytes <= 0 or size_bytes > hbm_capacity:
                raise ValueError(
                    f"Invalid allocation size for {block_id}."
                )

            next_hbm = remove_block(hbm, block_id)
            next_host = remove_block(host, block_id)

            possible_states = ensure_hbm_space(
                canonical_hbm(list(next_hbm)),
                next_host,
                size_bytes,
            )

            best = float("inf")

            for available_hbm, available_host in possible_states:

                final_hbm = canonical_hbm(
                    list(available_hbm) + [(block_id, size_bytes)]
                )

                result = minimum_future_cold_misses(
                    event_index + 1,
                    final_hbm,
                    available_host,
                )

                best = min(best, result)

            return int(best)

        # Access: HBM hits do not change residency.
        if operation == "access":

            block_id = event["block_id"]

            if any(
                cached_id == block_id
                for cached_id, _ in hbm
            ):
                return minimum_future_cold_misses(
                    event_index + 1,
                    hbm,
                    host,
                )

            # A host hit restores the block to HBM.
            host_index = next(
                (
                    index
                    for index, (cached_id, _) in enumerate(host)
                    if cached_id == block_id
                ),
                None,
            )

            if host_index is not None:

                block = host[host_index]

                next_host = (
                    host[:host_index] + host[host_index + 1:]
                )

                possible_states = ensure_hbm_space(
                    hbm,
                    next_host,
                    block[1],
                )

                best = float("inf")

                for available_hbm, available_host in possible_states:

                    restored_hbm = canonical_hbm(
                        list(available_hbm) + [block]
                    )

                    result = minimum_future_cold_misses(
                        event_index + 1,
                        restored_hbm,
                        available_host,
                    )

                    best = min(best, result)

                return int(best)

            # Neither tier contains the requested block.
            return 1 + minimum_future_cold_misses(
                event_index + 1,
                hbm,
                host,
            )

        # Time advances do not change cache residency.
        if operation == "advance_time":
            return minimum_future_cold_misses(
                event_index + 1,
                hbm,
                host,
            )

        # Metadata updates do not change residency or
        # directly affect the cold-miss objective.
        if operation == "update_metadata":
            return minimum_future_cold_misses(
                event_index + 1,
                hbm,
                host,
            )

        raise ValueError(
            f"Unsupported operation: {operation}"
        )

    optimum = minimum_future_cold_misses(
        0,
        tuple(),
        tuple(),
    )

    return {
        "workload": workload.get(
            "workload_name",
            workload_path.name,
        ),
        "file": workload_path.name,
        "hbm_capacity_bytes": hbm_capacity,
        "host_capacity_bytes": host_capacity,
        "event_count": len(events),
        "optimal_cold_misses": optimum,
        "solver_states": minimum_future_cold_misses.cache_info().currsize,
        "space_states": ensure_hbm_space.cache_info().currsize,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Exact future-aware two-tier memory oracle "
            "with the project's host FIFO behavior."
        )
    )

    parser.add_argument(
        "--workloads",
        nargs="*",
        help="Optional workload JSON paths.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    if args.workloads:
        workload_paths = [
            Path(path).resolve()
            for path in args.workloads
        ]
    else:
        workload_paths = sorted(
            (project_root / "workloads").glob("*.json")
        )

    if not workload_paths:
        raise FileNotFoundError(
            "No workload JSON files were found."
        )

    print("=" * 88)
    print("ACSS MEMCORE — EXACT OFFLINE TWO-TIER ORACLE")
    print("=" * 88)
    print("Objective: minimize total cold misses")
    print("HBM: same capacity as each workload")
    print("Host: same capacity and FIFO eviction as current implementation")
    print("Oracle: future-aware HBM victim selection")
    print("")

    results = []

    for workload_path in workload_paths:

        result = solve_workload(workload_path)
        results.append(result)

        print("-" * 88)
        print(f"Workload: {result['workload']}")
        print(f"File: {result['file']}")
        print(
            f"HBM capacity: {result['hbm_capacity_bytes']} bytes"
        )
        print(
            f"Host capacity: {result['host_capacity_bytes']} bytes"
        )
        print(f"Events: {result['event_count']}")
        print(
            f"Minimum possible cold misses: "
            f"{result['optimal_cold_misses']}"
        )
        print(
            f"DP event states explored: "
            f"{result['solver_states']}"
        )
        print(
            f"HBM-space states explored: "
            f"{result['space_states']}"
        )

    print("")
    print("=" * 88)
    print("ORACLE SUMMARY")
    print("=" * 88)

    print(
        f"{'Workload':42} {'Events':>8} {'Minimum cold misses':>22}"
    )

    for result in results:
        print(
            f"{result['file'][:42]:42} "
            f"{result['event_count']:>8} "
            f"{result['optimal_cold_misses']:>22}"
        )

    print("")
    print(
        "IMPORTANT: These are exact results for the stated synthetic "
        "workloads and model."
    )
    print(
        "They are not GPU benchmarks, latency measurements, or "
        "real-inference performance claims."
    )


if __name__ == "__main__":
    main()
