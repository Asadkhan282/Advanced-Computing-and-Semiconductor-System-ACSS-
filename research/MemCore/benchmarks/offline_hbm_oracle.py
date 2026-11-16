
import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Tuple


# ============================================================
# STATE DEFINITIONS
# ============================================================

Block = Tuple[str, int]
HBMState = Tuple[Block, ...]
HostState = Tuple[Block, ...]
MemoryState = Tuple[HBMState, HostState]


# ============================================================
# MEMORY STATE HELPERS
# ============================================================

def used_bytes(blocks: Tuple[Block, ...]) -> int:
    return sum(size for _, size in blocks)


def canonical_hbm(blocks) -> HBMState:
    """
    HBM victim order is not modeled as a replacement queue.
    Canonical sorting allows equivalent HBM states to share
    the same dynamic-programming cache entry.
    """
    return tuple(sorted(blocks))


def remove_block(blocks, block_id):
    return tuple(
        block
        for block in blocks
        if block[0] != block_id
    )


def validate_state(
    hbm: HBMState,
    host: HostState,
    hbm_capacity: int,
    host_capacity: int,
):
    """Validate tier capacities and unique block residency."""

    if used_bytes(hbm) > hbm_capacity:
        raise RuntimeError("HBM capacity exceeded.")

    if used_bytes(host) > host_capacity:
        raise RuntimeError("Host capacity exceeded.")

    hbm_ids = [block_id for block_id, _ in hbm]
    host_ids = [block_id for block_id, _ in host]

    if len(hbm_ids) != len(set(hbm_ids)):
        raise RuntimeError("Duplicate block in HBM.")

    if len(host_ids) != len(set(host_ids)):
        raise RuntimeError("Duplicate block in host memory.")

    if set(hbm_ids) & set(host_ids):
        raise RuntimeError(
            "A block cannot reside in both HBM and host memory."
        )


def place_in_host_fifo(
    host: HostState,
    block: Block,
    host_capacity: int,
) -> HostState:
    """
    Place an evicted HBM block into host memory.

    Host replacement follows FIFO:
    the oldest host-resident blocks are removed first.

    If the block cannot fit in host memory, it is dropped.
    """

    block_id, size_bytes = block

    if size_bytes > host_capacity:
        return host

    next_host = remove_block(host, block_id)

    while (
        next_host
        and used_bytes(next_host) + size_bytes > host_capacity
    ):
        next_host = next_host[1:]

    return next_host + (block,)


# ============================================================
# EXACT OFFLINE ORACLE
# ============================================================

def solve_workload(workload_path: Path) -> dict:

    with open(workload_path, "r", encoding="utf-8") as file:
        workload = json.load(file)

    memory = workload["memory"]
    events = workload["events"]

    hbm_capacity = int(memory["hbm_capacity_bytes"])
    host_capacity = int(memory["host_capacity_bytes"])

    if hbm_capacity <= 0 or host_capacity < 0:
        raise ValueError("Invalid memory capacities.")

    if not isinstance(events, list):
        raise ValueError("Workload events must be a list.")

    # --------------------------------------------------------
    # HBM SPACE ENUMERATION
    # --------------------------------------------------------

    @lru_cache(maxsize=None)
    def ensure_hbm_space(
        hbm: HBMState,
        host: HostState,
        required_bytes: int,
    ) -> Tuple[MemoryState, ...]:

        if required_bytes > hbm_capacity:
            raise ValueError(
                "Requested block exceeds HBM capacity."
            )

        if used_bytes(hbm) + required_bytes <= hbm_capacity:
            return ((hbm, host),)

        if not hbm:
            raise RuntimeError(
                "Oracle could not free required HBM capacity."
            )

        possible_states = set()

        # The oracle explores every possible HBM victim.
        for index, victim in enumerate(hbm):

            next_hbm = (
                hbm[:index] + hbm[index + 1:]
            )

            next_host = place_in_host_fifo(
                host,
                victim,
                host_capacity,
            )

            for final_hbm, final_host in ensure_hbm_space(
                next_hbm,
                next_host,
                required_bytes,
            ):

                possible_states.add(
                    (final_hbm, final_host)
                )

        return tuple(sorted(possible_states))

    # --------------------------------------------------------
    # DYNAMIC PROGRAMMING
    # --------------------------------------------------------

    @lru_cache(maxsize=None)
    def minimum_future_hbm_misses(
        event_index: int,
        hbm: HBMState,
        host: HostState,
    ) -> int:

        if event_index >= len(events):
            return 0

        event = events[event_index]
        operation = event["operation"]

        # ====================================================
        # ALLOCATION
        # ====================================================

        if operation == "allocate":

            block_id = event["block_id"]
            size_bytes = int(event["size_bytes"])

            if size_bytes <= 0 or size_bytes > hbm_capacity:
                raise ValueError(
                    f"Invalid allocation size for {block_id}."
                )

            # Reallocation replaces the previous residency.
            next_hbm = remove_block(hbm, block_id)
            next_host = remove_block(host, block_id)

            possible_states = ensure_hbm_space(
                canonical_hbm(next_hbm),
                next_host,
                size_bytes,
            )

            best = float("inf")

            for available_hbm, available_host in possible_states:

                final_hbm = canonical_hbm(
                    list(available_hbm)
                    + [(block_id, size_bytes)]
                )

                result = minimum_future_hbm_misses(
                    event_index + 1,
                    final_hbm,
                    available_host,
                )

                best = min(best, result)

            return int(best)

        # ====================================================
        # ACCESS
        # ====================================================

        if operation == "access":

            block_id = event["block_id"]

            # ------------------------------------------------
            # CASE 1: HBM HIT
            # ------------------------------------------------

            if any(
                cached_id == block_id
                for cached_id, _ in hbm
            ):

                return minimum_future_hbm_misses(
                    event_index + 1,
                    hbm,
                    host,
                )

            # ------------------------------------------------
            # CASE 2: HOST HIT
            # ------------------------------------------------

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

                # Remove the block from host before restoration.
                next_host = (
                    host[:host_index]
                    + host[host_index + 1:]
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

                    result = minimum_future_hbm_misses(
                        event_index + 1,
                        restored_hbm,
                        available_host,
                    )

                    best = min(best, result)

                # CRITICAL CORRECTION:
                # A host hit requires a host-to-HBM restoration.
                # The simulator counts this as one HBM miss.
                return int(best) + 1

            # ------------------------------------------------
            # CASE 3: COLD MISS
            # ------------------------------------------------

            return 1 + minimum_future_hbm_misses(
                event_index + 1,
                hbm,
                host,
            )

        # ====================================================
        # NON-RESIDENCY OPERATIONS
        # ====================================================

        if operation in (
            "advance_time",
            "update_metadata",
        ):

            return minimum_future_hbm_misses(
                event_index + 1,
                hbm,
                host,
            )

        raise ValueError(
            f"Unsupported operation: {operation}"
        )

    # --------------------------------------------------------
    # INITIAL STATE AND SOLUTION
    # --------------------------------------------------------

    initial_hbm = tuple()
    initial_host = tuple()

    optimum = minimum_future_hbm_misses(
        0,
        initial_hbm,
        initial_host,
    )

    cache_info = minimum_future_hbm_misses.cache_info()

    return {
        "workload": workload.get(
            "workload_name",
            workload_path.stem,
        ),
        "file": workload_path.name,
        "event_count": len(events),
        "optimal_hbm_misses": optimum,
        "dp_states": cache_info.currsize,
    }


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Exact offline oracle minimizing total HBM misses."
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
            "No workload JSON files found."
        )

    print("=" * 90)
    print("ACSS MEMCORE - EXACT OFFLINE HBM-MISS ORACLE")
    print("=" * 90)

    print(
        "Objective: minimize host restorations + cold misses"
    )

    print("Host replacement: FIFO")
    print("HBM victim selection: exact future-aware enumeration")
    print("")

    for workload_path in workload_paths:

        result = solve_workload(workload_path)

        print(f"Workload: {result['file']}")

        print(
            f"  Events: {result['event_count']}"
        )

        print(
            f"  Minimum total HBM misses: "
            f"{result['optimal_hbm_misses']}"
        )

        print(
            f"  DP states: {result['dp_states']}"
        )

        print("")

    print("=" * 90)
    print("IMPORTANT")

    print(
        "This is an exact oracle for the modeled synthetic "
        "two-tier memory system."
    )

    print(
        "It is not a GPU benchmark, latency measurement, "
        "or real-inference performance claim."
    )


if __name__ == "__main__":
    main()
