
from types import MethodType

from benchmarks import generate_robustness_workloads as generator
from benchmarks.compare_prediction_shifts import apply_shift
from benchmarks.compare_reuse_guard import make_rank
from memcore.memory import MemoryManager


SEEDS = range(200, 210)

FAMILIES = [
    "zipf",
    "bursty",
    "recurring",
    "prediction_mismatch",
]

SCENARIOS = [
    "CLEAN",
    "LATE_INVERSION",
    "TEMPORARY_INVERSION",
]

MODES = [
    "RAW_FIXED",
    "GLOBAL_GUARD_SOFT",
    "GLOBAL_GUARD_HARD",
]

BOUNDARIES = (80, 160, 240)

PHASE_NAMES = [
    "BEFORE",
    "SHIFT",
    "RECOVERY",
]


def replay(workload, mode):

    config = workload["memory"]

    policy = (
        "SLO_MARGIN"
        if mode == "RAW_FIXED"
        else "ADAPTIVE_MARGIN"
    )

    manager = MemoryManager(
        hbm_capacity_bytes=config["hbm_capacity_bytes"],
        host_capacity_bytes=config["host_capacity_bytes"],
        policy=policy,
    )

    progress = {"accesses": 0}

    counters = {
        "rank_calls": 0,
        "failure_evidence_calls": 0,
    }

    phase_guards = [
        {"calls": 0, "active": 0}
        for _ in range(3)
    ]

    if mode != "RAW_FIXED":

        base_rank = make_rank(mode, counters)

        def traced_rank(self, contexts):

            phase = min(
                progress["accesses"] // 80,
                2,
            )

            before_calls = counters["rank_calls"]
            before_active = counters[
                "failure_evidence_calls"
            ]

            ranked = base_rank(self, contexts)

            phase_guards[phase]["calls"] += (
                counters["rank_calls"]
                - before_calls
            )

            phase_guards[phase]["active"] += (
                counters["failure_evidence_calls"]
                - before_active
            )

            return ranked

        manager._rank_contexts_adaptively = MethodType(
            traced_rank,
            manager,
        )

    previous_hits = 0
    phase_hits = []
    phase_host_hits = []
    previous_host_hits = 0

    for event in workload["events"]:

        operation = event["operation"].lower()

        if operation == "allocate":

            metadata = {
                key: event[key]
                for key in (
                    "deadline_ms",
                    "reuse_probability",
                    "estimated_transfer_ms",
                    "quality_risk",
                )
                if key in event
            }

            manager.insert(
                event["block_id"],
                event["size_bytes"],
                **metadata,
            )

        elif operation == "access":

            manager.access(event["block_id"])

            progress["accesses"] += 1

            if progress["accesses"] in BOUNDARIES:

                report = manager.report()

                current_hits = report["hbm_hits"]
                current_host_hits = report["host_hits"]

                phase_hits.append(
                    current_hits - previous_hits
                )

                phase_host_hits.append(
                    current_host_hits - previous_host_hits
                )

                previous_hits = current_hits
                previous_host_hits = current_host_hits

        elif operation == "advance_time":

            manager.advance_time(
                event["delta_ms"]
            )

        elif operation == "update_metadata":

            metadata = {
                key: event[key]
                for key in (
                    "deadline_ms",
                    "reuse_probability",
                    "estimated_transfer_ms",
                    "quality_risk",
                )
                if key in event
            }

            manager.update_metadata(
                event["block_id"],
                **metadata,
            )

        else:

            raise ValueError(
                f"Unknown operation: {operation}"
            )

        manager.validate()

    assert progress["accesses"] == 240
    assert len(phase_hits) == 3

    final = manager.report()

    assert sum(phase_hits) == final["hbm_hits"]
    assert sum(phase_host_hits) == final["host_hits"]

    return {
        "phase_hits": phase_hits,
        "phase_host_hits": phase_host_hits,
        "phase_guards": phase_guards,
        "total_hits": final["hbm_hits"],
    }


def get_phase_rate(
    results,
    scenario,
    family,
    mode,
    phase,
):

    total_hits = sum(
        results[
            (scenario, family, seed, mode)
        ]["phase_hits"][phase]
        for seed in SEEDS
    )

    total_accesses = len(SEEDS) * 80

    return (
        total_hits / total_accesses
        if total_accesses
        else 0.0
    )


def get_guard_activation(
    results,
    scenario,
    family,
    mode,
    phase,
):

    records = [
        results[
            (scenario, family, seed, mode)
        ]["phase_guards"][phase]
        for seed in SEEDS
    ]

    calls = sum(
        record["calls"]
        for record in records
    )

    active = sum(
        record["active"]
        for record in records
    )

    return active / calls if calls else 0.0


def main():

    results = {}

    print("=" * 112)
    print("ACSS MEMCORE - GUARD RECOVERY DIAGNOSTICS")
    print("=" * 112)

    print("Seeds:", list(SEEDS))
    print("Families:", FAMILIES)
    print("Scenarios:", SCENARIOS)
    print("Modes:", MODES)
    print("Phase boundaries:", BOUNDARIES)
    print()

    for family in FAMILIES:

        for seed in SEEDS:

            original = generator.build_workload(
                family=family,
                seed=seed,
                hbm_blocks=4,
                host_blocks=8,
                metadata_mode="informed",
            )

            for scenario in SCENARIOS:

                workload, changed = apply_shift(
                    original,
                    scenario,
                )

                if scenario != "CLEAN" and changed == 0:
                    raise AssertionError(
                        "No metadata changes"
                    )

                for mode in MODES:

                    results[
                        (scenario, family, seed, mode)
                    ] = replay(
                        workload,
                        mode,
                    )

        print("Completed:", family)

    print()
    print("=" * 112)
    print("PHASE-LEVEL PERFORMANCE AND GUARD ACTIVATION")
    print("=" * 112)

    print(
        f"{'SCENARIO':22}"
        f"{'FAMILY':22}"
        f"{'PHASE':11}"
        f"{'RAW %':>9}"
        f"{'SOFT %':>9}"
        f"{'HARD %':>9}"
        f"{'DELTA':>9}"
        f"{'GUARD %':>10}"
    )

    print("-" * 112)

    for scenario in SCENARIOS:

        for family in FAMILIES:

            for phase in range(3):

                raw = get_phase_rate(
                    results,
                    scenario,
                    family,
                    "RAW_FIXED",
                    phase,
                )

                soft = get_phase_rate(
                    results,
                    scenario,
                    family,
                    "GLOBAL_GUARD_SOFT",
                    phase,
                )

                hard = get_phase_rate(
                    results,
                    scenario,
                    family,
                    "GLOBAL_GUARD_HARD",
                    phase,
                )

                guard = get_guard_activation(
                    results,
                    scenario,
                    family,
                    "GLOBAL_GUARD_HARD",
                    phase,
                )

                delta = (hard - raw) * 100

                print(
                    f"{scenario:22}"
                    f"{family:22}"
                    f"{PHASE_NAMES[phase]:11}"
                    f"{raw * 100:8.2f}%"
                    f"{soft * 100:8.2f}%"
                    f"{hard * 100:8.2f}%"
                    f"{delta:+8.2f}"
                    f"{guard * 100:9.2f}%"
                )

    print()
    print("=" * 112)
    print("TEMPORARY-INVERSION RECOVERY: PAIRED COMPARISON")
    print("=" * 112)

    for family in FAMILIES:

        wins = 0
        ties = 0
        losses = 0

        for seed in SEEDS:

            raw = results[
                (
                    "TEMPORARY_INVERSION",
                    family,
                    seed,
                    "RAW_FIXED",
                )
            ]["phase_hits"][2]

            hard = results[
                (
                    "TEMPORARY_INVERSION",
                    family,
                    seed,
                    "GLOBAL_GUARD_HARD",
                )
            ]["phase_hits"][2]

            if hard > raw:
                wins += 1
            elif hard < raw:
                losses += 1
            else:
                ties += 1

        guard = get_guard_activation(
            results,
            "TEMPORARY_INVERSION",
            family,
            "GLOBAL_GUARD_HARD",
            2,
        )

        print(
            f"{family:22}"
            f" W/T/L={wins}/{ties}/{losses}"
            f" recovery_guard={guard * 100:.2f}%"
        )

    expected = (
        len(SEEDS)
        * len(FAMILIES)
        * len(SCENARIOS)
        * len(MODES)
    )

    print()
    print("=" * 112)
    print("VALIDATION")
    print("=" * 112)

    print("Expected runs:", expected)
    print("Completed runs:", len(results))

    assert len(results) == expected

    print("Execution count: PASS")
    print("No production code modified.")
    print("Synthetic simulator results only.")


if __name__ == "__main__":
    main()
