
from types import MethodType

from benchmarks import generate_robustness_workloads as generator
from benchmarks.compare_prediction_shifts import apply_shift
from benchmarks.compare_reuse_guard import (
    make_rank,
    MIN_OBSERVATIONS,
    BRIER_DISADVANTAGE,
)
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

WINDOWS = {
    "RECENT_20": 20,
    "RECENT_40": 40,
    "RECENT_60": 60,
}

MODES = [
    "RAW_FIXED",
    "GLOBAL_GUARD_HARD",
    *WINDOWS,
]

PHASE_SIZE = 80
PHASE_COUNT = 3


def make_recent_rank(window, counters):

    def rank(self, contexts):

        if not contexts:
            return []

        # Completed prediction outcomes only.
        # No future workload events are inspected.
        history = list(
            self.reuse_prediction_observations
        )[-window:]

        global_prevalence = (
            self.get_observed_reuse_rate()
        )

        failure = False

        if len(history) >= MIN_OBSERVATIONS:

            recent_prevalence = (
                sum(observed for _, observed in history)
                / len(history)
            )

            prediction_brier = sum(
                (predicted - observed) ** 2
                for predicted, observed in history
            ) / len(history)

            baseline_brier = sum(
                (recent_prevalence - observed) ** 2
                for _, observed in history
            ) / len(history)

            failure = (
                prediction_brier
                > baseline_brier + BRIER_DISADVANTAGE
            )

        counters["rank_calls"] += 1

        if failure:
            counters["failure_evidence_calls"] += 1

        adjusted = []

        for context in contexts:

            item = dict(context)

            # Isolate guard evidence window.
            # Keep the original global-prevalence fallback.

            if failure:
                item["reuse_probability"] = global_prevalence

            adjusted.append(item)

        ranked = self.scheduler.rank(
            adjusted,
            now_ms=self.current_time_ms,
        )

        for item in ranked:
            # Fixed eviction margin.
            item["reliability"] = 0.0

        return ranked

    return rank


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
        for _ in range(PHASE_COUNT)
    ]

    if mode != "RAW_FIXED":

        if mode == "GLOBAL_GUARD_HARD":

            base_rank = make_rank(
                "GLOBAL_GUARD_HARD",
                counters,
            )

        else:

            base_rank = make_recent_rank(
                WINDOWS[mode],
                counters,
            )

        def traced_rank(self, contexts):

            phase = min(
                progress["accesses"] // PHASE_SIZE,
                PHASE_COUNT - 1,
            )

            before_calls = counters["rank_calls"]
            before_active = counters[
                "failure_evidence_calls"
            ]

            ranked = base_rank(self, contexts)

            phase_guards[phase]["calls"] += (
                counters["rank_calls"] - before_calls
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

    phase_hits = []
    previous_hits = 0

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

            if progress["accesses"] % PHASE_SIZE == 0:

                current_hits = manager.report()[
                    "hbm_hits"
                ]

                phase_hits.append(
                    current_hits - previous_hits
                )

                previous_hits = current_hits

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
                f"Unexpected operation: {operation}"
            )

        manager.validate()

    report = manager.report()

    assert progress["accesses"] == 240
    assert len(phase_hits) == PHASE_COUNT
    assert sum(phase_hits) == report["hbm_hits"]

    assert (
        report["hbm_hits"]
        + report["host_hits"]
        + report["cold_misses"]
        == progress["accesses"]
    )

    return {
        "hbm_hits": report["hbm_hits"],
        "phase_hits": phase_hits,
        "phase_guards": phase_guards,
        "accesses": progress["accesses"],
    }


def rate(results, scenario, family, mode, phase=None):

    records = [
        results[(scenario, family, seed, mode)]
        for seed in SEEDS
    ]

    if phase is None:

        hits = sum(r["hbm_hits"] for r in records)
        accesses = sum(r["accesses"] for r in records)

    else:

        hits = sum(
            r["phase_hits"][phase]
            for r in records
        )

        accesses = len(records) * PHASE_SIZE

    return hits / accesses if accesses else 0.0


def activation(results, scenario, family, mode, phase):

    records = [
        results[
            (scenario, family, seed, mode)
        ]["phase_guards"][phase]
        for seed in SEEDS
    ]

    calls = sum(r["calls"] for r in records)
    active = sum(r["active"] for r in records)

    return active / calls if calls else 0.0


def main():

    results = {}

    print("=" * 115)
    print("ACSS MEMCORE - ROLLING EVIDENCE WINDOW EXPERIMENT")
    print("=" * 115)

    print("Seeds:", list(SEEDS))
    print("Families:", FAMILIES)
    print("Scenarios:", SCENARIOS)
    print("Modes:", MODES)
    print("Minimum observations:", MIN_OBSERVATIONS)
    print("Brier disadvantage:", BRIER_DISADVANTAGE)
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
                        "Shift scenario modified no predictions."
                    )

                for mode in MODES:

                    results[
                        (scenario, family, seed, mode)
                    ] = replay(workload, mode)

        print("Completed:", family)

    print()
    print("=" * 115)
    print("OVERALL HBM HIT RATE BY SCENARIO")
    print("=" * 115)

    for scenario in SCENARIOS:

        print(f"\n{scenario}")

        print(
            f"{'MODE':23}"
            f"{'HBM HIT %':>12}"
            f"{'DELTA pp':>12}"
        )

        baseline_records = [
            results[(scenario, family, seed, "RAW_FIXED")]
            for family in FAMILIES
            for seed in SEEDS
        ]

        baseline_hits = sum(
            r["hbm_hits"]
            for r in baseline_records
        )

        total_accesses = sum(
            r["accesses"]
            for r in baseline_records
        )

        baseline = baseline_hits / total_accesses

        for mode in MODES:

            records = [
                results[(scenario, family, seed, mode)]
                for family in FAMILIES
                for seed in SEEDS
            ]

            hit_rate = (
                sum(r["hbm_hits"] for r in records)
                / sum(r["accesses"] for r in records)
            )

            print(
                f"{mode:23}"
                f"{hit_rate * 100:11.2f}%"
                f"{(hit_rate - baseline) * 100:+11.2f}"
            )

    print()
    print("=" * 115)
    print("TEMPORARY INVERSION - PHASE 3 RECOVERY")
    print("=" * 115)

    print(
        f"{'FAMILY':22}"
        f"{'RAW %':>10}"
        f"{'HARD %':>10}"
        f"{'R20 %':>10}"
        f"{'R40 %':>10}"
        f"{'R60 %':>10}"
    )

    print("-" * 115)

    scenario = "TEMPORARY_INVERSION"
    phase = 2

    for family in FAMILIES:

        values = [
            rate(
                results,
                scenario,
                family,
                mode,
                phase,
            ) * 100
            for mode in MODES
        ]

        print(
            f"{family:22}"
            + "".join(f"{v:9.2f}%" for v in values)
        )

    print()
    print("=" * 115)
    print("TEMPORARY INVERSION - GUARD ACTIVATION BY PHASE")
    print("=" * 115)

    print(
        f"{'FAMILY':22}"
        f"{'PHASE':>8}"
        f"{'HARD %':>12}"
        f"{'R20 %':>12}"
        f"{'R40 %':>12}"
        f"{'R60 %':>12}"
    )

    print("-" * 115)

    for family in FAMILIES:

        for phase in range(PHASE_COUNT):

            values = [
                activation(
                    results,
                    "TEMPORARY_INVERSION",
                    family,
                    mode,
                    phase,
                ) * 100
                for mode in MODES[1:]
            ]

            print(
                f"{family:22}"
                f"{phase + 1:8}"
                + "".join(
                    f"{value:11.2f}%"
                    for value in values
                )
            )

    print()
    print("=" * 115)
    print("RECOVERY - PAIRED WINS / TIES / LOSSES")
    print("=" * 115)

    for mode in MODES[1:]:

        wins = ties = losses = 0

        for family in FAMILIES:

            for seed in SEEDS:

                reference = results[
                    (
                        "TEMPORARY_INVERSION",
                        family,
                        seed,
                        "RAW_FIXED",
                    )
                ]["phase_hits"][2]

                candidate = results[
                    (
                        "TEMPORARY_INVERSION",
                        family,
                        seed,
                        mode,
                    )
                ]["phase_hits"][2]

                if candidate > reference:
                    wins += 1
                elif candidate < reference:
                    losses += 1
                else:
                    ties += 1

        print(
            f"{mode:23}"
            f"W/T/L={wins}/{ties}/{losses}"
        )

    expected = (
        len(SEEDS)
        * len(FAMILIES)
        * len(SCENARIOS)
        * len(MODES)
    )

    print()
    print("=" * 115)
    print("VALIDATION")
    print("=" * 115)

    print("Expected executions:", expected)
    print("Completed executions:", len(results))

    assert len(results) == expected

    print("Execution count: PASS")
    print("All results are synthetic simulator measurements.")
    print("Production MemCore policies remain unchanged.")


if __name__ == "__main__":
    main()
