
from types import MethodType

from benchmarks import generate_robustness_workloads as generator
from benchmarks.compare_prediction_shifts import apply_shift
from benchmarks.compare_guard_windows import replay as standard_replay
from benchmarks.compare_fast_recovery_guard import (
    replay_fast,
    metadata_from,
    prediction_brier,
    baseline_brier,
)
from memcore.memory import MemoryManager


SEEDS = range(500, 510)
FAMILIES = list(generator.FAMILIES)

SCENARIOS = [
    "CLEAN",
    "LATE_INVERSION",
    "TEMPORARY_INVERSION",
]

MODES = [
    "RAW_FIXED",
    "RECENT_20",
    "FAST_DROP_12",
    "SKILL_8",
    "SKILL_12",
]

SKILL_WINDOWS = {
    "SKILL_8": 8,
    "SKILL_12": 12,
}

SLOW_WINDOW = 20
FAILURE_THRESHOLD = 0.02
RECOVERY_SKILL_THRESHOLD = 0.02
PHASE_SIZE = 80


def replay_skill(workload, fast_window):

    config = workload["memory"]

    manager = MemoryManager(
        hbm_capacity_bytes=config["hbm_capacity_bytes"],
        host_capacity_bytes=config["host_capacity_bytes"],
        policy="ADAPTIVE_MARGIN",
    )

    progress = {"accesses": 0}

    phase_guards = [
        {
            "calls": 0,
            "active": 0,
            "skill_release": 0,
        }
        for _ in range(3)
    ]

    def skill_rank(self, contexts):

        if not contexts:
            return []

        history = list(
            self.reuse_prediction_observations
        )

        slow = history[-SLOW_WINDOW:]
        fast = history[-fast_window:]

        failure = False
        recovered_skill = False

        if len(slow) >= SLOW_WINDOW:

            slow_error = prediction_brier(slow)
            slow_reference = baseline_brier(slow)

            failure = (
                slow_error - slow_reference
                > FAILURE_THRESHOLD
            )

            if len(fast) >= fast_window:

                fast_error = prediction_brier(fast)
                fast_reference = baseline_brier(fast)

                recovered_skill = (
                    fast_error
                    <= fast_reference
                    - RECOVERY_SKILL_THRESHOLD
                )

        guard_active = (
            failure and not recovered_skill
        )

        phase = min(
            progress["accesses"] // PHASE_SIZE,
            2,
        )

        counters = phase_guards[phase]
        counters["calls"] += 1

        if guard_active:
            counters["active"] += 1

        if failure and recovered_skill:
            counters["skill_release"] += 1

        prevalence = self.get_observed_reuse_rate()

        adjusted = []

        for context in contexts:

            item = dict(context)

            if guard_active:
                item["reuse_probability"] = prevalence

            adjusted.append(item)

        ranked = self.scheduler.rank(
            adjusted,
            now_ms=self.current_time_ms,
        )

        for item in ranked:
            # Isolate reuse adjustment:
            # fixed SLO eviction margin.
            item["reliability"] = 0.0

        return ranked

    manager._rank_contexts_adaptively = MethodType(
        skill_rank,
        manager,
    )

    phase_hits = []
    previous_hits = 0

    for event in workload["events"]:

        operation = event["operation"].lower()

        if operation == "allocate":

            manager.insert(
                event["block_id"],
                event["size_bytes"],
                **metadata_from(event),
            )

        elif operation == "access":

            manager.access(event["block_id"])
            progress["accesses"] += 1

            if progress["accesses"] % PHASE_SIZE == 0:

                hits = manager.report()["hbm_hits"]

                phase_hits.append(
                    hits - previous_hits
                )

                previous_hits = hits

        elif operation == "advance_time":

            manager.advance_time(
                event["delta_ms"]
            )

        elif operation == "update_metadata":

            manager.update_metadata(
                event["block_id"],
                **metadata_from(event),
            )

        else:
            raise ValueError(
                f"Unknown operation: {operation}"
            )

        manager.validate()

    report = manager.report()

    assert progress["accesses"] == 240
    assert len(phase_hits) == 3
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


def execute(workload, mode):

    if mode in SKILL_WINDOWS:
        return replay_skill(
            workload,
            SKILL_WINDOWS[mode],
        )

    if mode == "FAST_DROP_12":
        return replay_fast(workload, 12)

    return standard_replay(workload, mode)


def records_for(results, scenario, mode, family=None):

    families = [family] if family else FAMILIES

    return [
        results[(scenario, f, seed, mode)]
        for f in families
        for seed in SEEDS
    ]


def hit_rate(records, phase=None):

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


def activation(records, phase):

    calls = sum(
        r["phase_guards"][phase]["calls"]
        for r in records
    )

    active = sum(
        r["phase_guards"][phase]["active"]
        for r in records
    )

    return active / calls if calls else 0.0


def paired_counts(results, scenario, mode):

    wins = ties = losses = 0

    for family in FAMILIES:

        for seed in SEEDS:

            candidate = results[
                (scenario, family, seed, mode)
            ]["hbm_hits"]

            baseline = results[
                (scenario, family, seed, "RAW_FIXED")
            ]["hbm_hits"]

            if candidate > baseline:
                wins += 1
            elif candidate < baseline:
                losses += 1
            else:
                ties += 1

    return wins, ties, losses


def main():

    results = {}

    print("=" * 110)
    print("ACSS MEMCORE - SKILL-BASED RECOVERY EXPERIMENT")
    print("=" * 110)

    print("Seeds:", list(SEEDS))
    print("Families:", FAMILIES)
    print("Scenarios:", SCENARIOS)
    print("Modes:", MODES)
    print("Slow window:", SLOW_WINDOW)
    print("Failure threshold:", FAILURE_THRESHOLD)
    print("Recovery skill threshold:", RECOVERY_SKILL_THRESHOLD)
    print("Metadata: informed")
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

                if scenario != "CLEAN":
                    assert changed > 0

                for mode in MODES:

                    results[
                        (scenario, family, seed, mode)
                    ] = execute(workload, mode)

        print("Completed:", family)

    print()
    print("=" * 110)
    print("OVERALL HBM HIT RATE")
    print("=" * 110)

    for scenario in SCENARIOS:

        baseline = hit_rate(
            records_for(
                results,
                scenario,
                "RAW_FIXED",
            )
        )

        print(f"\nScenario: {scenario}")

        print(
            f"{'MODE':20}"
            f"{'HBM %':>10}"
            f"{'DELTA pp':>12}"
            f"{'WINS':>8}"
            f"{'TIES':>8}"
            f"{'LOSSES':>9}"
        )

        for mode in MODES:

            rate = hit_rate(
                records_for(results, scenario, mode)
            )

            w, t, l = paired_counts(
                results,
                scenario,
                mode,
            )

            print(
                f"{mode:20}"
                f"{rate * 100:9.2f}%"
                f"{(rate - baseline) * 100:+11.2f}"
                f"{w:8}{t:8}{l:9}"
            )

    print()
    print("=" * 110)
    print("TEMPORARY INVERSION - RECOVERY HBM HIT RATE")
    print("=" * 110)

    print(
        f"{'FAMILY':22}"
        f"{'RAW %':>10}"
        f"{'R20 %':>10}"
        f"{'DROP12 %':>11}"
        f"{'SKILL8 %':>11}"
        f"{'SKILL12 %':>12}"
    )

    for family in FAMILIES:

        values = [
            hit_rate(
                records_for(
                    results,
                    "TEMPORARY_INVERSION",
                    mode,
                    family,
                ),
                phase=2,
            ) * 100
            for mode in MODES
        ]

        print(
            f"{family:22}"
            + "".join(
                f"{value:10.2f}%"
                for value in values
            )
        )

    print()
    print("=" * 110)
    print("RECOVERY GUARD ACTIVATION")
    print("=" * 110)

    print(
        f"{'FAMILY':22}"
        f"{'R20 %':>11}"
        f"{'DROP12 %':>11}"
        f"{'SKILL8 %':>11}"
        f"{'SKILL12 %':>12}"
    )

    for family in FAMILIES:

        values = [
            activation(
                records_for(
                    results,
                    "TEMPORARY_INVERSION",
                    mode,
                    family,
                ),
                phase=2,
            ) * 100
            for mode in MODES[1:]
        ]

        print(
            f"{family:22}"
            + "".join(
                f"{value:10.2f}%"
                for value in values
            )
        )

    print()
    print("=" * 110)
    print("SKILL-BASED RELEASE SIGNALS")
    print("=" * 110)

    for scenario in SCENARIOS:

        for family in FAMILIES:

            counts = []

            for mode in SKILL_WINDOWS:

                records = records_for(
                    results,
                    scenario,
                    mode,
                    family,
                )

                releases = sum(
                    phase["skill_release"]
                    for record in records
                    for phase in record["phase_guards"]
                )

                counts.append(releases)

            print(
                f"{scenario:22}"
                f"{family:22}"
                f"skill8={counts[0]:5}"
                f" skill12={counts[1]:5}"
            )

    expected = (
        len(SEEDS)
        * len(FAMILIES)
        * len(SCENARIOS)
        * len(MODES)
    )

    print()
    print("=" * 110)
    print("VALIDATION")
    print("=" * 110)

    print("Expected executions:", expected)
    print("Completed executions:", len(results))

    assert len(results) == expected

    print("Execution count: PASS")
    print("Production MemCore policies unchanged.")
    print("Synthetic simulator results only.")


if __name__ == "__main__":
    main()
