
from types import MethodType

from benchmarks import generate_robustness_workloads as generator
from benchmarks.compare_prediction_shifts import apply_shift
from benchmarks.compare_guard_windows import replay as standard_replay
from memcore.memory import MemoryManager


SEEDS = range(400, 410)
FAMILIES = list(generator.FAMILIES)

SCENARIOS = [
    "CLEAN",
    "LATE_INVERSION",
    "TEMPORARY_INVERSION",
]

FAST_WINDOWS = {
    "FAST_DROP_8": 8,
    "FAST_DROP_12": 12,
}

MODES = [
    "RAW_FIXED",
    "RECENT_20",
    *FAST_WINDOWS,
]

SLOW_WINDOW = 20
FAILURE_THRESHOLD = 0.02
RECOVERY_DROP = 0.10
PHASE_SIZE = 80


def metadata_from(event):
    return {
        key: event[key]
        for key in (
            "deadline_ms",
            "reuse_probability",
            "estimated_transfer_ms",
            "quality_risk",
        )
        if key in event
    }


def prediction_brier(history):

    if not history:
        return 0.0

    return sum(
        (predicted - observed) ** 2
        for predicted, observed in history
    ) / len(history)


def baseline_brier(history):

    if not history:
        return 0.0

    prevalence = sum(
        observed
        for _, observed in history
    ) / len(history)

    return prevalence * (1.0 - prevalence)


def replay_fast(workload, fast_window):

    config = workload["memory"]

    manager = MemoryManager(
        hbm_capacity_bytes=config["hbm_capacity_bytes"],
        host_capacity_bytes=config["host_capacity_bytes"],
        policy="ADAPTIVE_MARGIN",
    )

    progress = {"accesses": 0}

    phase_guards = [
        {"calls": 0, "active": 0, "fast_release": 0}
        for _ in range(3)
    ]

    def experimental_rank(self, contexts):

        if not contexts:
            return []

        history = list(
            self.reuse_prediction_observations
        )

        slow = history[-SLOW_WINDOW:]
        fast = history[-fast_window:]

        failure = False
        recovery_signal = False

        if len(slow) >= SLOW_WINDOW:

            slow_pred = prediction_brier(slow)
            slow_baseline = baseline_brier(slow)

            failure = (
                slow_pred - slow_baseline
                > FAILURE_THRESHOLD
            )

            fast_pred = prediction_brier(fast)

            recovery_signal = (
                slow_pred - fast_pred
                >= RECOVERY_DROP
            )

        # A recent fall in prediction error can
        # override the slower failure detector.
        #
        # No knowledge of the actual recovery
        # boundary is used.

        guard_active = (
            failure and not recovery_signal
        )

        phase = min(
            progress["accesses"] // PHASE_SIZE,
            2,
        )

        phase_guards[phase]["calls"] += 1

        if guard_active:
            phase_guards[phase]["active"] += 1

        if failure and recovery_signal:
            phase_guards[phase]["fast_release"] += 1

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
            # Fixed SLO margin.
            item["reliability"] = 0.0

        return ranked

    manager._rank_contexts_adaptively = MethodType(
        experimental_rank,
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

    if mode in FAST_WINDOWS:
        return replay_fast(
            workload,
            FAST_WINDOWS[mode],
        )

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

        hits = sum(
            r["hbm_hits"]
            for r in records
        )

        accesses = sum(
            r["accesses"]
            for r in records
        )

    else:

        hits = sum(
            r["phase_hits"][phase]
            for r in records
        )

        accesses = len(records) * PHASE_SIZE

    return hits / accesses if accesses else 0.0


def guard_activation(records, phase):

    calls = sum(
        r["phase_guards"][phase]["calls"]
        for r in records
    )

    active = sum(
        r["phase_guards"][phase]["active"]
        for r in records
    )

    return active / calls if calls else 0.0


def paired_counts(results, scenario, mode, phase=None):

    wins = ties = losses = 0

    for family in FAMILIES:

        for seed in SEEDS:

            candidate = results[
                (scenario, family, seed, mode)
            ]

            baseline = results[
                (scenario, family, seed, "RAW_FIXED")
            ]

            if phase is None:

                a = candidate["hbm_hits"]
                b = baseline["hbm_hits"]

            else:

                a = candidate["phase_hits"][phase]
                b = baseline["phase_hits"][phase]

            if a > b:
                wins += 1
            elif a < b:
                losses += 1
            else:
                ties += 1

    return wins, ties, losses


def main():

    results = {}

    print("=" * 110)
    print("ACSS MEMCORE - FAST RECOVERY DETECTOR")
    print("=" * 110)

    print("Seeds:", list(SEEDS))
    print("Families:", FAMILIES)
    print("Scenarios:", SCENARIOS)
    print("Modes:", MODES)
    print("Slow window:", SLOW_WINDOW)
    print("Failure threshold:", FAILURE_THRESHOLD)
    print("Fast error reduction threshold:", RECOVERY_DROP)
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
        f"{'FAST8 %':>10}"
        f"{'FAST12 %':>11}"
    )

    for family in FAMILIES:

        rates = [
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
                f"{rate:9.2f}%"
                for rate in rates
            )
        )

    print()
    print("=" * 110)
    print("TEMPORARY INVERSION - RECOVERY GUARD ACTIVATION")
    print("=" * 110)

    print(
        f"{'FAMILY':22}"
        f"{'R20 %':>11}"
        f"{'FAST8 %':>11}"
        f"{'FAST12 %':>12}"
    )

    for family in FAMILIES:

        rates = [
            guard_activation(
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
                f"{rate:10.2f}%"
                for rate in rates
            )
        )

    print()
    print("=" * 110)
    print("RECOVERY PAIRED WINS / TIES / LOSSES")
    print("=" * 110)

    for mode in MODES[1:]:

        w, t, l = paired_counts(
            results,
            "TEMPORARY_INVERSION",
            mode,
            phase=2,
        )

        print(
            f"{mode:20}"
            f"W/T/L={w}/{t}/{l}"
        )

    print()
    print("=" * 110)
    print("FAST RECOVERY SIGNAL COUNTS")
    print("=" * 110)

    for scenario in SCENARIOS:

        for mode in FAST_WINDOWS:

            records = records_for(
                results,
                scenario,
                mode,
            )

            releases = sum(
                phase["fast_release"]
                for record in records
                for phase in record["phase_guards"]
            )

            print(
                f"{scenario:22}"
                f"{mode:16}"
                f"release_signal_calls={releases}"
            )

    expected = (
        len(FAMILIES)
        * len(SEEDS)
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
    print("No future access information used by the guard.")
    print("Synthetic simulator results only.")
    print("Production MemCore policies unchanged.")


if __name__ == "__main__":
    main()
