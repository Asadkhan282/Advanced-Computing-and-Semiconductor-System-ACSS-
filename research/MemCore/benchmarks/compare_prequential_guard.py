
from types import MethodType

from benchmarks import generate_robustness_workloads as generator
from benchmarks.compare_prediction_shifts import apply_shift
from benchmarks.compare_guard_windows import replay as legacy_replay
from memcore.memory import MemoryManager


SEEDS = range(700, 710)
FAMILIES = list(generator.FAMILIES)

SCENARIOS = [
    "CLEAN",
    "LATE_INVERSION",
    "TEMPORARY_INVERSION",
]

MODES = [
    "RAW_FIXED",
    "RECENT_20",
    "PREQ_20",
    "PREQ_20_FAST12",
]

SLOW_WINDOW = 20
FAST_WINDOW = 12

FAILURE_THRESHOLD = 0.02
RECOVERY_THRESHOLD = -0.02

PHASE_SIZE = 80
TOTAL_ACCESSES = 240


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


def replay_prequential(workload, mode):

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
            "released": 0,
        }
        for _ in range(3)
    ]

    def prequential_rank(self, contexts):

        if not contexts:
            return []

        # Completed observations only.
        # Every baseline was frozen before its outcome.

        slow = self.get_prequential_reuse_metrics(
            window=SLOW_WINDOW
        )

        failure = (
            slow["observations"] >= SLOW_WINDOW
            and slow["excess_brier"] is not None
            and slow["excess_brier"] > FAILURE_THRESHOLD
        )

        recovered = False

        if mode == "PREQ_20_FAST12" and failure:

            fast = self.get_prequential_reuse_metrics(
                window=FAST_WINDOW
            )

            recovered = (
                fast["observations"] >= FAST_WINDOW
                and fast["excess_brier"] is not None
                and fast["excess_brier"]
                <= RECOVERY_THRESHOLD
            )

        guard_active = failure and not recovered

        phase = min(
            progress["accesses"] // PHASE_SIZE,
            2,
        )

        counters = phase_guards[phase]

        counters["calls"] += 1

        if guard_active:
            counters["active"] += 1

        if failure and recovered:
            counters["released"] += 1

        # Keep the original fallback strategy so that
        # only the evidence decision changes.
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

        # Keep the SLO eviction margin fixed.
        for item in ranked:
            item["reliability"] = 0.0

        return ranked

    manager._rank_contexts_adaptively = MethodType(
        prequential_rank,
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

    assert progress["accesses"] == TOTAL_ACCESSES
    assert len(phase_hits) == 3
    assert sum(phase_hits) == report["hbm_hits"]

    assert (
        report["hbm_hits"]
        + report["host_hits"]
        + report["cold_misses"]
        == TOTAL_ACCESSES
    )

    return {
        "hbm_hits": report["hbm_hits"],
        "phase_hits": phase_hits,
        "phase_guards": phase_guards,
        "accesses": progress["accesses"],
    }


def execute(workload, mode):

    if mode in ("PREQ_20", "PREQ_20_FAST12"):
        return replay_prequential(workload, mode)

    return legacy_replay(workload, mode)


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


def activation(records, phase=None):

    selected = []

    for record in records:
        if phase is None:
            selected.extend(record["phase_guards"])
        else:
            selected.append(record["phase_guards"][phase])

    calls = sum(p["calls"] for p in selected)
    active = sum(p["active"] for p in selected)

    return 100 * active / calls if calls else 0.0


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

    print("=" * 112)
    print("ACSS MEMCORE - PREQUENTIAL EVIDENCE GUARD")
    print("=" * 112)

    print("Seeds:", list(SEEDS))
    print("Families:", FAMILIES)
    print("Scenarios:", SCENARIOS)
    print("Modes:", MODES)
    print("Metadata: informed")
    print("Slow window:", SLOW_WINDOW)
    print("Fast window:", FAST_WINDOW)
    print("Failure threshold:", FAILURE_THRESHOLD)
    print("Recovery threshold:", RECOVERY_THRESHOLD)
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
    print("=" * 112)
    print("OVERALL HBM HIT RATE")
    print("=" * 112)

    for scenario in SCENARIOS:

        baseline = hit_rate(
            records_for(results, scenario, "RAW_FIXED")
        )

        print(f"\nScenario: {scenario}")

        print(
            f"{'MODE':22}"
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
                results, scenario, mode
            )

            print(
                f"{mode:22}"
                f"{rate * 100:9.2f}%"
                f"{(rate - baseline) * 100:+11.2f}"
                f"{w:8}{t:8}{l:9}"
            )

    print()
    print("=" * 112)
    print("PER-FAMILY HBM HIT RATE")
    print("=" * 112)

    print(
        f"{'SCENARIO':22}"
        f"{'FAMILY':22}"
        f"{'RAW %':>10}"
        f"{'R20 %':>10}"
        f"{'PREQ20 %':>11}"
        f"{'FAST12 %':>11}"
    )

    for scenario in SCENARIOS:

        for family in FAMILIES:

            rates = [
                hit_rate(
                    records_for(
                        results, scenario, mode, family
                    )
                ) * 100
                for mode in MODES
            ]

            print(
                f"{scenario:22}"
                f"{family:22}"
                + "".join(
                    f"{value:10.2f}%"
                    for value in rates
                )
            )

    print()
    print("=" * 112)
    print("TEMPORARY INVERSION - RECOVERY HBM HIT RATE")
    print("=" * 112)

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
                f"{value:10.2f}%"
                for value in rates
            )
        )

    print()
    print("=" * 112)
    print("TEMPORARY INVERSION - GUARD ACTIVATION BY PHASE")
    print("=" * 112)

    print(
        f"{'FAMILY':22}"
        f"{'PHASE':>8}"
        f"{'R20 %':>12}"
        f"{'PREQ20 %':>12}"
        f"{'FAST12 %':>12}"
    )

    for family in FAMILIES:

        for phase in range(3):

            rates = [
                activation(
                    records_for(
                        results,
                        "TEMPORARY_INVERSION",
                        mode,
                        family,
                    ),
                    phase=phase,
                )
                for mode in MODES[1:]
            ]

            print(
                f"{family:22}"
                f"{phase + 1:8}"
                + "".join(
                    f"{value:11.2f}%"
                    for value in rates
                )
            )

    print()
    print("=" * 112)
    print("RECOVERY PAIRED WINS / TIES / LOSSES")
    print("=" * 112)

    for mode in MODES[1:]:

        w, t, l = paired_counts(
            results,
            "TEMPORARY_INVERSION",
            mode,
            phase=2,
        )

        print(
            f"{mode:22}"
            f"W/T/L={w}/{t}/{l}"
        )

    print()
    print("=" * 112)
    print("VALIDATION")
    print("=" * 112)

    expected = (
        len(SEEDS)
        * len(FAMILIES)
        * len(SCENARIOS)
        * len(MODES)
    )

    print("Expected executions:", expected)
    print("Completed executions:", len(results))

    assert len(results) == expected

    print("Execution count: PASS")
    print("No future reuse outcomes used for guard decisions.")
    print("Prequential baselines frozen at prediction registration.")
    print("Synthetic simulator results only.")
    print("Production eviction policies unchanged.")


if __name__ == "__main__":
    main()
