
from benchmarks import generate_robustness_workloads as generator
from benchmarks.compare_prediction_shifts import apply_shift
from benchmarks.compare_guard_windows import replay


SEEDS = range(300, 320)
FAMILIES = list(generator.FAMILIES)

SCENARIOS = [
    "CLEAN",
    "LATE_INVERSION",
    "TEMPORARY_INVERSION",
]

MODES = [
    "RAW_FIXED",
    "GLOBAL_GUARD_HARD",
    "RECENT_20",
    "RECENT_40",
    "RECENT_60",
]

PHASE_SIZE = 80


def records_for(results, scenario, mode, family=None):

    families = [family] if family is not None else FAMILIES

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


def compare_paired(results, scenario, mode, phase=None):

    wins = ties = losses = 0

    for family in FAMILIES:

        for seed in SEEDS:

            candidate = results[
                (scenario, family, seed, mode)
            ]

            reference = results[
                (scenario, family, seed, "RAW_FIXED")
            ]

            if phase is None:

                candidate_hits = candidate["hbm_hits"]
                reference_hits = reference["hbm_hits"]

            else:

                candidate_hits = candidate["phase_hits"][phase]
                reference_hits = reference["phase_hits"][phase]

            if candidate_hits > reference_hits:
                wins += 1

            elif candidate_hits < reference_hits:
                losses += 1

            else:
                ties += 1

    return wins, ties, losses


def main():

    results = {}

    expected = (
        len(SEEDS)
        * len(FAMILIES)
        * len(SCENARIOS)
        * len(MODES)
    )

    print("=" * 110)
    print("ACSS MEMCORE - HELD-OUT GUARD WINDOW VALIDATION")
    print("=" * 110)

    print("Seeds:", list(SEEDS))
    print("Families:", FAMILIES)
    print("Scenarios:", SCENARIOS)
    print("Policies:", MODES)
    print("Metadata: informed")
    print("Expected executions:", expected)
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
                        "Expected metadata inversion: "
                        f"{scenario}, {family}, {seed}"
                    )

                for mode in MODES:

                    result = replay(workload, mode)

                    results[
                        (scenario, family, seed, mode)
                    ] = result

        print("Completed:", family)

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    print()
    print("=" * 110)
    print("OVERALL HBM HIT RATE - HELD-OUT SEEDS")
    print("=" * 110)

    for scenario in SCENARIOS:

        baseline = hit_rate(
            records_for(
                results,
                scenario,
                "RAW_FIXED",
            )
        )

        print()
        print("Scenario:", scenario)

        print(
            f"{'MODE':23}"
            f"{'HBM HIT %':>12}"
            f"{'DELTA pp':>12}"
            f"{'WINS':>8}"
            f"{'TIES':>8}"
            f"{'LOSSES':>9}"
        )

        print("-" * 80)

        for mode in MODES:

            rate = hit_rate(
                records_for(results, scenario, mode)
            )

            wins, ties, losses = compare_paired(
                results,
                scenario,
                mode,
            )

            print(
                f"{mode:23}"
                f"{rate * 100:11.2f}%"
                f"{(rate - baseline) * 100:+11.2f}"
                f"{wins:8}"
                f"{ties:8}"
                f"{losses:9}"
            )

    # ========================================================
    # PER-FAMILY RESULTS
    # ========================================================

    print()
    print("=" * 110)
    print("PER-FAMILY HBM HIT RATE")
    print("=" * 110)

    print(
        f"{'SCENARIO':22}"
        f"{'FAMILY':22}"
        f"{'RAW %':>10}"
        f"{'HARD %':>10}"
        f"{'R20 %':>10}"
        f"{'R20 DELTA':>12}"
    )

    print("-" * 110)

    for scenario in SCENARIOS:

        for family in FAMILIES:

            raw = hit_rate(
                records_for(
                    results,
                    scenario,
                    "RAW_FIXED",
                    family,
                )
            )

            hard = hit_rate(
                records_for(
                    results,
                    scenario,
                    "GLOBAL_GUARD_HARD",
                    family,
                )
            )

            recent20 = hit_rate(
                records_for(
                    results,
                    scenario,
                    "RECENT_20",
                    family,
                )
            )

            print(
                f"{scenario:22}"
                f"{family:22}"
                f"{raw * 100:9.2f}%"
                f"{hard * 100:9.2f}%"
                f"{recent20 * 100:9.2f}%"
                f"{(recent20 - raw) * 100:+11.2f}"
            )

    # ========================================================
    # TEMPORARY-INVERSION RECOVERY
    # ========================================================

    scenario = "TEMPORARY_INVERSION"
    recovery_phase = 2

    print()
    print("=" * 110)
    print("TEMPORARY INVERSION - RECOVERY PHASE")
    print("=" * 110)

    print(
        f"{'FAMILY':22}"
        f"{'RAW %':>10}"
        f"{'HARD %':>10}"
        f"{'R20 %':>10}"
        f"{'R40 %':>10}"
        f"{'R60 %':>10}"
    )

    print("-" * 110)

    for family in FAMILIES:

        values = []

        for mode in MODES:

            records = records_for(
                results,
                scenario,
                mode,
                family,
            )

            values.append(
                hit_rate(
                    records,
                    recovery_phase,
                ) * 100
            )

        print(
            f"{family:22}"
            + "".join(
                f"{value:9.2f}%"
                for value in values
            )
        )

    # ========================================================
    # GUARD ACTIVATION IN RECOVERY
    # ========================================================

    print()
    print("=" * 110)
    print("TEMPORARY INVERSION - RECOVERY GUARD ACTIVATION")
    print("=" * 110)

    print(
        f"{'FAMILY':22}"
        f"{'HARD %':>12}"
        f"{'R20 %':>12}"
        f"{'R40 %':>12}"
        f"{'R60 %':>12}"
    )

    print("-" * 110)

    for family in FAMILIES:

        values = []

        for mode in MODES[1:]:

            records = records_for(
                results,
                scenario,
                mode,
                family,
            )

            values.append(
                guard_activation(
                    records,
                    recovery_phase,
                ) * 100
            )

        print(
            f"{family:22}"
            + "".join(
                f"{value:11.2f}%"
                for value in values
            )
        )

    # ========================================================
    # RECOVERY PAIRED RESULTS
    # ========================================================

    print()
    print("=" * 110)
    print("RECOVERY PHASE - PAIRED WINS / TIES / LOSSES")
    print("=" * 110)

    for mode in MODES[1:]:

        wins, ties, losses = compare_paired(
            results,
            scenario,
            mode,
            phase=recovery_phase,
        )

        print(
            f"{mode:23}"
            f"W/T/L={wins}/{ties}/{losses}"
        )

    # ========================================================
    # WORST RECENT_20 RECOVERY CASES
    # ========================================================

    print()
    print("=" * 110)
    print("RECENT_20 - WORST RECOVERY CASES")
    print("=" * 110)

    differences = []

    for family in FAMILIES:

        for seed in SEEDS:

            raw = results[
                (scenario, family, seed, "RAW_FIXED")
            ]["phase_hits"][recovery_phase]

            recent = results[
                (scenario, family, seed, "RECENT_20")
            ]["phase_hits"][recovery_phase]

            differences.append(
                (recent - raw, family, seed)
            )

    for delta, family, seed in sorted(differences)[:10]:

        print(
            f"family={family:22}"
            f" seed={seed}"
            f" recovery_HBM_hit_delta={delta:+d}"
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 110)
    print("VALIDATION SUMMARY")
    print("=" * 110)

    print("Expected executions:", expected)
    print("Completed executions:", len(results))

    assert len(results) == expected

    print("Execution count: PASS")
    print("All policies evaluated on identical access traces.")
    print("Seed range:", SEEDS.start, "-", SEEDS.stop - 1)
    print("Metadata mode: informed")
    print("Guard thresholds remain unchanged.")
    print("Results are synthetic simulator measurements.")
    print("Production MemCore code remains unchanged.")


if __name__ == "__main__":
    main()
