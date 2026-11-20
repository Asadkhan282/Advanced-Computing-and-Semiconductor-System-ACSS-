
from benchmarks import generate_robustness_workloads as generator
from benchmarks.compare_prediction_shifts import apply_shift
from benchmarks.compare_prequential_guard import execute, MODES


SEEDS = range(800, 820)
FAMILIES = list(generator.FAMILIES)

CONDITIONS = [
    ("default_metadata", "CLEAN"),
    ("informed", "CLEAN"),
    ("informed", "LATE_INVERSION"),
    ("informed", "TEMPORARY_INVERSION"),
]

PHASE_SIZE = 80
TOTAL_ACCESSES = 240


def get_records(results, metadata, scenario, mode, family=None):
    families = [family] if family is not None else FAMILIES

    return [
        results[(metadata, scenario, f, seed, mode)]
        for f in families
        for seed in SEEDS
    ]


def hit_rate(records, phase=None):
    if phase is None:
        hits = sum(r["hbm_hits"] for r in records)
        accesses = sum(r["accesses"] for r in records)
    else:
        hits = sum(r["phase_hits"][phase] for r in records)
        accesses = len(records) * PHASE_SIZE

    return hits / accesses if accesses else 0.0


def paired_counts(results, metadata, scenario, mode, phase=None):
    wins = ties = losses = 0

    for family in FAMILIES:
        for seed in SEEDS:
            candidate = results[
                (metadata, scenario, family, seed, mode)
            ]
            baseline = results[
                (metadata, scenario, family, seed, "RAW_FIXED")
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


def guard_activation(records, phase):
    calls = sum(
        r["phase_guards"][phase]["calls"]
        for r in records
    )
    active = sum(
        r["phase_guards"][phase]["active"]
        for r in records
    )

    return 100.0 * active / calls if calls else 0.0


def main():
    results = {}

    expected = (
        len(SEEDS)
        * len(FAMILIES)
        * len(CONDITIONS)
        * len(MODES)
    )

    print("=" * 110)
    print("ACSS MEMCORE - HELD-OUT PREQUENTIAL GUARD VALIDATION")
    print("=" * 110)
    print("Seeds:", list(SEEDS))
    print("Families:", FAMILIES)
    print("Conditions:", CONDITIONS)
    print("Modes:", MODES)
    print("Expected executions:", expected)
    print("Thresholds: unchanged from compare_prequential_guard")
    print()

    for metadata, scenario in CONDITIONS:
        for family in FAMILIES:
            print(f"Running {metadata:18} {scenario:22} {family}")

            for seed in SEEDS:
                original = generator.build_workload(
                    family=family,
                    seed=seed,
                    hbm_blocks=4,
                    host_blocks=8,
                    metadata_mode=metadata,
                )

                workload, changed = apply_shift(
                    original,
                    scenario,
                )

                if scenario != "CLEAN" and changed == 0:
                    raise AssertionError(
                        f"No metadata change: {family}, {seed}"
                    )

                for mode in MODES:
                    result = execute(workload, mode)

                    assert result["accesses"] == TOTAL_ACCESSES
                    assert len(result["phase_hits"]) == 3
                    assert sum(result["phase_hits"]) == result["hbm_hits"]

                    results[
                        (metadata, scenario, family, seed, mode)
                    ] = result

    print()
    print("=" * 110)
    print("OVERALL RESULTS")
    print("=" * 110)

    for metadata, scenario in CONDITIONS:
        print(f"\nCondition: {metadata} / {scenario}")

        baseline = hit_rate(
            get_records(
                results, metadata, scenario, "RAW_FIXED"
            )
        )

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
                get_records(results, metadata, scenario, mode)
            )

            w, t, l = paired_counts(
                results, metadata, scenario, mode
            )

            print(
                f"{mode:22}"
                f"{rate * 100:9.2f}%"
                f"{(rate - baseline) * 100:+11.2f}"
                f"{w:8}{t:8}{l:9}"
            )

    print()
    print("=" * 110)
    print("PER-FAMILY PERFORMANCE: PREQ_20_FAST12")
    print("=" * 110)

    print(
        f"{'METADATA':18}"
        f"{'SCENARIO':22}"
        f"{'FAMILY':22}"
        f"{'RAW %':>10}"
        f"{'PREQ %':>10}"
        f"{'DELTA':>10}"
    )

    for metadata, scenario in CONDITIONS:
        for family in FAMILIES:
            raw = hit_rate(
                get_records(
                    results, metadata, scenario,
                    "RAW_FIXED", family
                )
            )

            preq = hit_rate(
                get_records(
                    results, metadata, scenario,
                    "PREQ_20_FAST12", family
                )
            )

            print(
                f"{metadata:18}"
                f"{scenario:22}"
                f"{family:22}"
                f"{raw * 100:9.2f}%"
                f"{preq * 100:9.2f}%"
                f"{(preq - raw) * 100:+9.2f}"
            )

    metadata = "informed"
    scenario = "TEMPORARY_INVERSION"
    recovery_phase = 2

    print()
    print("=" * 110)
    print("TEMPORARY INVERSION - RECOVERY PHASE")
    print("=" * 110)

    print(
        f"{'FAMILY':22}"
        f"{'RAW %':>10}"
        f"{'R20 %':>10}"
        f"{'PREQ20 %':>11}"
        f"{'FAST12 %':>11}"
    )

    for family in FAMILIES:
        values = [
            hit_rate(
                get_records(
                    results, metadata, scenario, mode, family
                ),
                phase=recovery_phase,
            ) * 100
            for mode in MODES
        ]

        print(
            f"{family:22}"
            + "".join(f"{value:10.2f}%" for value in values)
        )

    print()
    print("=" * 110)
    print("RECOVERY GUARD ACTIVATION")
    print("=" * 110)

    print(
        f"{'FAMILY':22}"
        f"{'R20 %':>12}"
        f"{'PREQ20 %':>12}"
        f"{'FAST12 %':>12}"
    )

    for family in FAMILIES:
        values = [
            guard_activation(
                get_records(
                    results, metadata, scenario, mode, family
                ),
                recovery_phase,
            )
            for mode in MODES[1:]
        ]

        print(
            f"{family:22}"
            + "".join(f"{v:11.2f}%" for v in values)
        )

    print()
    print("=" * 110)
    print("RECOVERY PAIRED RESULTS VS RAW BASELINE")
    print("=" * 110)

    for mode in MODES[1:]:
        w, t, l = paired_counts(
            results,
            metadata,
            scenario,
            mode,
            phase=recovery_phase,
        )

        print(f"{mode:22} W/T/L={w}/{t}/{l}")

    print()
    print("=" * 110)
    print("WORST PREQUENTIAL RECOVERY CASES")
    print("=" * 110)

    differences = []

    for family in FAMILIES:
        for seed in SEEDS:
            raw = results[
                (metadata, scenario, family, seed, "RAW_FIXED")
            ]["phase_hits"][recovery_phase]

            preq = results[
                (metadata, scenario, family, seed, "PREQ_20_FAST12")
            ]["phase_hits"][recovery_phase]

            differences.append((preq - raw, family, seed))

    for delta, family, seed in sorted(differences)[:10]:
        print(
            f"family={family:22} "
            f"seed={seed} "
            f"recovery_HBM_hit_delta={delta:+d}"
        )

    print()
    print("=" * 110)
    print("VALIDATION")
    print("=" * 110)

    print("Expected executions:", expected)
    print("Completed executions:", len(results))

    assert len(results) == expected

    print("Execution count: PASS")
    print("All compared policies used the same workload events.")
    print("No future reuse outcomes used in guard decisions.")
    print("Production eviction policies remain unchanged.")
    print("All results are synthetic simulator measurements.")


if __name__ == "__main__":
    main()
