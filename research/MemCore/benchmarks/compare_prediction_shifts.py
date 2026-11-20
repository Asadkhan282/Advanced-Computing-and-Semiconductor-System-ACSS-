
from copy import deepcopy

from benchmarks import generate_robustness_workloads as generator
from benchmarks.compare_reuse_guard import execute


SEEDS = range(200, 210)
FAMILIES = list(generator.FAMILIES)

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


def apply_shift(workload, scenario):

    modified = deepcopy(workload)

    access_count = 0
    changed_updates = 0

    for event in modified["events"]:

        operation = event["operation"].lower()

        if operation == "access":
            access_count += 1
            continue

        if operation != "update_metadata":
            continue

        if "reuse_probability" not in event:
            continue

        if scenario == "CLEAN":
            invert = False

        elif scenario == "LATE_INVERSION":
            invert = access_count >= 120

        elif scenario == "TEMPORARY_INVERSION":
            invert = 80 <= access_count < 160

        else:
            raise ValueError(scenario)

        if invert:

            original = float(
                event["reuse_probability"]
            )

            if not 0.0 <= original <= 1.0:
                raise ValueError(
                    "Invalid reuse probability"
                )

            event["reuse_probability"] = (
                1.0 - original
            )

            changed_updates += 1

    return modified, changed_updates


def aggregate(records):

    hits = sum(
        record["hbm_hits"]
        for record in records
    )

    accesses = sum(
        record["access_requests"]
        for record in records
    )

    rank_calls = sum(
        record["rank_calls"]
        for record in records
    )

    guard_calls = sum(
        record["failure_evidence_calls"]
        for record in records
    )

    return {
        "hbm_rate": (
            hits / accesses
            if accesses
            else 0.0
        ),
        "rank_calls": rank_calls,
        "guard_calls": guard_calls,
    }


def get_records(results, scenario, family, mode):

    return [
        results[(scenario, family, seed, mode)]
        for seed in SEEDS
    ]


def main():

    results = {}
    changed_counts = {}

    print("=" * 110)
    print("ACSS MEMCORE - PREDICTION SHIFT ROBUSTNESS")
    print("=" * 110)

    print("Seeds:", list(SEEDS))
    print("Families:", FAMILIES)
    print("Scenarios:", SCENARIOS)
    print("Policies:", MODES)
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

                changed_counts[
                    (scenario, family, seed)
                ] = changed

                if scenario != "CLEAN" and changed == 0:
                    raise AssertionError(
                        f"No predictions modified: "
                        f"{scenario}, {family}, {seed}"
                    )

                for mode in MODES:

                    results[
                        (scenario, family, seed, mode)
                    ] = execute(
                        workload,
                        mode,
                    )

        print("Completed family:", family)

    print()
    print("=" * 110)
    print("PER-FAMILY RESULTS")
    print("=" * 110)

    print(
        f"{'SCENARIO':22}"
        f"{'FAMILY':22}"
        f"{'RAW %':>10}"
        f"{'SOFT %':>10}"
        f"{'HARD %':>10}"
        f"{'DELTA pp':>12}"
    )

    print("-" * 110)

    for scenario in SCENARIOS:

        for family in FAMILIES:

            rates = {}

            for mode in MODES:

                records = get_records(
                    results,
                    scenario,
                    family,
                    mode,
                )

                rates[mode] = (
                    aggregate(records)["hbm_rate"]
                    * 100.0
                )

            delta = (
                rates["GLOBAL_GUARD_HARD"]
                - rates["RAW_FIXED"]
            )

            print(
                f"{scenario:22}"
                f"{family:22}"
                f"{rates['RAW_FIXED']:9.2f}%"
                f"{rates['GLOBAL_GUARD_SOFT']:9.2f}%"
                f"{rates['GLOBAL_GUARD_HARD']:9.2f}%"
                f"{delta:+11.2f}"
            )

    print()
    print("=" * 110)
    print("SCENARIO SUMMARY")
    print("=" * 110)

    for scenario in SCENARIOS:

        print(f"\nScenario: {scenario}")

        baseline_records = [
            results[
                (scenario, family, seed, "RAW_FIXED")
            ]
            for family in FAMILIES
            for seed in SEEDS
        ]

        baseline_rate = aggregate(
            baseline_records
        )["hbm_rate"]

        for mode in MODES:

            records = [
                results[(scenario, family, seed, mode)]
                for family in FAMILIES
                for seed in SEEDS
            ]

            stats = aggregate(records)

            wins = 0
            ties = 0
            losses = 0

            for family in FAMILIES:

                for seed in SEEDS:

                    candidate = results[
                        (scenario, family, seed, mode)
                    ]["hbm_hits"]

                    reference = results[
                        (
                            scenario,
                            family,
                            seed,
                            "RAW_FIXED",
                        )
                    ]["hbm_hits"]

                    if candidate > reference:
                        wins += 1

                    elif candidate < reference:
                        losses += 1

                    else:
                        ties += 1

            delta = (
                stats["hbm_rate"]
                - baseline_rate
            ) * 100.0

            print(
                f"{mode:22}"
                f" hit={stats['hbm_rate'] * 100:6.2f}%"
                f" delta={delta:+6.2f}pp"
                f" W/T/L={wins}/{ties}/{losses}"
            )

    print()
    print("=" * 110)
    print("HARD GUARD ACTIVATION")
    print("=" * 110)

    for scenario in SCENARIOS:

        for family in FAMILIES:

            records = get_records(
                results,
                scenario,
                family,
                "GLOBAL_GUARD_HARD",
            )

            stats = aggregate(records)

            calls = stats["rank_calls"]
            guarded = stats["guard_calls"]

            percentage = (
                100.0 * guarded / calls
                if calls
                else 0.0
            )

            modifications = sum(
                changed_counts[
                    (scenario, family, seed)
                ]
                for seed in SEEDS
            )

            print(
                f"{scenario:22}"
                f"{family:22}"
                f" changed={modifications:4}"
                f" guard={percentage:6.2f}%"
            )

    print()
    print("=" * 110)
    print("VALIDATION SUMMARY")
    print("=" * 110)

    expected = (
        len(SCENARIOS)
        * len(FAMILIES)
        * len(SEEDS)
        * len(MODES)
    )

    print("Expected executions:", expected)
    print("Completed executions:", len(results))

    if len(results) != expected:
        raise AssertionError(
            "Incomplete experiment"
        )

    print("Execution count: PASS")

    print()
    print(
        "Prediction inversion changes metadata only."
    )
    print(
        "Access sequences are preserved."
    )
    print(
        "Results are synthetic simulator measurements."
    )
    print(
        "No production policies were modified."
    )


if __name__ == "__main__":
    main()
