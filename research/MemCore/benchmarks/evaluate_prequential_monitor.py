
from benchmarks import generate_robustness_workloads as generator
from benchmarks.compare_prediction_shifts import apply_shift
from memcore.memory import MemoryManager


SEEDS = range(600, 610)

FAMILIES = list(generator.FAMILIES)

SCENARIOS = [
    "CLEAN",
    "LATE_INVERSION",
    "TEMPORARY_INVERSION",
]

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


def calculate_metrics(observations):

    count = len(observations)

    if count == 0:
        return {
            "observations": 0,
            "model_brier": None,
            "baseline_brier": None,
            "excess_brier": None,
        }

    model_brier = sum(
        (predicted - outcome) ** 2
        for predicted, baseline, outcome in observations
    ) / count

    baseline_brier = sum(
        (baseline - outcome) ** 2
        for predicted, baseline, outcome in observations
    ) / count

    return {
        "observations": count,
        "model_brier": model_brier,
        "baseline_brier": baseline_brier,
        "excess_brier": model_brier - baseline_brier,
    }


def replay(workload):

    config = workload["memory"]

    manager = MemoryManager(
        hbm_capacity_bytes=config["hbm_capacity_bytes"],
        host_capacity_bytes=config["host_capacity_bytes"],
        policy="SLO_MARGIN",
    )

    phase_records = [[], [], []]
    access_count = 0

    for event in workload["events"]:

        operation = event["operation"].lower()

        if operation == "allocate":

            manager.insert(
                event["block_id"],
                event["size_bytes"],
                **metadata_from(event),
            )

        elif operation == "access":

            pending_before = len(
                manager.pending_reuse_predictions
            )

            manager.access(event["block_id"])

            pending_after = len(
                manager.pending_reuse_predictions
            )

            # During an access event, completed prediction
            # records are finalized, but no new workload
            # metadata prediction is registered.

            completed = pending_before - pending_after

            if not (
                0 <= completed
                <= manager.REUSE_OBSERVATION_WINDOW
            ):
                raise AssertionError(
                    "Unexpected prediction lifecycle change"
                )

            phase = min(
                access_count // PHASE_SIZE,
                2,
            )

            if completed:

                latest = list(
                    manager.reuse_prediction_prequential_observations
                )[-completed:]

                if len(latest) != completed:
                    raise AssertionError(
                        "Completed observations missing"
                    )

                phase_records[phase].extend(latest)

            access_count += 1

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

    if access_count != TOTAL_ACCESSES:
        raise AssertionError(
            f"Expected {TOTAL_ACCESSES} accesses, "
            f"received {access_count}"
        )

    observations = [
        entry
        for phase in phase_records
        for entry in phase
    ]

    # Validate that our phase-level observation capture
    # agrees with MemoryManager's bounded history.

    retained = list(
        manager.reuse_prediction_prequential_observations
    )

    expected_retained = observations[
        -manager.REUSE_OBSERVATION_WINDOW:
    ]

    if retained != expected_retained:
        raise AssertionError(
            "Prequential history consistency failure"
        )

    manager_metrics = manager.get_prequential_reuse_metrics()

    if manager_metrics["observations"] != len(retained):
        raise AssertionError(
            "Prequential observation count mismatch"
        )

    report = manager.report()

    return {
        "phases": phase_records,
        "observations": observations,
        "hbm_hits": report["hbm_hits"],
        "accesses": access_count,
    }


def gather(results, scenario, family=None, phase=None):

    selected_families = (
        [family] if family is not None else FAMILIES
    )

    records = []

    for selected_family in selected_families:

        for seed in SEEDS:

            result = results[
                (scenario, selected_family, seed)
            ]

            if phase is None:
                records.extend(result["observations"])
            else:
                records.extend(result["phases"][phase])

    return calculate_metrics(records)


def fmt(value):

    if value is None:
        return "N/A"

    return f"{value:+.4f}"


def main():

    results = {}

    print("=" * 110)
    print("ACSS MEMCORE - PREQUENTIAL MONITOR EVALUATION")
    print("=" * 110)

    print("Seeds:", list(SEEDS))
    print("Families:", FAMILIES)
    print("Scenarios:", SCENARIOS)
    print("Policy: SLO_MARGIN")
    print("Metadata: informed")
    print("Phase size:", PHASE_SIZE)
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
                        f"No metadata shift: "
                        f"{scenario}, {family}, {seed}"
                    )

                results[
                    (scenario, family, seed)
                ] = replay(workload)

        print("Completed:", family)

    # ------------------------------------------------------
    # Check that changing metadata does not change the
    # baseline forecasts or observation outcomes.
    # ------------------------------------------------------

    print()
    print("=" * 110)
    print("CONTROLLED TRACE VALIDATION")
    print("=" * 110)

    comparisons = 0

    for family in FAMILIES:

        for seed in SEEDS:

            reference = results[
                ("CLEAN", family, seed)
            ]["observations"]

            baseline_signature = [
                (baseline, observed)
                for predicted, baseline, observed in reference
            ]

            for scenario in SCENARIOS[1:]:

                candidate = results[
                    (scenario, family, seed)
                ]["observations"]

                candidate_signature = [
                    (baseline, observed)
                    for predicted, baseline, observed in candidate
                ]

                if candidate_signature != baseline_signature:
                    raise AssertionError(
                        "Prediction outcomes or frozen "
                        "baselines differ between scenarios: "
                        f"{family}, {seed}, {scenario}"
                    )

                comparisons += 1

    print("Comparisons:", comparisons)
    print("Frozen baseline and outcome consistency: PASS")

    # ------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------

    print()
    print("=" * 110)
    print("OVERALL PREQUENTIAL FORECAST QUALITY")
    print("=" * 110)

    print(
        f"{'SCENARIO':24}"
        f"{'OBSERVATIONS':>15}"
        f"{'MODEL BRIER':>15}"
        f"{'BASELINE BRIER':>18}"
        f"{'EXCESS BRIER':>16}"
    )

    for scenario in SCENARIOS:

        metrics = gather(results, scenario)

        print(
            f"{scenario:24}"
            f"{metrics['observations']:15}"
            f"{metrics['model_brier']:15.4f}"
            f"{metrics['baseline_brier']:18.4f}"
            f"{metrics['excess_brier']:+16.4f}"
        )

    # ------------------------------------------------------
    # Per-family metrics
    # ------------------------------------------------------

    print()
    print("=" * 110)
    print("PER-FAMILY PREQUENTIAL EXCESS BRIER")
    print("=" * 110)

    print(
        f"{'FAMILY':24}"
        f"{'CLEAN':>15}"
        f"{'LATE SHIFT':>15}"
        f"{'TEMP SHIFT':>15}"
    )

    for family in FAMILIES:

        values = [
            gather(results, scenario, family)[
                "excess_brier"
            ]
            for scenario in SCENARIOS
        ]

        print(
            f"{family:24}"
            + "".join(
                f"{fmt(value):>15}"
                for value in values
            )
        )

    # ------------------------------------------------------
    # Temporary inversion phase diagnostics
    # ------------------------------------------------------

    print()
    print("=" * 110)
    print("TEMPORARY INVERSION - PHASE-LEVEL FORECAST QUALITY")
    print("=" * 110)

    print(
        f"{'FAMILY':22}"
        f"{'PHASE':>9}"
        f"{'OBS':>9}"
        f"{'MODEL':>12}"
        f"{'BASELINE':>12}"
        f"{'EXCESS':>12}"
        f"{'CLEAN EXCESS':>15}"
    )

    print("-" * 110)

    for family in FAMILIES:

        for phase in range(3):

            shifted = gather(
                results,
                "TEMPORARY_INVERSION",
                family,
                phase,
            )

            clean = gather(
                results,
                "CLEAN",
                family,
                phase,
            )

            print(
                f"{family:22}"
                f"{phase + 1:9}"
                f"{shifted['observations']:9}"
                f"{fmt(shifted['model_brier']):>12}"
                f"{fmt(shifted['baseline_brier']):>12}"
                f"{fmt(shifted['excess_brier']):>12}"
                f"{fmt(clean['excess_brier']):>15}"
            )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    expected = (
        len(FAMILIES)
        * len(SEEDS)
        * len(SCENARIOS)
    )

    print()
    print("=" * 110)
    print("VALIDATION")
    print("=" * 110)

    print("Expected executions:", expected)
    print("Completed executions:", len(results))

    assert len(results) == expected

    print("Execution count: PASS")
    print("All comparison outcomes were observed online.")
    print("Baselines were frozen before outcomes.")
    print("Synthetic simulator measurements only.")
    print("No production eviction policies were modified.")


if __name__ == "__main__":
    main()
