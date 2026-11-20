
from benchmarks import generate_robustness_workloads as generator
from benchmarks.compare_prediction_shifts import apply_shift
from benchmarks.compare_prequential_guard import execute
from memcore.memory import MemoryManager


SEEDS = range(900, 910)
FAMILIES = list(generator.FAMILIES)

CONDITIONS = [
    ("default_metadata", "CLEAN"),
    ("informed", "CLEAN"),
    ("informed", "LATE_INVERSION"),
    ("informed", "TEMPORARY_INVERSION"),
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


def replay_production(workload):

    config = workload["memory"]

    manager = MemoryManager(
        hbm_capacity_bytes=config["hbm_capacity_bytes"],
        host_capacity_bytes=config["host_capacity_bytes"],
        policy="PREQUENTIAL_SLO_MARGIN",
    )

    access_count = 0
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
            access_count += 1

            if access_count % PHASE_SIZE == 0:

                current_hits = manager.report()["hbm_hits"]

                phase_hits.append(
                    current_hits - previous_hits
                )

                previous_hits = current_hits

        elif operation == "advance_time":

            manager.advance_time(event["delta_ms"])

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

    assert access_count == TOTAL_ACCESSES
    assert len(phase_hits) == 3
    assert sum(phase_hits) == report["hbm_hits"]

    assert (
        report["hbm_hits"]
        + report["host_hits"]
        + report["cold_misses"]
        == access_count
    )

    return {
        "hbm_hits": report["hbm_hits"],
        "phase_hits": phase_hits,
        "accesses": access_count,
    }


def compare(reference, production):

    return (
        reference["hbm_hits"] == production["hbm_hits"]
        and reference["phase_hits"] == production["phase_hits"]
        and reference["accesses"] == production["accesses"]
    )


def main():

    matches = 0
    mismatches = []
    total = 0

    print("=" * 110)
    print("ACSS MEMCORE - PRODUCTION PREQUENTIAL EQUIVALENCE")
    print("=" * 110)

    print("Reference: PREQ_20_FAST12")
    print("Production: PREQUENTIAL_SLO_MARGIN")
    print("Seeds:", list(SEEDS))
    print("Families:", FAMILIES)
    print("Conditions:", CONDITIONS)
    print()

    for metadata, scenario in CONDITIONS:

        condition_matches = 0
        condition_total = 0

        for family in FAMILIES:

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

                if scenario != "CLEAN":
                    assert changed > 0

                reference = execute(
                    workload,
                    "PREQ_20_FAST12",
                )

                production = replay_production(workload)

                total += 1
                condition_total += 1

                if compare(reference, production):

                    matches += 1
                    condition_matches += 1

                else:

                    mismatches.append({
                        "metadata": metadata,
                        "scenario": scenario,
                        "family": family,
                        "seed": seed,
                        "reference_hits": reference["hbm_hits"],
                        "production_hits": production["hbm_hits"],
                        "reference_phases": reference["phase_hits"],
                        "production_phases": production["phase_hits"],
                    })

        print(
            f"{metadata:18} "
            f"{scenario:22} "
            f"matches={condition_matches}/{condition_total}"
        )

    print()
    print("=" * 110)
    print("EQUIVALENCE SUMMARY")
    print("=" * 110)

    print("Compared workload instances:", total)
    print("Matching instances:", matches)
    print("Mismatching instances:", len(mismatches))

    print()

    if mismatches:

        print("MISMATCH DETAILS")
        print("-" * 110)

        for mismatch in mismatches[:15]:

            print(
                f"{mismatch['metadata']} "
                f"{mismatch['scenario']} "
                f"{mismatch['family']} "
                f"seed={mismatch['seed']}"
            )

            print(
                "  Reference hits:",
                mismatch["reference_hits"],
            )

            print(
                "  Production hits:",
                mismatch["production_hits"],
            )

            print(
                "  Reference phases:",
                mismatch["reference_phases"],
            )

            print(
                "  Production phases:",
                mismatch["production_phases"],
            )

        print()
        print("EQUIVALENCE: FAIL")

        raise AssertionError(
            f"{len(mismatches)} workload instances differ."
        )

    print("EQUIVALENCE: PASS")
    print("Aggregate and phase-level HBM hits match.")
    print("Production implementation matches research results.")
    print("Synthetic simulator only.")
    print("No production files modified.")


if __name__ == "__main__":
    main()
