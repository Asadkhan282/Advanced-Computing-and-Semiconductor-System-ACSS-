
from types import MethodType

from benchmarks import generate_robustness_workloads as generator
from benchmarks.compare_prediction_shifts import apply_shift
from benchmarks.compare_guard_windows import (
    replay,
    make_recent_rank,
)
from memcore.memory import MemoryManager


SEEDS = range(300, 320)

FAMILIES = [
    "zipf",
    "bursty",
    "recurring",
    "prediction_mismatch",
]

SCENARIO = "TEMPORARY_INVERSION"
RECOVERY_START = 160
PHASE_SIZE = 80


def oracle_release_replay(workload):

    config = workload["memory"]

    manager = MemoryManager(
        hbm_capacity_bytes=config["hbm_capacity_bytes"],
        host_capacity_bytes=config["host_capacity_bytes"],
        policy="ADAPTIVE_MARGIN",
    )

    progress = {"accesses": 0}

    counters = {
        "rank_calls": 0,
        "failure_evidence_calls": 0,
    }

    release_calls = {"count": 0}

    recent_rank = make_recent_rank(
        20,
        counters,
    )

    def controlled_rank(self, contexts):

        if not contexts:
            return []

        # Oracle intervention:
        # At the known recovery boundary, force normal
        # SLO-margin ranking while retaining memory state.

        if progress["accesses"] >= RECOVERY_START:

            release_calls["count"] += 1

            ranked = self.scheduler.rank(
                contexts,
                now_ms=self.current_time_ms,
            )

            for item in ranked:
                item["reliability"] = 0.0

            return ranked

        return recent_rank(self, contexts)

    manager._rank_contexts_adaptively = MethodType(
        controlled_rank,
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

                current_hits = manager.report()["hbm_hits"]

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
        "accesses": progress["accesses"],
        "release_calls": release_calls["count"],
    }


def phase_rate(records, phase=2):

    hits = sum(
        record["phase_hits"][phase]
        for record in records
    )

    accesses = len(records) * PHASE_SIZE

    return hits / accesses if accesses else 0.0


def paired_counts(results, family, candidate_mode):

    wins = ties = losses = 0

    for seed in SEEDS:

        reference = results[
            (family, seed, "RAW_FIXED")
        ]["phase_hits"][2]

        candidate = results[
            (family, seed, candidate_mode)
        ]["phase_hits"][2]

        if candidate > reference:
            wins += 1
        elif candidate < reference:
            losses += 1
        else:
            ties += 1

    return wins, ties, losses


def main():

    results = {}

    print("=" * 110)
    print("ACSS MEMCORE - RECOVERY CARRYOVER DIAGNOSTIC")
    print("=" * 110)

    print("Seeds:", list(SEEDS))
    print("Families:", FAMILIES)
    print("Scenario:", SCENARIO)
    print("Oracle release after access:", RECOVERY_START)
    print("Evidence window: 20 observations")
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

            workload, changed = apply_shift(
                original,
                SCENARIO,
            )

            assert changed > 0

            baseline = replay(
                workload,
                "RAW_FIXED",
            )

            recent = replay(
                workload,
                "RECENT_20",
            )

            oracle = oracle_release_replay(
                workload
            )

            # The intervention must not alter decisions
            # before recovery begins.

            assert (
                oracle["phase_hits"][:2]
                == recent["phase_hits"][:2]
            ), (
                "Pre-recovery results differ: "
                f"{family}, seed={seed}"
            )

            results[(family, seed, "RAW_FIXED")] = baseline
            results[(family, seed, "RECENT_20")] = recent
            results[(family, seed, "ORACLE_RELEASE")] = oracle

        print("Completed:", family)

    print()
    print("=" * 110)
    print("PHASE 3 - RECOVERY HBM HIT RATE")
    print("=" * 110)

    print(
        f"{'FAMILY':22}"
        f"{'RAW %':>10}"
        f"{'R20 %':>10}"
        f"{'ORACLE %':>11}"
        f"{'ORACLE-R20':>13}"
        f"{'ORACLE-RAW':>13}"
    )

    print("-" * 110)

    for family in FAMILIES:

        rates = {}

        for mode in (
            "RAW_FIXED",
            "RECENT_20",
            "ORACLE_RELEASE",
        ):

            records = [
                results[(family, seed, mode)]
                for seed in SEEDS
            ]

            rates[mode] = phase_rate(records) * 100

        print(
            f"{family:22}"
            f"{rates['RAW_FIXED']:9.2f}%"
            f"{rates['RECENT_20']:9.2f}%"
            f"{rates['ORACLE_RELEASE']:10.2f}%"
            f"{rates['ORACLE_RELEASE'] - rates['RECENT_20']:+12.2f}"
            f"{rates['ORACLE_RELEASE'] - rates['RAW_FIXED']:+12.2f}"
        )

    print()
    print("=" * 110)
    print("RECOVERY PAIRED WINS / TIES / LOSSES VS BASELINE")
    print("=" * 110)

    for family in FAMILIES:

        for mode in ("RECENT_20", "ORACLE_RELEASE"):

            wins, ties, losses = paired_counts(
                results,
                family,
                mode,
            )

            print(
                f"{family:22}"
                f"{mode:17}"
                f"W/T/L={wins}/{ties}/{losses}"
            )

    print()
    print("=" * 110)
    print("AGGREGATED RECOVERY RESULTS")
    print("=" * 110)

    all_rates = {}

    for mode in (
        "RAW_FIXED",
        "RECENT_20",
        "ORACLE_RELEASE",
    ):

        records = [
            results[(family, seed, mode)]
            for family in FAMILIES
            for seed in SEEDS
        ]

        all_rates[mode] = phase_rate(records) * 100

        print(
            f"{mode:20}"
            f"{all_rates[mode]:8.2f}%"
        )

    print()
    print(
        "Oracle improvement over Recent 20:",
        f"{all_rates['ORACLE_RELEASE'] - all_rates['RECENT_20']:+.2f} pp",
    )

    print(
        "Oracle difference from raw baseline:",
        f"{all_rates['ORACLE_RELEASE'] - all_rates['RAW_FIXED']:+.2f} pp",
    )

    release_calls = sum(
        results[
            (family, seed, "ORACLE_RELEASE")
        ]["release_calls"]
        for family in FAMILIES
        for seed in SEEDS
    )

    print("Oracle-release ranking calls:", release_calls)

    print()
    print("=" * 110)
    print("VALIDATION")
    print("=" * 110)

    expected = (
        len(FAMILIES)
        * len(SEEDS)
        * 3
    )

    print("Expected executions:", expected)
    print("Completed executions:", len(results))

    assert len(results) == expected

    print("Execution count: PASS")
    print("Pre-recovery equivalence: PASS")
    print("HBM state is never reset at recovery.")
    print("Oracle uses a known recovery boundary.")
    print("Results are synthetic simulator measurements.")
    print("Production MemCore remains unchanged.")


if __name__ == "__main__":
    main()
