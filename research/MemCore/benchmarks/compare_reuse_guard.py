
from types import MethodType

from benchmarks import generate_robustness_workloads as generator
from memcore.memory import MemoryManager


SEEDS = range(42, 52)
FAMILIES = list(generator.FAMILIES)

MODES = [
    "RAW_FIXED",
    "CONTEXT_BLEND",
    "GLOBAL_GUARD_SOFT",
    "GLOBAL_GUARD_HARD",
]

MIN_OBSERVATIONS = 20
BRIER_DISADVANTAGE = 0.02


def make_rank(mode, counters):

    def experimental_rank(self, contexts):

        if not contexts:
            return []

        history = list(self.reuse_prediction_observations)
        prevalence = self.get_observed_reuse_rate()

        # All observations used here are historical.
        # The manager does not inspect future workload events.

        if history:

            predicted_brier = sum(
                (predicted - observed) ** 2
                for predicted, observed in history
            ) / len(history)

            baseline_brier = sum(
                (prevalence - observed) ** 2
                for _, observed in history
            ) / len(history)

        else:
            predicted_brier = 0.0
            baseline_brier = 0.0

        evidence_of_failure = (
            len(history) >= MIN_OBSERVATIONS
            and predicted_brier
                > baseline_brier + BRIER_DISADVANTAGE
        )

        counters["rank_calls"] += 1

        if evidence_of_failure:
            counters["failure_evidence_calls"] += 1

        adjusted_contexts = []

        for context in contexts:

            block_id = context["block_id"]
            predicted = float(context["reuse_probability"])

            if mode == "CONTEXT_BLEND":

                trust = self.get_context_reuse_reliability(
                    block_id
                )

            elif mode == "GLOBAL_GUARD_SOFT":

                trust = (
                    0.5
                    if evidence_of_failure
                    else 1.0
                )

            elif mode == "GLOBAL_GUARD_HARD":

                trust = (
                    0.0
                    if evidence_of_failure
                    else 1.0
                )

            else:
                raise ValueError(mode)

            effective = (
                trust * predicted
                + (1.0 - trust) * prevalence
            )

            item = dict(context)
            item["reuse_probability"] = effective

            adjusted_contexts.append(item)

        ranked = self.scheduler.rank(
            adjusted_contexts,
            now_ms=self.current_time_ms,
        )

        for item in ranked:
            # Keep the eviction margin fixed.
            # Only the reuse-probability mechanism changes.
            item["reliability"] = 0.0

        return ranked

    return experimental_rank


def execute(workload, mode):

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

    counters = {
        "rank_calls": 0,
        "failure_evidence_calls": 0,
    }

    if mode != "RAW_FIXED":

        manager._rank_contexts_adaptively = MethodType(
            make_rank(mode, counters),
            manager,
        )

    access_requests = 0

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
            access_requests += 1

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

    metrics = manager.report()

    hits = metrics["hbm_hits"]
    host_hits = metrics["host_hits"]
    cold_misses = metrics["cold_misses"]

    assert (
        hits + host_hits + cold_misses
        == access_requests
    ), "Access accounting mismatch"

    return {
        "hbm_hits": hits,
        "host_hits": host_hits,
        "cold_misses": cold_misses,
        "access_requests": access_requests,
        **counters,
    }


def hit_rate(records):

    hits = sum(
        record["hbm_hits"]
        for record in records
    )

    accesses = sum(
        record["access_requests"]
        for record in records
    )

    return hits / accesses if accesses else 0.0


def main():

    results = {}

    print("=" * 105)
    print("ACSS MEMCORE - EVIDENCE-GATED REUSE EXPERIMENT")
    print("=" * 105)

    print("Seeds:", list(SEEDS))
    print("Families:", FAMILIES)
    print("Modes:", MODES)
    print("Metadata: informed")
    print("Minimum observations:", MIN_OBSERVATIONS)
    print("Brier disadvantage:", BRIER_DISADVANTAGE)
    print()

    for family in FAMILIES:

        print("Running:", family)

        for seed in SEEDS:

            workload = generator.build_workload(
                family=family,
                seed=seed,
                hbm_blocks=4,
                host_blocks=8,
                metadata_mode="informed",
            )

            for mode in MODES:

                results[(family, seed, mode)] = execute(
                    workload,
                    mode,
                )

    print()
    print("=" * 105)
    print("PER-FAMILY HBM HIT RATE")
    print("=" * 105)

    print(
        f"{'FAMILY':24}"
        f"{'RAW':>12}"
        f"{'CONTEXT':>12}"
        f"{'SOFT':>12}"
        f"{'HARD':>12}"
    )

    for family in FAMILIES:

        rates = []

        for mode in MODES:

            records = [
                results[(family, seed, mode)]
                for seed in SEEDS
            ]

            rates.append(
                hit_rate(records) * 100
            )

        print(
            f"{family:24}"
            + "".join(
                f"{rate:11.2f}%"
                for rate in rates
            )
        )

    print()
    print("=" * 105)
    print("OVERALL COMPARISON")
    print("=" * 105)

    baseline_records = [
        results[(family, seed, "RAW_FIXED")]
        for family in FAMILIES
        for seed in SEEDS
    ]

    baseline = hit_rate(baseline_records)

    print(
        f"{'MODE':24}"
        f"{'HBM HIT %':>12}"
        f"{'DELTA pp':>12}"
        f"{'WINS':>8}"
        f"{'TIES':>8}"
        f"{'LOSSES':>9}"
    )

    for mode in MODES:

        records = [
            results[(family, seed, mode)]
            for family in FAMILIES
            for seed in SEEDS
        ]

        rate = hit_rate(records)

        wins = ties = losses = 0

        for family in FAMILIES:
            for seed in SEEDS:

                candidate = results[
                    (family, seed, mode)
                ]["hbm_hits"]

                reference = results[
                    (family, seed, "RAW_FIXED")
                ]["hbm_hits"]

                if candidate > reference:
                    wins += 1
                elif candidate < reference:
                    losses += 1
                else:
                    ties += 1

        print(
            f"{mode:24}"
            f"{rate * 100:11.2f}%"
            f"{(rate-baseline)*100:+11.2f}"
            f"{wins:8}"
            f"{ties:8}"
            f"{losses:9}"
        )

    print()
    print("=" * 105)
    print("GUARD ACTIVATION DIAGNOSTICS")
    print("=" * 105)

    for mode in MODES[1:]:

        calls = sum(
            results[(family, seed, mode)]["rank_calls"]
            for family in FAMILIES
            for seed in SEEDS
        )

        failures = sum(
            results[(family, seed, mode)][
                "failure_evidence_calls"
            ]
            for family in FAMILIES
            for seed in SEEDS
        )

        percentage = (
            100 * failures / calls
            if calls else 0.0
        )

        print(
            f"{mode:24}"
            f"rankings={calls:6} "
            f"failure_evidence={failures:6} "
            f"({percentage:.2f}%)"
        )

    print()
    print("Experimental synthetic simulator results only.")
    print("Thresholds are exploratory, not validated.")
    print("No production memory policies were changed.")


if __name__ == "__main__":
    main()
