
from types import MethodType

from benchmarks import generate_robustness_workloads as generator
from memcore.memory import MemoryManager


SEEDS = range(42, 52)
FAMILIES = list(generator.FAMILIES)

MODES = [
    "SLO_MARGIN",
    "RAW_FIXED",
    "ADAPTIVE_REUSE_ONLY",
    "ADAPTIVE_MARGIN_ONLY",
    "BOTH_ADAPTIVE",
    "NATIVE_ADAPTIVE",
]

EXPERIMENTS = {
    "RAW_FIXED": (False, False),
    "ADAPTIVE_REUSE_ONLY": (True, False),
    "ADAPTIVE_MARGIN_ONLY": (False, True),
    "BOTH_ADAPTIVE": (True, True),
}


def make_rank(adaptive_reuse, adaptive_margin):

    def experimental_rank(self, contexts):

        if not contexts:
            return []

        prevalence = self.get_observed_reuse_rate()

        adjusted_contexts = []
        details = {}

        for context in contexts:

            block_id = context["block_id"]

            predicted = float(
                context["reuse_probability"]
            )

            context_reliability = float(
                self.get_context_reuse_reliability(block_id)
            )

            # Factor 1: Should reliability modify the
            # reuse probability used by the scheduler?

            if adaptive_reuse:

                effective_reuse = (
                    context_reliability * predicted
                    + (1.0 - context_reliability)
                    * prevalence
                )

            else:
                effective_reuse = predicted

            # Factor 2: Should reliability modify the
            # SLO margin used to select eviction victims?
            #
            # ADAPTIVE_MARGIN uses:
            # margin = slo_margin * (1 - reliability)
            #
            # reliability=0 restores the full fixed margin.

            if adaptive_margin:
                margin_reliability = context_reliability
            else:
                margin_reliability = 0.0

            item = dict(context)

            item["reuse_probability"] = effective_reuse

            adjusted_contexts.append(item)

            details[block_id] = {
                "reliability": margin_reliability,
                "context_reliability": context_reliability,
                "predicted_reuse_probability": predicted,
                "effective_reuse_probability": effective_reuse,
            }

        ranked = self.scheduler.rank(
            adjusted_contexts,
            now_ms=self.current_time_ms,
        )

        for item in ranked:
            item.update(details[item["block_id"]])

        return ranked

    return experimental_rank


def execute(workload, mode):

    config = workload["memory"]

    policy = (
        "SLO_MARGIN"
        if mode == "SLO_MARGIN"
        else "ADAPTIVE_MARGIN"
    )

    manager = MemoryManager(
        hbm_capacity_bytes=config["hbm_capacity_bytes"],
        host_capacity_bytes=config["host_capacity_bytes"],
        policy=policy,
    )

    if mode in EXPERIMENTS:

        adaptive_reuse, adaptive_margin = EXPERIMENTS[mode]

        manager._rank_contexts_adaptively = MethodType(
            make_rank(
                adaptive_reuse=adaptive_reuse,
                adaptive_margin=adaptive_margin,
            ),
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
                f"Unknown workload operation: {operation}"
            )

        manager.validate()

    metrics = manager.report()

    hbm_hits = metrics["hbm_hits"]
    host_hits = metrics["host_hits"]
    cold_misses = metrics["cold_misses"]

    if hbm_hits + host_hits + cold_misses != access_requests:
        raise AssertionError(
            "Access accounting mismatch."
        )

    return {
        "access_requests": access_requests,
        "hbm_hits": hbm_hits,
        "host_hits": host_hits,
        "cold_misses": cold_misses,
        "hbm_rate": (
            hbm_hits / access_requests
            if access_requests
            else 0.0
        ),
    }


def mean_rate(records):

    total_hits = sum(
        record["hbm_hits"]
        for record in records
    )

    total_accesses = sum(
        record["access_requests"]
        for record in records
    )

    if total_accesses == 0:
        return 0.0

    return total_hits / total_accesses


def same_result(a, b):

    return all(
        a[key] == b[key]
        for key in (
            "access_requests",
            "hbm_hits",
            "host_hits",
            "cold_misses",
        )
    )


def main():

    results = {}

    print("=" * 112)
    print("ACSS MEMCORE - CONTROLLED ADAPTIVE POLICY ABLATION")
    print("=" * 112)

    print("Seeds:", list(SEEDS))
    print("Families:", FAMILIES)
    print("Metadata: informed")
    print("HBM capacity: 4 context blocks")
    print("Host capacity: 8 context blocks")
    print("Modes:", MODES)
    print()

    for family in FAMILIES:

        print(f"Running family: {family}")

        for seed in SEEDS:

            workload = generator.build_workload(
                family=family,
                seed=seed,
                hbm_blocks=4,
                host_blocks=8,
                metadata_mode="informed",
            )

            for mode in MODES:

                result = execute(
                    workload=workload,
                    mode=mode,
                )

                results[(family, seed, mode)] = result

    print()
    print("=" * 112)
    print("VALIDATION CHECKS")
    print("=" * 112)

    raw_matches = 0
    native_matches = 0
    total_cases = len(FAMILIES) * len(SEEDS)

    for family in FAMILIES:

        for seed in SEEDS:

            baseline = results[
                (family, seed, "SLO_MARGIN")
            ]

            raw_fixed = results[
                (family, seed, "RAW_FIXED")
            ]

            experimental_native = results[
                (family, seed, "BOTH_ADAPTIVE")
            ]

            actual_native = results[
                (family, seed, "NATIVE_ADAPTIVE")
            ]

            if same_result(baseline, raw_fixed):
                raw_matches += 1

            if same_result(
                experimental_native,
                actual_native,
            ):
                native_matches += 1

    print(
        f"RAW_FIXED vs SLO_MARGIN: "
        f"{raw_matches}/{total_cases} matching"
    )

    print(
        f"BOTH_ADAPTIVE vs NATIVE_ADAPTIVE: "
        f"{native_matches}/{total_cases} matching"
    )

    if native_matches != total_cases:
        print(
            "WARNING: Experimental adaptive behavior "
            "does not fully reproduce production."
        )

    if raw_matches != total_cases:
        print(
            "WARNING: RAW_FIXED and SLO_MARGIN differ. "
            "Review victim-selection differences before "
            "treating them as equivalent."
        )

    print()
    print("=" * 112)
    print("PER-FAMILY RESULTS: MEAN HBM HIT RATE")
    print("=" * 112)

    print(
        f"{'FAMILY':24}"
        f"{'SLO_MARGIN':>13}"
        f"{'RAW_FIXED':>13}"
        f"{'REUSE_ONLY':>13}"
        f"{'MARGIN_ONLY':>13}"
        f"{'BOTH':>13}"
    )

    print("-" * 112)

    display_modes = [
        "SLO_MARGIN",
        "RAW_FIXED",
        "ADAPTIVE_REUSE_ONLY",
        "ADAPTIVE_MARGIN_ONLY",
        "BOTH_ADAPTIVE",
    ]

    for family in FAMILIES:

        rates = []

        for mode in display_modes:

            records = [
                results[(family, seed, mode)]
                for seed in SEEDS
            ]

            rates.append(mean_rate(records) * 100)

        print(
            f"{family:24}"
            + "".join(
                f"{rate:12.2f}%"
                for rate in rates
            )
        )

    print()
    print("=" * 112)
    print("OVERALL RESULTS")
    print("=" * 112)

    baseline_records = [
        results[(family, seed, "SLO_MARGIN")]
        for family in FAMILIES
        for seed in SEEDS
    ]

    baseline_rate = mean_rate(baseline_records)

    print(
        f"{'MODE':26}"
        f"{'HBM HIT %':>13}"
        f"{'DELTA (pp)':>14}"
        f"{'WINS':>8}"
        f"{'TIES':>8}"
        f"{'LOSSES':>9}"
    )

    print("-" * 112)

    for mode in MODES:

        records = [
            results[(family, seed, mode)]
            for family in FAMILIES
            for seed in SEEDS
        ]

        rate = mean_rate(records)

        wins = 0
        ties = 0
        losses = 0

        for family in FAMILIES:

            for seed in SEEDS:

                candidate = results[
                    (family, seed, mode)
                ]["hbm_hits"]

                reference = results[
                    (family, seed, "SLO_MARGIN")
                ]["hbm_hits"]

                if candidate > reference:
                    wins += 1
                elif candidate == reference:
                    ties += 1
                else:
                    losses += 1

        delta_pp = (
            rate - baseline_rate
        ) * 100

        print(
            f"{mode:26}"
            f"{rate * 100:12.2f}%"
            f"{delta_pp:+13.2f}"
            f"{wins:8}"
            f"{ties:8}"
            f"{losses:9}"
        )

    print()
    print("Total paired cases:", total_cases)
    print("Results describe synthetic workloads only.")
    print("No GPU latency or hardware performance measured.")


if __name__ == "__main__":
    main()
