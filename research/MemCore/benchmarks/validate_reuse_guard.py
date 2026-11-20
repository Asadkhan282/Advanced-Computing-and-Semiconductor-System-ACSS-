
from benchmarks import generate_robustness_workloads as generator
from benchmarks.compare_reuse_guard import (
    execute,
    MODES,
    MIN_OBSERVATIONS,
    BRIER_DISADVANTAGE,
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

SEEDS = range(100, 120)

FAMILIES = list(generator.FAMILIES)

METADATA_MODES = [
    "default_metadata",
    "informed",
]


# ============================================================
# RESULT SUMMARIZATION
# ============================================================

def summarize(records):

    hits = sum(
        record["hbm_hits"]
        for record in records
    )

    accesses = sum(
        record["access_requests"]
        for record in records
    )

    host_hits = sum(
        record["host_hits"]
        for record in records
    )

    cold_misses = sum(
        record["cold_misses"]
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
        "rate": (
            hits / accesses
            if accesses
            else 0.0
        ),
        "hits": hits,
        "host_hits": host_hits,
        "cold_misses": cold_misses,
        "accesses": accesses,
        "rank_calls": rank_calls,
        "guard_calls": guard_calls,
    }


# ============================================================
# PAIRED COMPARISON
# ============================================================

def paired_counts(
    results,
    metadata_mode,
    families,
    mode,
):

    wins = 0
    ties = 0
    losses = 0

    for family in families:

        for seed in SEEDS:

            candidate = results[
                (
                    metadata_mode,
                    family,
                    seed,
                    mode,
                )
            ]["hbm_hits"]

            baseline = results[
                (
                    metadata_mode,
                    family,
                    seed,
                    "RAW_FIXED",
                )
            ]["hbm_hits"]

            if candidate > baseline:
                wins += 1

            elif candidate < baseline:
                losses += 1

            else:
                ties += 1

    return wins, ties, losses


# ============================================================
# COLLECT EXPERIMENT RESULTS
# ============================================================

def collect(
    results,
    metadata_modes,
    families,
    mode,
):

    return [
        results[
            (
                metadata,
                family,
                seed,
                mode,
            )
        ]
        for metadata in metadata_modes
        for family in families
        for seed in SEEDS
    ]


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    results = {}

    total_workloads = (
        len(SEEDS)
        * len(FAMILIES)
        * len(METADATA_MODES)
    )

    total_executions = (
        total_workloads
        * len(MODES)
    )

    print("=" * 110)
    print(
        "ACSS MEMCORE - HELD-OUT REUSE GUARD VALIDATION"
    )
    print("=" * 110)

    print("Seeds:", list(SEEDS))
    print("Families:", FAMILIES)
    print("Metadata modes:", METADATA_MODES)
    print("Policies:", MODES)

    print(
        "Guard minimum observations:",
        MIN_OBSERVATIONS,
    )

    print(
        "Brier disadvantage:",
        BRIER_DISADVANTAGE,
    )

    print("Total workload instances:", total_workloads)
    print("Total policy executions:", total_executions)
    print()

    # ========================================================
    # EXECUTE WORKLOADS
    # ========================================================

    for metadata in METADATA_MODES:

        for family in FAMILIES:

            print(
                f"Running {metadata:18} / {family}"
            )

            for seed in SEEDS:

                workload = generator.build_workload(
                    family=family,
                    seed=seed,
                    hbm_blocks=4,
                    host_blocks=8,
                    metadata_mode=metadata,
                )

                for mode in MODES:

                    result = execute(
                        workload,
                        mode,
                    )

                    results[
                        (
                            metadata,
                            family,
                            seed,
                            mode,
                        )
                    ] = result

    # ========================================================
    # PER-FAMILY RESULTS
    # ========================================================

    print()
    print("=" * 110)
    print("PER-FAMILY HBM HIT RATE")
    print("=" * 110)

    print(
        f"{'METADATA':18} "
        f"{'FAMILY':22} "
        f"{'RAW %':>9} "
        f"{'SOFT %':>9} "
        f"{'HARD %':>9} "
        f"{'HARD DELTA':>12}"
    )

    print("-" * 110)

    for metadata in METADATA_MODES:

        for family in FAMILIES:

            rates = {}

            for mode in MODES:

                records = collect(
                    results,
                    [metadata],
                    [family],
                    mode,
                )

                rates[mode] = (
                    summarize(records)["rate"]
                    * 100
                )

            delta = (
                rates["GLOBAL_GUARD_HARD"]
                - rates["RAW_FIXED"]
            )

            print(
                f"{metadata:18} "
                f"{family:22} "
                f"{rates['RAW_FIXED']:8.2f}% "
                f"{rates['GLOBAL_GUARD_SOFT']:8.2f}% "
                f"{rates['GLOBAL_GUARD_HARD']:8.2f}% "
                f"{delta:+11.2f} pp"
            )

    # ========================================================
    # OVERALL HELD-OUT RESULTS
    # ========================================================

    print()
    print("=" * 110)
    print("HELD-OUT OVERALL RESULTS")
    print("=" * 110)

    metadata_groups = [
        ["default_metadata"],
        ["informed"],
        METADATA_MODES,
    ]

    for metadata_group in metadata_groups:

        label = "+".join(metadata_group)

        baseline_records = collect(
            results,
            metadata_group,
            FAMILIES,
            "RAW_FIXED",
        )

        baseline = summarize(
            baseline_records
        )["rate"]

        print()
        print(f"Metadata: {label}")

        print("-" * 85)

        print(
            f"{'MODE':24} "
            f"{'HBM %':>9} "
            f"{'DELTA pp':>10} "
            f"{'WINS':>6} "
            f"{'TIES':>6} "
            f"{'LOSSES':>7}"
        )

        print("-" * 85)

        for mode in MODES:

            records = collect(
                results,
                metadata_group,
                FAMILIES,
                mode,
            )

            stats = summarize(records)

            wins = 0
            ties = 0
            losses = 0

            for metadata in metadata_group:

                w, t, l = paired_counts(
                    results,
                    metadata,
                    FAMILIES,
                    mode,
                )

                wins += w
                ties += t
                losses += l

            delta_pp = (
                stats["rate"] - baseline
            ) * 100

            print(
                f"{mode:24} "
                f"{stats['rate'] * 100:8.2f}% "
                f"{delta_pp:+9.2f} "
                f"{wins:6} "
                f"{ties:6} "
                f"{losses:7}"
            )

    # ========================================================
    # GUARD ACTIVATION DIAGNOSTICS
    # ========================================================

    print()
    print("=" * 110)
    print("GUARD ACTIVATION BY FAMILY")
    print("=" * 110)

    for metadata in METADATA_MODES:

        for family in FAMILIES:

            records = collect(
                results,
                [metadata],
                [family],
                "GLOBAL_GUARD_HARD",
            )

            stats = summarize(records)

            calls = stats["rank_calls"]

            activated = stats["guard_calls"]

            percentage = (
                100.0 * activated / calls
                if calls
                else 0.0
            )

            print(
                f"{metadata:18} "
                f"{family:22} "
                f"{activated:6}/{calls:<6} "
                f"({percentage:6.2f}%)"
            )

    # ========================================================
    # WORST PAIRED OUTCOMES
    # ========================================================

    print()
    print("=" * 110)
    print("HARD GUARD - WORST PAIRED OUTCOMES")
    print("=" * 110)

    differences = []

    for metadata in METADATA_MODES:

        for family in FAMILIES:

            for seed in SEEDS:

                baseline_hits = results[
                    (
                        metadata,
                        family,
                        seed,
                        "RAW_FIXED",
                    )
                ]["hbm_hits"]

                hard_hits = results[
                    (
                        metadata,
                        family,
                        seed,
                        "GLOBAL_GUARD_HARD",
                    )
                ]["hbm_hits"]

                differences.append(
                    (
                        hard_hits - baseline_hits,
                        metadata,
                        family,
                        seed,
                    )
                )

    for (
        delta,
        metadata,
        family,
        seed,
    ) in sorted(differences)[:10]:

        print(
            f"metadata={metadata:18} "
            f"family={family:22} "
            f"seed={seed:3} "
            f"HBM_hit_delta={delta:+d}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 110)
    print("VALIDATION SUMMARY")
    print("=" * 110)

    print(
        "Completed workload instances:",
        total_workloads,
    )

    print(
        "Completed policy executions:",
        len(results),
    )

    if len(results) != total_executions:

        raise AssertionError(
            "Experiment execution count mismatch."
        )

    print("Execution count: PASS")

    print(
        "Held-out seeds:",
        SEEDS.start,
        "to",
        SEEDS.stop - 1,
    )

    print(
        "Guard thresholds unchanged:",
        MIN_OBSERVATIONS,
        BRIER_DISADVANTAGE,
    )

    print()
    print(
        "Results describe synthetic simulator behavior."
    )

    print(
        "No GPU latency, energy, or hardware "
        "performance was measured."
    )

    print(
        "No MemCore production policies were changed."
    )


if __name__ == "__main__":
    main()
