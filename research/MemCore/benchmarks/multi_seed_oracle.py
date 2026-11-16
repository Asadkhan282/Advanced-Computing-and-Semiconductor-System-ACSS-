
import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parent

sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from generate_robustness_workloads import build_workload
from offline_hbm_oracle import solve_workload
from memcore.simulator import run_workload


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

POLICIES = (
    "LRU",
    "FIFO",
    "SLO",
    "SLO_MARGIN",
    "ADAPTIVE_SLO",
    "ADAPTIVE_MARGIN",
)

FAMILIES = (
    "uniform",
    "zipf",
    "bursty",
    "recurring",
    "prediction_mismatch",
    "capacity_pressure",
)

CONTEXTS = 6

# Capture two 30-access phases in the bursty workload.
ACCESS_COUNT = 60

HBM_BLOCKS = 3
HOST_BLOCKS = 3

DEFAULT_SEEDS = 10
MAX_SEEDS = 20

METADATA_MODE_ALIASES = {
    "default": "default_metadata",
    "default_metadata": "default_metadata",
    "informed": "informed",
}


# ============================================================
# METADATA MODE NORMALIZATION
# ============================================================

def normalize_metadata_mode(metadata_mode):

    if metadata_mode not in METADATA_MODE_ALIASES:
        raise ValueError(
            f"Unsupported metadata mode: {metadata_mode}. "
            f"Supported modes: {tuple(METADATA_MODE_ALIASES)}"
        )

    return METADATA_MODE_ALIASES[metadata_mode]


# ============================================================
# RESULT VALIDATION
# ============================================================

def validate_result(result, expected_accesses):

    memory = result["memory_metrics"]

    hbm_misses = (
        memory["host_hits"]
        + memory["cold_misses"]
    )

    actual_accesses = (
        result["hbm_hits"]
        + result["host_hits"]
        + result["cold_misses"]
    )

    if actual_accesses != expected_accesses:
        raise RuntimeError(
            f"Expected {expected_accesses} accesses, "
            f"got {actual_accesses}."
        )

    return {
        "hbm_hits": result["hbm_hits"],
        "host_hits": result["host_hits"],
        "cold_misses": result["cold_misses"],
        "hbm_misses": hbm_misses,
        "hit_rate": (
            result["hbm_hits"] / actual_accesses
            if actual_accesses else 0.0
        ),
    }


# ============================================================
# WORKLOAD VALIDATION
# ============================================================

def validate_workload(workload, seed):

    allocations = [
        event
        for event in workload["events"]
        if event["operation"] == "allocate"
    ]

    allocated_ids = {
        event["block_id"]
        for event in allocations
    }

    if len(allocated_ids) != CONTEXTS:
        raise RuntimeError(
            f"Seed {seed}: expected exactly {CONTEXTS} "
            f"unique contexts, found {len(allocated_ids)}."
        )

    total_capacity = HBM_BLOCKS + HOST_BLOCKS

    if len(allocated_ids) > total_capacity:
        raise RuntimeError(
            f"Seed {seed}: context count exceeds "
            "combined HBM and host capacity."
        )

    memory_config = workload["memory"]

    if memory_config["hbm_capacity_bytes"] <= 0:
        raise RuntimeError(
            f"Seed {seed}: invalid HBM capacity."
        )

    if memory_config["host_capacity_bytes"] <= 0:
        raise RuntimeError(
            f"Seed {seed}: invalid host capacity."
        )


# ============================================================
# WORKLOAD TRIMMING
# ============================================================

def trim_workload(workload, seed):

    trimmed_events = []
    access_seen = 0

    for event in workload["events"]:

        trimmed_events.append(event)

        if event["operation"] == "access":

            access_seen += 1

            if access_seen == ACCESS_COUNT:
                break

    if access_seen != ACCESS_COUNT:
        raise RuntimeError(
            f"Seed {seed}: could not obtain "
            f"{ACCESS_COUNT} access requests."
        )

    workload["events"] = trimmed_events

    actual_accesses = sum(
        event["operation"] == "access"
        for event in workload["events"]
    )

    if actual_accesses != ACCESS_COUNT:
        raise RuntimeError(
            f"Seed {seed}: expected {ACCESS_COUNT} "
            f"accesses after trimming, got {actual_accesses}."
        )

    return workload


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(family, seeds, metadata_mode):

    metadata_mode = normalize_metadata_mode(metadata_mode)

    records = []
    oracle_states = []
    oracle_seconds = []

    print("=" * 88)
    print("MEMCORE MULTI-SEED EXACT ORACLE EVALUATION")
    print("=" * 88)

    print(f"Family: {family}")
    print(f"Metadata mode: {metadata_mode}")
    print(f"Seeds: {len(seeds)}")
    print(f"Contexts: {CONTEXTS}")
    print(f"Accesses per trace: {ACCESS_COUNT}")
    print(f"HBM blocks: {HBM_BLOCKS}")
    print(f"Host blocks: {HOST_BLOCKS}")
    print(f"Policies: {len(POLICIES)}")
    print("")

    with tempfile.TemporaryDirectory(
        prefix="memcore_multi_seed_"
    ) as temporary_directory:

        temporary_path = Path(temporary_directory)

        for seed in seeds:

            # ------------------------------------------------
            # Generate workload with exactly six contexts.
            # ------------------------------------------------

            workload = build_workload(
                family=family,
                seed=seed,
                hbm_blocks=HBM_BLOCKS,
                host_blocks=HOST_BLOCKS,
                metadata_mode=metadata_mode,
                context_count=CONTEXTS,
            )

            validate_workload(workload, seed)

            # ------------------------------------------------
            # Preserve events through ACCESS_COUNT accesses.
            # ------------------------------------------------

            workload = trim_workload(workload, seed)

            # ------------------------------------------------
            # Save temporary workload.
            # ------------------------------------------------

            workload_path = temporary_path / (
                f"{family}_{metadata_mode}_{seed}.json"
            )

            workload_path.write_text(
                json.dumps(workload, indent=2),
                encoding="utf-8",
            )

            # ------------------------------------------------
            # Run exact offline HBM-miss oracle.
            # ------------------------------------------------

            start = time.perf_counter()

            oracle = solve_workload(workload_path)

            oracle_elapsed = time.perf_counter() - start

            optimum = oracle["optimal_hbm_misses"]
            dp_states = oracle["dp_states"]

            oracle_states.append(dp_states)
            oracle_seconds.append(oracle_elapsed)

            print(
                f"Seed {seed}: "
                f"oracle={optimum}, "
                f"DP states={dp_states}, "
                f"time={oracle_elapsed:.3f}s"
            )

            seed_results = {}

            # ------------------------------------------------
            # Evaluate all six online policies.
            # ------------------------------------------------

            for policy in POLICIES:

                result = run_workload(
                    workload_path,
                    policy=policy,
                    verbose=False,
                    trace_memory=False,
                )

                metrics = validate_result(
                    result,
                    ACCESS_COUNT,
                )

                regret = metrics["hbm_misses"] - optimum

                if regret < 0:
                    raise RuntimeError(
                        f"{policy} beat the exact oracle "
                        f"on seed {seed}. "
                        "Check model consistency."
                    )

                seed_results[policy] = {
                    **metrics,
                    "regret": regret,
                }

                records.append({
                    "seed": seed,
                    "policy": policy,
                    **metrics,
                    "oracle_hbm_misses": optimum,
                    "regret": regret,
                    "dp_states": dp_states,
                })

            # ------------------------------------------------
            # Identify per-seed winner(s).
            # ------------------------------------------------

            best_regret = min(
                item["regret"]
                for item in seed_results.values()
            )

            winners = [
                policy
                for policy, item in seed_results.items()
                if item["regret"] == best_regret
            ]

            print(
                f"  Best policy regret: {best_regret}; "
                f"winner(s): {', '.join(winners)}"
            )

            # ------------------------------------------------
            # Print per-policy results.
            # ------------------------------------------------

            for policy in POLICIES:

                item = seed_results[policy]

                print(
                    f"  {policy:16s} "
                    f"misses={item['hbm_misses']:2d} "
                    f"hits={item['hbm_hits']:2d} "
                    f"regret={item['regret']:2d}"
                )

            print("")

    # ========================================================
    # AGGREGATE RESULTS
    # ========================================================

    summaries = []

    for policy in POLICIES:

        policy_records = [
            item
            for item in records
            if item["policy"] == policy
        ]

        regrets = [
            item["regret"]
            for item in policy_records
        ]

        hit_rates = [
            item["hit_rate"]
            for item in policy_records
        ]

        summaries.append({

            "policy": policy,

            "mean_regret": statistics.mean(regrets),

            "median_regret": statistics.median(regrets),

            "worst_regret": max(regrets),

            "zero_regret_seeds": sum(
                regret == 0
                for regret in regrets
            ),

            "mean_hbm_misses": statistics.mean(
                item["hbm_misses"]
                for item in policy_records
            ),

            "mean_hbm_hit_rate": statistics.mean(
                hit_rates
            ),

        })

    summaries.sort(
        key=lambda item: (
            item["mean_regret"],
            item["worst_regret"],
            item["policy"],
        )
    )

    # ========================================================
    # DISPLAY AGGREGATE RESULTS
    # ========================================================

    print("=" * 88)
    print("AGGREGATE POLICY RESULTS")
    print("=" * 88)

    print(
        f"{'Policy':16s} "
        f"{'Mean regret':>12s} "
        f"{'Median':>9s} "
        f"{'Worst':>8s} "
        f"{'Zero-regret':>12s} "
        f"{'Mean misses':>12s} "
        f"{'Mean hit rate':>14s}"
    )

    for item in summaries:

        print(
            f"{item['policy']:16s} "
            f"{item['mean_regret']:12.3f} "
            f"{item['median_regret']:9.1f} "
            f"{item['worst_regret']:8d} "
            f"{item['zero_regret_seeds']:12d} "
            f"{item['mean_hbm_misses']:12.3f} "
            f"{item['mean_hbm_hit_rate'] * 100:13.2f}%"
        )

    print("")

    # ========================================================
    # ORACLE COMPLEXITY
    # ========================================================

    print(
        f"Oracle DP states: "
        f"mean={statistics.mean(oracle_states):.1f}, "
        f"max={max(oracle_states)}"
    )

    print(
        f"Oracle runtime: "
        f"mean={statistics.mean(oracle_seconds):.3f}s, "
        f"total={sum(oracle_seconds):.3f}s"
    )

    # ========================================================
    # MACHINE-READABLE RESULTS
    # ========================================================

    output = {

        "experiment": "multi_seed_exact_hbm_oracle",

        "family": family,

        "metadata_mode": metadata_mode,

        "seeds": seeds,

        "contexts": CONTEXTS,

        "access_count": ACCESS_COUNT,

        "hbm_blocks": HBM_BLOCKS,

        "host_blocks": HOST_BLOCKS,

        "trace_method": (
            f"First {ACCESS_COUNT} access events from the "
            "generated workload; preceding allocation and "
            "metadata events are preserved."
        ),

        "objective": (
            "host restorations + cold misses"
        ),

        "oracle": {

            "mean_dp_states": statistics.mean(
                oracle_states
            ),

            "max_dp_states": max(
                oracle_states
            ),

            "mean_runtime_seconds": statistics.mean(
                oracle_seconds
            ),

            "total_runtime_seconds": sum(
                oracle_seconds
            ),

        },

        "summary": summaries,

        "records": records,

    }

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_directory = (
        PROJECT_ROOT / "benchmarks" / "results"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_directory / (
        f"multi_seed_{family}_{metadata_mode}.json"
    )

    output_file.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("")
    print(f"Saved results: {output_file}")

    print(
        "Synthetic simulator/oracle experiment only."
    )

    print(
        "No GPU performance or real-inference claims."
    )


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare MemCore policies against an exact "
            "offline HBM-miss oracle across random seeds."
        )
    )

    parser.add_argument(
        "--family",
        choices=FAMILIES,
        default="uniform",
    )

    parser.add_argument(
        "--metadata-mode",
        choices=(
            "informed",
            "default",
            "default_metadata",
        ),
        default="default",
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed-count",
        type=int,
        default=DEFAULT_SEEDS,
    )

    args = parser.parse_args()

    if args.seed_count <= 0:
        parser.error(
            "--seed-count must be positive"
        )

    if args.seed_count > MAX_SEEDS:
        parser.error(
            f"Maximum {MAX_SEEDS} seeds per run. "
            "Expand only after reviewing oracle state growth."
        )

    if args.seed_start < 0:
        parser.error(
            "--seed-start cannot be negative"
        )

    seeds = list(
        range(
            args.seed_start,
            args.seed_start + args.seed_count,
        )
    )

    run_experiment(
        family=args.family,
        seeds=seeds,
        metadata_mode=args.metadata_mode,
    )


if __name__ == "__main__":
    main()
