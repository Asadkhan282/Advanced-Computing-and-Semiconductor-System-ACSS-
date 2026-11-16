
import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "workloads" / "robustness"

SEED = 20261109
BLOCK_SIZE = 256
ACCESS_COUNT = 240
PREDICTION_HORIZON = 20

FAMILIES = [
    "uniform",
    "zipf",
    "bursty",
    "recurring",
    "prediction_mismatch",
    "capacity_pressure",
]

METADATA_MODES = [
    "informed",
    "default_metadata",
]


def probability_of_reuse(per_request_probability):
    """Probability of at least one access in the next horizon."""

    if not 0.0 <= per_request_probability <= 1.0:
        raise ValueError(
            "Per-request probability must be between 0 and 1."
        )

    return 1.0 - (
        1.0 - per_request_probability
    ) ** PREDICTION_HORIZON


def make_access_sequence(rng, family, block_ids, count):
    """Generate a reproducible synthetic access sequence."""

    if not block_ids:
        raise ValueError("block_ids cannot be empty.")

    if count < 0:
        raise ValueError("count cannot be negative.")

    if family == "uniform":
        return [
            rng.choice(block_ids)
            for _ in range(count)
        ]

    if family == "zipf":
        weights = [
            1.0 / ((i + 1) ** 1.15)
            for i in range(len(block_ids))
        ]

        return rng.choices(
            block_ids,
            weights=weights,
            k=count,
        )

    if family == "bursty":
        groups = [
            block_ids[i:i + 3]
            for i in range(0, len(block_ids), 3)
        ]

        return [
            (
                rng.choice(groups[(i // 30) % len(groups)])
                if rng.random() < 0.85
                else rng.choice(block_ids)
            )
            for i in range(count)
        ]

    if family == "recurring":
        groups = [
            block_ids[i:i + 4]
            for i in range(0, len(block_ids), 4)
        ]

        return [
            (
                rng.choice(groups[(i // 20) % len(groups)])
                if rng.random() < 0.80
                else rng.choice(block_ids)
            )
            for i in range(count)
        ]

    if family == "prediction_mismatch":
        midpoint = len(block_ids) // 2

        low_popularity = block_ids[:midpoint]
        high_popularity = block_ids[midpoint:]

        return [
            (
                rng.choice(low_popularity)
                if rng.random() < 0.85
                else rng.choice(high_popularity)
            )
            for _ in range(count)
        ]

    if family == "capacity_pressure":
        return [
            block_ids[i % len(block_ids)]
            for i in range(count)
        ]

    raise ValueError(
        f"Unknown workload family: {family}"
    )


def make_reuse_probabilities(
    family,
    block_count,
    window_start,
):
    """
    Predict the probability of at least one access during
    the next PREDICTION_HORIZON global access requests.

    Predictions use the synthetic workload model and known
    phase schedule, never the realized future access sequence.
    """

    if block_count < 1:
        raise ValueError(
            "block_count must be at least 1."
        )

    if window_start < 0:
        raise ValueError(
            "window_start cannot be negative."
        )

    probabilities = []

    if family == "uniform":

        q = 1.0 / block_count

        probabilities = [
            probability_of_reuse(q)
            for _ in range(block_count)
        ]

    elif family == "zipf":

        weights = [
            1.0 / ((i + 1) ** 1.15)
            for i in range(block_count)
        ]

        total_weight = sum(weights)

        probabilities = [
            probability_of_reuse(
                weight / total_weight
            )
            for weight in weights
        ]

    elif family in ("bursty", "recurring"):

        group_size = (
            3 if family == "bursty" else 4
        )

        group_period = (
            30 if family == "bursty" else 20
        )

        active_weight = (
            0.85 if family == "bursty" else 0.80
        )

        group_count = (
            block_count + group_size - 1
        ) // group_size

        for block_index in range(block_count):

            probability_of_no_reuse = 1.0

            for offset in range(PREDICTION_HORIZON):

                request_index = window_start + offset

                phase = request_index // group_period
                active_group = phase % group_count

                active_start = (
                    active_group * group_size
                )

                active_end = min(
                    active_start + group_size,
                    block_count,
                )

                is_active = (
                    active_start
                    <= block_index
                    < active_end
                )

                if is_active:

                    q = (
                        active_weight
                        / (active_end - active_start)
                        + (1.0 - active_weight)
                        / block_count
                    )

                else:

                    q = (
                        (1.0 - active_weight)
                        / block_count
                    )

                probability_of_no_reuse *= (
                    1.0 - q
                )

            probabilities.append(
                1.0 - probability_of_no_reuse
            )

    elif family == "prediction_mismatch":

        midpoint = block_count // 2

        probabilities = [
            0.10 if i < midpoint else 0.90
            for i in range(block_count)
        ]

    elif family == "capacity_pressure":

        probabilities = []

        for block_index in range(block_count):

            will_be_accessed = any(
                (
                    window_start + offset
                ) % block_count == block_index

                for offset in range(
                    PREDICTION_HORIZON
                )
            )

            probabilities.append(
                1.0 if will_be_accessed else 0.0
            )

    else:

        raise ValueError(
            f"Unknown workload family: {family}"
        )

    return [
        round(probability, 4)
        for probability in probabilities
    ]


def make_metadata(
    rng,
    family,
    block_count,
    window_start,
):
    """
    Generate initial scheduling metadata.

    Deadline, transfer cost and quality risk are initialized
    here and remain unchanged during rolling prediction refreshes.
    """

    reuse_probabilities = make_reuse_probabilities(
        family,
        block_count,
        window_start,
    )

    metadata = []

    for probability in reuse_probabilities:

        metadata.append({
            "deadline_ms": rng.choice(
                [50, 100, 250, 500, 1000]
            ),

            "reuse_probability": probability,

            "estimated_transfer_ms": round(
                rng.uniform(0.2, 8.0),
                4,
            ),

            "quality_risk": round(
                rng.uniform(0.0, 1.0),
                4,
            ),
        })

    return metadata


def build_workload(
    family,
    seed,
    hbm_blocks,
    host_blocks,
    metadata_mode,
    context_count=None,
):
    """
    Build a reproducible synthetic workload.

    context_count:
        Optional explicit number of contexts.

        If omitted:
            capacity_pressure uses HBM + host blocks.
            Other families retain the original 12 contexts.

        If supplied:
            Exactly context_count contexts are generated.
    """

    if family not in FAMILIES:
        raise ValueError(
            f"Unknown workload family: {family}"
        )

    if metadata_mode not in METADATA_MODES:
        raise ValueError(
            f"Unknown metadata mode: {metadata_mode}"
        )

    if hbm_blocks < 1 or host_blocks < 1:
        raise ValueError(
            "HBM and host capacities must each be positive."
        )

    if context_count is not None:

        if (
            not isinstance(context_count, int)
            or isinstance(context_count, bool)
            or context_count < 1
        ):
            raise ValueError(
                "context_count must be a positive integer."
            )

        block_count = context_count

    elif family == "capacity_pressure":

        block_count = (
            hbm_blocks + host_blocks
        )

    else:

        block_count = 12

    rng = random.Random(seed)

    block_ids = [
        f"CTX_{i:02d}"
        for i in range(block_count)
    ]

    events = []

    initial_metadata = None

    if metadata_mode == "informed":

        initial_metadata = make_metadata(
            rng,
            family,
            block_count,
            window_start=0,
        )

    # Allocate all contexts before access begins.
    for index, block_id in enumerate(block_ids):

        event = {
            "operation": "allocate",
            "block_id": block_id,
            "size_bytes": BLOCK_SIZE,
        }

        if initial_metadata is not None:

            event.update(
                initial_metadata[index]
            )

        events.append(event)

    # Independent RNG: prediction generation cannot influence
    # the actual access sequence.
    access_rng = random.Random(
        seed + 1000
    )

    access_sequence = make_access_sequence(
        access_rng,
        family,
        block_ids,
        ACCESS_COUNT,
    )

    # Emit rolling metadata updates after each completed window.
    for index, block_id in enumerate(access_sequence):

        if (
            index > 0
            and index % PREDICTION_HORIZON == 0
        ):

            events.append({
                "operation": "advance_time",
                "delta_ms": 10.0,
            })

            if metadata_mode == "informed":

                # Predict the next window using the synthetic
                # workload model, not realized future accesses.
                refreshed_probabilities = (
                    make_reuse_probabilities(
                        family,
                        block_count,
                        window_start=index,
                    )
                )

                for context_index, context_id in enumerate(
                    block_ids
                ):

                    events.append({
                        "operation": "update_metadata",
                        "block_id": context_id,

                        "reuse_probability": (
                            refreshed_probabilities[
                                context_index
                            ]
                        ),
                    })

        events.append({
            "operation": "access",
            "block_id": block_id,
        })

    return {
        "workload_name": (
            f"{family}_{metadata_mode}"
        ),

        "description": (
            f"Synthetic {family} workload; "
            f"metadata mode: {metadata_mode}. "
            f"Contexts: {block_count}. "
            f"Reuse prediction horizon: "
            f"{PREDICTION_HORIZON} global access requests. "
            f"Informed predictions are refreshed between "
            f"completed observation windows using the synthetic "
            f"workload model, not realized future accesses. "
            f"Deadline, transfer-cost and quality-risk metadata "
            f"remain fixed after initial allocation."
        ),

        "memory": {
            "hbm_capacity_bytes": (
                hbm_blocks * BLOCK_SIZE
            ),

            "host_capacity_bytes": (
                host_blocks * BLOCK_SIZE
            ),
        },

        "events": events,
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible MemCore robustness workloads."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--hbm-blocks",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--host-blocks",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--contexts",
        type=int,
        default=None,
        help=(
            "Optional explicit context count. "
            "If omitted, original family defaults are preserved."
        ),
    )

    parser.add_argument(
        "--families",
        nargs="+",
        choices=FAMILIES,
        default=FAMILIES,
    )

    args = parser.parse_args()

    if args.hbm_blocks < 1 or args.host_blocks < 1:
        raise ValueError(
            "HBM and host capacities must each be positive."
        )

    if (
        args.contexts is not None
        and args.contexts < 1
    ):
        raise ValueError(
            "--contexts must be a positive integer."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated = []

    for family_index, family in enumerate(
        args.families
    ):

        for metadata_mode in METADATA_MODES:

            workload = build_workload(
                family=family,
                seed=args.seed + family_index,
                hbm_blocks=args.hbm_blocks,
                host_blocks=args.host_blocks,
                metadata_mode=metadata_mode,
                context_count=args.contexts,
            )

            filename = (
                f"{family}_{metadata_mode}.json"
            )

            output_path = (
                OUTPUT_DIR / filename
            )

            output_path.write_text(
                json.dumps(
                    workload,
                    indent=2,
                ),
                encoding="utf-8",
            )

            generated.append(
                output_path
            )

            update_count = sum(
                event["operation"] == "update_metadata"
                for event in workload["events"]
            )

            allocation_count = sum(
                event["operation"] == "allocate"
                for event in workload["events"]
            )

            print(
                f"Generated: {output_path.name} "
                f"({allocation_count} contexts, "
                f"{len(workload['events'])} events, "
                f"{update_count} metadata updates)"
            )

    print()
    print(
        f"Workloads generated: {len(generated)}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
