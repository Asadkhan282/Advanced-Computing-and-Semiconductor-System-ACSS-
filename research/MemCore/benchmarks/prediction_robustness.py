import math
import random
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memcore.scheduler import SLOAwareScheduler


SEED = 42
NOW_MS = 50.0

NUM_CONTEXTS = 100
NUM_TRIALS = 30
TOP_K = 10

ERROR_LEVELS = [0.0, 0.10, 0.25, 0.50]

EVICTION_MARGIN = 0.03


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def generate_contexts(seed=SEED):
    rng = random.Random(seed)
    contexts = []

    for index in range(NUM_CONTEXTS):
        contexts.append({
            "block_id": f"CTX_{index:03d}",
            "access_count": rng.randint(0, 30),
            "deadline_ms": rng.uniform(55.0, 2000.0),
            "reuse_probability": rng.uniform(0.0, 1.0),
            "estimated_transfer_ms": rng.uniform(0.1, 30.0),
            "quality_risk": rng.uniform(0.0, 1.0),

            # Simulated recency information for near-tie fallback.
            "last_access_tick": rng.randint(0, 1000),
        })

    return contexts


def perturb_contexts(contexts, error_level, rng):
    predicted = []

    for context in contexts:
        item = dict(context)

        if error_level > 0:
            item["reuse_probability"] = clamp(
                context["reuse_probability"]
                + rng.gauss(0.0, error_level),
                0.0,
                1.0,
            )

            item["quality_risk"] = clamp(
                context["quality_risk"]
                + rng.gauss(0.0, error_level),
                0.0,
                1.0,
            )

            multiplicative_noise = math.exp(
                rng.gauss(0.0, error_level)
            )

            item["estimated_transfer_ms"] = (
                context["estimated_transfer_ms"]
                * multiplicative_noise
            )

        predicted.append(item)

    return predicted


def ranking_positions(ranking):
    return {
        item["block_id"]: position
        for position, item in enumerate(ranking)
    }


def kendall_tau(reference, candidate):
    reference_positions = ranking_positions(reference)
    candidate_positions = ranking_positions(candidate)

    block_ids = list(reference_positions)

    concordant = 0
    discordant = 0

    for i in range(len(block_ids)):
        for j in range(i + 1, len(block_ids)):
            a = block_ids[i]
            b = block_ids[j]

            reference_order = (
                reference_positions[a] - reference_positions[b]
            )

            candidate_order = (
                candidate_positions[a] - candidate_positions[b]
            )

            if reference_order * candidate_order > 0:
                concordant += 1
            else:
                discordant += 1

    total = concordant + discordant

    if total == 0:
        return 1.0

    return (concordant - discordant) / total


def top_k_overlap(reference, candidate, k=TOP_K):
    reference_ids = {
        item["block_id"] for item in reference[:k]
    }

    candidate_ids = {
        item["block_id"] for item in candidate[:k]
    }

    return len(reference_ids & candidate_ids) / k


def score_deviation(reference, candidate):
    reference_scores = {
        item["block_id"]: item["score"]
        for item in reference
    }

    candidate_scores = {
        item["block_id"]: item["score"]
        for item in candidate
    }

    differences = [
        abs(reference_scores[block_id] - candidate_scores[block_id])
        for block_id in reference_scores
    ]

    return sum(differences) / len(differences)


def run_experiment():
    print("=" * 110)
    print("ACSS MEMCORE — EVICTION ROBUSTNESS COMPARISON")
    print("=" * 110)

    print(f"\nRandom seed: {SEED}")
    print(f"Contexts: {NUM_CONTEXTS}")
    print(f"Trials per error level: {NUM_TRIALS}")
    print(f"Eviction margin: {EVICTION_MARGIN}")
    print(f"Current simulated time: {NOW_MS:.2f} ms")

    scheduler = SLOAwareScheduler()

    ground_truth = generate_contexts()

    oracle_ranking = scheduler.rank(
        ground_truth,
        now_ms=NOW_MS
    )

    original_oracle_victim = oracle_ranking[-1]["block_id"]

    robust_oracle_victim = scheduler.select_victim(
        ground_truth,
        now_ms=NOW_MS,
        margin=EVICTION_MARGIN
    )

    print(f"\nOriginal policy oracle victim: {original_oracle_victim}")
    print(f"Margin policy oracle victim:   {robust_oracle_victim}")

    print("\nROBUSTNESS RESULTS")
    print("-" * 110)

    print(
        f"{'Error':>8}"
        f"{'Kendall tau':>14}"
        f"{'Top-10 overlap':>17}"
        f"{'Original agreement':>21}"
        f"{'Margin agreement':>20}"
        f"{'Score deviation':>19}"
    )

    for error_level in ERROR_LEVELS:
        tau_values = []
        overlap_values = []

        original_matches = 0
        margin_matches = 0

        deviations = []

        for trial in range(NUM_TRIALS):
            rng = random.Random(
                SEED + int(error_level * 10000) + trial
            )

            predicted_contexts = perturb_contexts(
                ground_truth,
                error_level,
                rng
            )

            predicted_ranking = scheduler.rank(
                predicted_contexts,
                now_ms=NOW_MS
            )

            # Original policy: evict the lowest-scoring context.
            original_predicted_victim = (
                predicted_ranking[-1]["block_id"]
            )

            # New policy: use the margin and recency fallback.
            margin_predicted_victim = scheduler.select_victim(
                predicted_contexts,
                now_ms=NOW_MS,
                margin=EVICTION_MARGIN
            )

            tau_values.append(
                kendall_tau(
                    oracle_ranking,
                    predicted_ranking
                )
            )

            overlap_values.append(
                top_k_overlap(
                    oracle_ranking,
                    predicted_ranking
                )
            )

            if original_predicted_victim == original_oracle_victim:
                original_matches += 1

            if margin_predicted_victim == robust_oracle_victim:
                margin_matches += 1

            deviations.append(
                score_deviation(
                    oracle_ranking,
                    predicted_ranking
                )
            )

        mean_tau = sum(tau_values) / NUM_TRIALS
        mean_overlap = sum(overlap_values) / NUM_TRIALS

        original_agreement = original_matches / NUM_TRIALS
        margin_agreement = margin_matches / NUM_TRIALS

        mean_deviation = sum(deviations) / NUM_TRIALS

        print(
            f"{error_level * 100:7.0f}%"
            f"{mean_tau:14.4f}"
            f"{mean_overlap * 100:16.2f}%"
            f"{original_agreement * 100:20.2f}%"
            f"{margin_agreement * 100:19.2f}%"
            f"{mean_deviation:19.4f}"
        )

    print("\nINTERPRETATION")
    print("-" * 110)

    print("Kendall tau and Top-10 overlap measure ranking stability.")
    print("Original agreement measures lowest-score victim stability.")
    print("Margin agreement measures the new policy's victim stability.")
    print("Each policy is compared against its own ground-truth decision.")

    print(
        "\nThese experiments use synthetic metadata and simulated "
        "recency values."
    )

    print(
        "They do not establish real-world prediction accuracy or "
        "GPU inference performance."
    )

    print("=" * 110)


if __name__ == "__main__":
    run_experiment()