
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memcore.scheduler import SLOAwareScheduler, SchedulerWeights


NOW_MS = 50.0

CONTEXTS = [
    {
        "block_id": "CTX_A",
        "access_count": 12,
        "deadline_ms": 60.0,
        "reuse_probability": 0.90,
        "estimated_transfer_ms": 1.0,
        "quality_risk": 0.10,
    },
    {
        "block_id": "CTX_B",
        "access_count": 2,
        "deadline_ms": 500.0,
        "reuse_probability": 0.20,
        "estimated_transfer_ms": 15.0,
        "quality_risk": 0.90,
    },
    {
        "block_id": "CTX_C",
        "access_count": 8,
        "deadline_ms": 100.0,
        "reuse_probability": 0.70,
        "estimated_transfer_ms": 8.0,
        "quality_risk": 0.50,
    },
    {
        "block_id": "CTX_D",
        "access_count": 0,
        "deadline_ms": 1000.0,
        "reuse_probability": 0.05,
        "estimated_transfer_ms": 0.5,
        "quality_risk": 0.05,
    },
    {
        "block_id": "CTX_E",
        "access_count": 5,
        "deadline_ms": 55.0,
        "reuse_probability": 0.50,
        "estimated_transfer_ms": 25.0,
        "quality_risk": 0.70,
    },
]


CONFIGURATIONS = {
    "Baseline": SchedulerWeights(
        deadline=0.30,
        reuse=0.25,
        hotness=0.20,
        transfer_cost=0.15,
        quality_risk=0.10,
    ),

    "Deadline dominant": SchedulerWeights(
        deadline=0.70,
        reuse=0.10,
        hotness=0.05,
        transfer_cost=0.10,
        quality_risk=0.05,
    ),

    "Reuse dominant": SchedulerWeights(
        deadline=0.10,
        reuse=0.65,
        hotness=0.10,
        transfer_cost=0.05,
        quality_risk=0.10,
    ),

    "Hotness dominant": SchedulerWeights(
        deadline=0.10,
        reuse=0.10,
        hotness=0.65,
        transfer_cost=0.05,
        quality_risk=0.10,
    ),

    "Transfer dominant": SchedulerWeights(
        deadline=0.10,
        reuse=0.10,
        hotness=0.05,
        transfer_cost=0.65,
        quality_risk=0.10,
    ),

    "Quality dominant": SchedulerWeights(
        deadline=0.10,
        reuse=0.10,
        hotness=0.05,
        transfer_cost=0.10,
        quality_risk=0.65,
    ),

    "Balanced": SchedulerWeights(
        deadline=0.20,
        reuse=0.20,
        hotness=0.20,
        transfer_cost=0.20,
        quality_risk=0.20,
    ),
}


def run_experiment():
    print("=" * 100)
    print("ACSS MEMCORE — SLO SCHEDULER WEIGHT SENSITIVITY")
    print("=" * 100)

    print(f"\nSimulated current time: {NOW_MS:.2f} ms")
    print(f"Number of contexts: {len(CONTEXTS)}")

    print("\nCONTEXT METADATA")
    print("-" * 100)

    for context in CONTEXTS:
        print(
            f"{context['block_id']:8}"
            f" Accesses={context['access_count']:2}"
            f" Deadline={context['deadline_ms']:8.1f} ms"
            f" Reuse={context['reuse_probability']:.2f}"
            f" Transfer={context['estimated_transfer_ms']:6.2f} ms"
            f" QualityRisk={context['quality_risk']:.2f}"
        )

    results = {}

    for name, weights in CONFIGURATIONS.items():
        scheduler = SLOAwareScheduler(weights=weights)

        ranking = scheduler.rank(
            CONTEXTS,
            now_ms=NOW_MS
        )

        results[name] = ranking

    print("\nWEIGHT CONFIGURATIONS")
    print("-" * 100)

    print(
        f"{'Configuration':24}"
        f"{'Deadline':>10}"
        f"{'Reuse':>10}"
        f"{'Hotness':>10}"
        f"{'Transfer':>10}"
        f"{'Quality':>10}"
    )

    for name, weights in CONFIGURATIONS.items():
        print(
            f"{name:24}"
            f"{weights.deadline:10.2f}"
            f"{weights.reuse:10.2f}"
            f"{weights.hotness:10.2f}"
            f"{weights.transfer_cost:10.2f}"
            f"{weights.quality_risk:10.2f}"
        )

    print("\nRANKING RESULTS")
    print("-" * 100)

    for name, ranking in results.items():
        order = " > ".join(
            item["block_id"] for item in ranking
        )

        eviction_candidate = ranking[-1]

        print(f"\n{name}")
        print(f"Priority ranking: {order}")
        print(
            f"Eviction candidate: {eviction_candidate['block_id']}"
            f" (score={eviction_candidate['score']:.4f})"
        )

    print("\nDETAILED SCORE MATRIX")
    print("-" * 100)

    print(
        f"{'Context':12}"
        + "".join(f"{name[:13]:>16}" for name in CONFIGURATIONS)
    )

    for context in CONTEXTS:
        block_id = context["block_id"]

        print(f"{block_id:12}", end="")

        for name in CONFIGURATIONS:
            matching = next(
                item for item in results[name]
                if item["block_id"] == block_id
            )

            print(f"{matching['score']:16.4f}", end="")

        print()

    print("\nEVICTION DECISION SUMMARY")
    print("-" * 100)

    eviction_candidates = {}

    for name, ranking in results.items():
        victim = ranking[-1]["block_id"]
        eviction_candidates[name] = victim
        print(f"{name:24} -> {victim}")

    unique_victims = set(eviction_candidates.values())

    print("\nINTERPRETATION")
    print("-" * 100)

    print(
        f"Number of distinct eviction decisions: "
        f"{len(unique_victims)}"
    )

    print(
        f"Eviction candidates observed: "
        f"{', '.join(sorted(unique_victims))}"
    )

    print(
        "\nA changing eviction candidate demonstrates that scheduler "
        "weights influence residency decisions."
    )

    print(
        "It does not demonstrate better real-world GPU performance."
    )

    print(
        "\nAll results are deterministic calculations using synthetic "
        "context metadata."
    )

    print("=" * 100)


if __name__ == "__main__":
    run_experiment()
