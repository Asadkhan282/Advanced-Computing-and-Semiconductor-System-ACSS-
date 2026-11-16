from memcore.scheduler import SLOAwareScheduler


MARGINS = [0.00, 0.01, 0.03, 0.05, 0.10]

# The reuse-probability differences create different score gaps.
# All other scoring metadata remains identical.
REUSE_CASES = [
    ("Very close", 0.52),
    ("Close", 0.58),
    ("Moderate", 0.66),
    ("Wide", 0.90),
]

RECENCY_CASES = [
    ("B older, A newer", 10, 1),
    ("A older, B newer", 1, 10),
]


def make_contexts(b_reuse, a_tick, b_tick):
    return [
        {
            "block_id": "CTX_A",
            "access_count": 3,
            "deadline_ms": 1000.0,
            "reuse_probability": 0.50,
            "estimated_transfer_ms": 2.0,
            "quality_risk": 0.50,
            "last_access_tick": a_tick,
        },
        {
            "block_id": "CTX_B",
            "access_count": 3,
            "deadline_ms": 1000.0,
            "reuse_probability": b_reuse,
            "estimated_transfer_ms": 2.0,
            "quality_risk": 0.50,
            "last_access_tick": b_tick,
        },
    ]


def main():
    scheduler = SLOAwareScheduler()

    print("=" * 100)
    print("ACSS MEMCORE — SLO MARGIN SENSITIVITY EXPERIMENT")
    print("=" * 100)

    print("Margins:", MARGINS)
    print("Scoring metadata: identical except reuse probability.")
    print("CTX_A reuse probability: 0.50")
    print("Simulated current time: 0 ms")

    total_cases = 0
    changed_cases = 0

    for reuse_label, b_reuse in REUSE_CASES:
        for recency_label, a_tick, b_tick in RECENCY_CASES:
            contexts = make_contexts(b_reuse, a_tick, b_tick)

            print("\n" + "-" * 100)
            print(f"REUSE CASE: {reuse_label}")
            print(f"CTX_B reuse probability: {b_reuse:.2f}")
            print(f"RECENCY: {recency_label}")

            ranked = scheduler.rank(contexts, now_ms=0.0)
            print("Scheduler ranking:", ranked)

            oracle_victim = scheduler.select_victim(
                contexts,
                now_ms=0.0,
                margin=0.0,
            )

            print("Zero-margin victim:", oracle_victim)
            print("\nMargin       Selected victim       Changed from zero-margin")

            for margin in MARGINS:
                victim = scheduler.select_victim(
                    contexts,
                    now_ms=0.0,
                    margin=margin,
                )

                changed = victim != oracle_victim

                if changed:
                    changed_cases += 1

                total_cases += 1

                print(
                    f"{margin:<12.2f}"
                    f"{victim:<23}"
                    f"{'YES' if changed else 'NO'}"
                )

    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY")
    print("=" * 100)
    print("Total margin decisions:", total_cases)
    print("Decisions changed from zero-margin SLO:", changed_cases)

    if total_cases:
        print(
            "Change frequency: "
            f"{changed_cases / total_cases * 100:.2f}%"
        )

    print("\nIMPORTANT LIMITATIONS")
    print("This is a deterministic synthetic scheduler experiment.")
    print("It does not establish optimal margin selection or GPU performance.")
    print("The zero-margin decision is a reference policy, not ground truth.")


if __name__ == "__main__":
    main()