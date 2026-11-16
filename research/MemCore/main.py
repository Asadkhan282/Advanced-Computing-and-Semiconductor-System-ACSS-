
import json

from memcore.simulator import run_workload


def main():

    print("=" * 60)
    print("             ACSS MEMCORE SIMULATOR")
    print("=" * 60)

    results = run_workload(
        "workloads/agent_workload.json"
    )

    print("\n" + "=" * 60)
    print("                 RESULTS")
    print("=" * 60)

    print(
        json.dumps(
            results,
            indent=4
        )
    )


if __name__ == "__main__":
    main()
