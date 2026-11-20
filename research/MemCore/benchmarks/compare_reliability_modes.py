
from types import MethodType

from benchmarks import generate_robustness_workloads as generator
from memcore.memory import MemoryManager


SEEDS = range(42, 52)

POLICIES = [
    "ADAPTIVE_SLO",
    "ADAPTIVE_MARGIN",
]

FAMILIES = list(generator.FAMILIES)

MODES = [
    "CONTEXT_ONLY",
    "GLOBAL_ONLY",
    "HYBRID",
]


def make_rank(mode):

    def adaptive_rank(self, contexts):

        if not contexts:
            return []

        prevalence = self.get_observed_reuse_rate()
        global_reliability = self.get_reuse_reliability()

        adjusted = []
        details = {}

        for context in contexts:

            block_id = context["block_id"]
            predicted = float(context["reuse_probability"])

            metrics = self.get_context_reuse_prediction_metrics(
                block_id
            )

            observations = metrics["observations"]

            context_reliability = (
                self.get_context_reuse_reliability(block_id)
            )

            if mode == "CONTEXT_ONLY":

                reliability = context_reliability

            elif mode == "GLOBAL_ONLY":

                reliability = global_reliability

            elif mode == "HYBRID":

                reliability = (
                    global_reliability
                    if observations < self.REUSE_MIN_OBSERVATIONS
                    else min(
                        global_reliability,
                        context_reliability,
                    )
                )

            else:
                raise ValueError(mode)

            effective = (
                reliability * predicted
                + (1.0 - reliability) * prevalence
            )

            item = dict(context)
            item["reuse_probability"] = effective

            adjusted.append(item)

            details[block_id] = {
                "reliability": reliability,
                "global_reliability": global_reliability,
                "context_reliability": context_reliability,
                "effective_reuse_probability": effective,
            }

        ranked = self.scheduler.rank(
            adjusted,
            now_ms=self.current_time_ms,
        )

        for item in ranked:
            item.update(details[item["block_id"]])

        return ranked

    return adaptive_rank


def execute(workload, policy, mode):

    config = workload["memory"]

    manager = MemoryManager(
        hbm_capacity_bytes=config["hbm_capacity_bytes"],
        host_capacity_bytes=config["host_capacity_bytes"],
        policy=policy,
    )

    manager._rank_contexts_adaptively = MethodType(
        make_rank(mode),
        manager,
    )

    access_requests = 0

    for event in workload["events"]:

        operation = event["operation"]

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

            manager.advance_time(event["delta_ms"])

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

        manager.validate()

    metrics = manager.report()

    return {
        "hbm_hits": metrics["hbm_hits"],
        "host_hits": metrics["host_hits"],
        "cold_misses": metrics["cold_misses"],
        "hbm_rate": (
            metrics["hbm_hits"] / access_requests
            if access_requests
            else 0.0
        ),
    }


def main():

    results = {}

    print("MemCore global/context reliability experiment")
    print("Seeds:", list(SEEDS))
    print("Families:", FAMILIES)
    print("Policies:", POLICIES)
    print("Modes:", MODES)
    print()

    for family in FAMILIES:

        for policy in POLICIES:

            for mode in MODES:

                records = []

                for seed in SEEDS:

                    workload = generator.build_workload(
                        family=family,
                        seed=seed,
                        hbm_blocks=4,
                        host_blocks=8,
                        metadata_mode="informed",
                    )

                    result = execute(
                        workload,
                        policy,
                        mode,
                    )

                    records.append({
                        "seed": seed,
                        **result,
                    })

                results[(family, policy, mode)] = records

    print("=" * 105)

    print(
        f"{'FAMILY':22} {'POLICY':18} {'MODE':15} "
        f"{'HBM HIT %':>11} {'HOST HITS':>11}"
    )

    print("=" * 105)

    for family in FAMILIES:

        for policy in POLICIES:

            for mode in MODES:

                records = results[(family, policy, mode)]

                rate = sum(
                    record["hbm_rate"]
                    for record in records
                ) / len(records)

                host_hits = sum(
                    record["host_hits"]
                    for record in records
                ) / len(records)

                print(
                    f"{family:22} {policy:18} {mode:15} "
                    f"{rate * 100:10.2f}% {host_hits:11.2f}"
                )

    print("=" * 105)
    print("\nOVERALL RESULTS")

    for policy in POLICIES:

        print(f"\n{policy}")

        for mode in MODES:

            records = [
                record
                for family in FAMILIES
                for record in results[(family, policy, mode)]
            ]

            rate = sum(
                record["hbm_rate"]
                for record in records
            ) / len(records)

            print(
                f"  {mode:15}: {rate * 100:.2f}%"
            )


if __name__ == "__main__":
    main()
