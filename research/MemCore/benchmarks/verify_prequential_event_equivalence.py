
from itertools import zip_longest

from memcore.memory import MemoryManager
from benchmarks import generate_robustness_workloads as generator
from benchmarks.compare_prediction_shifts import apply_shift
import benchmarks.compare_prequential_guard as research
import benchmarks.validate_production_prequential_equivalence as production


SEEDS = range(1000, 1010)

CONDITIONS = [
    ("default_metadata", "CLEAN"),
    ("informed", "CLEAN"),
    ("informed", "LATE_INVERSION"),
    ("informed", "TEMPORARY_INVERSION"),
]


class TracedManager(MemoryManager):
    last_created = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = []
        self.victims = []
        self._trace_depth = 0
        TracedManager.last_created = self

    def _snapshot(self, operation):
        pending = tuple(sorted(
            (
                block_id,
                p["predicted_probability"],
                p["baseline_probability"],
                p["observed_reuse"],
                p["expires_at"],
            )
            for block_id, p in
            self.pending_reuse_predictions.items()
        ))

        return (
            operation,
            tuple(self.hbm.keys()),
            tuple(self.host.keys()),
            self.hbm_used,
            self.host_used,
            self.logical_clock,
            self.access_request_clock,
            self.current_time_ms,
            tuple(sorted(self.metrics.items())),
            pending,
            tuple(
                self.reuse_prediction_prequential_observations
            ),
        )

    def _track(self, operation, *args, **kwargs):
        outer = self._trace_depth == 0
        self._trace_depth += 1

        try:
            result = getattr(
                super(), operation
            )(*args, **kwargs)

            if outer:
                self.events.append(
                    self._snapshot(operation)
                )

            return result
        finally:
            self._trace_depth -= 1

    def insert(self, *args, **kwargs):
        return self._track("insert", *args, **kwargs)

    def access(self, *args, **kwargs):
        return self._track("access", *args, **kwargs)

    def update_metadata(self, *args, **kwargs):
        return self._track(
            "update_metadata", *args, **kwargs
        )

    def advance_time(self, *args, **kwargs):
        return self._track(
            "advance_time", *args, **kwargs
        )

    def _select_hbm_victim(self):
        before = tuple(self.hbm.keys())

        victim = super()._select_hbm_victim()

        guard = self.get_prequential_guard_status()

        self.victims.append((
            len(self.events),
            before,
            victim,
            guard["guard_active"],
            guard["failure_detected"],
            guard["recovery_detected"],
        ))

        return victim


def run_research(workload):
    original = research.MemoryManager
    research.MemoryManager = TracedManager

    try:
        result = research.replay_prequential(
            workload, "PREQ_20_FAST12"
        )
        manager = TracedManager.last_created
    finally:
        research.MemoryManager = original

    return result, manager


def run_production(workload):
    original = production.MemoryManager
    production.MemoryManager = TracedManager

    try:
        result = production.replay_production(workload)
        manager = TracedManager.last_created
    finally:
        production.MemoryManager = original

    return result, manager


def first_difference(a, b):
    missing = object()

    for index, (left, right) in enumerate(
        zip_longest(a, b, fillvalue=missing)
    ):
        if left != right:
            return index, left, right

    return None


def main():
    total = 0
    matches = 0
    mismatches = []

    print("=" * 90)
    print("ACSS MEMCORE - EVENT-BY-EVENT POLICY EQUIVALENCE")
    print("=" * 90)
    print("Seeds:", list(SEEDS))
    print("Research: PREQ_20_FAST12")
    print("Integrated: PREQUENTIAL_SLO_MARGIN")
    print()

    for metadata, scenario in CONDITIONS:
        condition_matches = 0

        for family in generator.FAMILIES:
            for seed in SEEDS:
                original = generator.build_workload(
                    family=family,
                    seed=seed,
                    hbm_blocks=4,
                    host_blocks=8,
                    metadata_mode=metadata,
                )

                workload, changed = apply_shift(
                    original, scenario
                )

                if scenario != "CLEAN":
                    assert changed > 0

                ref_result, ref_manager = run_research(
                    workload
                )

                prod_result, prod_manager = run_production(
                    workload
                )

                same = (
                    ref_result["hbm_hits"]
                    == prod_result["hbm_hits"]
                    and ref_result["phase_hits"]
                    == prod_result["phase_hits"]
                    and ref_manager.events == prod_manager.events
                    and ref_manager.victims == prod_manager.victims
                )

                total += 1

                if same:
                    matches += 1
                    condition_matches += 1
                else:
                    mismatches.append((
                        metadata,
                        scenario,
                        family,
                        seed,
                        ref_result,
                        prod_result,
                        first_difference(
                            ref_manager.events,
                            prod_manager.events,
                        ),
                        first_difference(
                            ref_manager.victims,
                            prod_manager.victims,
                        ),
                    ))

        print(
            f"{metadata:18} {scenario:22} "
            f"matches={condition_matches}/60"
        )

    print()
    print("=" * 90)
    print("VALIDATION SUMMARY")
    print("=" * 90)
    print("Compared workloads:", total)
    print("Identical traces:", matches)
    print("Mismatching traces:", len(mismatches))

    if mismatches:
        print()
        print("FIRST FIVE MISMATCHES")

        for item in mismatches[:5]:
            (
                metadata, scenario, family, seed,
                ref_result, prod_result,
                event_diff, victim_diff,
            ) = item

            print(
                f"\n{metadata} / {scenario} / "
                f"{family} / seed={seed}"
            )
            print("Research result:", ref_result["phase_hits"])
            print("Production result:", prod_result["phase_hits"])

            if event_diff is not None:
                print("First event mismatch index:", event_diff[0])
                print("Research event:", event_diff[1])
                print("Production event:", event_diff[2])

            if victim_diff is not None:
                print("First victim mismatch index:", victim_diff[0])
                print("Research victim:", victim_diff[1])
                print("Production victim:", victim_diff[2])

        raise AssertionError(
            f"{len(mismatches)} workload traces differ."
        )

    assert total == 240

    print()
    print("EVENT-LEVEL EQUIVALENCE: PASS")
    print("HBM and host contents: MATCH")
    print("Eviction decisions: MATCH")
    print("Prediction histories: MATCH")
    print("Memory counters: MATCH")
    print()
    print("All results are synthetic simulator measurements.")
    print("Production source files were not modified.")


if __name__ == "__main__":
    main()
