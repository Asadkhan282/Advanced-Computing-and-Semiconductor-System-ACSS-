import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SchedulerWeights:
    """
    Weights for HBM residency priority.

    All weights must be non-negative.
    They must sum to 1.0.
    """

    deadline: float = 0.30
    reuse: float = 0.25
    hotness: float = 0.20
    transfer_cost: float = 0.15
    quality_risk: float = 0.10

    def validate(self):
        values = (
            self.deadline,
            self.reuse,
            self.hotness,
            self.transfer_cost,
            self.quality_risk,
        )

        if any(
            not math.isfinite(value) or value < 0
            for value in values
        ):
            raise ValueError(
                "Scheduler weights must be finite and non-negative."
            )

        if not math.isclose(
            sum(values),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Scheduler weights must sum to 1.0."
            )


class SLOAwareScheduler:
    """
    Deterministic heuristic scheduler for context HBM residency.

    A higher score means a context has a stronger claim to remain in HBM.

    deadline_ms:
        Absolute simulated deadline timestamp in milliseconds.

    now_ms:
        Current simulated timestamp in milliseconds.

    quality_risk:
        Estimated risk of losing useful information or quality if the
        context is moved, compressed, or recomputed.

    This is a heuristic policy, not a learned model or a measured
    GPU performance predictor.
    """

    def __init__(
        self,
        weights: Optional[SchedulerWeights] = None,
        deadline_scale_ms: float = 100.0,
        transfer_scale_ms: float = 10.0,
        hotness_scale: float = 5.0,
    ):
        self.weights = weights or SchedulerWeights()
        self.weights.validate()

        scales = (
            deadline_scale_ms,
            transfer_scale_ms,
            hotness_scale,
        )

        if any(
            not math.isfinite(value) or value <= 0
            for value in scales
        ):
            raise ValueError(
                "Scheduler scales must be finite and greater than zero."
            )

        self.deadline_scale_ms = deadline_scale_ms
        self.transfer_scale_ms = transfer_scale_ms
        self.hotness_scale = hotness_scale

    @staticmethod
    def _validate_metadata(metadata: Dict):
        required_fields = (
            "block_id",
            "access_count",
            "reuse_probability",
            "estimated_transfer_ms",
            "quality_risk",
        )

        for field in required_fields:
            if field not in metadata:
                raise ValueError(
                    f"Missing metadata field: {field}"
                )

        if (
            not isinstance(metadata["block_id"], str)
            or not metadata["block_id"]
        ):
            raise ValueError(
                "block_id must be a non-empty string."
            )

        access_count = metadata["access_count"]

        if (
            isinstance(access_count, bool)
            or not isinstance(access_count, int)
        ):
            raise ValueError(
                "access_count must be a non-negative integer."
            )

        if access_count < 0:
            raise ValueError(
                "access_count cannot be negative."
            )

        for field in ("reuse_probability", "quality_risk"):
            value = metadata[field]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(
                    f"{field} must be between 0.0 and 1.0."
                )

        transfer_ms = metadata["estimated_transfer_ms"]

        if (
            isinstance(transfer_ms, bool)
            or not isinstance(transfer_ms, (int, float))
            or not math.isfinite(transfer_ms)
            or transfer_ms < 0
        ):
            raise ValueError(
                "estimated_transfer_ms must be finite and non-negative."
            )

        deadline_ms = metadata.get("deadline_ms")

        if deadline_ms is not None:
            if (
                isinstance(deadline_ms, bool)
                or not isinstance(deadline_ms, (int, float))
                or not math.isfinite(deadline_ms)
                or deadline_ms < 0
            ):
                raise ValueError(
                    "deadline_ms must be None or a finite "
                    "non-negative number."
                )

        last_access_tick = metadata.get("last_access_tick", 0)

        if (
            isinstance(last_access_tick, bool)
            or not isinstance(last_access_tick, int)
            or last_access_tick < 0
        ):
            raise ValueError(
                "last_access_tick must be a non-negative integer."
            )

    def score_components(
        self,
        metadata: Dict,
        now_ms: float,
    ) -> Dict[str, float]:
        """
        Calculate normalized factors contributing to HBM residency priority.
        """

        self._validate_metadata(metadata)

        if (
            isinstance(now_ms, bool)
            or not isinstance(now_ms, (int, float))
            or not math.isfinite(now_ms)
            or now_ms < 0
        ):
            raise ValueError(
                "now_ms must be finite and non-negative."
            )

        deadline_ms = metadata.get("deadline_ms")

        if deadline_ms is None:
            deadline_urgency = 0.0
        else:
            slack_ms = max(
                deadline_ms - now_ms,
                0.0,
            )

            deadline_urgency = 1.0 / (
                1.0 + slack_ms / self.deadline_scale_ms
            )

        reuse = float(
            metadata["reuse_probability"]
        )

        access_count = metadata["access_count"]

        hotness = 1.0 - math.exp(
            -access_count / self.hotness_scale
        )

        transfer_ms = float(
            metadata["estimated_transfer_ms"]
        )

        transfer_cost = transfer_ms / (
            transfer_ms + self.transfer_scale_ms
        )

        quality_risk = float(
            metadata["quality_risk"]
        )

        return {
            "deadline_urgency": deadline_urgency,
            "reuse": reuse,
            "hotness": hotness,
            "transfer_cost": transfer_cost,
            "quality_risk": quality_risk,
        }

    def score(
        self,
        metadata: Dict,
        now_ms: float,
    ) -> float:
        """
        Return normalized HBM residency priority between 0 and 1.
        """

        components = self.score_components(
            metadata,
            now_ms,
        )

        weights = self.weights

        result = (
            weights.deadline * components["deadline_urgency"]
            + weights.reuse * components["reuse"]
            + weights.hotness * components["hotness"]
            + weights.transfer_cost * components["transfer_cost"]
            + weights.quality_risk * components["quality_risk"]
        )

        return result

    def rank(
        self,
        contexts: List[Dict],
        now_ms: float,
    ) -> List[Dict]:
        """
        Rank contexts from highest to lowest HBM residency priority.

        Ties are resolved by block_id for deterministic ordering.
        """

        ranked = []

        for metadata in contexts:
            components = self.score_components(
                metadata,
                now_ms,
            )

            weights = self.weights

            score = (
                weights.deadline * components["deadline_urgency"]
                + weights.reuse * components["reuse"]
                + weights.hotness * components["hotness"]
                + weights.transfer_cost * components["transfer_cost"]
                + weights.quality_risk * components["quality_risk"]
            )

            ranked.append({
                "block_id": metadata["block_id"],
                "score": score,
                "components": components,
            })

        ranked.sort(
            key=lambda item: (
                -item["score"],
                item["block_id"],
            )
        )

        return ranked

    def select_victim(
        self,
        contexts: List[Dict],
        now_ms: float,
        margin: float = 0.03,
    ) -> str:
        """
        Select an eviction candidate using a score-margin fallback.

        Contexts close to the lowest score are treated as near-ties.
        Among those candidates, select the least recently accessed.
        """

        if (
            isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(margin)
            or margin < 0
        ):
            raise ValueError(
                "margin must be finite and non-negative."
            )

        if not contexts:
            raise ValueError(
                "Cannot select a victim from an empty context list."
            )

        ranked = self.rank(
            contexts,
            now_ms=now_ms,
        )

        lowest_score = ranked[-1]["score"]

        candidate_ids = {
            item["block_id"]
            for item in ranked
            if item["score"] <= lowest_score + margin
        }

        candidates = [
            context
            for context in contexts
            if context["block_id"] in candidate_ids
        ]

        victim = min(
            candidates,
            key=lambda item: (
                item.get("last_access_tick", 0),
                item["access_count"],
                item["block_id"],
            ),
        )

        return victim["block_id"]

    def rank_adaptive(
        self,
        contexts: List[Dict],
        now_ms: float,
        reuse_reliability: float,
    ) -> List[Dict]:
        """
        Rank contexts using reuse weight adjusted by observed
        prediction reliability.

        The removed reuse weight is redistributed proportionally
        across the other scheduler weights.
        """

        if (
            isinstance(reuse_reliability, bool)
            or not isinstance(reuse_reliability, (int, float))
            or not math.isfinite(reuse_reliability)
            or not 0.0 <= reuse_reliability <= 1.0
        ):
            raise ValueError(
                "reuse_reliability must be between 0 and 1."
            )

        if not contexts:
            raise ValueError(
                "Cannot rank an empty context list."
            )

        weights = self.weights

        original_other_weight = 1.0 - weights.reuse

        adjusted_reuse_weight = (
            weights.reuse * reuse_reliability
        )

        if original_other_weight > 0:
            other_scale = (
                1.0 - adjusted_reuse_weight
            ) / original_other_weight
        else:
            other_scale = 0.0

        ranked = []

        for metadata in contexts:
            components = self.score_components(
                metadata,
                now_ms,
            )

            score = (
                adjusted_reuse_weight * components["reuse"]
                + other_scale * (
                    weights.deadline * components["deadline_urgency"]
                    + weights.hotness * components["hotness"]
                    + weights.transfer_cost * components["transfer_cost"]
                    + weights.quality_risk * components["quality_risk"]
                )
            )

            ranked.append({
                "block_id": metadata["block_id"],
                "score": score,
                "components": components,
            })

        ranked.sort(
            key=lambda item: (
                -item["score"],
                item["block_id"],
            )
        )

        return ranked

    def adaptive_score(
        self,
        metadata: Dict,
        now_ms: float,
        reuse_reliability: float,
    ) -> float:
        """
        Return one context's reliability-adjusted score.
        """

        ranked = self.rank_adaptive(
            [metadata],
            now_ms,
            reuse_reliability,
        )

        return ranked[0]["score"]