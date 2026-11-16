
import heapq
import math
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Optional

from memcore.scheduler import SLOAwareScheduler


@dataclass
class MemoryBlock:
    """A memory block with context-management metadata."""

    block_id: str
    size_bytes: int

    access_count: int = 0
    created_tick: int = 0
    last_access_tick: int = 0

    deadline_ms: Optional[float] = None
    reuse_probability: float = 0.5
    estimated_transfer_ms: float = 1.0
    quality_risk: float = 0.0


class MemoryManager:
    """
    MemCore two-tier memory simulator.

    Policies:
        LRU, FIFO, SLO, SLO_MARGIN,
        ADAPTIVE_SLO, ADAPTIVE_MARGIN.

    Reuse prediction:
        Predicts whether a block will be accessed within the
        next REUSE_OBSERVATION_HORIZON access requests.

    Prediction metrics:
        MAE, Brier score, Brier skill score,
        observed reuse prevalence.

    Adaptive policies:
        Use per-context prediction reliability to adjust
        reuse probabilities while retaining fixed scheduler
        weights across all contexts.

    This is a deterministic software simulator, not a physical
    HBM controller or a real GPU inference benchmark.
    """

    SUPPORTED_POLICIES = (
        "LRU",
        "FIFO",
        "SLO",
        "SLO_MARGIN",
        "ADAPTIVE_SLO",
        "ADAPTIVE_MARGIN",
    )

    DEFAULT_SLO_MARGIN = 0.03

    REUSE_OBSERVATION_HORIZON = 20
    REUSE_OBSERVATION_WINDOW = 100
    REUSE_MIN_OBSERVATIONS = 5

    DEFAULT_REUSE_PREVALENCE = 0.5

    def __init__(
        self,
        hbm_capacity_bytes: int,
        host_capacity_bytes: int,
        policy: str = "LRU",
        scheduler: Optional[SLOAwareScheduler] = None,
        current_time_ms: float = 0.0,
        slo_margin: float = DEFAULT_SLO_MARGIN,
    ):

        if (
            not isinstance(hbm_capacity_bytes, int)
            or isinstance(hbm_capacity_bytes, bool)
            or hbm_capacity_bytes <= 0
        ):
            raise ValueError(
                "HBM capacity must be a positive integer."
            )

        if (
            not isinstance(host_capacity_bytes, int)
            or isinstance(host_capacity_bytes, bool)
            or host_capacity_bytes < 0
        ):
            raise ValueError(
                "Host capacity must be a non-negative integer."
            )

        if not isinstance(policy, str):
            raise ValueError("Policy must be a string.")

        policy = policy.upper()

        if policy not in self.SUPPORTED_POLICIES:
            raise ValueError(
                f"Unsupported policy: {policy}. "
                f"Choose from {self.SUPPORTED_POLICIES}."
            )

        self._validate_nonnegative(
            current_time_ms,
            "current_time_ms",
        )

        self._validate_nonnegative(
            slo_margin,
            "slo_margin",
        )

        self.hbm_capacity = hbm_capacity_bytes
        self.host_capacity = host_capacity_bytes

        self.policy = policy
        self.slo_margin = float(slo_margin)

        self.scheduler = scheduler or SLOAwareScheduler()

        self.hbm = OrderedDict()
        self.host = OrderedDict()

        self.hbm_used = 0
        self.host_used = 0

        self.logical_clock = 0
        self.current_time_ms = float(current_time_ms)

        self.access_request_clock = 0

        # Active prediction records keyed by block ID.
        self.pending_reuse_predictions = {}

        # Heap entries:
        # (expires_at, sequence_number, block_id, prediction_record)
        self.reuse_prediction_expiry_heap = []
        self._reuse_prediction_sequence = 0

        # Global completed prediction history:
        # (predicted_probability, observed_reuse)
        self.reuse_prediction_observations = deque(
            maxlen=self.REUSE_OBSERVATION_WINDOW
        )

        # Context-specific completed prediction history:
        # (block_id, predicted_probability, observed_reuse)
        self.reuse_prediction_context_observations = deque(
            maxlen=self.REUSE_OBSERVATION_WINDOW
        )

        self._reuse_prediction_metrics_cache = None

        self.reuse_prediction_censored = 0

        self.metrics = {
            "hbm_hits": 0,
            "host_hits": 0,
            "cold_misses": 0,
            "hbm_evictions": 0,
            "host_evictions": 0,
            "bytes_moved_to_host": 0,
            "bytes_restored_to_hbm": 0,
            "bytes_dropped": 0,
        }

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    @staticmethod
    def _validate_probability(value, field_name):
        """Validate a finite probability between zero and one."""

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(
                f"{field_name} must be a finite number "
                "between 0 and 1."
            )

    @staticmethod
    def _validate_nonnegative(value, field_name):
        """Validate a finite non-negative numeric value."""

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                f"{field_name} must be a finite "
                "non-negative number."
            )

    # ---------------------------------------------------------
    # Simulated time
    # ---------------------------------------------------------

    def advance_time(self, delta_ms: float):
        """Advance simulated time independently of logical_clock."""

        self._validate_nonnegative(
            delta_ms,
            "delta_ms",
        )

        self.current_time_ms += float(delta_ms)

        return self.current_time_ms

    # ---------------------------------------------------------
    # Reuse prediction lifecycle
    # ---------------------------------------------------------

    def _register_reuse_prediction(
        self,
        block_id: str,
        probability: float,
    ):
        """Register a new prediction observation window."""

        if block_id in self.pending_reuse_predictions:
            raise RuntimeError(
                f"Prediction already active for '{block_id}'."
            )

        expires_at = (
            self.access_request_clock
            + self.REUSE_OBSERVATION_HORIZON
        )

        self._reuse_prediction_sequence += 1

        prediction = {
            "block_id": block_id,
            "predicted_probability": float(probability),
            "observed_reuse": False,
            "expires_at": expires_at,
        }

        self.pending_reuse_predictions[block_id] = prediction

        heapq.heappush(
            self.reuse_prediction_expiry_heap,
            (
                expires_at,
                self._reuse_prediction_sequence,
                block_id,
                prediction,
            ),
        )

    def _record_reuse_access(self, block_id: str):
        """
        Record one access request.

        HBM hits, host hits and cold misses all count.

        An access on the final request of the observation window
        counts as reuse.
        """

        self.access_request_clock += 1

        pending = self.pending_reuse_predictions.get(block_id)

        if (
            pending is not None
            and self.access_request_clock <= pending["expires_at"]
        ):
            pending["observed_reuse"] = True

        # Finalize expired predictions.
        while (
            self.reuse_prediction_expiry_heap
            and self.reuse_prediction_expiry_heap[0][0]
            <= self.access_request_clock
        ):

            (
                expires_at,
                sequence_number,
                pending_id,
                prediction,
            ) = heapq.heappop(
                self.reuse_prediction_expiry_heap
            )

            current_prediction = (
                self.pending_reuse_predictions.get(pending_id)
            )

            # Ignore stale heap entries.
            if current_prediction is not prediction:
                continue

            if prediction["expires_at"] != expires_at:
                continue

            self.pending_reuse_predictions.pop(pending_id)

            observed = (
                1.0 if prediction["observed_reuse"] else 0.0
            )

            predicted = prediction["predicted_probability"]

            # Global observation.
            self.reuse_prediction_observations.append(
                (predicted, observed)
            )

            # Context-specific observation.
            self.reuse_prediction_context_observations.append(
                (
                    prediction["block_id"],
                    predicted,
                    observed,
                )
            )

            self._reuse_prediction_metrics_cache = None

    # ---------------------------------------------------------
    # Prediction metric calculation
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_metrics_from_observations(observations):
        """
        Calculate MAE, Brier score, Brier skill score,
        and observed reuse prevalence.

        Input format:
            (predicted_probability, observed_outcome)

        observed_reuse_rate is based exclusively on completed
        observations, never on predicted probabilities.
        """

        if not observations:
            return {
                "observations": 0,
                "mae": None,
                "brier_score": None,
                "brier_skill_score": None,
                "observed_reuse_rate": None,
            }

        count = len(observations)

        absolute_errors = []
        squared_errors = []
        observed_values = []

        for predicted, observed in observations:

            absolute_errors.append(
                abs(predicted - observed)
            )

            squared_errors.append(
                (predicted - observed) ** 2
            )

            observed_values.append(observed)

        mae = sum(absolute_errors) / count
        brier_score = sum(squared_errors) / count

        prevalence = sum(observed_values) / count

        baseline_brier_score = (
            prevalence * (1.0 - prevalence)
        )

        if baseline_brier_score <= 1e-12:
            brier_skill_score = None
        else:
            brier_skill_score = (
                1.0 - brier_score / baseline_brier_score
            )

        return {
            "observations": count,
            "mae": mae,
            "brier_score": brier_score,
            "brier_skill_score": brier_skill_score,
            "observed_reuse_rate": prevalence,
        }

    # ---------------------------------------------------------
    # Global prediction metrics
    # ---------------------------------------------------------

    def _calculate_reuse_prediction_metrics(self):
        """Calculate metrics across all completed predictions."""

        return self._calculate_metrics_from_observations(
            self.reuse_prediction_observations
        )

    def get_reuse_prediction_metrics(self):
        """Return cached global prediction metrics."""

        if self._reuse_prediction_metrics_cache is None:
            self._reuse_prediction_metrics_cache = (
                self._calculate_reuse_prediction_metrics()
            )

        return dict(self._reuse_prediction_metrics_cache)

    def get_reuse_prediction_mae(self):
        return self.get_reuse_prediction_metrics()["mae"]

    def get_reuse_brier_score(self):
        return self.get_reuse_prediction_metrics()["brier_score"]

    def get_reuse_brier_skill_score(self):
        return self.get_reuse_prediction_metrics()[
            "brier_skill_score"
        ]

    def get_observed_reuse_rate(self):
        """
        Return empirical global reuse prevalence.

        Uses completed observation outcomes only.

        When no completed observations exist, return the
        neutral default reuse prevalence of 0.5.
        """

        metrics = self.get_reuse_prediction_metrics()

        observed_rate = metrics["observed_reuse_rate"]

        if observed_rate is None:
            return self.DEFAULT_REUSE_PREVALENCE

        return observed_rate

    def get_reuse_reliability(self):
        """Return the experimental global reliability estimate."""

        metrics = self.get_reuse_prediction_metrics()

        if metrics["observations"] < self.REUSE_MIN_OBSERVATIONS:
            return 0.0

        skill = metrics["brier_skill_score"]

        if skill is None or skill <= 0.0:
            return 0.0

        return min(1.0, max(0.0, skill))

    # ---------------------------------------------------------
    # Context-specific prediction metrics
    # ---------------------------------------------------------

    def get_context_reuse_prediction_metrics(
        self,
        block_id: str,
    ):
        """
        Calculate prediction metrics for one context.

        Only completed observations belonging to block_id
        are included.
        """

        if not isinstance(block_id, str) or not block_id:
            raise ValueError(
                "block_id must be a non-empty string."
            )

        observations = [
            (predicted, observed)
            for (
                context_id,
                predicted,
                observed,
            ) in self.reuse_prediction_context_observations
            if context_id == block_id
        ]

        return self._calculate_metrics_from_observations(
            observations
        )

    def get_context_reuse_reliability(
        self,
        block_id: str,
    ) -> float:
        """
        Return experimental reliability for an individual context.

        Reliability is zero when:
            - Fewer than five observations exist.
            - Brier skill is undefined.
            - Brier skill is zero or negative.

        Otherwise, positive Brier skill is clamped to [0, 1].
        """

        metrics = self.get_context_reuse_prediction_metrics(
            block_id
        )

        if metrics["observations"] < self.REUSE_MIN_OBSERVATIONS:
            return 0.0

        skill = metrics["brier_skill_score"]

        if skill is None or skill <= 0.0:
            return 0.0

        return min(1.0, max(0.0, skill))

    # ---------------------------------------------------------
    # Access
    # ---------------------------------------------------------

    def access(self, block_id: str) -> str:
        """Access a block in HBM or host memory."""

        self.logical_clock += 1

        self._record_reuse_access(block_id)

        if block_id in self.hbm:

            block = self.hbm[block_id]

            block.access_count += 1
            block.last_access_tick = self.logical_clock

            if self.policy == "LRU":
                self.hbm.move_to_end(block_id)

            self.metrics["hbm_hits"] += 1

            return "HBM_HIT"

        if block_id in self.host:

            block = self.host.pop(block_id)

            self.host_used -= block.size_bytes

            block.access_count += 1
            block.last_access_tick = self.logical_clock

            self.metrics["host_hits"] += 1
            self.metrics["bytes_restored_to_hbm"] += (
                block.size_bytes
            )

            self._make_hbm_space(block.size_bytes)
            self._place_in_hbm(block)

            self.validate()

            return "HOST_HIT"

        self.metrics["cold_misses"] += 1

        return "COLD_MISS"

    # ---------------------------------------------------------
    # Allocation
    # ---------------------------------------------------------

    def insert(
        self,
        block_id: str,
        size_bytes: int,
        deadline_ms: Optional[float] = None,
        reuse_probability: float = 0.5,
        estimated_transfer_ms: float = 1.0,
        quality_risk: float = 0.0,
    ) -> str:
        """Insert a block with scheduling metadata."""

        if not isinstance(block_id, str) or not block_id:
            raise ValueError(
                "Block ID must be a non-empty string."
            )

        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes <= 0
        ):
            raise ValueError(
                "Block size must be a positive integer."
            )

        if size_bytes > self.hbm_capacity:
            raise ValueError(
                "Block size exceeds total HBM capacity."
            )

        if deadline_ms is not None:
            self._validate_nonnegative(
                deadline_ms,
                "deadline_ms",
            )

        self._validate_probability(
            reuse_probability,
            "reuse_probability",
        )

        self._validate_nonnegative(
            estimated_transfer_ms,
            "estimated_transfer_ms",
        )

        self._validate_probability(
            quality_risk,
            "quality_risk",
        )

        self.logical_clock += 1

        # Cancel unfinished prediction from an older allocation.
        if block_id in self.pending_reuse_predictions:

            self.pending_reuse_predictions.pop(block_id)

            self.reuse_prediction_censored += 1

        # Remove an existing copy.
        if block_id in self.hbm:

            old_block = self.hbm.pop(block_id)
            self.hbm_used -= old_block.size_bytes

        elif block_id in self.host:

            old_block = self.host.pop(block_id)
            self.host_used -= old_block.size_bytes

        block = MemoryBlock(
            block_id=block_id,
            size_bytes=size_bytes,
            access_count=0,
            created_tick=self.logical_clock,
            last_access_tick=self.logical_clock,
            deadline_ms=deadline_ms,
            reuse_probability=float(reuse_probability),
            estimated_transfer_ms=float(
                estimated_transfer_ms
            ),
            quality_risk=float(quality_risk),
        )

        self._make_hbm_space(size_bytes)
        self._place_in_hbm(block)

        self._register_reuse_prediction(
            block_id,
            reuse_probability,
        )

        self.validate()

        return "ALLOCATED"

    # ---------------------------------------------------------
    # Metadata updates
    # ---------------------------------------------------------

    def update_metadata(
        self,
        block_id: str,
        deadline_ms: Optional[float] = None,
        reuse_probability: Optional[float] = None,
        estimated_transfer_ms: Optional[float] = None,
        quality_risk: Optional[float] = None,
    ) -> str:
        """Update metadata without reallocating the block."""

        if block_id in self.hbm:
            block = self.hbm[block_id]

        elif block_id in self.host:
            block = self.host[block_id]

        else:
            raise KeyError(
                f"Block '{block_id}' does not exist in MemCore."
            )

        # Validate all supplied fields before mutation.
        if deadline_ms is not None:
            self._validate_nonnegative(
                deadline_ms,
                "deadline_ms",
            )

        if reuse_probability is not None:

            self._validate_probability(
                reuse_probability,
                "reuse_probability",
            )

            if block_id in self.pending_reuse_predictions:
                raise RuntimeError(
                    f"Cannot refresh reuse prediction for "
                    f"'{block_id}': the previous prediction "
                    "window is still active."
                )

        if estimated_transfer_ms is not None:
            self._validate_nonnegative(
                estimated_transfer_ms,
                "estimated_transfer_ms",
            )

        if quality_risk is not None:
            self._validate_probability(
                quality_risk,
                "quality_risk",
            )

        if deadline_ms is not None:
            block.deadline_ms = float(deadline_ms)

        if estimated_transfer_ms is not None:
            block.estimated_transfer_ms = float(
                estimated_transfer_ms
            )

        if quality_risk is not None:
            block.quality_risk = float(quality_risk)

        if reuse_probability is not None:

            block.reuse_probability = float(
                reuse_probability
            )

            self._register_reuse_prediction(
                block_id,
                reuse_probability,
            )

        self.validate()

        return "METADATA_UPDATED"

    # ---------------------------------------------------------
    # Adaptive reuse probability
    # ---------------------------------------------------------

    def get_effective_reuse_probability(
        self,
        block_id: str,
        predicted_probability: float,
        global_prevalence: Optional[float] = None,
    ) -> float:
        """
        Blend a context prediction with global observed prevalence.

        Formula:

            effective_probability =
                reliability * predicted_probability
                + (1 - reliability) * global_prevalence

        Reliability is an experimental trust heuristic based
        on context-specific Brier skill.

        It is not a calibrated probability of correctness.
        """

        self._validate_probability(
            predicted_probability,
            "predicted_probability",
        )

        if global_prevalence is None:
            global_prevalence = self.get_observed_reuse_rate()

        self._validate_probability(
            global_prevalence,
            "global_prevalence",
        )

        reliability = self.get_context_reuse_reliability(
            block_id
        )

        effective_probability = (
            reliability * predicted_probability
            + (1.0 - reliability) * global_prevalence
        )

        return min(
            1.0,
            max(0.0, effective_probability),
        )

    # ---------------------------------------------------------
    # Adaptive ranking
    # ---------------------------------------------------------

    def _rank_contexts_adaptively(self, contexts):
        """
        Rank contexts with comparable scheduler scores.

        Every context is evaluated using the same fixed scheduler
        weights. Only its reuse_probability input is adjusted.

        This avoids comparing scores calculated with different
        per-context scheduler weight distributions.
        """

        if not contexts:
            return []

        global_prevalence = self.get_observed_reuse_rate()

        adjusted_contexts = []
        context_details = {}

        for context in contexts:

            block_id = context["block_id"]

            predicted_probability = float(
                context["reuse_probability"]
            )

            reliability = (
                self.get_context_reuse_reliability(block_id)
            )

            effective_probability = (
                reliability * predicted_probability
                + (1.0 - reliability) * global_prevalence
            )

            adjusted_context = dict(context)

            # Change only the reuse input.
            # All scheduler weights remain unchanged.
            adjusted_context["reuse_probability"] = (
                effective_probability
            )

            adjusted_contexts.append(adjusted_context)

            context_details[block_id] = {
                "reliability": reliability,
                "predicted_reuse_probability": (
                    predicted_probability
                ),
                "effective_reuse_probability": (
                    effective_probability
                ),
                "global_reuse_prevalence": (
                    global_prevalence
                ),
            }

        # Rank all contexts together with the normal fixed weights.
        ranked = self.scheduler.rank(
            adjusted_contexts,
            now_ms=self.current_time_ms,
        )

        for item in ranked:

            details = context_details[item["block_id"]]

            item["reliability"] = details["reliability"]

            item["predicted_reuse_probability"] = (
                details["predicted_reuse_probability"]
            )

            item["effective_reuse_probability"] = (
                details["effective_reuse_probability"]
            )

            item["global_reuse_prevalence"] = (
                details["global_reuse_prevalence"]
            )

        return ranked

    # ---------------------------------------------------------
    # HBM victim selection
    # ---------------------------------------------------------

    def _select_hbm_victim(self):
        """Select the HBM block to evict."""

        if not self.hbm:
            raise RuntimeError(
                "No HBM blocks available for eviction."
            )

        if self.policy in ("LRU", "FIFO"):
            return next(iter(self.hbm))

        contexts = [
            self.get_block_metadata(block_id)
            for block_id in self.hbm
        ]

        if self.policy == "SLO":

            ranked = self.scheduler.rank(
                contexts,
                now_ms=self.current_time_ms,
            )

            victim = min(
                ranked,
                key=lambda item: (
                    item["score"],
                    item["block_id"],
                ),
            )

            return victim["block_id"]

        if self.policy == "SLO_MARGIN":

            return self.scheduler.select_victim(
                contexts,
                now_ms=self.current_time_ms,
                margin=self.slo_margin,
            )

        if self.policy in (
            "ADAPTIVE_SLO",
            "ADAPTIVE_MARGIN",
        ):

            ranked = self._rank_contexts_adaptively(
                contexts
            )

            if self.policy == "ADAPTIVE_SLO":

                victim = min(
                    ranked,
                    key=lambda item: (
                        item["score"],
                        item["block_id"],
                    ),
                )

                return victim["block_id"]

            lowest_score = min(
                item["score"] for item in ranked
            )

            # Preserve the existing per-context margin behavior.
            # Less reliable predictions receive a wider margin.
            candidates = [
                item
                for item in ranked
                if item["score"]
                <= (
                    lowest_score
                    + self.slo_margin
                    * (1.0 - item["reliability"])
                )
            ]

            context_by_id = {
                context["block_id"]: context
                for context in contexts
            }

            victim = min(
                candidates,
                key=lambda item: (
                    context_by_id[
                        item["block_id"]
                    ]["last_access_tick"],
                    context_by_id[
                        item["block_id"]
                    ]["access_count"],
                    item["block_id"],
                ),
            )

            return victim["block_id"]

        raise RuntimeError(
            f"Unsupported eviction policy: {self.policy}"
        )

    # ---------------------------------------------------------
    # Memory tier management
    # ---------------------------------------------------------

    def _make_hbm_space(self, required_bytes: int):
        """Evict blocks until sufficient HBM capacity is available."""

        while (
            self.hbm_used + required_bytes
            > self.hbm_capacity
        ):

            if not self.hbm:
                raise RuntimeError(
                    "Unable to free enough HBM capacity."
                )

            block_id = self._select_hbm_victim()
            block = self.hbm.pop(block_id)

            self.hbm_used -= block.size_bytes

            self.metrics["hbm_evictions"] += 1

            self._place_in_host(block)

    def _place_in_hbm(self, block: MemoryBlock):
        """Place a block in HBM."""

        self.hbm[block.block_id] = block
        self.hbm_used += block.size_bytes

    def _place_in_host(self, block: MemoryBlock):
        """Place an evicted block in host memory using FIFO."""

        if block.size_bytes > self.host_capacity:

            self.metrics["host_evictions"] += 1
            self.metrics["bytes_dropped"] += block.size_bytes

            return

        while (
            self.host_used + block.size_bytes
            > self.host_capacity
        ):

            if not self.host:
                break

            _, evicted_block = self.host.popitem(last=False)

            self.host_used -= evicted_block.size_bytes

            self.metrics["host_evictions"] += 1
            self.metrics["bytes_dropped"] += (
                evicted_block.size_bytes
            )

        if (
            self.host_used + block.size_bytes
            <= self.host_capacity
        ):

            self.host[block.block_id] = block
            self.host_used += block.size_bytes

            self.metrics["bytes_moved_to_host"] += (
                block.size_bytes
            )

        else:

            self.metrics["host_evictions"] += 1
            self.metrics["bytes_dropped"] += block.size_bytes

    # ---------------------------------------------------------
    # Metadata inspection
    # ---------------------------------------------------------

    def get_block_metadata(self, block_id: str) -> dict:
        """Retrieve metadata for a block in either memory tier."""

        if block_id in self.hbm:

            block = self.hbm[block_id]
            tier = "HBM"

        elif block_id in self.host:

            block = self.host[block_id]
            tier = "HOST"

        else:
            raise KeyError(
                f"Block '{block_id}' does not exist in MemCore."
            )

        return {
            "block_id": block.block_id,
            "size_bytes": block.size_bytes,
            "tier": tier,
            "access_count": block.access_count,
            "created_tick": block.created_tick,
            "last_access_tick": block.last_access_tick,
            "idle_ticks": (
                self.logical_clock - block.last_access_tick
            ),
            "deadline_ms": block.deadline_ms,
            "reuse_probability": block.reuse_probability,
            "estimated_transfer_ms": (
                block.estimated_transfer_ms
            ),
            "quality_risk": block.quality_risk,
        }

    # ---------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------

    def report(self) -> dict:
        """Return memory usage, access metrics and prediction state."""

        total_accesses = (
            self.metrics["hbm_hits"]
            + self.metrics["host_hits"]
            + self.metrics["cold_misses"]
        )

        hbm_hit_rate = (
            self.metrics["hbm_hits"] / total_accesses
            if total_accesses
            else 0.0
        )

        cached_hit_rate = (
            (
                self.metrics["hbm_hits"]
                + self.metrics["host_hits"]
            ) / total_accesses
            if total_accesses
            else 0.0
        )

        prediction_metrics = (
            self.get_reuse_prediction_metrics()
        )

        # Report context reliability for currently resident blocks.
        context_reliability = {}

        for block_id in list(self.hbm) + list(self.host):
            context_reliability[block_id] = (
                self.get_context_reuse_reliability(block_id)
            )

        return {
            "policy": self.policy,
            "slo_margin": self.slo_margin,
            "logical_clock": self.logical_clock,
            "current_time_ms": self.current_time_ms,

            "hbm_used_bytes": self.hbm_used,
            "hbm_capacity_bytes": self.hbm_capacity,

            "host_used_bytes": self.host_used,
            "host_capacity_bytes": self.host_capacity,

            "hbm_blocks": len(self.hbm),
            "host_blocks": len(self.host),

            "hbm_hit_rate": hbm_hit_rate,
            "cached_hit_rate": cached_hit_rate,

            "access_request_clock": self.access_request_clock,

            "reuse_prediction_observations": (
                prediction_metrics["observations"]
            ),

            "reuse_prediction_pending": (
                len(self.pending_reuse_predictions)
            ),

            "reuse_prediction_censored": (
                self.reuse_prediction_censored
            ),

            "reuse_prediction_mae": (
                prediction_metrics["mae"]
            ),

            "reuse_prediction_brier_score": (
                prediction_metrics["brier_score"]
            ),

            "reuse_prediction_brier_skill_score": (
                prediction_metrics["brier_skill_score"]
            ),

            "observed_reuse_rate": (
                prediction_metrics["observed_reuse_rate"]
            ),

            "global_reuse_prevalence": (
                self.get_observed_reuse_rate()
            ),

            "reuse_reliability": (
                self.get_reuse_reliability()
            ),

            "context_reuse_reliability": (
                context_reliability
            ),

            **self.metrics,
        }

    # ---------------------------------------------------------
    # Invariant checks
    # ---------------------------------------------------------

    def validate(self) -> bool:
        """Verify accounting, capacity and metadata invariants."""

        assert self.hbm_used == sum(
            block.size_bytes
            for block in self.hbm.values()
        ), "HBM byte accounting mismatch."

        assert self.host_used == sum(
            block.size_bytes
            for block in self.host.values()
        ), "Host byte accounting mismatch."

        assert 0 <= self.hbm_used <= self.hbm_capacity
        assert 0 <= self.host_used <= self.host_capacity

        assert not (
            set(self.hbm.keys()) & set(self.host.keys())
        ), "Duplicate block found across memory tiers."

        assert self.policy in self.SUPPORTED_POLICIES

        assert math.isfinite(self.current_time_ms)
        assert self.current_time_ms >= 0

        assert math.isfinite(self.slo_margin)
        assert self.slo_margin >= 0

        assert self.access_request_clock >= 0
        assert self.reuse_prediction_censored >= 0

        assert len(
            self.reuse_prediction_observations
        ) <= self.REUSE_OBSERVATION_WINDOW

        assert len(
            self.reuse_prediction_context_observations
        ) <= self.REUSE_OBSERVATION_WINDOW

        # Validate global completed observations.
        for predicted, observed in self.reuse_prediction_observations:

            assert math.isfinite(predicted)
            assert 0.0 <= predicted <= 1.0
            assert observed in (0.0, 1.0)

        # Validate context-specific completed observations.
        for (
            context_block_id,
            predicted_probability,
            observed_reuse,
        ) in self.reuse_prediction_context_observations:

            assert isinstance(context_block_id, str)
            assert context_block_id

            assert math.isfinite(predicted_probability)
            assert 0.0 <= predicted_probability <= 1.0

            assert observed_reuse in (0.0, 1.0)

        # Validate active prediction records.
        for block_id, prediction in (
            self.pending_reuse_predictions.items()
        ):

            assert isinstance(block_id, str)
            assert prediction["block_id"] == block_id

            assert math.isfinite(
                prediction["predicted_probability"]
            )

            assert 0.0 <= prediction[
                "predicted_probability"
            ] <= 1.0

            assert isinstance(
                prediction["observed_reuse"],
                bool,
            )

            assert prediction["expires_at"] > (
                self.access_request_clock
            )

        # Validate all resident blocks.
        for block in (
            list(self.hbm.values())
            + list(self.host.values())
        ):

            assert block.size_bytes > 0
            assert block.access_count >= 0

            assert block.created_tick <= self.logical_clock
            assert block.last_access_tick <= self.logical_clock

            assert 0.0 <= block.reuse_probability <= 1.0
            assert 0.0 <= block.quality_risk <= 1.0

            assert block.estimated_transfer_ms >= 0

            if block.deadline_ms is not None:
                assert block.deadline_ms >= 0

        return True
