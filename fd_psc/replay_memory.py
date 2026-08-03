"""Continuous-window replay, balanced reservoirs, and GRASP scheduling."""

from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch


class ReplayError(RuntimeError):
    pass


def _derived_seed(seed: int, stable_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{stable_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _as_unit_vector(value: Any, *, name: str) -> torch.Tensor:
    vector = torch.as_tensor(value, dtype=torch.float32).detach().cpu().flatten().clone()
    if vector.numel() == 0 or not torch.isfinite(vector).all():
        raise ReplayError(f"{name} must be a finite non-empty vector")
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or float(norm) <= 1.0e-12:
        raise ReplayError(f"{name} must have non-zero norm")
    return vector / norm


def _cosine_distance(left: torch.Tensor, right: torch.Tensor, *, name: str) -> float:
    """Return a numerically bounded cosine distance between unit vectors."""

    if left.numel() != right.numel():
        raise ReplayError(f"{name} dimension mismatch")
    similarity = float(torch.dot(left, right).item())
    if not math.isfinite(similarity):
        raise ReplayError(f"{name} must be finite")
    similarity = min(1.0, max(-1.0, similarity))
    return min(2.0, max(0.0, 1.0 - similarity))


@dataclass
class ReplayWindow:
    """One contiguous replayable JEPA window with stable provenance."""

    window_id: str
    trajectory_id: str
    transition_ids: Tuple[str, ...]
    frame_ids: Tuple[str, ...]
    timesteps: Tuple[int, ...]
    content_hash: str
    context_identifier: str
    context_embedding: Any
    visual_latent: Any
    proprio: Any
    actions: Any
    time_positions: Any = None
    prediction_mask: Any = None
    residual: Any = None
    contact: Any = None
    contact_available: bool = False
    dynamics_change: Any = None
    dynamics_change_available: bool = False
    source_episode: str = ""
    provenance: str = "episode_support"
    committed: bool = False
    preprocess_hash: str = ""
    base_checkpoint_hash: str = ""
    latent_adapter_schema: str = ""
    difficulty_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # These fields are populated on a replay *snapshot*, not when the window
    # is admitted.  A repair trajectory therefore sees one frozen online
    # cluster center even though the persistent prototype may move after a
    # later successful slow commit.
    frozen_context_cluster_id: str = ""
    frozen_context_cluster_prototype: Any = None
    frozen_context_cluster_distance: Optional[float] = None

    def __post_init__(self) -> None:
        self.window_id = str(self.window_id)
        self.trajectory_id = str(self.trajectory_id)
        self.transition_ids = tuple(str(item) for item in self.transition_ids)
        self.frame_ids = tuple(str(item) for item in self.frame_ids)
        self.timesteps = tuple(int(item) for item in self.timesteps)
        self.content_hash = str(self.content_hash).lower()
        self.context_identifier = str(self.context_identifier)
        self.source_episode = str(self.source_episode)
        self.frozen_context_cluster_id = str(self.frozen_context_cluster_id)
        if not self.window_id or not self.trajectory_id or not self.context_identifier:
            raise ReplayError("window_id, trajectory_id, and context_identifier must be non-empty")
        if not self.transition_ids or not self.frame_ids or not self.timesteps:
            raise ReplayError("a replay window requires transition, frame, and timestep identities")
        if len(set(self.transition_ids)) != len(self.transition_ids):
            raise ReplayError("transition_ids must be unique within a window")
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ReplayError("frame_ids must be unique within a window")
        if len(self.frame_ids) != len(self.timesteps):
            raise ReplayError("frame_ids and timesteps must describe the same continuous frames")
        if len(self.transition_ids) != max(0, len(self.timesteps) - 1):
            raise ReplayError("a continuous window requires exactly one transition between adjacent frames")
        if any(b != a + 1 for a, b in zip(self.timesteps, self.timesteps[1:])):
            raise ReplayError("replay timesteps must form one contiguous trajectory window")
        if len(self.content_hash) != 64 or any(ch not in "0123456789abcdef" for ch in self.content_hash):
            raise ReplayError("content_hash must be a SHA-256 hex digest")
        self.context_embedding = _as_unit_vector(self.context_embedding, name="context_embedding")
        if self.difficulty_score is not None and not math.isfinite(float(self.difficulty_score)):
            raise ReplayError("difficulty_score must be finite")
        if self.frozen_context_cluster_prototype is not None:
            if not self.frozen_context_cluster_id:
                raise ReplayError("a frozen cluster prototype requires a stable cluster id")
            prototype = _as_unit_vector(
                self.frozen_context_cluster_prototype,
                name="frozen context cluster prototype",
            )
            distance = _cosine_distance(
                self.context_embedding,
                prototype,
                name="frozen context cluster prototype",
            )
            if self.frozen_context_cluster_distance is not None and not math.isclose(
                float(self.frozen_context_cluster_distance),
                distance,
                rel_tol=1.0e-6,
                abs_tol=1.0e-6,
            ):
                raise ReplayError("frozen context cluster distance does not match its prototype")
            self.frozen_context_cluster_prototype = prototype
            self.frozen_context_cluster_distance = distance
        elif self.frozen_context_cluster_distance is not None:
            raise ReplayError("a frozen cluster distance requires its frozen prototype")

    def clone(self) -> "ReplayWindow":
        return copy.deepcopy(self)

    def with_frozen_context_cluster(
        self,
        cluster_id: str,
        prototype: Any,
    ) -> "ReplayWindow":
        """Clone this window and attach one auditable cluster-center snapshot."""

        stable_cluster_id = str(cluster_id)
        if not stable_cluster_id:
            raise ReplayError("frozen context cluster id must be non-empty")
        center = _as_unit_vector(prototype, name="frozen context cluster prototype")
        distance = _cosine_distance(
            self.context_embedding,
            center,
            name="frozen context cluster prototype",
        )
        result = self.clone()
        result.frozen_context_cluster_id = stable_cluster_id
        result.frozen_context_cluster_prototype = center.clone()
        result.frozen_context_cluster_distance = distance
        return result

    def difficulty(self) -> float:
        # ``difficulty_score`` is an optional precomputed residual/stability
        # signal (the trainer stores theta0 JEPA residual here).  Contact and
        # dynamics events remain additive rather than being silently masked by
        # that cached residual.
        value = float(self.difficulty_score) if self.difficulty_score is not None else 0.0
        if self.difficulty_score is None and self.residual is not None:
            residual = torch.as_tensor(self.residual, dtype=torch.float32)
            if residual.numel() and torch.isfinite(residual).all():
                value += float(torch.linalg.vector_norm(residual).item())
        if self.contact_available and bool(torch.as_tensor(self.contact).any()):
            value += 1.0
        if self.dynamics_change_available and bool(torch.as_tensor(self.dynamics_change).any()):
            value += 1.0
        return value


@dataclass
class _Cluster:
    cluster_id: str
    prototype: torch.Tensor
    seen_count: int = 0
    admission_count: int = 0
    windows: List[ReplayWindow] = field(default_factory=list)
    rng_state: object = None


@dataclass(frozen=True)
class ReplayAdmission:
    admitted: bool
    cluster_id: str
    replaced_window_id: Optional[str]
    reason: str


class ClusterBalancedReplay:
    """Deterministic online clustering plus per-cluster reservoir sampling."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        capacity: int,
        *,
        maximum_context_clusters: int,
        new_cluster_similarity_threshold: float,
        minimum_windows_per_cluster: int,
        seed: int,
    ) -> None:
        if int(capacity) < 0 or int(maximum_context_clusters) < 0 or int(minimum_windows_per_cluster) < 0:
            raise ValueError("replay capacities must be non-negative")
        if not -1.0 <= float(new_cluster_similarity_threshold) <= 1.0:
            raise ValueError("new_cluster_similarity_threshold must be in [-1,1]")
        self.capacity = int(capacity)
        self.maximum_context_clusters = int(maximum_context_clusters)
        self.new_cluster_similarity_threshold = float(new_cluster_similarity_threshold)
        self.minimum_windows_per_cluster = int(minimum_windows_per_cluster)
        self.seed = int(seed)
        self._clusters: Dict[str, _Cluster] = {}
        self._next_cluster_index = 0
        self._seen_window_ids: set[str] = set()
        self._sample_rng = random.Random(_derived_seed(self.seed, "balanced-sampler"))

    def __len__(self) -> int:
        return sum(len(cluster.windows) for cluster in self._clusters.values())

    @property
    def cluster_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._clusters))

    @property
    def minimum_capacity_degraded(self) -> bool:
        return bool(self._clusters) and self.capacity < len(self._clusters) * self.minimum_windows_per_cluster

    def _quotas(self) -> Dict[str, int]:
        ids = sorted(self._clusters)
        if not ids:
            return {}
        base, remainder = divmod(self.capacity, len(ids))
        return {cluster_id: base + int(i < remainder) for i, cluster_id in enumerate(ids)}

    def _cluster_rng(self, cluster: _Cluster) -> random.Random:
        rng = random.Random()
        if cluster.rng_state is None:
            rng.seed(_derived_seed(self.seed, cluster.cluster_id))
        else:
            rng.setstate(cluster.rng_state)
        return rng

    def _save_cluster_rng(self, cluster: _Cluster, rng: random.Random) -> None:
        cluster.rng_state = rng.getstate()

    def _rebalance_to_quotas(self) -> None:
        quotas = self._quotas()
        for cluster_id in sorted(self._clusters):
            cluster = self._clusters[cluster_id]
            quota = quotas[cluster_id]
            if len(cluster.windows) <= quota:
                continue
            rng = self._cluster_rng(cluster)
            keep = sorted(rng.sample(range(len(cluster.windows)), quota)) if quota else []
            cluster.windows = [cluster.windows[index] for index in keep]
            self._save_cluster_rng(cluster, rng)

    def _new_cluster(self, embedding: torch.Tensor) -> _Cluster:
        if self.maximum_context_clusters <= 0:
            raise ReplayError("maximum_context_clusters=0 disables replay admission")
        cluster_id = f"cluster-{self._next_cluster_index:08d}"
        self._next_cluster_index += 1
        cluster = _Cluster(cluster_id=cluster_id, prototype=embedding.clone())
        rng = random.Random(_derived_seed(self.seed, cluster_id))
        cluster.rng_state = rng.getstate()
        self._clusters[cluster_id] = cluster
        self._rebalance_to_quotas()
        return cluster

    def _nearest_cluster(self, embedding: torch.Tensor) -> Tuple[Optional[_Cluster], float]:
        best: Optional[_Cluster] = None
        best_similarity = -float("inf")
        for cluster_id in sorted(self._clusters):
            cluster = self._clusters[cluster_id]
            if cluster.prototype.numel() != embedding.numel():
                continue
            similarity = float(torch.dot(cluster.prototype, embedding).item())
            if similarity > best_similarity:
                best, best_similarity = cluster, similarity
        return best, best_similarity

    def _assign_cluster(self, embedding: torch.Tensor) -> _Cluster:
        nearest, similarity = self._nearest_cluster(embedding)
        if nearest is None:
            return self._new_cluster(embedding)
        if (
            similarity < self.new_cluster_similarity_threshold
            and len(self._clusters) < self.maximum_context_clusters
        ):
            return self._new_cluster(embedding)
        return nearest

    def add_committed_window(self, window: ReplayWindow, *, commit_kind: str) -> ReplayAdmission:
        if commit_kind != "slow":
            raise ReplayError("global historical replay may only be updated by a successful slow commit")
        if not window.committed:
            raise ReplayError("window must be marked committed before historical replay admission")
        if window.provenance != "episode_support":
            raise ReplayError("historical replay only accepts support-derived windows")
        if self.capacity == 0:
            return ReplayAdmission(False, "", None, "capacity_zero")
        if window.window_id in self._seen_window_ids:
            raise ReplayError(f"duplicate replay window_id: {window.window_id}")
        embedding = _as_unit_vector(window.context_embedding, name="context_embedding")
        cluster = self._assign_cluster(embedding)
        if cluster.prototype.numel() != embedding.numel():
            raise ReplayError("context embedding dimension changed within a replay state")
        self._seen_window_ids.add(window.window_id)
        previous_seen = cluster.seen_count
        cluster.seen_count += 1
        mean = (cluster.prototype * float(previous_seen) + embedding) / float(cluster.seen_count)
        cluster.prototype = _as_unit_vector(mean, name="cluster prototype")
        quota = self._quotas()[cluster.cluster_id]
        if quota <= 0:
            return ReplayAdmission(False, cluster.cluster_id, None, "fair_capacity_degradation")
        stored = window.clone()
        # Cluster annotations describe a read snapshot.  Never persist a
        # caller's stale annotation as part of reservoir state.
        stored.frozen_context_cluster_id = ""
        stored.frozen_context_cluster_prototype = None
        stored.frozen_context_cluster_distance = None
        if len(cluster.windows) < quota:
            cluster.windows.append(stored)
            cluster.admission_count += 1
            return ReplayAdmission(True, cluster.cluster_id, None, "under_quota")
        rng = self._cluster_rng(cluster)
        index = rng.randrange(cluster.seen_count)
        replaced: Optional[str] = None
        admitted = index < quota
        if admitted:
            replaced = cluster.windows[index].window_id
            cluster.windows[index] = stored
            cluster.admission_count += 1
        self._save_cluster_rng(cluster, rng)
        return ReplayAdmission(
            admitted,
            cluster.cluster_id,
            replaced,
            "reservoir_replace" if admitted else "reservoir_reject",
        )

    def add_committed_windows(self, windows: Iterable[ReplayWindow], *, commit_kind: str) -> List[ReplayAdmission]:
        return [self.add_committed_window(window, commit_kind=commit_kind) for window in windows]

    @staticmethod
    def _snapshot_window(cluster: _Cluster, window: ReplayWindow) -> ReplayWindow:
        return window.with_frozen_context_cluster(cluster.cluster_id, cluster.prototype)

    def windows(self) -> Tuple[ReplayWindow, ...]:
        return tuple(
            self._snapshot_window(self._clusters[cluster_id], window)
            for cluster_id in sorted(self._clusters)
            for window in sorted(self._clusters[cluster_id].windows, key=lambda item: item.window_id)
        )

    def sample_balanced(self, count: int, *, allow_replacement: bool = True) -> Tuple[ReplayWindow, ...]:
        if int(count) < 0:
            raise ValueError("sample count must be non-negative")
        nonempty = [self._clusters[key] for key in sorted(self._clusters) if self._clusters[key].windows]
        if not nonempty or count == 0:
            return ()
        if not allow_replacement:
            count = min(int(count), sum(len(cluster.windows) for cluster in nonempty))
        result: List[ReplayWindow] = []
        available: Dict[str, List[int]] = {
            cluster.cluster_id: list(range(len(cluster.windows))) for cluster in nonempty
        }
        cursor = 0
        while len(result) < count:
            cluster = nonempty[cursor % len(nonempty)]
            cursor += 1
            indices = available[cluster.cluster_id]
            if indices:
                choice_pos = self._sample_rng.randrange(len(indices))
                index = indices.pop(choice_pos)
            elif allow_replacement:
                index = self._sample_rng.randrange(len(cluster.windows))
            else:
                if all(not items for items in available.values()):
                    break
                continue
            result.append(self._snapshot_window(cluster, cluster.windows[index]))
        return tuple(result)

    def state_dict(self) -> Dict[str, Any]:
        clusters = {}
        for cluster_id, cluster in sorted(self._clusters.items()):
            clusters[cluster_id] = {
                "prototype": cluster.prototype.clone(),
                "seen_count": cluster.seen_count,
                "admission_count": cluster.admission_count,
                "windows": copy.deepcopy(cluster.windows),
                "rng_state": copy.deepcopy(cluster.rng_state),
            }
        return {
            "schema_version": self.SCHEMA_VERSION,
            "config": {
                "capacity": self.capacity,
                "maximum_context_clusters": self.maximum_context_clusters,
                "new_cluster_similarity_threshold": self.new_cluster_similarity_threshold,
                "minimum_windows_per_cluster": self.minimum_windows_per_cluster,
                "seed": self.seed,
            },
            "clusters": clusters,
            "next_cluster_index": self._next_cluster_index,
            "seen_window_ids": sorted(self._seen_window_ids),
            "sample_rng_state": copy.deepcopy(self._sample_rng.getstate()),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise ReplayError("unsupported replay schema")
        expected = {
            "capacity": self.capacity,
            "maximum_context_clusters": self.maximum_context_clusters,
            "new_cluster_similarity_threshold": self.new_cluster_similarity_threshold,
            "minimum_windows_per_cluster": self.minimum_windows_per_cluster,
            "seed": self.seed,
        }
        if dict(state.get("config", {})) != expected:
            raise ReplayError("replay checkpoint/config mismatch")
        restored: Dict[str, _Cluster] = {}
        for cluster_id, raw in sorted(dict(state.get("clusters", {})).items()):
            restored[str(cluster_id)] = _Cluster(
                cluster_id=str(cluster_id),
                prototype=_as_unit_vector(raw["prototype"], name="cluster prototype"),
                seen_count=int(raw["seen_count"]),
                admission_count=int(raw["admission_count"]),
                windows=copy.deepcopy(list(raw["windows"])),
                rng_state=copy.deepcopy(raw["rng_state"]),
            )
        self._clusters = restored
        self._next_cluster_index = int(state.get("next_cluster_index", 0))
        self._seen_window_ids = set(str(item) for item in state.get("seen_window_ids", []))
        self._sample_rng.setstate(copy.deepcopy(state["sample_rng_state"]))
        if len(self) > self.capacity:
            raise ReplayError("restored replay exceeds configured capacity")


@dataclass(frozen=True)
class GRASPBatch:
    phase: str
    windows: Tuple[ReplayWindow, ...]
    duplicate_rate: float


@dataclass(frozen=True)
class GRASPWindowScore:
    """Auditable easy/hard score against a frozen context-cluster center."""

    window_id: str
    cluster_id: str
    context_cluster_distance: float
    residual_contact_dynamics: float
    total_score: float


class GRASPSampler:
    """Easy -> balanced -> hard cumulative repair sampling schedule."""

    PHASES = ("easy", "balanced", "hard")
    FRACTIONS = (0.3, 0.4, 0.3)

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._rng = random.Random(_derived_seed(self.seed, "grasp"))

    @classmethod
    def phase_counts(cls, total_steps: int) -> Dict[str, int]:
        if int(total_steps) < 0:
            raise ValueError("total_steps must be non-negative")
        raw = [float(total_steps) * fraction for fraction in cls.FRACTIONS]
        floors = [math.floor(value) for value in raw]
        remainder = int(total_steps) - sum(floors)
        order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - floors[i]), i))
        for index in order[:remainder]:
            floors[index] += 1
        return dict(zip(cls.PHASES, floors))

    @classmethod
    def phase_for_step(cls, step_index: int, total_steps: int) -> str:
        if not 0 <= int(step_index) < int(total_steps):
            raise IndexError("GRASP step index is out of range")
        counts = cls.phase_counts(total_steps)
        boundary = 0
        for phase in cls.PHASES:
            boundary += counts[phase]
            if step_index < boundary:
                return phase
        raise AssertionError("invalid GRASP phase allocation")

    @staticmethod
    def _fallback_cluster_id(window: ReplayWindow) -> str:
        cluster_id = str(getattr(window, "frozen_context_cluster_id", ""))
        return cluster_id or f"context:{window.context_identifier}"

    @classmethod
    def score_windows(cls, windows: Sequence[ReplayWindow]) -> Tuple[GRASPWindowScore, ...]:
        """Compute stable scores without mutating replay or sampler RNG state.

        Historical replay snapshots carry the online cluster id, center, and
        cosine distance.  Exception-local or legacy windows do not, so their
        context identifier defines a deterministic fallback cluster whose
        normalized mean is frozen for this input snapshot.
        """

        ordered = sorted(windows, key=lambda item: item.window_id)
        if len({item.window_id for item in ordered}) != len(ordered):
            raise ReplayError("GRASP requires unique replay window ids")

        fallback_groups: Dict[str, List[ReplayWindow]] = {}
        distances: Dict[str, Tuple[str, float]] = {}
        for window in ordered:
            cluster_id = cls._fallback_cluster_id(window)
            prototype = getattr(window, "frozen_context_cluster_prototype", None)
            recorded_distance = getattr(window, "frozen_context_cluster_distance", None)
            if prototype is not None:
                center = _as_unit_vector(
                    prototype,
                    name="frozen context cluster prototype",
                )
                distance = _cosine_distance(
                    window.context_embedding,
                    center,
                    name="frozen context cluster prototype",
                )
                if recorded_distance is not None and not math.isclose(
                    float(recorded_distance),
                    distance,
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-6,
                ):
                    raise ReplayError(
                        f"frozen cluster distance changed for replay window {window.window_id}"
                    )
                distances[window.window_id] = (cluster_id, distance)
            elif recorded_distance is not None:
                raise ReplayError(
                    f"replay window {window.window_id} has a cluster distance without a center"
                )
            else:
                fallback_groups.setdefault(cluster_id, []).append(window)

        for cluster_id, group in sorted(fallback_groups.items()):
            members = sorted(group, key=lambda item: item.window_id)
            dimension = members[0].context_embedding.numel()
            if any(item.context_embedding.numel() != dimension for item in members):
                raise ReplayError(f"context cluster {cluster_id} changes embedding dimension")
            center_sum = torch.stack([item.context_embedding for item in members], dim=0).sum(dim=0)
            center_norm = torch.linalg.vector_norm(center_sum)
            # An exactly cancelling fallback mean has no direction.  Stable
            # window-id order supplies a deterministic medoid-like center.
            if not torch.isfinite(center_norm):
                raise ReplayError(f"context cluster {cluster_id} has a non-finite center")
            if float(center_norm) <= 1.0e-12:
                center = members[0].context_embedding.clone()
            else:
                center = center_sum / center_norm
            for window in members:
                distances[window.window_id] = (
                    cluster_id,
                    _cosine_distance(
                        window.context_embedding,
                        center,
                        name=f"context cluster {cluster_id}",
                    ),
                )

        scores: List[GRASPWindowScore] = []
        for window in ordered:
            cluster_id, distance = distances[window.window_id]
            signal = float(window.difficulty())
            if not math.isfinite(signal):
                raise ReplayError(f"non-finite difficulty for replay window {window.window_id}")
            total = signal + distance
            if not math.isfinite(total):
                raise ReplayError(f"non-finite GRASP score for replay window {window.window_id}")
            scores.append(
                GRASPWindowScore(
                    window_id=window.window_id,
                    cluster_id=cluster_id,
                    context_cluster_distance=distance,
                    residual_contact_dynamics=signal,
                    total_score=total,
                )
            )
        return tuple(scores)

    def _sample_balanced(
        self,
        windows: Sequence[ReplayWindow],
        scores: Mapping[str, GRASPWindowScore],
        *,
        step_index: int,
        batch_size: int,
    ) -> List[ReplayWindow]:
        buckets: Dict[str, List[ReplayWindow]] = {}
        for window in sorted(windows, key=lambda item: item.window_id):
            buckets.setdefault(scores[window.window_id].cluster_id, []).append(window)
        cluster_ids = sorted(buckets)
        available = {cluster_id: list(items) for cluster_id, items in buckets.items()}
        selections: List[ReplayWindow] = []
        for offset in range(batch_size):
            cluster_id = cluster_ids[(int(step_index) + offset) % len(cluster_ids)]
            pool = available[cluster_id]
            if pool:
                index = self._rng.randrange(len(pool))
                selections.append(pool.pop(index).clone())
            else:
                source = buckets[cluster_id]
                selections.append(source[self._rng.randrange(len(source))].clone())
        return selections

    def sample(
        self,
        windows: Sequence[ReplayWindow],
        *,
        step_index: int,
        total_steps: int,
        batch_size: int,
    ) -> GRASPBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not windows:
            return GRASPBatch(self.phase_for_step(step_index, total_steps), (), 0.0)
        phase = self.phase_for_step(step_index, total_steps)
        score_values = self.score_windows(windows)
        scores = {item.window_id: item for item in score_values}
        easy_order = sorted(
            windows,
            key=lambda item: (scores[item.window_id].total_score, item.window_id),
        )
        hard_order = sorted(
            windows,
            key=lambda item: (-scores[item.window_id].total_score, item.window_id),
        )
        if phase == "easy":
            pool = easy_order[: max(1, math.ceil(len(easy_order) * 0.3))]
        elif phase == "hard":
            pool = hard_order[: max(1, math.ceil(len(hard_order) * 0.3))]
        else:
            pool = list(easy_order)
        selections: List[ReplayWindow] = []
        if phase == "balanced":
            selections = self._sample_balanced(
                pool,
                scores,
                step_index=step_index,
                batch_size=batch_size,
            )
        elif len(pool) >= batch_size:
            selections = [item.clone() for item in pool[:batch_size]]
        else:
            selections = [item.clone() for item in pool]
            while len(selections) < batch_size:
                selections.append(pool[self._rng.randrange(len(pool))].clone())
        unique = len({item.window_id for item in selections})
        duplicate_rate = 1.0 - unique / float(len(selections))
        return GRASPBatch(phase, tuple(selections), duplicate_rate)

    def state_dict(self) -> Dict[str, Any]:
        return {"seed": self.seed, "rng_state": copy.deepcopy(self._rng.getstate())}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("seed", -1)) != self.seed:
            raise ReplayError("GRASP seed mismatch")
        self._rng.setstate(copy.deepcopy(state["rng_state"]))


__all__ = [
    "ClusterBalancedReplay",
    "GRASPBatch",
    "GRASPSampler",
    "GRASPWindowScore",
    "ReplayAdmission",
    "ReplayError",
    "ReplayWindow",
]
