from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RoiCandidate:
    """描述候選區域集合 B_t 中的單一 2D ROI。"""

    roi_id: str
    bbox: list[int]
    area_px: int
    touch_border: bool
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DepthFrameResult:
    """描述單幀相對深度推論結果。"""

    backend: str
    relative_depth: list[list[float]] | None = None
    confidence: list[list[float]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScaleAlignmentResult:
    """描述由路面幾何估得的尺度對齊結果。"""

    scale_factor: float
    candidate_count: int
    selected_count: int
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlaneEstimate:
    """描述局部世界座標系中的基準地平面 Π_t。"""

    normal: list[float]
    offset: float
    inlier_ratio: float
    support_count: int
    source: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateObservation:
    """描述候選區域三維點集 S_{t,k}^{cand} 形成的 cluster 級觀測。"""

    candidate_id: str
    roi_id: str
    obstacle_type: str
    bbox: list[int]
    centroid_2d: list[float]
    anchor: list[float]
    height_m: float
    width_m: float
    distance_m: float
    z_range_m: list[float]
    point_count: int
    abnormal_count: int
    support_points: list[list[float]] = field(default_factory=list)
    candidate_type: str = ""  # "curb/step" | "pothole/pothole" | ""
    metadata: dict[str, Any] = field(default_factory=dict)
    object_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CrossFrameMatch:
    """描述相鄰兩幀之間的候選延續摘要 M_t。"""

    prev_candidate_id: str
    curr_candidate_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskAssessment:
    """描述單一追蹤物件的風險矩陣與控制判定。"""

    object_id: str
    risk_weight: str
    decision: str
    bypassable: bool
    w_clear_m: float
    d_brake_m: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrackedObject:
    """描述跨幀維持的障礙物物件 O_k。"""

    object_id: str
    roi_id: str
    candidate_id: str
    obstacle_type: str
    anchor: list[float]
    height_m: float
    width_m: float
    distance_m: float
    state: str
    miss_count: int
    hit_count: int
    last_seen_frame: str
    support_points: list[list[float]] = field(default_factory=list)
    risk: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameState:
    """描述單幀在新版論文管線下的完整輸出。"""

    frame_id: str
    depth_stats: dict[str, Any]
    scale_alignment: dict[str, Any]
    road_mask_stats: dict[str, Any]
    plane_model: dict[str, Any]
    roi_candidates: list[dict[str, Any]]
    candidate_clusters: list[dict[str, Any]]
    cross_frame_matches: list[dict[str, Any]]
    tracked_objects: list[dict[str, Any]]
    risk_events: list[dict[str, Any]]
    timing_ms: dict[str, float]
    temporal_measurements: list[dict[str, Any]] = field(default_factory=list)
    raw_risk_assessments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
