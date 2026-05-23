from __future__ import annotations


def track_feature_flow(*_args, **_kwargs):
    """舊版 SfM 特徵追蹤已被新版 DA3 管線取代。"""

    raise RuntimeError(
        "track_feature_flow() has been removed from the active pipeline. "
        "The thesis-aligned architecture now uses DA3 dense depth instead of SfM feature flow."
    )
