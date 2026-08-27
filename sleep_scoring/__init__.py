"""Capture back end: read a position log, turn it into a clean posture timeline.

Statistics, data-quality gating and risk scoring live in the ``sleep_statistics``
and ``scoring_model`` modules at the top level. This package stops at the
timeline, which is the last point where the two capture formats still differ.
"""

from .config import Config, DEFAULT_LABEL_MAP, Posture
from .intervals import Interval, Timeline, build_timeline
from .loading import LoadResult, Sample, load_any, load_csv, load_log

__all__ = [
    "Config", "DEFAULT_LABEL_MAP", "Posture",
    "Interval", "Timeline", "build_timeline",
    "LoadResult", "Sample", "load_any", "load_csv", "load_log",
]
