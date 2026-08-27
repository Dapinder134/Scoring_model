"""Tunable thresholds and the position vocabulary."""

from dataclasses import dataclass, field
from typing import Dict, Optional


class Posture:
    """Canonical posture labels used everywhere downstream."""

    SUPINE = "supine"
    PRONE = "prone"
    LEFT = "left"
    RIGHT = "right"
    ABSENT = "absent"        # nobody detected in frame
    UNMONITORED = "unmonitored"  # gap too long to attribute to any posture

    BODY = (SUPINE, PRONE, LEFT, RIGHT)
    NON_BODY = (ABSENT, UNMONITORED)


#: Maps the capture app's ``Position`` strings onto canonical postures.
#: Keys are compared case-insensitively with surrounding whitespace stripped.
DEFAULT_LABEL_MAP: Dict[str, str] = {
    "back": Posture.SUPINE,
    "supine": Posture.SUPINE,
    "front": Posture.PRONE,
    "prone": Posture.PRONE,
    "stomach": Posture.PRONE,
    "face down": Posture.PRONE,
    "left side": Posture.LEFT,
    "left": Posture.LEFT,
    "right side": Posture.RIGHT,
    "right": Posture.RIGHT,
    "no person detected": Posture.ABSENT,
    "no person": Posture.ABSENT,
    "none": Posture.ABSENT,
    "": Posture.ABSENT,
}


@dataclass
class Config:
    """Thresholds for cleaning the raw label stream and gating the result.

    The defaults encode one core assumption: a *real* change of sleeping
    posture is a rare, slow event (a handful per hour at most), so anything
    faster than ``min_dwell_s`` is classifier noise rather than movement.
    """

    # --- interval semantics -------------------------------------------------
    #: A row's posture is held until the next row's timestamp (the user's
    #: "unbroken stretch" rule). If set, a gap longer than this is not
    #: attributed to the posture and becomes UNMONITORED instead. ``None``
    #: applies the rule literally, with no cap.
    max_hold_s: Optional[float] = None
    #: Duration credited to the final row, which has no successor to bound it.
    tail_hold_s: float = 0.0
    #: Merge rows sharing a timestamp into one winning row before any duration
    #: is attributed. Disable only to reproduce the uncleaned stream.
    collapse_duplicates: bool = True
    #: A stretch longer than this rests on a single sample rather than on
    #: continuous observation, and is counted as inferred rather than observed.
    evidence_gap_s: float = 120.0

    # --- de-noising ---------------------------------------------------------
    #: A posture must persist at least this long to count as a genuine turn.
    #: Shorter runs are absorbed into their neighbours.
    min_dwell_s: float = 30.0
    #: An ABSENT stretch shorter than this is treated as detector dropout and
    #: bridged with the surrounding posture. Longer ones are real absences.
    max_bridge_absence_s: float = 120.0

    # --- data-quality gates -------------------------------------------------
    #: Minimum analysed in-bed time before risk scores are considered usable.
    min_scoreable_h: float = 2.0
    #: Minimum fraction of in-bed time carrying an actual posture label.
    min_posture_coverage: float = 0.25
    #: Raw label changes per hour above which the capture stream is flagged as
    #: implausibly flickery. Real sleepers turn far less often than this.
    max_plausible_raw_transitions_per_h: float = 60.0

    # --- feature scaling (mirrors the risk model's normalisation) -----------
    bout_cap_h: float = 4.0
    transitions_cap_per_h: float = 12.0

    label_map: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LABEL_MAP))

    def canonical(self, raw_label: str) -> Optional[str]:
        """Return the canonical posture for a raw label, or ``None`` if unknown."""
        if raw_label is None:
            return Posture.ABSENT
        return self.label_map.get(str(raw_label).strip().lower())
