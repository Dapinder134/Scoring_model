"""Read a capture CSV and report everything wrong with it.

Loading deliberately never raises on bad rows: a night of sleep data is worth
scoring even when parts of it are unusable, so problems are collected into
:class:`LoadResult.issues` and the caller decides what to do with them.
"""

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from .config import Config, Posture

TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S")

LANDMARKS = (
    "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
)
REQUIRED_COLUMNS = ("Timestamp", "Session_Name", "Position")


@dataclass
class Sample:
    """One logged row, after parsing."""

    timestamp: datetime
    session: str
    raw_position: str
    posture: str
    visibility: Dict[str, float] = field(default_factory=dict)
    row_number: int = 0

    @property
    def mean_visibility(self) -> float:
        vals = [v for v in self.visibility.values() if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def has_landmarks(self) -> bool:
        return bool(self.visibility)


@dataclass
class LoadResult:
    samples: List[Sample]
    issues: List[str] = field(default_factory=list)
    raw_row_count: int = 0
    skipped_rows: int = 0
    unknown_labels: Dict[str, int] = field(default_factory=dict)
    sessions: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    path: str = ""

    def issue(self, text: str) -> None:
        self.issues.append(text)


def _parse_timestamp(text: str) -> Optional[datetime]:
    text = (text or "").strip()
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _parse_float(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def load_csv(path: str, config: Optional[Config] = None) -> LoadResult:
    """Parse a capture CSV into ordered :class:`Sample` objects."""
    config = config or Config()
    result = LoadResult(samples=[], path=path)

    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        result.columns = list(reader.fieldnames or [])

        missing = [c for c in REQUIRED_COLUMNS if c not in result.columns]
        if missing:
            result.issue(f"CSV is missing required column(s): {', '.join(missing)}")
            return result

        vis_columns = [f"{lm}_Visibility" for lm in LANDMARKS
                       if f"{lm}_Visibility" in result.columns]
        if len(vis_columns) < len(LANDMARKS):
            absent = [lm for lm in LANDMARKS if f"{lm}_Visibility" not in result.columns]
            result.issue(f"No visibility column for landmark(s): {', '.join(absent)}")

        sessions = set()
        for row_number, row in enumerate(reader, start=2):
            result.raw_row_count += 1
            stamp = _parse_timestamp(row.get("Timestamp", ""))
            if stamp is None:
                result.skipped_rows += 1
                if result.skipped_rows <= 5:
                    result.issue(
                        f"row {row_number}: unparseable Timestamp "
                        f"{row.get('Timestamp', '')!r} - row dropped"
                    )
                continue

            raw_position = (row.get("Position") or "").strip()
            posture = config.canonical(raw_position)
            if posture is None:
                result.unknown_labels[raw_position] = (
                    result.unknown_labels.get(raw_position, 0) + 1
                )
                posture = Posture.ABSENT

            visibility = {}
            for lm in LANDMARKS:
                col = f"{lm}_Visibility"
                if col not in result.columns:
                    continue
                value = _parse_float(row.get(col, ""))
                if not math.isnan(value):
                    visibility[lm] = value

            sessions.add((row.get("Session_Name") or "").strip())
            result.samples.append(
                Sample(
                    timestamp=stamp,
                    session=(row.get("Session_Name") or "").strip(),
                    raw_position=raw_position,
                    posture=posture,
                    visibility=visibility,
                    row_number=row_number,
                )
            )

    result.sessions = sorted(s for s in sessions if s)
    _audit(result, config)
    return result


def load_log(path: str, config: Optional[Config] = None) -> LoadResult:
    """Parse the plain-text ``YYYY-MM-DD HH:MM:SS, <label>`` log format.

    Same output as :func:`load_csv`, minus landmark visibility, so both capture
    formats reach the timeline builder as one kind of thing.
    """
    config = config or Config()
    result = LoadResult(samples=[], path=path)
    result.columns = ["Timestamp", "Position"]
    sessions = set()

    with open(path, "r", encoding="utf-8-sig") as handle:
        for row_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            stamp_text, _, raw_position = line.partition(",")
            stamp = _parse_timestamp(stamp_text)
            if stamp is None:
                result.raw_row_count += 1
                result.skipped_rows += 1
                if result.skipped_rows <= 5:
                    result.issue(f"line {row_number}: unparseable record {line!r} - dropped")
                continue

            result.raw_row_count += 1
            raw_position = raw_position.strip()
            posture = config.canonical(raw_position)
            if posture is None:
                result.unknown_labels[raw_position] = (
                    result.unknown_labels.get(raw_position, 0) + 1
                )
                posture = Posture.ABSENT

            result.samples.append(Sample(
                timestamp=stamp, session="", raw_position=raw_position,
                posture=posture, visibility={}, row_number=row_number,
            ))

    result.sessions = sorted(s for s in sessions if s)
    _audit(result, config)
    return result


def load_any(path: str, config: Optional[Config] = None) -> LoadResult:
    """Dispatch on file extension: ``.csv`` uses the capture CSV reader."""
    if path.lower().endswith(".csv"):
        return load_csv(path, config)
    return load_log(path, config)


def _audit(result: LoadResult, config: Config) -> None:
    """Attach ordering / vocabulary / consistency findings to ``result``."""
    samples = result.samples
    if not samples:
        result.issue("No usable rows found.")
        return

    if result.unknown_labels:
        pretty = ", ".join(
            f"{label!r} x{count}" for label, count in sorted(result.unknown_labels.items())
        )
        result.issue(
            f"Unrecognised Position label(s) treated as 'no person': {pretty}. "
            "Add them to Config.label_map if they are real postures."
        )

    out_of_order = sum(
        1 for a, b in zip(samples, samples[1:]) if b.timestamp < a.timestamp
    )
    if out_of_order:
        result.issue(f"{out_of_order} row(s) go backwards in time; re-sorting by timestamp.")
        samples.sort(key=lambda s: (s.timestamp, s.row_number))

    if len(result.sessions) > 1:
        result.issue(
            f"File mixes {len(result.sessions)} session names ({', '.join(result.sessions)}); "
            "scores assume a single night."
        )

    # Postures that never appear at all are worth naming: the risk model has
    # terms that can only ever fire on labels the capture app is able to emit.
    seen = {s.posture for s in samples}
    for posture in (Posture.PRONE,):
        if posture not in seen:
            result.issue(
                f"No '{posture}' rows in this file - the capture app never emitted a "
                "front/stomach label, so every prone-weighted risk term stays at zero."
            )

    detected = [s for s in samples if s.posture in Posture.BODY]
    if detected:
        weak = {}
        for lm in LANDMARKS:
            vals = [s.visibility[lm] for s in detected if lm in s.visibility]
            if vals and sum(vals) / len(vals) < 0.5:
                weak[lm] = sum(vals) / len(vals)
        if weak:
            pretty = ", ".join(f"{k}={v:.2f}" for k, v in sorted(weak.items()))
            result.issue(
                f"Mean landmark visibility below 0.5 on detected rows ({pretty}); "
                "postures derived from these joints are unreliable."
            )

    mislabelled = [
        s for s in samples if s.posture in Posture.BODY and not s.has_landmarks
    ]
    if mislabelled and any(s.has_landmarks for s in samples):
        result.issue(
            f"{len(mislabelled)} row(s) carry a posture label but no landmark data at all."
        )
