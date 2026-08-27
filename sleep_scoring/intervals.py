"""Turn a stream of instantaneous labels into a posture timeline.

The governing rule, stated by the data owner: a row's posture is held until
the next row arrives. A row at 05:06:10 followed by one at 05:10:13 means the
sleeper held that posture for the whole 4m03s -- an unbroken stretch -- and if
the next row repeats the posture the stretch simply continues.

Applied literally to this capture format that rule is not enough, because the
raw stream contains two artefacts that a naive reading turns into nonsense:

1. Several rows share one timestamp, often disagreeing about the posture.
2. The logger alternates a detection row with a "no person" row at ~1 Hz, so
   the label changes on essentially every row.

Left alone, artefact 2 alone produces ~500 posture changes an hour. Nobody
turns over eight times a minute for seven hours. The stages below remove both
artefacts before any duration is attributed.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence

from .config import Config, Posture
from .loading import Sample


@dataclass
class Interval:
    """A contiguous stretch during which one posture was held."""

    start: datetime
    end: datetime
    posture: str
    #: Number of raw samples that fell inside this interval.
    support: int = 1
    #: True when the stretch was reconstructed across a detector dropout
    #: rather than being directly observed.
    bridged: bool = False
    #: Seconds within this interval carried by a single sample across a long
    #: gap, i.e. inferred from the hold rule rather than repeatedly observed.
    inferred_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return (self.end - self.start).total_seconds()

    @property
    def duration_h(self) -> float:
        return self.duration_s / 3600.0


@dataclass
class Timeline:
    intervals: List[Interval] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    #: Diagnostics captured before de-noising, for comparison.
    raw_transitions: int = 0
    raw_transitions_per_h: float = 0.0
    raw_interval_count: int = 0
    duplicate_timestamps: int = 0
    conflicting_timestamps: int = 0
    dropped_zero_length: int = 0
    bridged_absences: int = 0
    bridged_absence_s: float = 0.0
    absorbed_runs: int = 0
    absorbed_run_s: float = 0.0
    capped_hold_s: float = 0.0

    @property
    def span_s(self) -> float:
        if not self.intervals:
            return 0.0
        return (self.intervals[-1].end - self.intervals[0].start).total_seconds()

    def total_s(self, postures: Sequence[str]) -> float:
        return sum(i.duration_s for i in self.intervals if i.posture in postures)

    def note(self, text: str) -> None:
        self.notes.append(text)


def _collapse_duplicate_timestamps(samples: List[Sample], timeline: Timeline) -> List[Sample]:
    """Reduce every timestamp to a single winning sample.

    Two rows cannot describe the same instant. Where they disagree, a row with
    a real posture beats a "no person" row -- a detection is positive evidence
    that somebody was there, while a non-detection is merely absence of it --
    and among competing postures the one with the clearest landmarks wins.
    """
    if not samples:
        return []

    winners: List[Sample] = []
    bucket: List[Sample] = [samples[0]]

    def resolve(group: List[Sample]) -> Sample:
        if len(group) == 1:
            return group[0]
        timeline.duplicate_timestamps += len(group) - 1
        detected = [s for s in group if s.posture in Posture.BODY]
        if detected and len(detected) != len(group):
            timeline.conflicting_timestamps += 1
            group = detected
        elif len({s.posture for s in group}) > 1:
            timeline.conflicting_timestamps += 1
        return max(group, key=lambda s: (s.mean_visibility, s.row_number))

    for sample in samples[1:]:
        if sample.timestamp == bucket[0].timestamp:
            bucket.append(sample)
        else:
            winners.append(resolve(bucket))
            bucket = [sample]
    winners.append(resolve(bucket))
    return winners


def _raw_intervals(samples: List[Sample], config: Config, timeline: Timeline) -> List[Interval]:
    """Apply the hold-until-next-row rule."""
    intervals: List[Interval] = []
    for current, following in zip(samples, samples[1:]):
        span = (following.timestamp - current.timestamp).total_seconds()
        if span <= 0:
            timeline.dropped_zero_length += 1
            continue
        if config.max_hold_s is not None and span > config.max_hold_s:
            timeline.capped_hold_s += span - config.max_hold_s
            intervals.append(Interval(
                start=current.timestamp,
                end=current.timestamp + timedelta(seconds=config.max_hold_s),
                posture=current.posture,
            ))
            intervals.append(Interval(
                start=current.timestamp + timedelta(seconds=config.max_hold_s),
                end=following.timestamp,
                posture=Posture.UNMONITORED,
            ))
        else:
            intervals.append(Interval(
                start=current.timestamp, end=following.timestamp, posture=current.posture,
                inferred_s=span if span > config.evidence_gap_s else 0.0,
            ))

    if samples and config.tail_hold_s > 0:
        last = samples[-1]
        intervals.append(Interval(
            start=last.timestamp,
            end=last.timestamp + timedelta(seconds=config.tail_hold_s),
            posture=last.posture,
        ))
    return intervals


def _merge_adjacent(intervals: List[Interval]) -> List[Interval]:
    """Fuse neighbouring intervals that share a posture."""
    merged: List[Interval] = []
    for interval in intervals:
        if merged and merged[-1].posture == interval.posture and merged[-1].end == interval.start:
            merged[-1].end = interval.end
            merged[-1].support += interval.support
            merged[-1].bridged = merged[-1].bridged or interval.bridged
            merged[-1].inferred_s += interval.inferred_s
        else:
            merged.append(Interval(
                start=interval.start, end=interval.end, posture=interval.posture,
                support=interval.support, bridged=interval.bridged,
                inferred_s=interval.inferred_s,
            ))
    return merged


def _bridge_short_absences(
    intervals: List[Interval], config: Config, timeline: Timeline
) -> List[Interval]:
    """Fill brief non-detections with the posture that surrounds them.

    A pose detector that loses the sleeper for a second has not witnessed them
    leaving the bed. Only a sustained absence means the bed is actually empty.
    """
    if not intervals:
        return []

    out: List[Interval] = []
    for index, interval in enumerate(intervals):
        if (
            interval.posture != Posture.ABSENT
            or interval.duration_s >= config.max_bridge_absence_s
        ):
            out.append(interval)
            continue

        previous = next(
            (i for i in reversed(out) if i.posture in Posture.BODY), None
        )
        following = next(
            (i for i in intervals[index + 1:] if i.posture in Posture.BODY), None
        )
        # Only bridge when there is real posture on at least one side; a short
        # absence at the very start or end of the file is left as absence.
        carrier = previous or following
        if carrier is None:
            out.append(interval)
            continue

        timeline.bridged_absences += 1
        timeline.bridged_absence_s += interval.duration_s
        out.append(Interval(
            start=interval.start, end=interval.end, posture=carrier.posture,
            support=0, bridged=True, inferred_s=interval.inferred_s,
        ))
    return _merge_adjacent(out)


def _absorb_short_runs(
    intervals: List[Interval], config: Config, timeline: Timeline
) -> List[Interval]:
    """Remove posture runs too brief to be a real turn.

    Repeatedly takes the shortest sub-threshold body-posture run and folds it
    into whichever *body-posture* neighbour is longer, until every remaining run
    clears ``min_dwell_s``. Working shortest-first keeps the most confident runs
    intact and stops a single blip from deciding the shape of the night.

    A short run with no body-posture neighbour is left exactly as it is. Folding
    it into the surrounding non-detection would convert a real observation --
    the camera did see the sleeper -- into "nobody there", which is the one
    direction this stage must never move data.
    """
    if config.min_dwell_s <= 0:
        return intervals

    working = _merge_adjacent(intervals)
    unabsorbable = set()
    while len(working) > 1:
        candidates = [
            index for index, interval in enumerate(working)
            if interval.posture in Posture.BODY
            and interval.duration_s < config.min_dwell_s
            and (interval.start, interval.posture) not in unabsorbable
        ]
        if not candidates:
            break

        index = min(candidates, key=lambda i: working[i].duration_s)
        target = working[index]
        left = working[index - 1] if index > 0 else None
        right = working[index + 1] if index + 1 < len(working) else None

        body_options = [
            i for i in (left, right) if i is not None and i.posture in Posture.BODY
        ]
        if not body_options:
            unabsorbable.add((target.start, target.posture))
            continue

        chosen = max(body_options, key=lambda i: i.duration_s)
        timeline.absorbed_runs += 1
        timeline.absorbed_run_s += target.duration_s
        target.posture = chosen.posture
        target.bridged = True
        working = _merge_adjacent(working)

    return working


def build_timeline(samples: List[Sample], config: Optional[Config] = None) -> Timeline:
    """Full cleaning pipeline: samples in, posture timeline out."""
    config = config or Config()
    timeline = Timeline()
    if not samples:
        timeline.note("No samples to build a timeline from.")
        return timeline

    ordered = sorted(samples, key=lambda s: (s.timestamp, s.row_number))

    # Diagnostics on the untouched stream, so the report can show how much
    # of the apparent movement was real.
    raw_changes = sum(
        1 for a, b in zip(ordered, ordered[1:]) if a.posture != b.posture
    )
    span_h = (ordered[-1].timestamp - ordered[0].timestamp).total_seconds() / 3600.0
    timeline.raw_transitions = raw_changes
    timeline.raw_transitions_per_h = raw_changes / span_h if span_h > 0 else 0.0

    deduped = (
        _collapse_duplicate_timestamps(ordered, timeline)
        if config.collapse_duplicates else ordered
    )
    raw = _raw_intervals(deduped, config, timeline)
    timeline.raw_interval_count = len(_merge_adjacent(raw))

    bridged = _bridge_short_absences(_merge_adjacent(raw), config, timeline)
    timeline.intervals = _absorb_short_runs(bridged, config, timeline)

    if timeline.raw_transitions_per_h > config.max_plausible_raw_transitions_per_h:
        timeline.note(
            f"Raw label stream changes posture {timeline.raw_transitions_per_h:.0f} "
            f"times per hour, far above the {config.max_plausible_raw_transitions_per_h:.0f}/h "
            "plausibility limit. The capture stream is flickering, not the sleeper."
        )
    if timeline.conflicting_timestamps:
        timeline.note(
            f"{timeline.conflicting_timestamps} timestamp(s) carried disagreeing "
            "Position values; resolved in favour of the detected posture with the "
            "clearest landmarks."
        )
    return timeline
