"""
Sleep posture statistics engine.

Reads a position log, splits it into sessions, removes classifier flicker,
applies the morning X reassignment, computes every per-night statistic, checks
the data quality constraints, and hands the result to the scoring model.

Two capture formats are accepted:

    the capture CSV     Timestamp,Session_Name,Position,<landmark columns>
    the plain-text log  YYYY-MM-DD HH:MM:SS, <label>

Parsing and timeline construction are delegated to the ``sleep_scoring``
package, which owns the awkward parts of the real capture format: rows sharing
a timestamp, detector dropout, and the sample-and-hold rule. This module owns
everything from the timeline upwards.

Run it:
    python sleep_statistics.py Night_001.csv
    python sleep_statistics.py sleep_positions_1.txt --session 4
    python sleep_statistics.py Night_001.csv --session 1 --reassign 5=1 11=2
"""

import argparse
import collections
from dataclasses import dataclass
from datetime import datetime, timedelta

import scoring_model
from sleep_scoring import Posture, build_timeline, load_any
from sleep_scoring import Config as CaptureConfig

ABSENT = "Absent"
UNCLASSIFIED = "No Person Detected"
POSTURES = ["Back", "Front", "Left Side", "Right Side"]

# X codes used in the morning review.
X_LABELS = {0: ABSENT, 1: "Back", 2: "Front", 3: "Left Side", 4: "Right Side"}

# The timeline speaks canonical postures; this module speaks capture labels.
CANONICAL_TO_LABEL = {
    Posture.SUPINE: "Back",
    Posture.PRONE: "Front",
    Posture.LEFT: "Left Side",
    Posture.RIGHT: "Right Side",
    Posture.ABSENT: UNCLASSIFIED,
    Posture.UNMONITORED: UNCLASSIFIED,
}


@dataclass
class Config:
    """Every constant the pipeline uses. Load from an external file in production."""

    # FIX 1 - was 600 s. A ten-minute gap is not a new night, and the capture
    # files routinely go quiet for an hour while the sleeper is still in bed.
    # At 600 s one night split into ten "sessions", and because time between
    # sessions belongs to no session at all, 76.5% of the recording was
    # silently discarded. Four hours separates genuinely different recordings
    # without cutting a single night apart.
    session_gap_s: int = 14400

    debounce_s: float = 30.0          # runs shorter than this are classifier noise
    final_hold_s: float = 0.0         # time credited to a session's last record
    nominal_interval_s: float = 210.0 # intended capture interval, drives resolution limits

    # FIX 2 - non-detections shorter than this are detector dropout, not the
    # sleeper leaving the bed, and are refilled with the surrounding posture.
    bridge_absence_s: float = 120.0
    # Cap on how long one record's posture may be held. None applies the
    # sample-and-hold rule literally, with no cap.
    max_hold_s: float = None
    # A stretch longer than this rests on a single record rather than on
    # continuous observation, and is reported as inferred.
    evidence_gap_s: float = 120.0

    min_scored_hours: float = 4.0
    max_absent_pct: float = 20.0
    max_trans_per_hour: float = 30.0
    # FIX 3 - was 4. Requiring all four postures in one night fails every
    # sleeper who never lies on their front, which is most of them. Two
    # distinct positions is the point below which a night carries no
    # positional information at all.
    min_distinct_positions: int = 2

    def max_detectable_tph(self):
        return 3600.0 / self.nominal_interval_s if self.nominal_interval_s else float("inf")

    def bout_resolution_min(self):
        return self.nominal_interval_s / 60.0

    def debounce_is_meaningful(self):
        """
        FIX 4 - the comparison was the wrong way round. Under sample-and-hold at
        a nominal 210 s interval the shortest run that can exist is 210 s, so a
        5 s debounce window can never absorb anything - yet the original
        reported that as "meaningful: yes" and a 300 s window, which does fire,
        as no. Debounce only does work when its window exceeds the interval.
        """
        return self.debounce_s > self.nominal_interval_s

    def capture_config(self):
        """Translate into the back end's settings."""
        return CaptureConfig(
            min_dwell_s=self.debounce_s,
            max_bridge_absence_s=self.bridge_absence_s,
            max_hold_s=self.max_hold_s,
            tail_hold_s=self.final_hold_s,
            evidence_gap_s=self.evidence_gap_s,
        )


@dataclass
class Run:
    session: int
    index: int
    raw_label: str
    start: datetime
    duration: float
    x: int = 0
    inferred_s: float = 0.0

    @property
    def effective_label(self):
        if self.raw_label != UNCLASSIFIED:
            return self.raw_label
        return X_LABELS.get(self.x, ABSENT)

    @property
    def is_posture(self):
        return self.effective_label != ABSENT


# ----------------------------------------------------------------- parsing

def load_log(path, cfg=None):
    """Read either capture format and return (records, LoadResult).

    ``records`` is the list of (timestamp, label) pairs the rest of this module
    used to parse for itself. The audit findings ride along on the LoadResult.
    """
    capture = (cfg or Config()).capture_config()
    result = load_any(path, capture)
    rows = [(s.timestamp, s.raw_position) for s in result.samples]
    rows.sort(key=lambda r: r[0])
    return rows, result


def split_sessions(intervals, gap_s):
    """Split the timeline where the sleeper was away long enough to end a night.

    Returns (sessions, between_s). A boundary gap belongs to no session, so its
    duration is returned separately rather than being dropped on the floor.
    """
    if not intervals:
        return [], 0.0
    sessions, current, between = [], [], 0.0
    for interval in intervals:
        boundary = (
            interval.posture in Posture.NON_BODY and interval.duration_s > gap_s
        )
        if boundary:
            if current:
                sessions.append(current)
                current = []
            between += interval.duration_s
        else:
            current.append(interval)
    if current:
        sessions.append(current)
    return sessions, between


def make_run_objects(session_no, session_intervals):
    """One Run per timeline interval, numbered for the morning review."""
    out = []
    for index, interval in enumerate(session_intervals, start=1):
        out.append(Run(
            session=session_no,
            index=index,
            raw_label=CANONICAL_TO_LABEL.get(interval.posture, UNCLASSIFIED),
            start=interval.start,
            duration=interval.duration_s,
            inferred_s=interval.inferred_s,
        ))
    return out


def build_sessions(path, cfg):
    """Full front-to-back read: file in, {session number: [Run]} out."""
    rows, load_result = load_log(path, cfg)
    if not rows:
        return {}, load_result, None, 0.0
    timeline = build_timeline(load_result.samples, cfg.capture_config())
    sessions, between_s = split_sessions(timeline.intervals, cfg.session_gap_s)
    all_runs = {i: make_run_objects(i, s) for i, s in enumerate(sessions, start=1)}
    return all_runs, load_result, timeline, between_s


# ----------------------------------------------------------------- statistics

def compute_statistics(runs):
    """
    Every per-night figure, computed after reassignment.

    Bout grouping is the subtle part. Consecutive runs sharing an effective
    label merge into one bout. A run left at X = 0 (Absent) breaks the chain,
    so two Back runs separated by an absence stay separate - the sleeper left
    the bed, so it is not unbroken time lying there.
    """
    total_s = sum(r.duration for r in runs)
    by_label = collections.defaultdict(float)
    for r in runs:
        by_label[r.effective_label] += r.duration

    absent_s = by_label.get(ABSENT, 0.0)
    scored_s = total_s - absent_s
    scored_h = scored_s / 3600.0

    # bouts
    bouts, current = [], None
    for r in runs:
        if not r.is_posture:
            current = None
            continue
        if current is not None and current["label"] == r.effective_label:
            current["duration"] += r.duration
        else:
            current = {"label": r.effective_label, "duration": r.duration,
                       "start": r.start}
            bouts.append(current)

    # transitions: a bout whose label differs from the previous bout's label
    transitions = sum(1 for a, b in zip(bouts, bouts[1:]) if a["label"] != b["label"])

    posture_runs = [r for r in runs if r.is_posture]
    longest_bout_s = max((b["duration"] for b in bouts), default=0.0)
    posture_total = sum(by_label.get(p, 0.0) for p in POSTURES)
    inferred_s = sum(r.inferred_s for r in posture_runs)

    pct = {p: (100.0 * by_label.get(p, 0.0) / scored_s if scored_s else 0.0) for p in POSTURES}

    return {
        "total_s": total_s,
        "absent_s": absent_s,
        "scored_s": scored_s,
        "scored_h": scored_h,
        "absent_pct_of_total": 100.0 * absent_s / total_s if total_s else 0.0,
        "seconds": {p: by_label.get(p, 0.0) for p in POSTURES},
        "pct": pct,
        "n_runs": len(runs),
        "n_posture_runs": len(posture_runs),
        "n_bouts": len(bouts),
        "transitions": transitions,
        "trans_per_hour": transitions / scored_h if scored_h else 0.0,
        "longest_bout_s": longest_bout_s,
        "longest_bout_h": longest_bout_s / 3600.0,
        "mean_bout_s": posture_total / len(bouts) if bouts else 0.0,
        "distinct_positions": sum(1 for p in POSTURES if by_label.get(p, 0.0) > 0),
        "bouts": bouts,
        "inferred_s": inferred_s,
        "inferred_pct": 100.0 * inferred_s / scored_s if scored_s else 0.0,
        "sum_check": posture_total + absent_s - total_s,
        "pct_check": sum(pct.values()),
    }


def check_constraints(stats, cfg):
    """Every gate, with the actual value beside its threshold."""
    c = []
    c.append(("Scored length at least the minimum", stats["scored_h"], cfg.min_scored_hours,
              stats["scored_h"] >= cfg.min_scored_hours, "hours"))
    c.append(("Unresolved absent time within limit", stats["absent_pct_of_total"], cfg.max_absent_pct,
              stats["absent_pct_of_total"] <= cfg.max_absent_pct, "%"))
    c.append(("Transition rate physically plausible", stats["trans_per_hour"], cfg.max_trans_per_hour,
              stats["trans_per_hour"] <= cfg.max_trans_per_hour, "per hour"))
    c.append(("Transition rate within capture resolution", stats["trans_per_hour"],
              cfg.max_detectable_tph(), stats["trans_per_hour"] <= cfg.max_detectable_tph(),
              "per hour"))
    c.append(("At least the minimum distinct positions", stats["distinct_positions"],
              cfg.min_distinct_positions, stats["distinct_positions"] >= cfg.min_distinct_positions,
              "count"))
    c.append(("Components sum to total time", stats["sum_check"], 0.0,
              abs(stats["sum_check"]) < 0.5, "seconds"))
    c.append(("Percentages sum to 100", stats["pct_check"], 100.0,
              abs(stats["pct_check"] - 100.0) < 0.5 or stats["scored_s"] == 0, "%"))
    c.append(("Longest bout not longer than session", stats["longest_bout_h"], stats["scored_h"],
              stats["longest_bout_h"] <= stats["scored_h"] + 1e-6, "hours"))
    return c


# ----------------------------------------------------------------- reporting

def hms(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h else f"{m:d}m {s:02d}s"


def rule(char="-", width=78):
    print(char * width)


def print_capture_audit(load_result, timeline, between_s):
    """What the raw stream looked like before any of it was cleaned up."""
    rule("=")
    print("CAPTURE AUDIT")
    rule("=")
    print(f"   {'Records read':<38}{load_result.raw_row_count:>10}")
    print(f"   {'Records unparseable':<38}{load_result.skipped_rows:>10}")
    if timeline is not None:
        print(f"   {'Raw label changes':<38}{timeline.raw_transitions:>10}")
        print(f"   {'Raw label changes per hour':<38}{timeline.raw_transitions_per_h:>10.1f}")
        print(f"   {'Rows sharing a timestamp':<38}{timeline.duplicate_timestamps:>10}")
        print(f"   {'Timestamps with disagreeing labels':<38}{timeline.conflicting_timestamps:>10}")
        print(f"   {'Dropouts bridged':<38}{timeline.bridged_absences:>10}"
              f"   ({hms(timeline.bridged_absence_s)})")
        print(f"   {'Short runs absorbed':<38}{timeline.absorbed_runs:>10}"
              f"   ({hms(timeline.absorbed_run_s)})")
        print(f"   {'Intervals after cleaning':<38}{len(timeline.intervals):>10}")
    if between_s:
        print(f"   {'Time between sessions (not scored)':<38}{hms(between_s):>10}")
    for issue in load_result.issues + (timeline.notes if timeline else []):
        print(f"   NOTE: {issue}")
    print()


def print_session_overview(all_runs, cfg):
    rule("=")
    print("SESSION OVERVIEW")
    rule("=")
    print(f"{'#':>3} {'start':19} {'duration':>11} {'runs':>6} {'trans/h':>9} "
          f"{'longest':>10} {'absent%':>9} {'positions':>10}")
    rule()
    for sess_no, runs in sorted(all_runs.items()):
        s = compute_statistics(runs)
        print(f"{sess_no:>3} {str(runs[0].start):19} {hms(s['total_s']):>11} "
              f"{s['n_runs']:>6} {s['trans_per_hour']:>9.1f} "
              f"{hms(s['longest_bout_s']):>10} {s['absent_pct_of_total']:>8.1f}% "
              f"{s['distinct_positions']:>10}")
    print()


def print_statistics(stats, sess_no, cfg):
    rule("=")
    print(f"NIGHT STATISTICS  -  session {sess_no}")
    rule("=")

    print("\nA. TIME BUDGET")
    print(f"   {'Total recorded':<34}{hms(stats['total_s']):>14}")
    print(f"   {'Not in bed (X = 0)':<34}{hms(stats['absent_s']):>14}")
    print(f"   {'Scored time (the denominator)':<34}{hms(stats['scored_s']):>14}"
          f"   = {stats['scored_h']:.3f} h")
    for p in POSTURES:
        print(f"   {p:<34}{hms(stats['seconds'][p]):>14}")
    print(f"   {'check: components - total':<34}{stats['sum_check']:>13.1f}s")

    print("\nB. POSITION PERCENTAGES  (of scored time)")
    for p in POSTURES:
        pctv = stats["pct"][p]
        bar = "#" * int(round(pctv / 2))
        print(f"   {p:<16}{pctv:>6.1f}%  {bar}")
    print(f"   {'check: sum':<16}{stats['pct_check']:>6.1f}%")
    print(f"   {'absent share':<16}{stats['absent_pct_of_total']:>6.1f}%  (of total recorded)")

    print("\nC. MOVEMENT")
    print(f"   {'Runs after debounce':<34}{stats['n_runs']:>10}")
    print(f"   {'Posture runs':<34}{stats['n_posture_runs']:>10}")
    print(f"   {'Bouts (merged runs)':<34}{stats['n_bouts']:>10}")
    print(f"   {'Position transitions':<34}{stats['transitions']:>10}")
    print(f"   {'Transitions per hour':<34}{stats['trans_per_hour']:>10.1f}")
    print(f"   {'Longest unbroken bout':<34}{hms(stats['longest_bout_s']):>10}"
          f"   = {stats['longest_bout_h']:.3f} h")
    print(f"   {'Mean bout length':<34}{hms(stats['mean_bout_s']):>10}")
    print(f"   {'Distinct positions observed':<34}{stats['distinct_positions']:>10} of 4")

    print("\nD. EVIDENCE QUALITY")
    print(f"   {'Scored time held from one record':<34}{hms(stats['inferred_s']):>10}"
          f"   = {stats['inferred_pct']:.1f}% of scored time")
    if stats["inferred_pct"] > 50:
        print("   WARNING: most of this night is inferred from isolated records rather")
        print("            than observed continuously. Treat the bout figures as soft,")
        print("            and note that a single record before a long gap decides the")
        print("            whole gap, so contested timestamps carry unusual weight.")

    print("\nE. NORMALISED INPUTS  (0 to 1, fed to the scoring model)")
    x = {
        "x_sup":   stats["pct"]["Back"] / 100.0,
        "x_pro":   stats["pct"]["Front"] / 100.0,
        "x_left":  stats["pct"]["Left Side"] / 100.0,
        "x_right": stats["pct"]["Right Side"] / 100.0,
        "x_bout":  min(1.0, stats["longest_bout_h"] / scoring_model.BOUT_CAP_HOURS),
        "x_trans": min(1.0, stats["trans_per_hour"] / scoring_model.TRANS_CAP_PER_HOUR),
    }
    for k, v in x.items():
        print(f"   {k:<12}{v:>8.3f}")
    print()


def print_constraints(constraints, cfg):
    rule("=")
    print("DATA QUALITY CONSTRAINTS")
    rule("=")
    print(f"   {'constraint':<42}{'actual':>10}{'threshold':>12}  result")
    rule()
    for name, actual, threshold, ok, unit in constraints:
        print(f"   {name:<42}{actual:>10.2f}{threshold:>12.2f}  "
              f"{'PASS' if ok else 'FAIL'}")
    failed = [c for c in constraints if not c[3]]
    print()
    if failed:
        print(f"   OVERALL GATE: FAIL  ({len(failed)} of {len(constraints)} constraints)")
        for name, *_ in failed:
            print(f"      - {name}")
    else:
        print("   OVERALL GATE: PASS")
    print()
    print(f"   Capture model: nominal interval {cfg.nominal_interval_s:.0f}s, so at most "
          f"{cfg.max_detectable_tph():.1f} transitions/hour are")
    print(f"   detectable and bout lengths carry +/- {cfg.bout_resolution_min():.1f} minutes of error. "
          f"Debounce meaningful: "
          f"{'yes' if cfg.debounce_is_meaningful() else 'no'}.")
    print()
    return not failed


def print_scores(stats, gate_passed, prone_detectable):
    rule("=")
    print("RISK SCORES")
    rule("=")
    if not gate_passed:
        print("\n   SUPPRESSED - the constraints above failed, so any score would be")
        print("   describing a recording that cannot represent a night's sleep.\n")
        return

    try:
        results = scoring_model.calculate_sleep_risks(
            stats["pct"]["Back"], stats["pct"]["Front"],
            stats["pct"]["Left Side"], stats["pct"]["Right Side"],
            stats["longest_bout_h"], stats["trans_per_hour"],
            prone_detectable=prone_detectable,
        )
    except scoring_model.InvalidPostureInput as exc:
        print(f"\n   REJECTED BY SCORING MODEL: {exc}\n")
        return

    results.sort(key=lambda r: r["score"], reverse=True)
    print(f"\n   {'condition':<38}{'score':>8}{'band':>11}{'cap':>7}")
    rule()
    for r in results:
        cap = "-" if r["capped_at"] >= 100 else f"{r['capped_at']:.0f}"
        print(f"   {r['condition']:<38}{r['score']:>8.1f}{r['band']:>11}{cap:>7}")

    top = results[0]
    runner_up = results[1]["score"] if len(results) > 1 else 0.0
    print()
    print(f"   Leading concern: {top['condition']} at {top['score']:.1f} "
          f"({top['score'] - runner_up:+.1f} clear of next)")
    print(f"   {top['summary']}")
    print()


# ----------------------------------------------------------------- entry point

def parse_reassign(pairs):
    out = {}
    for item in pairs or []:
        run_idx, _, x = item.partition("=")
        out[int(run_idx)] = int(x)
    return out


def main():
    ap = argparse.ArgumentParser(description="Sleep posture statistics")
    ap.add_argument("logfile", help="capture CSV or plain-text position log")
    ap.add_argument("--session", type=int, default=None,
                    help="session number to analyse in detail (default: the one with "
                         "the most scored posture time)")
    ap.add_argument("--reassign", nargs="*", metavar="RUN=X",
                    help="morning review, e.g. --reassign 5=1 11=2 (0 absent, 1 back, "
                         "2 front, 3 left, 4 right)")
    ap.add_argument("--debounce", type=float, default=None)
    ap.add_argument("--final-hold", type=float, default=None)
    ap.add_argument("--session-gap", type=float, default=None,
                    help="absence long enough to end a night, in seconds")
    ap.add_argument("--bridge-absence", type=float, default=None,
                    help="non-detections shorter than this are detector dropout")
    ap.add_argument("--max-hold", type=float, default=None,
                    help="cap on how long one record's posture may be held")
    args = ap.parse_args()

    cfg = Config()
    if args.debounce is not None:
        cfg.debounce_s = args.debounce
    if args.final_hold is not None:
        cfg.final_hold_s = args.final_hold
    if args.session_gap is not None:
        cfg.session_gap_s = args.session_gap
    if args.bridge_absence is not None:
        cfg.bridge_absence_s = args.bridge_absence
    if args.max_hold is not None:
        cfg.max_hold_s = args.max_hold

    rows, load_result = load_log(args.logfile, cfg)
    if not rows:
        print("No parseable records found.")
        return
    print(f"\nLoaded {len(rows)} records from {args.logfile}"
          + (f" ({load_result.skipped_rows} unparseable lines skipped)"
             if load_result.skipped_rows else ""))
    print(f"Span: {rows[0][0]} to {rows[-1][0]}")
    labels = collections.Counter(l for _, l in rows)
    print("Raw labels: " + ", ".join(f"{k} x{v}" for k, v in labels.most_common()))

    # Whether the classifier can see prone at all is a property of the whole
    # log, not of one session, and it changes how a low score should be read.
    prone_detectable = any(
        s.posture == Posture.PRONE for s in load_result.samples
    )
    missing = [p for p in POSTURES if p not in labels]
    if missing:
        print(f"WARNING: never emitted by the classifier: {', '.join(missing)}")
    print()

    all_runs, load_result, timeline, between_s = build_sessions(args.logfile, cfg)
    if not all_runs:
        print("No sessions could be built from this file.")
        return

    print_capture_audit(load_result, timeline, between_s)
    print_session_overview(all_runs, cfg)

    sess_no = args.session
    if sess_no is None:
        # FIX 5 - the original picked the longest session by total duration,
        # which on a real capture selected a block of pure absence with no
        # posture data in it at all. Pick by scored posture time instead.
        sess_no = max(all_runs, key=lambda k: compute_statistics(all_runs[k])["scored_s"])
        print(f"No --session given; analysing session {sess_no}, "
              f"which holds the most scored posture time.\n")
    if sess_no not in all_runs:
        print(f"Session {sess_no} does not exist. Available: {sorted(all_runs)}")
        return

    runs = all_runs[sess_no]

    reassign = parse_reassign(args.reassign)
    unresolved = [r for r in runs if r.raw_label == UNCLASSIFIED]
    if unresolved:
        print(f"MORNING REVIEW - {len(unresolved)} unclassified segment(s) in session {sess_no}")
        for r in unresolved:
            if r.index in reassign:
                r.x = reassign[r.index]
            flag = "" if r.index in reassign else "   <-- still X=0, excluded"
            print(f"   run {r.index:>3}  {r.start}  {hms(r.duration):>9}  "
                  f"X={r.x} -> {r.effective_label}{flag}")
        print()

    stats = compute_statistics(runs)
    print_statistics(stats, sess_no, cfg)
    gate = print_constraints(check_constraints(stats, cfg), cfg)
    print_scores(stats, gate, prone_detectable)


if __name__ == "__main__":
    main()
