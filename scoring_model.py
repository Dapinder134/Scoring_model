"""
Sleep posture risk scoring model.

Converts the six per-night posture statistics into eight independent
condition scores on a 0-100 scale.

NOTE ON THE WEIGHTS: the coefficients below are heuristic. The mechanisms
they encode are supported by the literature cited in the project proposal
(Joosten 2014, Oksenberg 1997, Cary 2019, Defloor 2005), but the numeric
values themselves are not clinically validated and no calibration dataset
exists for them. Treat them as a documented limitation, not as evidence.
"""

import math

# Band thresholds. Kept as module constants so they can be moved into the
# external config file without touching the logic.
MODERATE_THRESHOLD = 30.0
HIGH_THRESHOLD = 60.0

# Normalisation ceilings.
BOUT_CAP_HOURS = 4.0
TRANS_CAP_PER_HOUR = 12.0

# Order: [w_sup, w_pro, w_left, w_right, w_bout, w_trans]
WEIGHT_MATRIX = {
    "Positional obstructive sleep apnoea": [0.65, 0.00, 0.00, 0.00, 0.35, -0.10],
    "Habitual snoring":                    [0.70, 0.00, 0.00, 0.00, 0.30, -0.05],
    "Nocturnal reflux (GORD)":             [0.45, 0.00, -0.15, 0.55, 0.00, 0.00],
    "Neck strain":                         [0.00, 0.70, 0.00, 0.00, 0.30, -0.05],
    "Lower back strain":                   [0.45, 0.55, 0.00, 0.00, 0.00, -0.05],
    "Shoulder / hip joint loading":        [0.00, 0.00, 0.30, 0.30, 0.40, -0.15],
    "Pressure injury risk":                [0.25, 0.00, 0.00, 0.00, 0.75, -0.30],
    "Sleep fragmentation":                 [0.00, 0.00, 0.00, 0.00, -0.20, 1.00],
}

CONDITION_INFO = {
    "Positional obstructive sleep apnoea": {
        "mechanism": "Lying flat lets the tongue and soft palate fall backwards under gravity, "
                     "narrowing or briefly closing the airway. Breathing pauses, oxygen dips, and "
                     "the sleeper partially wakes to restore airflow.",
        "action": "Encourage side sleeping; raise the head of the bed. Persistent symptoms warrant "
                  "referral to a sleep physician.",
    },
    "Habitual snoring": {
        "mechanism": "The same mechanism, but the airway narrows without fully closing, so relaxed "
                     "soft tissue vibrates in the moving airstream.",
        "action": "Side sleeping; head elevation. Weight and alcohol are known aggravators.",
    },
    "Nocturnal reflux (GORD)": {
        "mechanism": "Right lateral places the gastro-oesophageal junction below the pool of stomach "
                     "contents, so acid tracks upward more easily. Lying flat removes the gravity "
                     "gradient that normally keeps it down.",
        "action": "Favour left lateral; raise the head of the bed; avoid eating close to bedtime.",
    },
    "Neck strain": {
        "mechanism": "Face-down sleeping forces the head to stay rotated near the end of its range "
                     "for hours, loading the cervical facet joints and surrounding soft tissue.",
        "action": "Reduce prone time; if prone is unavoidable, use a very thin pillow or none at all.",
    },
    "Lower back strain": {
        "mechanism": "Prone hyperextends the lumbar spine. Flat supine without knee support flattens "
                     "the natural lumbar curve. Both sustain a loaded end-range posture.",
        "action": "Pillow under the pelvis when prone; pillow under the knees when supine.",
    },
    "Shoulder / hip joint loading": {
        "mechanism": "Side sleeping concentrates body weight onto the dependent shoulder and hip. "
                     "Held long enough this compresses the joint capsule and surrounding tendons.",
        "action": "Alternate sides through the night; pillow between the knees; adequate mattress "
                  "give at the shoulder.",
    },
    "Pressure injury risk": {
        "mechanism": "Sustained pressure over a bony prominence exceeds capillary filling pressure "
                     "and cuts local blood flow. Tissue damage begins well before the sleeper notices.",
        "action": "Reposition at least two-hourly; pressure-redistributing mattress or overlay. "
                  "Chiefly a concern for people with limited mobility.",
    },
    "Sleep fragmentation": {
        "mechanism": "A high transition count suggests the sleeper is not settling into sustained "
                     "sleep. Frequent repositioning is usually a symptom of an underlying disturbance "
                     "rather than a cause in itself.",
        "action": "Treat as a flag for further investigation: sleep environment, pain, caffeine "
                  "timing, or an untreated breathing disorder.",
    },
}


class InvalidPostureInput(ValueError):
    """Raised when the supplied statistics cannot describe a real night."""


def _finite(name, value):
    """
    FIX 4 - NaN defeated every guard in the original _validate. `nan < 0` is False,
    and `abs(nan - 100) > 0.5` is also False, so a NaN percentage passed validation
    untouched. It then reached `min(100.0, nan)`, which Python resolves to 100.0,
    and every condition came back 100.0 / High. A dropped landmark therefore
    produced maximum alarm on all eight conditions - the worst possible direction
    for a missing value to fail in.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise InvalidPostureInput(f"{name} is not a number ({value!r})")
    if math.isnan(value):
        raise InvalidPostureInput(f"{name} is NaN")
    if math.isinf(value):
        raise InvalidPostureInput(f"{name} is infinite")
    return value


def _validate(supine_pct, prone_pct, left_pct, right_pct, bout_hours, trans_per_hour):
    """
    FIX 3 - the original accepted anything. Percentages summing to 360, negative
    bout lengths and 999 transitions per hour all produced confident scores.
    """
    parts = {"supine": supine_pct, "prone": prone_pct,
             "left": left_pct, "right": right_pct}
    for name, v in parts.items():
        v = _finite(f"{name} percentage", v)
        if v < 0:
            raise InvalidPostureInput(f"{name} percentage is negative ({v})")
    bout_hours = _finite("bout_hours", bout_hours)
    trans_per_hour = _finite("trans_per_hour", trans_per_hour)
    if bout_hours < 0:
        raise InvalidPostureInput(f"bout_hours is negative ({bout_hours})")
    if trans_per_hour < 0:
        raise InvalidPostureInput(f"trans_per_hour is negative ({trans_per_hour})")

    total = supine_pct + prone_pct + left_pct + right_pct
    if abs(total - 100.0) > 0.5:
        raise InvalidPostureInput(
            f"posture percentages sum to {total:.1f}, not 100. They must be taken over "
            f"SCORED time (total recorded time minus time the sleeper was not in bed), "
            f"not over total recorded time."
        )


def band_for(score):
    """
    FIX 1 - the original used `score <= 29` for Low and `score <= 59` for Moderate,
    which left 29.1 to 29.9 and 59.1 to 59.9 falling into the wrong band. A score of
    29.1 was reported as Moderate when the specification puts Moderate at 30 and above.
    """
    if score >= HIGH_THRESHOLD:
        return "High"
    if score >= MODERATE_THRESHOLD:
        return "Moderate"
    return "Low"


def max_achievable(weights, prone_available=True):
    """
    FIX 2 - a condition cannot always reach 100, so a low score is sometimes a blind
    spot rather than a finding.

    Two separate effects cap a condition, and the original handled only the first:

    1. A missing prone measurement zeroes the prone term. Neck strain then tops out
       at 30 and lower back strain at 45.
    2. The four posture shares are a simplex - they sum to 100 - so a condition
       weighted on two different postures can never max both. Nocturnal reflux
       (0.45 supine + 0.55 right) tops out at 55 and shoulder / hip loading at 70,
       whatever the classifier can see.

    The original returned 100.0 for every condition with a zero prone weight, so
    effect 2 went unreported and those two ceilings were invisible. It also only
    happened to be right about effect 1 because the positive weights in those two
    rows sum to exactly 1.0; `100 * (1 - w_prone)` is not the ceiling in general.

    The bout and transition terms are independent of the postures, so each
    contributes its positive part on top of the best single posture.
    """
    postures = [weights[0], weights[2], weights[3]]
    if prone_available:
        postures.append(weights[1])
    best_posture = max(max(postures), 0.0)
    return round(100.0 * (best_posture + max(weights[4], 0.0) + max(weights[5], 0.0)), 1)


def calculate_sleep_risks(supine_pct, prone_pct, left_pct, right_pct,
                          bout_hours, trans_per_hour, validate=True,
                          prone_detectable=None):
    """
    Returns a list of dicts, one per condition, each with:
        condition, score, band, capped_at, summary

    All four percentages must be taken over SCORED time - total recorded time
    minus any period the sleeper confirmed they were not in bed. They must sum
    to 100.

    FIX 5 - `prone_detectable` says whether the classifier is able to emit prone
    at all, which is not the same question as whether this sleeper was prone
    tonight. Inferring it from `prone_pct > 0`, as the original did, labels a
    genuine finding - somebody who simply never lies on their front - as a
    measurement blind spot. Pass it explicitly from the capture layer, which
    knows what labels the classifier produced across the whole log. Left at
    None it falls back to the original inference.
    """
    if validate:
        _validate(supine_pct, prone_pct, left_pct, right_pct, bout_hours, trans_per_hour)

    x_sup = supine_pct / 100.0
    x_pro = prone_pct / 100.0
    x_left = left_pct / 100.0
    x_right = right_pct / 100.0
    x_bout = min(1.0, bout_hours / BOUT_CAP_HOURS)
    x_trans = min(1.0, trans_per_hour / TRANS_CAP_PER_HOUR)

    inputs = [x_sup, x_pro, x_left, x_right, x_bout, x_trans]
    prone_available = (prone_pct > 0) if prone_detectable is None else bool(prone_detectable)

    results = []
    for condition, weights in WEIGHT_MATRIX.items():
        raw = 100.0 * sum(w * x for w, x in zip(weights, inputs))
        score = round(max(0.0, min(100.0, raw)), 1)
        band = band_for(score)
        cap = max_achievable(weights, prone_available)

        info = CONDITION_INFO[condition]
        if band == "High":
            summary = f"{info['mechanism']} Corrective action: {info['action']}"
        elif band == "Moderate":
            summary = info["mechanism"]
        else:
            summary = "The pattern looks unremarkable."

        cap_with_prone = max_achievable(weights, True)
        if cap < cap_with_prone:
            summary += (f" NOTE: without a prone measurement this condition cannot exceed "
                        f"{cap:.0f}, so a low result here is a blind spot rather than a finding.")
        elif cap < HIGH_THRESHOLD:
            summary += (f" NOTE: the weights cap this condition at {cap:.0f} for any posture "
                        f"mix, so it can never reach High.")
        elif cap < 100.0:
            summary += (f" NOTE: the weights cap this condition at {cap:.0f} for any posture mix.")

        results.append({
            "condition": condition,
            "score": score,
            "band": band,
            "capped_at": cap,
            "summary": summary,
        })

    return results
