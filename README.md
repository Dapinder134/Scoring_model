# Scoring_model

Sleep posture scoring: position log → cleaned posture timeline → per-night statistics
→ data-quality gate → eight condition scores.

```bash
python sleep_statistics.py data/Night_001_20260821_222832.csv
python sleep_statistics.py sleep_positions_1.txt --session 4
python sleep_statistics.py data/Night_001_20260821_222832.csv --reassign 4=4 6=4 8=4
python -m unittest discover -s tests -t .        # 64 tests
```

Pure standard library, no dependencies.

## Layout

```
scoring_model.py      the risk model: weights, bands, ceilings
sleep_statistics.py   sessions, statistics, quality gate, reporting, CLI
sleep_scoring/        capture back end
  config.py             thresholds and the posture vocabulary
  loading.py            reads both capture formats, audits the raw stream
  intervals.py          sample-and-hold, dedup, dropout bridging, debounce
tests/                64 tests, including both real captures end to end
data/                 the capture files
```

Both capture formats are accepted and dispatched on extension:

```
capture CSV      Timestamp,Session_Name,Position,<32 landmark columns>
plain-text log   YYYY-MM-DD HH:MM:SS, <label>
```

## The hold rule

A record's posture is held until the next record arrives. A record at 05:06:10
followed by one at 05:10:13 is a single unbroken 4m03s stretch, and a repeat of
the same posture continues the stretch rather than starting a new one.

Applied to the real capture format that rule needs three guards first, because
the raw stream carries artefacts that a direct reading turns into nonsense:

| Artefact | In `Night_001_20260821_222832.csv` |
|---|---|
| Records sharing a timestamp | 2245, of which 1211 disagree about the posture |
| Label changes per hour | 512 |
| Detected/absent alternation | 99.3% strict |

Nobody turns over eight times a minute for seven hours, so `intervals.py`
removes all three before any duration is attributed: it collapses each
timestamp to one winning record (a detection beats a non-detection; landmark
confidence breaks the rest), refills non-detections shorter than
`bridge_absence_s`, and absorbs posture runs shorter than `debounce_s` into
their longer *posture* neighbour.

## Configuration

| Setting | Default | Meaning |
|---|---|---|
| `session_gap_s` | 14400 | Absence long enough to end a night |
| `debounce_s` | 30 | A posture must hold this long to be a real turn |
| `bridge_absence_s` | 120 | Shorter non-detections are detector dropout |
| `max_hold_s` | `None` | Cap on holding one record; `None` applies the hold rule literally |
| `evidence_gap_s` | 120 | Longer stretches count as inferred, not observed |
| `min_scored_hours` | 4.0 | Below this the night is not scored |
| `max_absent_pct` | 20.0 | Unresolved absence above this fails the gate |
| `min_distinct_positions` | 2 | Below this a night carries no positional information |

## Fixes applied

### `scoring_model.py`

1. **Band boundaries** — `score <= 29` / `score <= 59` left 29.1–29.9 and
   59.1–59.9 in the wrong band. Now `>= 30` is Moderate and `>= 60` is High.
2. **Ceilings** — `100 * (1 - w_prone)` only described the prone blind spot,
   and only by coincidence (it is right just when a row's positive weights sum
   to 1.0). It also returned 100 for every condition with a zero prone weight,
   hiding the fact that the four posture shares are a simplex. `max_achievable`
   now takes the best single posture plus the positive bout and transition
   terms, and is verified against a brute-force sweep of the whole input space:

   | condition | was | now (prone) | now (no prone) |
   |---|---|---|---|
   | Nocturnal reflux (GORD) | 100 | **55** | **55** |
   | Shoulder / hip joint loading | 100 | **70** | **70** |
   | Neck strain | 30 | 100 | 30 |
   | Lower back strain | 45 | 55 | 45 |

3. **Input validation** — percentages must sum to 100, nothing may be negative.
4. **NaN and infinity** — these defeated every guard. `nan < 0` is False and
   `abs(nan - 100) > 0.5` is also False, so a NaN percentage passed validation,
   reached `min(100.0, nan)` — which Python resolves to `100.0` — and returned
   100.0 / High on all eight conditions. A dropped landmark produced maximum
   alarm. Both are now rejected on every input.
5. **`prone_detectable`** — inferring it from `prone_pct > 0` labelled a
   sleeper who simply never lies on their front as a measurement blind spot.
   The capture layer now passes it explicitly, since whether the classifier can
   emit prone is a property of the whole log, not of one night.

### `sleep_statistics.py`

1. **Session gap 600 s → 14400 s.** A ten-minute gap is not a new night. At
   600 s one real capture split into ten sessions, and because time between
   sessions belongs to no session, **5.56 h of 7.27 h — 76.5% of the
   recording — was silently discarded.** `split_sessions` now also returns
   boundary time so nothing can vanish unreported.
2. **Debounce no longer lets absence swallow detections.** Folding a short run
   into whichever neighbour was longer meant a long non-detection absorbed real
   observations: 3324 records containing 1659 genuine Right Side detections
   collapsed into one 33-minute "No Person Detected" run. Short runs now fold
   only into posture neighbours, and a detection with no posture neighbour is
   left alone.
3. **`min_distinct_positions` 4 → 2.** Requiring all four postures in one night
   fails every sleeper who never lies prone, which is most of them.
4. **`debounce_is_meaningful()`** compared the wrong way round. Under
   sample-and-hold at a 210 s interval the shortest possible run is 210 s, so a
   5 s window can never fire — yet it reported "meaningful: yes", and a 300 s
   window, which does fire, as no.
5. **Default session choice** picked the longest by total duration, which on a
   real capture selected a block of pure absence with no posture data in it.
   It now picks by scored posture time.
6. **Both capture formats** are read by the same loader, so the 35-column CSV
   no longer needs converting by hand.

## Known limits

- **82.6% of `Night_001_20260821_222832.csv` is inferred**, not observed: the
  logger goes quiet for up to 50 minutes at a time, and under the hold rule a
  single record then decides that whole stretch. Bout lengths are soft, and a
  contested timestamp immediately before a long gap carries unusual weight. Set
  `--max-hold` to cap it.
- The CSV resolves contested timestamps using landmark visibility; the text log
  has no such column and can only take the later record. On this capture that
  one difference moves 67 minutes between Left and Right. Prefer the CSV.
- The `Position` vocabulary in these files is `Back`, `Left Side`, `Right Side`,
  `No Person Detected` — no prone label, so every prone-weighted term is dead.
  Extend `sleep_scoring.config.DEFAULT_LABEL_MAP` when the classifier gains one.
- `Back` is assumed supine. If the classifier also labels face-down as `Back`,
  supine and prone are being conflated.
- The weights are heuristic and uncalibrated; see the note in `scoring_model.py`.
