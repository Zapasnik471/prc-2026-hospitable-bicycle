# Experiment log — team hospitable-bicycle

Everything we tried, in the order we tried it, including what did not work.
Leaderboard (LB) = public ranking RMSE on Jan + Jul 2026 (344,841 departures).
Validation (val) = train on 10 months of 2025, evaluate on Jan + Jul 2025 with **every row kept**
(see "Validation protocol"). Run-to-run noise on val: ~0.5 ALL / ~1.3 TRIM (TRIM = rows with y ≤ 3 h).

## 1. Validation protocol (the most important early lesson)

Our first models filtered "artifacts" (labels > 3 h, BLOCK equal to SCHED, ~1-day labels) from both
training and validation. Clean validation RMSE was ~314, but the leaderboard said 588. The ranking
set contains these records, and with a squared-error metric a handful of 1-day labels dominates.
From then on the holdout was **never filtered**; that with-tail validation tracked the leaderboard
(605.6 val vs 588.6 LB) and every later decision was made on it.

## 2. Leaderboard history

| Version | LB RMSE | Change |
|---|---|---|
| v1 | 599.65 | CatBoost on schedule-anchored features, artifacts filtered |
| v2 | 588.56 | tuned CatBoost (depth 10, lr 0.05, l2 3) |
| v8 | 324.34 | **AOBT block** (MVT − AOBT_3, MVT − SCHED, ...), **hurdle model**, no filtering of training data |
| v9 | 322.76 | LIRF rule (no AOBT and s ≥ 9 h → predict s) |
| v11 | 319.58 | queue features anchored at the actual take-off (MVT) |
| v13 | 319.35 | 3-seed average |
| v14 | 317.46 | carrier from the airport-side flight number |
| v15 | 317.38 | bulk regressor = 0.7 CatBoost + 0.3 LightGBM |
| v16 | 309.37 | turnaround features (link to the last arrival at the same stand) |
| v17 | 306.46 | disruption meters (airport regime in the 60 min before take-off) |
| v18 | 272.99 | **monster rules A + B** (replace the LIRF rule) |
| v19 | 272.70 | 5-seed average |
| v20 | 269.87 | turnaround vs flight plan (EOBT − in-block, EOBT − AOBT, ground ratio) |
| v21 | 269.84 | record-quirk encodings |
| v22 | 270.10 | linear-tree LightGBM — reverted |
| v23 | 267.88 | **LOBT window** (clip predictions to the window) |
| v24 | 266.96 | window + per-airport offset for predictions below the window |
| **v25** | **265.49** | time-to-take-off features from LOBT / EOBT / IOBT — **final** |

## 3. Key findings

### 3.1 The flight's own AOBT and MVT carry most of the signal
The ranking file blanks only BLOCK and TAXITIME of departures; the organisers confirmed that all other
columns of the flight may be used. `MVT − AOBT_3` (NM's view of the taxi-out) and `MVT − SCHED` are
the two anchors of the label, but neither is a copy of it (`analysis/record_quirks.py`, 2025):

| Airport | departures | AOBT missing | echo: BLOCK = SCHED (±60 s) | BLOCK ≈ AOBT (±120 s) | BLOCK ≫ 10 min before AOBT | BLOCK ≫ 10 min after AOBT | y > 1 h |
|---|---|---|---|---|---|---|---|
| LTFM | 272,965 | 1.34 % | 10.5 % | 17.1 % | 3.6 % | 12.1 % | 0.29 % |
| EHAM | 247,951 | 1.65 % | 6.9 % | 44.9 % | 0.1 % | 5.1 % | 0.03 % |
| LFPG | 239,552 | 1.58 % | 8.2 % | 40.2 % | 4.1 % | 2.9 % | 0.17 % |
| EGLL | 239,546 | 0.59 % | 9.8 % | 36.4 % | 2.6 % | 3.7 % | 0.30 % |
| EDDF | 230,141 | 0.85 % | 7.0 % | 41.5 % | 1.5 % | 0.1 % | 0.03 % |
| LEMD | 212,242 | 0.45 % | 8.6 % | 48.8 % | 2.8 % | 0.1 % | 0.00 % |
| LEBL | 179,705 | 1.04 % | 9.9 % | 47.6 % | 3.8 % | 0.4 % | 0.00 % |
| EDDM | 167,334 | 0.60 % | 10.3 % | 47.7 % | 7.2 % | 0.1 % | 0.04 % |
| LIRF | 160,704 | 0.93 % | **20.7 %** | 33.6 % | **13.0 %** | 0.4 % | **1.22 %** |
| LSZH | 134,907 | 1.70 % | 5.8 % | 44.8 % | 2.6 % | 0.1 % | 0.02 % |

Overall AOBT is missing for 1.08 % of departures (1.58 % in the ranking set) and BLOCK is within two
minutes of AOBT for only 39 % of them, so the model has to learn per airport and per carrier how the
airport records off-block. That is what the AOBT block, the 'record-quirk' encodings (echo / early /
late rates by airport × carrier and airport × stand) and the 1 % AOBT masking are for. Adding the AOBT
block and stopping the training-set filtering took LB from 588 to 324.

### 3.2 Hurdle model for the heavy tail
A single RMSE regressor is pulled toward the rare multi-hour labels. A classifier for `y > 1 h`
combined with separate bulk and log-target tail regressors (`yhat = (1 − p)·mu_lo + p·mu_hi`)
was clearly better on the with-tail validation.

### 3.3 Monster records at LIRF
In 2025, 49 of the 52 labels above 9 h are at LIRF (48 of them without AOBT; the other three are two
LFPG rows and one LSZH row, also without AOBT). Most are flights whose BLOCK equals SCHED ("echo"). The rest form a cluster at one day plus a normal taxi (C = 87,536 s).
Rule B (LIRF, no AOBT, s ≥ 32,400 s; 47 rows in 2025): below s = 50,000 all 27/27 labels equal s, so predict s; above,
predict the mixture `p·C + (1 − p)·s` with the cluster share per s band (0.42 for 50–64.8 ks,
0.83 for 64.8–86.4 ks, 0.50 above). Rule A (AOBT present but MVT − AOBT ≥ 32,400 s, i.e. AOBT itself is
a day off) is based on one 2025 example; the single ranking row it touches is independently confirmed by
the LOBT window, which forces its label above 83,214 s. All probabilities were estimated on 2025 and the
rules were submitted once (v18): no leaderboard probing. See `analysis/monster_records.py`.

### 3.4 LOBT window
In 100 % of the 2,062,577 departures of 2025 that have an LOBT, `|BLOCK − LOBT| ≤ 3606 s` (minimum and
maximum exactly ∓3606). Hence `MVT − LOBT − 3606 ≤ y ≤ MVT − LOBT + 3606`. Clipping predictions to
this window improved validation from 339.75 to 312.55 on only 138 changed rows (all of their labels
inside the window). When a prediction falls below a deep window (lower bound `lo ≥ 1800 s`, 909 rows in 2025), the truth
is not at the edge but well inside it: `y − lo` has mean 1831 s and median 1685 s, from ~250–900 s at
LEMD / LEBL / EDDM to 2962 s (median 3603 s, i.e. near the top of the window) at LIRF. Such rows are
moved to `lo + E[y − lo]` per airport, shrunk toward the global mean with weight 30
(global 1831 s, LIRF 2758, EGLL 1933, EHAM 1259, …). The idea of the window comes from team
outgoing-boat's public repository. See `analysis/lobt_window.py`.

### 3.5 Where the remaining error is
On validation the 10 worst rows hold ~47 % of the squared error: two LFPG rows whose BLOCK is a day
early (unpredictable from the inputs), LIRF monsters, and LIRF flights without AOBT in the middle
s range, where whether the airport echoed SCHED is close to a coin flip and only a calibrated hedge
is possible.

## 4. Final configuration (v25)

Features: base schedule-anchored block, AOBT block (1 % AOBT masking), queue at MVT, carrier,
turnaround, disruption, turnaround vs plan, time-to-take-off / LOBT window, target and quirk encodings,
8 % flight-plan masking. Model: hurdle (CatBoost depth 10, lr 0.05, l2 3; tail head depth 6;
bulk 0.7 CatBoost + 0.3 LightGBM), full-data refit with fixed iterations (750 / 4800 / 900, LightGBM 1654),
5 seeds. Post-processing: seed average → monster rules → LOBT window.
Validation of the refactored code (seed 42): ALL 334.87 / TRIM 239.94, with the LOBT window 309.10.

## 5. What did not work

| Idea | Result |
|---|---|
| Hyper-parameter grid (depth 8 / 12, l2 10, hurdle threshold 2700 / 5400 s, tail depth 8) | all worse than depth 10 / l2 3 / 3600 s / depth 6 |
| Linear-tree LightGBM | LB 270.10 vs 269.84 |
| Bulk regressor learning a residual to MVT − AOBT (`--resid`) | worse |
| Traffic state at our own pushback (`--v10`) | no gain |
| Explicit wake leader/follower pairs as numeric separation features beyond the base block | no gain |
| Extra METAR de-icing features (temperature change, snow accumulation) | zero importance, removed |
| Out-of-fold corrector model on top of the hurdle | no gain |
| Dedicated LIRF echo classifier head | no gain over the hurdle + rules |
| Per-airport caps on predictions | worse |
| Calibrating the tail with a function of s | worse |
| Mixture rule for LIRF mid-s lane | worse than the hedge |
| OPDI ground-event data (runway exit events) | coverage and quality insufficient |
| Own off-block detection from open ADS-B tracks (adsb.lol, pilot on 15 Jul 2025) | only −1.4 % SSE on covered rows, low coverage; not worth the pipeline |
