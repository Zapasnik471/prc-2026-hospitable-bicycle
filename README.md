# PRC Data Challenge 2026 — taxi-out time (team hospitable-bicycle)

Prediction of the taxi-out time `TAXITIME_SEC_mvt = MVT - BLOCK` (actual take-off minus actual off-block)
for departures at ten European airports (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTFM, LSZH).
Metric: RMSE on Jan + Jul 2026. **Final leaderboard RMSE: 265.49** (submission v25).

## Approach in one paragraph

A gradient-boosted **hurdle model** (CatBoost classifier for `y > 1 h` × {bulk regressor = 0.7 CatBoost +
0.3 LightGBM, log-target tail regressor}) on ~113 features, averaged over 5 seeds, followed by two
data-driven **post-processing rules**: (1) *monster rules* for records whose off-block time is about one day
off, with probabilities estimated on 2025, and (2) the **LOBT window**: in 100 % of 2025 departures
`|BLOCK − LOBT| ≤ 3606 s`, which bounds the label to `[MVT − LOBT − 3606, MVT − LOBT + 3606]`.
Validation always keeps the full tail (no filtering of the holdout), because the leaderboard does.

## Pipeline

```
2025 movements ──► features (src/features) ──► target + quirk encodings ──► flight-plan masking
                                                                              │
2026 ranking   ──► features ──────────────────────────────────────────────────┤
                                                                              ▼
                              hurdle model, full-data refit, 5 seeds (src/model.py)
                                                                              │
                     seed average ─► monster rules ─► LOBT window (src/postprocess.py) ─► submission
```

| Step | Module | What it does |
|---|---|---|
| base features | `src/features/base.py` | planned/actual congestion, queue position, time, flight-plan revisions, wake pair, weather, geometry |
| AOBT block | `src/features/aobt.py` | `MVT − AOBT_3`, `MVT − SCHED` and derived flags; 1 % AOBT masking |
| queue at take-off | `src/features/queue.py` | runway/airport traffic around the actual take-off time |
| turnaround | `src/features/turnaround.py` | link to the last arrival at the same stand; buffer vs schedule and flight plan |
| disruption | `src/features/disruption.py` | airport regime in the 60 min before take-off |
| LOBT window | `src/features/plan.py` | time from LOBT/EOBT/IOBT to take-off, window lower bound |
| carrier | `src/features/carrier.py` | airline prefix of the airport-side flight number |
| encodings | `src/encoding.py`, `src/features/quirks.py` | out-of-fold target encodings; rates of 'record quirks' (BLOCK echoing SCHED, BLOCK far from AOBT) |
| model | `src/model.py` | hurdle model |
| post-processing | `src/postprocess.py` | seed averaging, monster rules, LOBT window |

## Repository layout

```
config.py            paths, constants, final hyper-parameters (no flags: one configuration)
run.py               reproduce the final submission
validate.py          10-month train / Jan+Jul 2025 holdout
src/                 pipeline code (see table above)
data_prep/           scripts that build the external data files
tools/               check_equivalence.py (refactor safety net, see below)
src/echo.py          experiment not used in the final submission (validate.py / run.py --echo), see docs/experiments.md
legacy/              the original experiment scripts that produced v25, kept for transparency
analysis/            probes on 2025 behind the rules: lobt_window, monster_records, record_quirks
docs/experiments.md  full experiment log incl. dead ends
```

## Reproducing

```bash
pip install -r requirements.txt

# competition data (OpenSky S3, credentials from the organisers) -> data/competition/
# external data -> data/external/
python data_prep/fetch_metar.py
python data_prep/build_metar_features.py
python data_prep/build_airports.py
python data_prep/build_osm_stands.py
python data_prep/build_taxi_distance.py

python validate.py --gpu --seed 42      # holdout RMSE, see below
python run.py --gpu                     # 5 seeds -> output/submissions/hospitable-bicycle_vfinal.parquet
```

Paths can be changed with `PRC_DATA_DIR`, `PRC_EXT_DIR`, `PRC_OUT_DIR`. All our runs used Google Colab
(A100, CatBoost on GPU); one seed takes about 20 min, the full 5-seed run about 1 h 45 min (Python 3.13, versions pinned in `requirements.txt`). GPU training is not bit-for-bit deterministic;
run-to-run noise is ~0.5–1.5 RMSE on the holdout.

### Validation (Jan + Jul 2025 holdout, all rows kept, seed 42, GPU)

| | ALL | TRIM (y ≤ 3 h) |
|---|---|---|
| model + LIRF rule | 334.87 | 239.94 |
| + LOBT window | 309.10 | 239.98 |

The original scripts gave 336.2 / 241.2 for the same configuration; features are bit-identical
(see below), so the difference is GPU training noise.

### Refactor check

The code in `src/` is a restructuring of the scripts in `legacy/`. `tools/check_equivalence.py` verifies
that the refactored pipeline produces identical model inputs (every feature column, both the validation
and the submission path) and identical post-processing outputs:

```bash
python tools/check_equivalence.py features --seed 42
python tools/check_equivalence.py post --seed-files <5 seed files> --avg avgj.parquet --final hospitable-bicycle_v25.parquet
```

## Key findings

1. **Use the flight's own AOBT and MVT.** The organisers allow every non-blanked column; `MVT − AOBT_3`
   and `MVT − SCHED` anchor the label (BLOCK is within 2 min of AOBT for 39 % of flights and equals SCHED
   for 6–21 % depending on the airport); with a hurdle model and an unfiltered training set this took
   the leaderboard from 588 to 324.
2. **Validate with the tail.** Filtering ~1-day records from the holdout gave a clean 314 that meant
   nothing; the unfiltered holdout tracks the leaderboard.
3. **LOBT window.** `|BLOCK − LOBT| ≤ 3606 s` in 100 % of 2025 departures, which bounds every label;
   predictions are clipped and moved inside the window (validation 339.75 → 312.55).
4. **Monster records.** 49 of 52 labels above 9 h in 2025 are at LIRF (BLOCK copying SCHED, or one day
   plus taxi); rules with probabilities estimated on 2025 (leaderboard 306 → 273).

Full log with every leaderboard step and all dead ends: [`docs/experiments.md`](docs/experiments.md).

## External data and licences

| Data | Source | Licence | Used for |
|---|---|---|---|
| METAR (ASOS) | Iowa Environmental Mesonet (Iowa State University), `mesonet.agron.iastate.edu/cgi-bin/request/asos.py` | free for any lawful purpose, provided as-is (IEM API terms); METARs are public aviation weather reports | weather features |
| Airports, runways | OurAirports (`davidmegginson.github.io/ourairports-data`) | Public Domain | route geometry, runway thresholds |
| Stand positions | OpenStreetMap via Overpass API | ODbL 1.0, © OpenStreetMap contributors | stand → runway distance |
| Aircraft type → wake category | own compilation of the public ICAO Doc 8643 wake categories (`src/features/wake.py`) | part of this repository (GPL-3.0) | wake features |

No data files are included in this repository. The competition data (OpenSky Network / PRC) is
used under the challenge terms and is not redistributed; external files are rebuilt with `data_prep/`.

## Acknowledgements

Ideas taken from other teams' public repositories are credited in the code:
the LOBT window (team outgoing-boat, `github.com/EnioAguiar/prc-taxiout-2026`),
turnaround and disruption meters (team kind-mango, `github.com/skylinkapi/prc-data-challenge-2026-kind-mango`),
airport-regime features (team joyous-rainbow).

## Originality

All code in this repository was written by team hospitable-bicycle. Other teams' public repositories
were read for ideas only; every idea we adopted was re-derived and verified on our own 2025 data before
use (for the LOBT window: `analysis/lobt_window.py`), and is credited above and in the code. No
third-party competition code was copied or adapted.

## Licence

GPL-3.0, see `LICENSE`. Third-party data attributions: `NOTICE`.
