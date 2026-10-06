"""
Step 1/2 (stand geometry): stand coordinates from OpenStreetMap (aeroway=parking_position) via the
Overpass API -> <EXT_DIR>/osm_stands.parquet  (icao, stand_code, stand_lat, stand_lon).
Data (c) OpenStreetMap contributors, ODbL. Needs airports.parquet (build_airports.py).

    python data_prep/build_osm_stands.py
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

OUT = C.EXT_DIR / "osm_stands.parquet"
OVERPASS = "https://overpass-api.de/api/interpreter"
BOX = 0.08   # degrees around the airport reference point
HEADERS = {"User-Agent": "prc-data-challenge-student/1.0 (taxi-out research)", "Accept": "application/json"}


def norm(c):
    return c.replace(" ", "").upper() if c else None


def fetch(lat, lon):
    s, w, n, e = lat - BOX, lon - BOX, lat + BOX, lon + BOX
    q = f'''[out:json][timeout:90];
nwr["aeroway"="parking_position"]({s},{w},{n},{e});
out center tags;'''
    data = urllib.parse.urlencode({"data": q}).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(OVERPASS, data=data, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as ex:
            print(f"    attempt {attempt + 1} failed ({ex}); waiting 15 s")
            time.sleep(15)
    return {"elements": []}


def main():
    ap = pl.read_parquet(C.AIRPORTS_FILE)
    coords = {r["icao"]: (r["lat"], r["lon"]) for r in ap.filter(pl.col("icao").is_in(C.AIRPORTS)).to_dicts()}
    rows = []
    for icao in C.AIRPORTS:
        if icao not in coords:
            print(f"{icao}: no coordinates, skipped")
            continue
        seen = {}
        for el in fetch(*coords[icao]).get("elements", []):
            t = el.get("tags", {})
            code = norm(t.get("ref") or t.get("gate"))
            if not code:
                continue
            # node has lat/lon, way/relation has a center
            plat = el.get("lat") or (el.get("center") or {}).get("lat")
            plon = el.get("lon") or (el.get("center") or {}).get("lon")
            if plat is None or plon is None:
                continue
            seen.setdefault(code, []).append((plat, plon))
        for code, pts in seen.items():   # several objects with the same code -> mean position
            rows.append({"icao": icao, "stand_code": code,
                         "stand_lat": sum(p[0] for p in pts) / len(pts),
                         "stand_lon": sum(p[1] for p in pts) / len(pts)})
        print(f"{icao}: {len(seen)} stands with coordinates")
        time.sleep(3)
    out = pl.DataFrame(rows)
    out.write_parquet(OUT)
    print(f"wrote {out.height:,} stands -> {OUT}")


if __name__ == "__main__":
    main()
