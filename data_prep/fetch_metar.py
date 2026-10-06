"""
Step 1/2 (weather): download raw METAR from the Iowa Environmental Mesonet (IEM) ASOS service
into <EXT_DIR>/metar_raw/metar_<ICAO>.csv. Resumable (skips files already downloaded), with retries.

    python data_prep/fetch_metar.py
"""
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

OUT = C.EXT_DIR / "metar_raw"
STS = "2025-01-01T00:00:00Z"            # 2025 (training) ...
ETS = "2026-08-01T00:00:00Z"            # ... through July 2026 (covers the ranking months)
DATA_VARS = ["tmpf", "dwpf", "relh", "p01i", "vsby", "sknt", "drct", "wxcodes"]
ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
RETRIES, TIMEOUT = 4, 120


def build_url(station: str) -> str:
    parts = [f"station={station}"] + [f"data={v}" for v in DATA_VARS]
    parts += [f"sts={STS}", f"ets={ETS}", "tz=Etc/UTC", "format=onlycomma", "latlon=no",
              "missing=M", "trace=T", "report_type=3", "report_type=4"]   # routine + special METARs
    return ENDPOINT + "?" + "&".join(parts)


def fetch(station: str) -> None:
    dest = OUT / f"metar_{station}.csv"
    if dest.exists() and dest.stat().st_size > 200:
        print(f"  {station}: already downloaded")
        return
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(build_url(station), timeout=TIMEOUT) as r:
                body = r.read().decode("utf-8", errors="replace")
            if body.count("\n") < 2:
                raise ValueError("empty response")
            dest.write_text(body, encoding="utf-8")
            print(f"  {station}: OK ({body.count(chr(10))} rows)")
            return
        except (urllib.error.URLError, ValueError, TimeoutError) as e:
            print(f"  {station}: attempt {attempt}/{RETRIES} failed ({e}); waiting {5 * attempt}s")
            time.sleep(5 * attempt)
    print(f"  {station}: FAILED, run the script again")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ap in C.AIRPORTS:
        fetch(ap)
        time.sleep(2)   # be polite to the IEM server


if __name__ == "__main__":
    main()
