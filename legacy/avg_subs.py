"""
USREDNIANIE SUBMISJI (seed averaging).

Kazdy trening tego samego modelu rozni sie na LB o ~2 pkt (szum treningu).
Srednia z K niezaleznych treningow zmniejsza ten szum ~K-krotnie.

WAZNE: usredniaj SUROWE submisje (bez reguly LIRF), a regule naloz na wynik.
(Uzycie plikow _lirf tez jest poprawne - 20 wierszy ma wtedy wszedzie te sama
wartosc s - ale czysciej jest usrednic surowe i dopiero potem nalozyc regule.)

    python avg_subs.py wynik.parquet sub_a.parquet sub_b.parquet sub_c.parquet ...
    python lirf_rule.py --apply --base wynik.parquet
"""
import sys
import numpy as np
import polars as pl

TARGET, ID = "TAXITIME_SEC_mvt", "MVT_ID_mvt"
out_path, paths = sys.argv[1], sys.argv[2:]
assert len(paths) >= 2, "podaj co najmniej 2 submisje"

subs = [pl.read_parquet(p) for p in paths]
ids = subs[0][ID]
for p, s in zip(paths, subs):
    assert s.height == subs[0].height, f"{p}: inna liczba wierszy"
    assert (s[ID] == ids).all(), f"{p}: inna kolejnosc/zestaw ID"
    assert s[TARGET].null_count() == 0, f"{p}: nulle"

P = np.column_stack([s[TARGET].to_numpy().astype(float) for s in subs])
print(f"{len(paths)} submisji, {P.shape[0]:,} wierszy")
for p, col in zip(paths, P.T):
    print(f"  {p}: srednia {col.mean():,.1f}")

# ile szumu miedzy modelami: srednia suma kwadratow odchylen od sredniej
mean = P.mean(axis=1)
noise = ((P - mean[:, None]) ** 2).sum(axis=0)
print(f"\nrozrzut kazdego modelu wokol sredniej (suma kw.): "
      + ", ".join(f"{n:.2e}" for n in noise))
print(f"-> szacowany zysk z usrednienia wzgledem pojedynczego modelu: ~{noise.mean():.2e} SSE "
      f"(~{np.sqrt(noise.mean() / P.shape[0]):.1f} s RMSE-szumu na wiersz)")

out = subs[0].select(ID).with_columns(pl.Series(TARGET, np.round(mean).astype(np.int32)))
assert (out[ID] == ids).all() and out[TARGET].null_count() == 0
out.write_parquet(out_path)
print(f"\nzapisano -> {out_path}  (srednia {out[TARGET].mean():,.1f})")
