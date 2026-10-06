import sys, polars as pl, numpy as np
def sc(f):
    d = pl.read_parquet(f)
    y = d["TAXITIME_SEC_mvt"].to_numpy(); yh = d["yhat"].to_numpy().clip(0)
    m = ((d["ADEP_mvt"] == "LIRF") & d["aobt_null"] & (d["s"] >= 32400)).to_numpy()
    yh = np.where(m, d["s"].to_numpy(), yh)
    t = y <= 10800
    li = ((d["ADEP_mvt"] == "LIRF") & d["aobt_null"]).to_numpy()
    return (round(np.sqrt(((y-yh)**2).mean()), 2), round(np.sqrt(((y[t]-yh[t])**2).mean()), 2),
            f"LIRF-null SSE {((y[li]-yh[li])**2).sum():.3e}")
for f in sys.argv[1:]:
    print(f, sc(f))
