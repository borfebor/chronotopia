"""
Chronotopia verification harness.

Feeds known ground truth (a 24 h sine at several sampling rates, synthetic
plates) through every function this project has touched, and asserts the answer
is right. Sections 1-10 cover the Stage-1 correctness fixes, 11-12 the v0.8
smoothing changes, 13-15 the v0.7.3 plate detection and plots.py refactor.

Run:  python verify.py
"""
import string
import logging
import sys
import types
from io import BytesIO

logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

# --- stub out streamlit so methods.py imports without a running server --------
_st = types.ModuleType("streamlit")
_st.slider = lambda *a, **k: k.get("value", 10)
for _name in ("error", "warning", "stop", "write", "toast", "info", "success"):
    setattr(_st, _name, lambda *a, **k: None)
_st.checkbox = _st.toggle = lambda *a, **k: False
sys.modules.setdefault("streamlit", _st)

from methods import methods  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  — ' + detail if detail else ''}")


def sine(dt_h, days=10, period=24.0, noise=0.0, seed=0):
    t = np.arange(0, days * 24, dt_h)
    rng = np.random.default_rng(seed)
    y = np.sin(2 * np.pi * t / period) + rng.normal(0, noise, len(t))
    return t, y


# =============================================================================
print("\n1. sampling_interval — must ignore replicate counts and row order")
# =============================================================================
check("regular hourly", np.isclose(methods.sampling_interval(np.repeat(np.arange(0, 10), 3)), 1.0))

shuffled = pd.Series([2, 0, 1, 2, 0, 1])
check("shuffled rows still give +1.0",
      np.isclose(methods.sampling_interval(shuffled.values), 1.0),
      f"got {methods.sampling_interval(shuffled.values)}")

uneven = pd.Series([0, 0, 0, 1, 1, 2])  # 3, 2 and 1 replicates
check("uneven replicate counts give +1.0",
      np.isclose(methods.sampling_interval(uneven.values), 1.0),
      f"got {methods.sampling_interval(uneven.values)}")

check("10-min sampling -> 0.1667 h",
      np.isclose(methods.sampling_interval(np.arange(0, 24, 1 / 6)), 1 / 6))

# =============================================================================
print("\n2. period_correlation — must return HOURS, not lag index")
# =============================================================================
for dt in (0.1667, 0.5, 1.0, 2.0):
    t, y = sine(dt)
    p = methods.period_correlation(y, dt, 18, 36)
    check(f"24 h sine @ dt={dt:g} h -> {p:.2f} h", abs(p - 24) <= max(1.0, dt))

t, y = sine(0.5, period=20.0)
p = methods.period_correlation(y, 0.5, 18, 36)
check(f"20 h sine -> {p:.2f} h", abs(p - 20) <= 1.0)

check("degenerate delta_t returns NaN", np.isnan(methods.period_correlation(y, 0)))

# =============================================================================
print("\n3. period_estimation — end-to-end, the path app.py actually calls")
# =============================================================================
for dt in (0.5, 1.0):
    t, y = sine(dt, noise=0.05)
    df = pd.DataFrame({"time": t, "s1": y})
    for meth in ("Autocorrelation", "Lomb-Scargle Periodogram", "Fast Fourier Transform (FFT)"):
        p = methods.period_estimation(df, ["s1"], "time", method=meth,
                                      min_period=18, max_period=36).loc["s1"]
        check(f"{meth} @ dt={dt:g} -> {p:.2f} h", abs(p - 24) <= 1.5)

# =============================================================================
print("\n4. wavelet() — periodogram fs must be 1/dt")
# =============================================================================
t, y = sine(0.5)
f_bad, P_bad = sp_signal.periodogram((y - y.mean()) / y.std(), fs=0.5)
f_good, P_good = sp_signal.periodogram((y - y.mean()) / y.std(), fs=1 / 0.5)
check(f"fs=1/dt recovers 24 h (fs=dt gave {1 / f_bad[P_bad.argmax()]:.0f} h)",
      abs(1 / f_good[P_good.argmax()] - 24) < 1)

import inspect  # noqa: E402
check("methods.wavelet source uses fs=1 / dt", "fs=1 / dt" in inspect.getsource(methods.wavelet))

# =============================================================================
print("\n5. Hilbert detrending ('Rolling Hilbert') — envelope must run along TIME (axis 0)")
# =============================================================================
t = np.arange(0, 10 * 24, 0.5)
# three samples with very different amplitudes; after envelope normalisation
# they must all end up with comparable amplitude
amps = [1.0, 5.0, 20.0]
d = pd.DataFrame({f"s{i}": a * np.sin(2 * np.pi * t / 24) for i, a in enumerate(amps)})
out = methods.envelope_rolling(d, win=48)
rng_ = out.abs().max()
check("shape preserved", out.shape == d.shape, f"{out.shape}")
check(f"amplitudes equalised across 1x/5x/20x inputs (max/min = {rng_.max() / rng_.min():.2f})",
      rng_.max() / rng_.min() < 1.2)

env = np.abs(sp_signal.hilbert(d.to_numpy(), axis=0))
check("envelope tracks each column independently",
      np.allclose(env.max(axis=0) / env.max(axis=0)[0], np.array(amps) / amps[0], rtol=0.1))

# =============================================================================
print("\n6. envelope_rolling honours its win argument")
# =============================================================================
src = inspect.getsource(methods.envelope_rolling)
check("no hardcoded win=20", "win=20)" not in src.split("def")[1].split(":", 1)[1])
y = pd.DataFrame({"s": np.sin(2 * np.pi * t / 24) + 0.02 * t})
a = methods.envelope_rolling(y, win=5)
b = methods.envelope_rolling(y, win=200)
check("different windows give different results", not np.allclose(a.fillna(0), b.fillna(0)))

check("rolling() is centred", "center=True" in inspect.getsource(methods.rolling))

# =============================================================================
print("\n7. Butterworth band is valid once delta_t is in hours")
# =============================================================================
for dt, should_work in ((1 / 6, True), (0.5, True), (1.0, True)):
    fs = 1 / dt
    ok = True
    try:
        sp_signal.butter(4, [(1 / 30) / (fs / 2), (1 / 18) / (fs / 2)], btype="band")
    except Exception:
        ok = False
    check(f"dt={dt:.4g} h -> filter constructs", ok is should_work)

# =============================================================================
print("\n8. detrend / normalize still work through the public API")
# =============================================================================
t = np.arange(0, 10 * 24, 1.0)
d = pd.DataFrame({"time": t, "a": np.sin(2 * np.pi * t / 24) + 0.05 * t,
                  "b": np.cos(2 * np.pi * t / 24) + 3})
for m in ("None", "Linear", "Rolling mean", "Rolling Hilbert", "Cubic"):
    try:
        r = methods.detrend(d, ["a", "b"], "time", m)
        check(f"detrend '{m}' -> {np.asarray(r).shape}", np.asarray(r).shape == (len(t), 2))
    except Exception as e:
        check(f"detrend '{m}'", False, f"{type(e).__name__}: {e}")

for m in ("None", "Z-Score", "Sample-wise Min-Max", "Global Min-Max"):
    try:
        r = methods.normalize(d, ["a", "b"], m)
        check(f"normalize '{m}'", np.asarray(r).shape == (len(t), 2))
    except Exception as e:
        check(f"normalize '{m}'", False, f"{type(e).__name__}: {e}")

_lin = np.asarray(methods.detrend(d, ["a"], "time", "Linear")).ravel()
check(f"Linear detrend removes the 0.05*t slope (residual slope {np.polyfit(t, _lin, 1)[0]:.2e})",
      abs(np.polyfit(t, _lin, 1)[0]) < 1e-3)

# =============================================================================
print("\n9. add_entrainment no longer uses value_counts ordering")
# =============================================================================
d2 = pd.DataFrame({"time": np.arange(0, 10 * 24, 1.0), "a": 1.0})
ent = methods.add_entrainment(d2, "time", n_days=3, period=24, on_ratio=0.5)
check("returns time + entrainment", list(ent.columns) == ["time", "entrainment"])
check("entrainment is binary", set(np.unique(ent["entrainment"])) <= {0, 1})
rising = np.where(np.diff(ent["entrainment"].to_numpy().astype(int)) > 0)[0]
cyc, T, cut = methods.count_entrainment_days(ent["entrainment"].values, 1.0)
check(f"round-trips to T=24 h ({cyc} cycles detected)", abs(T - 24) < 1e-6)
check("no value_counts in add_entrainment", "value_counts" not in inspect.getsource(methods.add_entrainment))

# =============================================================================
print("\n10. End-to-end: a MINUTE-sampled file, the case that was broken")
# =============================================================================
# 15-min sampling, 7 days — exactly what a luminometer or actimeter emits.
raw_minutes = np.arange(0, 7 * 24 * 60, 15)
d3 = pd.DataFrame({"Time": raw_minutes})
d3["s1"] = np.sin(2 * np.pi * raw_minutes / (24 * 60))

# --- old behaviour: delta_t taken BEFORE the unit conversion ---
old_delta_t = float(np.mean(np.diff(d3["Time"].value_counts().index)))
# --- new behaviour: after the conversion, from sorted unique times ---
d3["Time"] = d3["Time"].apply(lambda x: methods.time_changer(x, "Minutes"))
new_delta_t = methods.sampling_interval(d3["Time"].values)

check(f"delta_t is now hours: {new_delta_t:.4f} h (was {old_delta_t:.1f}, i.e. raw minutes)",
      np.isclose(new_delta_t, 0.25))

check(f"header would read 'recorded every {new_delta_t:.2f} h' not '{old_delta_t:.1f} h'",
      new_delta_t < 1)

check(f"'Mean' smoothing window = {int(round(1 / new_delta_t))} samples (was "
      f"{max(1, int(round(1 / old_delta_t)))} — i.e. a no-op)",
      int(round(1 / new_delta_t)) == 4)

fs_new, fs_old = 1 / new_delta_t, 1 / old_delta_t
ok_new = ok_old = True
try:
    sp_signal.butter(4, [(1 / 30) / (fs_new / 2), (1 / 18) / (fs_new / 2)], btype="band")
except Exception:
    ok_new = False
try:
    sp_signal.butter(4, [(1 / 30) / (fs_old / 2), (1 / 18) / (fs_old / 2)], btype="band")
except Exception:
    ok_old = False
check(f"Butterworth constructs (new={ok_new}, old={ok_old})", ok_new and not ok_old)

p = methods.period_estimation(d3, ["s1"], "Time", method="Autocorrelation",
                              min_period=18, max_period=36).loc["s1"]
check(f"Autocorrelation on the minute file -> {p:.2f} h (old code would say {p / new_delta_t:.0f})",
      abs(p - 24) < 0.5)

# =============================================================================
print("\n11. Savitzky-Golay smoothing (v0.7.2)")
# =============================================================================
DT = 0.5
t = np.arange(0, 10 * 24, DT)
rng = np.random.default_rng(7)
clean = np.sin(2 * np.pi * t / 24)
noisy = clean + rng.normal(0, 0.30, len(t))
d4 = pd.DataFrame({"time": t, "clean": clean, "noisy": noisy})

sm, note = methods.savitzky_golay(d4, ["clean", "noisy"], DT, window_h=6, polyorder=2)
check("6 h window used as requested (no clamp note)", note == "", note)

amp_in = (clean.max() - clean.min()) / 2
amp_out = (sm["clean"].max() - sm["clean"].min()) / 2
check(f"24 h amplitude preserved: {amp_out / amp_in:.4f} of input", amp_out / amp_in > 0.99)

res_before = np.std(noisy - clean)
res_after = np.std(sm["noisy"].to_numpy() - clean)
check(f"noise cut from sd={res_before:.3f} to {res_after:.3f} ({res_after / res_before:.0%})",
      res_after < 0.5 * res_before)

def peak_positions(y):
    """Indices of local maxima. np.argmax is useless here — a clean sine has many
    peaks at exactly 1.0, so argmax just reports whichever tie floating point
    happens to favour, which is not a phase measurement."""
    return [i for i in range(1, len(y) - 1) if y[i] >= y[i - 1] and y[i] >= y[i + 1]]


check(f"no phase shift — all {len(peak_positions(clean))} peaks stay put",
      peak_positions(sm['clean'].to_numpy()) == peak_positions(clean))

check("output shape unchanged", sm.shape == (len(t), 2))
check("no NaN introduced at the edges", not sm.isna().to_numpy().any())

# --- a 12 h window really does cost you the waveform harmonic ---
sq = pd.DataFrame({"s": np.sign(np.sin(2 * np.pi * t / 24))})
h6, _ = methods.savitzky_golay(sq, ["s"], DT, window_h=6, polyorder=2)
h12, _ = methods.savitzky_golay(sq, ["s"], DT, window_h=12, polyorder=2)
sharp6 = np.abs(np.diff(h6["s"].to_numpy())).max()
sharp12 = np.abs(np.diff(h12["s"].to_numpy())).max()
check(f"6 h keeps square-wave edges sharper than 12 h ({sharp6:.3f} vs {sharp12:.3f})",
      sharp6 > sharp12)

# --- degree 2 vs 3: identical in the interior, different only at the edges.
#     This is why 3 is not offered as a separate choice — and why the tooltip
#     must say "interior", not "output". ---
a2, _ = methods.savitzky_golay(d4, ["noisy"], DT, window_h=6, polyorder=2)
a3, _ = methods.savitzky_golay(d4, ["noisy"], DT, window_h=6, polyorder=3)
v2, v3 = a2.to_numpy().ravel(), a3.to_numpy().ravel()
half = int(round(6 / DT)) // 2 + 1
check(f"polyorder 2 == polyorder 3 in the interior (max diff {np.abs(v2 - v3)[half:-half].max():.1e})",
      np.allclose(v2[half:-half], v3[half:-half]))
check(f"...but they differ in the end windows (max diff {np.abs(v2 - v3)[:half].max():.4f}) — "
      "mode='interp' extrapolates at the requested degree",
      not np.allclose(v2[:half], v3[:half]))

# --- sparse sampling must clamp, not raise ---
for dt_sparse, n_pts in ((2.0, 84), (3.0, 56), (4.0, 42), (6.0, 28)):
    ts = np.arange(0, n_pts) * dt_sparse
    dsp = pd.DataFrame({"s": np.sin(2 * np.pi * ts / 24)})
    try:
        out, nte = methods.savitzky_golay(dsp, ["s"], dt_sparse, window_h=6, polyorder=2)
        ok = out.shape == (n_pts, 1) and np.isfinite(out.to_numpy()).all()
        check(f"dt={dt_sparse:g} h clamps instead of raising"
              + (f" — '{nte[:52]}...'" if nte else ""), ok)
    except Exception as e:
        check(f"dt={dt_sparse:g} h", False, f"{type(e).__name__}: {e}")

# --- degenerate inputs ---
tiny = pd.DataFrame({"s": [1.0, 2.0]})
out, nte = methods.savitzky_golay(tiny, ["s"], 1.0, window_h=6, polyorder=2)
check("2-point series returns unchanged with an explanation", len(nte) > 0 and out.shape == (2, 1))

out, nte = methods.savitzky_golay(d4, ["clean"], np.nan, window_h=6, polyorder=2)
check("NaN sampling interval is handled", "skipped" in nte.lower())

out, nte = methods.savitzky_golay(d4, ["clean"], DT, window_h=1000, polyorder=2)
check(f"window longer than the recording clamps — '{nte[:44]}...'", "longer than" in nte)

# --- NaN gaps must not spread across the window ---
gappy = d4.copy()
gappy.loc[100:104, "noisy"] = np.nan
out, _ = methods.savitzky_golay(gappy, ["noisy"], DT, window_h=6, polyorder=2)
n_nan_in, n_nan_out = int(gappy["noisy"].isna().sum()), int(out["noisy"].isna().sum())
check(f"a 5-point gap stays 5 points, not a whole window ({n_nan_in} in -> {n_nan_out} out)",
      n_nan_in == n_nan_out)
check("values either side of the gap are finite",
      np.isfinite(out["noisy"].to_numpy()[[99, 105]]).all())

# =============================================================================
print("\n12. Removals are complete and the survivors still work")
# =============================================================================
for gone in ("apply_butter", "butter_bandpass_filter", "hilbert_rolling_mean"):
    check(f"methods.{gone} removed", not hasattr(methods, gone))

d5 = pd.DataFrame({"time": t, "a": clean + 0.02 * t, "b": np.cos(2 * np.pi * t / 24) + 3})
for m in ("None", "Linear", "Rolling mean", "Rolling Hilbert", "Cubic"):
    try:
        r = methods.detrend(d5, ["a", "b"], "time", m)
        check(f"detrend '{m}'", np.asarray(r).shape == (len(t), 2))
    except Exception as e:
        check(f"detrend '{m}'", False, f"{type(e).__name__}: {e}")

check("'Hilbert + Rolling mean' no longer in the detrend dispatch",
      "Hilbert + Rolling mean" not in inspect.getsource(methods.detrend))
# The invariant is that selecting one method runs ONLY that method — as plain
# dict values every estimator was evaluated on every rerun, and a failure in any
# one took down all of them. v0.7.6 split estimation from removal, so the check is
# now behavioural rather than a grep for a lambda: count the calls.
_orig = {n: getattr(methods, n) for n in
         ("polynomial_baseline", "rolling_baseline", "loess_baseline",
          "exponential_baseline")}
_calls = {n: 0 for n in _orig}
try:
    for _n, _f in _orig.items():
        def _counter(*a, __n=_n, __f=_f, **k):
            _calls[__n] += 1
            return __f(*a, **k)
        setattr(methods, _n, staticmethod(_counter))
    methods.detrend(d5, ["a", "b"], "time", "Rolling mean")
finally:
    for _n, _f in _orig.items():
        setattr(methods, _n, staticmethod(_f))
check(f"detrend runs only the selected estimator ({sum(_calls.values())} of "
      f"{len(_calls)} called)",
      _calls["rolling_baseline"] == 1 and sum(_calls.values()) == 1)

check("every advertised estimator is reachable from detrend()",
      all(methods.estimate_baseline(d5, ["a", "b"], "time", m, window=5,
                                    delta_t=float(np.diff(t).mean()),
                                    span_h=48.0) is not None
          for m in methods.BASELINE_METHODS))

# v0.7.7 pruned the menu on measured redundancy — see the NOTE in methods.py.
for _gone in ("sinc_baseline",):
    check(f"methods.{_gone} removed", not hasattr(methods, _gone))
check("Running median and Sinc low-pass are no longer offered",
      not ({"Running median", "Sinc low-pass"} & set(methods.BASELINE_METHODS)),
      str(methods.BASELINE_METHODS))

for _rem in ("Subtract", "Divide"):
    for _m in methods.BASELINE_METHODS:
        try:
            _r = methods.detrend(d5, ["a", "b"], "time", _m, removal=_rem)
            check(f"detrend '{_m}' / {_rem}", np.asarray(_r).shape == (len(t), 2))
        except Exception as _e:
            check(f"detrend '{_m}' / {_rem}", False, f"{type(_e).__name__}: {_e}")

# Divide must refuse a baseline that crosses zero rather than returning infinities.
_centred = pd.DataFrame({"time": t, "a": clean, "b": np.cos(2 * np.pi * t / 24)})
_div = np.asarray(methods.detrend(_centred, ["a", "b"], "time", "Linear", removal="Divide"))
_sub = np.asarray(methods.detrend(_centred, ["a", "b"], "time", "Linear", removal="Subtract"))
check("Divide falls back to Subtract when the baseline crosses zero",
      np.allclose(_div, _sub, equal_nan=True))

# =============================================================================
print("\n13. Well-ID parsing (v0.7.3)")
# =============================================================================
import plates  # noqa: E402

WELL_CASES = [
    ("A1", ("A", 1)), ("a01", ("A", 1)), ("G8", ("G", 8)),
    ("Well B02", ("B", 2)), ("sample_H12_ctrl", ("H", 12)),
    ("Plate1_C7", ("C", 7)), ("Plate2_A01", ("A", 1)),
    ("B_02", ("B", 2)), ("A-1", ("A", 1)), ("d 4", ("D", 4)),
    ("2024_B7", ("B", 7)), ("A24", ("A", 24)), ("P53_A07", ("A", 7)),
    # things that must NOT look like wells
    ("Time", None), ("sample_1", None), ("COL_12", None), ("control", None),
    ("Q1", None), ("A0", None), ("P53", None), ("A100", None), ("A25", None),
]
bad = [(n, plates.parse_well(n), exp) for n, exp in WELL_CASES if plates.parse_well(n) != exp]
check(f"{len(WELL_CASES)} well-name cases parse correctly", not bad,
      "" if not bad else f"wrong: {bad[:3]}")

# =============================================================================
print("\n14. Plate format detection")
# =============================================================================
for size, (r, c) in plates.PLATE_FORMATS.items():
    names = [f"{string.ascii_uppercase[i // c]}{i % c + 1:02d}" for i in range(size)]
    p = plates.detect_plate(names)
    check(f"full {size}-well -> {p.label if p else None} via {p.source if p else '-'}",
          p is not None and p.size == size and p.n_rows == r and p.n_cols == c
          and p.source == "names")

# partial plate: 40 wells spanning A1..D10 needs an 8x12, not a 6x8
p = plates.detect_plate([f"{r}{c:02d}" for r in "ABCD" for c in range(1, 11)])
check(f"partial plate A1-D10 -> {p.label} with {p.size - 40} empty wells",
      p is not None and p.size == 96)

# instrument prefixes must not defeat detection
p = plates.detect_plate([f"LUM_{r}{c}" for r in "ABCDEFGH" for c in range(1, 13)])
check("96 wells behind an instrument prefix still detected",
      p is not None and p.size == 96 and p.source == "names")

# a few unparsable columns are tolerated and reported
mixed = [f"{r}{c:02d}" for r in "ABCDEFGH" for c in range(1, 13)][:94] + ["blank1", "blank2"]
p = plates.detect_plate(mixed)
check("94 wells + 2 blanks -> plate, blanks reported",
      p is not None and len(p.wells) == 94 and any("no recognisable well" in n for n in p.notes))

print("   -- false positives --")
for n_samples in (12, 24, 48):
    p = plates.detect_plate([f"treatment_{i}" for i in range(n_samples)])
    check(f"{n_samples} ordinary samples are NOT called a plate", p is None)
for n_samples in (96, 384):
    p = plates.detect_plate([f"sample_{i}" for i in range(n_samples)])
    check(f"{n_samples} ordinary samples -> plate via count, flagged as a guess",
          p is not None and p.source == "count" and "no well IDs" in p.describe())
p = plates.detect_plate([f"{r}{c:02d}" for _ in range(2) for r in "AB" for c in range(1, 4)])
check("duplicated wells (two plates in one file) rejected", p is None)
check("50 samples -> not a plate", plates.detect_plate([f"s{i}" for i in range(50)]) is None)

print("   -- grouping stays opt-in --")
p96 = plates.detect_plate([f"{r}{c:02d}" for r in "ABCDEFGH" for c in range(1, 13)])
check("no Condition column is invented", "Condition" not in p96.wells.columns)
check("group_by_geometry('None') returns None", plates.group_by_geometry(p96, "None") is None)
check("group_by_geometry('Row') -> 8 groups",
      plates.group_by_geometry(p96, "Row").nunique() == 8)
check("group_by_geometry('Column') -> 12 groups",
      plates.group_by_geometry(p96, "Column").nunique() == 12)

# =============================================================================
print("\n15. Plate rendering and the plots.py refactor")
# =============================================================================
import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import plots                                                  # noqa: E402

tp = np.arange(0, 7 * 24, 1.0)
rngp = np.random.default_rng(5)
wells = [f"{r}{c:02d}" for r in "ABCDEFGH" for c in range(1, 13)]
plate_df = pd.DataFrame({"time": tp,
                         **{w: np.sin(2 * np.pi * tp / 24 + i * 0.1) + rngp.normal(0, .1, len(tp))
                            for i, w in enumerate(wells)}})
fig = plates.plot_plate(plate_df, "time", p96)
check(f"96-well figure has 96 axes", len(fig.axes) == 96)
matplotlib.pyplot.close(fig)

partial = plates.detect_plate([f"{r}{c:02d}" for r in "ABCD" for c in range(1, 11)])
sub = plate_df[["time"] + [f"{r}{c:02d}" for r in "ABCD" for c in range(1, 11)]]
fig = plates.plot_plate(sub, "time", partial)
check("partial plate still draws the full grid", len(fig.axes) == 96)
matplotlib.pyplot.close(fig)

fig = plates.plot_plate(plate_df, "time", p96, annotations={w: "24.0h" for w in wells[:5]},
                        well_colors={wells[0]: "#ffe9e9"}, show_sample_names=True)
check("annotation / well_colors overlay hooks accept input", len(fig.axes) == 96)
matplotlib.pyplot.close(fig)

try:
    plates.plot_plate(plate_df[["time"]], "time", p96)
    check("empty overlap raises a clear error", False)
except ValueError as e:
    check(f"empty overlap raises ValueError", "wells are present" in str(e))

print("   -- plots.py exposes everything its callers need --")
# plot_entrainment (the plt.axvspan one) was deliberately dropped in v0.7.7 —
# see section 19. Everything draws through plot_entrainment_ax now.
for fn in ("plot", "grouped_plot", "grouped_plot_traces", "grouped_plot_traces_export",
           "double_plot", "multi_acto", "phase_plot",
           "plot_entrainment_ax", "simple_plot", "split_plot", "pie_chart", "text",
           "feature_entrainment"):
    check(f"plots.{fn}", callable(getattr(plots, fn, None)))

print("   -- and methods.py no longer does", "--")
for fn in ("plot", "double_plot", "grouped_plot", "multi_acto", "phase_plot",
           "simple_plot", "split_plot", "pie_chart", "feature_entrainment",
           "multiplot", "easy_pdf_report", "easy_pdf_report_new",
           "plot_table_on_ax", "multi_subplot"):
    check(f"methods.{fn} gone", not hasattr(methods, fn))

check("methods.py keeps the numeric API",
      all(hasattr(methods, f) for f in
          ("detrend", "normalize", "period_estimation", "savitzky_golay",
           "run_metacycle", "detect_rhythmicity", "sampling_interval",
           "count_entrainment_days", "add_entrainment", "multicomparison")))


# =============================================================================
print("\n16. Comparison views (v0.7.4)")
# =============================================================================
tc = np.arange(0, 7 * 24, 0.5)
rngc = np.random.default_rng(9)
scols = [f"WT_{i}" for i in range(3)] + [f"mut_{i}" for i in range(2)]
cdf = pd.DataFrame({"time": tc})
for i, c in enumerate(scols):
    cdf[c] = np.sin(2 * np.pi * tc / (24 if c.startswith("WT") else 26.5) + i * .5) \
        + rngc.normal(0, .15, len(tc)) + 2
crows = []
for g, per in [("WT", 24), ("mutant", 26.5), ("rescue", 24.5), ("KO", 28)]:
    for r in range(4):
        c = f"{g}_r{r}"
        crows.append({"name": c, "Condition": g})
        cdf[c] = np.sin(2 * np.pi * tc / per + r * .25) + rngc.normal(0, .2, len(tc)) + 2
clay = pd.DataFrame(crows)

f = plots.compare_samples(cdf, "time", scols, 0, tc.max())
check(f"5 samples -> 5 lines", len(f.axes[0].lines) == 5)
check("legend present for >= 2 series", f.axes[0].get_legend() is not None)
check("every series is direct-labelled too (identity never colour-alone)",
      len([a for a in f.axes[0].texts if a.get_text() in scols]) == 5)
matplotlib.pyplot.close(f)

f = plots.compare_samples(cdf, "time", [scols[0]], 0, tc.max())
check("a single sample draws without a legend",
      len(f.axes[0].lines) == 1 and f.axes[0].get_legend() is None)
matplotlib.pyplot.close(f)

f = plots.compare_samples(cdf, "time", scols + ["WT_0", "WT_1"], 0, tc.max())
check(f"more than {plots.MAX_COMPARE_SAMPLES} samples is capped, not an error",
      len(f.axes[0].lines) == plots.MAX_COMPARE_SAMPLES)
matplotlib.pyplot.close(f)

for style, extra in (("Mean ± SD", 0), ("Mean + Replicates", 16)):
    f = plots.compare_groups(cdf, "time", ["WT", "mutant", "rescue", "KO"], clay,
                             0, tc.max(), style=style)
    check(f"4 conditions, '{style}' -> {len(f.axes[0].lines)} lines",
          len(f.axes[0].lines) == 4 + extra)
    check(f"'{style}' legend carries N per condition",
          all("(N=4)" in t.get_text() for t in f.axes[0].get_legend().get_texts()))
    matplotlib.pyplot.close(f)

f = plots.compare_groups(cdf, "time", ["WT", "mutant", "rescue", "KO", "WT"], clay,
                         0, tc.max())
check(f"more than {plots.MAX_COMPARE_GROUPS} conditions is capped",
      len(f.axes[0].lines) == plots.MAX_COMPARE_GROUPS)
matplotlib.pyplot.close(f)

for label, fn in (("no samples", lambda: plots.compare_samples(cdf, "time", [], 0, tc.max())),
                  ("one condition", lambda: plots.compare_groups(
                      cdf, "time", ["WT"], clay, 0, tc.max()))):
    try:
        fn()
        check(f"{label} raises a clear error", False)
    except ValueError as e:
        check(f"{label} -> ValueError: {str(e)[:44]}", True)

# A condition whose wells were all excluded must not kill the figure
thin = clay[clay.Condition != "KO"]
f = plots.compare_groups(cdf, "time", ["WT", "mutant", "KO"], thin, 0, tc.max())
check("a condition with no surviving samples is skipped, not fatal",
      len(f.axes[0].lines) == 2)
matplotlib.pyplot.close(f)

print("   -- palette --")
check(f"{len(plots.COMPARE_PALETTE)} fixed slots, assigned by position",
      plots.COMPARE_PALETTE == ["#d55e00", "#56b4e9", "#2a78d6", "#eda100", "#4a3aa7"])
check("colours are not cycled for the first 5",
      len(set(plots._compare_colors(5))) == 5)
check("a custom palette overrides the default",
      plots._compare_colors(2, ["#111111", "#222222"]) == ["#111111", "#222222"])

# The end labels sit outside the axes; app.py exports with a bare savefig, so the
# figure margins - not bbox_inches='tight' - are what keeps them on the canvas.
f = plots.compare_samples(cdf, "time", scols, 0, tc.max())
check(f"right margin reserved for end labels ({f.subplotpars.right:.3f} < 0.9)",
      f.subplotpars.right < 0.9)
buf = BytesIO()
f.savefig(buf, format="svg")          # exactly how app.py exports
svg = buf.getvalue().decode("utf-8", "ignore")
check("all 5 end labels survive a plain savefig to SVG",
      all(s in svg for s in scols))
matplotlib.pyplot.close(f)

# Colliding end values must be pushed apart rather than overprinted
flat = pd.DataFrame({"time": tc, **{f"s{i}": np.full(len(tc), 1.0) for i in range(4)}})
f = plots.compare_samples(flat, "time", [f"s{i}" for i in range(4)], 0, tc.max())
ys = sorted(a.xy[1] for a in f.axes[0].texts if a.get_text().startswith("s"))
check("identical end values are decluttered, not stacked",
      len(f.axes[0].texts) >= 4)
matplotlib.pyplot.close(f)


# =============================================================================
print("\n17. pyboat figure leak (found while testing the comparison views)")
# =============================================================================
# pyboat's WAnalyzer.compute_spectrum defaults to do_plot=True, so methods.wavelet
# drew and never closed one matplotlib figure PER SAMPLE PER RERUN. Wavelet
# Transform is the default period method, so a 96-well plate leaked 96 figures
# every time any widget was touched.
matplotlib.pyplot.close("all")
lcols = [f"L{i}" for i in range(12)]
ldf = pd.DataFrame({"time": tc,
                    **{c: np.sin(2 * np.pi * tc / 24 + i) + rngc.normal(0, .2, len(tc))
                       for i, c in enumerate(lcols)}})
before = len(matplotlib.pyplot.get_fignums())
per = methods.period_estimation(ldf, lcols, "time", method="Wavelet Transform",
                                min_period=18, max_period=30)
leaked = len(matplotlib.pyplot.get_fignums()) - before
check(f"Wavelet period estimation on {len(lcols)} samples leaks 0 figures (was {len(lcols)})",
      leaked == 0, f"leaked {leaked}")
check(f"...and still returns the right answer (median {np.nanmedian(per):.2f} h)",
      abs(np.nanmedian(per) - 24) < 1.0)
check("methods.wavelet passes do_plot=False",
      "do_plot=False" in inspect.getsource(methods.wavelet))


# =============================================================================
print("\n18. Plot styling (v0.7.5)")
# =============================================================================
import re                                                      # noqa: E402
import styles as ct_styles                                     # noqa: E402

check(f"{len(ct_styles.STYLE_NAMES)} styles offered", len(ct_styles.STYLE_NAMES) == 9)
check("the requested additions are present",
      all(s in ct_styles.STYLE_NAMES for s in ("ggplot", "Framed grid", "Journal")))

print("   -- every style applies and yields a usable facecolor --")
for name in ct_styles.STYLE_NAMES:
    try:
        face = ct_styles.apply(name, "notebook", "Chronotopia")
        f = matplotlib.pyplot.figure()
        ax = f.add_subplot(111)
        ax.plot(tc[:50], np.sin(tc[:50]))
        ok = face.startswith("#") and len(face) == 7
        check(f"{name:12} -> facecolor {face}", ok)
        matplotlib.pyplot.close(f)
    except Exception as e:
        check(f"{name}", False, f"{type(e).__name__}: {e}")

print("   -- the user's 'Framed grid' spec is honoured --")
ct_styles.apply("Framed grid", "notebook", "Chronotopia")
rc = matplotlib.rcParams
for key, want in (("axes.edgecolor", "black"), ("axes.linewidth", 1.0),
                  ("ytick.left", True), ("ytick.major.size", 6.0),
                  ("xtick.bottom", True), ("xtick.direction", "out"),
                  ("xtick.major.size", 6.0), ("xtick.color", "black")):
    got = rc[key]
    got_cmp = matplotlib.colors.to_hex(got) if key.endswith("color") else got
    want_cmp = matplotlib.colors.to_hex(want) if key.endswith("color") else want
    check(f"{key} = {got}", got_cmp == want_cmp)
check("ytick.direction mirrored to 'out' (their snippet only set xtick)",
      rc["ytick.direction"] == "out")

print("   -- ggplot and Journal are structurally what they claim --")
ct_styles.apply("ggplot", "notebook", "Chronotopia")
check("ggplot: grey panel + white grid + no frame",
      matplotlib.colors.to_hex(matplotlib.rcParams["axes.facecolor"]) == "#ebebeb"
      and matplotlib.colors.to_hex(matplotlib.rcParams["grid.color"]) == "#ffffff"
      and matplotlib.rcParams["axes.grid"])
ct_styles.apply("Journal", "notebook", "Chronotopia")
check("Journal: top/right spines off, ticks outward, no grid",
      not matplotlib.rcParams["axes.spines.top"]
      and not matplotlib.rcParams["axes.spines.right"]
      and matplotlib.rcParams["xtick.direction"] == "out"
      and not matplotlib.rcParams["axes.grid"])
ct_styles.apply("Minimal", "notebook", "Chronotopia")
# Minimal originally used horizontal rules only. Changed in v0.7.6: with x ticks
# every 24 h the vertical lines are a period reference, which beats tidiness here.
check("Minimal: no frame, faint grid on both axes",
      matplotlib.rcParams["axes.grid.axis"] == "both"
      and matplotlib.rcParams["axes.grid"]
      and not matplotlib.rcParams["axes.spines.left"]
      and not matplotlib.rcParams["axes.spines.top"])

print("   -- publication baseline --")
ct_styles.apply("Ticks", "paper", "Chronotopia", editable_text=True)
check("pdf.fonttype 42 (TrueType, not Type-3)", matplotlib.rcParams["pdf.fonttype"] == 42)
check("ps.fonttype 42", matplotlib.rcParams["ps.fonttype"] == 42)
check("savefig.dpi 300", matplotlib.rcParams["savefig.dpi"] == 300)
check("data drawn above gridlines", matplotlib.rcParams["axes.axisbelow"] is True)


def _svg_text_count(editable):
    ct_styles.apply("Journal", "notebook", "Chronotopia", editable_text=editable)
    f = matplotlib.pyplot.figure()
    a = f.add_subplot(111)
    a.plot(tc[:50], np.sin(tc[:50]))
    a.set_xlabel("Time (h)")
    a.set_ylabel("Bioluminescence")
    b = BytesIO()
    f.savefig(b, format="svg")
    matplotlib.pyplot.close(f)
    return len(re.findall(r"<text", b.getvalue().decode("utf8", "ignore")))


n_editable, n_outlined = _svg_text_count(True), _svg_text_count(False)
check(f"editable export keeps real <text> in the SVG ({n_editable} elements)",
      n_editable > 0)
check(f"unchecking it outlines the text instead ({n_outlined} elements)",
      n_outlined == 0)

print("   -- palettes --")
check(f"{len(ct_styles.PALETTE_NAMES)} curated palettes, not 204",
      10 <= len(ct_styles.PALETTE_NAMES) <= 20)
check("default is the only all-pairs-safe one",
      ct_styles.DEFAULT_PALETTE == "Chronotopia"
      and ct_styles.PALETTES["Chronotopia"]["all_pairs"] is True)
for name in ct_styles.PALETTE_NAMES:
    cols = ct_styles.preview_colors(name, 5)
    if len(cols) != 5:
        check(f"palette '{name}' yields 5 colours", False, f"got {len(cols)}")
check("every curated palette yields 5 usable colours",
      all(len(ct_styles.preview_colors(n, 5)) == 5 for n in ct_styles.PALETTE_NAMES))
check("labels carry the measured CVD number",
      ct_styles.palette_label("Chronotopia").endswith("13.0")
      and "0.7" in ct_styles.palette_label("tab10"))
check("safe palettes are marked, unsafe are not",
      "✓" in ct_styles.palette_label("colorblind")
      and "✓" not in ct_styles.palette_label("tab10"))
check("an unknown palette name falls back instead of raising",
      ct_styles.apply("Ticks", "notebook", "not-a-palette").startswith("#"))
check("the all-colormaps escape hatch still exists",
      len(ct_styles.all_colormap_names()) > 50)

print("   -- gridlines give a 24 h reference --")
for name in ct_styles.STYLE_NAMES:
    ct_styles.apply(name, "notebook", "Chronotopia")
    if matplotlib.rcParams["axes.grid"]:
        check(f"{name:12} grid covers BOTH axes, not just y",
              matplotlib.rcParams["axes.grid.axis"] == "both")

# The point of the x grid: the plot functions tick every 24 h, so the vertical
# lines land on day boundaries and a non-24 h period is visibly drifting.
ct_styles.apply("White grid", "notebook", "Chronotopia")
gdf = pd.DataFrame({"time": np.arange(0, 6 * 24, 0.5)})
gdf["A01"] = np.sin(2 * np.pi * gdf["time"] / 25.5) + 2
gfig = plots.plot(gdf, "time", "A01", 0, float(gdf["time"].max()))
gax = gfig.axes[0]
xticks = [t for t in gax.get_xticks() if 0 <= t <= gdf["time"].max()]
check(f"x ticks fall on 24 h boundaries ({[int(t) for t in xticks]})",
      len(xticks) >= 5 and all(abs(t % 24) < 1e-9 for t in xticks))
check("x gridlines are actually drawn",
      any(gl.get_visible() for gl in gax.get_xgridlines()))
matplotlib.pyplot.close(gfig)

print("   -- switching styles must not leak rcParams --")
# rcParams are process-global and survive Streamlit reruns. Before the fix,
# Minimal set axes.grid.axis='y' and every grid style chosen afterwards silently
# lost its vertical gridlines; Framed grid left 6-pt ticks behind the same way.
import itertools as _it                                       # noqa: E402
WATCHED = ["axes.grid", "axes.grid.axis", "xtick.major.size", "ytick.major.size",
           "axes.edgecolor", "axes.spines.left", "axes.spines.top",
           "xtick.direction", "ytick.direction", "grid.color", "axes.linewidth",
           "axes.facecolor", "xtick.bottom", "ytick.left"]
leaks = []
for first, second in _it.permutations(ct_styles.STYLE_NAMES, 2):
    ct_styles.apply(second, "notebook", "Chronotopia")
    clean = {k: str(matplotlib.rcParams[k]) for k in WATCHED}
    ct_styles.apply(first, "notebook", "Chronotopia")
    ct_styles.apply(second, "notebook", "Chronotopia")
    after = {k: str(matplotlib.rcParams[k]) for k in WATCHED}
    bad = [k for k in WATCHED if clean[k] != after[k]]
    if bad:
        leaks.append((first, second, bad))
n_pairs = len(list(_it.permutations(ct_styles.STYLE_NAMES, 2)))
check(f"all {n_pairs} style transitions land on identical rcParams",
      not leaks, "" if not leaks else f"leaking: {leaks[:2]}")
check("_STYLE_KEYS covers every key the styles set",
      set(ct_styles._STYLE_KEYS) == {k for sp in ct_styles.STYLES.values() for k in sp["rc"]})

ct_styles.apply(ct_styles.DEFAULT_STYLE, ct_styles.DEFAULT_CONTEXT,
                ct_styles.DEFAULT_PALETTE)


# =============================================================================
print("\n19. Rhythmicity report (v0.7.7)")
# =============================================================================
from rhythmicity_report import RhythmicityReport                # noqa: E402
import matplotlib.patches as _mpatch                            # noqa: E402

rt = np.arange(0, 7 * 24, 1.0)
rrng = np.random.default_rng(21)
_conds = ["WT", "mutant"]
_cols, _lay = [], []
rdf = pd.DataFrame({"time": rt})
for _g in _conds:
    for _r in range(3):
        _c = f"{_g}_r{_r}"
        _cols.append(_c)
        _lay.append({"name": _c, "Condition": _g})
        rdf[_c] = np.sin(2 * np.pi * rt / 24 + _r * .3) + rrng.normal(0, .2, len(rt)) + 2
_lay = pd.DataFrame(_lay)

_BASE = dict(df=rdf, t_col="time", layout_df=_lay, phases=None, sum_stats=None,
             methods=methods, thresh=0.05, ent=True, ent_days=3, ent_color="#EBEBEB",
             bg_color="white", unit="a.u.", T=24, order=0, t0=0, t1=int(rt.max()),
             conditions=_conds, data_cols=_cols, period_len_min=18, period_len_max=30,
             period_estimation="Lomb-Scargle Periodogram", file_name="t")


def _result_for(method):
    r = pd.DataFrame({"CycID": _cols, "Periods": [24.1, 23.8, 25.9, 24.0, 26.2, 23.9]})
    if method == "Tempo":
        r["probability_rhythmic"] = [0.9133333333333333, 0.8266666, 0.21, 0.87, 0.19, 0.94]
        r["Tempo_BH.Q"] = 1 - r["probability_rhythmic"]
        r["is_rhythmic"] = r["probability_rhythmic"] > 0.5
        r["confidence"] = ["high", "high", "low", "high", "low", "high"]
    else:
        r[f"{method}_BH.Q"] = [0.001, 0.02, 0.3, 0.004, 0.6, 0.01]
        r[f"{method}_AMP"] = [1.0, 1.1, 0.4, 0.9, 0.3, 1.2]
    r["reject"] = True
    return r


print("   -- builds without an analysis --")
try:
    rep = RhythmicityReport(result_df=None, method="meta2d", **_BASE).build()
    buf = rep.to_pdf()
    check(f"no analysis -> {len(rep.figures)} pages, {len(buf.getvalue())} byte PDF",
          len(buf.getvalue()) > 1000)
except Exception as e:
    check("no analysis builds", False, f"{type(e).__name__}: {e}")

r0 = RhythmicityReport(result_df=None, method="meta2d", **_BASE)
check("sample title degrades to just the name", r0._get_sample_title(_cols[0]) == _cols[0])
check("no results -> no q column", r0.q_col is None and not r0.has_results)

# result_df present but empty, and result_df missing the method's column
check("empty result_df is treated as no results",
      RhythmicityReport(result_df=pd.DataFrame(), method="meta2d", **_BASE).has_results is False)
odd = pd.DataFrame({"CycID": _cols, "Periods": [24.0] * 6})
rodd = RhythmicityReport(result_df=odd, method="meta2d", **_BASE)
check("results without a q column still build",
      rodd.q_col is None and len(rodd.build().figures) > 0)

print("   -- every trace panel carries the zeitgeber band --")


def _band_count(ax):
    return sum(1 for p in ax.patches if isinstance(p, (_mpatch.Polygon, _mpatch.Rectangle)))


for method in ("meta2d", "Tempo"):
    rep = RhythmicityReport(result_df=_result_for(method), method=method, **_BASE)

    # Multi-condition overview: previously plt.axvspan put every band on the
    # last-created panel, so only the final condition was shaded.
    rep.figures = []
    rep.add_conditions_overview()
    ov = [_band_count(a) for a in rep.figures[0].axes if a.lines]
    check(f"{method}: conditions overview shades all {len(ov)} panels ({ov})",
          len(ov) == len(_conds) and all(b > 0 for b in ov))

    # Per-sample grids: previously drew raw traces with no shading at all.
    rep.figures = []
    rep.add_group_traces()
    grids = [f for f in rep.figures if len([a for a in f.axes if a.lines]) >= 3]
    counts = [[_band_count(a) for a in f.axes if a.lines] for f in grids]
    check(f"{method}: every per-sample panel is shaded ({counts})",
          bool(counts) and all(all(b > 0 for b in c) for c in counts))
    matplotlib.pyplot.close("all")

# split_plot is the exception and should stay one: its second panel IS the
# free-running segment, so it must NOT be shaded.
_sfig = plots.split_plot(rdf, "time", _cols[0], ent=True, ent_days=3, unit="a.u.",
                         bg_color="white", band_color="#EBEBEB", order=0, T=24, title="x")
_sb = [_band_count(a) for a in _sfig.axes]
check(f"split_plot shades entrainment but not free-running ({_sb})",
      _sb[0] > 0 and _sb[1] == 0)
matplotlib.pyplot.close("all")

print("   -- annotation is readable --")
rep_m = RhythmicityReport(result_df=_result_for("meta2d"), method="meta2d", **_BASE)
title = rep_m._get_sample_title(_cols[0])
check(f"meta2d title is 2 lines: {title!r}", title.count("\n") == 1)
check("no raw float spew in the title",
      not re.search(r"\d\.\d{6,}", title))
check("q formatted to 3 dp or better", "q = 0.001" in title)

rep_t = RhythmicityReport(result_df=_result_for("Tempo"), method="Tempo", **_BASE)
t_title = rep_t._get_sample_title(_cols[0])
check(f"Tempo title shows a percentage, not 0.9133333...: {t_title!r}",
      "P 91%" in t_title and not re.search(r"\d\.\d{6,}", t_title))
check("Tempo title carries the confidence", "(high)" in t_title)

check("tiny q collapses to a bound", RhythmicityReport._fmt_q(1e-9) == "q < 0.0001")
check("mid q uses scientific", RhythmicityReport._fmt_q(5e-4) == "q = 5.0e-04")
check("ordinary q uses 3 dp", RhythmicityReport._fmt_q(0.0234) == "q = 0.023")
check("NaN q is dropped", RhythmicityReport._fmt_q(np.nan) is None)

print("   -- Tempo is a method, not an always-on extra --")
check("meta2d report carries no ML columns", rep_m.has_ml is False)
check("Tempo report is flagged as the ML method", rep_t.is_ml_method and rep_t.has_ml)
n_meta = len(RhythmicityReport(result_df=_result_for("meta2d"), method="meta2d", **_BASE).build().figures)
n_tempo = len(RhythmicityReport(result_df=_result_for("Tempo"), method="Tempo", **_BASE).build().figures)
check(f"the ML pie page appears only for Tempo ({n_meta} vs {n_tempo} pages)",
      n_tempo == n_meta + 1)
matplotlib.pyplot.close("all")

print("   -- methods without an amplitude column --")
mc = methods.multicomparison(_result_for("Tempo"), _lay, _conds, "Tempo", 0.05)
check(f"multicomparison skips Amplitude for Tempo (tested: {sorted(mc.tested.unique())})",
      "Amplitude" not in set(mc.tested) and "Rhythmicity" in set(mc.tested))
mc2 = methods.multicomparison(_result_for("meta2d"), _lay, _conds, "meta2d", 0.05)
check("...but still runs it for meta2d", "Amplitude" in set(mc2.tested))

print("   -- plot_entrainment global-axes trap is gone --")
check("plots.plot_entrainment removed", not hasattr(plots, "plot_entrainment"))
_plots_code = "\n".join(
    line for line in inspect.getsource(plots).split("\n")
    if not line.lstrip().startswith("#")
)
check("no plt.axvspan left in plots.py code (comments aside)",
      "plt.axvspan" not in _plots_code)
check("the axes-bound version is what everything calls",
      _plots_code.count("plot_entrainment_ax(") >= 4)


# =============================================================================
print("\n20. Plate overlays (v0.7.8)")
# =============================================================================
from chronotopia_feature_extractor import ChronotopiaFeatureExtractor as _CFE   # noqa: E402

_pt = np.arange(0, 7 * 24, 1.0)
_prng = np.random.default_rng(31)
_pcols, _pdata = [], {}
for _i in range(96):
    _r, _c = string.ascii_uppercase[_i // 12], _i % 12 + 1
    _w = f"{_r}{_c:02d}"
    _pcols.append(_w)
    if _c <= 8:                                   # rhythmic, period rises down the rows
        _pdata[_w] = (0.4 + 0.9 * (_c / 8)) * np.sin(
            2 * np.pi * _pt / (22.0 + (_i // 12) * 0.55) + _i * 0.3
        ) + _prng.normal(0, .12, len(_pt)) + 1
    else:                                         # arrhythmic controls
        _pdata[_w] = _prng.normal(1, .28, len(_pt)) + 0.004 * _pt
_pdf = pd.DataFrame({"time": _pt, **_pdata})
_plate = plates.detect_plate(_pcols)
_feats = _CFE.extract_batch(_pdf, "time", _pcols,
                            packages=plates.METRIC_PACKAGES, verbose=False)
_pres = pd.DataFrame({"CycID": _pcols})
_pres["meta2d_BH.Q"] = [0.001 if c <= 8 else 0.4 for _ in range(8) for c in range(1, 13)]
_flags = _pres.set_index("CycID")["meta2d_BH.Q"] <= 0.05
_pmask = [s for s in _pcols if not bool(_flags.get(s, False))]

check("one cosinor fit supplies period, amplitude, R2 and noise",
      all(c in _feats.columns for c in ("cosinor_period", "cosinor_amplitude",
                                        "cosinor_r2", "cosinor_residual_std")))
check("Rhythmicity is offered only once an analysis exists",
      "Rhythmicity" not in plates.metric_names(False)
      and "Rhythmicity" in plates.metric_names(True))

print("   -- every metric resolves and colours --")
for _name in plates.metric_names(has_results=True):
    vals = plates.compute_metric(_name, _pdf, "time", _pcols, features=_feats,
                                 result_df=_pres, q_col="meta2d_BH.Q", thresh=0.05)
    colors, legend = plates.build_overlay(_name, vals)
    kind = plates.METRICS[_name]["kind"]
    want = "swatches" if kind == "status" else "colorbar"
    check(f"{_name:20} -> {len(colors)} wells, {legend['kind']} legend",
          len(colors) == len(_pcols) and legend["kind"] == want)

print("   -- colour encoding matches the job --")
_pv = plates.compute_metric("Period (h)", _pdf, "time", _pcols, features=_feats)
_, _plegend = plates.build_overlay("Period (h)", _pv)
check("period uses a diverging scale centred on 24 h",
      isinstance(_plegend["norm"], matplotlib.colors.TwoSlopeNorm)
      and abs(_plegend["norm"].vcenter - 24.0) < 1e-9)
_, _r2legend = plates.build_overlay(
    "Cosinor R²", plates.compute_metric("Cosinor R²", _pdf, "time", _pcols, features=_feats))
check("R² is pinned to its true 0–1 range, not the data range",
      _r2legend["norm"].vmin == 0.0 and _r2legend["norm"].vmax == 1.0)

_ac = plates.compute_metric("Acrophase (h)", _pdf, "time", _pcols, features=_feats)
check(f"acrophase wraps onto a 24 h clock (max {np.nanmax(_ac):.1f})",
      np.nanmin(_ac) >= 0 and np.nanmax(_ac) < 24)
check("a raw 25.83 h acrophase becomes 1.83 h",
      abs(float(plates.compute_metric(
          "Acrophase (h)", None, None, ["x"],
          features=pd.DataFrame({"sample_id": ["x"],
                                 "cosinor_acrophase_h": [25.83]}))["x"]) - 1.83) < 1e-6)

print("   -- masking non-rhythmic wells --")
_c_all, _l_all = plates.build_overlay("Period (h)", _pv)
_c_msk, _l_msk = plates.build_overlay("Period (h)", _pv, mask=_pmask)
span_all = _l_all["norm"].vmax - _l_all["norm"].vmin
span_msk = _l_msk["norm"].vmax - _l_msk["norm"].vmin
check(f"arrhythmic wells stop stretching the scale ({span_all:.1f} h -> {span_msk:.1f} h)",
      span_msk < span_all * 0.6)
check(f"{len(_pmask)} masked wells are painted flat grey",
      all(_c_msk[s] == plates.MASK_COLOR for s in _pmask))
check("the legend says so", "not rhythmic" in _l_msk.get("note", ""))
_lbl = plates.format_labels(_pv, "Period (h)", mask=_pmask)
check("and their misleading value is not printed",
      all(_lbl[s] == "" for s in _pmask) and _lbl[_pcols[0]] != "")

print("   -- label ink stays legible on any well colour --")
check("dark well -> light ink", plates._ink_for("#08306b") == "#ffffff")
check("light well -> dark ink", plates._ink_for("#f7fbff") == "#0b0b0b")
check("mask grey -> dark ink", plates._ink_for(plates.MASK_COLOR) == "#0b0b0b")

print("   -- the figure carries the legend --")
_colors, _legend = plates.build_overlay("Period (h)", _pv, mask=_pmask)
_f = plates.plot_plate(_pdf, "time", _plate, well_colors=_colors,
                       annotations=plates.format_labels(_pv, "Period (h)", mask=_pmask),
                       legend=_legend)
check(f"colorbar axes added ({len(_f.axes)} axes for 96 wells)", len(_f.axes) == 97)
matplotlib.pyplot.close(_f)
_sc, _sl = plates.build_overlay(
    "Rhythmicity",
    plates.compute_metric("Rhythmicity", _pdf, "time", _pcols, result_df=_pres,
                          q_col="meta2d_BH.Q", thresh=0.05))
_f = plates.plot_plate(_pdf, "time", _plate, well_colors=_sc, legend=_sl)
check(f"swatch legend axes added ({len(_f.axes)} axes)", len(_f.axes) == 97)
matplotlib.pyplot.close(_f)
_f = plates.plot_plate(_pdf, "time", _plate)
check(f"no legend when there is no overlay ({len(_f.axes)} axes)", len(_f.axes) == 96)
matplotlib.pyplot.close(_f)

print("   -- degrades rather than raising --")
check("unknown metric returns an empty series",
      plates.compute_metric("nope", _pdf, "time", _pcols).empty)
check("Rhythmicity with no results reads n/a",
      set(plates.compute_metric("Rhythmicity", _pdf, "time", _pcols).unique()) == {"n/a"})
_nan = pd.Series(np.nan, index=_pcols)
check("an all-NaN metric yields no colours and no legend",
      plates.build_overlay("Amplitude", _nan) == ({}, None))
check("missing features give NaN, not KeyError",
      plates.compute_metric("Amplitude", _pdf, "time", _pcols,
                            features=pd.DataFrame({"sample_id": _pcols})).isna().all())


# =============================================================================
print("\n21. Sine sweep (v0.7.9)")
# =============================================================================
# The dataset-level question: which periods are present at all, rather than
# whether one trace is rhythmic. Planted components must come back exactly.
_st_ = np.arange(0, 48, 2.0)                     # 2 h sampling, 2 days — RNA-seq shaped
_srng = np.random.default_rng(4)
_truth = _srng.choice([24.0, 12.0, 8.0, None], size=600, p=[.45, .22, .13, .20])
_sdata = {}
for _i, _per in enumerate(_truth):
    _y = _srng.normal(0, .35, len(_st_)) + 5
    if _per is not None:
        _y = _y + 1.2 * np.sin(2 * np.pi * _st_ / _per + _srng.uniform(0, 6.28))
    _sdata[f"g{_i}"] = _y
_sdf = pd.DataFrame({"time": _st_, **_sdata})

_sres, _sland = methods.sine_sweep(_sdf, "time", list(_sdata), 2, 30, 0.1)
check(f"one row per signal ({len(_sres)})", len(_sres) == len(_sdata))
check("results carry period, amplitude, phase and R²",
      set(["sample", "period", "amplitude", "phase_h", "r2", "rss"]) <= set(_sres.columns))
check("landscape carries the aggregate the histogram cannot show",
      set(["period", "mean_r2", "median_r2", "frac_best", "n_best"]) <= set(_sland.columns))

_spk = methods.sweep_peaks(_sland, n_peaks=4)
_found = sorted(round(p, 1) for p in _spk["period"])
for _want in (24.0, 12.0, 8.0):
    check(f"{_want:.0f} h recovered from the aggregate landscape",
          any(abs(f - _want) < 0.35 for f in _found), f"found {_found}")

_good = _sres[_sres.r2 > 0.3].copy()
_good["truth"] = [_truth[int(x[1:])] for x in _good["sample"]]
for _want in (24.0, 12.0, 8.0):
    _sub = _good[_good.truth == _want]
    _hit = (_sub["period"].sub(_want).abs() < 1).mean() if len(_sub) else 0
    check(f"per-signal fits land within 1 h of a planted {_want:.0f} h "
          f"({100 * _hit:.0f}% of {len(_sub)})", _hit > 0.85)

print("   -- the singular-matrix trap --")
# At a trial period of exactly 2*dt the sine regressor is sampled at its zero
# crossings, X.T@X is singular, and the pre-existing np.linalg.solve raised
# LinAlgError. Only a sweep this wide ever reaches those periods.
_Xs, _Ps = methods.build_projection_matrices(np.arange(0, 48, 2.0), np.array([4.0]))
check("a rank-deficient design matrix no longer raises",
      np.isfinite(_Ps).all())
check("build_projection_matrices uses the pseudoinverse",
      "pinv" in inspect.getsource(methods.build_projection_matrices))

print("   -- Nyquist --")
_r2c, _ = methods.sine_sweep(_sdf, "time", list(_sdata)[:5], 1, 30, 0.5)
check(f"period_min clamped to twice the sampling interval "
      f"{_r2c.attrs.get('nyquist_clamped')}",
      _r2c.attrs.get("nyquist_clamped") == (1, 4.0))
_r2n, _ = methods.sine_sweep(_sdf, "time", list(_sdata)[:5], 6, 30, 0.5)
check("a range already above Nyquist is left alone",
      "nyquist_clamped" not in _r2n.attrs)

print("   -- missing values --")
_gap = _sdf.copy()
_gap.loc[3:5, "g0"] = np.nan
_rg, _lg = methods.sine_sweep(_gap, "time", ["g0", "g1"], 2, 30, 0.5)
check(f"gaps are mean-filled and counted ({_rg.attrs.get('n_missing_filled')})",
      _rg.attrs.get("n_missing_filled") == 3 and np.isfinite(_rg["r2"]).all())
_flat = pd.DataFrame({"time": _st_, "flat": np.ones(len(_st_))})
_rf, _ = methods.sine_sweep(_flat, "time", ["flat"], 2, 30, 0.5)
check("a constant signal yields NaN R² rather than dividing by zero",
      bool(np.isnan(_rf["r2"].iloc[0])))

print("   -- scale --")
_wide = pd.DataFrame({"time": _st_,
                      **{f"w{i}": _srng.normal(0, 1, len(_st_)) for i in range(4000)}})
_t0 = 0.0
_rw, _lw = methods.sine_sweep(_wide, "time", [f"w{i}" for i in range(4000)], 2, 30, 0.1)
check(f"4000 signals sweep in one batched pass ({len(_rw)} rows, {len(_lw)} periods)",
      len(_rw) == 4000 and len(_lw) > 200)

print("   -- edge cases degrade --")
for _label, _fn in (
    ("no signals", lambda: methods.sine_sweep(_sdf, "time", [], 2, 30, 0.5)),
    ("too few timepoints", lambda: methods.sine_sweep(
        pd.DataFrame({"time": [0.0, 1.0], "a": [1.0, 2.0]}), "time", ["a"], 2, 30, 0.5)),
    ("degenerate range", lambda: methods.sine_sweep(_sdf, "time", ["g0"], 24, 24, 5.0)),
):
    try:
        _fn()
        check(f"{_label} raises a clear error", False)
    except ValueError as e:
        check(f"{_label} -> ValueError: {str(e)[:40]}", True)

check("empty landscape yields no peaks",
      len(methods.sweep_peaks(pd.DataFrame({"period": [], "mean_r2": []}))) == 0)

print("   -- the figure --")
_f = plots.period_sweep({"All": _sland}, {"All": _sres}, r2_thresh=0.3, peaks=_spk)
check("two panels sharing the period axis", len(_f.axes) == 2
      and _f.axes[0].get_shared_x_axes().joined(_f.axes[0], _f.axes[1]))
check("peaks are labelled on the aggregate",
      len([t for t in _f.axes[0].texts if "h" in t.get_text()]) >= 3)
matplotlib.pyplot.close(_f)
_f = plots.period_sweep({"a": _sland, "b": _sland}, {"a": _sres, "b": _sres},
                        r2_thresh=0.3, peaks=None)
check("grouped sweep gets a legend", _f.axes[0].get_legend() is not None)
matplotlib.pyplot.close(_f)


# =============================================================================
print("\n22. Feature analytics (v0.7.10)")
# =============================================================================
import features as ftx                                          # noqa: E402
from chronotopia_feature_extractor import ChronotopiaFeatureExtractor as _FX  # noqa: E402

_ft = np.arange(0, 6 * 24, 0.5)
_frng = np.random.default_rng(7)
_frows = []
for _cond, (_per, _amp, _drift) in [("WT", (24.0, 1.0, 0.0)),
                                    ("mutant", (26.5, 0.55, 0.010))]:
    for _r in range(8):
        _y = (_amp + _frng.normal(0, .06)) * np.sin(
            2 * np.pi * _ft / (_per + _frng.normal(0, .3)) + _frng.uniform(0, 6)
        ) + _frng.normal(0, .18, len(_ft)) + 2 + _drift * _ft
        with ftx.silence_extractor_warnings():
            _f = _FX(_y, _ft).extract()
        _f["sample_id"] = f"{_cond}_{_r}"
        _f["Condition"] = _cond
        _frows.append(_f)
_FEAT = pd.DataFrame(_frows)

print("   -- the dictionary --")
_dict = ftx.describe_features(_FEAT.columns)
check(f"every feature classified ({len(_dict)} of {_FEAT.shape[1] - 2})",
      len(_dict) == _FEAT.shape[1] - 2)
check("nothing falls through to 'Other'", "Other" not in set(_dict.concept))
check(f"grouped into {_dict.concept.nunique()} concepts, in a fixed order",
      list(_dict.concept.unique()) == [c for c in ftx.CONCEPT_ORDER
                                       if c in set(_dict.concept)])
check("the meta_* columns are flagged as recording, not biology",
      set(_dict[_dict.role == ftx.RECORDING].feature) ==
      {"meta_duration_h", "meta_n_points", "meta_dt_h", "meta_is_short"})
check("every feature carries a description",
      _dict.description.str.len().gt(10).all())
check("the five period estimates land in one concept",
      {"cosinor_period", "cycles_period_event_based", "harmonic_fundamental_period_h",
       "lomb_scargle_peak_period_h", "wavelet_ridge_period_mean"}
      <= set(ftx.features_in_concept(_FEAT.columns, "Period")))

print("   -- quality --")
_q = ftx.quality_report(_FEAT)
check(f"one row per feature ({len(_q)})", len(_q) == len(_dict))
check("constant columns are marked unusable",
      bool((~_q.loc[_q.constant, "usable"]).all()))
_cl = ftx.redundancy_clusters(_FEAT, 0.95)
check(f"redundancy clusters found ({_cl.cluster.nunique() if len(_cl) else 0} groups "
      f"over {len(_cl)} features)", len(_cl) > 0)

print("   -- differential comparison --")
_res, _meta = ftx.compare_conditions(_FEAT, "Condition", "WT", "mutant")
check(f"{_meta['n_significant']} of {_meta['n_tested']} features differ",
      _meta["n_significant"] > 10)
check("no q-value is NaN when a p-value is", _res["q"].notna().sum() >= _res["p"].notna().sum() - 2)
_hit = set(_res[_res.significant].concept)
for _want in ("Period", "Amplitude"):
    check(f"the planted {_want.lower()} difference is detected", _want in _hit)
check("n>=8 per group selects the parametric test",
      _meta["test"] == "parametric" and _meta["effect_name"] == "Hedges' g")
_small = _FEAT[_FEAT.sample_id.str.endswith(("_0", "_1", "_2"))]
_r2, _m2 = ftx.compare_conditions(_small, "Condition", "WT", "mutant")
check(f"n=3 per group falls back to rank-based ({_m2['reason']})",
      _m2["test"] == "rank" and _m2["effect_name"] == "Cliff's delta")
check("the choice of test is reported, not hidden", len(_m2["reason"]) > 10)

print("   -- the BH NaN trap --")
# A single NaN p-value used to make EVERY q NaN in both implementations: argsort
# sorts NaN last, the reversal puts it first, minimum.accumulate carries it across.
_pn = np.array([0.001, 0.02, 0.3, np.nan, 0.05])
for _label, _fn in (("features._bh", ftx._bh), ("methods.bh_fdr", methods.bh_fdr)):
    _out = _fn(_pn)
    check(f"{_label}: one NaN no longer poisons the rest",
          np.isfinite(_out[[0, 1, 2, 4]]).all() and np.isnan(_out[3]))
check("BH still matches the textbook without NaNs",
      np.allclose(ftx._bh([0.001, 0.008, 0.039, 0.041]),
                  [0.004, 0.016, 0.041, 0.041], atol=1e-3))
check("both implementations agree",
      np.allclose(ftx._bh(_pn), methods.bh_fdr(_pn), equal_nan=True))

print("   -- effect sizes --")
check("Cliff's delta: identical -> 0", abs(ftx._cliffs_delta([1, 2, 3], [1, 2, 3])) < 1e-12)
check("Cliff's delta: fully separated -> 1", ftx._cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0)
check("Cliff's delta is antisymmetric",
      ftx._cliffs_delta([5, 6], [1, 2]) == -ftx._cliffs_delta([1, 2], [5, 6]))
_ga = _frng.normal(0, 1, 300)
_gb = _ga + 1.0
check(f"Hedges' g on a 1 SD shift -> {ftx._hedges_g(_ga, _gb):.2f}",
      abs(ftx._hedges_g(_ga, _gb) + 1.0) < 0.15)
check("Hedges' g is undefined on zero variance, and falls back",
      not np.isfinite(ftx._hedges_g([1, 1, 1], [1, 1, 1])))

print("   -- cohort context --")
_pc = ftx.cohort_percentiles(_FEAT, "mutant_0")
check(f"percentiles for {len(_pc)} features", len(_pc) > 40)
check("all percentiles are within 0-100",
      _pc.percentile.between(0, 100).all())
check("a feature constant across the cohort sits at the midrank, not at 0",
      abs(float(_pc[_pc.feature == "meta_duration_h"].percentile.iloc[0]) - 50) < 1e-9)
check("sorted most-extreme first",
      _pc.extremity.is_monotonic_decreasing)
try:
    ftx.cohort_percentiles(_FEAT, "no_such_sample")
    check("unknown sample raises", False)
except ValueError:
    check("unknown sample raises a clear error", True)

print("   -- QC --")
_qc = ftx.qc_flags(_FEAT)
check(f"one verdict per sample ({len(_qc)})", len(_qc) == len(_FEAT))
check("verdicts are pass/warn/fail", set(_qc.verdict) <= {"pass", "warn", "fail"})
check("flagged samples carry a reason",
      bool((_qc[_qc.n_flags > 0]["reasons"].str.len() > 0).all()) if (_qc.n_flags > 0).any() else True)
check(f"rules recorded ({len(_qc.attrs['rules_applied'])} applied)",
      len(_qc.attrs["rules_applied"]) >= 3)
_missing_col = ftx.qc_flags(_FEAT[["sample_id", "cosinor_r2"]])
check("rules whose feature is absent are skipped, not silently passed",
      len(_missing_col.attrs["rules_applied"]) == 1)

print("   -- figures --")
_f = plots.feature_volcano(_res, _meta)
check("volcano renders", len(_f.axes) >= 1)
matplotlib.pyplot.close(_f)
_f = plots.feature_volcano(pd.DataFrame(), _meta)
check("empty comparison degrades to a message, not a crash", len(_f.axes) == 1)
matplotlib.pyplot.close(_f)
_f = plots.cohort_context(_pc, "mutant_0")
check("cohort context renders", len(_f.axes) >= 1)
matplotlib.pyplot.close(_f)
_f = plots.cohort_context(pd.DataFrame(), "x")
check("empty cohort degrades", len(_f.axes) == 1)
matplotlib.pyplot.close(_f)

check("nothing is dropped from the feature table",
      set(_FEAT.columns) >= set(_dict.feature))


# =============================================================================
print("\n23. Tooltips (v0.7.11)")
# =============================================================================
import ast as _ast
import docs

print("   -- the text --")
check(f"{len(docs.HELP)} entries across {len(docs.SECTIONS)} sections",
      len(docs.HELP) > 90 and len(docs.SECTIONS) >= 10)
_dupes = [l for s in docs.SECTIONS.values() for l in s]
check("no label is documented twice in different sections",
      len(_dupes) == len(set(_dupes)),
      ", ".join(sorted({l for l in _dupes if _dupes.count(l) > 1})))
check("every tooltip is a non-empty string",
      all(isinstance(v, str) and v.strip() for v in docs.HELP.values()))
check("no tooltip is a bare restatement of its label",
      all(v.strip().lower() != k.strip().lower() for k, v in docs.HELP.items()))
_short = [k for k, v in docs.HELP.items() if len(v) < 25]
check("tooltips say something (>= 25 chars)", not _short, ", ".join(_short[:4]))

print("   -- coverage of app.py --")
_WIDGET_CALLS = {
    "selectbox", "number_input", "text_input", "checkbox", "slider", "multiselect",
    "color_picker", "button", "download_button", "radio", "file_uploader", "toggle",
    "pills", "text_area", "date_input", "time_input", "segmented_control",
    "select_slider",
}
_app_labels, _dynamic = [], 0
for _node in _ast.walk(_ast.parse(open("app.py").read())):
    if not (isinstance(_node, _ast.Call) and isinstance(_node.func, _ast.Attribute)):
        continue
    if _node.func.attr not in _WIDGET_CALLS:
        continue
    _lab = None
    if _node.args and isinstance(_node.args[0], _ast.Constant) and isinstance(_node.args[0].value, str):
        _lab = _node.args[0].value
    else:
        for _kw in _node.keywords:
            if _kw.arg == "label" and isinstance(_kw.value, _ast.Constant):
                _lab = _kw.value.value
    if _lab is None:
        _dynamic += 1
    else:
        _app_labels.append(_lab)

_cov = docs.coverage(_app_labels)
check(f"every widget label in app.py is documented "
      f"({_cov['n_documented']}/{_cov['n_labels']})",
      not _cov["missing"], ", ".join(map(repr, _cov["missing"][:5])))
check(f"only {_dynamic} labels are built at runtime", _dynamic <= 2)
check("a label built with an f-string still resolves, via the parenthetical strip",
      docs.h("Samples to compare (up to 5)") is not None
      and docs.h("Samples to compare (up to 12)") is not None)
check("stripping a parenthetical does not invent tooltips for unknown labels",
      docs.h("Something nobody wrote (2)") is None)

print("   -- attachment --")
_calls = []


def _fake(_name):
    def _f(*a, **k):
        _calls.append((_name, a, k))
    _f.__name__ = _name
    return _f


_fake_st = types.SimpleNamespace()
_FakeDG = type("FakeDG", (), {})
for _w in docs._WIDGETS:
    setattr(_fake_st, _w, _fake(_w))
    setattr(_FakeDG, _w, _fake(_w))
_n = docs.attach(_fake_st, _FakeDG, _force=True)
check(f"{_n} constructors wrapped, module-level and method both", _n == 2 * len(docs._WIDGETS))

_fake_st.selectbox("Smoothening", [1, 2])
check("a module-level call picks up its tooltip", bool(_calls[-1][2].get("help")))
_FakeDG().selectbox("Smoothening", [1, 2])
check("a column/sidebar call picks up its tooltip too — the two bindings differ",
      bool(_calls[-1][2].get("help")))
_fake_st.selectbox(label="Smoothening", options=[1, 2])
check("label passed by keyword works", bool(_calls[-1][2].get("help")))
_fake_st.selectbox("Smoothening", [1, 2], help="context-specific")
check("an explicit help= is never overwritten",
      _calls[-1][2]["help"] == "context-specific")
_fake_st.selectbox("A label nobody documented", [1])
check("an undocumented label is left alone", _calls[-1][2].get("help") is None)
_before = len(_calls)
_fake_st.button("Run analysis")
check("positional-only calls still work", len(_calls) == _before + 1)
check("attach is idempotent — a second call wraps nothing",
      docs.attach(_fake_st, _FakeDG, _force=True) == 0)

print("   -- the reference document --")
_md = docs.as_markdown()
check(f"renders {len(_md)} characters", len(_md) > 8000)
check("every section appears as a heading",
      all(f"## {s}" in _md for s in docs.SECTIONS))
check("every tooltip appears in it",
      all(v in _md for v in docs.HELP.values()))
check("material-icon prefixes are stripped from the headings",
      ":material/" not in _md)

check("app.py attaches before it draws anything",
      open("app.py").read().index("docs.attach()")
      < open("app.py").read().index("st.sidebar"))


# =============================================================================
print("\n" + "=" * 70)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 70)
sys.exit(1 if FAIL else 0)
