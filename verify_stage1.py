"""
Stage-1 verification harness.

Feeds known ground truth (a 24 h sine at several sampling rates) through every
function touched by the Stage-1 fixes, and asserts the answer is right.

Run:  python verify_stage1.py
"""
import sys
import types

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
print("\n11. Savitzky-Golay smoothing (v0.8)")
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
print("\n" + "=" * 70)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 70)
sys.exit(1 if FAIL else 0)
