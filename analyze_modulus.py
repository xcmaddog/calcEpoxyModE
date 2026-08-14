"""
analyze_modulus.py

Extracts the Modulus of Elasticity (E) from a cylindrical compression-test
CSV exported by the MTS TestSuite DAQ, using extensometer strain directly.

Expected test-data file layout (in DATA_DIR, one file per sample) --
this now comes in two flavors and the loader auto-detects which one it's
looking at, by scanning for the header row (the one containing
"Displacement") rather than assuming a fixed number of preamble lines:

  Old format (Test/Test Run/Date preamble):
    line 1: File Path: ...
    line 2: Test: ...
    line 3: Test Run: ...
    line 4: Date: ...
    line 5: (blank)
    line 6: (blank)
    line 7: Running Time, Axial Displacement, Axial Force, Axial AI1_AOX
    line 8: sec, mm, N, %                                    <- units
    line 9+: data

  New format (no preamble):
    line 1: Time, Displacement, Force, Force 3
    line 2: (s), (mm), (N), (lbf)                             <- units
    line 3+: data

  Both formats have the same *column order* (time, displacement, force,
  extensometer channel), which is all the loader actually relies on -- it
  reads by position, not by header name. That matters because the new
  format's 4th column is labeled "Force 3 (lbf)", but it is NOT a force:
  it's the same extensometer strain reading as before (in %), just on a
  channel that never got relabeled after being repurposed. The loader
  treats it as strain_pct regardless of what the header claims.

Expected measurements file (MEASUREMENTS_CSV, one row per sample):
    File Name, Diameter, Length, epsilon L
    Trial Run 1, 0.496, 0.9245, 9.65
    ...
    Diameter and Length are in inches. epsilon L is the extensometer's
    gauge length in mm -- it isn't used in the modulus calculation
    (the strain channel is already true engineering strain, just in
    percent), it's just carried through and reported for the record.

    "File Name" is matched against the sample name loosely: underscores
    and extra whitespace are treated the same, and matching is case
    insensitive. So a data file named "Trial_Run_1.csv" will match a
    "File Name" of "Trial Run 1".

Note on sign convention: this DAQ records displacement, force, and strain
all as *negative* during compression (the actuator moves in its negative
direction). The script flips force and strain automatically, so
everything below is described as if compression were positive.

Note on contact: the platens are already in contact with the sample at
the start of the recording (~40-60 N preload), so the very first sample
is the true zero point -- no contact detection is needed. A sanity check
just confirms the first Force reading is in that expected preload band
before trusting the rest of the calculation (PRELOAD_RANGE_N,
MACHINE_NOISE_N below).

Method
------
1. Zero strain at the first sample (strain = Axial AI1_AOX / 100, offset
   so it's zero at index 0).

2. Compute engineering stress (Force / cross-sectional area, from the
   measured diameter).

3. Find the initial straight-line portion of the stress-strain curve
   with a sliding-window linear regression, choosing the window with
   the best R^2 (restricted to the region before peak force, so we
   don't accidentally fit yielded/post-peak data).

4. Fit a line to that window: stress = E*strain + b. The slope IS the
   modulus of elasticity -- it's the same number you'd get by picking
   any point on the tangent line and dividing its stress by its strain
   measured from where the extended tangent crosses the strain axis
   (that axis crossing is simply -b/E). We report that intercept too,
   as a sanity check that the toe region was handled correctly.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import linregress, rankdata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MEASUREMENTS_CSV = os.path.join(SCRIPT_DIR, "Trial_Run_Measurements.csv")

IN_TO_MM = 25.4
PRELOAD_RANGE_N = (40.0, 60.0)   # expected Force at the very first sample
MACHINE_NOISE_N = 6.0            # ~1 std of Force noise on this machine


def _normalize_name(name):
    return re.sub(r"[\s_]+", " ", name).strip().lower()


def _detect_data_start(path):
    """
    Finds where the actual data starts by locating the header row (the
    one naming the columns) rather than assuming a fixed preamble length
    -- old-format files have a 6-line Test/Test Run/Date preamble before
    it, new-format files have none. "Displacement" is used as the anchor
    because it appears verbatim in both formats' header row and nowhere
    else in the preamble.
    """
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for i, line in enumerate(fh):
            if "displacement" in line.lower():
                return i + 2  # skip the header line and the units line after it
    raise ValueError(f"Could not find a header row containing 'Displacement' in {path}")


def load_test_data(sample_name):
    target = _normalize_name(sample_name)
    candidates = {
        f[:-4]: f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")
    }
    match = next((fname for stem, fname in candidates.items()
                  if _normalize_name(stem) == target), None)

    if match is None:
        raise FileNotFoundError(
            f"No file matching '{sample_name}' in {DATA_DIR}.\n"
            f"Available samples: {', '.join(sorted(candidates)) if candidates else '(none found)'}"
        )

    path = os.path.join(DATA_DIR, match)
    skip = _detect_data_start(path)

    df = pd.read_csv(
        path,
        skiprows=skip,
        names=["time_s", "displacement_mm", "force_N", "strain_pct"],
        header=None,
    )
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df["force_N"].to_numpy(), df["strain_pct"].to_numpy()


def load_measurements(sample_name):
    if not os.path.isfile(MEASUREMENTS_CSV):
        raise FileNotFoundError(f"Measurements file not found: {MEASUREMENTS_CSV}")

    df = pd.read_csv(MEASUREMENTS_CSV)
    df.columns = [c.strip() for c in df.columns]
    target = _normalize_name(sample_name)
    match = df[df["File Name"].apply(_normalize_name) == target]

    if match.empty:
        available = ", ".join(df["File Name"].tolist())
        raise ValueError(
            f"No row for '{sample_name}' in {MEASUREMENTS_CSV}.\n"
            f"Available: {available}"
        )

    row = match.iloc[0]
    return {
        "diameter_in": float(row["Diameter"]),
        "length_in": float(row["Length"]),
        "gauge_length_mm": float(row["epsilon L"]),
    }


def normalize_sign(force, strain_pct):
    """
    Force and strain are both recorded negative during compression on this
    setup. Flip each (independently, based on its own trend) so both
    increase positively as the test proceeds.
    """
    n = len(force)
    edge = max(20, n // 100)
    force_sign = -1.0 if force[-edge:].mean() < force[:edge].mean() else 1.0
    strain_sign = -1.0 if strain_pct[-edge:].mean() < strain_pct[:edge].mean() else 1.0
    return force * force_sign, strain_pct * strain_sign


def check_preload(force):
    """
    Sanity check only (doesn't affect the calculation): the platens are
    expected to already be in contact at the start of the record, with
    roughly 40-60 N of preload. If the first reading is well outside that
    band -- more than a few multiples of the ~6 N machine noise beyond
    it -- something is probably off (test started before contact, wrong
    file, sign issue, etc.) and it's worth a look before trusting E.
    """
    first_force = force[0]
    low, high = PRELOAD_RANGE_N
    margin = 3 * MACHINE_NOISE_N
    if not (low - margin <= first_force <= high + margin):
        print(
            f"Warning: first Force reading is {first_force:.1f} N, outside "
            f"the expected {low}-{high} N preload band (+/- {margin:.0f} N "
            f"margin). Double check this sample started in contact."
        )


def find_modulus_window(strain, stress, win_frac=0.08, smooth_frac=0.06,
                         min_start_frac=0.04, max_start_frac=0.45):
    """
    Finds the modulus window by scoring every candidate point on two
    things at once, rather than using one to gate the other:
      - the local derivative of the *smoothed* stress-strain curve
        (Savitzky-Golay on both strain and stress, then a finite
        difference) -- a stable stand-in for "how steep is it here",
      - the R^2 of a window centered on that point, evaluated against
        the *raw, unsmoothed* data -- "how trustworthy is a straight-line
        reading here".
    Both series are converted to percentile ranks (0-1, robust to
    outliers -- see below) and summed; the point with the highest
    combined rank wins, and its window (fit on raw data) is the answer.

    Candidates are restricted to `min_start_frac`-`max_start_frac` of the
    pre-peak range, to avoid the initial settling-artifact spike and
    post-yield data, same as before.

    Why rank, not min-max: an earlier version of this used plain min-max
    normalization, and it broke on one A-series sample where the smoothed
    derivative had a single point with a near-zero denominator (a tiny
    but nonzero step in smoothed strain) blowing up to ~11 million, versus
    tens or hundreds everywhere else. Min-max crushes everything else to
    ~0 next to an outlier like that, so the "highest combined score" was
    just that one degenerate point (R^2 ~ 0.03 -- pure noise). Percentile
    rank only cares about ordering, so one absurd value can't dominate --
    it's still an outlier, but it just ranks 1st out of n rather than
    rescaling everything else into irrelevance.

    This replaced an earlier version that required R^2 > 0.95 outright
    and fell back to a plain best-R^2 search when nothing cleared that
    bar -- which happened on a B-6 sample whose best achievable R^2
    anywhere in the valid range was ~0.84. This scoring approach doesn't
    need a fallback: it always has a well-defined best point, and it
    happened to land within 1% of that sample's previously-known value.
    """
    peak_idx = int(np.argmax(stress))
    n = peak_idx
    if n < 20:
        raise RuntimeError("Not enough pre-peak data to fit a linear region.")

    win = max(10, int(n * win_frac))
    half = win // 2
    smooth_w = max(11, int(n * smooth_frac))
    if smooth_w % 2 == 0:
        smooth_w += 1
    smooth_w = min(smooth_w, n - 1 if (n - 1) % 2 == 1 else n - 2)

    strain_s = savgol_filter(strain[:n], smooth_w, 3)
    stress_s = savgol_filter(stress[:n], smooth_w, 3)
    d_strain = np.diff(strain_s)
    d_stress = np.diff(stress_s)
    d_strain = np.where(np.abs(d_strain) < 1e-12, 1e-12, d_strain)
    deriv = d_stress / d_strain  # smoothed local slope, length n-1

    lo = int(n * min_start_frac)
    hi = min(int(n * max_start_frac), len(deriv))
    if hi <= lo:
        raise RuntimeError("min_start_frac/max_start_frac leave no candidates.")

    idxs = np.arange(lo, hi)
    r2_vals = np.full(len(idxs), np.nan)
    for j, i in enumerate(idxs):
        start = max(0, i - half)
        end = min(n, start + win)
        if end - start < 10:
            continue
        result = linregress(strain[start:end], stress[start:end])
        r2_vals[j] = result.rvalue ** 2

    valid = ~np.isnan(r2_vals)
    idxs, r2_vals, deriv_vals = idxs[valid], r2_vals[valid], deriv[idxs[valid]]

    r2_rank = (rankdata(r2_vals) - 1) / (len(r2_vals) - 1)
    deriv_rank = (rankdata(deriv_vals) - 1) / (len(deriv_vals) - 1)
    best_j = np.argmax(r2_rank + deriv_rank)
    best_i = idxs[best_j]

    start = max(0, best_i - half)
    end = min(n, start + win)
    result = linregress(strain[start:end], stress[start:end])
    return result.rvalue ** 2, start, end


def sig_figs(x, n=3):
    if x == 0:
        return 0.0
    from math import log10, floor
    d = n - int(floor(log10(abs(x)))) - 1
    return round(x, d)


def analyze(sample_name):
    force, strain_pct = load_test_data(sample_name)
    meas = load_measurements(sample_name)

    force, strain_pct = normalize_sign(force, strain_pct)
    check_preload(force)

    # Platens are already in contact at the first sample (~40-60 N
    # preload), so the first reading IS the zero point -- no detection
    # needed.
    strain0_pct = strain_pct[0]

    diameter_mm = meas["diameter_in"] * IN_TO_MM
    area_mm2 = np.pi / 4 * diameter_mm ** 2

    strain = (strain_pct - strain0_pct) / 100.0   # true strain, mm/mm
    stress_MPa = force / area_mm2                 # N/mm^2 = MPa

    r2, start, end = find_modulus_window(strain, stress_MPa)
    fit = linregress(strain[start:end], stress_MPa[start:end])

    E_MPa = fit.slope
    E_GPa = E_MPa / 1000.0
    E_psi = E_MPa * 145.037738

    strain_axis_intercept = -fit.intercept / fit.slope

    return {
        "sample": sample_name,
        "diameter_in": meas["diameter_in"],
        "length_in": meas["length_in"],
        "gauge_length_mm": meas["gauge_length_mm"],
        "area_mm2": area_mm2,
        "initial_force_N": force[0],
        "fit_strain_range": (strain[start], strain[end - 1]),
        "fit_r_squared": r2,
        "toe_compensated_strain_zero": strain_axis_intercept,
        "E_GPa": sig_figs(E_GPa),
        "E_psi": sig_figs(E_psi),
        # Full curve + fit details, for plotting.
        "strain": strain,
        "stress_MPa": stress_MPa,
        "fit_start_idx": start,
        "fit_end_idx": end,
        "fit_slope": fit.slope,
        "fit_intercept": fit.intercept,
    }


def plot_stress_strain(results, save_path=None, show=True):
    """
    Plots the full stress-strain curve, highlights the window used for the
    modulus fit, and draws that fit's tangent line (extended back to the
    strain axis, so the toe-compensation is visible).
    """
    import matplotlib.pyplot as plt

    strain = results["strain"]
    stress = results["stress_MPa"]
    start, end = results["fit_start_idx"], results["fit_end_idx"]
    slope, intercept = results["fit_slope"], results["fit_intercept"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(strain, stress, color="0.6", lw=1, label="Full curve")
    ax.plot(strain[start:end], stress[start:end], color="tab:red", lw=2.5,
             label="Region used for E fit")

    # Tangent line, extended from the strain-axis intercept through a bit
    # past the fit window, so the toe compensation is visible.
    x0 = results["toe_compensated_strain_zero"]
    x1 = strain[end - 1] * 1.15
    xs = np.array([x0, x1])
    ax.plot(xs, slope * xs + intercept, "--", color="tab:blue", lw=1.5,
             label="Tangent (extended)")
    ax.axhline(0, color="black", lw=0.8)
    ax.plot(x0, 0, "o", color="tab:blue", ms=5)

    ax.set_xlabel("Engineering strain (mm/mm)")
    ax.set_ylabel("Engineering stress (MPa)")
    ax.set_title(
        f"{results['sample']}: E = {results['E_GPa']} GPa "
        f"(R^2 = {results['fit_r_squared']:.4f})"
    )
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    sample = input("Sample name (e.g. Trial Run 1 or Trial_Run_1): ").strip()

    results = analyze(sample)

    print()
    print(f"Sample:                     {results['sample']}")
    print(f"Diameter:                   {results['diameter_in']:.4f} in")
    print(f"Length:                     {results['length_in']:.4f} in")
    #print(f"Extensometer gauge length:  {results['gauge_length_mm']:.3f} mm")
    print(f"Cross-sectional area:       {results['area_mm2']:.4f} mm^2")
    print(f"Initial (preload) force:    {results['initial_force_N']:.2f} N")
    print(f"Linear-fit strain range:    {results['fit_strain_range'][0]:.5f} to "
          f"{results['fit_strain_range'][1]:.5f}")
    print(f"Linear-fit R^2:             {results['fit_r_squared']:.5f}")
    print(f"Toe-compensated strain 0:   {results['toe_compensated_strain_zero']:.5f}")
    print()
    print(f"Modulus of Elasticity:      {results['E_GPa']} GPa  "
          f"({results['E_psi']} psi)")

    show_plot = input("\nShow stress/strain plot? (y/n): ").strip().lower()
    if show_plot.startswith("y"):
        safe_name = re.sub(r"[^\w\-]+", "_", results["sample"])
        save_path = os.path.join(SCRIPT_DIR, f"{safe_name}_stress_strain.png")
        plot_stress_strain(results, save_path=save_path)


if __name__ == "__main__":
    main()