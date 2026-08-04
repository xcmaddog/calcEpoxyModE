# calcEpoxyModE

A Python script that calculates the modulus of elasticity of a cylindrical
sample from compression-test data (MTS TestSuite DAQ export + extensometer
strain), using a tangent-modulus fit with automatic toe compensation.

## Requirements

- Python 3.8+
- `numpy`, `pandas`, `scipy`
- `matplotlib` (optional — only needed if you want the stress/strain plot)

```
pip install numpy pandas scipy matplotlib
```

## Setup

Place `analyze_modulus.py` in a folder alongside:

```
your-folder/
├── analyze_modulus.py
├── Trial_Run_Measurements.csv
└── data/
    ├── Sample_1.csv
    ├── Sample_2.csv
    └── ...
```

- **`data/`** — one raw CSV per test run, exported directly from MTS
  TestSuite. Expects the standard export layout: 7 metadata lines, a
  header line, a units line, then data with columns `Running Time` (s),
  `Axial Displacement` (mm), `Axial Force` (N), and `Axial AI1_AOX` (%) —
  the extensometer's strain reading.

- **`Trial_Run_Measurements.csv`** — one row per sample, with columns:

  | Column | Meaning | Units |
  |---|---|---|
  | `File Name` | Matches the sample's data CSV (case/spacing-insensitive — `Trial Run 1` matches `Trial_Run_1.csv`) | — |
  | `Diameter` | Measured specimen diameter | inches |
  | `Length` | Measured specimen length | inches |
  | `epsilon L` | Extensometer gauge length | mm |

  `epsilon L` is carried through and reported for the record, but isn't
  currently used in the calculation — the extensometer channel is already
  true engineering strain, just expressed as a percentage.

## Usage

```
python analyze_modulus.py
```

You'll be prompted for a sample name (either `Trial Run 1` or
`Trial_Run_1` works), then the script prints the results and offers to
save/show a stress-strain plot.

## What it does

1. **Sign convention** — this DAQ records displacement, force, and strain
   all as *negative* during compression. The script detects and flips
   this automatically.
2. **Zero point** — assumes the platens are already in contact at the
   start of the recording (~40–60 N preload), so the first data point is
   taken as the zero-strain reference. If the first Force reading falls
   well outside that expected band, the script prints a warning rather
   than silently trusting the data.
3. **Stress/strain** — stress = Force / cross-sectional area (from the
   measured diameter); strain = extensometer % / 100.
4. **Modulus fit** — searches the *early* part of the loading curve (before
   peak stress) for the straight-line window with the best R², then fits
   a line to it. The slope of that line is the modulus of elasticity —
   equivalent to picking any point on the tangent line and dividing its
   stress by its strain measured from where the extended tangent crosses
   the strain axis (standard toe-compensation).
5. **Output** — modulus in GPa and psi (3 sig figs), the R² of the fit,
   the strain range used, and the toe-compensated strain-axis intercept
   (useful as a sanity check — it should be close to zero).

## Sample output

```
Sample name (e.g. Trial Run 1 or Trial_Run_1): Trial Run 1

Sample:                     Trial Run 1
Diameter:                   0.4960 in
Length:                     0.9245 in
Extensometer gauge length:  9.650 mm
Cross-sectional area:       124.6581 mm^2
Initial (preload) force:    54.26 N
Linear-fit strain range:    0.00610 to 0.01138
Linear-fit R^2:             0.99022
Toe-compensated strain 0:   -0.00049

Modulus of Elasticity:      2.31 GPa  (335000.0 psi)

Show stress/strain plot? (y/n):
```

If you say yes, it saves (and tries to display) a plot of the full curve
with the fitted region highlighted and the tangent line drawn through it.

## Notes

This was built around one specific MTS TestSuite setup and its ~6 N noise
floor. If you're running it against a different machine or specimen
geometry, the constants worth checking first are `PRELOAD_RANGE_N` and
`MACHINE_NOISE_N` near the top of the file, and `min_frac` /
`max_start_frac` in `best_linear_window()` (how wide a straight-line
window it looks for, and how early in the curve it's allowed to start).