# Research Notes

Working notes: decisions, findings, and open questions. Every claim below says where it comes from.
- **[paper]**: stated in the cited source.
- **[data]**: checked directly in the files in `data/raw/`.
- **[me]**: my own interpretation or standard textbook knowledge.

---

## Day 1, Block 3: Reading

### 1. Zhang et al. 2020, *Nature Communications* 11:1706

Source: open-access copy at https://pmc.ncbi.nlm.nih.gov/articles/PMC7136228 (DOI 10.1038/s41467-020-15235-7)

**Setup [paper]**
- 12 commercial **Eunicell LR2032 coin cells**, 45 mAh, **LiCoO2/graphite**.
- Temperatures: 25, 35 and 45 °C.
- Cycling: 1C (45 mA) constant-current/constant-voltage charge to 4.2 V, then 2C (90 mA) constant-current discharge to 3 V.
- Impedance measured from **0.02 Hz to 20 kHz** with a 5 mA excitation current.
- More than 20,000 spectra in total.

**What are the 9 "states"?**
- **[paper]** Impedance is measured "at nine different stages of charging/discharging during every even-numbered cycle".
- **[paper, Supplementary Fig. 1]** The states, measured during constant-current/constant-voltage charging and discharging:

| State | When | DC current flowing? |
|---|---|---|
| I | Before charging | no |
| II | Start of charging | yes |
| III | After 20 min of charging | yes |
| IV | After charging, before resting | — |
| **V** | **After a 15 min rest (fully charged)** | **no** |
| VI | Start of discharging | yes |
| VII | After 10 min of discharging | yes |
| VIII | After discharging, before resting | — |
| IX | After a 15 min rest (fully discharged) | no |

- **[me]** States V and IX are the "electrochemically stable" ones: rested, with no current flowing. That's why the measurements are most consistent there.
- **[paper, Supplementary Fig. 2]** They also tested a model trained on state V against spectra from the other states for cell 25C02. Possible future work for us: does a frequency budget chosen at state V still work at other states?
- **[paper, Supplementary Fig. 3]** The **imaginary part** at 17.80 Hz and 2.16 Hz has a positive linear correlation with cycle number for 25C01. This matches my feature-index check below.
- **[paper, Supplementary Fig. 4]** The same train/test grouping as in the main text.
- **[paper, Supplementary Table 1]** RMSE of the impedance-based method compared with a discharge-curve method.
- **[peer-review file]** Reviewer 1 notes that all cells were cycled identically at a constant temperature, so the model is only shown to work under those conditions. **The same limitation applies to us; include it in the report.**

**Which state did they use?**
- **[paper]** **State V** for capacity and remaining-useful-life estimation (Figs. 1a and 2–4).
- **[paper]** Reason: the model "is most accurate at electrochemically stable states", where measurements are more consistent.
- **[data]** The tutorial CSV also uses state V. Its first row (cell 1, cycle 1) has ZReal = 0.38470 and 0.39156 Ω at 20 kHz and 15.8 kHz. These match the first two rows of `EIS_state_V_25C01.txt` exactly.

**How did they split training and test data?**
- **[paper]** By **cell**, not by spectrum.
  - Train: 25C01–25C04, 35C01, 45C01.
  - Test: 25C05–25C08 (plus, presumably, 35C02 and 45C02).
- **[me]** This confirms our rule: split by cell. My plan goes further by holding out each cell in turn (leave-one-cell-out), which reports the spread across cells instead of a single split.

**How is capacity matched to each impedance measurement?**
- **[paper]** This is **not stated explicitly**. Capacity is measured after every **odd** cycle, and impedance during every **even** cycle. The paper only says the target is "the capacity corresponding to the EIS spectrum".
- **[me]** The likely pairing is each impedance cycle with the adjacent odd cycle's capacity. **Check the tutorial notebook or the capacity files before trusting any pairing.**

**Model [paper]**
- Gaussian process regression (GPR) with a zero mean and a diagonal squared-exponential kernel with ARD (one length-scale per input). Hyperparameters are chosen by maximising the marginal likelihood.
- Inputs: real and imaginary parts at 60 frequencies (120 features), standardised with the training-set mean and standard deviation.
- Metrics: R² is shown in the figures. RMSE is not reported in the main text.

**Most important finding for our project: which frequencies matter [paper + data]**
- **[paper]** At 25 °C, ARD gives the largest weights to **features 91 and 100, corresponding to 17.80 Hz and 2.16 Hz**. At 35 and 45 °C, 17.80 Hz alone is the salient frequency.
- **[data]** In the raw files, 17.796 Hz is frequency #31 and 2.161 Hz is frequency #40 of 60. With features ordered as [60 real, 60 imaginary], features 91 and 100 are the **imaginary part at #31 and #40**. The numbers are consistent.
- **[me]** Zhang's own model already suggests that **a few mid-to-low frequencies carry most of the aging information**. That is a strong motivation for our frequency-budget question. Our E2 experiment tests it systematically instead of reading it off one model.

### 2. Kallel, Petrychenko, Kanoun 2022, *Applied Sciences* 12(2):885

Sources: https://doi.org/10.3390/app12020885 (abstract checked via Crossref)
- **Status:** MDPI blocked automated access. **I've read only the abstract; read the full paper in a browser.**

**What it says [paper, abstract]**
- They build a detailed electrical model of the cell, simplify it to fit the measured spectra, and use the physical meaning of its parameters.
- Proposed SoH indicator: **R_SEI / (R_SEI + R_ct)**, the ratio of the SEI resistance to the SEI resistance plus the charge-transfer resistance. It describes "charge-transfer efficiency" at the electrode surface.
- Experiment: 4 cells of the same type, cycled 400 times, with impedance measured every 50 cycles.
- Result: the ratio correlates systematically with cycle number. They propose it as a basis for **online** SoH assessment.

**Which impedance features relate to aging? [me, to confirm in the full text]**
- The **SEI resistance** and the **charge-transfer resistance**, both read from the mid-frequency arc or arcs.
- Their approach uses physically meaningful features, which is the opposite of Zhang's approach of feeding the whole spectrum into a black box.
- **Idea for our project:** add a "physics-feature" baseline alongside the ohmic-resistance baseline.

**TODO (full text):**
- Which battery type?
- Which equivalent circuit?
- Frequency range?
- How do R0 and the Warburg element behave with aging?

### 3. SSD'26 press release: Fast A\*-mRMR (Kallel & Kanoun, Best Paper)

Source: https://www.tu-chemnitz.de/tu/pressestelle/aktuell/13437/en

**What problem does frequency selection solve? [paper = press release]**
- One impedance measurement can have **up to 480 data points**. That is too many for the small microcontrollers in battery management systems to process in real time.

**Method [press release]**
- They adapt **A\* search** (a path-finding algorithm) to choose informative points without trying every combination.
- It uses an **mRMR** criterion (maximum relevance, minimum redundancy): high relevance to state of charge, low overlap with points already chosen.

**Reported results [press release]**
- 1.27 s computation time, described as more than 500× faster than comparable methods.
- 3.62–3.93 % error.
- Data: physics-based digital-twin data, 4 cells × 17 SoC levels × 3 temperatures.

**Surprising finding [press release]**
- "allowing a controlled degree of similarity between selected data points can actually improve predictions."
- **[me]** Some redundancy helps. I'll watch for this in E2: mRMR may *not* beat simple selectors, and that's a valid outcome.

**How our work differs [me]**
- They estimate **SoC**; we estimate **SoH**.
- They use digital-twin data; we use real measured data (Zhang).
- We use **standard mRMR, not their A\*-mRMR**. Never claim to reproduce their method.
- The number of points their method finally selects isn't in the press release.

### 4. Key concepts [me, standard definitions]

| Term | Meaning | Where it appears on the Nyquist plot |
|---|---|---|
| **Impedance Z** | Complex ratio V/I under a small AC excitation at frequency f. Z = Re(Z) + j·Im(Z), or equivalently a magnitude \|Z\| and a phase | — |
| **Nyquist plot** | x = Re(Z), y = −Im(Z), one point per frequency. **High frequency on the left, low frequency on the right.** The frequency itself is not on an axis | The whole curve |
| **Bode plot** | \|Z\| and phase plotted against log(f). It shows the frequency explicitly, which is useful for seeing *which* frequency does what | — |
| **Ohmic resistance R0** | Electrolyte, separator, current collectors and contacts. It acts instantly, so it has no frequency dependence | Where the curve crosses the real axis at high frequency (left) |
| **SEI resistance R_SEI** | Resistance of the solid-electrolyte interphase film on the anode, which grows with aging | A small first arc at high-to-mid frequency (sometimes merged with the next arc) |
| **Charge-transfer resistance R_ct** | Resistance of the electrochemical reaction at the electrode surface, in parallel with the double-layer capacitance | The main **semicircle** at mid frequency; its diameter ≈ R_ct |
| **Warburg tail** | Solid-state diffusion of lithium | Roughly a 45° line at **low frequency** (right) |
| **Inductive tail** | Cables and cell geometry | Points below the real axis at the highest frequencies |

**[data]** That inductive tail is visible in our data. In `EIS_state_V_25C01.txt`, the column labelled "-Im(Z)" is **negative** at 20 kHz (−0.035 Ω). In other words, the curve dips below the axis at the highest frequencies. This isn't an error, but it matters when estimating R0: use the zero crossing, not the first point.

---

## Facts checked in the data [data]

- Raw file format: tab-separated, 7 columns: `time/s`, `cycle number`, `freq/Hz`, `Re(Z)/Ohm`, `-Im(Z)/Ohm`, `|Z|/Ohm`, `Phase(Z)/deg`.
- 60 frequencies per spectrum, from 20004 Hz down to 0.01999 Hz, in descending order.
- `EIS_state_V_25C01.txt` has 15,660 rows = **261 spectra** (261 unique cycle numbers).

## Day 1, Block 4: Exploratory analysis results [data]

Source of truth: `notebooks/01_eda.ipynb`. The numbers below are copied from its outputs.

**Inventory**
- 88 EIS files covering 12 cells: 8 at 25 °C, 2 at 35 °C and 2 at 45 °C.
- States I–V and IX exist for all 12 cells. States VI–VIII exist only for some 25 °C cells.
- `25C42` appears only once (state VI). It isn't one of the paper's 12 cells, so we **ignore it**.
- Every spectrum has 60 frequencies, from 20004 Hz down to 0.01999 Hz.

**State V: spectra, labels and lifetime per cell**

| cell | spectra | capacity cycles | labelled spectra | final SoH (last capacity cycle) |
|---|---|---|---|---|
| 25C01 | 261 | 350 | 261 | 0.239 |
| 25C02 | 250 | 181 | 181 | 0.710 |
| 25C03 | 229 | 202 | 202 | 0.748 |
| 25C04 | 81 | 35 | **35** | 0.866 |
| 25C05 | 275 | 328 | 275 | 0.001 (cell collapsed, see below) |
| 25C06 | 212 | 213 | 212 | 0.534 |
| 25C07 | 140 | 148 | 140 | 0.430 |
| 25C08 | 86 | 37 | **37** | 0.761 |
| 35C01 | 327 | 327 | 327 | 0.360 |
| 35C02 | 318 | 318 | 318 | 0.670 |
| 45C01 | 299 | 300 | 299 | 0.729 |
| 45C02 | 310 | 310 | 310 | 0.734 |

- Labelled spectra at 25 °C: 1,343. That is exactly the size of the tutorial CSV.
- **Earlier open question resolved:** nothing was subsampled. The cells just have very different lifetimes, from 35 to 275 labelled spectra.

**How capacity labels are paired with spectra (resolved)**
- Capacity files are time series. Per cycle, the maximum capacity where `ox/red == 0` is the **discharge** capacity.
- **Impedance cycle n is paired with the discharge capacity of capacity-file cycle n − 1.** This reproduces the tutorial's `Q` exactly.
- SoH label: `q = Q / Q(first cycle)`.

**Check of the tutorial CSV against the raw files**
- 3 random rows (seed 0): rows 854, 686 and 1140, from cells 25C05 and 25C06.
- Real part, imaginary part and Q are **identical** to the raw files (maximum difference 0.0).
- `seriesIdx` k corresponds to cell 25C0k, state V.
- The tutorial's `ZImag_*` columns contain **−Im(Z)**, the same sign as the raw column, even though the name suggests otherwise.

**Problems in the raw files (handled in the loader)**
1. The 35 °C and 45 °C EIS files have **no header row**; the 25 °C files do.
2. In the capacity files, the header has spaces instead of a tab between `cycle number` and `ox/red`.
3. Capacity files have two layouts: 4 columns, or 6 columns with `Ewe/V` and `I/mA`. In both, capacity is the last column.
4. The last capacity cycle sometimes has no discharge value (NaN), so it is dropped.
5. `.DS_Store` and `__MACOSX` are macOS junk files; ignore them.

**What the figures show (first reading; check against your own observations)**
- `eda_nyquist_aging.png`, cell 25C01 at cycles 1, 131 and 261:
  - The mid-frequency **semicircle grows strongly** with aging. The minimum between the arc and the tail moves from Re ≈ 1.05 Ω to ≈ 1.15 Ω, then to ≈ 1.57 Ω.
  - The high-frequency intercept (≈ R0) moves much less, from about 0.38 to 0.44 Ω.
  - **[me]** This suggests that charge-transfer/interface processes, not ohmic resistance, dominate the aging signal. That is consistent with Zhang's ARD result (17.8 Hz and 2.16 Hz) and with Kallel 2022's focus on R_SEI and R_ct. It also predicts that our ohmic-only baseline will be weak.
- `eda_capacity_fade.png`:
  - The 25 °C cells degrade very unevenly. **25C07 and 25C05 show sudden drops** (a "knee") instead of a smooth fade.
  - 25C05 falls to about 0 mAh by cycle 300. Its final SoH of 0.001 is a dead cell, not a parsing error. Only cycles that have impedance spectra are used for labels.
  - The 35/45 °C cells start with higher capacity and fade more smoothly. 35C01 has a knee near cycle 300.
  - **[me]** Implications for modelling:
    - Held-out cells will differ a lot, so the per-cell spread of errors matters.
    - Cells 25C04 and 25C08 contribute only 35–37 spectra each.
- **Estimated sweep time** (one period per frequency = Σ 1/f):
  - All 60 frequencies: **239.5 s**.
  - Only the frequencies ≥ 1 Hz: **4.5 s**.
  - **[me]** The lowest decade of frequencies takes most of the time. This is a lower-bound estimate (no settling or averaging), and it is the core motivation for E3.
  - Note that 2.16 Hz, one of Zhang's two key frequencies, is above 1 Hz.

## Day 2, Block 1: Data module

**Code**
- `src/frugal_eis/data.py`: `load_dataset(state, temps)` returns an `EISDataset` with `X_re`, `X_im`, `freqs`, `soh`, `capacity`, `cell_id`, `cycle` and `temp`.
- `src/frugal_eis/features.py`: `full_spectrum`, `full_spectrum_names` and `ohmic_feature`.
- The package is installed in editable mode (`pip install -e .`), so notebooks, experiments and tests can all do `import frugal_eis`.

**Decisions**
- **SoH is relative:** capacity ÷ first-cycle capacity *of the same cell*.
  - Why: initial capacities differ (34–42 mAh; the 35/45 °C cells start higher). A relative SoH measures each cell's own fade, so every cell starts at SoH = 1.
  - Downside: errors in the first-cycle capacity affect every label of that cell.
  - The alternative, capacity ÷ nominal 45 mAh, would put all cells below 1 from the start.
- **`X_im` stores −Im(Z), as in the raw files.** It is positive above the real axis. I didn't flip the sign, to avoid one easy source of bugs.
- **R0 estimate:** linear interpolation of Re(Z) where −Im crosses zero at high frequency. The first point is inductive, so taking Re at the highest frequency would be biased.
- Cells missing a state (VI–VIII) are skipped with a warning, not an error.

**Validation [data]** (checked against the tutorial CSV)
- State V at 25 °C: **1,343 spectra × 60 frequencies**: the same rows, in the same order, as the tutorial CSV.
- Maximum difference: Re 0.0, −Im 0.0, capacity 5e-14, SoH 6e-16, i.e. identical.
- SoH range at 25 °C: 0.184–1.0, and the first spectrum of each cell is exactly 1.0.
- `full_spectrum_names`: index 90 = `mIm@17.8Hz` and index 99 = `mIm@2.16Hz`. These are Zhang's "91st and 100th" features (1-based), which confirms the [Re, −Im] ordering.
- R0 ranges from 0.274 to 0.476 Ω. For 25C01 it goes from 0.397 to 0.449 Ω.
- **The correlation between R0 and SoH at 25 °C is −0.445.** That is weak to moderate, as the Nyquist plot suggested. Expect the ohmic-only baseline to be clearly worse than full-spectrum models.
- All temperatures, state V: 2,597 spectra (1,343 at 25 °C, 645 at 35 °C, 609 at 45 °C).
- State VII: 424 spectra from 4 cells, with 4 skip warnings, as expected.

**Fixed:** `requirements.txt` was UTF-16 (written by `pip freeze >` in Windows PowerShell 5.1). I replaced it with a UTF-8 file of pinned direct dependencies.

## Day 2, Block 2: Tests

Run with `python -m pytest -q`: **18 passed** (about 40 s, mostly data loading). Tests that need the raw data skip automatically if it isn't downloaded.

| File | What it checks |
|---|---|
| `tests/test_data.py` | Consistent shapes, no NaNs, SoH range, one temperature per cell, the 12 expected cells (25C42 never loaded), frequencies descending, cycles increasing, an exact match with the tutorial CSV, a warning for missing states, an error for an invalid state |
| `tests/test_evaluate.py` | **Leakage**: in every leave-one-cell-out fold, train and test cells are disjoint and the test fold is exactly the held-out cell; every cell is held out once. Run on synthetic interleaved groups and on the real 12 cells |
| `tests/test_features.py` | Feature order of `full_spectrum`; `ohmic_feature` on hand-made spectra with known answers (zero crossing at 0.41 Ω, no inductive part, rows independent, error if −Im never crosses zero) |

**Changed from the plan: the SoH bound is (0, 1], not (0.3, 1.1)** [data]
- 25 spectra of **25C05** have SoH below 0.3, with a minimum of 0.184. That cell genuinely collapses (see `eda_capacity_fade.png`); it isn't a parsing error.
- Loosening a test so it passes would hide information, so the test instead checks what must always hold: SoH > 0, SoH ≤ 1, and exactly 1.0 at each cell's first spectrum.
- Minimum SoH per cell: 25C05 0.184, 35C01 0.360, 25C01 0.370, 25C07 0.442; all others ≥ 0.53.
- **[me]** For modelling, 25C05's collapse region is far outside what other cells show. When 25C05 is held out, the model must extrapolate below the training range, so expect its error to be large. Report it rather than removing the cell.

**Checking the leakage test itself:** with a random row split (`KFold(shuffle=True)`) swapped in for `loco_splits`, the test fails as it should. So it actually detects leakage.

`src/frugal_eis/evaluate.py` currently contains only `loco_splits()`. The full evaluation harness comes in Block 3.

## Day 2, Block 3: Evaluation harness

**Code** (`src/frugal_eis/evaluate.py`)
- `loco_evaluate(model_factory, X, y, groups, model_name=, cycle=, experiment=)` returns a `LOCOResult`:
  - `.per_cell`: one row per held-out cell, with rmse, mae, r2, n_train, n_test and fit time.
  - `.predictions`: one row per spectrum, with cell, cycle, y_true and y_pred.
  - `.summary()`: mean ± std over cells, plus metrics pooled over all predictions.
- It uses `LeaveOneGroupOut` (through `loco_splits`).
- `model_factory` builds a **fresh unfitted model for every fold** (an estimator instance is cloned). Scalers and selectors inside a Pipeline therefore only ever see training cells.
- Passing `experiment="e1"` writes `results/e1.csv` (per cell) and `results/e1_predictions.csv`.
- `save_results([r1, r2, ...], "e1")` combines several models into one file. `summarize([...])` gives one row per model.

**Tests:** 5 new ones, 23 in total, all passing.
- A perfectly linear model gives RMSE 0.
- The metrics equal sklearn's.
- A spy model confirms the held-out cell is never in the training data.
- CSV files are written and combined correctly.
- A length mismatch raises an error.

**Smoke run on real data: always predicting the training mean** (output in a temporary folder, not in `results/`) [data]
- RMSE per held-out cell: 0.041 (25C02) to 0.224 (25C05). **Mean 0.129 ± 0.058.**
- **[me]** This is the floor: any real model must beat RMSE ≈ 0.13. Add it as a third reference row in E1.

**Per-cell R² is misleading; use RMSE/MAE as the headline metrics** [data + me]
- With the mean predictor, 25C04 gets **R² = −33**, but its RMSE (0.166) is no worse than 25C01's (0.168, R² ≈ 0).
- The reason: R² divides by the variance of the true values *within the held-out cell*. 25C04 only goes from SoH 1.0 to 0.86 over 35 spectra, so the variance is tiny and R² explodes.
- The mean of per-cell R² (−5.2 ± 11.5) is dominated by such cells and means little.
- **Decision:** report RMSE and MAE per cell (mean ± std) as the main results. Report R² only **pooled** over all predictions, or per cell with this warning attached.

## Day 2, Block 4: First models (E1 partial)

Run: `python -m experiments.e1_partial` → `results/e1_partial.csv` (per cell) and `results/e1_partial_predictions.csv`.
Setup: state V, 25 °C, 1,343 spectra, 8 cells, leave-one-cell-out.

**Final results** [data] (RMSE/MAE are mean ± std across the 8 held-out cells; R² is pooled)

| model | features | RMSE | MAE | R² pooled |
|---|---|---|---|---|
| mean_predictor (reference) | — | 0.129 ± 0.058 | 0.104 ± 0.045 | −0.04 |
| ohmic_linear (physics baseline) | R0 only | 0.108 ± 0.058 | 0.090 ± 0.052 | 0.15 |
| ridge_full (RidgeCV, row-wise alpha selection, as originally planned) | 120 | 0.148 ± 0.077 | 0.136 ± 0.082 | −0.50 |
| **ridge_full_groupcv** (alpha by inner leave-one-cell-out) | 120 | **0.089 ± 0.038** | **0.082 ± 0.039** | **0.49** |

**What happened, step by step (the order matters, so I'm keeping it honest)**
1. **First run, as originally planned** (`RidgeCV`, grid 1e-4…1e3): ridge on the full spectrum was **worse than the mean predictor** (RMSE 0.149). Its chosen alphas were tiny (0.0009–0.005).
2. **Diagnosis [me]:** by default, RidgeCV picks alpha by leaving out single **rows**. Neighbouring spectra of the same cell are almost identical, so this inner validation leaks, just as a random outer split would. It rewards a model that memorises cell-specific detail, and that detail doesn't transfer to an unseen cell.
3. **Fix:** `GroupRidgeCV` in `models.py` picks alpha by leaving out one *training cell* at a time, mirroring the outer evaluation. `loco_evaluate` now passes the training cells' ids to `fit` when the model accepts `groups`; a test confirms the held-out cell is never among them.
4. **Second run:** RMSE 0.086, but the chosen alpha sat at the **upper edge of the grid (1000) in 7 of 8 folds**, so the true optimum could lie beyond it.
5. **Third run (final):** the grid was widened to 1e-4…1e6 (41 values) **for both ridge variants**, so they're still compared on the same grid.
   - Chosen alphas are now inside the range: 10–17,800.
   - RMSE 0.089, slightly worse than run 2's 0.086. I report the wide-grid number anyway. Choosing the grid that happened to score better would be cherry-picking, and 0.003 is well within the cell-to-cell spread (std 0.038).
   - The row-wise RidgeCV still picks alphas of 0.001–0.006 on the wide grid, so the diagnosis holds.

**Takeaways [me]**
- **How you tune is as important as the model.** Choosing hyperparameters with a leaky inner split turned a useful model into one worse than a constant. From now on, **every tuned model in this project tunes with cell-wise inner validation.** Note that GPR fits its kernel hyperparameters by marginal likelihood on the training rows, which may have the same problem; check this on Day 3.
- The full spectrum beats R0 alone (0.089 vs 0.108), but **not for every cell**:
  - On 25C03, R0 is much better (0.042 vs 0.167).
  - On 25C05 and 25C07, the full spectrum is much better (0.106 vs 0.196, and 0.038 vs 0.171).
  - Hypothesis for later: different cells age through different mechanisms.
- Large alphas (10³–10⁴) mean strong shrinkage. With 120 highly correlated features and only 7 training cells, the model can afford very little flexibility. That supports the frequency-budget idea: most of the spectrum is redundant.
- The spread across cells is large (std ≈ 40% of the mean). With 8 cells, differences of about 0.01 RMSE between models are **not** meaningful.

## Day 3, Blocks 1–4: Five models, E1, figures, README

**Block 1: models** (`models.py`)
- `random_forest` (300 trees, seed 0).
- `ARDGaussianProcess` / `gpr_ard(stride)`: scaler + GPR with kernel ConstantKernel × RBF (one length scale per feature) + WhiteKernel, `normalize_y=True`. It exposes `length_scales_` and `noise_level_`.
- `mlp`: scaler + MLP (64, 32), `early_stopping`, seed 0.
- 7 new tests in `tests/test_models.py`; 32 in total, all passing. They check that GPR's ARD ranks the one relevant toy feature first.

**GPR setup: decided by timing one fold** (25C01 held out, 1,082 training rows) [data]
- A first timing run (RF, MLP, then GPR on all 120 features with every 2nd spectrum) **ran 68 minutes without finishing** the GPR. I stopped it.
- Second run, unbuffered:

| Model | Fit time, one fold | RMSE on 25C01 |
|---|---|---|
| RF | 8.2 s | 0.050 |
| MLP | 0.6 s | 0.273 |
| GPR, −Im only, every 4th spectrum | 23.8 s | 0.151 |

- GPR with every 2nd spectrum was still running after about 4 minutes, so I stopped it.
- **Decision:** E1 uses GPR on **−Im only (60 features), every 4th training spectrum**. This is the plan's "if slow" fallback; it took 259 s for all 8 folds. Recorded as a limitation.

**Block 2: E1 results** (`python -m experiments.e1_full_spectrum`, about 5 min) [data]

| Model | RMSE mean ± std | MAE mean ± std | R² pooled |
|---|---|---|---|
| random_forest | **0.077 ± 0.026** | 0.068 ± 0.027 | 0.71 |
| ridge_groupcv | 0.089 ± 0.038 | 0.082 ± 0.039 | 0.49 |
| ohmic_linear | 0.108 ± 0.058 | 0.090 ± 0.052 | 0.15 |
| mean_predictor | 0.129 ± 0.058 | 0.104 ± 0.045 | −0.04 |
| gpr_ard_neg_im_s4 | 0.139 ± 0.052 | 0.120 ± 0.041 | −0.18 |
| mlp | 0.196 ± 0.143 | 0.164 ± 0.131 | −2.57 |

Files: `results/e1_full_spectrum.csv` (summary), `_per_cell.csv`, `_predictions.csv`.

**What the results say [data + me]**
- **The random forest is best, but it can't extrapolate.** On 25C05 (true minimum SoH 0.184), its lowest prediction is 0.378, just above the lowest SoH in its training data (0.370). Trees can only output values seen in training.
- **GPR is worse than the floor.**
  - Fitted noise was ≈10⁻⁴ or lower in every fold (max 4.4e-4).
  - **fig1b** shows why: for 25C01, the GPR mean follows the truth for about 17 cycles, then **falls back to the training average (≈0.72)** with a very wide ±2σ band. The held-out spectra are "far" from all training spectra in the learned ARD metric.
  - Its uncertainty is honest: 89.7 % of true values lie inside ±2σ.
  - **[me]** This is the Day 2 lesson again: hyperparameters fitted to training *rows* (here by marginal likelihood) can't anticipate cell-to-cell differences.
  - It is **not** a reproduction of Zhang et al. They used a fixed 4-cell training split, both Re and Im, and all spectra. Our setup differs on all three, so this says nothing about GPR in general.
- **The MLP is the worst and least stable** (std 0.143).
  - Early stopping halted at 45–96 iterations on a random 10 % row split. That split is the same leaky pattern: validation rows are near-copies of training rows, so its stopping point says little about unseen cells.
  - A possible fix for later: stop early on a held-out *training cell*.
- The spread across cells is large (std 33–73 % of mean RMSE). With 8 cells, rankings closer than about 0.01–0.02 RMSE are not meaningful.

**Block 3: figures** (`python -m experiments.e1_figures`)
- `fig1_pred_vs_true.png`: the best model (random forest), coloured by held-out cell, with a y = x line and per-cell RMSE in the legend.
- `fig1b_gpr_uncertainty.png`: 25C01 truth vs GPR mean ± 2σ. It refits the GPR exactly as in the E1 fold.

**Block 4: README**
- Research question, dataset credit, method, E1 table, Figure 1, observations, how to reproduce, limitations, and a "work in progress: frequency-budget experiments next" line.
- The table was **generated from `results/e1_full_spectrum.csv`**, not typed.
- Checking the claims against the outputs fixed two of my own wordings:
  - the spread range (33–73 %, not 30–70 %);
  - the setup of the hour-long GPR fit (every 2nd spectrum, not every spectrum).

## Day 4, Block 1: Frequency selectors

**Code** (`src/frugal_eis/selection.py`)
- `rank_mrmr`, `rank_rfe` and `rank_ard` each return a full ranking of the 60 frequencies. `select_mrmr/rfe/ard(X, y, k)` return its first k.
- Controls: `select_logspaced(freqs, k)` and `select_random(k, seed)`.
- `FrequencySelector(method, k)` is a scikit-learn transformer that keeps the Re and −Im columns of the selected frequencies. Inside a Pipeline under `loco_evaluate`, selection is fitted on training cells only, and a test confirms it.
- Rankings are cached per training set, so E2 ranks once per fold, not once per k.

**Design decisions**
- **The unit is the frequency.** Methods rank the 120 columns; a frequency enters the ranking when either of its columns does. Keeping a frequency keeps both Re and −Im, because the instrument measures both at once.
- **mRMR is written by hand** (the plan allowed it): greedy, score = |corr(x, y)| − mean |corr(x, chosen)|. It's transparent and has no pandas dependency. It is **not** the group's A\*-mRMR.
- **RFE:** ridge with alpha = 1 on standardised columns, step 1, eliminated down to one column, which gives the full ranking.
- **ARD:** GPR on −Im only, every 4th spectrum (the 120-feature fit is infeasible, Day 3). Frequencies tied at the upper length-scale bound are ordered by |corr(−Im, SoH)|.
- **Log-spaced:** the grid is log-uniform (ratio ≈ 1.265 per step at both ends), so "evenly on a log scale" equals evenly spaced indices. k = 1 is the geometric middle.

**Tests:** 11 in `tests/test_selection.py`, 43 in total, all passing.
- The key synthetic test: one frequency is a near-copy of the best one. mRMR must pick the weaker but **independent** frequency second, while ranking by relevance alone would pick the copy. It does.

**Preview: top 5 when ranking on all 8 cells.** This is not a result; E2 ranks inside each fold.

| Method | Top 5 |
|---|---|
| mRMR | 0.040, 597, 0.020, 0.051, 1200 Hz |
| RFE | 2.74, 0.020, 17.8, 35.9, 0.671 Hz |
| ARD | 20000, 57.4, 0.165, 8.82, 953 Hz |
| log-spaced, k = 5 | 20000, 597, 17.8, 0.671, 0.020 Hz |

**[me]**
- The methods disagree a lot.
- mRMR favours the **slowest, most expensive** frequencies (0.02–0.05 Hz, i.e. 20–50 s per period). It has no notion of measurement cost, which is the motivation for E3, error vs sweep time. A cost-aware selection could be a natural extension.
- RFE recovers 17.8 Hz (Zhang's ARD frequency) and 2.74 Hz (next to his 2.16 Hz).
- ARD ranks 20 kHz first. That point lies in the inductive, cable-dominated region, which is suspicious. To check in E2: whether it survives inside the folds, and whether it actually helps.

## Day 4, Block 2: Experiment E2, error vs frequency budget

**Code:** `experiments/e2_frequency_budget.py` writes `results/e2_budget.csv` (1,584 rows = 198 configurations × 8 held-out cells). Columns: model, selector, k, seed, cell, rmse, mae, n_train, n_test, selected_freqs (Hz), alpha. Run time 37.6 min; 7.4 min of that was the ARD ranking (once per fold, then cached).

**Design decisions**
- **Budgets:** k ∈ {1, 2, 3, 5, 8, 12, 20} plus "all" (60). "All" is the same for every selector, so it is computed once.
- **Selection is fitted inside each fold on the 7 training cells** (`SelectedModel` under `loco_evaluate`), and the selected frequencies are logged per fold.
- **Model: ridge, not GPR.** The plan allowed this if time was short, but the real reason is that GPR was worse than the mean predictor in E1 (0.139 vs 0.129). Alpha is tuned by inner leave-one-cell-out, as in E1. `SelectedModel` passes the cell ids through to the tuning (a new test checks this; 44 tests now).
- **Robustness check:** the random forest (E1's best model) with the same selectors.
- **Random control:** 20 seeds per k, ridge only, to keep the run under an hour.

**Sanity check:** k = all reproduces E1 exactly (ridge 0.0894, RF 0.0767).

## Day 4, Block 3: Figure and notes

`experiments/e2_figures.py` writes `results/figures/fig2_rmse_vs_budget.png` and `results/e2_budget_summary.csv`. In the figure:
- x is k on a log axis; y is RMSE, mean ± std over the 8 held-out cells; one line per selector;
- one panel per model;
- the grey band is the random control: 10th–90th percentile over 20 seeds of the mean per-cell RMSE (ridge panel only).

Mean per-cell RMSE (from `e2_budget_summary.csv`):

| Model | Selector | k=1 | 2 | 3 | 5 | 8 | 12 | 20 | all 60 |
|---|---|---|---|---|---|---|---|---|---|
| ridge | mRMR | **0.080** | 0.087 | 0.086 | 0.084 | 0.083 | 0.086 | 0.086 | 0.089 |
| ridge | log-spaced | 0.109 | 0.094 | 0.089 | 0.091 | 0.091 | 0.092 | 0.092 | |
| ridge | RFE | 0.113 | 0.107 | 0.103 | 0.101 | 0.093 | 0.095 | 0.098 | |
| ridge | ARD | 0.109 | 0.099 | 0.096 | 0.103 | 0.100 | 0.099 | 0.095 | |
| ridge | random, median of 20 seeds | 0.110 | 0.090 | 0.095 | 0.089 | 0.091 | 0.090 | 0.090 | |
| RF | mRMR | 0.112 | 0.079 | 0.077 | 0.084 | 0.077 | 0.079 | 0.084 | 0.077 |
| RF | log-spaced | 0.122 | 0.077 | 0.082 | 0.084 | 0.077 | **0.071** | 0.076 | |
| RF | RFE | 0.109 | 0.105 | 0.102 | 0.107 | 0.093 | 0.096 | 0.084 | |
| RF | ARD | 0.136 | 0.111 | 0.105 | 0.108 | 0.108 | 0.105 | 0.079 | |

**[me] What the figure actually shows**
1. **A few frequencies are as good as all 60.** Ridge with mRMR is at or below the full spectrum (0.089) at every k. The random forest with mRMR or log-spaced frequencies matches its full spectrum (0.077) from k = 2. For SoH at 25 °C, the full 60-point sweep is mostly redundant.
2. **The one frequency that matters is slow: about 0.04–0.05 Hz.** At k = 1, mRMR chose 0.040 Hz in 6 folds and 0.051 Hz in 2, and ridge on it reaches 0.080. The random control confirms this independently. Every random single frequency at or below 0.13 Hz scores 0.078–0.083, 0.16 Hz scores 0.090, and every one at 0.26 Hz or above scores 0.099–0.117. This is the low-frequency, diffusion-dominated end of the spectrum.
3. **Selection does not clearly beat log-spacing or random beyond k = 1.**
   - For ridge, mRMR is better than log-spaced on 3–6 of 8 cells depending on k, and sits at the good edge of the random band rather than outside it.
   - For the random forest, log-spaced is as good as or better than mRMR (best overall: log-spaced k = 12, 0.071).
   - **RFE and ARD are worse than random** at most budgets for ridge, and worse than log-spaced at most budgets for both models.
   - So the honest result is: the data-driven selectors add little over a simple log-spaced grid, except mRMR at k = 1, which finds the slow frequency.
4. **The selectors are unstable across folds.** Beyond the first, slow frequency, mRMR's second pick varies between 73 Hz and 4.9 kHz depending on which cell is held out. ARD's single pick is 20 kHz in 2 of 8 folds; that is the inductive, cable-dominated region flagged on Day 4, and it did not help. Different picks giving similar errors is more evidence of redundancy.
5. **The cell-to-cell spread dominates.** The std across held-out cells (≈ 0.03–0.05) is larger than almost every gap between selectors (≈ 0.01–0.02). With 8 cells, rankings between methods are not reliable; only "few frequencies ≈ all" and "a slow frequency is best at k = 1" are consistent patterns. The catch for practice: 0.04 Hz means about 25 s per period. "Fewest frequencies" is not the same as "fastest measurement", and E3 measures exactly that trade-off.

## Day 5, Block 1: E3, estimated acquisition time

**Code**
- `src/frugal_eis/timing.py`: `sweep_time(freqs, n_periods=3, settle_s=0)` = Σ (n_periods / f + settle_s). 5 tests in `tests/test_timing.py`; 49 tests in total.
- `experiments/e3_figures.py` re-uses E2's per-fold results (no new training). It converts each fold's selected frequencies into a sweep time and writes `results/e3_time.csv` (mean ± std over folds of time and RMSE) and `results/figures/fig3_rmse_vs_time.png`.

**Assumptions (also in the docstring)**
- The estimate is a **lower bound, not a measurement**: 3 periods per frequency, one frequency after another.
- Settling, averaging and instrument overhead are ignored, and so is the 15 min rest before state V (the same for every budget).
- `selected_freqs` is stored to 4 significant digits, so times are off by < 0.05 %.

**Numbers [data, from `e3_time.csv`]**
- **Full sweep (60 frequencies): 719 s ≈ 12 min**, with RMSE 0.089 (ridge) and 0.077 (RF).
- mRMR k = 1 (0.040 or 0.051 Hz): about 70 s, RMSE 0.080 (ridge), **about 10× faster than the full sweep with no loss**.
- **Among the selectors (excluding random), the cheapest point that matches the best full-spectrum error** (RF, 0.077) is the random forest with 2 log-spaced frequencies (20 kHz and 0.020 Hz): 150 s, RMSE 0.077.
- Log-spaced with k ≥ 2 always includes 0.020 Hz, the slowest frequency (150 s alone). Its cost barely depends on k.
- **ARD picks fast frequencies**: k = 1 takes 0.07 s, k = 3 takes 6 s, but the error is 0.096–0.11 for ridge. **RFE** sits in between (38–270 s).

**[me] A lead from the random control, not a result**
- Ridge's best point overall is a *random* draw: k = 2, seed 1 = **35.9 Hz + 17.8 Hz**, RMSE 0.076 ± 0.046 in **0.25 s**.
- 17.8 Hz is the frequency Zhang et al.'s ARD ranked highest. So two mid-frequency points, about 3000× faster than the full sweep, matched the best full-spectrum model here.
- **But it was picked after the fact from 140 random draws, so its error is optimistic** (best of many). To make it a fair result, the pair must be chosen *before* looking at the test cells. Two ways:
  - a "prior-knowledge" selector that fixes Zhang's 17.8 Hz (+ 2.16 Hz) from the paper;
  - a cost-aware selector that maximises relevance per second inside each fold.
- If either holds up under leave-one-cell-out, it is the strongest finding of the project: mid frequencies (charge transfer / SEI region) carry the aging signal at a fraction of the time cost.

## Day 5, Block 2: E4, noise robustness

**Code**
- `features.multiplicative_noise(X, σ, rng)` = X · (1 + σ·N(0,1)) per entry, i.e. Re and Im perturbed independently (1 new test, 50 in total).
- `experiments/e4_noise.py` writes `results/e4_noise.csv` (2,000 rows = 5 models × 8 cells × 5 σ × 10 seeds). Run time 7 min.
- `experiments/e4_figures.py` writes `fig4_noise_robustness.png` and `e4_noise_summary.csv`.

**Design decisions**
- **Models are trained on clean spectra; noise is added to the test spectra only.** σ ∈ {0, 0.5, 1, 2, 5} %, 10 seeds.
- **Identical noise for every model:** one noisy copy per (fold, σ, seed), seeded by (seed, fold) and given to all models.
- **Which small subset:** the lowest mean E2 RMSE with k ≤ 5, random control excluded, chosen automatically from `e2_budget_summary.csv`:
  - ridge → mRMR, k = 1 (≈ 0.04 Hz);
  - RF → log-spaced, k = 2 (20 kHz + 0.020 Hz).
  - Caveat: this choice uses E2's held-out results, so the small subsets start with a slight optimistic bias.
- Frequency selection is refitted per fold on the clean training cells, as in E2.
- **GPR full spectrum is included because the plan asks for it**: E1's setup (−Im only, every 4th spectrum). Ridge and RF are compared too, because GPR is not the competitive model here.

**Sanity check:** σ = 0 reproduces E1/E2 (GPR 0.139, RF 0.077, ridge 0.089, ridge mRMR k = 1 0.080, RF log-spaced k = 2 0.077).

**Results: mean per-cell RMSE [data, `e4_noise_summary.csv`]**

| Model | σ = 0 | 0.5 % | 1 % | 2 % | 5 % |
|---|---|---|---|---|---|
| ridge, 60 freq. | 0.089 | 0.089 | 0.090 | 0.090 | 0.091 |
| ridge, mRMR 1 freq. | 0.080 | 0.080 | 0.080 | 0.083 | **0.098** |
| RF, 60 freq. | 0.077 | 0.076 | 0.076 | 0.075 | 0.074 |
| RF, log-spaced 2 freq. | 0.077 | 0.077 | 0.078 | 0.079 | **0.095** |
| GPR, 60 freq. (−Im) | 0.139 | 0.140 | 0.142 | 0.147 | 0.159 |

Over the 10 seeds, the cell-averaged RMSE spans at most 0.012 (GPR at 5 %) and at most 0.007 for the other models.

**[me] What it shows**
1. **Up to 1 % noise, the small subsets lose nothing** (≤ 0.001 RMSE) and stay better than or equal to the full spectrum.
2. **From 2 % they start to degrade, and at 5 % they lose their advantage:** +0.018 (ridge, 1 freq.) and +0.019 (RF, 2 freq.). Ridge on 60 frequencies is then better than ridge on 1 (0.091 vs 0.098).
3. **The full spectrum is almost immune** (ridge +0.002 at 5 %). Sixty frequencies carrying redundant information average independent noise out; one or two frequencies cannot. **This is the price of frugality: fewer frequencies, less averaging.** The same logic says a frugal sweep could spend the saved time on repeated readings of its few frequencies.
4. **RF on the full spectrum gets slightly *better* with noise** (0.077 → 0.074), driven by cells 25C04 (0.120 → 0.100) and 25C02. I read this as an artefact, not a benefit: noise pushes the trees' outputs towards the middle of the training range, which happens to help cells the forest over- or under-predicts. Do not report it as noise "helping".
5. **GPR is the worst at every level and degrades the most** (+0.020 at 5 %), consistent with E1: its fit has a very small noise term, so it is sensitive to inputs away from the training spectra.
- **Open:** what noise level is realistic for this instrument? I have not checked the instrument's accuracy spec. If it is around 1 % or better, the frugal subsets are safe; that needs a source before it goes in the report.

## Day 5, Block 3: Where on the spectrum do the selectors pick?

**Code:** `experiments/e5_selected_on_nyquist.py` writes `results/figures/fig5_selected_on_nyquist.png` and `results/e5_selected_k5.csv`. No new training; it uses E2's per-fold k = 5 selections.

**Design decisions**
- **Typical spectrum:** cell 25C01 (the longest-lived) at SoH ≈ 0.90, i.e. cycle 22, mid-life.
- **Which 5 per selector:** selections differ between folds, so the figure marks the 5 frequencies each selector chose **most often over the 8 folds**. Because that hides the picks that change from fold to fold, the table below also counts **all 40 per-fold picks**.
- **Regions are read off the spectrum, not fitted with a circuit** [data + me]:
  - **inductive / ohmic:** f ≥ 9.9 kHz, where −Im(Z) crosses zero; the real-axis intercept ≈ R0;
  - **SEI + charge transfer:** 9.9 kHz > f > 1.07 Hz, the depressed arc (apex ≈ 36 Hz). The SEI and charge-transfer arcs overlap here, so they are not separated;
  - **diffusion:** f ≤ 1.07 Hz, below the valley between arc and tail; the rising Warburg-like tail.
  - The boundaries come from this one spectrum and shift a little with aging.
- First version bug: I located "the top of the arc" as the global maximum of −Im, but in aged spectra the tail ends higher than the arc. I fixed it by taking the first local maximum.

**Results [data, `e5_selected_k5.csv` and script output]**

| Selector | Most-often-chosen 5 (Hz, folds out of 8) | All 40 per-fold picks: ohmic / arc / diffusion |
|---|---|---|
| mRMR | 0.064 (3), 0.051 (7), 0.040 (8), 0.032 (3), 0.020 (4) | 0 / 14 / 26 |
| RFE | 472 (6), 11.1 (4), 2.74 (5), 0.53 (3), 0.020 (5) | 1 / 27 / 12 |
| ARD | 20000 (2), 57.4 (3), 45.4 (4), 3.46 (3), 0.165 (4) | 5 / 27 / 8 |
| log-spaced (fixed) | 20000, 597, 17.8, 0.67, 0.020 (8 each) | 8 / 16 / 16 |

**[me] Answer to "which part do they pick?"**
- **mRMR picks the diffusion tail**: its 4–5 most stable picks are all between 0.02 and 0.064 Hz. The 14 arc picks are each fold's single "different" frequency, a different arc point in almost every fold (73 Hz to 4.9 kHz, see Day 4). Physically, this matches lithium transport in the electrodes slowing down with aging, but that is an interpretation, not tested here.
- **RFE and ARD mostly pick the SEI / charge-transfer arc.** That is where Zhang et al.'s ARD found 17.8 Hz and 2.16 Hz, and where Kallel et al.'s R_SEI / (R_SEI + R_ct) indicator lives. ARD also takes 20 kHz in some folds (inductive, cable-dominated; suspicious since Day 4).
- **Almost nobody picks the ohmic region.** Only ARD (5/40) and log-spaced, by construction. This agrees with E1, where the R0-only physics baseline (0.108) was well behind the spectrum-based models.
- **This ties the selectors to the cost result (E3):** mRMR is the most accurate at k = 1–5 *because* it uses the slow diffusion tail, which costs 47–150 s per point (3 periods at 0.064–0.020 Hz), while the arc frequencies cost under a millisecond to about 3 s. The open lead from E3 (35.9 + 17.8 Hz, arc apex region) is exactly the cheap alternative to test fairly.

## Open questions / TODO
- [x] Supplementary Fig. 1 of Zhang 2020: definitions of states I–IX (table in section 1)
- [x] Capacity ↔ impedance pairing: impedance cycle n ↔ discharge capacity of capacity cycle n − 1 (see Block 4)
- [x] Why the tutorial CSV has fewer cycles per cell: it doesn't. Lifetimes vary from 35 to 275 labelled spectra
- [ ] The paper says capacity is measured on odd cycles and impedance on even cycles, but the files number them 1, 2, 3, … How does the file numbering relate to the paper's? (Low priority; the pairing reproduces the tutorial.)
- [ ] Read the full text of Kallel 2022 (in a browser)
- [x] `25C42`: one state VI file only, not among the paper's 12 cells, so ignored
