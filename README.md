# US heatwaves, soil moisture regimes, and BAM preconditioning

Spring atmospheric variability leaves an imprint on the land surface, but how this imprint conditions summer surface heating remains unclear. Here, we identify a recurring spring soil-moisture pattern associated with the **baroclinic annular mode (BAM)**, which describes fluctuations in midlatitude storm activity. Negative early-April BAM states are followed by a circumglobal zonal wavenumber-5 circulation pattern and a sustained precipitation deficit over the central United States, providing a pathway for the development and persistence of spring soil-moisture anomalies into summer. Moisture deficits persist into summer, accompanied by stronger inverse coupling between soil moisture and sensible heat flux and increased heatwave event frequency. During heatwaves, soils are drier across contrasting climatological moisture regimes, with pronounced suppression of evaporative cooling where soil moisture falls below the critical threshold. These findings support land-surface priming as a pathway linking spring atmospheric variability to summer coupling and heatwave susceptibility, connecting seasonal preconditioning with individual extremes across the weather–climate continuum.

This repository supports the ERA5 + GLEAM–based analysis behind that narrative (heatwave catalogs typically 1950–present; BAM composites from **1980** onward). Paths and case names are set in `src/config_v2.yaml`.

---

## Scientific analysis (brief)

1. **Prepare fields** — GLEAM SMrz/SMs/E/H (and related) over the US; ERA5 precipitation and Z500 anomalies (annual-cycle window removal); monthly / smoothed daily SM anomalies for EOF input.
2. **Detect** contiguous US summer heatwaves with Teng-style object tracking (percentile threshold, minimum area/duration, limited center-of-mass motion).
3. **Diagnose spring SMrz EOFs** (Apr–May smoothed anomalies; NCL `2_EOF.ncl`) and classify years by standardized mean PC\* (late April + all May): positive / weak / negative extremes; compare JJA heatwave maps by phase.
4. **Map SM–LE regimes** — breakpoint fits on evaporative / LE–SM relationships, then spatial regime separation maps.
5. **Regional regime diagnostics** on the heatwave cluster footprint (all HW days / area), not only the max-intensity point.
6. **Coupling** — Dirmeyer-style sensitivity index \(S \propto b_f \times s_w\) (EF vs SM anomalies by month).
7. **BAM–PC\* timing** — correlate daily BAM on each April calendar day with late-April + all-May EOF3 PC\* (years ≥ 1980); highlights the early-April / BAMnet window.
8. **BAM persistence (BAMnet)** — over **5–17 Apr** (23 Apr − 12 d ± 6 d), with |BAM| > 1.5:
   - \(M_\pm = f_\pm \times \mathrm{mean}(|\mathrm{BAM}|)\) on days beyond ±1.5  
   - **BAMnet** \(= M_+ - M_-\) (peak- vs trough-dominated)
9. **Circulation** — regress late-April + all-May mean Z500′ (zonal \(k \ge 4\)) on standardized EOF3 PC\*; related BAM composites.
10. **Heatwave odds** — JJA event-count ratios for years with \(M_\mathrm{net} > 0\) vs \(M_\mathrm{net} < 0\) (Mₙₑₜ window: late April + all May, |BAM| > 1.5).
11. **Precipitation** — central-US accumulated PR′ time series and running-accumulation maps around the BAMnet window (pos vs neg years).

### Main outputs


| Output | Location (typical) |
| ------ | ------------------ |
| Heatwave event catalog | `{path_case_land}/Heat_waves_events_list.csv` (+ masks / SAT maps) |
| GLEAM / ERA5 anomaly NetCDFs | `{path_outputs_land}/` (SMrz, PR′, Z500′, smoothed Apr–May SM, …) |
| SMrz EOF / PC timeseries | `{path_outputs_land}/EOF_timeseries_…_SMrz_…_AprMay_US.nc` |
| Breakpoint / regime fields | `{path_outputs_land}/` + maps under `{path_case_land}/Figures/` |
| Coupling (SI) fields | `{path_outputs_land}/` |
| BAM / EOF / Z500 / PR figures | `{path_case_land}/../Figures/` or `src/Figures/` (script-dependent) |


---

## Data flow

```
GLEAM (SMrz, SMs, E, H, …) + ERA5 (TS, PR, Z500) + BAM index
        │
        ▼
[0] Field prep (anomalies, smoothing) ──► NetCDFs on scratch
        │
        ├─► [NCL] 2_EOF.ncl ──► SMrz EOF / PC file
        │
        ▼
[1] Detect & track HWs ──► Heat_waves_events_list.csv
        │
        ▼
[2] EOF3 / PC* year classes + HW maps
        │
        ▼
[3] Breakpoints (LE–SM) ──► [3] Regime maps
        │
        ▼
[4] Regional SM–LE densities on HW footprint
        │
        ▼
[5] Coupling sensitivity (EF–SM)
        │
        ▼
[6] Daily BAM vs EOF3 PC* (calendar-day corr)
        │
        ▼
[6b] BAMnet persistence vs SMrz′ / PC*
        │
        ├──────────────────┐
        ▼                  ▼
[7] Z500′ ~ EOF3 PC*   [8] HW ratio by M_net sign
        │
        ▼
[9] Precipitation evolution (BAMnet pos / neg)
```

**Order:** edit `src/config_v2.yaml`, then run `src/run_analysis.sh` (full pipeline). Compute EOFs with `2_EOF.ncl` after step 0 (and before step 2) if the EOF NetCDF is not already on scratch. Run `6_BAM_precondition.py` before `6_BAM_phases_persistent_precondition.py`.

---

## How to run

Assumes a UNIX environment (SLURM on Negishi/Purdue-style cluster). Set paths, conda env, and `case` in `config_v2.yaml` (and in the header of `run_analysis.sh` for output directories) before submitting.

```bash
cd src

# Full land–atmosphere + BAM pipeline (steps 0–9)
sbatch run_analysis.sh
# or: bash run_analysis.sh
```

**`run_analysis.sh` runs (in order):**

`0_data.py` — GLEAM/ERA5 prep, anomalies, smoothed SM  
`1_heatwaves_detection_teng_clusters_object.py` — Teng-style US heatwave catalog  
`2_analysis_EOF_meanPC_AprMay.py` — Apr–May SMrz EOF / PC\* diagnostics and HW maps  
`3_breakpoints_optimized_LE.py` — LE–SM breakpoint fits  
`3_map_separation_regimes_LE.py` — spatial SM–LE regime maps  
`4_analysis_regimes_LE_allHWdays_allHWarea_regional.py` — regimes on HW footprint  
`5_coupling_conditional.py` — Dirmeyer-style coupling sensitivity  
`6_BAM_precondition.py` — daily BAM vs late-Apr+May EOF3 PC\* (calendar-day correlation)  
`6_BAM_phases_persistent_precondition.py` — BAMnet persistence vs SM / PC\*  
`7_z500_regression_BAM_EOF3.py` — Z500′ regression on EOF3 PC\*  
`8_prob_HW_BAM_persistent_precondition_thres125_lateApr_allMay.py` — JJA HW ratios by Mₙₑₜ  
`9_precipitation_evolution_BAM.py` — PR′ composites around the BAMnet window  

Scripts that take `--config` use `config_v2.yaml`. Scripts `6_BAM_precondition.py`, `6_BAM_phases_persistent_precondition.py`, 8, and 9 load that file from the script directory automatically.

**Prerequisite (EOFs):** after `0_data.py` has written smoothed Apr–May SMrz anomalies, run `2_EOF.ncl` (NCL) to produce `path_file_eof` if it does not already exist.

Dependencies: see `environment.yml`. External data paths are listed in `data_sources.md` and `config_v2.yaml`.

---

## Files in `src/`


| File | What it does | Useful beyond this paper? |
| ---- | ------------ | ------------------------- |
| `0_data.py` | Build US GLEAM stacks and ERA5 PR/Z500 (and SM) anomaly NetCDFs. | Yes — reusable field prep. |
| `1_heatwaves_detection_teng_clusters_object.py` | Object-based US heatwave detection (Teng-style), severity, event CSV. | Yes — reusable heatwave catalog builder. |
| `2_EOF.ncl` | NCL EOF/PC of smoothed Apr–May SMrz anomalies. | Yes — drop-in EOF step for other SM fields. |
| `2_analysis_EOF_meanPC_AprMay.py` | PC\* year classes, EOF maps, HW composites by phase. | Yes — template for SM-mode vs HW typing. |
| `3_breakpoints_optimized_LE.py` | Optimize LE–SM breakpoints (multiprocess). | Yes — regime breakpoint engine. |
| `3_map_separation_regimes_LE.py` | Map separated SM–LE regimes. | Yes — spatial regime visualization. |
| `4_analysis_regimes_LE_allHWdays_allHWarea_regional.py` | SM–LE densities over the HW cluster footprint. | Yes — event-footprint regime stats. |
| `5_coupling_conditional.py` | Monthly EF–SM sensitivity index (Dirmeyer-style). | Yes — coupling SI pipeline. |
| `6_BAM_precondition.py` | Calendar-day correlation of daily BAM vs late-Apr+May EOF3 PC\*. | Yes — BAM–SM timing diagnostic. |
| `6_BAM_phases_persistent_precondition.py` | BAMnet (5–17 Apr, \|BAM\|>1.5) vs SMrz′ / EOF3. | Yes — BAM persistence indices. |
| `7_z500_regression_BAM_EOF3.py` | Z500′ (\(k\ge4\)) regressed on EOF3 PC\*; BAM-related maps. | Yes — circulation–EOF regression. |
| `8_prob_HW_BAM_persistent_precondition_thres125_lateApr_allMay.py` | JJA HW event ratios for Mₙₑₜ pos vs neg. | Yes — conditional HW probability maps. |
| `9_precipitation_evolution_BAM.py` | Central-US PR′ TS and accumulation maps by BAMnet sign. | Yes — precip composites about a BAM window. |
| `Functions.py` | Shared helpers (anomalies, annual cycle, maps, FFT filter, I/O). | Yes — utility library for related scripts. |
| `config_v2.yaml` | Paths, case name, BAM/EOF windows, optional CLI defaults. | Yes — single config for the pipeline. |
| `run_analysis.sh` | SLURM driver for steps 0–9 in order. | Yes — end-to-end batch recipe. |


---

## Notes

- Edit **`config_v2.yaml`** before running: `path_outputs_land`, `path_case_land`, `path_bam`, `path_file_eof`, `z500_anom_file`, GLEAM/ERA5 inputs, and `apr_day_start` / `apr_day_end`.
- **BAMnet window** (scripts `6_BAM_phases_persistent_precondition.py` and 9): 5–17 April, \|BAM\| > 1.5, years ≥ 1980. **Mₙₑₜ for HW ratios** (script 8) uses late April + all May with the same \|BAM\| threshold. Script `6_BAM_precondition.py` uses the same PC\* window and shades 5–17 Apr on the calendar-day correlation plot.
- Central-US box used in several figures: **256–270°E, 27–41°N**.
- Large NetCDF intermediates belong on scratch (`path_outputs_land`); keep paper figures and small CSVs under the case directory or `Figures/`.
