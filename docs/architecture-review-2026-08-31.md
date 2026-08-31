# Architecture Review — Deferred Backlog

**Repo:** tesis-hakam · **Branch at review:** `fix/walkforward-cv-and-leakage`
**Reviewed:** 31 August 2026
**Status:** 🔴 **DEFERRED — do not act on this before the thesis defense.**

> Every item below is a refactor of working code. Nothing here is required to
> defend the thesis. The one exception is **C1**, which is a possible
> *correctness* problem in the SHAP chapter — read its "Before the defense"
> note, because it may be worth a 10-minute check even while everything else waits.

Ordered by strength. `Strong` = clear payoff, evidence is concrete.
`Worth exploring` = real friction, solution needs a design pass.
`Speculative` = only worth it if the situation changes.

---

## Contents

| # | Candidate | Strength | Risk if ignored |
|---|---|---|---|
| [C1](#c1--deepen-shap-into-one-explainer-module) | Deepen SHAP into one explainer module | Strong | **Thesis may be wrong**, not just stale |
| [C2](#c2--derive-the-thesis-numbers-instead-of-retyping-them) | Derive thesis numbers instead of retyping | Strong | Silent staleness on every rerun |
| [C3](#c3--collapse-artifact-loading-into-one-module) | Collapse artifact loading into one module | Strong | Broken paths, no provenance |
| [C4](#c4--reunite-the-forked-stage-b-and-figure-code) | Reunite forked Stage B and figure code | Strong | Official vs exploratory results diverge |
| [C5](#c5--one-variable-registry-one-transform) | One variable registry, one transform | Strong | Already fired once (`44247c9`) |
| [C6](#c6--give-the-leakage-guards-a-test-surface) | Give the leakage guards a test surface | Worth exploring | Regressions surface 45 min into a grid run |
| [C7](#c7--replayable-revisions-instead-of-a-chain-of-binaries) | Replayable revisions instead of binaries | Speculative | Only if another revision round comes |

**Suggested order after the defense:** C1 → C3 → C5 → C2 → C4 → C6 → C7.
(C3 and C5 are small and unblock C2; C1 stands alone and comes first because
it is the only one that can change a published number.)

---

## C1 · Deepen SHAP into one explainer module

**Strength:** Strong · **Dependency category:** in-process

### Files

| File | Location | What it does |
|---|---|---|
| `modelling/03_shap_analysis.ipynb` | cell 8, cell 10 | `build_lag_matrix` — **assumes** feature order |
| `modelling/analyze_shap_direction.py` | `:27–42`, `:74–89` | `build_lag_matrix` — **assumes** feature order |
| `modelling/create_shap_lag_example_docx.py` | `:89–109`, `:125–135` | `build_exact_darts_matrix` — **reads** feature order |

### Problem

Three implementations reconstruct the fitted model's lag-feature matrix
independently, and they do not agree.

Two of them build the matrix as *all target lags first, then each covariate's
lags in column order*:

```python
for lag in range(1, lags + 1):
    row.append(target[t - lag])
    names.append(f"IHSG_lag_{lag}")
for col, name in enumerate(covariate_names):
    for lag in range(1, lags + 1):
        row.append(covariates[t - lag, col])
        names.append(f"{name}_lag_{lag}")
```

The third reads the ordering off the fitted model via
`model.lagged_feature_names`, and carries a comment stating that Darts
**interleaves** covariates rather than grouping them.

**At most one of these can be right.** If the third is correct, then the SHAP
values behind Tabel 4.9 and the entire signed-SHAP / arah-SHAP revision are
attributed to the wrong variables — NPL_Ratio's contribution could actually
belong to Tin, and so on.

Two secondary divergences in the same code:

- `03_shap_analysis.ipynb` subsamples with `sample_size = min(500, len(X_test))`;
  both scripts explain the full 389-observation matrix. The notebook's figures
  and the scripts' CSVs are therefore **not numerically comparable**, though the
  thesis presents them as one analysis.
- The Darts-unwrapping hack and the SHAP squeeze are copy-pasted three times:
  ```python
  underlying = model.model
  if hasattr(underlying, "__iter__") and not hasattr(underlying, "predict"):
      underlying = underlying[0]
  explainer = shap.TreeExplainer(underlying)
  raw = np.asarray(explainer.shap_values(X))
  if raw.ndim == 3:
      raw = raw[0]
  ```

The `test_start = max(PROD_TEST_START, WINDOW + 1)` guard is likewise
triplicated (`03` cell 8; `analyze_shap_direction.py:69`;
`create_shap_lag_example_docx.py:124`).

### ⚠️ Before the defense

The additivity check passing (`1.33e-15`) does **not** rule this out — additivity
holds regardless of how you *label* the columns. Reconstruction can be perfect
while every attribution is mislabelled.

A cheap check that does not require any refactor:

```python
# in modelling/, with the production model fitted as in analyze_shap_direction.py
assert list(model.lagged_feature_names) == feature_names, \
    "assumed lag order does not match darts"
```

If that assertion passes, C1 is pure tidiness and can wait with the rest.
If it fails, the SHAP tables need regenerating before the numbers are defended.

### Solution

One module — `modelling/shap_lib.py` — exposing a single interface:

```python
explain_winner() -> (contributions, base_value, predictions)
# contributions indexed by (date, variable, lag)
```

The implementation loads the winning configuration, refits, reads the feature
order **off the model once**, unwraps Darts, squeezes the SHAP array, and
verifies additivity. Three adapters consume it: the importance table (`03`),
the direction study (`analyze_shap_direction.py`), and the per-lag example
(`create_shap_lag_example_docx.py`).

### Wins

- Feature order read, not assumed
- Locality: misattribution concentrates in one module
- Darts unwrap stops leaking into every consumer
- One sample rule across all three consumers
- The interface becomes the test surface
- Delete two divergent implementations

### Done when

- [ ] `model.lagged_feature_names` is the only source of feature order
- [ ] `build_lag_matrix` exists zero times outside `shap_lib`
- [ ] Notebook and scripts explain the same observation set
- [ ] Regenerated SHAP tables compared against the current Tabel 4.9

---

## C2 · Derive the thesis numbers instead of retyping them

**Strength:** Strong · **Dependency category:** in-process

### Files

| File | Result literals |
|---|---|
| `draft/build_tesis_docx.py` | 24 |
| `draft/revise_final_no_gdp_benchmark_shap.py` | 17 |
| `draft/buat_final_draft.py` | 9 |
| `draft/create_notulen_response_{docx,pdf}.py` | 6 |
| `draft/revise_v3_mape_benchmark_rf.py` | 2 |
| **Total** | **58** |

Plus `README.md`, which still reports 21 covariate scenarios and 2,520 fits —
the pre-GDP-exclusion figures.

Counted literals: `432` ×11, `0,6360` ×10, `2.408` ×8, `504`, `14 kovariat` ×5,
`18 skenario` ×2, and the `0,63xx` MAPE variants.

### Problem

The results are retyped as string literals into the document-generation code,
with no link back to `stage_a_summary.csv`. Any rerun of Stage A silently
invalidates the thesis, and the scope decision survives only as a string
replacement:

```python
# draft/revise_final_no_gdp_benchmark_shap.py:217
"15 kovariat": "14 kovariat",
"504 kombinasi": "432 kombinasi",
```

Meanwhile the authoritative scope statement — which covariates are excluded —
lives in a docx-rewriting script rather than in `cv_lib`:

```python
# draft/revise_final_no_gdp_benchmark_shap.py:33
GDP_SCENARIOS = {"GDP", "All_Macro_no_UST", "All_Covariates"}
```

…while `cv_lib.py:158` still asserts the pre-revision count:

```python
assert len(SCENARIO_COVARIATES) == 21, f"expected 21 scenarios, got {len(SCENARIO_COVARIATES)}"
```

### Solution

One module — `modelling/thesis_facts.py` — that reads `stage_a_summary.csv`,
`nested_cv_*.csv`, `descriptive_stats_transformed.csv`, and the
`shap_direction_*` CSVs, and exposes named facts with Indonesian decimal
formatting (`0,6360` not `0.6360`). The docx scripts and README interpolate
those facts instead of retyping them.

Derived, never typed: `n_covariates`, `n_scenarios`, `n_configurations`,
`n_observations`, `winner_mape`, `winner_model`, `winner_covariates`.

### Wins

- Leverage: one interface, 58 call sites
- Scope counts computed, not asserted
- Reconciliation becomes a test instead of a chore
- Locality: stale numbers fail loudly
- README stops drifting from results

### Done when

- [ ] No `432` / `0,6360` / `2.408` literal in any `draft/*.py`
- [ ] `cv_lib` holds the single scope declaration; `== 21` assert is derived or gone
- [ ] README figures regenerate from the same module

---

## C3 · Collapse artifact loading into one module

**Strength:** Strong · **Dependency category:** in-process

### Files — 9 copies of one line

| File | Location |
|---|---|
| `modelling/02c_model_selection_cv.ipynb` | cell 1 |
| `modelling/02d_nested_tuning_cv.ipynb` | cell 1 |
| `modelling/03_shap_analysis.ipynb` | cell 1 |
| `modelling/04_visualization.ipynb` | cell 1 |
| `modelling/explore_lib.py` | `:36` |
| `modelling/plot_lightgbm_tuned.py` | `:21` |
| `modelling/analyze_shap_direction.py` | `:46` |
| `modelling/create_shap_lag_example_docx.py` | `:113` |
| `modelling/make_descriptive_stats.py` | `:54` |
| `draft/figures/make_thesis_diagrams.py` | `:167` |

```python
sorted(glob.glob("saved_models/df_merged_*.joblib"), reverse=True)[0]
```

### Problem

Nine call sites each re-derive which `df_merged` is current, under four
different path conventions — bare relative `"saved_models/…"` (breaks unless the
process runs from `modelling/`), `HERE / "saved_models"`, `ROOT / "saved_models/…"`,
and `os.path.join(modelling_dir, …)`. It is a silent latest-wins rule, and no
downstream artifact records which run it actually read.

**Already broken:** `plot_lightgbm_tuned.py:22` loads
`saved_models/explore_lightgbm_screening1_w20h1_tuning.joblib`, which does not
exist — only `explore_randomforest_wti_w120h1_tuning.joblib` is present.
Conversely `explore_randomforest_wti_w120h1_{folds,summary}.csv` exist in
`modelling/` but that configuration is not in `run_explore_tuning.py`'s
`CONFIGS` list.

### Solution

`modelling/artifacts.py`, resolving `saved_models/` from its own file location:

```python
artifacts.current_merged()   # newest df_merged, with recorded provenance
artifacts.final_config()
artifacts.tuning()
```

### Wins

- Leverage: one interface, 9 call sites
- Working-directory coupling disappears
- Provenance recorded once
- Missing artifacts fail in one place, with one message
- Deletion test: concentrates

### Done when

- [ ] `glob.glob` appears once in `modelling/`
- [ ] `plot_lightgbm_tuned.py`'s missing joblib is resolved or the script retired
- [ ] `run_explore_tuning.py` CONFIGS matches what is actually on disk

---

## C4 · Reunite the forked Stage B and figure code

**Strength:** Strong · **Dependency category:** in-process

### Files

| Pair | Duplication |
|---|---|
| `02d_nested_tuning_cv.ipynb` cells 3/5/7 ↔ `explore_lib.py:51–152` | Line-for-line |
| `04_visualization.ipynb` cell 9 ↔ `plot_lightgbm_tuned.py:51–84` | IEEE `rcParams` block, identical |
| `cv_lib.py:344–352` ↔ `cv_lib.py:448–458` | Price-reconstruction block, internal |

### Problem

The Stage B nested-CV loop exists twice. The duplicate spans the fold-boundary
assertion, the resume-from-checkpoint block, `optuna.create_study(direction="minimize",
sampler=TPESampler(seed=42))`, the 11-key `outer_results.append({...})` dict, and
the `summary = {...}` dict — byte-identical. The official Stage B result and the
exploratory result can therefore drift apart without anyone editing either on
purpose, which defeats the point of running the exploratory pass as a comparison.

The same pattern repeats in the figure code (`IEEE_WIDTH, IEEE_HEIGHT, FONT_PT`
and the full `rcParams` dict) and inside `cv_lib` itself, where
`inverse_and_metrics` and `predict_fold` each carry their own copy of the
log-price reconstruction.

Related: `04_visualization.ipynb` cell 3 and `plot_lightgbm_tuned.py:47` both
recompute MAPE inline as `np.mean(np.abs((actual - predicted) / actual)) * 100`
instead of calling `cv.compute_metrics`.

### Solution

Move the nested-CV loop into `cv_lib` as
`run_nested_cv(model, covariates, window, horizon, out_prefix)`. Both the
notebook (Stage B winner) and `run_explore_tuning.py` (non-winners) become
adapters that call it with a configuration. Extract the `rcParams` block into a
single figure-style module, and factor the reconstruction block out of the two
`cv_lib` functions that share it.

### Wins

- Official and exploratory results stay comparable by construction
- Notebook shrinks to a call
- Locality: tuning bugs concentrate
- One figure style, two figures
- Deletion test: concentrates

---

## C5 · One variable registry, one transform

**Strength:** Strong · **Dependency category:** in-process

### Files

| File | Location |
|---|---|
| `modelling/cv_lib.py` | `:31–32` — canonical |
| `modelling/make_descriptive_stats.py` | `:26–28` registry, `:63–66` own transform |
| `modelling/00_data_preprocessing.ipynb` | cell 2 — third declaration |

### Problem

`LEVEL_VARS` / `RATE_VARS` are declared three times, verbatim. `cv_lib`'s own
docstring names the risk without removing it: *"verbatim from
00_data_preprocessing.ipynb / 02b_default_params.ipynb so downstream notebooks
don't drift from them."*

Worse, `make_descriptive_stats.py` re-implements the transform:

```python
transformed[col] = np.diff(np.log(df[col].dropna().values))   # per column
```

while the pipeline goes through `cv_lib.fit_target_transform` /
`fit_covariate_transform` — `fill_missing_values` on a business-day-regularised
`TimeSeries`, then `Diff`, then a train-only `Scaler`. Two code paths, one
table. **Tabel 4.1 therefore describes data the pipeline never fits on.**

### This already fired

Commit `44247c9` (31 Aug 2026) reconciled
`modelling/descriptive_stats_transformed.csv` by hand: the tracked version had
`N=2442` and no `US_Treasury_10Y` row, while the working tree — which matched
the submitted proposal's Tabel 4.1 — had `N=2407` and included it. The tracked
file was a pre-leakage-fix snapshot that nothing forced to be regenerated.

### Solution

`make_descriptive_stats.py` imports the registry and the transform from
`cv_lib` instead of restating both. `00_data_preprocessing.ipynb` imports the
registry too, rather than declaring it.

### Wins

- Category drift becomes impossible
- One transform, two consumers
- Locality: Tabel 4.1 tracks the fit
- Manual CSV reconciliation stops recurring

---

## C6 · Give the leakage guards a test surface

**Strength:** Worth exploring · **Dependency category:** local-substitutable

### Files

- `modelling/cv_lib.py:500–518` — the only self-check in the repo
- No test module anywhere in the tree

### Problem

The guards that justify the entire methodology revision run as inline asserts
inside a 45-minute grid, or never:

| Guard | Where it runs today |
|---|---|
| Fold math (`expanding_window_folds`) | ✅ `__main__` self-check |
| Embargo produces a real date gap (`assert_embargo_gap`) | grid run only |
| Transform fitted on train only (`fit_target_transform`) | grid run only |
| Nested-CV isolation (`make_inner_cv_objective`) | grid run only |
| Metrics (`compute_metrics`) | never |
| Inverse transform / price reconstruction | never |

A regression in any of these surfaces 45 minutes into a run — or after the
numbers are already in the document. For a thesis whose contribution *is* the
leakage correction, that is the weakest point to be asked about in a defense.

### Solution

`modelling/test_cv_lib.py` driving `expanding_window_folds`,
`assert_embargo_gap`, `fit_target_transform`, `compute_metrics`, and
`make_inner_cv_objective` over short synthetic series. The seam is already
parameterised by integer index, so no Darts fit is required and the whole suite
runs in seconds.

### Wins

- Leakage guards run in seconds
- The interface is the test surface
- Regressions caught before the grid
- Defensible to the examiners

---

## C7 · Replayable revisions instead of a chain of binaries

**Strength:** Speculative · **Dependency category:** ports & adapters

### Files

- `draft/revise_final_no_gdp_benchmark_shap.py` (v1 → v2)
- `draft/revise_v3_mape_benchmark_rf.py` (v2 → v3)
- `draft/sisipkan_gambar_bab2.py`, `draft/perbarui_daftar_pustaka.py`
- 13 `final-draft-tesis-*.docx` snapshots in `draft/`

### Problem

Each revision is a one-shot script that string-replaces into the *previous*
`.docx`, so the thesis state lives in a chain of binaries no one can diff and no
script can safely re-run. The backup files are the version-control system.

### Solution

Express each revision as an ordered list of edits replayed from one base
document, so re-running is idempotent and the edits themselves are reviewable
in git.

### Wins

- Revisions become reviewable
- Re-running is idempotent
- Backups stop multiplying

### Why speculative

The thesis may be finished before the payoff lands. Worth doing only if another
revision round is actually coming.

---

## Vocabulary used here

**module** · **interface** · **implementation** · **depth / deep / shallow** ·
**seam** · **adapter** · **leverage** · **locality**

*Deletion test:* would deleting this module concentrate complexity, or merely
move it? Only "concentrates" justifies the refactor.
*Two adapters rule:* one adapter is a hypothetical seam; two make it real.

---

*Generated by `/improve-codebase-architecture`, 31 August 2026.*
