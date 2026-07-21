# ADR 043: Tyche uncertainty provider

- Status: Accepted
- Date: 2026-07-20
- Change class: [major] [arch]

## Context

Kwavers Analysis and the PINN solver independently implemented corrected
quantiles, means, variances, parameter sampling, sensitivity labels, and
calibration result storage. The Analysis rank selected the lower tail as
confidence increased. The PINN ensemble used `E[x²]-E[x]²`, which loses
variance through cancellation. Analysis also returned only the first member of
a multi-prediction interval request, represented absent calibration statistics
as zeroes, and labeled squared Pearson correlation as both first-order and total
Sobol indices. Its Morris path changed multiple parameters per step and divided
effects by a hard-coded parameter count.

Tyche is the Atlas owner for reproducible sampling, online moments,
correlation screening, and finite-sample conformal calibration. Its corrected
rank implements the empirical quantile in Section 1 and Theorem 1 of
[Angelopoulos and Bates](https://arxiv.org/abs/2107.07511):
`ceil((n+1)(1-alpha))`, capped at `n`. Tyche ADR 0001 records the exchangeable
rank proof at public revision
[`2b8fb14`](https://github.com/ryancinsight/tyche/blob/2b8fb14267a710e1438102666211494a3d6f179e/docs/adr/0001-reproducible-study-boundary.md#conformal-coverage).

## Decision

Kwavers depends on `tyche-core` at merged revision `2b8fb14`.

- Analysis computes absolute errors and even medians in prediction-native
  `f32`, then widens the completed score exactly for
  `ConformalCalibrator<f64>`. Returned scores use `Cow::Borrowed`; all requested
  prediction arrays remain present in each interval family.
- PINN conformal state stores a validated `ConformalCalibrator<f32>`. Its public
  miscoverage configuration becomes `f32`, matching model outputs and scores;
  no widen-compute-narrow path remains.
- PINN ensemble summaries use `Moments<f32>` with `PopulationVariance`. A
  dedicated statistics leaf owns validation and the Welford oracle boundary.
- Global sensitivity becomes const-generic correlation screening over Tyche
  `ParameterSpace`, deterministic `LatinHypercube`, and
  `CorrelationScreening`. It reports squared correlations by their actual name.
- The local pseudo-Sobol, bootstrap, and Morris implementations are removed.
  Genuine Morris and Saltelli/Sobol methods remain a Tyche provider capability
  gap; Kwavers does not retain mislabeled substitutes.
- Undefined pre-calibration distributions and zero-width coverage efficiency
  use `Option`, not fabricated numeric values.

The dependency direction is
`kwavers-analysis|kwavers-solver -> tyche-core -> eunomia`. Physics, model
inference, Leto arrays, and uncertainty presentation remain Kwavers-owned.

## Public migration

- `ConformalResult` now carries a score lifetime. Its interval values change
  from one `(Array2, Array2)` pair to `PredictionIntervalBatch` with aligned
  lower/upper vectors, its score array becomes `Cow<[f64]>`, and
  `coverage_probability` becomes `target_coverage_probability`.
- `CalibrationSummary::score_distribution` becomes
  `Option<ScoreDistribution>`.
- `ConformalValidationMetrics::coverage_efficiency` becomes `Option<f64>`.
- `PinnUncertaintyConfig::{conformal_alpha, variance_threshold}` change from
  `f64` to `f32`. `PinnConformalPredictor::new` becomes fallible; direct
  callers propagate its `KwaversResult`.
- Dynamic `SensitivityIndices` and `MorrisResults` are removed.
  `SensitivityAnalyzer::analyze` and
  `UncertaintyQuantifier::sensitivity_analysis` become const-generic and return
  Tyche `SensitivityReport<f64, PARAMETERS>` from a borrowed
  `ParameterSpace<f64, PARAMETERS>`.
- `MlUncertaintyConfig` gains `sensitivity_seed: tyche_core::Seed`.
- `UncertaintyQuantifier::generate_report` borrows
  `&[&dyn UncertaintyResult]`, and `UncertaintyReport` retains that slice
  without requiring caller boxes or collecting a duplicate reference vector.
  Type erasure remains confined to this cold heterogeneous reporting boundary.
- The workspace patches Apollo's Git source to the synchronized Atlas checkout.
  This keeps Coeus-transitive and direct FFT dependencies on one package
  identity as Coeus adopts remote provider declarations.

No compatibility wrappers or aliases preserve the superseded contracts.

## Proof and verification obligations

1. Five sorted scores at 90% confidence select the fifth score, not the first.
2. An even median between adjacent `f32` values rounds in `f32` before exact
   widening; no unrepresentable `f64` midpoint reaches prediction arithmetic.
3. The fixed confidence panel remains present and a distinct configured level
   identifies one additional interval batch by its exact probability.
4. Non-finite PINN model output reaches typed validation without executing the
   finite widened-`f32` bitwise invariant.
5. Borrowed conformity scores retain the calibration buffer address, and two
   input predictions yield two lower and two upper arrays at every level.
6. Welford population variance returns `1` for `[10000, 10002]`, where the
   superseded `f32` second-moment formula returns `0` by cancellation.
7. A one-parameter affine response has squared correlation `1` within the
   `O(n epsilon)` rounding bound, and replay is bitwise deterministic.
8. Empty, non-finite, dimension-mismatched, and uncalibrated inputs return
   typed errors. Undefined statistics remain `None`.
9. Package clippy, Nextest, doctests, Rustdoc, dependency audit, and public
   semver classification run against the delivered revision.

## Rejected alternatives

- Correcting each local formula preserves duplicate statistical ownership and
  permits future drift.
- Computing PINN ranks in `f64` and narrowing the selected score violates the
  declared `f32` model/scoring precision.
- Retaining dynamic pseudo-Sobol or Morris APIs behind deprecated aliases keeps
  mathematically false contracts and compatibility debt.
