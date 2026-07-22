# ADR 043: Tyche uncertainty provider

- Status: Accepted
- Date: 2026-07-20
- Change class: [major] [arch]
- Closed: 2026-07-22

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
rank proof at the integrated public revision
[`55ef4d0`](https://github.com/ryancinsight/tyche/blob/55ef4d0cf107d30799aece1ee26529d1f8f8e3cb/docs/adr/0001-reproducible-study-boundary.md#conformal-coverage).

## Decision

Kwavers depends on `tyche-core` at merged revision `55ef4d0`.

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
  Genuine Morris and Saltelli sensitivity methods remain Tyche provider
  capability gaps; Kwavers does not retain mislabeled substitutes.
- `kwavers-grid` owns only the physical transformation from a Tyche unit design
  into a validated geometric domain. One blanket `DesignSamplingExt`
  implementation collects counter, Latin-hypercube, and Sobol designs directly
  into the final Leto matrix. `CollocationSampler<G>` owns strategy selection
  and monomorphizes over the inline domain type.
- Rectangular boundary faces are selected with a borrowed Tyche
  `WeightedCategorical` distribution whose masses are their measures. Disk and
  ball interiors use inverse-area and inverse-volume radial maps; spherical
  boundaries use direct angle and uniform-cosine maps. The construction follows
  the inverse-volume rule in [Goodman, Monte Carlo notes, page
  19](https://math.nyu.edu/~goodman/teaching/MonteCarlo17/notes/Week1.pdf).
- Boundary charts remain domain-owned. Applying a Latin hypercube or Sobol unit
  cube directly to a boundary is undefined without a chart, so the configured
  collocation design affects the interior only.
- Undefined pre-calibration distributions and zero-width coverage efficiency
  use `Option`, not fabricated numeric values.

The dependency direction is
`kwavers-grid|kwavers-analysis|kwavers-solver -> tyche-core -> eunomia`.
Physical transforms, model inference, Leto arrays, and uncertainty presentation
remain Kwavers-owned.

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
- `RectangularDomain::{new_1d,new_2d,new_3d}` and
  `SphericalDomain::{new_2d,new_3d}` return `Result`; their fields become
  private and borrowed accessors expose validated active coordinates.
- `GeometricDomain::{sample_interior,sample_boundary}` take a required Tyche
  `Seed` and return `Result<Array2<f64>, GeometryError>`.
- `CollocationSampler` becomes `CollocationSampler<G: GeometricDomain>`, stores
  `G` directly instead of `Box<dyn GeometricDomain>`, and returns typed results.
  `AdaptiveRefinement` and its collocation-strategy variant are removed; the ML
  adaptive sampler is the canonical residual-driven implementation.
- `elastic_2d::ElasticCollocationSamplingStrategy` is removed. Elastic training
  configuration stores the canonical `CollocationSamplingStrategy` directly,
  so serialization, dispatch vocabulary, and extension ownership cannot drift.
- `CollocationSamplingStrategy` and `PinnGeometryInterfaceCondition` become
  non-exhaustive public enums; downstream matches require a wildcard so future
  complete design or interface variants do not force another enum-layout break.
- `MultiRegionDomain::new` returns `Result<_, MultiRegionError>`, validates
  region, material, interface, and dimension cardinalities, and keeps its
  collections private behind borrowed accessors. Interface sampling resets its
  acceptance quota for every adjacent pair and derives a distinct Tyche seed
  for each pair.
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
10. Every non-empty geometry and collocation sample matrix performs one
    allocation and zero reallocations; fixed-layout domain construction
    performs none.
11. Every strategy returns the requested cardinality in one, two, and three
    dimensions, replays bitwise for a fixed seed, and classifies every mapped
    point as strict interior. Solver dispatch equals direct Tyche-design
    collection element for element, while a fixed domain and seed produce the
    same boundary matrix for every interior strategy.
12. For a rectangle face `f` with measure `A_f`, selection has probability
    `A_f / sum(A)` and conditional density `1 / A_f`; their product is the
    constant surface density `1 / sum(A)`.
13. For a disk, `r = R sqrt(U)` gives `P(r <= q) = (q/R)^2`. For a ball,
    `r = R cbrt(U)` gives `P(r <= q) = (q/R)^3`. Uniform angle in two
    dimensions and uniform azimuth plus `cos(theta)` in three dimensions supply
    the remaining normalized angular measure.
14. Invalid bounds, center, radius, unit coordinate, dimension, measure, and
    design count fail before mutating caller output or allocating an
    intermediate design matrix. Addressable but unreservable output sizes
    return a typed allocation error instead of reaching an allocator abort.

## Rejected alternatives

- Correcting each local formula preserves duplicate statistical ownership and
  permits future drift.
- Computing PINN ranks in `f64` and narrowing the selected score violates the
  declared `f32` model/scoring precision.
- Retaining dynamic pseudo-Sobol or Morris APIs behind deprecated aliases keeps
  mathematically false contracts and compatibility debt.
- Retaining local LHS permutations or Sobol direction numbers duplicates the
  provider's replay contract and allocates intermediate designs.
- Rejection sampling a disk or ball has input-dependent work, discards Tyche
  points, and makes an exact-cardinality low-discrepancy design impossible.
- Storing collocation domains behind `Box<dyn GeometricDomain>` imposes heap and
  vtable costs where the caller already knows the concrete domain type.
- Treating Latin-hypercube or Sobol coordinates as boundary samples either
  discards almost every point or silently falls back to another distribution;
  both violate the selected design's contract.

## Closure evidence

PR #304 merged the collocation boundary as `9ad18523d`. Exact candidate head
`cc382dbc2243678fef55101aa106e9f8d7ad7bbf` passes ordinary CI
`29875284052`, architecture validation `29875284007`, and legacy audit
`29875283982`. The benchmark workflow failure on that head came from the
superseded complete statistical universe and is closed by ADR 045's bounded
replacement, whose exact head is green.
