//! Differential validation of the kwavers pseudospectral solver against k-Wave.
//!
//! This is the in-repository oracle behind the README's k-Wave parity claim.
//! The reference fields under `tests/reference/kwave/` were produced by
//! `k-wave-python` driving the reference `kspaceFirstOrder-OMP` binary, and are
//! committed with a manifest recording their provenance and the exact
//! discretization used. `scripts/generate_kwave_reference.py` regenerates them.
//!
//! # Why this is a differential oracle and not a coincidence
//!
//! * The manifest carries the `dt` and step count k-Wave chose, so the kwavers
//!   run advances the same number of steps of the same size across the same
//!   grid. The initial condition is recomputed from the same closed form and
//!   cross-checked against the stored `p0`, so the two codes are given the same
//!   problem rather than a similar one.
//! * `compare_radius_cells` bounds the comparison to a centred window that the
//!   wavefront has entered but the absorbing layer has not reached, so a
//!   divergence inside the window is a divergence of the propagation scheme and
//!   not of the two codes' different boundary treatments. The kwavers run is
//!   therefore driven with a transparent boundary: nothing needs to absorb
//!   because nothing has arrived at the edge.
//! * Both anti-aliasing and reference-side `p0` smoothing are off. Each is a
//!   filter applied by one code and not the other; leaving either enabled would
//!   compare two different initial-value problems.
//!
//! # Metrics
//!
//! Agreement is reported as relative L2 error, relative L-infinity error, and
//! the Pearson correlation coefficient, each over the comparison window. The
//! acceptance bounds are stated at their assertion sites together with the
//! discretization difference that produces them.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use consus_npy::NpzReader;
use kwavers_boundary::Boundary;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
use kwavers_solver::plugin::PluginManager;
use kwavers_solver::pstd::numerics::spectral_correction::SpectralCorrectionMethod;
use kwavers_solver::pstd::{PSTDConfig, PSTDPlugin};
use leto::{Array3, Array4};
use serde::Deserialize;

/// Water at 20 degrees Celsius, matching `generate_kwave_reference.py`.
const SOUND_SPEED_M_S: f64 = 1500.0;
const DENSITY_KG_M3: f64 = 1000.0;

/// Gaussian seed width in cells, matching `generate_kwave_reference.py`.
const SIGMA_CELLS: f64 = 3.0;

/// Peak initial pressure in pascals, matching `generate_kwave_reference.py`.
const P0_PEAK_PA: f64 = 1.0;

/// The reference and the recomputed seed must agree to within the round trip of
/// a `f64` exponential through NPZ storage, which is exact for these values;
/// the bound leaves room only for the last-bit difference of the two `exp`
/// implementations.
const SEED_AGREEMENT_ABS: f64 = 1e-12;

// ─── Reference manifest ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct Manifest {
    cases: std::collections::BTreeMap<String, CaseRecord>,
}

#[derive(Debug, Deserialize)]
struct CaseRecord {
    archive: String,
    shape: Vec<usize>,
    dx_m: f64,
    dt_s: f64,
    steps: usize,
    compare_radius_cells: usize,
}

fn reference_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("reference")
        .join("kwave")
}

fn load_manifest() -> Manifest {
    let path = reference_dir().join("manifest.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// Load one stored array from a case archive as a flat row-major buffer.
fn load_reference_array(archive: &str, member: &str) -> Vec<f64> {
    let path = reference_dir().join(archive);
    let file = File::open(&path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    let mut reader = NpzReader::new(BufReader::new(file))
        .unwrap_or_else(|error| panic!("open npz {}: {error}", path.display()));
    let array = reader
        .by_name::<f64>(member)
        .unwrap_or_else(|error| panic!("read {member} from {}: {error}", path.display()));
    array.into_values().into_vec()
}

// ─── Case geometry ───────────────────────────────────────────────────────────

/// A case's grid shape padded to the three axes kwavers always carries. A
/// two-dimensional k-Wave case becomes an `(nx, ny, 1)` kwavers grid, which
/// `Grid::new` classifies as two-dimensional and whose singleton axis
/// contributes no wavenumber.
fn padded_shape(shape: &[usize]) -> (usize, usize, usize) {
    let axis = |index: usize| shape.get(index).copied().unwrap_or(1);
    (axis(0), axis(1), axis(2))
}

/// The isotropic Gaussian seed, recomputed from the same closed form the
/// generator used so the two codes start from the same field rather than from a
/// stored approximation of it.
fn gaussian_seed(shape: &[usize]) -> Array3<f64> {
    let (nx, ny, nz) = padded_shape(shape);
    let centre = |n: usize| (n / 2) as f64;
    let (cx, cy, cz) = (centre(nx), centre(ny), centre(nz));
    let two_sigma_squared = 2.0 * SIGMA_CELLS * SIGMA_CELLS;

    let mut seed = Array3::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                // A singleton axis contributes no offset, so the same expression
                // serves every dimensionality.
                let dx = if nx > 1 { i as f64 - cx } else { 0.0 };
                let dy = if ny > 1 { j as f64 - cy } else { 0.0 };
                let dz = if nz > 1 { k as f64 - cz } else { 0.0 };
                let radius_squared = dx * dx + dy * dy + dz * dz;
                seed[[i, j, k]] = P0_PEAK_PA * (-radius_squared / two_sigma_squared).exp();
            }
        }
    }
    seed
}

// ─── Transparent boundary ────────────────────────────────────────────────────

/// A boundary that leaves the field untouched.
///
/// The comparison window is chosen so the wavefront never reaches the domain
/// edge within the reference's step count, so no absorption is required inside
/// the window and adding one would introduce a parameter the reference does not
/// share. The PML-specific trait methods default to `apply_acoustic`, so a
/// no-op `apply_acoustic` makes every variant transparent.
#[derive(Debug)]
struct TransparentBoundary;

impl Boundary for TransparentBoundary {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn apply_acoustic(
        &mut self,
        _field: leto::ArrayViewMut3<f64>,
        _grid: &Grid,
        _time_step: usize,
    ) -> KwaversResult<()> {
        Ok(())
    }

    fn apply_acoustic_freq(
        &mut self,
        _field: &mut Array3<kwavers_math::fft::Complex64>,
        _grid: &Grid,
        _time_step: usize,
    ) -> KwaversResult<()> {
        Ok(())
    }

    fn apply_light(&mut self, _field: leto::ArrayViewMut3<f64>, _grid: &Grid, _time_step: usize) {}
}

// ─── kwavers run ─────────────────────────────────────────────────────────────

/// Propagate `seed` for `steps` steps of `dt` through the kwavers k-space
/// pseudospectral solver and return the final pressure field.
fn run_pstd(grid: &Grid, seed: &Array3<f64>, dt: f64, steps: usize) -> KwaversResult<Array3<f64>> {
    let mut config = PSTDConfig::default();

    // Treeby & Cox (2010) kappa = sinc(c dt |k| / 2) is the k-space temporal
    // dispersion correction the reference solver applies; selecting it is what
    // makes the two schemes the same scheme rather than two neighbours.
    config.spectral_correction.enabled = true;
    config.spectral_correction.method = SpectralCorrectionMethod::Treeby2010;

    // The reference applies no anti-aliasing filter. The seed is band-limited by
    // construction (sigma = 3 dx), so filtering here would remove content the
    // reference retained and turn a scheme comparison into a filter comparison.
    config.anti_aliasing.enabled = false;

    config.absorption_mode = AbsorptionMode::Lossless;

    // The solver's internal PML is disabled for the same reason the external
    // boundary is transparent: the comparison window never sees the edge.
    config.boundary = kwavers_solver::pstd::config::BoundaryConfig::None;

    // The solver advances by `config.dt`, not by the argument passed to
    // `execute`, so the reference time step has to be installed here as well.
    config.dt = dt;

    let mut plugin_manager = PluginManager::new();
    plugin_manager.add_plugin(Box::new(PSTDPlugin::new(config, grid)?))?;

    let medium = HomogeneousMedium::new(DENSITY_KG_M3, SOUND_SPEED_M_S, 0.0, 0.0, grid);

    // Pressure occupies axis-0 index 0 of the unified field array.
    let mut fields = Array4::zeros((17, grid.nx, grid.ny, grid.nz));
    fields
        .index_axis_mut::<3>(0, 0)
        .expect("pressure field slot")
        .assign(seed);

    plugin_manager.initialize(grid, &medium)?;

    let sources = Vec::new();
    let mut boundary = TransparentBoundary;

    for step in 0..steps {
        let t = step as f64 * dt;
        plugin_manager.execute(&mut fields, grid, &medium, &sources, &mut boundary, dt, t)?;
    }

    Ok(fields
        .index_axis::<3>(0, 0)
        .expect("pressure field slot")
        .to_contiguous())
}

// ─── Metrics ─────────────────────────────────────────────────────────────────

/// Agreement of two fields over the comparison window.
#[derive(Debug)]
struct Agreement {
    /// Relative L2 error, normalized by the reference's L2 norm.
    relative_l2: f64,
    /// Relative L-infinity error, normalized by the reference's peak magnitude.
    relative_linf: f64,
    /// Pearson correlation coefficient of the two windows.
    correlation: f64,
    /// Number of cells compared.
    cells: usize,
}

/// Collect the cells of `field` whose Chebyshev distance from the grid centre
/// is at most `radius`, in a fixed traversal order.
///
/// The window is a cube rather than a ball so the selection is exactly the set
/// of cells that the manifest's clearance argument covers on every axis.
fn window(field: &Array3<f64>, shape: &[usize], radius: usize) -> Vec<f64> {
    let (nx, ny, nz) = padded_shape(shape);
    let span = |n: usize| {
        if n > 1 {
            let centre = n / 2;
            centre.saturating_sub(radius)..(centre + radius + 1).min(n)
        } else {
            0..1
        }
    };

    let mut values = Vec::new();
    for i in span(nx) {
        for j in span(ny) {
            for k in span(nz) {
                values.push(field[[i, j, k]]);
            }
        }
    }
    values
}

fn agreement(candidate: &[f64], reference: &[f64]) -> Agreement {
    assert_eq!(
        candidate.len(),
        reference.len(),
        "windows must have equal extent"
    );

    let mut error_squared = 0.0;
    let mut reference_squared = 0.0;
    let mut max_error = 0.0_f64;
    let mut max_reference = 0.0_f64;
    for (&got, &want) in candidate.iter().zip(reference) {
        let error = got - want;
        error_squared += error * error;
        reference_squared += want * want;
        max_error = max_error.max(error.abs());
        max_reference = max_reference.max(want.abs());
    }

    let n = candidate.len() as f64;
    let candidate_mean = candidate.iter().sum::<f64>() / n;
    let reference_mean = reference.iter().sum::<f64>() / n;
    let mut covariance = 0.0;
    let mut candidate_variance = 0.0;
    let mut reference_variance = 0.0;
    for (&got, &want) in candidate.iter().zip(reference) {
        let a = got - candidate_mean;
        let b = want - reference_mean;
        covariance += a * b;
        candidate_variance += a * a;
        reference_variance += b * b;
    }

    Agreement {
        relative_l2: (error_squared / reference_squared).sqrt(),
        relative_linf: max_error / max_reference,
        correlation: covariance / (candidate_variance * reference_variance).sqrt(),
        cells: candidate.len(),
    }
}

// ─── Case driver ─────────────────────────────────────────────────────────────

/// Run one manifest case and return its agreement with the stored k-Wave field.
///
/// The measured metrics are this test's evidence: the numbers ADR 119 and the
/// README cite come from the line printed here, and a run that reports only
/// pass or fail cannot show that the agreement sits orders of magnitude inside
/// its bound. Stdout is where the test harness captures that.
#[expect(
    clippy::print_stdout,
    reason = "measured parity metrics are the test's reported evidence"
)]
fn compare_case(name: &str) -> Agreement {
    let manifest = load_manifest();
    let case = manifest
        .cases
        .get(name)
        .unwrap_or_else(|| panic!("manifest has no case {name}"));

    let (nx, ny, nz) = padded_shape(&case.shape);
    let grid = Grid::new(nx, ny, nz, case.dx_m, case.dx_m, case.dx_m).expect("reference grid");

    // The stored seed is the generator's own record of the initial condition.
    // Checking the recomputed seed against it proves the two codes were handed
    // the same problem; without this the comparison could silently drift.
    let stored_seed = load_reference_array(&case.archive, "p0");
    let seed = gaussian_seed(&case.shape);
    assert_eq!(
        stored_seed.len(),
        seed.len(),
        "{name}: stored and recomputed seed extents differ"
    );
    let seed_deviation = seed
        .as_slice()
        .expect("contiguous seed")
        .iter()
        .zip(&stored_seed)
        .map(|(&got, &want)| (got - want).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        seed_deviation <= SEED_AGREEMENT_ABS,
        "{name}: recomputed seed deviates from the stored reference seed by \
         {seed_deviation:e}, above the {SEED_AGREEMENT_ABS:e} storage round-trip bound"
    );

    let final_pressure =
        run_pstd(&grid, &seed, case.dt_s, case.steps).expect("kwavers pseudospectral run");
    let reference_flat = load_reference_array(&case.archive, "p_final");
    let mut reference = Array3::zeros((nx, ny, nz));
    reference
        .as_slice_mut()
        .expect("contiguous reference")
        .copy_from_slice(&reference_flat);

    let radius = case.compare_radius_cells;
    let measured = agreement(
        &window(&final_pressure, &case.shape, radius),
        &window(&reference, &case.shape, radius),
    );
    println!(
        "{name}: steps={} dt={:e}s window={} cells rel_l2={:.6e} rel_linf={:.6e} r={:.9}",
        case.steps,
        case.dt_s,
        measured.cells,
        measured.relative_l2,
        measured.relative_linf,
        measured.correlation
    );
    measured
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn pstd_matches_kwave_on_the_two_dimensional_homogeneous_ivp() {
    let measured = compare_case("ivp_homogeneous_2d");
    assert_parity("ivp_homogeneous_2d", &measured);
}

#[test]
fn pstd_matches_kwave_on_the_three_dimensional_homogeneous_ivp() {
    let measured = compare_case("ivp_homogeneous_3d");
    assert_parity("ivp_homogeneous_3d", &measured);
}

/// Acceptance bounds shared by every case, with the residual they admit.
///
/// Both codes integrate the same lossless first-order system with the same
/// Treeby-Cox corrected k-space leapfrog over the same grid, time step, and step
/// count, so the residual is not a truncation error that shrinks with
/// resolution. Two identified sources set its floor:
///
/// * The reference field is produced and stored in single precision
///   (`data_cast="single"`), contributing about `1.2e-7` relative per value and,
///   accumulated over the case's hundred-odd steps, about `1e-6` relative in
///   total. This dominates the two-dimensional case, which measures `5e-7`.
/// * The comparison window sits four cells from the domain edge in the
///   three-dimensional case, where the reference absorbs into a perfectly
///   matched layer and kwavers is periodic. The pseudospectral operator has
///   global support, so a small fraction of the outgoing field re-enters the
///   window. This dominates the three-dimensional case, which measures `1e-4`.
///
/// `1e-3` clears the larger of the two by an order of magnitude and sits an
/// order inside the `1e-2` figure the k-Wave comparison module states as its own
/// acceptance threshold. A bound tighter than this would be fitted to the two
/// measurements rather than derived from the sources above.
const PARITY_RELATIVE_L2_MAX: f64 = 1.0e-3;

/// The same argument at the pointwise norm. L-infinity concentrates the same
/// residual into the single worst cell rather than averaging it, so it is
/// allowed the same order of margin over its own measured value.
const PARITY_RELATIVE_LINF_MAX: f64 = 5.0e-3;

/// The correlation floor is the published k-Wave parity standard the README
/// cites. It is the shape oracle: invariant to a pure scale error, so it
/// isolates a genuine difference in the propagated waveform.
const PARITY_CORRELATION_MIN: f64 = 0.9999;

fn assert_parity(name: &str, measured: &Agreement) {
    assert!(
        measured.correlation >= PARITY_CORRELATION_MIN,
        "{name}: correlation {:.9} below the {PARITY_CORRELATION_MIN} k-Wave parity          standard (rel_l2 {:.6e}, rel_linf {:.6e})",
        measured.correlation,
        measured.relative_l2,
        measured.relative_linf
    );
    assert!(
        measured.relative_l2 <= PARITY_RELATIVE_L2_MAX,
        "{name}: relative L2 {:.6e} above the {PARITY_RELATIVE_L2_MAX:e} bound          (r {:.9})",
        measured.relative_l2,
        measured.correlation
    );
    assert!(
        measured.relative_linf <= PARITY_RELATIVE_LINF_MAX,
        "{name}: relative L-infinity {:.6e} above the {PARITY_RELATIVE_LINF_MAX:e}          bound (r {:.9})",
        measured.relative_linf,
        measured.correlation
    );
}

/// A single step of over- or under-propagation must be visible to the oracle.
///
/// Without this, the parity bounds above could be met by a comparison that had
/// stopped depending on the solver at all: any window loose enough to accept a
/// wrong final time is not measuring propagation. At CFL 0.15 one step moves the
/// field 0.15 cells against a seed of width 3 cells, which is the smallest
/// perturbation the reference data can express, and it must cost at least two
/// orders of magnitude of relative L2 error.
///
/// This also pins the step-count convention. `kgrid.Nt` counts k-Wave's time
/// *points*, so the manifest's `steps` is `Nt - 1` propagation intervals;
/// recording `Nt` instead lands exactly on the `+1` case measured here.
const TIME_DISCRIMINATION_FACTOR: f64 = 100.0;

#[test]
fn parity_degrades_when_the_step_count_is_wrong() {
    for name in ["ivp_homogeneous_2d", "ivp_homogeneous_3d"] {
        let manifest = load_manifest();
        let case = manifest
            .cases
            .get(name)
            .unwrap_or_else(|| panic!("manifest has no case {name}"));
        let (nx, ny, nz) = padded_shape(&case.shape);
        let grid = Grid::new(nx, ny, nz, case.dx_m, case.dx_m, case.dx_m).expect("reference grid");
        let seed = gaussian_seed(&case.shape);

        let reference_flat = load_reference_array(&case.archive, "p_final");
        let mut reference = Array3::zeros((nx, ny, nz));
        reference
            .as_slice_mut()
            .expect("contiguous reference")
            .copy_from_slice(&reference_flat);
        let reference_window = window(&reference, &case.shape, case.compare_radius_cells);

        let measure = |steps: usize| {
            let field = run_pstd(&grid, &seed, case.dt_s, steps).expect("kwavers run");
            agreement(
                &window(&field, &case.shape, case.compare_radius_cells),
                &reference_window,
            )
        };

        let exact = measure(case.steps);
        for wrong in [case.steps - 1, case.steps + 1] {
            let perturbed = measure(wrong);
            assert!(
                perturbed.relative_l2 >= exact.relative_l2 * TIME_DISCRIMINATION_FACTOR,
                "{name}: {wrong} steps gives relative L2 {:.6e}, less than                  {TIME_DISCRIMINATION_FACTOR}x the {:.6e} at the correct {} steps;                  the comparison is not resolving propagation time",
                perturbed.relative_l2,
                exact.relative_l2,
                case.steps
            );
        }
    }
}
