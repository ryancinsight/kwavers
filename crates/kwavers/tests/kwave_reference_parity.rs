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
use kwavers_solver::fdtd::{FdtdConfig, FdtdPlugin};
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

#[derive(Debug, Clone, Deserialize)]
struct CaseRecord {
    archive: String,
    shape: Vec<usize>,
    dx_m: f64,
    dt_s: f64,
    steps: usize,
    compare_radius_cells: usize,
    /// Power-law absorption coefficient in k-Wave's own units,
    /// `dB/(MHz^alpha_power cm)`. Absent for a lossless case.
    #[serde(default)]
    alpha_coeff_db: Option<f64>,
    #[serde(default)]
    alpha_power: Option<f64>,
    /// A layered medium steps sound speed and density from their water values
    /// to these across a smoothed interface. Absent for a uniform medium.
    #[serde(default)]
    layer_sound_speed_m_s: Option<f64>,
    #[serde(default)]
    layer_density_kg_m3: Option<f64>,
    #[serde(default)]
    layer_interface_cell: Option<usize>,
    #[serde(default)]
    layer_transition_cells: Option<f64>,
}

impl CaseRecord {
    /// The absorption model this case's reference was produced with.
    ///
    /// `AbsorptionMode::PowerLaw` takes the coefficient in k-Wave's units
    /// unconverted, so the manifest value passes through unchanged and the two
    /// codes cannot disagree about units.
    fn absorption(&self) -> AbsorptionMode {
        match (self.alpha_coeff_db, self.alpha_power) {
            (Some(alpha_coeff), Some(alpha_power)) => AbsorptionMode::PowerLaw {
                alpha_coeff,
                alpha_power,
            },
            _ => AbsorptionMode::Lossless,
        }
    }
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
    // The NPY header records the writer's memory order, and a transposed array
    // is Fortran-contiguous. The generator forces C order for exactly this
    // reason; asserting it here keeps a future regeneration from silently
    // handing back a buffer this reader would index the wrong way round.
    assert!(
        !array.is_fortran_order(),
        "{}: {member} is stored in Fortran order; regenerate with the current          generator, which writes C order",
        path.display()
    );
    array.into_values().into_vec()
}

/// Load a stored reference field into a grid-shaped array.
///
/// Elements are placed by index rather than by copying into the backing slice.
/// The stored buffer is row-major over the case's shape, and an array's slice
/// order is its own business; on a square case the two happen to agree, so a
/// slice copy passes there and silently scrambles the moment the axes differ in
/// length. Indexing states the mapping instead of assuming it.
fn load_reference_field(case: &CaseRecord, member: &str) -> Array3<f64> {
    let (nx, ny, nz) = padded_shape(&case.shape);
    let flat = load_reference_array(&case.archive, member);
    assert_eq!(
        flat.len(),
        nx * ny * nz,
        "{}: stored {member} has {} values, expected {}",
        case.archive,
        flat.len(),
        nx * ny * nz
    );
    let mut field = Array3::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                field[[i, j, k]] = flat[(i * ny + j) * nz + k];
            }
        }
    }
    field
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

/// Build the medium the reference was run against.
///
/// `HomogeneousMedium::new` seeds `absorption_alpha` with water's coefficient,
/// and `initialize_absorption_operators` prefers the medium's coefficient over
/// the one in `PSTDConfig::absorption_mode` whenever the medium reports a
/// non-zero value — which this constructor guarantees it always does. The
/// config coefficient is therefore unreachable through this medium, and
/// absorption has to be set here for the solver to see it at all. See the
/// `PSTD-ABSORPTION-CONFIG-DEAD` board item.
///
/// Nonlinearity is set to zero because `PSTDConfig::nonlinearity` is false on
/// this path, so the medium's B/A never enters the pressure update.
fn reference_medium(
    grid: &Grid,
    case: &CaseRecord,
    absorption: AbsorptionMode,
) -> KwaversResult<Box<dyn kwavers_medium::Medium>> {
    let mut base = HomogeneousMedium::new(DENSITY_KG_M3, SOUND_SPEED_M_S, 0.0, 0.0, grid);
    let (alpha, power) = match absorption {
        AbsorptionMode::PowerLaw {
            alpha_coeff,
            alpha_power,
        } => (alpha_coeff, alpha_power),
        // A lossless run must be lossless: leaving the constructor's water
        // coefficient in place would make the "lossless" reference comparison
        // quietly absorbing.
        _ => (0.0, 1.5),
    };
    base.set_acoustic_properties(alpha, power, 0.0)?;

    let Some(interface) = case.layer_interface_cell else {
        return Ok(Box::new(base));
    };

    // Expand the uniform medium across the grid, then overwrite the two fields
    // the case varies. Expanding first keeps every property the case does not
    // vary identical to the uniform runs, so the pair isolates heterogeneity.
    let mut layered =
        kwavers_medium::heterogeneous::HeterogeneousMedium::from_homogeneous(&base, grid)
            .expect("expand the reference medium across the grid");
    let layer_c = case
        .layer_sound_speed_m_s
        .expect("a layered case records its layer sound speed");
    let layer_rho = case
        .layer_density_kg_m3
        .expect("a layered case records its layer density");
    let transition = case
        .layer_transition_cells
        .expect("a layered case records its transition width");

    for i in 0..grid.nx {
        // The same hyperbolic tangent the generator used, so the two codes see
        // the same medium rather than two roundings of the same intent.
        let blend = 0.5 * (1.0 + ((i as f64 - interface as f64) / transition).tanh());
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                layered.sound_speed[[i, j, k]] =
                    SOUND_SPEED_M_S + (layer_c - SOUND_SPEED_M_S) * blend;
                layered.density[[i, j, k]] = DENSITY_KG_M3 + (layer_rho - DENSITY_KG_M3) * blend;
            }
        }
    }
    Ok(Box::new(layered))
}

/// Propagate `seed` for `steps` steps of `dt` through the kwavers k-space
/// pseudospectral solver and return the final pressure field.
fn run_pstd(
    grid: &Grid,
    case: &CaseRecord,
    seed: &Array3<f64>,
    dt: f64,
    steps: usize,
    absorption: AbsorptionMode,
) -> KwaversResult<Array3<f64>> {
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

    config.absorption_mode = absorption.clone();

    // The solver's internal PML is disabled for the same reason the external
    // boundary is transparent: the comparison window never sees the edge.
    config.boundary = kwavers_solver::pstd::config::BoundaryConfig::None;

    // The solver advances by `config.dt`, not by the argument passed to
    // `execute`, so the reference time step has to be installed here as well.
    config.dt = dt;

    let mut plugin_manager = PluginManager::new();
    plugin_manager.add_plugin(Box::new(PSTDPlugin::new(config, grid)?))?;

    let medium = reference_medium(grid, case, absorption)?;

    // Pressure occupies axis-0 index 0 of the unified field array.
    let mut fields = Array4::zeros((17, grid.nx, grid.ny, grid.nz));
    fields
        .index_axis_mut::<3>(0, 0)
        .expect("pressure field slot")
        .assign(seed);

    plugin_manager.initialize(grid, medium.as_ref())?;

    let sources = Vec::new();
    let mut boundary = TransparentBoundary;

    for step in 0..steps {
        let t = step as f64 * dt;
        plugin_manager.execute(
            &mut fields,
            grid,
            medium.as_ref(),
            &sources,
            &mut boundary,
            dt,
            t,
        )?;
    }

    Ok(fields
        .index_axis::<3>(0, 0)
        .expect("pressure field slot")
        .to_contiguous())
}

/// Propagate `seed` through the kwavers finite-difference solver on the
/// reference's own discretization and return the final pressure field.
///
/// This is a cross-scheme comparison rather than a like-for-like one: k-Wave is
/// exact in space, and this solver is fourth-order in space and second-order in
/// time, so the two are expected to separate by that scheme's dispersion error
/// and not to agree to storage precision. It is included because it measures
/// that separation against a real reference instead of against the
/// pseudospectral solver, which shares its k-space machinery and cannot act as
/// an independent oracle for it.
fn run_fdtd(grid: &Grid, seed: &Array3<f64>, dt: f64, steps: usize) -> KwaversResult<Array3<f64>> {
    let config = FdtdConfig {
        spatial_order: 4,
        staggered_grid: true,
        // The time step comes from the reference, so the Courant factor here is
        // only the config's own stability guard and never selects `dt`.
        cfl_factor: 0.95,
        subgridding: false,
        subgrid_factor: 2,
        enable_gpu_acceleration: false,
        nt: steps,
        dt,
        sensor_mask: None,
        ..Default::default()
    };

    let mut plugin_manager = PluginManager::new();
    plugin_manager.add_plugin(Box::new(FdtdPlugin::new(config, grid)?))?;

    let medium = HomogeneousMedium::new(DENSITY_KG_M3, SOUND_SPEED_M_S, 0.0, 0.0, grid);

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

/// Report one case's measured agreement.
///
/// These metrics are the tests' evidence: the numbers ADR 119 and the README
/// cite come from this line, and a run that reports only pass or fail cannot
/// show that the agreement sits orders of magnitude inside its bound. Stdout is
/// where the test harness captures that.
#[expect(
    clippy::print_stdout,
    reason = "measured parity metrics are the tests' reported evidence"
)]
fn report(name: &str, steps: usize, dt: f64, measured: &Agreement) {
    println!(
        "{name}: steps={steps} dt={dt:e}s window={} cells rel_l2={:.6e} rel_linf={:.6e} r={:.9}",
        measured.cells, measured.relative_l2, measured.relative_linf, measured.correlation
    );
}

/// Run one manifest case and return its agreement with the stored k-Wave field.
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
    let stored_seed = load_reference_field(case, "p0");
    let seed = gaussian_seed(&case.shape);
    let (nx_s, ny_s, nz_s) = padded_shape(&case.shape);
    let mut seed_deviation = 0.0_f64;
    for i in 0..nx_s {
        for j in 0..ny_s {
            for k in 0..nz_s {
                seed_deviation =
                    seed_deviation.max((seed[[i, j, k]] - stored_seed[[i, j, k]]).abs());
            }
        }
    }
    assert!(
        seed_deviation <= SEED_AGREEMENT_ABS,
        "{name}: recomputed seed deviates from the stored reference seed by          {seed_deviation:e}, above the {SEED_AGREEMENT_ABS:e} storage round-trip bound"
    );

    let final_pressure = run_pstd(&grid, case, &seed, case.dt_s, case.steps, case.absorption())
        .expect("kwavers pseudospectral run");
    let reference = load_reference_field(case, "p_final");

    let radius = case.compare_radius_cells;
    let measured = agreement(
        &window(&final_pressure, &case.shape, radius),
        &window(&reference, &case.shape, radius),
    );
    report(name, case.steps, case.dt_s, &measured);
    measured
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn pstd_matches_kwave_on_the_two_dimensional_homogeneous_ivp() {
    let measured = compare_case("ivp_homogeneous_2d");
    assert_parity("ivp_homogeneous_2d", &measured, PARITY_RELATIVE_L2_MAX);
}

#[test]
fn pstd_matches_kwave_on_the_three_dimensional_homogeneous_ivp() {
    let measured = compare_case("ivp_homogeneous_3d");
    assert_parity("ivp_homogeneous_3d", &measured, PARITY_RELATIVE_L2_MAX);
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

/// L-infinity concentrates the same residual into the single worst cell rather
/// than averaging it. Across every case measured here the pointwise error runs
/// about twice the L2 error, so each case's L-infinity bound is derived from its
/// own L2 bound at five times that ratio rather than being stated independently.
const PARITY_LINF_TO_L2_RATIO: f64 = 5.0;

/// The absorbing case's bound.
///
/// Its residual is not the lossless case's storage-precision floor: the two
/// codes implement the fractional-Laplacian absorption operator and its
/// Kramers-Kronig dispersion partner `eta tan(pi y / 2)` with different
/// discretizations, and that difference does not shrink with resolution. The
/// measurement is `8.1e-3`; `2e-2` clears it with margin while staying far below
/// the `0.32` attenuation the case is testing, so the bound cannot be met by a
/// solver that has mismodelled the absorption it is meant to reproduce.
const ABSORBING_RELATIVE_L2_MAX: f64 = 2.0e-2;

/// The layered case's bound.
///
/// Heterogeneity moves the residual for reasons the uniform cases do not have:
/// the density gradient enters the momentum update through a term the two codes
/// discretize differently, and the reflected and transmitted amplitudes depend
/// on the interface profile as each code samples it. Neither shrinks with
/// resolution at fixed profile width, so the bound is stated at the level those
/// choices produce rather than as a convergence rate. The measurement is
/// `7.97e-3`, close to the absorbing case's `8.1e-3` and for the same kind of
/// reason; `2e-2` clears it with margin while staying an order below the `0.27`
/// separation from a uniform run that the case is testing.
const LAYERED_RELATIVE_L2_MAX: f64 = 2.0e-2;

/// The correlation floor is the published k-Wave parity standard the README
/// cites. It is the shape oracle: invariant to a pure scale error, so it
/// isolates a genuine difference in the propagated waveform.
const PARITY_CORRELATION_MIN: f64 = 0.9999;

fn assert_parity(name: &str, measured: &Agreement, relative_l2_max: f64) {
    assert!(
        measured.correlation >= PARITY_CORRELATION_MIN,
        "{name}: correlation {:.9} below the {PARITY_CORRELATION_MIN} k-Wave parity standard (rel_l2 {:.6e}, rel_linf {:.6e})",
        measured.correlation,
        measured.relative_l2,
        measured.relative_linf
    );
    assert!(
        measured.relative_l2 <= relative_l2_max,
        "{name}: relative L2 {:.6e} above the {relative_l2_max:e} bound          (r {:.9})",
        measured.relative_l2,
        measured.correlation
    );
    assert!(
        measured.relative_linf <= relative_l2_max * PARITY_LINF_TO_L2_RATIO,
        "{name}: relative L-infinity {:.6e} above the {:e} bound (r {:.9})",
        measured.relative_linf,
        relative_l2_max * PARITY_LINF_TO_L2_RATIO,
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

        let reference = load_reference_field(case, "p_final");
        let reference_window = window(&reference, &case.shape, case.compare_radius_cells);

        let measure = |steps: usize| {
            let field = run_pstd(&grid, case, &seed, case.dt_s, steps, case.absorption())
                .expect("kwavers run");
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

/// Measure the finite-difference solver against the same k-Wave reference.
///
/// Reported, not asserted at the pseudospectral bound: the two schemes differ by
/// construction, and the point of the measurement is the size of that
/// difference against an independent reference.
#[test]
fn fdtd_separates_from_kwave_by_its_dispersion_error() {
    let name = "ivp_homogeneous_2d";
    let manifest = load_manifest();
    let case = manifest
        .cases
        .get(name)
        .unwrap_or_else(|| panic!("manifest has no case {name}"));
    let (nx, ny, nz) = padded_shape(&case.shape);
    let grid = Grid::new(nx, ny, nz, case.dx_m, case.dx_m, case.dx_m).expect("reference grid");
    let seed = gaussian_seed(&case.shape);

    let reference = load_reference_field(case, "p_final");

    let field =
        run_fdtd(&grid, &seed, case.dt_s, case.steps).expect("kwavers finite-difference run");
    let measured = agreement(
        &window(&field, &case.shape, case.compare_radius_cells),
        &window(&reference, &case.shape, case.compare_radius_cells),
    );

    report("ivp_homogeneous_2d/fdtd", case.steps, case.dt_s, &measured);

    // A fourth-order-in-space staggered scheme carries a relative phase error of
    // order `(k dx)^4 / 30` per wavelength travelled. Because that error is
    // quartic in the wavenumber it is set by the highest resolved content, not
    // the dominant content: a `sigma = 3 dx` Gaussian has a spectral width of
    // `1/3` rad per cell, so its three-sigma edge sits near `k dx ~ 1`, giving
    // `1/30 ~ 3e-2`. The measured `2.5e-2` is that number. `5e-2` bounds it with
    // margin for the second-order temporal term, and is loose enough that this
    // test reports a scheme difference rather than policing one.
    assert!(
        measured.relative_l2 <= 5.0e-2,
        "{name}: finite-difference relative L2 {:.6e} exceeds the dispersion-error          band this scheme is expected to occupy (r {:.9})",
        measured.relative_l2,
        measured.correlation
    );

    // The waveform must still be the same waveform: a dispersion error shifts
    // phase, it does not decorrelate. Falling below this means the scheme is
    // wrong, not merely dispersive.
    assert!(
        measured.correlation >= 0.99,
        "{name}: finite-difference correlation {:.9} is too low to be dispersion          alone (rel_l2 {:.6e})",
        measured.correlation,
        measured.relative_l2
    );
}

/// The absorbing case must match k-Wave, and must differ from the lossless one.
///
/// Matching alone would not establish that the absorption model runs: a solver
/// that silently ignored `PowerLaw` would produce the lossless field, and if the
/// coefficient were small enough that field would still sit inside the parity
/// bound. The reference pair is therefore generated with identical grid, seed,
/// time step, and step count, one variable changed, and this test asserts both
/// halves — agreement with the absorbing reference, and separation from the
/// lossless one.
#[test]
fn pstd_matches_kwave_with_power_law_absorption() {
    let measured = compare_case("ivp_absorbing_2d");
    assert_parity("ivp_absorbing_2d", &measured, ABSORBING_RELATIVE_L2_MAX);

    let manifest = load_manifest();
    let absorbing = manifest
        .cases
        .get("ivp_absorbing_2d")
        .expect("manifest has the absorbing case");
    assert!(
        matches!(absorbing.absorption(), AbsorptionMode::PowerLaw { .. }),
        "the absorbing case's manifest record carries no power-law coefficient,          so this test would silently reduce to the lossless one"
    );

    // The two references share every parameter but absorption, so comparing the
    // stored fields measures the model's effect in the reference solver and
    // needs no kwavers run of its own.
    let lossless = manifest
        .cases
        .get("ivp_homogeneous_2d")
        .expect("manifest has the lossless case");
    assert_eq!(
        (absorbing.shape.clone(), absorbing.steps, absorbing.dt_s),
        (lossless.shape.clone(), lossless.steps, lossless.dt_s),
        "the absorbing and lossless cases must differ only in absorption"
    );

    let read = |case: &CaseRecord| {
        window(
            &load_reference_field(case, "p_final"),
            &case.shape,
            case.compare_radius_cells,
        )
    };
    let separation = agreement(&read(absorbing), &read(lossless));

    // At 40 dB/(MHz^1.5 cm) over the 1.5 mm this wave travels, with the seed's
    // dominant content near 0.8 MHz, the expected attenuation is
    // 40 * 0.8^1.5 * 0.15 cm ~ 4 dB, an amplitude factor near 0.6. The measured
    // separation is 0.53 relative L2. Requiring 0.2 keeps the guard far above
    // the parity bound it is protecting (1e-3) while staying well clear of the
    // measurement, so it fails on absorption being absent rather than on
    // absorption being slightly mismodelled — which the parity assertion above
    // already covers.
    assert!(
        separation.relative_l2 >= 0.2,
        "the absorbing and lossless reference fields differ by only {:.6e}          relative L2; the absorbing case cannot discriminate an absent          absorption model from a correct one",
        separation.relative_l2
    );
}

/// The layered case must match k-Wave, and must differ from the uniform one.
///
/// Same structure as the absorbing case and for the same reason: a solver that
/// ignored the medium arrays would reproduce the uniform field, so agreement
/// alone proves nothing. The reference pair shares grid, seed, and interface-free
/// physics; only sound speed and density vary.
#[test]
fn pstd_matches_kwave_on_a_layered_medium() {
    let measured = compare_case("ivp_layered_2d");

    let manifest = load_manifest();
    let layered = manifest
        .cases
        .get("ivp_layered_2d")
        .expect("manifest has the layered case");
    assert!(
        layered.layer_interface_cell.is_some(),
        "the layered case's manifest record carries no interface, so this test          would silently reduce to a uniform one"
    );

    // The uniform case runs a different time step, so the two stored fields are
    // not directly comparable. Comparing the layered reference against a kwavers
    // run on a *uniform* medium at the layered case's own discretization keeps
    // every other variable fixed and isolates the medium.
    let (nx, ny, nz) = padded_shape(&layered.shape);
    let grid = Grid::new(nx, ny, nz, layered.dx_m, layered.dx_m, layered.dx_m).expect("grid");
    let seed = gaussian_seed(&layered.shape);
    let uniform_record = CaseRecord {
        layer_sound_speed_m_s: None,
        layer_density_kg_m3: None,
        layer_interface_cell: None,
        layer_transition_cells: None,
        ..layered.clone()
    };
    let uniform = run_pstd(
        &grid,
        &uniform_record,
        &seed,
        layered.dt_s,
        layered.steps,
        AbsorptionMode::Lossless,
    )
    .expect("uniform-medium run at the layered case's discretization");

    let reference = load_reference_field(layered, "p_final");

    let radius = layered.compare_radius_cells;
    let separation = agreement(
        &window(&uniform, &layered.shape, radius),
        &window(&reference, &layered.shape, radius),
    );

    // The interface sits 8 cells from the seed and the wavefront reaches it
    // around step 53 of 120, so by the final step the reference carries a
    // transmitted wave, a reflection back through the seed, and a refracted
    // front, none of which the uniform run produces. Requiring 0.2 relative L2
    // keeps the guard far above the parity bound it protects while staying well
    // clear of the measurement, so it fails on the medium being ignored rather
    // than on it being slightly mismodelled.
    report(
        "ivp_layered_2d/uniform-vs-layered-reference",
        layered.steps,
        layered.dt_s,
        &separation,
    );
    assert_parity("ivp_layered_2d", &measured, LAYERED_RELATIVE_L2_MAX);
    assert!(
        separation.relative_l2 >= 0.2,
        "a uniform-medium run differs from the layered reference by only          {:.6e} relative L2; the layered case cannot discriminate a solver that          ignores the medium arrays from one that reads them",
        separation.relative_l2
    );
}
