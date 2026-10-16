//! Full-wave PSTD runs of the linear array in homogeneous brain tissue.
//!
//! Field amplitudes are taken at the drive frequency: a single-bin DFT of the
//! pressure over the whole run, divided by the same DFT of the drive. For a
//! linear medium that ratio is the steady-state (CW) transfer magnitude, the
//! quantity the Rayleigh oracle computes and the one a long neuromodulation
//! pulse reaches. A peak-hold over the sampled burst would instead read up to
//! `1 − cos(π f dt)` low (4.9% at 10 samples per period) and depend on the grid.

use super::{
    attenuation_np_m, element_drive, time_step_s, GridSpec, LinearArraySpec, ALPHA_DB_CM_MHZ,
    ALPHA_POWER, BURST_CYCLES, B_OVER_A, DENSITY_KG_M3, FREQUENCY_HZ, POINTS_PER_WAVELENGTH,
    SOUND_SPEED_M_S,
};
use anyhow::{anyhow, Result};
use eunomia::convert::IntegerTarget;
use eunomia::Complex64;
use eunomia::FloatElement;
use kwavers_boundary::cpml::CPMLConfig;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
use kwavers_solver::forward::pstd::config::{
    BoundaryConfig, KSpaceMethod, PSTDConfig, PowerLawDispersion,
};
use kwavers_solver::forward::pstd::PSTDSolver;
use kwavers_source::{GridSource, SourceMode};
use kwavers_transducer::transducers::physics::{
    rayleigh_pressure, PlanarAperture, PlanarApertureShape, RayleighIntegralSpec,
};
use leto::{Array2, Array3};
use std::f64::consts::TAU;

/// Side of the square calibration plane, metres (7.8 λ).
const CALIBRATION_PLANE_WIDTH_M: f64 = 6.0e-3;
/// Free margin between the plane edge and the lateral PML, metres. The beam of the
/// plane spreads by about λ/W = 7° and is 1.5 mm wider per side at the observation
/// depth; the margin keeps the diffracted field off the PML, whose absorption
/// degrades at grazing incidence.
const CALIBRATION_MARGIN_M: f64 = 4.0e-3;
/// Gauss–Legendre order for the calibration plane; the phase across the plane varies
/// by under 1.5 rad at the observation depth, which 16 nodes integrate to round-off.
const CALIBRATION_QUADRATURE_ORDER: usize = 16;
/// Depth of the calibration observation point below the source plane, metres:
/// 16.12 Np/m over it attenuates by 33%, so the absorbing check resolves α to about 1%.
const CALIBRATION_DEPTH_M: f64 = 25.0e-3;
/// Time after the latest arrival and the burst before the calibration run stops, seconds.
const CALIBRATION_TAIL_S: f64 = 2.0e-6;

/// Fields from one focused PSTD run.
#[derive(Debug, Clone)]
pub struct FocalRun {
    /// Amplitude at the drive frequency per unit surface pressure at every voxel
    /// (steady-state gain, Pa/Pa).
    pub harmonic_gain: Array3<f64>,
    /// Minimum (most negative) pressure over the run at every voxel, Pa.
    pub peak_neg: Array3<f64>,
}

/// Run configuration shared by the focused and calibration runs.
#[derive(Debug, Clone, Copy)]
pub struct RunOptions {
    /// Enable the Westervelt nonlinear term with the brain B/A.
    pub nonlinear: bool,
    /// Apply the brain power-law absorption, without dispersion (k-Wave
    /// `alpha_mode = 'no_dispersion'`, required for an exponent of 1).
    pub absorbing: bool,
    /// Number of time steps.
    pub steps: usize,
}

fn medium_for(grid: &Grid, options: RunOptions) -> HomogeneousMedium {
    let mut medium = HomogeneousMedium::new(DENSITY_KG_M3, SOUND_SPEED_M_S, 0.0, 0.0, grid);
    if options.nonlinear {
        medium.set_nonlinearity(B_OVER_A);
    }
    medium
}

fn absorption_mode(options: RunOptions) -> AbsorptionMode {
    if options.absorbing {
        AbsorptionMode::PowerLaw {
            alpha_coeff: Some(ALPHA_DB_CM_MHZ),
            alpha_power: ALPHA_POWER,
        }
    } else {
        AbsorptionMode::Lossless
    }
}

fn cpml(grid: &GridSpec) -> CPMLConfig {
    CPMLConfig::with_per_dimension_thickness(grid.pml_cells, grid.pml_cells, grid.pml_cells)
        .with_alpha(2.0)
}

fn pstd_config(grid: &GridSpec, options: RunOptions) -> PSTDConfig {
    PSTDConfig {
        dt: time_step_s(grid.dx),
        nt: options.steps,
        boundary: BoundaryConfig::CPML(cpml(grid)),
        pml_inside: true,
        smooth_sources: false,
        kspace_method: KSpaceMethod::StandardPSTD,
        absorption_mode: absorption_mode(options),
        power_law_dispersion: PowerLawDispersion::Omitted,
        nonlinearity: options.nonlinear,
        ..PSTDConfig::default()
    }
}

fn new_grid(grid: &GridSpec) -> Result<Grid> {
    Ok(Grid::new(
        grid.nx, grid.ny, grid.nz, grid.dx, grid.dx, grid.dx,
    )?)
}

/// `(cos ωt_n, sin ωt_n)` at the drive frequency for every step.
fn drive_frequency_kernel(steps: usize, dt: f64) -> Vec<(f64, f64)> {
    let omega_dt = TAU * FREQUENCY_HZ * dt;
    (0..steps)
        .map(|step| {
            let phase = omega_dt * f64::from_count(step);
            (phase.cos(), phase.sin())
        })
        .collect()
}

/// Magnitude of the drive-frequency DFT bin of `samples`.
fn drive_frequency_magnitude(samples: impl Iterator<Item = f64>, kernel: &[(f64, f64)]) -> f64 {
    let (re, im) = samples
        .zip(kernel)
        .fold((0.0, 0.0), |(re, im), (value, &(c, s))| {
            (re + value * c, im - value * s)
        });
    re.hypot(im)
}

/// Magnitude of the drive-frequency DFT of one undelayed element drive of amplitude `amplitude_pa`.
fn drive_reference(amplitude_pa: f64, kernel: &[(f64, f64)], dt: f64) -> f64 {
    let drive = element_drive(&[0.0], amplitude_pa, kernel.len(), dt);
    drive_frequency_magnitude((0..kernel.len()).map(|step| drive[[0, step]]), kernel)
}

/// Propagate the focused array and record the drive-frequency gain and peak-negative
/// pressure in every voxel.
///
/// # Errors
///
/// Returns an error if the grid, medium, source mask, or solver construction fails,
/// or if a non-finite pressure appears during propagation.
pub fn run_focused(
    grid: &GridSpec,
    array: &LinearArraySpec,
    focus_m: [f64; 3],
    amplitude_pa: f64,
    options: RunOptions,
) -> Result<FocalRun> {
    let dt = time_step_s(grid.dx);
    let delays = array.focus_delays_s(focus_m);
    let signals = element_drive(&delays, amplitude_pa, options.steps, dt);
    let kwave = array.kwave_array();

    let cells = new_grid(grid)?;
    let (mask, per_cell) = kwave
        .build_per_element_source(&cells, &signals)
        .map_err(|message| anyhow!(message))?;
    let source = GridSource {
        p_mask: Some(mask),
        p_signal: Some(per_cell),
        p_mode: SourceMode::Additive,
        ..GridSource::new_empty()
    };

    let medium = medium_for(&cells, options);
    let mut solver = PSTDSolver::new(pstd_config(grid, options), cells, &medium, source)?;

    let kernel = drive_frequency_kernel(options.steps, dt);
    let shape = (grid.nx, grid.ny, grid.nz);
    let mut re = Array3::<f64>::zeros(shape);
    let mut im = Array3::<f64>::zeros(shape);
    let mut peak_neg = Array3::<f64>::zeros(shape);
    for &(c, s) in &kernel {
        solver.step_forward()?;
        let field = solver.pressure_field();
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let value = field[[i, j, k]];
                    if !value.is_finite() {
                        return Err(anyhow!("non-finite pressure at ({i}, {j}, {k})"));
                    }
                    re[[i, j, k]] += value * c;
                    im[[i, j, k]] -= value * s;
                    peak_neg[[i, j, k]] = peak_neg[[i, j, k]].min(value);
                }
            }
        }
    }
    let reference = drive_reference(amplitude_pa, &kernel, dt);
    let mut harmonic_gain = Array3::<f64>::zeros(shape);
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                harmonic_gain[[i, j, k]] = re[[i, j, k]].hypot(im[[i, j, k]]) / reference;
            }
        }
    }
    Ok(FocalRun {
        harmonic_gain,
        peak_neg,
    })
}

/// Calibration of the additive pressure source and the absorption against the
/// baffled Rayleigh integral.
///
/// A [`CALIBRATION_PLANE_WIDTH_M`] square plane sits [`CALIBRATION_MARGIN_M`] inside
/// the lateral PML at `points_per_wavelength`; the drive-frequency gain on axis
/// [`CALIBRATION_DEPTH_M`] below it is divided by the baffled Rayleigh value of
/// the identical plane, with the brain attenuation applied to both when
/// `absorbing`. The ratio should be 1: the lossless run tests the source
/// scaling, the absorbing run the power-law absorption without dispersion.
///
/// # Errors
///
/// Returns an error if the solver fails or produces a non-finite field.
pub fn run_plane_wave_calibration(
    amplitude_pa: f64,
    points_per_wavelength: f64,
    absorbing: bool,
) -> Result<f64> {
    let dx = super::wavelength_m() / points_per_wavelength;
    // 10 CPML cells at the production λ/3, the same 3.3 λ thickness at any resolution.
    let pml_cells =
        usize::from_truncated((10.0 * points_per_wavelength / POINTS_PER_WAVELENGTH).round());
    let array_plane_k = pml_cells + 2;
    let depth_cells = usize::from_truncated((CALIBRATION_DEPTH_M / dx).round());
    let plane_cells = usize::from_truncated((CALIBRATION_PLANE_WIDTH_M / dx).round());
    let margin_cells = usize::from_truncated((CALIBRATION_MARGIN_M / dx).round());
    let lateral_cells = plane_cells + 2 * (margin_cells + pml_cells);
    let grid = GridSpec {
        nx: lateral_cells,
        ny: lateral_cells,
        nz: array_plane_k + depth_cells + 2 * pml_cells,
        dx,
        array_plane_k,
        pml_cells,
    };
    let lo = grid.pml_cells + margin_cells;
    let hi_i = lo + plane_cells;
    let hi_j = lo + plane_cells;
    let width_m = f64::from_count(hi_i - lo) * dx;
    let dt = time_step_s(dx);
    let latest_arrival_s = CALIBRATION_DEPTH_M.hypot(width_m) / SOUND_SPEED_M_S;
    let duration_s = latest_arrival_s + BURST_CYCLES / FREQUENCY_HZ + CALIBRATION_TAIL_S;
    let steps = usize::from_truncated((duration_s / dt).ceil());
    let drive = element_drive(&[0.0], amplitude_pa, steps, dt);

    let cells = new_grid(&grid)?;
    let plane = (hi_i - lo) * (hi_j - lo);
    let mut p_mask = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    for i in lo..hi_i {
        for j in lo..hi_j {
            p_mask[[i, j, grid.array_plane_k]] = 1.0;
        }
    }
    let mut p_signal = Array2::<f64>::zeros([plane, steps]);
    for row in 0..plane {
        for step in 0..steps {
            p_signal[[row, step]] = drive[[0, step]];
        }
    }
    let source = GridSource {
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        p_mode: SourceMode::Additive,
        ..GridSource::new_empty()
    };
    let options = RunOptions {
        nonlinear: false,
        absorbing,
        steps,
    };
    let medium = medium_for(&cells, options);
    let mut solver = PSTDSolver::new(pstd_config(&grid, options), cells, &medium, source)?;

    let observe = (grid.nx / 2, grid.ny / 2, grid.array_plane_k + depth_cells);
    let mut trace = Vec::with_capacity(steps);
    for _ in 0..steps {
        solver.step_forward()?;
        let value = solver.pressure_field()[[observe.0, observe.1, observe.2]];
        if !value.is_finite() {
            return Err(anyhow!("non-finite calibration pressure"));
        }
        trace.push(value);
    }
    let kernel = drive_frequency_kernel(steps, dt);
    let gain = drive_frequency_magnitude(trace.into_iter(), &kernel)
        / drive_reference(amplitude_pa, &kernel, dt);

    // Baffled Rayleigh value of the identical square plane, unit surface pressure.
    // The observation cell centre sits half a cell off the plane centre in x and y
    // (even cell counts), so both are evaluated at the cell, not the plane centre.
    let plane_center = [
        0.5 * f64::from_count(lo + hi_i - 1) * dx,
        0.5 * f64::from_count(lo + hi_j - 1) * dx,
        grid.array_plane_m(),
    ];
    let aperture = PlanarAperture::oriented(
        plane_center,
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        PlanarApertureShape::Rectangle {
            width_m,
            height_m: f64::from_count(hi_j - lo) * dx,
        },
        Complex64::new(1.0, 0.0),
    )
    .map_err(|error| anyhow!("{error:?}"))?;
    let attenuation = if absorbing { attenuation_np_m() } else { 0.0 };
    let spec = RayleighIntegralSpec::new(
        TAU * FREQUENCY_HZ / SOUND_SPEED_M_S,
        attenuation,
        CALIBRATION_QUADRATURE_ORDER,
        CALIBRATION_QUADRATURE_ORDER,
    )
    .map_err(|error| anyhow!("{error:?}"))?;
    let observation_m = [
        f64::from_count(observe.0) * dx,
        f64::from_count(observe.1) * dx,
        f64::from_count(observe.2) * dx,
    ];
    let baffled = rayleigh_pressure(&[observation_m], &[aperture], &spec)
        .map_err(|error| anyhow!("{error:?}"))?[0]
        .norm();
    Ok(gain / baffled)
}
