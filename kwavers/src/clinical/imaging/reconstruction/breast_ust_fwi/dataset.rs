//! PSTD acquisition dataset generation for breast UST FWI.
//!
//! The clinical layer owns acquisition orchestration: the solver layer provides
//! PSTD propagation and frequency-domain inversion, while this module maps the
//! breast ring-array protocol onto those solver contracts.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::boundary::CPMLConfig;
use crate::domain::grid::Grid;
use crate::domain::medium::heterogeneous::HeterogeneousFactory;
use crate::domain::sensor::recorder::simple::SensorRecorder;
use crate::domain::source::{GridSource, SourceMode};
use crate::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    MultiRowRingArray, ElementPosition,
};
use crate::solver::forward::pstd::config::BoundaryConfig;
use crate::solver::forward::pstd::{PSTDConfig, PSTDSolver};
use crate::solver::inverse::fwi::frequency_domain::FrequencyObservation;
use ndarray::{s, Array2, Array3, ArrayView1};
use num_complex::Complex64;
use std::f64::consts::PI;

pub const BREAST_UST_PSTD_DATASET_MODEL: &str =
    "clinical_breast_ust_multi_row_ring_pstd_frequency_dataset";

const PSTD_DATASET_CFL_LIMIT: f64 = 0.3;

/// PSTD acquisition settings for frequency-binned breast UST data.
#[derive(Clone, Copy, Debug)]
pub struct BreastUstPstdDatasetConfig {
    /// Uniform grid spacing [m] for the phantom sound-speed volume.
    pub spacing_m: f64,
    /// PSTD time step [s].
    pub time_step_s: f64,
    /// Number of continuous-wave cycles simulated for each frequency.
    pub cycles_per_frequency: usize,
    /// Number of trailing cycles used for the complex frequency bin.
    pub frequency_bin_cycles: usize,
    /// Scalar pressure-source amplitude [Pa] applied to each active row source.
    pub source_amplitude_pa: f64,
    /// Scalar density [kg/m^3] used with the sound-speed phantom.
    pub density_kg_m3: f64,
    /// CPML thickness in cells; zero disables CPML.
    pub cpml_thickness_cells: usize,
}

impl Default for BreastUstPstdDatasetConfig {
    fn default() -> Self {
        Self {
            spacing_m: 1.0e-3,
            time_step_s: 1.0e-7,
            cycles_per_frequency: 4,
            frequency_bin_cycles: 1,
            source_amplitude_pa: 1.0e3,
            density_kg_m3: 1000.0,
            cpml_thickness_cells: 8,
        }
    }
}

/// Frequency-binned multi-row ring acquisition data.
#[derive(Clone, Debug)]
pub struct BreastUstPstdDataset {
    /// Frequencies represented by `observed_pressure`.
    pub frequencies_hz: Vec<f64>,
    /// Complex pressure bins shaped `(frequency, transmit, receiver)`.
    pub observed_pressure: Array3<Complex64>,
    /// Number of cylindrical-wave transmit events per frequency.
    pub transmissions: usize,
    /// Number of ring receivers sampled per transmit.
    pub receivers: usize,
    /// PSTD steps executed for each frequency.
    pub time_steps_per_frequency: Vec<usize>,
    /// First absolute sample used for each frequency-domain bin.
    pub frequency_bin_start_steps_per_frequency: Vec<usize>,
    /// Clinical acquisition model identifier.
    pub model_family: &'static str,
}

impl BreastUstPstdDataset {
    /// Convert the stacked dataset into solver observations.
    #[must_use]
    pub fn observations(&self) -> Vec<FrequencyObservation> {
        self.frequencies_hz
            .iter()
            .enumerate()
            .map(|(index, &frequency_hz)| {
                FrequencyObservation::new(
                    frequency_hz,
                    self.observed_pressure.slice(s![index, .., ..]).to_owned(),
                )
            })
            .collect()
    }
}

/// Generate multi-row ring receiver data by PSTD simulation and frequency binning.
///
/// # Theorem
/// For each receiver trace `p_r[n]`, the returned complex datum is the
/// rectangular-quadrature first Fourier coefficient
/// `2/M * sum_{n=n0}^{n0+M-1} p_r[n] exp(-i 2 pi f n dt)` over the configured
/// trailing steady-state cycles. This is the discrete frequency bin consumed by
/// the frequency-domain FWI solver.
///
/// # Errors
/// Returns an error when geometry, medium, sampling, or PSTD stability
/// constraints are violated.
pub fn generate_breast_ust_pstd_frequency_dataset(
    sound_speed_m_s: &Array3<f64>,
    array: &MultiRowRingArray,
    frequencies_hz: &[f64],
    config: BreastUstPstdDatasetConfig,
) -> KwaversResult<BreastUstPstdDataset> {
    validate_config(&config)?;
    validate_sound_speed(sound_speed_m_s)?;
    validate_frequencies(frequencies_hz, config.time_step_s)?;
    validate_cfl(sound_speed_m_s, config)?;

    let receiver_indices =
        map_ring_points_to_grid(sound_speed_m_s.dim(), config, array.elements())?;
    let transmissions = array.circumferential_elements();
    let receivers = receiver_indices.len();
    let mut observed = Array3::<Complex64>::zeros((frequencies_hz.len(), transmissions, receivers));
    let mut time_steps_per_frequency = Vec::with_capacity(frequencies_hz.len());
    let mut bin_start_steps = Vec::with_capacity(frequencies_hz.len());

    for (frequency_index, &frequency_hz) in frequencies_hz.iter().enumerate() {
        let steps = time_steps_for_frequency(frequency_hz, config)?;
        let bin_start = frequency_bin_start_step(frequency_hz, config, steps)?;
        time_steps_per_frequency.push(steps);
        bin_start_steps.push(bin_start);

        for transmit_index in 0..transmissions {
            let source_points = array.cylindrical_source(transmit_index);
            let source_indices =
                map_ring_points_to_grid(sound_speed_m_s.dim(), config, &source_points)?;
            let traces = run_pstd_transmit(
                sound_speed_m_s,
                &receiver_indices,
                &source_indices,
                frequency_hz,
                steps,
                config,
            )?;

            for receiver in 0..receivers {
                observed[[frequency_index, transmit_index, receiver]] = frequency_bin(
                    traces.row(receiver),
                    frequency_hz,
                    config.time_step_s,
                    bin_start,
                );
            }
        }
    }

    Ok(BreastUstPstdDataset {
        frequencies_hz: frequencies_hz.to_vec(),
        observed_pressure: observed,
        transmissions,
        receivers,
        time_steps_per_frequency,
        frequency_bin_start_steps_per_frequency: bin_start_steps,
        model_family: BREAST_UST_PSTD_DATASET_MODEL,
    })
}

/// Snap ring-array coordinates to the same centered grid points used by PSTD.
///
/// # Errors
/// Returns an error when any source/receiver lies outside the centered PSTD
/// grid support or when the grid spacing is invalid.
pub fn snap_multi_row_ring_array_to_grid(
    array: &MultiRowRingArray,
    dimensions: (usize, usize, usize),
    spacing_m: f64,
) -> KwaversResult<MultiRowRingArray> {
    if !spacing_m.is_finite() || spacing_m <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "spacing_m must be positive and finite, got {spacing_m}"
        )));
    }
    let mut snapped = Vec::with_capacity(array.element_count());
    for &point in array.elements() {
        let index = map_ring_point_to_grid(dimensions, spacing_m, point)?;
        snapped.push(grid_index_to_ring_point(dimensions, spacing_m, index));
    }
    MultiRowRingArray::from_ordered_elements(
        array.circumferential_elements(),
        array.rows(),
        array.diameter_m(),
        array.row_spacing_m(),
        snapped,
    )
}

fn run_pstd_transmit(
    sound_speed_m_s: &Array3<f64>,
    receiver_indices: &[(usize, usize, usize)],
    source_indices: &[(usize, usize, usize)],
    frequency_hz: f64,
    steps: usize,
    config: BreastUstPstdDatasetConfig,
) -> KwaversResult<Array2<f64>> {
    let (nx, ny, nz) = sound_speed_m_s.dim();
    let grid = Grid::new(
        nx,
        ny,
        nz,
        config.spacing_m,
        config.spacing_m,
        config.spacing_m,
    )?;
    let density = Array3::from_elem((nx, ny, nz), config.density_kg_m3);
    let medium = HeterogeneousFactory::from_arrays(
        sound_speed_m_s.clone(),
        density,
        None,
        None,
        None,
        frequency_hz,
    )
    .map_err(KwaversError::InvalidInput)?;

    let mut p_mask = Array3::<f64>::zeros((nx, ny, nz));
    for &(i, j, k) in source_indices {
        p_mask[[i, j, k]] += 1.0;
    }
    let source = GridSource {
        p_mask: Some(p_mask),
        p_signal: Some(tone_signal(frequency_hz, steps, config)),
        p_mode: SourceMode::Additive,
        ..GridSource::new_empty()
    };

    let pstd_config = PSTDConfig {
        nt: steps,
        dt: config.time_step_s,
        boundary: pstd_boundary(config.cpml_thickness_cells),
        smooth_sources: false,
        ..Default::default()
    };
    let mut solver = PSTDSolver::new(pstd_config, grid, &medium, source)?;
    solver.sensor_recorder =
        SensorRecorder::from_ordered_indices(receiver_indices.to_vec(), steps)?;
    solver.run_orchestrated(steps)?.ok_or_else(|| {
        KwaversError::InvalidInput("PSTD acquisition produced no receiver data".into())
    })
}

fn pstd_boundary(cpml_thickness_cells: usize) -> BoundaryConfig {
    if cpml_thickness_cells == 0 {
        BoundaryConfig::None
    } else {
        BoundaryConfig::CPML(CPMLConfig::with_thickness(cpml_thickness_cells))
    }
}

fn tone_signal(frequency_hz: f64, steps: usize, config: BreastUstPstdDatasetConfig) -> Array2<f64> {
    Array2::from_shape_fn((1, steps), |(_, n)| {
        let phase = 2.0 * PI * frequency_hz * n as f64 * config.time_step_s;
        config.source_amplitude_pa * phase.sin()
    })
}

fn frequency_bin(
    samples: ArrayView1<'_, f64>,
    frequency_hz: f64,
    dt: f64,
    start_sample: usize,
) -> Complex64 {
    let window = samples.slice(s![start_sample..]);
    let scale = 2.0 / window.len() as f64;
    samples.iter().skip(start_sample).enumerate().fold(
        Complex64::new(0.0, 0.0),
        |acc, (n, &sample)| {
            let phase = -2.0 * PI * frequency_hz * (start_sample + n) as f64 * dt;
            acc + Complex64::new(phase.cos(), phase.sin()) * sample
        },
    ) * scale
}

fn map_ring_points_to_grid(
    dims: (usize, usize, usize),
    config: BreastUstPstdDatasetConfig,
    points: &[ElementPosition],
) -> KwaversResult<Vec<(usize, usize, usize)>> {
    points
        .iter()
        .map(|point| map_ring_point_to_grid(dims, config.spacing_m, *point))
        .collect()
}

fn map_ring_point_to_grid(
    (nx, ny, nz): (usize, usize, usize),
    spacing_m: f64,
    point: ElementPosition,
) -> KwaversResult<(usize, usize, usize)> {
    let center = [
        0.5 * (nx - 1) as f64 * spacing_m,
        0.5 * (ny - 1) as f64 * spacing_m,
        0.5 * (nz - 1) as f64 * spacing_m,
    ];
    let coord = [
        center[0] + point.x_m,
        center[1] + point.y_m,
        center[2] + point.z_m,
    ];
    let max = [
        (nx - 1) as f64 * spacing_m,
        (ny - 1) as f64 * spacing_m,
        (nz - 1) as f64 * spacing_m,
    ];
    for axis in 0..3 {
        if coord[axis] < 0.0 || coord[axis] > max[axis] {
            return Err(KwaversError::InvalidInput(format!(
                "ring point {:?} maps outside centered PSTD grid bounds {:?}",
                point, max
            )));
        }
    }
    Ok((
        (coord[0] / spacing_m).round() as usize,
        (coord[1] / spacing_m).round() as usize,
        (coord[2] / spacing_m).round() as usize,
    ))
}

fn grid_index_to_ring_point(
    (nx, ny, nz): (usize, usize, usize),
    spacing_m: f64,
    (ix, iy, iz): (usize, usize, usize),
) -> ElementPosition {
    let center = [
        0.5 * (nx - 1) as f64,
        0.5 * (ny - 1) as f64,
        0.5 * (nz - 1) as f64,
    ];
    ElementPosition {
        x_m: (ix as f64 - center[0]) * spacing_m,
        y_m: (iy as f64 - center[1]) * spacing_m,
        z_m: (iz as f64 - center[2]) * spacing_m,
    }
}

fn time_steps_for_frequency(
    frequency_hz: f64,
    config: BreastUstPstdDatasetConfig,
) -> KwaversResult<usize> {
    time_steps_for_cycles(
        frequency_hz,
        config.time_step_s,
        config.cycles_per_frequency,
    )
}

fn frequency_bin_start_step(
    frequency_hz: f64,
    config: BreastUstPstdDatasetConfig,
    total_steps: usize,
) -> KwaversResult<usize> {
    let bin_steps = time_steps_for_cycles(
        frequency_hz,
        config.time_step_s,
        config.frequency_bin_cycles,
    )?;
    Ok(total_steps.saturating_sub(bin_steps))
}

fn time_steps_for_cycles(frequency_hz: f64, dt: f64, cycles: usize) -> KwaversResult<usize> {
    let raw = (cycles as f64 / (frequency_hz * dt)).ceil();
    if !raw.is_finite() || raw > usize::MAX as f64 {
        return Err(KwaversError::InvalidInput(format!(
            "time-step count is not representable for frequency {frequency_hz}"
        )));
    }
    Ok((raw as usize).max(2))
}

fn validate_config(config: &BreastUstPstdDatasetConfig) -> KwaversResult<()> {
    if !config.spacing_m.is_finite() || config.spacing_m <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "spacing_m must be positive and finite, got {}",
            config.spacing_m
        )));
    }
    if !config.time_step_s.is_finite() || config.time_step_s <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "time_step_s must be positive and finite, got {}",
            config.time_step_s
        )));
    }
    if config.cycles_per_frequency == 0 {
        return Err(KwaversError::InvalidInput(
            "cycles_per_frequency must be positive".to_owned(),
        ));
    }
    if config.frequency_bin_cycles == 0 || config.frequency_bin_cycles > config.cycles_per_frequency
    {
        return Err(KwaversError::InvalidInput(format!(
            "frequency_bin_cycles must be in 1..={}, got {}",
            config.cycles_per_frequency, config.frequency_bin_cycles
        )));
    }
    if !config.source_amplitude_pa.is_finite() {
        return Err(KwaversError::InvalidInput(format!(
            "source_amplitude_pa must be finite, got {}",
            config.source_amplitude_pa
        )));
    }
    if !config.density_kg_m3.is_finite() || config.density_kg_m3 <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "density_kg_m3 must be positive and finite, got {}",
            config.density_kg_m3
        )));
    }
    Ok(())
}

fn validate_sound_speed(sound_speed_m_s: &Array3<f64>) -> KwaversResult<()> {
    if sound_speed_m_s.is_empty() {
        return Err(KwaversError::InvalidInput(
            "sound_speed_m_s volume must not be empty".to_owned(),
        ));
    }
    for &speed in sound_speed_m_s {
        if !speed.is_finite() || speed <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "sound speed must be positive and finite, got {speed}"
            )));
        }
    }
    Ok(())
}

fn validate_frequencies(frequencies_hz: &[f64], dt: f64) -> KwaversResult<()> {
    if frequencies_hz.is_empty() {
        return Err(KwaversError::InvalidInput(
            "frequencies_hz must not be empty".to_owned(),
        ));
    }
    let nyquist = 0.5 / dt;
    for &frequency_hz in frequencies_hz {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "frequency must be positive and finite, got {frequency_hz}"
            )));
        }
        if frequency_hz >= nyquist {
            return Err(KwaversError::InvalidInput(format!(
                "frequency {frequency_hz} Hz must be below Nyquist {nyquist} Hz"
            )));
        }
    }
    Ok(())
}

fn validate_cfl(
    sound_speed_m_s: &Array3<f64>,
    config: BreastUstPstdDatasetConfig,
) -> KwaversResult<()> {
    let max_sound_speed = sound_speed_m_s
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let cfl = max_sound_speed * config.time_step_s / config.spacing_m;
    if cfl > PSTD_DATASET_CFL_LIMIT {
        return Err(KwaversError::InvalidInput(format!(
            "PSTD acquisition CFL {cfl:.6} exceeds limit {PSTD_DATASET_CFL_LIMIT:.6}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
