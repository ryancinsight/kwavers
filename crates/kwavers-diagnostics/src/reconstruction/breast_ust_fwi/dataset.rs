//! PSTD acquisition dataset generation for breast UST FWI.
//!
//! The clinical layer owns acquisition orchestration: the solver layer provides
//! PSTD propagation and frequency-domain inversion, while this module maps the
//! breast ring-array protocol onto those solver contracts.

use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_domain::grid::Grid;
use kwavers_domain::medium::heterogeneous::HeterogeneousFactory;
use kwavers_domain::sensor::recorder::simple::SensorRecorder;
use kwavers_domain::source::{GridSource, SourceMode};
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;
use kwavers_solver::forward::pstd::{PSTDConfig, PSTDSolver};
use kwavers_solver::inverse::fwi::frequency_domain::FrequencyObservation;
use ndarray::{s, Array2, Array3};
use num_complex::Complex64;

mod signal;
mod validation;
use signal::{
    frequency_bin, frequency_bin_start_step, grid_index_to_ring_point, map_ring_point_to_grid,
    map_ring_points_to_grid, pstd_boundary, time_steps_for_frequency, tone_signal,
};
use validation::{validate_cfl, validate_config, validate_frequencies, validate_sound_speed};

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
            density_kg_m3: DENSITY_WATER_NOMINAL, // water coupling bath
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

    // GPU PSTD path (compile-time `gpu` feature): all axes power-of-2 and
    // ≤ 256 → drive the GPU-resident pipeline via the canonical kwavers entry.
    // Falls back to the CPU PSTDSolver path on shape rejection or runtime error.
    #[cfg(feature = "gpu")]
    {
        if nx.is_power_of_two()
            && ny.is_power_of_two()
            && nz.is_power_of_two()
            && nx <= 256
            && ny <= 256
            && nz <= 256
        {
            if let Some(traces) =
                try_run_gpu_pstd_transmit(&grid, &medium, &source, receiver_indices, steps, config)
            {
                return Ok(traces);
            }
        }
    }

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

#[cfg(feature = "gpu")]
fn try_run_gpu_pstd_transmit(
    grid: &Grid,
    medium: &kwavers_domain::medium::heterogeneous::HeterogeneousMedium,
    source: &GridSource,
    receiver_indices: &[(usize, usize, usize)],
    steps: usize,
    config: BreastUstPstdDatasetConfig,
) -> Option<Array2<f64>> {
    use kwavers_gpu::pstd_gpu::{run_gpu_pstd, GpuPstdRunConfig};
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let mut sensor_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    for &(i, j, k) in receiver_indices {
        sensor_mask[[i, j, k]] = true;
    }
    // GPU PSTD uses its own polynomial-PML inside the wgsl pipeline; clinical
    // dataset gen sets `pml_inside = true` when the CPU CPML thickness is
    // nonzero so the GPU path matches the CPU absorbing boundary semantics.
    let gpu_config = GpuPstdRunConfig {
        time_steps: steps,
        dt: config.time_step_s,
        alpha_coeff_db: 0.0,
        alpha_power: 1.0,
        pml_size: if config.cpml_thickness_cells == 0 {
            None
        } else {
            Some(config.cpml_thickness_cells)
        },
        pml_size_xyz: None,
        pml_inside: config.cpml_thickness_cells > 0,
        pml_alpha_xyz: None,
    };
    match run_gpu_pstd(grid, medium, source, &sensor_mask, gpu_config) {
        Ok(traces) => Some(traces),
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests;
