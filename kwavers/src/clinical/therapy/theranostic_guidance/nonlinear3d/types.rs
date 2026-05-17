//! Configuration, volume, aperture, and result types for the nonlinear 3-D solver.
//!
//! Grid-index arithmetic lives in [`super::grid`]; it is re-exported here so
//! that all existing `use super::types::{GridIndex, flat_index, grid_point_m}`
//! imports in sibling modules continue to compile unchanged.

use ndarray::Array3;

use crate::core::error::{KwaversError, KwaversResult};

use super::super::{AnatomyKind, Point3};

// Re-export geometry primitives — SSOT is `grid.rs`.
pub(crate) use super::grid::{flat_index, grid_point_m, GridIndex};

pub const THERANOSTIC_NONLINEAR_3D_MODEL: &str =
    "westervelt_3d_discrete_adjoint_fwi_plus_rayleigh_plesset_passive_inverse";
pub const THERANOSTIC_NONLINEAR_3D_PROPAGATOR: &str =
    "heterogeneous_westervelt_fdtd_3d_second_order_discrete_adjoint";
pub const THERANOSTIC_CAVITATION_INVERSE_MODEL: &str =
    "rayleigh_plesset_subharmonic_green_projected_gradient_inverse";

#[derive(Clone, Debug)]
pub struct Nonlinear3dConfig {
    pub anatomy: AnatomyKind,
    pub grid_size: usize,
    pub iterations: usize,
    pub element_count: usize,
    pub receiver_count: usize,
    pub source_encoding_count: usize,
    pub frequency_hz: f64,
    pub source_pressure_pa: f64,
    pub cycles: f64,
    pub cfl: f64,
    pub treatment_window_radius_m: f64,
    pub min_points_per_wavelength: f64,
    pub checkpoint_interval_steps: usize,
    pub lesion_delta_c_m_s: f64,
    pub lesion_delta_beta: f64,
    pub sound_speed_regularization: f64,
    pub nonlinearity_regularization: f64,
    pub gradient_smoothing_steps: usize,
    pub bubble_radius_m: f64,
    pub bubble_time_steps_per_period: usize,
    pub inertial_mi_threshold: f64,
    pub cavitation_iterations: usize,
    pub cavitation_regularization: f64,
}

impl Nonlinear3dConfig {
    #[must_use]
    pub fn new(anatomy: AnatomyKind) -> Self {
        Self {
            anatomy,
            grid_size: 20,
            iterations: 2,
            element_count: if matches!(anatomy, AnatomyKind::Brain) {
                128
            } else {
                96
            },
            receiver_count: 48,
            source_encoding_count: 3,
            frequency_hz: if matches!(anatomy, AnatomyKind::Brain) {
                650_000.0
            } else {
                500_000.0
            },
            source_pressure_pa: if matches!(anatomy, AnatomyKind::Brain) {
                1.5e5
            } else {
                6.0e6
            },
            cycles: 3.0,
            cfl: 0.42,
            treatment_window_radius_m: 0.04,
            min_points_per_wavelength: 6.0,
            checkpoint_interval_steps: 128,
            lesion_delta_c_m_s: -35.0,
            lesion_delta_beta: 0.85,
            sound_speed_regularization: 2.0e-3,
            nonlinearity_regularization: 1.0e-3,
            gradient_smoothing_steps: 2,
            bubble_radius_m: 2.0e-6,
            bubble_time_steps_per_period: 96,
            inertial_mi_threshold: 1.9,
            cavitation_iterations: 24,
            cavitation_regularization: 1.0e-4,
        }
    }

    pub fn validate(&self) -> KwaversResult<()> {
        if self.grid_size < 12 {
            return Err(KwaversError::InvalidInput(
                "nonlinear 3-D grid_size must be at least 12".to_owned(),
            ));
        }
        if self.iterations == 0 {
            return Err(KwaversError::InvalidInput(
                "nonlinear 3-D FWI iterations must be positive".to_owned(),
            ));
        }
        if self.element_count < 8 || self.receiver_count < 4 {
            return Err(KwaversError::InvalidInput(
                "nonlinear 3-D aperture requires at least 8 sources and 4 receivers".to_owned(),
            ));
        }
        if self.source_encoding_count == 0 {
            return Err(KwaversError::InvalidInput(
                "nonlinear 3-D source_encoding_count must be positive".to_owned(),
            ));
        }
        if self.checkpoint_interval_steps == 0 {
            return Err(KwaversError::InvalidInput(
                "nonlinear 3-D checkpoint_interval_steps must be positive".to_owned(),
            ));
        }
        for (name, value) in [
            ("frequency_hz", self.frequency_hz),
            ("source_pressure_pa", self.source_pressure_pa),
            ("cycles", self.cycles),
            ("cfl", self.cfl),
            ("treatment_window_radius_m", self.treatment_window_radius_m),
            ("min_points_per_wavelength", self.min_points_per_wavelength),
            (
                "sound_speed_regularization",
                self.sound_speed_regularization,
            ),
            (
                "nonlinearity_regularization",
                self.nonlinearity_regularization,
            ),
            ("bubble_radius_m", self.bubble_radius_m),
            ("inertial_mi_threshold", self.inertial_mi_threshold),
            ("cavitation_regularization", self.cavitation_regularization),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(KwaversError::InvalidInput(format!(
                    "{name} must be nonnegative and finite"
                )));
            }
        }
        for (name, value) in [
            ("frequency_hz", self.frequency_hz),
            ("source_pressure_pa", self.source_pressure_pa),
            ("cycles", self.cycles),
            ("cfl", self.cfl),
            ("min_points_per_wavelength", self.min_points_per_wavelength),
            ("bubble_radius_m", self.bubble_radius_m),
            ("inertial_mi_threshold", self.inertial_mi_threshold),
            ("cavitation_regularization", self.cavitation_regularization),
        ] {
            if value <= 0.0 {
                return Err(KwaversError::InvalidInput(format!(
                    "{name} must be positive"
                )));
            }
        }
        if !self.lesion_delta_c_m_s.is_finite() || self.lesion_delta_c_m_s == 0.0 {
            return Err(KwaversError::InvalidInput(
                "lesion_delta_c_m_s must be finite and nonzero".to_owned(),
            ));
        }
        if !self.lesion_delta_beta.is_finite() || self.lesion_delta_beta == 0.0 {
            return Err(KwaversError::InvalidInput(
                "lesion_delta_beta must be finite and nonzero".to_owned(),
            ));
        }
        if !(0.0..=0.95).contains(&self.cfl) {
            return Err(KwaversError::InvalidInput(
                "nonlinear 3-D CFL must lie in (0, 0.95]".to_owned(),
            ));
        }
        if self.bubble_time_steps_per_period < 24 || self.cavitation_iterations == 0 {
            return Err(KwaversError::InvalidInput(
                "Rayleigh-Plesset and cavitation inverse iteration counts are too small".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Nonlinear3dVolume {
    pub anatomy: AnatomyKind,
    pub ct_hu: Array3<f64>,
    pub label: Array3<i16>,
    pub body_mask: Array3<bool>,
    pub target_mask: Array3<bool>,
    pub inversion_mask: Array3<bool>,
    pub density_kg_m3: Array3<f64>,
    pub background_beta: Array3<f64>,
    pub true_beta: Array3<f64>,
    pub background_sound_speed_m_s: Array3<f64>,
    pub true_sound_speed_m_s: Array3<f64>,
    /// Per-voxel attenuation coefficient at 1 MHz, in Np/m. The
    /// frequency dependence follows a power law `α(f) = α(1MHz) · f_MHz^y`
    /// where `y` is the per-voxel exponent in `attenuation_power_law_y`. See
    /// `volume::attenuation_np_per_m_mhz_from_hu` for the tissue-class table
    /// and the Hamilton & Blackstock 1998 / Connor & Hynynen 2002 references.
    pub attenuation_np_per_m_mhz: Array3<f64>,
    /// Per-voxel power-law exponent `y` for biological tissue absorption
    /// (Treeby & Cox 2010 §I; Szabo 1995): `α(f) = α(1MHz) · f_MHz^y`.
    /// Soft tissue is slightly superlinear (`y ≈ 1.05`); skull bone follows
    /// the Stokes-Kirchhoff classical viscous law (`y ≈ 2`, Connor & Hynynen
    /// 2002). The leading-order `y = 1` simplification (linear scaling) is a
    /// good approximation for soft tissue but underestimates skull attenuation
    /// at high frequencies and overestimates it at low frequencies by up to
    /// 4× — critical for transcranial subharmonic cavitation receive paths.
    pub attenuation_power_law_y: Array3<f64>,
    pub spacing_m: f64,
    pub source_dimensions: [usize; 3],
    pub source_spacing_m: [f64; 3],
    pub crop_bounds_index: [usize; 6],
    pub aperture_direction: Option<[f64; 3]>,
    pub focus: GridIndex,
}

#[derive(Clone, Debug)]
pub(crate) struct Nonlinear3dAperture {
    pub sources: Vec<GridIndex>,
    pub receivers: Vec<GridIndex>,
    pub therapy_points_m: Vec<Point3>,
    pub receiver_points_m: Vec<Point3>,
    pub model_name: String,
    pub focus: GridIndex,
}

#[derive(Clone, Debug)]
pub struct VolumeReconstructionMetrics {
    pub dice_equal_area: f64,
    pub cnr: f64,
    pub nrmse: f64,
}

#[derive(Clone, Debug)]
pub struct Nonlinear3dResult {
    pub ct_hu: Array3<f64>,
    pub label: Array3<i16>,
    pub body_mask: Array3<bool>,
    pub target_mask: Array3<bool>,
    pub inversion_mask: Array3<bool>,
    pub background_sound_speed_m_s: Array3<f64>,
    pub true_sound_speed_m_s: Array3<f64>,
    pub reconstructed_sound_speed_m_s: Array3<f64>,
    pub reconstructed_delta_c_m_s: Array3<f64>,
    pub background_beta: Array3<f64>,
    pub true_beta: Array3<f64>,
    pub reconstructed_beta: Array3<f64>,
    pub reconstructed_delta_beta: Array3<f64>,
    pub multiparameter_fwi_score: Array3<f64>,
    pub nonlinear_fusion_score: Array3<f64>,
    pub westervelt_peak_pressure_pa: Array3<f64>,
    pub cavitation_source_density: Array3<f64>,
    pub reconstructed_cavitation_density: Array3<f64>,
    pub fwi_objective_history: Vec<f64>,
    pub cavitation_objective_history: Vec<f64>,
    pub therapy_points_m: Vec<Point3>,
    pub receiver_points_m: Vec<Point3>,
    pub spacing_m: f64,
    pub source_dimensions: [usize; 3],
    pub source_spacing_m: [f64; 3],
    pub crop_bounds_index: [usize; 6],
    pub treatment_window_radius_m: f64,
    pub wavelength_min_m: f64,
    pub points_per_wavelength_min: f64,
    pub resolution_meets_min_ppw: bool,
    pub dt_s: f64,
    pub time_steps: usize,
    pub active_voxels: usize,
    pub fwi_metrics: VolumeReconstructionMetrics,
    pub cavitation_metrics: VolumeReconstructionMetrics,
    pub fusion_metrics: VolumeReconstructionMetrics,
    pub aperture_model: String,
    pub model_family: &'static str,
    pub propagator_model: &'static str,
    pub cavitation_inverse_model: &'static str,
    pub is_full_wave_inversion: bool,
    pub uses_nonlinear_wave_propagation: bool,
    pub uses_rayleigh_plesset: bool,
}
