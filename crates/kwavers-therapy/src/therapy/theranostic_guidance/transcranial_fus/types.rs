use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use kwavers_core::constants::tissue_acoustics::DENSITY_BRAIN;
use leto::{Array1, Array2, Array3};

/// Full configuration for the transcranial FUS planning pipeline.
#[derive(Clone, Debug)]
pub struct TranscranialFusPlanConfig {
    /// Number of hemispherical transducer elements.
    pub element_count: usize,
    /// Centre frequency `Hz`.
    pub frequency_hz: f64,
    /// Hemisphere radius `m`.
    pub radius_m: f64,
    /// Minimum polar angle for the element cap `rad`.
    pub cap_min_polar_rad: f64,
    /// Maximum polar angle for the element cap `rad`.
    pub cap_max_polar_rad: f64,
    /// Brain sound speed [m/s].
    pub brain_c: f64,
    /// Skull sound speed [m/s].
    pub skull_c: f64,
    /// Desired peak pressure at focus `Pa`.
    pub target_peak_pa: f64,
    /// Number of samples along each skull ray.
    pub samples_per_ray: usize,
    /// Grid-point chunk size for Rayleigh integration (memory control).
    pub chunk_size: usize,
    /// Mechanical-index threshold for inertial cavitation probability.
    pub inertial_mi_threshold: f64,
    /// Brain tissue density [kg/m³].
    pub rho_brain: f64,
    /// Subspot raster pitch `m`.
    pub pitch_m: f64,
    /// Mechanical index for BBB opening.
    pub mechanical_index_bbb: f64,
    /// Total sonication time `s`.
    pub sonication_s: f64,
    /// Duty cycle (0, 1].
    pub duty_cycle: f64,
    /// Focal Gaussian radius for BBB dose accumulation `m`.
    pub focal_radius_m: f64,
    /// BBB Hill model D₅₀ dose.
    pub d50: f64,
    /// BBB Hill model cooperativity exponent.
    pub hill_n: f64,
}

impl Default for TranscranialFusPlanConfig {
    fn default() -> Self {
        Self {
            element_count: 1024,
            frequency_hz: 0.65 * MHZ_TO_HZ,
            radius_m: 0.150,
            cap_min_polar_rad: 0.22,
            cap_max_polar_rad: 1.18,
            brain_c: SOUND_SPEED_TISSUE,
            skull_c: 2800.0,
            target_peak_pa: MPA_TO_PA,
            samples_per_ray: 192,
            chunk_size: 512,
            inertial_mi_threshold: 1.9,
            rho_brain: DENSITY_BRAIN,
            pitch_m: 3.0e-3,
            mechanical_index_bbb: 0.45,
            sonication_s: 60.0,
            duty_cycle: 0.02,
            focal_radius_m: 2.0e-3,
            d50: 0.40,
            hill_n: 2.5,
        }
    }
}

/// All outputs produced by the transcranial FUS planning pipeline.
#[derive(Debug)]
pub struct TranscranialFusPlan {
    /// Synthesised peak-positive pressure field `Pa`, shape (nx, ny, nz).
    pub pressure_pa: Array3<f32>,
    /// Time-averaged intensity I = p²/(2ρc) [W/m²].
    pub intensity_w_m2: Array3<f32>,
    /// Mechanical index MI = p/1e6 / √f_MHz.
    pub mechanical_index: Array3<f32>,
    /// Inertial-cavitation probability (logistic in MI).
    pub cavitation_probability: Array3<f32>,
    /// Per-element phase correction `rad`.
    pub phases_rad: Array1<f64>,
    /// Per-element one-way skull delay `s`.
    pub delays_s: Array1<f64>,
    /// Per-element skull path length `m`.
    pub skull_lengths_m: Array1<f64>,
    /// Per-element amplitude transmission weight.
    pub amplitude_weights: Array1<f64>,
    /// Element positions in 3-D space `m`, shape (N, 3).
    pub element_positions_m: Array2<f64>,
    /// GBM subspot voxel indices, shape (M, 3).
    pub subspot_indices: Array2<usize>,
    /// BBB acoustic dose field.
    pub bbb_dose: Array3<f32>,
    /// BBB permeability field (Hill model).
    pub bbb_permeability: Array3<f32>,
    /// Stable cavitation probability field.
    pub bbb_stable_cavitation: Array3<f32>,
    /// Inertial cavitation risk field.
    pub bbb_inertial_risk: Array3<f32>,
    /// Grid index of the sonication focus.
    pub focus_index: [usize; 3],
    /// Number of active elements.
    pub element_count: usize,
    /// Operating frequency `Hz`.
    pub frequency_hz: f64,
}
