//! Geometry, medium, and array construction for the linear-array neuromodulation study.
//!
//! Every physical input is derived here once: wavelength from the brain sound speed,
//! pitch from the grating-lobe criterion, element count from the aperture. The
//! simulation and the analytical oracle consume the same element list.

pub mod analytic;
pub mod drive;
pub mod metrics;
pub mod plots;
pub mod simulate;

use anyhow::{anyhow, Result};
use eunomia::convert::IntegerTarget;
use eunomia::FloatElement;
use kwavers_transducer::transducers::physics::{PlanarApertureGeometry, PlanarApertureShape};
use kwavers_transducer::KWaveArray;
use leto::Array2;
use std::f64::consts::PI;

/// Brain gray-matter sound speed in m/s.
pub const SOUND_SPEED_M_S: f64 = kwavers_core::constants::tissue_acoustics::SOUND_SPEED_BRAIN;
/// Brain density in kg/m³.
pub const DENSITY_KG_M3: f64 = 1040.0;
/// Brain B/A nonlinearity parameter (Duck 1990 Table 4.16 brain mean).
pub const B_OVER_A: f64 = 6.55;
/// Brain gray-matter power-law absorption coefficient at 1 MHz, dB/(MHz·cm).
pub const ALPHA_DB_CM_MHZ: f64 = 0.7;
/// Power-law absorption exponent (gray-matter tabulation).
///
/// The causal fractional-Laplacian dispersion term is proportional to
/// `tan(πy/2)`, which diverges at `y = 1`; the full-wave runs therefore omit it
/// (`PowerLawDispersion::Omitted`, k-Wave's `alpha_mode = 'no_dispersion'`) and
/// keep the absorption term, matching the oracle's `e^{−α r}` per path.
pub const ALPHA_POWER: f64 = 1.0;
/// Drive frequency in Hz.
pub const FREQUENCY_HZ: f64 = 2.0e6;
/// Points per wavelength of the full-wave grid.
///
/// The k-space corrected PSTD scheme (k-Wave's) propagates exactly in a
/// homogeneous medium at any resolution above the Nyquist limit of 2; 3 keeps
/// the 2 MHz band inside the grid with margin for the burst's bandwidth.
pub const POINTS_PER_WAVELENGTH: f64 = 3.0;
/// Number of tone-burst cycles driven on every element.
pub const BURST_CYCLES: f64 = 8.0;
/// Courant number for the PSTD time step.
pub const CFL: f64 = 0.3;
/// Source amplitude in pascals for the linear runs (linear response is scale-free).
pub const SOURCE_AMPLITUDE_PA: f64 = 1.0e5;
/// Element kerf fraction: element width is `(1 - KERF_FRACTION) × pitch`.
pub const KERF_FRACTION: f64 = 0.1;
/// Grating-lobe-free steering limit for a 30° steering requirement, in radians.
pub const STEERING_LIMIT_RAD: f64 = PI / 6.0;

/// Array aperture, metres.
pub const ARRAY_APERTURE_M: f64 = 30.0e-3;
/// Array elevation, metres.
pub const ARRAY_ELEVATION_M: f64 = 1.5e-3;
/// Focal pressure target at 6–8 cm, pascals.
pub const TARGET_PRESSURE_PA: f64 = 120.0e3;
/// Drive voltage at which the target is to be reached, volts peak.
pub const TARGET_DRIVE_VP: f64 = 30.0;
/// Focal depths in question, millimetres.
pub const FOCUS_FOCI_MM: [f64; 2] = [70.0, 80.0];

/// Wavelength in brain tissue, metres.
#[must_use]
pub fn wavelength_m() -> f64 {
    SOUND_SPEED_M_S / FREQUENCY_HZ
}

/// Element pitch that avoids grating lobes up to the steering limit.
///
/// Steered to `θ_s`, a linear array of pitch `p` has grating lobes where
/// `sin θ = sin θ_s ± m λ/p`. Keeping the nearest one, `sin θ_s − λ/p`, out of
/// real space (`≤ −1`) at the full steering limit gives `p ≤ λ / (1 + sin θ_max)`.
#[must_use]
pub fn pitch_m() -> f64 {
    wavelength_m() / (1.0 + STEERING_LIMIT_RAD.sin())
}

/// Number of whole elements that fit a given linear aperture at the given pitch.
#[must_use]
pub fn element_count(aperture_m: f64, pitch: f64) -> usize {
    usize::from_floor(aperture_m / pitch)
}

/// Full-wave grid shared by the mm-scale runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridSpec {
    /// Cells along the lateral (array) axis.
    pub nx: usize,
    /// Cells along the elevation axis.
    pub ny: usize,
    /// Cells along depth (propagation) axis.
    pub nz: usize,
    /// Isotropic cell size in metres.
    pub dx: f64,
    /// Index of the array plane along depth.
    pub array_plane_k: usize,
    /// CPML thickness in cells on every face.
    pub pml_cells: usize,
}

impl GridSpec {
    /// Physical lateral centre of the grid in metres.
    #[must_use]
    pub fn lateral_center_m(&self) -> f64 {
        0.5 * f64::from_count(self.nx - 1) * self.dx
    }

    /// Physical elevation centre of the grid in metres.
    #[must_use]
    pub fn elevation_center_m(&self) -> f64 {
        0.5 * f64::from_count(self.ny - 1) * self.dx
    }

    /// Physical depth of the array plane in metres.
    #[must_use]
    pub fn array_plane_m(&self) -> f64 {
        f64::from_count(self.array_plane_k) * self.dx
    }
}

/// One linear array: its element centres and sizes in physical coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearArraySpec {
    /// Element pitch in metres.
    pub pitch_m: f64,
    /// Element width along the array in metres.
    pub element_width_m: f64,
    /// Element height (elevation extent) in metres.
    pub element_height_m: f64,
    /// Element centre x-coordinates in metres.
    pub x_m: Vec<f64>,
    /// Elevation centre in metres.
    pub y_m: f64,
    /// Array-plane depth in metres.
    pub z_m: f64,
}

impl LinearArraySpec {
    /// Build an array of whole elements centred on the grid's lateral axis.
    #[must_use]
    pub fn centered(aperture_m: f64, element_height_m: f64, grid: &GridSpec) -> Self {
        Self::at(
            aperture_m,
            element_height_m,
            [
                grid.lateral_center_m(),
                grid.elevation_center_m(),
                grid.array_plane_m(),
            ],
        )
    }

    /// Build an array of whole elements centred at `center_m` (x, y, z) in metres.
    #[must_use]
    pub fn at(aperture_m: f64, element_height_m: f64, center_m: [f64; 3]) -> Self {
        let pitch = pitch_m();
        let count = element_count(aperture_m, pitch);
        let x_m = (0..count)
            .map(|i| center_m[0] + (f64::from_count(i) - 0.5 * f64::from_count(count - 1)) * pitch)
            .collect();
        Self {
            pitch_m: pitch,
            element_width_m: (1.0 - KERF_FRACTION) * pitch,
            element_height_m,
            x_m,
            y_m: center_m[1],
            z_m: center_m[2],
        }
    }

    /// Number of elements.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.x_m.len()
    }

    /// Focal point at radial distance `focal_m` from the array centre, steered by `steer_rad`.
    #[must_use]
    pub fn focus_m(&self, focal_m: f64, steer_rad: f64, grid: &GridSpec) -> [f64; 3] {
        [
            grid.lateral_center_m() + focal_m * steer_rad.sin(),
            self.y_m,
            self.z_m + focal_m * steer_rad.cos(),
        ]
    }

    /// Rectangular element model for the full-wave solver, rasterized from the
    /// same planar apertures the Rayleigh oracle integrates.
    ///
    /// # Errors
    ///
    /// Returns an error if an element geometry is invalid.
    pub fn kwave_array(&self) -> Result<KWaveArray> {
        let mut array = KWaveArray::with_params(FREQUENCY_HZ, SOUND_SPEED_M_S);
        for &x in &self.x_m {
            let geometry = PlanarApertureGeometry::oriented(
                [x, self.y_m, self.z_m],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                PlanarApertureShape::Rectangle {
                    width_m: self.element_width_m,
                    height_m: self.element_height_m,
                },
            )
            .map_err(|error| anyhow!("{error:?}"))?;
            array.add_planar_aperture_element(geometry);
        }
        Ok(array)
    }

    /// Per-element focusing delays (s), zero for the farthest element:
    /// `τ_i = (max_j d_j − d_i) / c` with `d_i` the centre-to-focus distance.
    #[must_use]
    pub fn focus_delays_s(&self, focus: [f64; 3]) -> Vec<f64> {
        let distances: Vec<f64> = self
            .x_m
            .iter()
            .map(|&x| {
                (x - focus[0])
                    .hypot(self.y_m - focus[1])
                    .hypot(self.z_m - focus[2])
            })
            .collect();
        let farthest = distances.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        distances
            .iter()
            .map(|&distance| (farthest - distance) / SOUND_SPEED_M_S)
            .collect()
    }
}

/// Tone-burst drive for every element: Hann-enveloped, delayed by its focusing delay.
///
/// Element `i` emits `A · env(t − τ_i) · sin(2π f (t − τ_i))` for the burst
/// duration `BURST_CYCLES / f`, with `τ_i` from [`LinearArraySpec::focus_delays_s`].
#[must_use]
pub fn element_drive(delays_s: &[f64], amplitude_pa: f64, steps: usize, dt: f64) -> Array2<f64> {
    let burst_s = BURST_CYCLES / FREQUENCY_HZ;
    let mut signals = Array2::<f64>::zeros([delays_s.len(), steps]);
    for (i, &tau) in delays_s.iter().enumerate() {
        for step in 0..steps {
            let local = f64::from_count(step) * dt - tau;
            if (0.0..=burst_s).contains(&local) {
                let envelope = 0.5 * (1.0 - (2.0 * PI * local / burst_s).cos());
                signals[[i, step]] =
                    amplitude_pa * envelope * (2.0 * PI * FREQUENCY_HZ * local).sin();
            }
        }
    }
    signals
}

/// Time step for the PSTD run on the given grid.
#[must_use]
pub fn time_step_s(dx: f64) -> f64 {
    CFL * dx / SOUND_SPEED_M_S
}

/// Power-law absorption coefficient in Np/m at the drive frequency.
#[must_use]
pub fn attenuation_np_m() -> f64 {
    kwavers_core::units::alpha_db_per_mhz_cm_to_np_per_m(
        ALPHA_DB_CM_MHZ,
        FREQUENCY_HZ / kwavers_core::constants::numerical::MHZ_TO_HZ,
        ALPHA_POWER,
    )
}

/// Forward-wave factor of the solver's additive pressure source relative to the
/// baffled Rayleigh integral with surface pressure `A`.
///
/// The solver pre-scales additive sources by `2·dt/(N·c·dx)` (k-Wave's convention,
/// `kwavers-solver` `fdtd/source_handler/scaling.rs`), which makes the forward
/// wave equal the baffled value. The planar calibration in
/// `simulate::run_plane_wave_calibration` measures this factor; the oracle uses
/// the theoretical value and the measured residual is reported beside it.
pub const SOURCE_CONVENTION_FACTOR: f64 = 1.0;
