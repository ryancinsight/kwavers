//! Acoustic forward-projection model for pulse-echo SIRT reconstruction.
//!
//! Defines the transducer array geometry and computes the system matrix
//! entries A[s, v] that encode how strongly sensor *s* "sees" voxel *v*:
//!
//! ```text
//! A[s, v] = w(r) · exp(−2 · α_np · f_c · r)
//!
//! w(r) = 1 / r    (spherical spreading; 1/r² for intensity, 1/r for pressure)
//! r    = ‖x_sensor_s − x_voxel_v‖₂
//! α_np = attenuation coefficient [Np/(m·Hz)]
//! f_c  = centre frequency (Hz)
//! ```
//!
//! The factor **2** accounts for the round-trip (transmit + receive) path.
//!
//! ## Derivation
//!
//! For a plane-wave transmit followed by coherent receive (delay-and-sum
//! beamforming), the received echo amplitude from a point scatterer at
//! position **x_v** and sensor **s** at **x_s** is:
//!
//! ```text
//! e_s(τ = 2r/c) = σ_v · exp(−α·f·2r) / r
//! ```
//!
//! where σ_v is the voxel reflectivity.  Stacking e_s over all sensors gives
//! the linear system **b = A·σ** which SIRT inverts.
//!
//! ## References
//!
//! - Jensen JA (1991). "A model for the propagation and scattering of
//!   ultrasound in tissue." *J Acoust Soc Am* 89(1):182–190.
//! - Dines KA, Kak AC (1979). "Ultrasonic attenuation tomography of soft
//!   tissues." *Ultrasonic Imaging* 1(1):16–33.

use kwavers_core::constants::fundamental::ACOUSTIC_ABSORPTION_TISSUE;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use ndarray::{Array1, Array3};

/// Acoustic forward-projection geometry for pulse-echo ultrasound SIRT.
#[derive(Debug, Clone)]
pub struct AcousticProjectionGeometry {
    /// Lateral (x) positions of each sensor element (m).
    pub element_x: Vec<f64>,
    /// Axial (z) position of the sensor array plane (m).  Typically 0.
    pub element_z: f64,
    /// Speed of sound in the medium (m/s).
    pub sound_speed: f64,
    /// Tissue attenuation coefficient [dB/(cm·MHz)].
    /// Soft tissue typical: 0.5.  Water: ≈ 0.002.
    pub attenuation_db_cm_mhz: f64,
    /// Transducer centre frequency (Hz).
    pub center_frequency_hz: f64,
    /// Physical voxel spacing (dx, dy, dz) (m).
    pub voxel_spacing: (f64, f64, f64),
}

impl Default for AcousticProjectionGeometry {
    /// 128-element linear array at 0.3 mm pitch, centred at x = 0,
    /// 5 MHz centre frequency, soft-tissue attenuation, 0.3 mm isotropic grid.
    fn default() -> Self {
        Self {
            element_x: (0..128).map(|i| (i as f64 - 63.5) * 3e-4).collect(),
            element_z: 0.0,
            sound_speed: kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE,
            attenuation_db_cm_mhz: ACOUSTIC_ABSORPTION_TISSUE, // 0.5 dB/(cm·MHz) — Duck (1990)
            center_frequency_hz: 5.0 * MHZ_TO_HZ,
            voxel_spacing: (3e-4, 3e-4, 3e-4),
        }
    }
}

impl AcousticProjectionGeometry {
    /// Attenuation coefficient in Nepers per metre per Hz.
    ///
    /// Conversion from the conventional clinical unit dB/(cm·MHz):
    /// ```text
    /// α_np [Np/(m·Hz)] = α_dB [dB/(cm·MHz)]
    ///                    × ln(10)/20    [dB → Np]
    ///                    × 100          [1/cm → 1/m]
    ///                    × 1e-6         [1/MHz → 1/Hz]
    ///                  = α_dB × 1.15129e-4
    /// ```
    #[inline]
    #[must_use]
    pub fn alpha_nepers_per_m_per_hz(&self) -> f64 {
        self.attenuation_db_cm_mhz * (10.0_f64.ln() / 20.0) * 100.0 * 1e-6
    }
}

/// Compute forward projection **A · x** for a pulse-echo transducer array.
///
/// # Algorithm
///
/// For each sensor *s* and each voxel at grid index (i, j, k):
/// ```text
/// r_sv       = ‖x_s − x_v‖₂,   x_v = (i·dx, j·dy, k·dz)
/// A[s,v]     = exp(−2·α·f_c·r_sv) / r_sv
/// [A·x]_s   = Σ_{i,j,k} A[s,(i,j,k)] · x[i,j,k]
/// ```
///
/// # Complexity
///
/// O(N_sensors × N_voxels): 128 × 64³ ≈ 33 M multiply-adds per call.
pub(crate) fn project_acoustic(
    image: &Array3<f64>,
    geom: &AcousticProjectionGeometry,
) -> Array1<f64> {
    let (nx, ny, nz) = image.dim();
    let (dx, dy, dz) = geom.voxel_spacing;
    let alpha = geom.alpha_nepers_per_m_per_hz();
    let f_c = geom.center_frequency_hz;
    let zs = geom.element_z;

    let mut proj = Array1::zeros(geom.element_x.len());
    for (s, &xs) in geom.element_x.iter().enumerate() {
        let mut sum = 0.0_f64;
        for i in 0..nx {
            let xv = i as f64 * dx;
            let dx2 = (xv - xs) * (xv - xs);
            for j in 0..ny {
                let yv = j as f64 * dy;
                let dxy2 = yv.mul_add(yv, dx2);
                for k in 0..nz {
                    let zv = k as f64 * dz;
                    let r = (zv - zs).mul_add(zv - zs, dxy2).sqrt().max(1e-6);
                    let weight = (-2.0 * alpha * f_c * r).exp() / r;
                    sum += weight * image[[i, j, k]];
                }
            }
        }
        proj[s] = sum;
    }
    proj
}

/// Compute backprojection **Aᵀ · r** for a pulse-echo transducer array.
///
/// # Algorithm
///
/// The adjoint of `project_acoustic` distributes each sensor residual
/// r(s) back to all voxels weighted by the same A[s,v]:
/// ```text
/// [Aᵀ·r][i,j,k] = Σ_s A[s,(i,j,k)] · r(s)
/// ```
pub(crate) fn backproject_acoustic(
    residual: &Array1<f64>,
    shape: (usize, usize, usize),
    geom: &AcousticProjectionGeometry,
) -> Array3<f64> {
    let (nx, ny, nz) = shape;
    let (dx, dy, dz) = geom.voxel_spacing;
    let alpha = geom.alpha_nepers_per_m_per_hz();
    let f_c = geom.center_frequency_hz;
    let zs = geom.element_z;

    let mut image = Array3::zeros(shape);
    for (s, &xs) in geom.element_x.iter().enumerate() {
        let r_s = residual[s];
        for i in 0..nx {
            let xv = i as f64 * dx;
            let dx2 = (xv - xs) * (xv - xs);
            for j in 0..ny {
                let yv = j as f64 * dy;
                let dxy2 = yv.mul_add(yv, dx2);
                for k in 0..nz {
                    let zv = k as f64 * dz;
                    let r = (zv - zs).mul_add(zv - zs, dxy2).sqrt().max(1e-6);
                    let weight = (-2.0 * alpha * f_c * r).exp() / r;
                    image[[i, j, k]] += weight * r_s;
                }
            }
        }
    }
    image
}
