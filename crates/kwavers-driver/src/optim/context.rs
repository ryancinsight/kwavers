//! The input context types for a co-design point: the transducer array geometry, the device thermal
//! environment, the PDN decoupling bank, and the EMI test setup. Pure data with sensible defaults;
//! the physics lives in [`super::evaluate`] and [`super::kernels`].

use crate::physics::acoustic::wavelength_m;

/// Geometry of the phased-array transducer connected to the driver tile.
#[derive(Debug, Clone, Copy)]
pub struct ArrayGeometry {
    /// Number of elements (channels).
    pub n_elements: usize,
    /// Element pitch (m).
    pub pitch_m: f64,
    /// Focal depth (m).
    pub focal_m: f64,
    /// Steering angle (degrees).
    pub steer_deg: f64,
    /// Speed of sound in the coupling medium (m/s).
    pub speed_m_s: f64,
    /// Acoustic operating frequency (Hz).
    pub freq_hz: f64,
    /// Acoustic wavelength (m) — must equal `speed_m_s / freq_hz`.
    pub lambda_m: f64,
}

impl ArrayGeometry {
    /// Construct from fundamental parameters; derives `lambda_m` automatically.
    #[must_use]
    pub fn new(
        n_elements: usize,
        pitch_m: f64,
        focal_m: f64,
        steer_deg: f64,
        speed_m_s: f64,
        freq_hz: f64,
    ) -> Self {
        Self {
            n_elements,
            pitch_m,
            focal_m,
            steer_deg,
            speed_m_s,
            freq_hz,
            lambda_m: wavelength_m(speed_m_s, freq_hz),
        }
    }
}

/// Thermal context for the device under analysis.
#[derive(Debug, Clone, Copy)]
pub struct ThermalContext {
    /// Ambient temperature (K).
    pub t_ambient_k: f64,
    /// Board temperature rise at the device footprint (K), from [`crate::physics::thermal::solve_board`].
    pub board_rise_k: f64,
    /// Junction-to-case thermal resistance (K/W).
    pub theta_jc_k_per_w: f64,
    /// Temperature coefficient of Rds_on (K⁻¹); typical Si LDMOS ≈ 6.0e-3.
    pub alpha_rds_per_k: f64,
    /// Maximum allowed junction temperature (K).
    pub t_j_max_k: f64,
}

impl Default for ThermalContext {
    fn default() -> Self {
        Self {
            t_ambient_k: 298.0, // 25 °C
            board_rise_k: 0.0,
            theta_jc_k_per_w: 40.0, // typical HV-class pulser SOIC/QFN
            alpha_rds_per_k: 6.0e-3,
            t_j_max_k: 423.0, // 150 °C
        }
    }
}

/// PDN configuration: the decoupling capacitor bank on VPP near the device.
#[derive(Debug, Clone)]
pub struct PdnConfig {
    /// Capacitor bank: `(C_f, ESR_ohm, ESL_h)` per capacitor.
    pub caps: Vec<(f64, f64, f64)>,
    /// Maximum tolerable ripple voltage on VPP (V).
    pub v_ripple: f64,
    /// Peak transient current demand from the device (A).
    pub i_transient_a: f64,
    /// Commutation-loop area (mm²), from [`crate::physics::emi::CommutationLoop`].
    pub loop_area_mm2: f64,
}

/// EMI analysis context.
#[derive(Debug, Clone, Copy)]
pub struct EmiContext {
    /// Test distance (m) for radiated emission check (CISPR 22: 3 m or 10 m).
    pub test_distance_m: f64,
    /// Regulatory limit (dBµV/m) at the operating frequency (CISPR 22 class B: 30 dBµV/m at 30 MHz).
    pub limit_dbuv_m: f64,
}

impl Default for EmiContext {
    fn default() -> Self {
        Self {
            test_distance_m: 3.0,
            limit_dbuv_m: 40.0, // generous margin; 2 MHz is well below CISPR Class B band
        }
    }
}
