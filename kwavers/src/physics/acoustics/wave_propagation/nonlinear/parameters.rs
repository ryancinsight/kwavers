use crate::core::constants::acoustic_parameters::{NP_TO_DB, WATER_ABSORPTION_ALPHA_0};
use crate::core::constants::fundamental::{
    C_WATER, DENSITY_TISSUE, DENSITY_WATER, SOUND_SPEED_TISSUE,
};
use crate::core::constants::numerical::{CM_TO_M, MHZ_TO_HZ};
use crate::core::constants::tissue_acoustics::{
    B_OVER_A_BRAIN, B_OVER_A_FAT, B_OVER_A_KIDNEY, B_OVER_A_LIVER, B_OVER_A_SOFT_TISSUE,
    B_OVER_A_WATER, DENSITY_BRAIN, DENSITY_FAT, DENSITY_KIDNEY, DENSITY_LIVER, SOUND_SPEED_BRAIN,
    SOUND_SPEED_FAT, SOUND_SPEED_KIDNEY, SOUND_SPEED_LIVER,
};

/// Parameters defining the nonlinear propagation properties of a medium
#[derive(Debug, Clone, Copy)]
pub struct NonlinearParameters {
    /// Density of the medium (rho_0) [kg/m^3]
    pub density: f64,
    /// Small-signal sound speed (c_0) (m/s)
    pub sound_speed: f64,
    /// Nonlinear parameter B/A (dimensionless)
    pub b_over_a: f64,
    /// Coefficient of nonlinearity (beta = 1 + B/2A)
    pub beta: f64,
    /// Attenuation coefficient at 1 MHz [Np/m/MHz^1.1] or similar
    pub attenuation_coeff: f64,
    /// Frequency dependence of attenuation (exponent y, typically ~1.0 for tissue, 2.0 for water)
    pub attenuation_exponent: f64,
}

impl NonlinearParameters {
    /// Create parameters for typical soft tissue
    #[must_use]
    pub fn soft_tissue() -> Self {
        // B/A = 6.5 — representative median for soft tissues (Duck 1990 Table 4.16, Bjørnø 2002).
        let b_over_a = B_OVER_A_SOFT_TISSUE;
        Self {
            density: DENSITY_TISSUE,
            sound_speed: SOUND_SPEED_TISSUE,
            b_over_a,
            beta: 1.0 + b_over_a / 2.0,
            // 0.5 dB/(cm·MHz) → Np/(m·MHz): / CM_TO_M / NP_TO_DB (SSOT-sourced)
            attenuation_coeff: 0.5 / CM_TO_M / NP_TO_DB,
            attenuation_exponent: 1.1, // Typical for soft tissue
        }
    }

    /// Create parameters for water
    #[must_use]
    pub fn water() -> Self {
        // B/A for water at 20°C = 5.2 (Beyer 1960; Duck 1990 Table 4.16).
        let b_over_a = B_OVER_A_WATER;
        Self {
            density: DENSITY_WATER,
            sound_speed: C_WATER,
            b_over_a,
            beta: 1.0 + b_over_a / 2.0,
            // WATER_ABSORPTION_ALPHA_0 dB/(cm·MHz²) → Np/(m·MHz²) — water classical f² absorption.
            attenuation_coeff: WATER_ABSORPTION_ALPHA_0 / CM_TO_M / NP_TO_DB,
            attenuation_exponent: 2.0,
        }
    }

    /// Construct nonlinear-acoustic parameters for human liver parenchyma.
    ///
    /// ## Sources
    ///
    /// Routes density, sound speed, and B/A through the fundamental-constants
    /// SSOT (`DENSITY_LIVER`, `SOUND_SPEED_LIVER`, `B_OVER_A_LIVER`). β is
    /// derived as `1 + B/(2A)` (Hamilton & Blackstock 1998 Eq. 3.7); the
    /// attenuation power-law `α(f) = α₀ · f^y` uses α₀ = 0.45 dB/(cm·MHz^y),
    /// y = 1.05, the central value from Duck (1990) Table 4.20.
    #[must_use]
    pub fn liver() -> Self {
        Self {
            density: DENSITY_LIVER,
            sound_speed: SOUND_SPEED_LIVER,
            b_over_a: B_OVER_A_LIVER,
            beta: 1.0 + B_OVER_A_LIVER / 2.0,
            attenuation_coeff: 0.45 / CM_TO_M / NP_TO_DB,
            attenuation_exponent: 1.05,
        }
    }

    /// Construct nonlinear-acoustic parameters for human renal cortex.
    ///
    /// ## Sources
    ///
    /// Density / sound speed / B/A routed through the fundamental SSOT
    /// (`DENSITY_KIDNEY`, `SOUND_SPEED_KIDNEY`, `B_OVER_A_KIDNEY`).
    /// Attenuation α₀ = 1.0 dB/(cm·MHz^y), y = 1.0 — Duck (1990) Table 4.20
    /// reports kidney attenuation roughly linear in frequency and 2× that of
    /// liver, reflecting the medullary fibrous content.
    #[must_use]
    pub fn kidney() -> Self {
        Self {
            density: DENSITY_KIDNEY,
            sound_speed: SOUND_SPEED_KIDNEY,
            b_over_a: B_OVER_A_KIDNEY,
            beta: 1.0 + B_OVER_A_KIDNEY / 2.0,
            attenuation_coeff: 1.0 / CM_TO_M / NP_TO_DB,
            attenuation_exponent: 1.0,
        }
    }

    /// Construct nonlinear-acoustic parameters for human brain (mean grey +
    /// white matter).
    ///
    /// ## Sources
    ///
    /// Density / sound speed / B/A routed through the fundamental SSOT
    /// (`DENSITY_BRAIN`, `SOUND_SPEED_BRAIN`, `B_OVER_A_BRAIN`). Attenuation
    /// α₀ = 0.6 dB/(cm·MHz^y), y = 1.3 — Duck (1990) Table 4.20 and
    /// Goldman & Hueter (1956). The super-linear exponent y > 1 reflects
    /// the steep frequency dependence of cerebral attenuation that the
    /// transcranial-FUS literature relies on for skull-aberration models.
    #[must_use]
    pub fn brain() -> Self {
        Self {
            density: DENSITY_BRAIN,
            sound_speed: SOUND_SPEED_BRAIN,
            b_over_a: B_OVER_A_BRAIN,
            beta: 1.0 + B_OVER_A_BRAIN / 2.0,
            attenuation_coeff: 0.6 / CM_TO_M / NP_TO_DB,
            attenuation_exponent: 1.3,
        }
    }

    /// Construct nonlinear-acoustic parameters for human adipose tissue.
    ///
    /// ## Sources
    ///
    /// Density / sound speed / B/A routed through the fundamental SSOT
    /// (`DENSITY_FAT`, `SOUND_SPEED_FAT`, `B_OVER_A_FAT`). B/A = 9.6 makes
    /// fat the most nonlinear soft tissue in the body, relevant for breast
    /// imaging and subcutaneous-fat heating during transcutaneous HIFU.
    /// Attenuation α₀ = 0.6 dB/(cm·MHz^y), y = 1.0 — Duck (1990) Table 4.20.
    #[must_use]
    pub fn fat() -> Self {
        Self {
            density: DENSITY_FAT,
            sound_speed: SOUND_SPEED_FAT,
            b_over_a: B_OVER_A_FAT,
            beta: 1.0 + B_OVER_A_FAT / 2.0,
            attenuation_coeff: 0.6 / CM_TO_M / NP_TO_DB,
            attenuation_exponent: 1.0,
        }
    }

    /// Calculate attenuation at a specific frequency [Np/m]
    #[must_use]
    pub fn attenuation_at_frequency(&self, frequency_hz: f64) -> f64 {
        let f_mhz = frequency_hz / MHZ_TO_HZ;
        self.attenuation_coeff * f_mhz.powf(self.attenuation_exponent)
    }
}

/// Properties for tissue harmonic imaging simulation
#[derive(Debug, Clone)]
pub struct TissueHarmonicProperties {
    /// Fundamental frequency (Hz)
    pub fundamental_frequency: f64,
    /// Peak negative pressure of fundamental (Pa)
    pub fundamental_pressure: f64,
    /// Bandwidth of the transducer (fractional)
    pub fractional_bandwidth: f64,
    /// F-number of the imaging system
    pub f_number: f64,
    /// Focal depth (m)
    pub focal_depth: f64,
}
