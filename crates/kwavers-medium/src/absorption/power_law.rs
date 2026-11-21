//! Power-law absorption model

use aequitas::systems::si::quantities::{
    Dimensionless, Frequency, Length, ReciprocalLength, Velocity,
};
use kwavers_core::constants::acoustic_parameters::{ABSORPTION_TISSUE, WATER_ABSORPTION_ALPHA_0};
use kwavers_core::constants::numerical::{CM_TO_M, TWO_PI};
use kwavers_core::constants::{DB_TO_NP, REFERENCE_FREQUENCY_HZ};
use kwavers_core::error::{KwaversError, KwaversResult};
use serde::{Deserialize, Serialize};

mod wire;

/// A pressure-amplitude absorption law with a runtime exponent.
/// `α(f) = α_ref · (f/f_ref)^y` raises only a dimensionless ratio, so
/// fractional exponents leave the returned attenuation in reciprocal metres.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(try_from = "wire::PowerLawWire", into = "wire::PowerLawWire")]
pub struct PowerLawAbsorption {
    reference_loss: ReciprocalLength,
    reference_frequency: Frequency,
    exponent: Dimensionless,
    dispersion_correction: bool,
}

impl Default for PowerLawAbsorption {
    fn default() -> Self {
        Self::soft_tissue()
    }
}

impl PowerLawAbsorption {
    /// Construct a law from amplitude loss at the specified frequency.
    ///
    /// # Errors
    /// Rejects nonfinite inputs, negative loss or exponent, and nonpositive
    /// reference frequency.
    pub fn new(
        reference_loss: ReciprocalLength,
        reference_frequency: Frequency,
        exponent: Dimensionless,
        dispersion_correction: bool,
    ) -> KwaversResult<Self> {
        if !reference_loss.as_base().is_finite()
            || reference_loss.into_base() < 0.0
            || !reference_frequency.as_base().is_finite()
            || reference_frequency.into_base() <= 0.0
            || !exponent.as_base().is_finite()
            || exponent.into_base() < 0.0
        {
            return Err(KwaversError::InvalidInput(
                "Invalid reference amplitude loss, reference frequency, or power-law exponent"
                    .into(),
            ));
        }
        Ok(Self {
            reference_loss,
            reference_frequency,
            exponent,
            dispersion_correction,
        })
    }

    /// Water at 20°C with a quadratic frequency dependence.
    #[must_use]
    pub fn water() -> Self {
        Self {
            reference_loss: reference_loss_from_decibels_per_centimetre(WATER_ABSORPTION_ALPHA_0),
            reference_frequency: Frequency::from_base(REFERENCE_FREQUENCY_HZ),
            exponent: Dimensionless::from_base(2.0),
            dispersion_correction: false,
        }
    }

    /// Generic soft tissue with the Duck (1990) reference loss at 1 MHz.
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self {
            reference_loss: reference_loss_from_decibels_per_centimetre(ABSORPTION_TISSUE),
            reference_frequency: Frequency::from_base(REFERENCE_FREQUENCY_HZ),
            exponent: Dimensionless::from_base(1.1),
            dispersion_correction: true,
        }
    }

    /// Evaluate the pressure-amplitude attenuation at a physical frequency.
    ///
    /// # Errors
    /// Rejects negative or nonfinite frequencies and nonrepresentable losses.
    pub fn absorption_at_frequency(&self, frequency: Frequency) -> KwaversResult<ReciprocalLength> {
        if !frequency.as_base().is_finite() || frequency.into_base() < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Frequency must be finite and nonnegative".into(),
            ));
        }
        let loss = evaluate(
            self.reference_loss,
            frequency / self.reference_frequency,
            self.exponent,
        );
        if !loss.as_base().is_finite() {
            return Err(KwaversError::InvalidInput(
                "Power-law attenuation exceeds the scalar range".into(),
            ));
        }
        Ok(loss)
    }

    /// Phase velocity from `c_p = c₀/(1 + c₀ α tan(πy/2)/(2πf))`.
    ///
    /// # Errors
    /// Rejects invalid speed or frequency, a singular odd exponent when
    /// dispersion is enabled, and nonpositive phase velocity.
    pub fn phase_velocity(&self, frequency: Frequency, c0: Velocity) -> KwaversResult<Velocity> {
        if !c0.as_base().is_finite() || c0.into_base() <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Sound speed must be finite and positive".into(),
            ));
        }
        let loss = self.absorption_at_frequency(frequency)?;
        if !self.dispersion_correction || frequency.into_base() == 0.0 || loss.into_base() == 0.0 {
            return Ok(c0);
        }
        let exponent = self.exponent.into_base();
        if exponent % 2.0 == 1.0 {
            return Err(KwaversError::InvalidInput(
                "Odd exponents are singular in the phase-velocity law".into(),
            ));
        }
        let tan_term = (std::f64::consts::FRAC_PI_2 * exponent).tan();
        let correction = c0 * loss / (frequency * TWO_PI);
        let denominator = 1.0 + correction.into_base() * tan_term;
        let velocity = c0 / denominator;
        if !velocity.as_base().is_finite() || velocity.into_base() <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Power-law phase velocity must be finite and positive".into(),
            ));
        }
        Ok(velocity)
    }
}

#[inline]
fn reference_loss_from_decibels_per_centimetre(decibels: f64) -> ReciprocalLength {
    ReciprocalLength::from_base(decibels * (DB_TO_NP / CM_TO_M))
}

#[inline]
fn evaluate(
    reference_loss: ReciprocalLength,
    ratio: Dimensionless,
    exponent: Dimensionless,
) -> ReciprocalLength {
    if reference_loss.into_base() == 0.0 {
        return reference_loss;
    }
    reference_loss * ratio.into_base().powf(exponent.into_base())
}

/// Convert the k-Wave power-law absorption coefficient from
/// `dB/(MHz^y·cm)` to the spectral coefficient `Np/((rad/s)^y·m)`.
///
/// This is the single owner of the rad/s-prefactor conversion
/// (KW-POWER-LAW-INVERSE-FORMS): every fractional-Laplacian path converts
/// through it, never by hand-spelled factors.
///
/// # Theorem
/// For a power-law absorption prefactor `α_dB` defined at `1 MHz`,
/// the Treeby & Cox spectral coefficient `α_SI` satisfies:
///
/// ```text
/// α(f) [dB/cm] = α_dB · (f / 1 MHz)^y
/// α(ω) [Np/m]  = α_SI · ω^y
/// α_SI = α_dB · (ln(10)/20) · 100 · (1 / (2π · 10^6))^y
/// ```
///
/// The last line is the exact `k-Wave` `db2neper` conversion used before the
/// fractional-Laplacian coefficients `τ` and `η` are formed.
#[must_use]
#[inline]
pub fn power_law_db_cm_to_np_omega_m(alpha_db_cm: f64, alpha_power: f64) -> f64 {
    // The `/ CM_TO_M` factor converts dB/cm → dB/m (1/0.01 = 100).
    let reference_loss = reference_loss_from_decibels_per_centimetre(alpha_db_cm);
    // The raw spectral coefficient has runtime-dependent units. Strip the
    // reference angular frequency only at the kernel coefficient boundary.
    let omega_ref = Frequency::from_base(REFERENCE_FREQUENCY_HZ) * TWO_PI;
    reference_loss.into_base() * omega_ref.into_base().recip().powf(alpha_power)
}

/// Amplitude attenuation `α(f)` [Np·m⁻¹] at frequency `f` for the k-Wave
/// power-law prefactor `α_dB` [dB/(MHz^y·cm)] and exponent `y`.
///
/// ```text
///   α(f) = α_dB · (ln10/20) · 100 · (f / 1 MHz)^y
/// ```
///
/// This is the single owner of the at-frequency conversion: the conversion a
/// **time-domain** solver needs. The relaxation-spectrum fit
/// (`absorption::relaxation_fit`) is posed in Np·m⁻¹ at a reference
/// frequency, whereas [`power_law_db_cm_to_np_omega_m`] produces the
/// `Np/((rad/s)^y·m)` form that only the spectral fractional-Laplacian path
/// uses. Keeping the forms beside each other is deliberate — the units are
/// the easiest thing in unit conversion to get silently wrong.
///
/// Returns 0 for a non-positive or non-finite frequency.
#[must_use]
#[inline]
pub fn power_law_db_cm_to_np_m(alpha_db_cm: f64, alpha_power: f64, freq_hz: f64) -> f64 {
    if !freq_hz.is_finite() || freq_hz <= 0.0 {
        return 0.0;
    }
    evaluate(
        reference_loss_from_decibels_per_centimetre(alpha_db_cm),
        Frequency::from_base(freq_hz) / Frequency::from_base(REFERENCE_FREQUENCY_HZ),
        Dimensionless::from_base(alpha_power),
    )
    .into_base()
}

/// k-Wave `dB/(MHz^y·cm)` power-law prefactor that reproduces a given
/// amplitude attenuation `α(f)` [Np/m] at frequency `f` for exponent `y`.
///
/// Single owner of the inverse conversion:
///
/// ```text
///   α_db = α_Np_m · (CM_TO_M / DB_TO_NP) · (f_ref / f)^y
/// ```
///
/// Used to inject a non-power-law excess attenuation (e.g. a resonant
/// bubble-cloud α from Commander–Prosperetti) into a power-law medium's
/// prefactor so it produces that attenuation at the drive frequency. Exact at
/// `f`; off-frequency it then scales as the medium's `f^y`.
///
/// Returns 0 for a non-positive or non-finite frequency.
#[must_use]
#[inline]
pub fn np_m_to_power_law_db_cm(alpha_np_m: f64, freq_hz: f64, alpha_power: f64) -> f64 {
    if !freq_hz.is_finite() || freq_hz <= 0.0 {
        return 0.0;
    }
    alpha_np_m * (CM_TO_M / DB_TO_NP) * (REFERENCE_FREQUENCY_HZ / freq_hz).powf(alpha_power)
}

/// Exact inverse of [`power_law_db_cm_to_np_omega_m`]: the `dB/(MHz^y·cm)`
/// prefactor for a spectral coefficient `Np/((rad/s)^y·m)`.
///
/// ```text
///   α_dB = α_SI · (2π · 10^6)^y · (cm/m) / (ln10/20)
/// ```
#[must_use]
#[inline]
pub fn np_omega_m_to_power_law_db_cm(alpha_si: f64, alpha_power: f64) -> f64 {
    alpha_si * (TWO_PI * REFERENCE_FREQUENCY_HZ).powf(alpha_power) * CM_TO_M / DB_TO_NP
}

/// `dB/(cm·MHz)` prefactor as `Np/(m·MHz)` at the 1 MHz reference (the y = 1
/// linear case).
///
/// `const`-capable owner for `const` initializers, which cannot call the
/// `powf`-based owners: `α [Np/(m·MHz)] = α_dB · (ln10/20) · 100`.
/// Agrees with [`power_law_db_cm_to_np_m`] at `(y = 1, f = 1 MHz)` to 1 ulp
/// (`·DB·100` vs `·(DB/CM)` round the last bit differently); the DB_TO_NP
/// evaluation is canonical per the k-Wave oracle test below.
#[must_use]
#[inline]
pub const fn db_per_cm_mhz_to_np_per_m_mhz(alpha_db_cm_mhz: f64) -> f64 {
    alpha_db_cm_mhz * DB_TO_NP * 100.0
}

/// k-Wave `dB/(MHz^y·cm)` prefactor as an `Np/(m·Hz^y)` prefactor for solvers
/// that build per-frequency masks as `α(f) = prefac · f_Hz^y` (e.g. KZK).
///
/// Single owner of the Hz^y-prefactor conversion:
///
/// ```text
///   prefac = α_dB · (ln10/20) · 100 / (1 MHz)^y
/// ```
#[must_use]
#[inline]
pub fn power_law_db_cm_to_np_m_hz(alpha_db_cm: f64, alpha_power: f64) -> f64 {
    power_law_db_cm_to_np_m(alpha_db_cm, alpha_power, 1.0)
}

/// Power law model implementation
#[derive(Debug)]
pub struct PowerLawModel {
    config: PowerLawAbsorption,
}

impl PowerLawModel {
    /// Create a new power law model
    #[must_use]
    pub fn new(config: PowerLawAbsorption) -> Self {
        Self { config }
    }

    /// Apply the amplitude response to every complex sample in each plane.
    ///
    /// # Errors
    /// Returns an error for a mismatched frequency axis, an invalid distance,
    /// or an invalid evaluated loss. Validation precedes spectrum mutation.
    pub fn apply_frequency_domain(
        &self,
        spectrum: &mut leto::Array3<eunomia::Complex64>,
        frequencies: &[Frequency],
        distance: Length,
    ) -> KwaversResult<()> {
        if spectrum.shape()[0] != frequencies.len() {
            return Err(KwaversError::InvalidInput(
                "Power-law absorption frequency count must match spectrum axis 0".into(),
            ));
        }
        if !distance.as_base().is_finite() || distance.into_base() < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Propagation distance must be finite and nonnegative".into(),
            ));
        }
        let gains = frequencies
            .iter()
            .map(|&frequency| {
                let loss = self.config.absorption_at_frequency(frequency)?;
                Ok((-(loss * distance).into_base()).exp())
            })
            .collect::<KwaversResult<Vec<_>>>()?;
        let [_, ny, nz] = spectrum.shape();
        for (i, attenuation) in gains.into_iter().enumerate() {
            for j in 0..ny {
                for k in 0..nz {
                    spectrum[[i, j, k]] *= attenuation;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
