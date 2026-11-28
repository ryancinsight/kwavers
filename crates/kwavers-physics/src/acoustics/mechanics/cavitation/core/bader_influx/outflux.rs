use kwavers_core::constants::fundamental::BOLTZMANN;
use std::f64::consts::PI;

/// Radius samples over one compression phase.
#[derive(Debug, Clone, Copy)]
pub struct CompressionSamples<'a> {
    /// Bubble-wall radius at each sample [m].
    pub radius: &'a [f64],
    /// Sample times measured from the cycle start [s], one per radius.
    pub time: &'a [f64],
}

/// Liquid and gas state entering the compression-phase flux.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutfluxMedium {
    /// Gas diffusivity in the liquid `D` [m²/s].
    pub diffusivity: f64,
    /// Far-field dissolved-gas concentration `C∞` [molecules/m³].
    pub far_field_concentration: f64,
    /// Saturation concentration `C_0` at the reference pressure
    /// [molecules/m³].
    pub saturation_concentration: f64,
    /// Gas temperature `T` [K].
    pub temperature: f64,
    /// Polytropic exponent `γ` of the gas.
    pub polytropic_exponent: f64,
    /// Reference (ambient) pressure `P_0` [Pa].
    pub ambient_pressure: f64,
}

/// Net gas transfer [molecules] over a compression-phase radius
/// segment (prototype Build 6 generalisation of the Bader 2019a
/// Eq. (4) thin-boundary-layer flux):
///
/// ```text
/// Δn = ∫ 4π·R(t)²·D·(C∞ − C_s(R(t))) / δ(t) dt,
/// δ(t) = √(π·D·t),   C_s(R) = C_0·P_g(R)/P_0,
/// P_g(R) = P_gM·(R_M/R)^(3γ),  P_gM = n·k_B·T/((4π/3)·R_M³)
/// ```
///
/// Signed: positive = net influx during compression, negative = net
/// outflux. The outflux is driven by `C_s ~ R^(−3γ)` and dominated by
/// the small-`R` state. `time` is measured from the **cycle start**
/// (the boundary layer grows from cycle start in the primary
/// formulation), and the samples should cover the compression phase
/// (`R` falling from `R_M`). This is a generalisation anchored to the
/// published flux form, not itself a published law — it was
/// investigated in the prototype because the published influx-only
/// law cannot close the formation gas budget; its verdict there was
/// that the outflux is real but closes only ~1.33× of the gap.
#[must_use]
pub fn compression_phase_outflux(
    samples: CompressionSamples<'_>,
    medium: &OutfluxMedium,
    gas_content_molecules: f64,
    max_radius: f64,
) -> f64 {
    let CompressionSamples { radius, time } = samples;
    let OutfluxMedium {
        diffusivity,
        far_field_concentration,
        saturation_concentration,
        temperature,
        polytropic_exponent,
        ambient_pressure,
    } = *medium;
    if radius.len() < 2
        || radius.len() != time.len()
        || diffusivity <= 0.0
        || max_radius <= 0.0
        || temperature <= 0.0
        || ambient_pressure <= 0.0
    {
        return 0.0;
    }
    let p_gm =
        gas_content_molecules * BOLTZMANN * temperature / ((4.0 / 3.0) * PI * max_radius.powi(3));
    let integrand: Vec<f64> = radius
        .iter()
        .zip(time.iter())
        .map(|(&r, &t)| {
            if r <= 0.0 || t <= 0.0 {
                return 0.0;
            }
            let p_g = p_gm * (max_radius / r).powf(3.0 * polytropic_exponent);
            let c_s = saturation_concentration * p_g / ambient_pressure;
            let delta = (PI * diffusivity * t).sqrt();
            4.0 * PI * r * r * diffusivity * (far_field_concentration - c_s) / delta
        })
        .collect();
    let mut integral = 0.0;
    for i in 1..integrand.len() {
        integral += 0.5 * (integrand[i - 1] + integrand[i]) * (time[i] - time[i - 1]);
    }
    integral
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative bound on the two-sample flux against its closed form.
    ///
    /// Operation-count bound, each correctly rounded operation adding
    /// at most `ε/2` of relative error and relative errors of products
    /// and quotients adding. The `C_s` path (gas content, `P_gM`,
    /// `C_s`) takes at most 17 operations; with `C∞ = 2·C_0` and
    /// `C_s = C_0` its error enters `C∞ − C_s` with weight
    /// `C_s/(C∞ − C_s) = 1`. The flux and trapezoid chain takes at
    /// most 14 operations in the implementation and 14 in the
    /// reference. The sum is at most `(17 + 14 + 14)·ε/2 = 22.5ε`; the
    /// bound is `32ε`.
    const FLUX_RELATIVE_BOUND: f64 = 32.0 * f64::EPSILON;

    /// At `R = R_M` the polytropic factor is one, so
    /// `C_s = C_0·P_gM/P_0` and, with `P_gM = P_0`, `C_s = C_0`. The
    /// flux integrand `4πR²D(C∞ − C_s)/√(πDt)` is then the only
    /// time dependence and the two-sample integral is its trapezoid.
    #[test]
    fn two_samples_at_the_maximum_radius_integrate_the_boundary_layer_flux() {
        let radius = 2.0e-6_f64;
        let diffusivity = 2.0e-9_f64;
        let temperature = 293.15_f64;
        let ambient_pressure = 101_325.0_f64;
        let saturation = 2.4e25_f64;
        let far_field = 2.0 * saturation;
        let gas_content =
            ambient_pressure * (4.0 / 3.0) * PI * radius.powi(3) / (BOLTZMANN * temperature);
        let times = [1.0e-7_f64, 3.0e-7_f64];
        let radii = [radius, radius];
        let medium = OutfluxMedium {
            diffusivity,
            far_field_concentration: far_field,
            saturation_concentration: saturation,
            temperature,
            polytropic_exponent: 1.4,
            ambient_pressure,
        };
        let got = compression_phase_outflux(
            CompressionSamples {
                radius: &radii,
                time: &times,
            },
            &medium,
            gas_content,
            radius,
        );
        let flux = |t: f64| {
            4.0 * PI * radius * radius * diffusivity * (far_field - saturation)
                / (PI * diffusivity * t).sqrt()
        };
        let expected = 0.5 * (flux(times[0]) + flux(times[1])) * (times[1] - times[0]);
        let relative = ((got - expected) / expected).abs();
        assert!(
            relative <= FLUX_RELATIVE_BOUND,
            "relative difference {relative} exceeds {FLUX_RELATIVE_BOUND}"
        );
    }
}
