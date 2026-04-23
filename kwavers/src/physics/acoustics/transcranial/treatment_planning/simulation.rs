//! Fast acoustic intensity and bioheat calculations

use super::planner::TreatmentPlanner;
use super::types::{TargetVolume, TransducerSetup};
use crate::core::error::KwaversResult;
use ndarray::Array3;
use num_complex::Complex;

impl TreatmentPlanner {
    /// Simulate acoustic intensity field from a phased-array transducer using coherent
    /// superposition of spherical waves (Rayleigh-Sommerfeld approximation).
    ///
    /// # Theory
    ///
    /// Each element `n` at position `r_n` radiates a monopole spherical wave with
    /// complex pressure amplitude:
    ///
    /// ```text
    /// p_n(r) = A_n · exp(i φ_n) · exp(i k |r − r_n|) / (4π |r − r_n|)
    /// ```
    ///
    /// where:
    /// - `k = 2π f / c` is the wavenumber [rad/m]
    /// - `A_n` is the element amplitude (unity assumed)
    /// - `φ_n` is the phase delay applied to element `n`
    ///
    /// The total acoustic pressure is the coherent sum over all elements:
    ///
    /// ```text
    /// p(r) = Σ_n p_n(r)
    /// ```
    ///
    /// and the acoustic intensity (time-averaged) is:
    ///
    /// ```text
    /// I(r) = |p(r)|² / (2 ρ c)
    /// ```
    ///
    /// Reference: O'Neil HT (1949), *J Acoust Soc Am* 21(5):516–526;
    /// Daum DR & Hynynen K (1999), *IEEE Trans Biomed Eng* 46(9):1070–1082.
    pub(crate) fn simulate_acoustic_field(
        &self,
        setup: &TransducerSetup,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = self.brain_grid.dimensions();
        let mut acoustic_field = Array3::zeros((nx, ny, nz));

        // Wavenumber [rad/m]: k = 2π f / c_brain
        const C_BRAIN: f64 = 1546.0; // m/s — average brain speed of sound (Fry 1978)
        const RHO_BRAIN: f64 = 1040.0; // kg/m³ — brain tissue density
        let k_wave = 2.0 * std::f64::consts::PI * setup.frequency / C_BRAIN;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let xv = i as f64 * self.brain_grid.dx;
                    let yv = j as f64 * self.brain_grid.dy;
                    let zv = k as f64 * self.brain_grid.dz;

                    // Coherent superposition: p(r) = Σ_n exp(i k r_n) / r_n
                    let mut p_total = Complex::new(0.0_f64, 0.0_f64);
                    for (idx, element) in setup.element_positions.iter().enumerate() {
                        let dx = xv - element[0];
                        let dy = yv - element[1];
                        let dz = zv - element[2];
                        let r = (dx * dx + dy * dy + dz * dz).sqrt();
                        if r < 1e-9 {
                            continue;
                        }
                        // Phase delay applied to element `idx` (aberration correction)
                        let steer_phase = setup.element_phases.get(idx).copied().unwrap_or(0.0);
                        // Spherical wave: exp(i(k r + steer_phase)) / r
                        let phase = k_wave * r + steer_phase;
                        p_total += Complex::new(phase.cos(), phase.sin()) / r;
                    }

                    // I(r) = |p(r)|² / (2 ρ c)  [W/m²]
                    acoustic_field[[i, j, k]] = p_total.norm_sqr() / (2.0 * RHO_BRAIN * C_BRAIN);
                }
            }
        }

        Ok(acoustic_field)
    }

    /// Calculate thermal response using the Pennes bioheat equation in steady state.
    ///
    /// # Theory — Pennes Bioheat Equation (1948)
    ///
    /// The steady-state bioheat equation (no diffusion, spatially uniform):
    ///
    /// ```text
    /// 0 = Q(r) − W_b ρ_b c_b [T(r) − T_a]
    /// ```
    ///
    /// where:
    /// - `Q = 2α I` is the volumetric heat source [W/m³]
    /// - `α` = amplitude absorption coefficient [Np/m]
    /// - `W_b` = blood perfusion rate [m³/(m³·s)] = 1/s
    /// - `ρ_b c_b` = blood heat capacity density [J/(m³·K)]
    ///
    /// Solving:
    /// ```text
    /// ΔT = Q / (W_b · ρ_b · c_b)  [K]  (steady-state, no diffusion)
    /// ```
    ///
    /// Reference: Pennes HH (1948). "Analysis of tissue and arterial blood temperatures
    /// in the resting human forearm." *J Appl Physiol* 1(2):93–122.
    /// Nyborg WL (1988). *Phys Med Biol* 33(7):785–792.
    pub(crate) fn calculate_thermal_response(
        &self,
        acoustic_field: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = acoustic_field.dim();
        let mut temperature_field = Array3::zeros((nx, ny, nz));

        // Brain tissue parameters (Fry & Barger 1978; Duck 1990, §4)
        const F_MHZ: f64 = 1.0; // reference frequency [MHz] — adjust per setup
        const ALPHA_NP_PER_M: f64 = 0.5 * F_MHZ * 100.0 / 8.686; // 0.5 dB/MHz/cm → Np/m
        const W_B: f64 = 0.0064; // blood perfusion: 6.4 mL/(100g·s) = 6.4e-4 m³/(kg·s)
        const RHO_B: f64 = 1060.0; // blood density [kg/m³]
        const C_B: f64 = 3617.0; // blood specific heat [J/(kg·K)] (ICRU 1992)
        const T_BODY: f64 = 37.0; // arterial blood temperature [°C]

        // Product W_b · ρ_b · c_b [W/(m³·K)]
        let perfusion_sink = W_B * RHO_B * C_B;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let intensity = acoustic_field[[i, j, k]]; // [W/m²]

                    // Volumetric heat source: Q = 2α · I [W/m³]
                    let q = 2.0 * ALPHA_NP_PER_M * intensity;

                    // Steady-state Pennes bioheat: ΔT = Q / (W_b ρ_b c_b) [K]
                    let delta_t = q / perfusion_sink;

                    temperature_field[[i, j, k]] = T_BODY + delta_t;
                }
            }
        }

        Ok(temperature_field)
    }

    /// Estimate treatment time
    pub(crate) fn estimate_treatment_time(
        &self,
        _targets: &[TargetVolume],
        acoustic_field: &Array3<f64>,
    ) -> f64 {
        // Estimate based on required thermal dose
        let thermal_dose_target = 240.0; // CEM43
        let max_intensity = acoustic_field.iter().fold(0.0_f64, |a, &b| a.max(b));

        if max_intensity > 0.0 {
            // Simplified: t = thermal_dose / (absorption_rate * intensity)
            thermal_dose_target / (0.5 * max_intensity)
        } else {
            0.0
        }
    }
}
