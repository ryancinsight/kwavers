//! Construction of `FractionalLaplacianAbsorption` from per-voxel material
//! fields. Computes `dt · τ = dt · 2 α₀_ω · c^(y+1)` from the canonical
//! Treeby-Cox 2010 power-law absorption coefficients and builds the
//! corresponding `|k|^y` spectral-filter weights.

use crate::core::constants::fundamental::SOUND_SPEED_AIR;
use rayon::prelude::*;

use super::spectrum::build_k_power_spectrum;
use super::{AbsorptionBuilder, FractionalLaplacianAbsorption};

impl FractionalLaplacianAbsorption {
    /// Build an absorption operator for one Westervelt forward simulation.
    ///
    /// Returns `None` when the requested attenuation field is identically
    /// zero (the loss-free baseline). Skipping construction in that case
    /// preserves zero-cost behaviour: the forward and adjoint paths take
    /// the `Option::None` short-circuit and pay no FFT cost.
    ///
    /// # Inputs
    ///
    /// - `n`: cube side length (full grid is `n × n × n`).
    /// - `spacing_m`: isotropic grid spacing.
    /// - `dt_s`: FDTD timestep (already CFL-checked by the forward).
    /// - `speed_m_s`: per-voxel sound speed (length `n³`).
    /// - `attenuation_np_per_m_mhz`: per-voxel `α₀` at 1 MHz in Np/m.
    /// - `attenuation_power_law_y`: per-voxel `y` exponent; the global
    ///   spectral exponent is `representative_y(...)`.
    pub(crate) fn maybe_new(input: AbsorptionBuilder<'_>) -> Option<Self> {
        let cells = input.n * input.n * input.n;
        debug_assert_eq!(input.speed_m_s.len(), cells);
        debug_assert_eq!(input.attenuation_np_per_m_mhz.len(), cells);
        debug_assert_eq!(input.attenuation_power_law_y.len(), cells);

        let max_alpha = input
            .attenuation_np_per_m_mhz
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        if max_alpha <= 0.0 || !max_alpha.is_finite() {
            return None;
        }
        Some(Self::new(input))
    }

    fn new(input: AbsorptionBuilder<'_>) -> Self {
        let n = input.n;
        let cells = n * n * n;
        let y_exponent = representative_y(input.attenuation_power_law_y);
        let omega_ref = std::f64::consts::TAU * 1.0e6; // 1 MHz reference

        // Per-voxel τ and η for the wave-equation form of Treeby-Cox
        // 2010 §III.B (J. Biomed. Opt. 15(2) 021314, Eq. 11):
        //
        //     (1/c²·∂²/∂t² − ∇²) p = τ̃·∂L_y(p)/∂t + η̃·L_{y+1}(p)
        //     τ̃ = −2 α₀ c^(y−1)         (Eq. 9)
        //     η̃ = −2 α₀ c^y tan(π y / 2) (Eq. 10)
        //
        // Multiplying by c² yields the form integrated by the FDTD update:
        //
        //     ∂²p/∂t² = c²∇²p + c²·τ̃·∂L_y/∂t + c²·η̃·L_{y+1}(p)
        //
        // Discretising ∂L_y/∂t with a backward Euler step on the absorbing
        // pair (consistent with the leapfrog Westervelt stencil) and
        // multiplying through by dt² gives the FDTD coefficients
        //
        //     dt_tau  = dt · 2 α₀ c^(y+1)
        //     dt2_eta = dt² · 2 α₀ c^(y+2) tan(π y / 2)
        //
        // applied as
        //     next += −dt_tau · (L_y(p[n]) − L_y(p[n−1]))
        //             − dt2_eta · L_{y+1}(p[n])
        //
        // The PSTD `α` convention uses an angular-frequency basis
        // (`α(ω) = α₀ · ω^y`); the `α(f) = α₀_f · f_MHz^y` convention
        // in `attenuation_np_per_m_mhz` corresponds to
        // `α₀_omega = α₀_f / ω_ref^y`. We use `α₀_omega` here so that the
        // analytical plane-wave decay `α(ω) = α₀_omega · ω^y` evaluates
        // exactly to `α₀_f · f_MHz^y` Np/m at the test frequency.
        let dt_tau: Vec<f64> = (0..cells)
            .into_par_iter()
            .map(|i| {
                let alpha0_f = input.attenuation_np_per_m_mhz[i];
                let alpha0_omega = alpha0_f / omega_ref.powf(y_exponent);
                let c = input.speed_m_s[i].max(SOUND_SPEED_AIR);
                let tau = 2.0 * alpha0_omega * c.powf(y_exponent + 1.0);
                input.dt_s * tau
            })
            .collect();

        // η Kramers-Kronig dispersion is dropped — see module docs for the
        // von-Neumann stability argument. For y = 2 this is exact (η ≡ 0);
        // for y < 2 it is the documented Stokes-Kirchhoff approximation
        // that preserves the correct absorption magnitude.

        let k_pow_y = build_k_power_spectrum(n, input.spacing_m, y_exponent);

        Self {
            n,
            dt_tau,
            k_pow_y,
            prev_l_y: None,
        }
    }
}

/// Volume-area-weighted median power-law exponent across body voxels.
/// Skull (`y = 2`), soft tissue (`y = 1.05`), and air (`y = 1`) map to
/// distinct tissue classes; the spectral filter uses one global `y`. The
/// median minimizes the in-volume error vs. the per-voxel power law,
/// matching the canonical kwavers PSTD convention.
fn representative_y(power_law_y: &[f64]) -> f64 {
    if power_law_y.is_empty() {
        return 1.05;
    }
    let mut sorted: Vec<f64> = power_law_y.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::representative_y;

    /// Representative-y matches the per-class table: for a volume that is
    /// 80% soft tissue (`y = 1.05`) and 20% skull (`y = 2.0`), the median
    /// returns `y = 1.05` (since more than half the voxels have that value).
    #[test]
    fn representative_y_matches_volume_median() {
        let mut y_field = vec![1.05_f64; 800];
        y_field.extend(vec![2.0_f64; 200]);
        let y = representative_y(&y_field);
        assert!(
            (y - 1.05).abs() < 1.0e-12,
            "median y must be 1.05 for an 80/20 soft-tissue/skull mix; got {y}",
        );
    }
}
