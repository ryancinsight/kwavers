//! Lesion → acoustic-medium perturbation.
//!
//! The monitored slice can only show a lesion if the lesion *physically changes
//! the medium* the imaging pulse propagates through. This module maps the two
//! therapy mechanisms onto the acoustic sound-speed field that the monitor
//! reconstructs each frame — so the image change is a genuine consequence of the
//! simulated physics, not a painted-on overlay.
//!
//! # Thermal mechanism (CEM43 ablation)
//!
//! Sound speed in soft tissue rises with temperature at
//! `∂c/∂T ≈ 2 m/s/°C` over the sub-coagulation range (Duck 1990, *Physical
//! Properties of Tissue*; the basis of ultrasound/MR thermometry). We apply the
//! verified [`TemperatureCoefficients`] linear law per voxel. Cumulative thermal
//! damage is tracked separately as CEM43 dose (Sapareto & Dewey 1984); the
//! coagulation/ablation boundary is the iso-contour at
//! [`ABLATION_CEM43_THRESHOLD_MIN`].
//!
//! # Cavitation mechanism (mechanical / histotripsy)
//!
//! A residual gas void fraction `β` collapses the local sound speed via the Wood
//! (1930) mixture law — even `β ≈ 10⁻⁴` drops `c` from ~1500 m/s toward a few
//! hundred m/s, the strong impedance change that makes a cavitation cloud
//! hyperechoic on B-mode. We apply the verified
//! [`wood_sound_speed`] per voxel with the base tissue speed as the liquid speed.
//!
//! Both mechanisms feed [`perturb_sound_speed`], which returns the perturbed
//! field for the monitored-slice reconstruction.

use kwavers_physics::acoustics::bubble_dynamics::wood_sound_speed;
use kwavers_physics::thermal::TemperatureCoefficients;
use leto::Array3 as LetoArray3;
use ndarray::{Array3, Zip};

/// CEM43 thermal dose [equivalent minutes at 43 °C] at which soft tissue is
/// taken as coagulated/ablated.
///
/// 240 CEM43 is the widely used thermal-ablation iso-effect threshold for
/// brain/soft tissue (Damianou & Hynynen 1994; McDannold et al. 2010);
/// protein-denaturation onset is ~1 CEM43, full coagulative necrosis ~240.
pub const ABLATION_CEM43_THRESHOLD_MIN: f64 = 240.0;

/// Gas sound speed used in the Wood mixture law [m/s] (air at body temperature).
const C_GAS_DEFAULT: f64 = 343.0;
/// Gas density used in the Wood mixture law [kg/m³] (air at body temperature).
const RHO_GAS_DEFAULT: f64 = 1.2;
/// Liquid density used in the Wood mixture law [kg/m³] (water/soft tissue).
const RHO_LIQUID_DEFAULT: f64 = 1000.0;

/// Therapy mechanism that produces the lesion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TherapyMode {
    /// Thermal ablation: heating shifts sound speed; CEM43 marks necrosis.
    Thermal,
    /// Cavitation/histotripsy: bubble void fraction collapses sound speed.
    Cavitation,
}

/// State driving the medium perturbation for one monitor frame.
///
/// Borrows the per-voxel driving field so callers can reuse their solver buffers
/// without copying.
#[derive(Debug)]
pub enum LesionState<'a> {
    /// Thermal: per-voxel temperature [°C] with the linear sound-speed law.
    Thermal {
        /// Temperature field [°C], same shape as the base sound-speed field.
        temperature_c: &'a LetoArray3<f64>,
        /// Reference (baseline) temperature [°C] at which `base_c` is defined.
        reference_c: f64,
        /// Tissue temperature coefficients (e.g. [`TemperatureCoefficients::soft_tissue`]).
        coeff: TemperatureCoefficients,
    },
    /// Cavitation: per-voxel gas void fraction `β ∈ [0, 1)`.
    Cavitation {
        /// Void-fraction field, same shape as the base sound-speed field.
        void_fraction: &'a Array3<f64>,
    },
}

/// Perturbed sound speed from a temperature field via the linear thermal law
/// `c(T) = c₀ + (∂c/∂T)·(T − T_ref)` (Duck 1990).
///
/// `base_c`, `temperature_c` must share the same shape (same simulation grid).
#[must_use]
pub fn thermal_perturbed_sound_speed(
    base_c: &Array3<f64>,
    temperature_c: &LetoArray3<f64>,
    reference_c: f64,
    coeff: &TemperatureCoefficients,
) -> Array3<f64> {
    let [nx, ny, nz] = temperature_c.shape();
    assert_eq!(
        base_c.dim(),
        (nx, ny, nz),
        "invariant: thermal lesion perturbation requires matching base and temperature shapes"
    );
    let mut out = base_c.clone();
    for ((i, j, k), c) in out.indexed_iter_mut() {
        *c = coeff.sound_speed(*c, temperature_c[[i, j, k]], reference_c);
    }
    out
}

/// Perturbed sound speed from a gas void-fraction field via the Wood (1930)
/// mixture law, using the per-voxel base speed as the liquid sound speed.
///
/// `base_c`, `void_fraction` must share the same shape.
#[must_use]
pub fn cavitation_perturbed_sound_speed(
    base_c: &Array3<f64>,
    void_fraction: &Array3<f64>,
) -> Array3<f64> {
    let mut out = base_c.clone();
    Zip::from(&mut out).and(void_fraction).for_each(|c, &beta| {
        *c = wood_sound_speed(beta, *c, RHO_LIQUID_DEFAULT, C_GAS_DEFAULT, RHO_GAS_DEFAULT);
    });
    out
}

/// Dispatch the lesion → sound-speed perturbation for the active therapy mode.
#[must_use]
pub fn perturb_sound_speed(base_c: &Array3<f64>, state: &LesionState<'_>) -> Array3<f64> {
    match state {
        LesionState::Thermal {
            temperature_c,
            reference_c,
            coeff,
        } => thermal_perturbed_sound_speed(base_c, temperature_c, *reference_c, coeff),
        LesionState::Cavitation { void_fraction } => {
            cavitation_perturbed_sound_speed(base_c, void_fraction)
        }
    }
}

/// Binary lesion mask from a CEM43 thermal-dose field at a coagulation threshold.
///
/// Use [`ABLATION_CEM43_THRESHOLD_MIN`] for the standard ablation iso-effect.
#[must_use]
pub fn lesion_mask(cem43_dose: &LetoArray3<f64>, threshold_min: f64) -> LetoArray3<bool> {
    cem43_dose.mapv(|d| d >= threshold_min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn thermal_shift_matches_linear_law() {
        // soft_tissue ∂c/∂T = 2 m/s/°C; +10 °C over reference → +20 m/s.
        let base = Array3::from_elem((2, 2, 2), 1540.0);
        let temp = LetoArray3::from_elem([2, 2, 2], 47.0);
        let coeff = TemperatureCoefficients::soft_tissue();
        let out = thermal_perturbed_sound_speed(&base, &temp, 37.0, &coeff);
        for &c in out.iter() {
            assert!(
                (c - 1560.0).abs() < 1e-9,
                "expected 1540 + 2·10 = 1560, got {c}"
            );
        }
    }

    #[test]
    fn thermal_no_rise_is_identity() {
        let base = Array3::from_elem((2, 1, 3), 1500.0);
        let temp = LetoArray3::from_elem([2, 1, 3], 37.0);
        let coeff = TemperatureCoefficients::soft_tissue();
        let out = thermal_perturbed_sound_speed(&base, &temp, 37.0, &coeff);
        for &c in out.iter() {
            assert!((c - 1500.0).abs() < 1e-9);
        }
    }

    #[test]
    fn cavitation_zero_void_is_identity() {
        let base = Array3::from_elem((3, 1, 2), 1540.0);
        let beta = Array3::from_elem((3, 1, 2), 0.0);
        let out = cavitation_perturbed_sound_speed(&base, &beta);
        for &c in out.iter() {
            assert!((c - 1540.0).abs() < 1e-9, "β=0 must be identity, got {c}");
        }
    }

    #[test]
    fn cavitation_collapses_sound_speed_monotonically() {
        // Wood law: tiny void fraction collapses c far below the liquid speed,
        // monotonically decreasing in β.
        let base = Array3::from_elem((1, 1, 1), 1540.0);
        let c_at = |b: f64| {
            cavitation_perturbed_sound_speed(&base, &Array3::from_elem((1, 1, 1), b))[[0, 0, 0]]
        };
        let c_small = c_at(1e-4);
        let c_large = c_at(1e-2);
        assert!(c_small < 1540.0, "any β>0 lowers c, got {c_small}");
        assert!(
            c_large < c_small,
            "larger β lowers c further: {c_large} !< {c_small}"
        );
        // β = 10⁻⁴ already drops c below ~1000 m/s (Wood 1930 regime).
        assert!(
            c_small < 1000.0,
            "β=1e-4 should collapse c below 1000, got {c_small}"
        );
    }

    #[test]
    fn lesion_mask_thresholds_dose() {
        let mut dose = LetoArray3::from_elem([2, 1, 2], 0.0);
        dose[[0, 0, 0]] = ABLATION_CEM43_THRESHOLD_MIN + 1.0;
        dose[[1, 0, 1]] = ABLATION_CEM43_THRESHOLD_MIN - 1.0;
        let mask = lesion_mask(&dose, ABLATION_CEM43_THRESHOLD_MIN);
        assert!(mask[[0, 0, 0]], "above threshold → lesion");
        assert!(!mask[[1, 0, 1]], "below threshold → not lesion");
        assert_eq!(mask.iter().filter(|&&b| b).count(), 1);
    }

    #[test]
    fn dispatch_matches_direct_calls() {
        let base = Array3::from_elem((2, 2, 2), 1530.0);
        let temp = LetoArray3::from_elem([2, 2, 2], 50.0);
        let coeff = TemperatureCoefficients::soft_tissue();
        let via_dispatch = perturb_sound_speed(
            &base,
            &LesionState::Thermal {
                temperature_c: &temp,
                reference_c: 37.0,
                coeff,
            },
        );
        let direct = thermal_perturbed_sound_speed(&base, &temp, 37.0, &coeff);
        assert_eq!(via_dispatch, direct);
    }
}
