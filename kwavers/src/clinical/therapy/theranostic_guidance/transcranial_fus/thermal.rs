//! Heterogeneous Pennes bioheat solver for transcranial FUS therapy planning.
//!
//! # Physical model
//!
//! The Pennes bioheat equation with volumetric acoustic heating:
//!
//!   ρ·cₚ·∂T/∂t = k·∇²T − ωb·ρb·cb·(T − T_a) + Q
//!
//! where:
//! - ρ, cₚ, k : tissue density \[kg/m³\], specific heat \[J/(kg·K)\], thermal
//!   conductivity \[W/(m·K)\].
//! - ωb, ρb, cb : blood perfusion rate \[1/s\], density \[kg/m³\], specific
//!   heat \[J/(kg·K)\].
//! - T_a : arterial blood temperature (baseline) \[°C\].
//! - Q = 2·α·I : volumetric heating from acoustic absorption \[W/m³\].
//! - α : power-law acoustic absorption coefficient \[Np/m\].
//! - I : time-averaged acoustic intensity \[W/m²\].
//!
//! Rearranged for explicit Euler per-voxel update:
//!
//!   ∂T/∂t = κ·∇²T − Ω·(T − T_a) + H
//!
//! where κ = k/(ρ·cₚ) \[m²/s\], Ω = ωb·ρb·cb/(ρ·cₚ) \[1/s\],
//! H = 2α·I/(ρ·cₚ) \[K/s\].
//!
//! # Material properties
//!
//! Hasgall et al. (2022) IT'IS Foundation database v4.1:
//! - Skull cortical bone : ρ = 1908 kg/m³, cₚ = 1313 J/(kg·K), k = 0.32 W/(m·K).
//! - Brain white matter  : ρ = 1040 kg/m³, cₚ = 3650 J/(kg·K), k = 0.51 W/(m·K),
//!                         ωb = 0.009 /s.
//! - Water (background)  : ρ = 998 kg/m³, cₚ = 4182 J/(kg·K), k = 0.598 W/(m·K).
//! - Blood               : ρb = 1060 kg/m³, cb = 3840 J/(kg·K).
//!
//! # Absorption model
//!
//! Duck (1990) "Physical Properties of Tissue", Academic Press:
//! - Skull : 15   dB·cm⁻¹·MHz⁻¹.
//! - Brain : 3.5  dB·cm⁻¹·MHz⁻¹.
//! - Water : 0.002 dB·cm⁻¹·MHz⁻¹.
//!
//! Conversion: α\_Np\_m = α\_dB\_cm\_MHz · f\_MHz · 100/8.686.
//!
//! # CEM43 thermal dose
//!
//! Sapareto & Dewey (1984) cumulative equivalent minutes at 43 °C:
//!   CEM43 = ∫₀ᵀ R^(43 − T(t)) dt   \[min\]
//!   R = 0.5 if T ≥ 43 °C,  R = 0.25 if T < 43 °C.
//! Thermal lesion threshold: CEM43 ≥ 240 min (Dewhirst et al. 2003).
//!
//! # Numerical stability
//!
//! Explicit Euler is stable when dt ≤ h²/(6·κ_max).
//! For brain: κ = 0.51/(1040·3650) ≈ 1.34×10⁻⁷ m²/s.
//! At h = 2.5 mm: dt_max ≈ 7.8 s — a safety factor of ~31× over the
//! default dt = 0.25 s.
//!
//! # References
//! - Pennes, H. H. (1948). J. Appl. Physiol. 1(2), 93–122.
//! - Sapareto, S. A. & Dewey, W. C. (1984). Int. J. Radiat. Oncol. Biol.
//!   Phys. 10(6), 787–800.
//! - Dewhirst, M. W. et al. (2003). Int. J. Hyperthermia 19(3), 267–294.
//! - Hasgall et al. (2022). IT'IS database v4.1. doi:10.13099/VIP21000-04-1.
//! - Duck, F. A. (1990). Physical Properties of Tissue. Academic Press.

use ndarray::{Array3, Zip};

use crate::core::constants::fundamental::DENSITY_WATER;
use crate::core::constants::medical::{
    THERMAL_DOSE_R_ABOVE_43C, THERMAL_DOSE_R_BELOW_43C, THERMAL_DOSE_REFERENCE_TEMP_C,
    THERMAL_DOSE_THRESHOLD,
};
use crate::core::constants::thermodynamic::{SPECIFIC_HEAT_WATER, THERMAL_CONDUCTIVITY_WATER};
use crate::core::constants::tissue_acoustics::{DENSITY_BLOOD, DENSITY_BRAIN};
use crate::core::constants::tissue_thermal::{SPECIFIC_HEAT_BLOOD_PLASMA, SPECIFIC_HEAT_BRAIN_WHITE};

// ── Material constants (IT'IS v4.1 / ICRU-44 / Duck 1990) ────────────────────

const SKULL_RHO: f64 = 1908.0; // kg/m³
const SKULL_CP: f64 = 1313.0; // J/(kg·K)
const SKULL_K: f64 = 0.32; // W/(m·K)
const SKULL_PERF: f64 = 0.0; // 1/s (cortical bone: negligible)
const SKULL_ALPHA_DB_CM_MHZ: f64 = 15.0;

// SSOT: DENSITY_BRAIN = 1040.0 kg/m³ (tissue_acoustics::DENSITY_BRAIN, Duck 1990 Table 4.1)
const BRAIN_K: f64 = 0.51; // W/(m·K) — IT'IS v4.1 (differs from Duck 1990 canonical 0.50)
const BRAIN_PERF: f64 = 0.009; // 1/s — IT'IS v4.1 (differs from canonical BLOOD_PERFUSION_RATE_BRAIN=0.0064)
const BRAIN_ALPHA_DB_CM_MHZ: f64 = 3.5;

// SSOT: DENSITY_WATER=998.2, SPECIFIC_HEAT_WATER=4182.0, THERMAL_CONDUCTIVITY_WATER=0.598
const WATER_PERF: f64 = 0.0; // 1/s
const WATER_ALPHA_DB_CM_MHZ: f64 = 0.002;

// SSOT: DENSITY_BLOOD=1060.0 kg/m³ (ICRU-44/Duck 1990), SPECIFIC_HEAT_BLOOD_PLASMA=3840.0 J/(kg·K)

/// 1 Np/m = 8.686 dB/m; 1 dB/cm = 100 dB/m → α_Np_m = α_dB_cm·100/8.686.
const DB_CM_TO_NP_M: f64 = 100.0 / 8.686;

/// CEM43 lesion threshold [min] — Dewhirst et al. (2003).
const CEM43_LESION_THRESHOLD: f64 = THERMAL_DOSE_THRESHOLD;

// ── Output type ──────────────────────────────────────────────────────────────

/// Outputs of the heterogeneous transcranial Pennes bioheat simulation.
#[derive(Debug)]
pub struct TranscranialThermalResult {
    /// Peak temperature during sonication \[°C\], shape (nx, ny, nz).
    pub peak_temperature_c: Array3<f32>,
    /// Temperature at end of sonication \[°C\], shape (nx, ny, nz).
    pub final_temperature_c: Array3<f32>,
    /// Cumulative CEM43 thermal dose \[min\], shape (nx, ny, nz).
    pub cem43_min: Array3<f32>,
    /// Thermal lesion: CEM43 ≥ 240 min within brain (non-skull) tissue.
    pub lesion_mask: Array3<bool>,
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Simulate heterogeneous Pennes bioheat for transcranial FUS ablation.
///
/// All physics — Laplacian, perfusion, absorption heating, CEM43 integration —
/// execute in Rust. No Python-side computation is required.
///
/// # Parameters
/// - `intensity_w_m2`: steady-state acoustic intensity \[W/m²\].
/// - `skull_mask`    : voxels classified as skull cortical bone.
/// - `brain_mask`    : voxels classified as brain parenchyma.
/// - `spacing_m`     : voxel edge lengths \[m\] (dx, dy, dz).
/// - `frequency_hz`  : operating frequency; determines α-to-heat conversion.
/// - `sonication_s`  : total sonication duration \[s\].
/// - `dt_s`          : explicit Euler time step \[s\].
/// - `baseline_c`    : initial and arterial blood temperature \[°C\].
///
/// # Returns
/// [`TranscranialThermalResult`] with peak/final temperature, CEM43, and lesion mask.
pub fn transcranial_pennes_thermal_dose(
    intensity_w_m2: &Array3<f32>,
    skull_mask: &Array3<bool>,
    brain_mask: &Array3<bool>,
    spacing_m: [f64; 3],
    frequency_hz: f64,
    sonication_s: f64,
    dt_s: f64,
    baseline_c: f64,
) -> TranscranialThermalResult {
    let (nx, ny, nz) = intensity_w_m2.dim();
    let freq_mhz = frequency_hz * 1.0e-6;

    // Per-tissue absorption [Np/m] at operating frequency.
    let alpha_skull = SKULL_ALPHA_DB_CM_MHZ * freq_mhz * DB_CM_TO_NP_M;
    let alpha_brain = BRAIN_ALPHA_DB_CM_MHZ * freq_mhz * DB_CM_TO_NP_M;
    let alpha_water = WATER_ALPHA_DB_CM_MHZ * freq_mhz * DB_CM_TO_NP_M;

    // Pre-compute per-voxel derived coefficients in one pass.
    // kappa    = k/(ρ·cₚ)            [m²/s]   thermal diffusivity
    // perf_c   = ωb·ρb·cb/(ρ·cₚ)    [1/s]    effective perfusion decay
    // heat_rcp = 2α·I/(ρ·cₚ)        [K/s]    acoustic heating rate
    let mut kappa = Array3::<f64>::zeros((nx, ny, nz));
    let mut perf_c = Array3::<f64>::zeros((nx, ny, nz));
    let mut heat_rcp = Array3::<f64>::zeros((nx, ny, nz));

    Zip::from(&mut kappa)
        .and(&mut perf_c)
        .and(&mut heat_rcp)
        .and(skull_mask)
        .and(brain_mask)
        .and(intensity_w_m2)
        .for_each(|kap, pc, hr, &is_skull, &is_brain, &i_val| {
            let (rho, cp, k, perf, alpha) = if is_skull {
                (SKULL_RHO, SKULL_CP, SKULL_K, SKULL_PERF, alpha_skull)
            } else if is_brain {
                (DENSITY_BRAIN, SPECIFIC_HEAT_BRAIN_WHITE, BRAIN_K, BRAIN_PERF, alpha_brain)
            } else {
                (DENSITY_WATER, SPECIFIC_HEAT_WATER, THERMAL_CONDUCTIVITY_WATER, WATER_PERF, alpha_water)
            };
            let rho_cp = rho * cp;
            *kap = k / rho_cp;
            *pc = perf * DENSITY_BLOOD * SPECIFIC_HEAT_BLOOD_PLASMA / rho_cp;
            *hr = 2.0 * alpha * f64::from(i_val) / rho_cp;
        });

    // Explicit Euler time loop.
    let mut temp = Array3::<f64>::from_elem((nx, ny, nz), baseline_c);
    let mut peak = Array3::<f64>::from_elem((nx, ny, nz), baseline_c);
    let mut cem43 = Array3::<f64>::zeros((nx, ny, nz));
    let steps = ((sonication_s / dt_s).ceil() as usize).max(1);
    let [dx, dy, dz] = spacing_m;
    let dx2 = dx * dx;
    let dy2 = dy * dy;
    let dz2 = dz * dz;

    for _ in 0..steps {
        let lap = laplacian_neumann_3d(&temp, dx2, dy2, dz2);

        // dT/dt = kappa*∇²T - perf_c*(T-T_a) + heat_rcp
        let mut new_temp = Array3::<f64>::zeros((nx, ny, nz));
        Zip::from(&mut new_temp)
            .and(&temp)
            .and(&lap)
            .and(&kappa)
            .and(&perf_c)
            .and(&heat_rcp)
            .for_each(|nt, &t, &l, &kap, &pc, &hr| {
                *nt = t + dt_s * (kap.mul_add(l, hr) - pc * (t - baseline_c));
            });
        temp = new_temp;

        // Update peak temperature and accumulate CEM43.
        Zip::from(&mut peak)
            .and(&mut cem43)
            .and(&temp)
            .for_each(|p, c, &t| {
                if t > *p {
                    *p = t;
                }
                // Sapareto & Dewey (1984) CEM43 integrand, explicit Euler in time [min].
                let r: f64 = if t >= THERMAL_DOSE_REFERENCE_TEMP_C {
                    THERMAL_DOSE_R_ABOVE_43C
                } else {
                    THERMAL_DOSE_R_BELOW_43C
                };
                *c += (dt_s / 60.0) * r.powf(THERMAL_DOSE_REFERENCE_TEMP_C - t);
            });
    }

    // Lesion mask: CEM43 >= 240 min AND in brain AND not in skull.
    let mut lesion_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    Zip::from(&mut lesion_mask)
        .and(&cem43)
        .and(brain_mask)
        .and(skull_mask)
        .for_each(|b, &c, &is_brain, &is_skull| {
            *b = c >= CEM43_LESION_THRESHOLD && is_brain && !is_skull;
        });

    TranscranialThermalResult {
        peak_temperature_c: peak.mapv(|v| v as f32),
        final_temperature_c: temp.mapv(|v| v as f32),
        cem43_min: cem43.mapv(|v| v as f32),
        lesion_mask,
    }
}

// ── Numerical helper ─────────────────────────────────────────────────────────

/// Second-order central finite-difference Laplacian with zero-gradient Neumann BCs.
///
/// For interior voxel (i,j,k):
///   ∇²T ≈ (T[i+1]+T[i-1]-2T)/dx² + (T[j+1]+T[j-1]-2T)/dy² + (T[k+1]+T[k-1]-2T)/dz²
///
/// Boundary rule: ghost cell equals the nearest boundary value (∂T/∂n = 0).
/// At i=0: T[-1] = T[0]; at i=nx-1: T[nx] = T[nx-1].
fn laplacian_neumann_3d(arr: &Array3<f64>, dx2: f64, dy2: f64, dz2: f64) -> Array3<f64> {
    let (nx, ny, nz) = arr.dim();
    let mut lap = Array3::<f64>::zeros((nx, ny, nz));
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let center = arr[[ix, iy, iz]];
                let xp = if ix + 1 < nx { arr[[ix + 1, iy, iz]] } else { center };
                let xm = if ix > 0 { arr[[ix - 1, iy, iz]] } else { center };
                let yp = if iy + 1 < ny { arr[[ix, iy + 1, iz]] } else { center };
                let ym = if iy > 0 { arr[[ix, iy - 1, iz]] } else { center };
                let zp = if iz + 1 < nz { arr[[ix, iy, iz + 1]] } else { center };
                let zm = if iz > 0 { arr[[ix, iy, iz - 1]] } else { center };
                lap[[ix, iy, iz]] = (xp + xm - 2.0 * center) / dx2
                    + (yp + ym - 2.0 * center) / dy2
                    + (zp + zm - 2.0 * center) / dz2;
            }
        }
    }
    lap
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniform temperature → Laplacian = 0 everywhere.
    #[test]
    fn laplacian_uniform_field_is_zero() {
        let arr = Array3::from_elem((8, 8, 8), 37.0_f64);
        let lap = laplacian_neumann_3d(&arr, 1e-3_f64.powi(2), 1e-3_f64.powi(2), 1e-3_f64.powi(2));
        for &v in lap.iter() {
            assert!(v.abs() < 1e-10, "expected zero Laplacian, got {v}");
        }
    }

    /// Baseline-only run (no intensity) should keep temperature flat.
    #[test]
    fn pennes_no_heating_stays_at_baseline() {
        let shape = (8, 8, 8);
        let intensity = Array3::<f32>::zeros(shape);
        let skull = Array3::<bool>::from_elem(shape, false);
        let brain = Array3::<bool>::from_elem(shape, true);
        let result = transcranial_pennes_thermal_dose(
            &intensity,
            &skull,
            &brain,
            [1e-3, 1e-3, 1e-3],
            650_000.0,
            1.0,
            0.25,
            37.0,
        );
        for &t in result.peak_temperature_c.iter() {
            assert!(
                (t as f64 - 37.0).abs() < 0.01,
                "expected ~37 °C, got {t}"
            );
        }
        assert!(result.lesion_mask.iter().all(|&b| !b));
    }

    /// High-intensity skull voxel should heat up faster than brain voxel.
    #[test]
    fn skull_heats_faster_than_brain_under_equal_intensity() {
        let shape = (4, 4, 4);
        let intensity = Array3::<f32>::from_elem(shape, 1e5_f32);
        let mut skull = Array3::<bool>::from_elem(shape, false);
        let mut brain = Array3::<bool>::from_elem(shape, false);
        // Single skull voxel at [0,0,0], single brain voxel at [3,3,3].
        skull[[0, 0, 0]] = true;
        brain[[3, 3, 3]] = true;
        let result = transcranial_pennes_thermal_dose(
            &intensity,
            &skull,
            &brain,
            [1e-3, 1e-3, 1e-3],
            650_000.0,
            1.0,
            0.25,
            37.0,
        );
        let skull_peak = result.peak_temperature_c[[0, 0, 0]] as f64;
        let brain_peak = result.peak_temperature_c[[3, 3, 3]] as f64;
        // Skull α=15 dB/cm/MHz >> brain α=3.5 dB/cm/MHz → skull heats faster.
        assert!(
            skull_peak > brain_peak,
            "skull peak {skull_peak:.2} should exceed brain peak {brain_peak:.2}"
        );
        // Both should rise above baseline.
        assert!(skull_peak > 37.0, "skull peak {skull_peak:.2} should exceed baseline");
        assert!(brain_peak > 37.0, "brain peak {brain_peak:.2} should exceed baseline");
    }
}
