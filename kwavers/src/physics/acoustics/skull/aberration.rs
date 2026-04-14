//! Aberration correction for transcranial ultrasound.
//!
//! # Theorem: Time-Reversal Reciprocity (Fink 1992)
//!
//! The linear acoustic wave equation is invariant under time reversal `t → −t`.
//! If `p(x,t)` is a solution, then `p*(x,t) = p(x, T−t)` is also a solution.
//! This gives the **time-reversal mirror (TRM)** principle:
//!
//! 1. **Forward pass** — transmit from a point source at the focus; record the
//!    distorted wavefield `p(xᵢ, t)` at each transducer element `i`.
//! 2. **Phase conjugation** — in the frequency domain, conjugate the received
//!    spectrum: `P_back(xᵢ, ω) = P*(xᵢ, ω)`.
//! 3. **Back-propagation** — re-transmit `p*(xᵢ, t)`; reciprocity guarantees
//!    refocusing at the original source location.
//!
//! # Phase-Screen Model (Clement & Hynynen 2002; Aubry 2003)
//!
//! For a continuous-wave (CW) single-frequency source, the skull introduces a
//! spatially varying phase delay. Treating the skull as a thin phase screen at
//! the aperture plane, the phase accumulated along a z-propagating ray at
//! transducer position `(x,y)` is:
//!
//! ```text
//!   φ(x,y) = Σ_z  [k_skull(x,y,z) − k_water] · Δz
//! ```
//!
//! where the sum runs over all skull voxels along the ray and
//! `k = 2πf/c` is the local wavenumber.
//!
//! ## Theorem (phase-screen correction)
//!
//! **Statement.** For a planar transducer array with elements at `{(xᵢ,yᵢ)}`
//! and a skull modelled as a phase screen, the corrected drive phase for
//! element `i` that pre-compensates skull aberration at the focus is:
//!
//! ```text
//!   φ_corr,i = −φ(xᵢ, yᵢ) = −Σ_z [k_skull(xᵢ,yᵢ,z) − k_water] Δz
//! ```
//!
//! **Proof.** Under Born single-scattering the field at the focus is:
//! ```text
//!   p_focus = Σ_i  A_i · exp(i[φ_geom,i + φ_skull,i + φ_drive,i])
//! ```
//! Setting `φ_drive,i = −φ_skull,i` removes the skull phase, leaving pure
//! geometric focusing.  QED.
//!
//! **Corollary.** The phase-screen model is exact in the single-scattering
//! (Born) limit.  Multiple reflections within the diploe introduce a residual
//! error bounded by the skull reflectivity (≈ 20–30 % for cortical bone;
//! Pinton et al. 2012).
//!
//! # Running Phase Integral (Volumetric Map)
//!
//! For volumetric analysis, the phase accumulated from the transducer face
//! (z = 0) to depth plane `z = z_k` is:
//!
//! ```text
//!   Φ(x,y,z) = Σ_{z'=0}^{z}  [k_skull(x,y,z') − k_water] · Δz'
//! ```
//!
//! This running integral `Φ(x,y,z)` is what
//! [`AberrationCorrection::compute_time_reversal_phases`] returns.
//! The transducer-element correction is `−Φ(x,y,z_max)`.
//!
//! # References
//!
//! - Fink M (1992). Time reversal of ultrasonic fields — Part I.
//!   IEEE Trans. Ultrason. Ferroelectr. Freq. Control **39**(5), 555–566.
//! - Clement GT, Hynynen K (2002). A non-invasive method for focusing
//!   ultrasound through the human skull. Phys. Med. Biol. **47**(8), 1219–1236.
//!   DOI: 10.1088/0031-9155/47/8/301
//! - Aubry J-F, Tanter M, Pernot M, Thomas J-L, Fink M (2003).
//!   Experimental demonstration of noninvasive transskull adaptive focusing
//!   based on prior CT scans. J. Acoust. Soc. Am. **113**(1), 84–93.
//!   DOI: 10.1121/1.1529663
//! - Pinton G, Aubry J-F, Bossy E, Muller M, Pernot M, Tanter M (2012).
//!   Attenuation, scattering, and absorption of ultrasound in the skull bone.
//!   Med. Phys. **39**(1), 299–307. DOI: 10.1118/1.3668316
//! - Tanter M, Thomas J-L, Fink M (1998). Focusing and steering through
//!   absorbing and aberrating layers. J. Acoust. Soc. Am. **103**(5), 2403–2410.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::physics::acoustics::skull::HeterogeneousSkull;
use ndarray::{Array1, Array2, Array3};
use std::f64::consts::PI;

/// Water sound speed used when no temperature data are available [m/s].
const C_WATER_DEFAULT: f64 = 1482.0; // water at 22 °C

/// Aberration correction calculator using the CT-based phase-screen model.
///
/// Implements:
/// - **Volumetric phase map** [`compute_time_reversal_phases`] —
///   running k-space integral `Φ(x,y,z)` from z=0 to each depth plane.
/// - **Element correction array** [`compute_element_corrections`] —
///   scalar phase per transducer element `φ_corr,i = −Φ(xᵢ,yᵢ,z_max)`.
/// - **Correction field** [`compute_correction_phases`] —
///   3D array of the negated phase map, ready to apply to drive signals.
///
/// # Reference
///
/// Clement & Hynynen (2002), Aubry et al. (2003).
#[derive(Debug)]
pub struct AberrationCorrection<'a> {
    grid: &'a Grid,
    skull: &'a HeterogeneousSkull,
    /// Water sound speed [m/s] (default: 1482.0 m/s at 22 °C).
    pub c_water: f64,
}

impl<'a> AberrationCorrection<'a> {
    /// Construct an aberration correction calculator.
    ///
    /// `c_water` defaults to 1482 m/s; override with [`with_water_speed`] if
    /// the water temperature is known.
    pub fn new(grid: &'a Grid, skull: &'a HeterogeneousSkull) -> Self {
        Self {
            grid,
            skull,
            c_water: C_WATER_DEFAULT,
        }
    }

    /// Override the reference water sound speed [m/s].
    ///
    /// Use `WaterProperties::sound_speed(T_celsius)` from the constants crate
    /// when operating at a specific temperature.
    #[must_use]
    pub fn with_water_speed(mut self, c_water: f64) -> Self {
        self.c_water = c_water;
        self
    }

    /// Compute the volumetric running phase integral `Φ(x,y,z)`.
    ///
    /// ## Algorithm
    ///
    /// For each voxel `(i, j, k)`, the returned phase is the cumulative
    /// integral from the transducer face (z = 0) to that depth:
    ///
    /// ```text
    ///   Φ(i,j,k) = Σ_{k'=0}^{k}  [k_skull(i,j,k') − k_water] · dz
    /// ```
    ///
    /// where `k_skull(i,j,k') = 2πf / c_skull(i,j,k')` and
    /// `k_water = 2πf / c_water`.
    ///
    /// Voxels where `c_local = c_water` contribute zero to the integral.
    ///
    /// ## Returns
    ///
    /// Phase in radians.  Positive values indicate a phase advance (local
    /// speed slower than water); negative for faster-than-water regions
    /// (cortical bone, where c_bone > c_water).
    pub fn compute_time_reversal_phases(&self, frequency: f64) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let dz = self.grid.dz;
        let k_water = 2.0 * PI * frequency / self.c_water;

        let mut phases = Array3::zeros((nx, ny, nz));

        for i in 0..nx {
            for j in 0..ny {
                let mut running_phase = 0.0_f64;
                for k in 0..nz {
                    let c_local = self.skull.sound_speed[[i, j, k]];
                    if c_local > 0.0 {
                        let k_local = 2.0 * PI * frequency / c_local;
                        running_phase += (k_local - k_water) * dz;
                    }
                    phases[[i, j, k]] = running_phase;
                }
            }
        }

        Ok(phases)
    }

    /// Compute the correction phase field `Φ_corr = −Φ(x,y,z)`.
    ///
    /// This is the negation of [`compute_time_reversal_phases`].  Applying
    /// `Φ_corr` as a drive phase shift pre-compensates skull aberration in
    /// the Born (single-scattering) approximation (Aubry et al. 2003, §II.B).
    pub fn compute_correction_phases(&self, frequency: f64) -> KwaversResult<Array3<f64>> {
        let phases = self.compute_time_reversal_phases(frequency)?;
        Ok(phases.mapv(|phi| -phi))
    }

    /// Compute scalar phase correction for each element of a 2D planar array.
    ///
    /// ## Algorithm
    ///
    /// For element `l` at aperture position `(x_m, y_m)`:
    /// ```text
    ///   φ_corr,l = −Σ_{z=0}^{z_max}  [k_skull(i,j,z) − k_water] · dz
    ///            = −Φ(i, j, Nz−1)
    /// ```
    /// where `(i, j)` is the grid cell nearest to `(x_m, y_m)`.
    ///
    /// ## Arguments
    ///
    /// * `frequency`   — source frequency [Hz]
    /// * `element_x_m` — x-positions of transducer elements [m]
    /// * `element_y_m` — y-positions of transducer elements [m]
    ///
    /// ## Returns
    ///
    /// Phase corrections in radians.  Add to geometric focusing delays for
    /// aberration-corrected drive signals.
    ///
    /// ## Reference
    ///
    /// Clement & Hynynen (2002). Phys. Med. Biol. 47(8), 1219–1236.
    pub fn compute_element_corrections(
        &self,
        frequency: f64,
        element_x_m: &[f64],
        element_y_m: &[f64],
    ) -> KwaversResult<Array1<f64>> {
        assert_eq!(
            element_x_m.len(),
            element_y_m.len(),
            "element_x_m and element_y_m must have the same length"
        );

        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;
        let k_water = 2.0 * PI * frequency / self.c_water;
        let dz = self.grid.dz;
        let n_elem = element_x_m.len();
        let mut corrections = Array1::zeros(n_elem);

        for (elem_idx, (&xm, &ym)) in element_x_m.iter().zip(element_y_m.iter()).enumerate() {
            let i = ((xm / self.grid.dx).round() as isize).clamp(0, nx as isize - 1) as usize;
            let j = ((ym / self.grid.dy).round() as isize).clamp(0, ny as isize - 1) as usize;

            let mut total_phase = 0.0_f64;
            for k in 0..nz {
                let c_local = self.skull.sound_speed[[i, j, k]];
                if c_local > 0.0 {
                    let k_local = 2.0 * PI * frequency / c_local;
                    total_phase += (k_local - k_water) * dz;
                }
            }
            corrections[elem_idx] = -total_phase;
        }

        Ok(corrections)
    }

    /// Compute element phase corrections from a pre-computed phase map.
    ///
    /// Re-uses an existing `phases` map from [`compute_time_reversal_phases`]
    /// to avoid redundant integration.  Correction for element `l` is
    /// `-phases[gi, gj, Nz−1]` where `(gi, gj)` is nearest to element `l`.
    pub fn element_corrections_from_map(
        &self,
        phases: &Array3<f64>,
        element_x_m: &[f64],
        element_y_m: &[f64],
    ) -> Array1<f64> {
        assert_eq!(element_x_m.len(), element_y_m.len());
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;
        let n_elem = element_x_m.len();
        let mut corr = Array1::zeros(n_elem);
        for (idx, (&xm, &ym)) in element_x_m.iter().zip(element_y_m.iter()).enumerate() {
            let i = ((xm / self.grid.dx).round() as isize).clamp(0, nx as isize - 1) as usize;
            let j = ((ym / self.grid.dy).round() as isize).clamp(0, ny as isize - 1) as usize;
            corr[idx] = -phases[[i, j, nz - 1]];
        }
        corr
    }

    /// Compute the 2D phase aberration map at the aperture plane `z = z_max`.
    ///
    /// Returns `Φ(x,y) = Φ(x,y,z_max)` — the quantity directly measured in
    /// hydrophone experiments and compared against CT-predicted corrections in
    /// Aubry et al. (2003), Fig. 5.
    pub fn aperture_phase_map(&self, frequency: f64) -> KwaversResult<Array2<f64>> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;
        let k_water = 2.0 * PI * frequency / self.c_water;
        let dz = self.grid.dz;

        let mut map = Array2::zeros((nx, ny));

        for i in 0..nx {
            for j in 0..ny {
                let mut total = 0.0_f64;
                for k in 0..nz {
                    let c_local = self.skull.sound_speed[[i, j, k]];
                    if c_local > 0.0 {
                        let k_local = 2.0 * PI * frequency / c_local;
                        total += (k_local - k_water) * dz;
                    }
                }
                map[[i, j]] = total;
            }
        }

        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::physics::acoustics::skull::HeterogeneousSkull;
    use ndarray::Array3;

    /// Build a minimal skull: uniform bone slab in the centre z-layers,
    /// water (c = C_WATER_DEFAULT) everywhere else.
    fn make_test_skull(
        nx: usize,
        ny: usize,
        nz: usize,
        bone_start: usize,
        bone_end: usize,
        c_bone: f64,
    ) -> HeterogeneousSkull {
        let mut sound_speed = Array3::from_elem((nx, ny, nz), C_WATER_DEFAULT);
        let density = Array3::from_elem((nx, ny, nz), 998.0_f64);
        let attenuation = Array3::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for k in bone_start..bone_end.min(nz) {
                    sound_speed[[i, j, k]] = c_bone;
                }
            }
        }
        HeterogeneousSkull {
            sound_speed,
            density,
            attenuation,
        }
    }

    /// Phase in a pure-water grid must be zero everywhere.
    ///
    /// When `c_local = c_water` for all voxels, `k_local - k_water = 0`
    /// and the running integral is identically zero.
    #[test]
    fn test_zero_phase_in_water() {
        let grid = Grid::new(8, 8, 16, 1e-3, 1e-3, 1e-3).unwrap();
        let skull = make_test_skull(8, 8, 16, 0, 0, C_WATER_DEFAULT); // no bone
        let ac = AberrationCorrection::new(&grid, &skull);
        let phases = ac.compute_time_reversal_phases(500e3).unwrap();
        for v in phases.iter() {
            assert!(
                v.abs() < 1e-12,
                "Phase in pure water must be zero, got {v:.3e}"
            );
        }
    }

    /// A uniform bone slab must produce the analytically correct total phase.
    ///
    /// ## Analytical result
    ///
    /// With `f = 500 kHz`, `c_bone = 3000 m/s`, `c_water = 1482 m/s`,
    /// `d = n_bone × dz`:
    /// ```text
    ///   Δk = 2πf(1/c_bone − 1/c_water)
    ///   φ  = Δk × d
    /// ```
    /// Since `c_bone > c_water`, `Δk < 0`, so `φ < 0` (phase advance:
    /// bone is faster than water).
    ///
    /// Tolerance: 1e-10 relative error (limited by floating-point arithmetic).
    #[test]
    fn test_uniform_bone_slab_phase() {
        let f = 500e3_f64; // 500 kHz
        let c_bone = 3000.0_f64;
        let dz = 1e-3_f64;
        let n_bone = 4_usize;
        let nz = 16_usize;

        let grid = Grid::new(4, 4, nz, 1e-3, 1e-3, dz).unwrap();
        let skull = make_test_skull(4, 4, nz, 4, 4 + n_bone, c_bone);
        let ac = AberrationCorrection::new(&grid, &skull).with_water_speed(C_WATER_DEFAULT);

        let phases = ac.compute_time_reversal_phases(f).unwrap();

        let k_water = 2.0 * PI * f / C_WATER_DEFAULT;
        let k_bone = 2.0 * PI * f / c_bone;
        let expected = (k_bone - k_water) * (n_bone as f64 * dz);

        let phi_computed = phases[[0, 0, nz - 1]];
        let rel_err = (phi_computed - expected).abs() / expected.abs().max(1e-12);
        assert!(
            rel_err < 1e-10,
            "Phase integral for uniform slab: expected {expected:.6} rad, \
             got {phi_computed:.6} rad, rel_err={rel_err:.3e}"
        );
    }

    /// Correction phases must be the exact negation of the aberration phases.
    ///
    /// `Φ_corr(x,y,z) + Φ_aberr(x,y,z) = 0` everywhere (floating-point exact).
    #[test]
    fn test_correction_is_negation_of_aberration() {
        let grid = Grid::new(6, 6, 12, 1e-3, 1e-3, 1e-3).unwrap();
        let skull = make_test_skull(6, 6, 12, 3, 7, 2800.0);
        let ac = AberrationCorrection::new(&grid, &skull);
        let aberr = ac.compute_time_reversal_phases(1e6).unwrap();
        let corr = ac.compute_correction_phases(1e6).unwrap();
        for (a, c) in aberr.iter().zip(corr.iter()) {
            assert!(
                (a + c).abs() < 1e-14,
                "Φ_aberr + Φ_corr must equal zero: {a:.3e} + {c:.3e} = {:.3e}",
                a + c
            );
        }
    }

    /// Element corrections must equal the negation of the aperture phase map.
    ///
    /// `compute_element_corrections` and `element_corrections_from_map` must
    /// agree to floating-point precision.
    #[test]
    fn test_element_corrections_match_aperture_map() {
        let grid = Grid::new(8, 8, 12, 1e-3, 1e-3, 1e-3).unwrap();
        let skull = make_test_skull(8, 8, 12, 2, 6, 3100.0);
        let ac = AberrationCorrection::new(&grid, &skull);

        let f = 700e3_f64;
        let phases = ac.compute_time_reversal_phases(f).unwrap();
        let aperture = ac.aperture_phase_map(f).unwrap();

        // Elements at integer grid positions along x, fixed y=2 mm
        let x_pos: Vec<f64> = (0..8).map(|ii| ii as f64 * 1e-3).collect();
        let y_pos: Vec<f64> = vec![2e-3; 8];

        let corr = ac.compute_element_corrections(f, &x_pos, &y_pos).unwrap();
        let corr_from_map = ac.element_corrections_from_map(&phases, &x_pos, &y_pos);

        let j_grid = (2e-3_f64 / 1e-3).round() as usize; // = 2
        for k in 0..8 {
            let c = corr[k];
            let cm = corr_from_map[k];
            assert!(
                (c - cm).abs() < 1e-12,
                "Element {k}: direct={c:.6} rad, from_map={cm:.6} rad"
            );
            let expected_corr = -aperture[[k, j_grid]];
            let rel = (c - expected_corr).abs() / expected_corr.abs().max(1e-10);
            assert!(
                rel < 1e-10,
                "Element {k}: correction={c:.6}, aperture-based={expected_corr:.6}, rel={rel:.3e}"
            );
        }
    }

    /// Phase grows monotonically through the bone slab and is flat before and after.
    ///
    /// - Before bone (k < bone_start): `Φ = 0`.
    /// - During bone: `Φ` changes monotonically (linear for uniform bone).
    /// - After bone: `Φ` is constant (no more skull material).
    #[test]
    fn test_phase_monotone_through_bone_then_flat() {
        let grid = Grid::new(4, 4, 20, 1e-3, 1e-3, 1e-3).unwrap();
        let skull = make_test_skull(4, 4, 20, 6, 10, 3000.0); // bone at z=6..10
        let ac = AberrationCorrection::new(&grid, &skull);
        let phases = ac.compute_time_reversal_phases(500e3).unwrap();

        // Before bone: phase must be zero
        for k in 0..6 {
            assert!(
                phases[[0, 0, k]].abs() < 1e-12,
                "Phase before bone must be zero at k={k}, got {:.3e}",
                phases[[0, 0, k]]
            );
        }
        // During bone (c_bone > c_water → k_bone < k_water → δφ < 0 per step)
        for k in 6..9 {
            assert!(
                phases[[0, 0, k + 1]] < phases[[0, 0, k]],
                "Phase must decrease through bone: Φ[{k}]={:.6} → Φ[{}]={:.6}",
                phases[[0, 0, k]],
                k + 1,
                phases[[0, 0, k + 1]]
            );
        }
        // After bone: phase constant
        let phi_after = phases[[0, 0, 10]];
        for k in 10..20 {
            assert!(
                (phases[[0, 0, k]] - phi_after).abs() < 1e-12,
                "Phase after bone must be constant at k={k}: \
                 got {:.6}, expected {phi_after:.6}",
                phases[[0, 0, k]]
            );
        }
    }

    /// For a spatially uniform bone slab the aperture phase map must be
    /// identical at every `(x, y)` position.
    #[test]
    fn test_phase_spatially_uniform_for_uniform_skull() {
        let grid = Grid::new(6, 6, 10, 1e-3, 1e-3, 1e-3).unwrap();
        let skull = make_test_skull(6, 6, 10, 2, 6, 2800.0);
        let ac = AberrationCorrection::new(&grid, &skull);
        let aperture = ac.aperture_phase_map(500e3).unwrap();

        let ref_phi = aperture[[0, 0]];
        for i in 0..6 {
            for j in 0..6 {
                let phi = aperture[[i, j]];
                assert!(
                    (phi - ref_phi).abs() < 1e-12,
                    "Aperture phase at ({i},{j}) = {phi:.6} differs from \
                     reference {ref_phi:.6}"
                );
            }
        }
    }

    /// Phase correction must scale linearly with frequency.
    ///
    /// Since `Δk = 2πf · (1/c_bone − 1/c_water)` is proportional to f,
    /// doubling f must double the total phase.
    #[test]
    fn test_phase_scales_linearly_with_frequency() {
        let grid = Grid::new(4, 4, 12, 1e-3, 1e-3, 1e-3).unwrap();
        let skull = make_test_skull(4, 4, 12, 3, 7, 2800.0);
        let ac = AberrationCorrection::new(&grid, &skull);

        let f1 = 500e3_f64;
        let f2 = 1e6_f64; // 2× frequency

        let p1 = ac.compute_time_reversal_phases(f1).unwrap();
        let p2 = ac.compute_time_reversal_phases(f2).unwrap();

        let phi1 = p1[[0, 0, 11]];
        let phi2 = p2[[0, 0, 11]];

        let ratio = phi2 / phi1;
        assert!(
            (ratio - 2.0).abs() < 1e-10,
            "Phase must scale linearly with frequency: ratio={ratio:.6}, expected 2.0"
        );
    }
}
