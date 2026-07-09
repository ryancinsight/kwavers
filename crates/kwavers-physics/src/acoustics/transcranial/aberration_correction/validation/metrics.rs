use super::super::phase_correction::TranscranialAberrationCorrection;
use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
use kwavers_math::numerics::operators::interpolation::trilinear_index_space;
use leto::Array3;

impl TranscranialAberrationCorrection {
    /// Calculate focal acoustic intensity at the target using trilinear interpolation.
    ///
    /// ## Theorem: Trilinear Interpolation of Acoustic Intensity
    /// For a pressure-squared field `field[i,j,k] = |p(i,j,k)|²`, the intensity
    /// at a sub-voxel point r = (x, y, z) is obtained by trilinear interpolation:
    /// ```text
    ///   I(r) = TRILINEAR(field, x/Δx, y/Δy, z/Δz) / (2 ρ₀ c₀)
    /// ```
    ///
    /// ## Algorithm
    /// 1. Convert physical coordinates to fractional grid indices (xi, yj, zk).
    /// 2. Clamp to [0, N−2] so the upper corner is always valid.
    /// 3. Perform 3D trilinear interpolation over the enclosing unit cell.
    /// 4. Divide by 2ρ₀c₀ (ρ₀ = 1000 kg/m³, c₀ = reference_speed).
    ///
    /// ## References
    /// - O'Neil (1949). J. Acoust. Soc. Am. 21(5):516–526.
    /// - Press et al. (2007). Numerical Recipes §3.6.
    pub(super) fn calculate_focal_intensity(
        &self,
        field: &Array3<f64>,
        target_point: &[f64; 3],
    ) -> f64 {
        let [nx, ny, nz] = field.shape();
        let xi = (target_point[0] / self.grid.dx).clamp(0.0, nx.saturating_sub(2) as f64);
        let yj = (target_point[1] / self.grid.dy).clamp(0.0, ny.saturating_sub(2) as f64);
        let zk = (target_point[2] / self.grid.dz).clamp(0.0, nz.saturating_sub(2) as f64);

        let p2_interp = trilinear_index_space(field, xi, yj, zk);
        let rho0 = DENSITY_WATER_NOMINAL;
        p2_interp / (2.0 * rho0 * self.reference_speed)
    }

    /// Calculate peak sidelobe level relative to the main lobe (linear ratio).
    ///
    /// ## Theorem: 6 dB Main Lobe Exclusion
    /// The main lobe is defined as the −6 dB region: all cells where
    /// `I ≥ I_peak / 4`. The bounding box of this region is excluded from the
    /// sidelobe search. The returned value is the linear ratio `I_side / I_peak`;
    /// caller converts to dB via `10·log10(ratio)`.
    ///
    /// ## References
    /// - Zhu & Steinberg (1993). IEEE Trans. UFFC 40(6):726–737.
    pub(super) fn calculate_sidelobe_level(
        &self,
        field: &Array3<f64>,
        _target_point: &[f64; 3],
    ) -> f64 {
        let [nx, ny, nz] = field.shape();

        let i_peak = field.iter().copied().fold(0.0_f64, f64::max);
        if i_peak <= 0.0 {
            return 0.0;
        }

        let threshold_6db = i_peak * 0.25;
        let mut lobe = MainLobeBounds::empty(nx, ny, nz);

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if field[[i, j, k]] >= threshold_6db {
                        lobe.include(i, j, k);
                    }
                }
            }
        }

        let mut max_sidelobe = 0.0_f64;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if lobe.excludes(i, j, k) {
                        max_sidelobe = max_sidelobe.max(field[[i, j, k]]);
                    }
                }
            }
        }

        max_sidelobe / i_peak
    }

    /// Calculate the geometric-mean FWHM focal spot size (metres).
    ///
    /// ## Theorem: FWHM Resolution Metric
    /// For each axis α ∈ {x, y, z}, the FWHM of the 1D profile through the peak
    /// voxel is:
    /// ```text
    ///   FWHM_α = (last_half_max_index − first_half_max_index) · Δα   (m)
    /// ```
    /// The scalar focal spot size is the geometric mean of the three FWHM values:
    /// ```text
    ///   FWHM_geom = (FWHM_x · FWHM_y · FWHM_z)^(1/3)   (m)
    /// ```
    ///
    /// ## References
    /// - Treeby & Cox (2010). J. Biomed. Opt. 15(2):021314.
    /// - Goodman (2005). Introduction to Fourier Optics §6.2.
    pub(super) fn calculate_focal_spot_size(
        &self,
        field: &Array3<f64>,
        _target_point: &[f64; 3],
    ) -> f64 {
        let [nx, ny, nz] = field.shape();

        let i_peak = field.iter().copied().fold(0.0_f64, f64::max);
        if i_peak <= 0.0 {
            return 0.0;
        }

        let (pi, pj, pk) = peak_voxel(field, i_peak);
        let half_max = i_peak * 0.5;

        let fwhm_x = fwhm_axis(nx, self.grid.dx, |i| field[[i, pj, pk]], half_max);
        let fwhm_y = fwhm_axis(ny, self.grid.dy, |j| field[[pi, j, pk]], half_max);
        let fwhm_z = fwhm_axis(nz, self.grid.dz, |k| field[[pi, pj, k]], half_max);

        if fwhm_x > 0.0 && fwhm_y > 0.0 && fwhm_z > 0.0 {
            (fwhm_x * fwhm_y * fwhm_z).cbrt()
        } else {
            fwhm_x.max(fwhm_y).max(fwhm_z)
        }
    }
}

#[derive(Clone, Copy)]
struct MainLobeBounds {
    i_min: usize,
    i_max: usize,
    j_min: usize,
    j_max: usize,
    k_min: usize,
    k_max: usize,
}

impl MainLobeBounds {
    fn empty(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            i_min: nx,
            i_max: 0,
            j_min: ny,
            j_max: 0,
            k_min: nz,
            k_max: 0,
        }
    }

    fn include(&mut self, i: usize, j: usize, k: usize) {
        self.i_min = self.i_min.min(i);
        self.i_max = self.i_max.max(i);
        self.j_min = self.j_min.min(j);
        self.j_max = self.j_max.max(j);
        self.k_min = self.k_min.min(k);
        self.k_max = self.k_max.max(k);
    }

    fn excludes(&self, i: usize, j: usize, k: usize) -> bool {
        i < self.i_min
            || i > self.i_max
            || j < self.j_min
            || j > self.j_max
            || k < self.k_min
            || k > self.k_max
    }
}

fn peak_voxel(field: &Array3<f64>, i_peak: f64) -> (usize, usize, usize) {
    let [nx, ny, nz] = field.shape();

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if field[[i, j, k]] >= i_peak * (1.0 - 1e-12) {
                    return (i, j, k);
                }
            }
        }
    }

    (0, 0, 0)
}

fn fwhm_axis<F>(len: usize, delta: f64, sample: F, half_max: f64) -> f64
where
    F: Fn(usize) -> f64,
{
    let mut first = None;
    let mut last = None;

    for idx in 0..len {
        if sample(idx) >= half_max {
            first.get_or_insert(idx);
            last = Some(idx);
        }
    }

    match (first, last) {
        (Some(f), Some(l)) if l >= f => (l - f) as f64 * delta,
        _ => 0.0,
    }
}
