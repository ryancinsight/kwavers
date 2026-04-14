//! Diverging Wave Imaging and Synthetic Transmit Aperture (STA)
//!
//! This module implements geometric delay calculations for diverging wave (virtual source)
//! ultrasound imaging, enabling wide-field-of-view imaging with synthetic transmit focusing.
//!
//! # Mathematical Foundation
//!
//! ## Virtual Source Model (Papadacci et al. 2014; Jensen et al. 2006)
//!
//! A diverging wave is produced by creating a virtual point source at depth F **behind**
//! the transducer face (z = −F, F > 0).  This mimics spherical emission from the virtual
//! source, broadening the beam beyond the physical aperture.
//!
//! **Transmit delay** for element i at lateral position xᵢ, imaging point (x, z):
//! ```text
//!   τ_tx(x, z, i) = [sqrt((x − xᵢ)² + (z + F)²) − F] / c        (s)
//! ```
//! The subtraction of F/c normalises t = 0 to the transducer face (z = 0 plane).
//!
//! **Receive delay** for element j at position xⱼ, imaging point (x, z):
//! ```text
//!   τ_rx(x, z, j) = sqrt((x − xⱼ)² + z²) / c                      (s)
//! ```
//!
//! **Total STA delay** for transmit element i, receive element j:
//! ```text
//!   τ_STA(x, z, i, j) = τ_tx(x, z, i) + τ_rx(x, z, j)             (s)
//! ```
//!
//! ## Synthetic Transmit Aperture (STA) Beamforming
//!
//! The coherently compounded image (Jensen et al. 2006, Eq. 2):
//! ```text
//!   I(x, z) = Σᵢ Σⱼ wᵢⱼ · sⱼ[t = τ_STA(x, z, i, j)]
//! ```
//! where sⱼ[t] is the RF signal received by element j and wᵢⱼ are apodization weights.
//!
//! ## F-number Apodization
//!
//! At imaging depth z and transmit virtual source depth F, the effective aperture size
//! seen by a point scatterer is:
//! ```text
//!   D_eff = (z + F) / F_number
//! ```
//! Elements outside |xⱼ − x| > D_eff/2 receive zero weight (rectangular apodization).
//! For Hann-windowed apodization (reduces side lobes):
//! ```text
//!   w(x, xⱼ, z) = 0.5 · [1 + cos(π · |xⱼ − x| / (D_eff/2))]
//!                  if |xⱼ − x| ≤ D_eff/2, else 0
//! ```
//!
//! ## PRF Limit
//!
//! **Theorem** (Tanter & Fink 2014, §II.A):  For unambiguous imaging to depth z_max,
//! the pulse repetition frequency satisfies:
//! ```text
//!   PRF_max = c / (2 · z_max)
//! ```
//! **Proof**: Round-trip travel time to depth z_max is T_rt = 2z_max/c.  The next
//! transmission must not overlap with returning echoes, so T_pulse ≥ T_rt, hence
//! PRF = 1/T_pulse ≤ c/(2z_max). □
//!
//! # Coordinate System
//!
//! - **x**: Lateral position (m), parallel to transducer face
//! - **z**: Axial depth (m), perpendicular to transducer face (positive downward)
//! - Virtual source at (xᵢ, −F) — F > 0 places it above (behind) the face
//! - All delays in seconds; multiply by sampling frequency for sample indices
//!
//! # References
//!
//! - Jensen, J.A., et al. (2006). "Synthetic aperture ultrasound imaging."
//!   *Ultrasonics*, 44, e5–e15. DOI: 10.1016/j.ultras.2006.07.017
//!
//! - Papadacci, C., et al. (2014). "High-contrast ultrafast imaging of the heart."
//!   *IEEE TUFFC*, 61(2), 288–301. DOI: 10.1109/TUFFC.2014.2909
//!
//! - Tanter, M., & Fink, M. (2014). "Ultrafast imaging in biomedical ultrasound."
//!   *IEEE TUFFC*, 61(1), 102–119. DOI: 10.1109/TUFFC.2014.2882
//!
//! - Montaldo, G., et al. (2009). "Coherent plane-wave compounding for very high
//!   frame rate ultrasonography." *IEEE TUFFC*, 56(3), 489–506.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2, Array3};

/// Diverging wave (virtual source) imaging configuration
#[derive(Debug, Clone)]
pub struct DivergingWaveConfig {
    /// Lateral positions of transducer elements (m)
    pub element_positions: Vec<f64>,
    /// Speed of sound in medium (m/s)
    pub sound_speed: f64,
    /// Virtual source depth behind the transducer face (m, positive value)
    ///
    /// A larger F creates a broader diverging wave (wider field of view).
    /// Typical range: 5–20 mm (5–20× element pitch).
    pub virtual_source_depth: f64,
    /// F-number for apodization (default 1.5)
    ///
    /// Higher F-number → narrower apodization → better side-lobe suppression.
    pub f_number: f64,
    /// Sampling frequency (Hz), used for converting delays to sample indices
    pub sampling_frequency: f64,
}

impl Default for DivergingWaveConfig {
    fn default() -> Self {
        // 128-element array with 0.3 mm pitch (cardiac imaging, Papadacci et al. 2014)
        let n_elem = 128usize;
        let pitch = 3.0e-4; // 0.3 mm
        let x_start = -(n_elem as f64 - 1.0) / 2.0 * pitch;
        let element_positions: Vec<f64> = (0..n_elem).map(|i| x_start + i as f64 * pitch).collect();

        Self {
            element_positions,
            sound_speed: 1540.0,         // Soft tissue
            virtual_source_depth: 0.010, // 10 mm behind transducer
            f_number: 1.5,
            sampling_frequency: 40.0e6, // 40 MHz
        }
    }
}

/// Diverging wave processor for synthetic transmit aperture imaging
#[derive(Debug, Clone)]
pub struct DivergingWave {
    pub config: DivergingWaveConfig,
}

impl DivergingWave {
    /// Create a new diverging wave processor with the given configuration.
    pub fn new(config: DivergingWaveConfig) -> Self {
        Self { config }
    }

    /// Create with default cardiac imaging configuration (128 elements, 0.3 mm pitch).
    pub fn cardiac(element_positions: Vec<f64>) -> Self {
        Self::new(DivergingWaveConfig {
            element_positions,
            ..DivergingWaveConfig::default()
        })
    }

    /// Number of transducer elements.
    pub fn n_elements(&self) -> usize {
        self.config.element_positions.len()
    }

    /// Transmit delay from virtual source i to imaging point (x, z).
    ///
    /// ## Algorithm
    ///
    /// Virtual source for element i is at (xᵢ, −F).  The spherical wave reaches
    /// point (x, z) at time:
    /// ```text
    ///   τ_tx(x, z, i) = [sqrt((x − xᵢ)² + (z + F)²) − F] / c
    /// ```
    ///
    /// The subtraction of F/c aligns t = 0 with the transducer face (z = 0), so
    /// τ_tx ≥ 0 for all z ≥ 0 (forward medium).
    ///
    /// # Arguments
    /// * `x`        - Lateral position of imaging point (m)
    /// * `z`        - Axial depth of imaging point (m, must be ≥ 0)
    /// * `elem_idx` - Transmitting element index
    pub fn transmit_delay(&self, x: f64, z: f64, elem_idx: usize) -> KwaversResult<f64> {
        self.check_elem(elem_idx)?;
        let xi = self.config.element_positions[elem_idx];
        let f = self.config.virtual_source_depth;
        let c = self.config.sound_speed;
        let dx = x - xi;
        let dz = z + f;
        let delay = ((dx * dx + dz * dz).sqrt() - f) / c;
        Ok(delay)
    }

    /// Receive delay from imaging point (x, z) to element j.
    ///
    /// ## Algorithm
    ///
    /// Standard time-of-flight from scatterer at (x, z) to element j at (xⱼ, 0):
    /// ```text
    ///   τ_rx(x, z, j) = sqrt((x − xⱼ)² + z²) / c
    /// ```
    ///
    /// # Arguments
    /// * `x`        - Lateral position of imaging point (m)
    /// * `z`        - Axial depth of imaging point (m)
    /// * `elem_idx` - Receiving element index
    pub fn receive_delay(&self, x: f64, z: f64, elem_idx: usize) -> KwaversResult<f64> {
        self.check_elem(elem_idx)?;
        let xj = self.config.element_positions[elem_idx];
        let c = self.config.sound_speed;
        let dx = x - xj;
        let delay = (dx * dx + z * z).sqrt() / c;
        Ok(delay)
    }

    /// Total STA delay for transmit element i, receive element j at imaging point (x, z).
    ///
    /// ```text
    ///   τ_STA(x, z, i, j) = τ_tx(x, z, i) + τ_rx(x, z, j)
    /// ```
    pub fn sta_delay(&self, x: f64, z: f64, tx_idx: usize, rx_idx: usize) -> KwaversResult<f64> {
        Ok(self.transmit_delay(x, z, tx_idx)? + self.receive_delay(x, z, rx_idx)?)
    }

    /// Hann apodization weight for a receive element at this imaging point.
    ///
    /// ## Algorithm
    ///
    /// Effective aperture half-width at depth z:
    /// ```text
    ///   D_half = (z + F) / (2 · F_number)
    /// ```
    /// Hann weight:
    /// ```text
    ///   w = 0.5 · [1 + cos(π · |xⱼ − x| / D_half)]   if |xⱼ − x| ≤ D_half
    ///   w = 0                                            otherwise
    /// ```
    ///
    /// Reference: Synnevåg et al. (2007), *IEEE TUFFC* 54(11):2213–2220, Eq. (1).
    pub fn hann_apodization(&self, x: f64, z: f64, elem_idx: usize) -> f64 {
        let xj = self.config.element_positions[elem_idx];
        let f = self.config.virtual_source_depth;
        let d_half = (z + f) / (2.0 * self.config.f_number);
        let dist = (xj - x).abs();
        if dist > d_half || d_half < 1e-30 {
            0.0
        } else {
            0.5 * (1.0 + (std::f64::consts::PI * dist / d_half).cos())
        }
    }

    /// Maximum unambiguous PRF for imaging to depth `z_max`.
    ///
    /// ## Theorem (Tanter & Fink 2014, §II.A)
    ///
    /// Round-trip travel time to depth z_max is `T_rt = 2·z_max/c`.  The next pulse
    /// must not interfere with returning echoes, so:
    /// ```text
    ///   PRF_max = c / (2 · z_max)     (Hz)
    /// ```
    pub fn max_prf(&self, z_max: f64) -> f64 {
        self.config.sound_speed / (2.0 * z_max)
    }

    /// Compute the full STA delay table: `delays[tx, rx, pixel]` for all
    /// transmit/receive element pairs across an image grid.
    ///
    /// # Arguments
    /// * `x_pixels` - Lateral pixel positions (m)
    /// * `z_pixels` - Axial pixel depths (m)
    ///
    /// # Returns
    /// Array of shape `[n_tx × n_rx × n_pixels]` where `n_pixels = nx × nz`.
    /// Indexed as `delays[[tx, rx, iz * nx + ix]]`.
    ///
    /// # Notes
    ///
    /// For large arrays this is O(N² × N_pixels) in memory.  For real-time systems
    /// consider computing delays on the fly or using the per-point `sta_delay()`.
    pub fn sta_delay_table(
        &self,
        x_pixels: &Array1<f64>,
        z_pixels: &Array1<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let n = self.n_elements();
        if n == 0 {
            return Err(KwaversError::InvalidInput(
                "Element positions not set".to_string(),
            ));
        }
        let nx = x_pixels.len();
        let nz = z_pixels.len();
        let n_pixels = nx * nz;

        let mut table = Array3::zeros((n, n, n_pixels));

        for tx in 0..n {
            let xi = self.config.element_positions[tx];
            let f = self.config.virtual_source_depth;
            let c = self.config.sound_speed;

            for (iz, &z) in z_pixels.iter().enumerate() {
                for (ix, &x) in x_pixels.iter().enumerate() {
                    let pixel = iz * nx + ix;
                    // Transmit delay (same for all rx at this pixel)
                    let dx_tx = x - xi;
                    let dz_tx = z + f;
                    let tau_tx = ((dx_tx * dx_tx + dz_tx * dz_tx).sqrt() - f) / c;

                    for rx in 0..n {
                        let xj = self.config.element_positions[rx];
                        let dx_rx = x - xj;
                        let tau_rx = (dx_rx * dx_rx + z * z).sqrt() / c;
                        table[[tx, rx, pixel]] = tau_tx + tau_rx;
                    }
                }
            }
        }

        Ok(table)
    }

    /// Transmit-only delay surface: `delays[elem, pixel]` for a single transmit event.
    ///
    /// This is used to compute the transmit wavefront shape for a specific element firing.
    ///
    /// # Returns
    /// Array of shape `[n_elements × (nx · nz)]`.
    pub fn transmit_delay_surface(
        &self,
        x_pixels: &Array1<f64>,
        z_pixels: &Array1<f64>,
    ) -> KwaversResult<Array2<f64>> {
        let n = self.n_elements();
        if n == 0 {
            return Err(KwaversError::InvalidInput(
                "Element positions not set".to_string(),
            ));
        }
        let nx = x_pixels.len();
        let nz = z_pixels.len();
        let mut delays = Array2::zeros((n, nx * nz));

        let f = self.config.virtual_source_depth;
        let c = self.config.sound_speed;

        for (tx, &xi) in self.config.element_positions.iter().enumerate() {
            for (iz, &z) in z_pixels.iter().enumerate() {
                for (ix, &x) in x_pixels.iter().enumerate() {
                    let dx = x - xi;
                    let dz = z + f;
                    delays[[tx, iz * nx + ix]] = ((dx * dx + dz * dz).sqrt() - f) / c;
                }
            }
        }

        Ok(delays)
    }

    fn check_elem(&self, idx: usize) -> KwaversResult<()> {
        if idx >= self.n_elements() {
            return Err(KwaversError::InvalidInput(format!(
                "Element index {idx} out of range (n_elements={})",
                self.n_elements()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small N-element array with uniform pitch `pitch` centred at x=0.
    fn uniform_array(n: usize, pitch: f64) -> DivergingWave {
        let x0 = -(n as f64 - 1.0) / 2.0 * pitch;
        let positions: Vec<f64> = (0..n).map(|i| x0 + i as f64 * pitch).collect();
        DivergingWave::new(DivergingWaveConfig {
            element_positions: positions,
            sound_speed: 1540.0,
            virtual_source_depth: 0.010,
            f_number: 1.5,
            sampling_frequency: 40.0e6,
        })
    }

    /// PRF_max = c / (2·z_max).
    ///
    /// For c=1540 m/s, z_max=40 mm: PRF_max = 19 250 Hz.
    #[test]
    fn test_max_prf_formula() {
        let dw = uniform_array(8, 3.0e-4);
        let z_max = 0.040; // 40 mm
        let prf = dw.max_prf(z_max);
        let expected = 1540.0 / (2.0 * 0.040);
        assert!(
            (prf - expected).abs() / expected < 1e-10,
            "PRF_max = {prf:.2} Hz, expected {expected:.2} Hz"
        );
    }

    /// On-axis (x=0) with center element (x_i=0) the transmit delay reduces to:
    ///   τ_tx(0, z, center) = [sqrt(0 + (z+F)²) − F] / c = z / c
    ///
    /// This matches a plane-wave normal-incidence transmit delay at z, confirming
    /// the virtual-source formula collapses correctly on axis.
    #[test]
    fn test_on_axis_tx_delay_equals_z_over_c() {
        let dw = uniform_array(9, 3.0e-4); // 9 elements → center at index 4 (x=0)
        let z = 0.020; // 20 mm
        let center = 4;
        // Center element is at x=0 by construction
        assert!(
            dw.config.element_positions[center].abs() < 1e-12,
            "Center element must be at x=0"
        );
        let tau = dw.transmit_delay(0.0, z, center).unwrap();
        let expected = z / dw.config.sound_speed;
        assert!(
            (tau - expected).abs() < 1e-12,
            "On-axis τ_tx = {tau:.6e} s, expected z/c = {expected:.6e} s"
        );
    }

    /// Receive delay at z=0 from a scatterer directly on the element face equals zero.
    #[test]
    fn test_receive_delay_at_face_is_zero() {
        let dw = uniform_array(8, 3.0e-4);
        // Element 3 at position x₃
        let x3 = dw.config.element_positions[3];
        let tau = dw.receive_delay(x3, 0.0, 3).unwrap();
        assert!(tau.abs() < 1e-12, "τ_rx at face must be 0, got {tau:.4e}");
    }

    /// Transmit delays are non-negative for all imaging depths z ≥ 0.
    ///
    /// This follows from ||(x−xᵢ, z+F)|| ≥ F for any (x, z) with z ≥ 0 and F > 0.
    #[test]
    fn test_transmit_delays_non_negative() {
        let dw = uniform_array(8, 3.0e-4);
        let x_test = [-0.01, 0.0, 0.01];
        let z_test = [0.005, 0.010, 0.030, 0.050];
        for &x in &x_test {
            for &z in &z_test {
                for elem in 0..dw.n_elements() {
                    let tau = dw.transmit_delay(x, z, elem).unwrap();
                    assert!(
                        tau >= -1e-15,
                        "τ_tx negative: x={x:.3e} z={z:.3e} elem={elem} τ={tau:.4e}"
                    );
                }
            }
        }
    }

    /// Lateral symmetry: for symmetric element array and on-axis x=0,
    /// the transmit delay from element i must equal that from the mirror element.
    #[test]
    fn test_lateral_symmetry() {
        let n = 8;
        let dw = uniform_array(n, 3.0e-4);
        let z = 0.020;
        let x = 0.005; // off-axis
        for i in 0..n / 2 {
            let j = n - 1 - i; // mirror element
                               // Symmetric test: transmit from i with query x, vs from j with query -x
            let tau_i = dw.transmit_delay(x, z, i).unwrap();
            let tau_j = dw.transmit_delay(-x, z, j).unwrap();
            assert!(
                (tau_i - tau_j).abs() < 1e-12,
                "Symmetry broken: i={i} j={j} τ_i={tau_i:.6e} τ_j={tau_j:.6e}"
            );
        }
    }

    /// STA delay = transmit + receive.
    #[test]
    fn test_sta_delay_is_sum() {
        let dw = uniform_array(8, 3.0e-4);
        let (x, z, tx, rx) = (0.003, 0.015, 2, 5);
        let tau_tx = dw.transmit_delay(x, z, tx).unwrap();
        let tau_rx = dw.receive_delay(x, z, rx).unwrap();
        let tau_sta = dw.sta_delay(x, z, tx, rx).unwrap();
        assert!(
            (tau_sta - (tau_tx + tau_rx)).abs() < 1e-15,
            "STA delay {tau_sta:.6e} ≠ τ_tx + τ_rx = {:.6e}",
            tau_tx + tau_rx
        );
    }

    /// Hann apodization weight at distance 0 from centre of aperture must equal 1.0.
    #[test]
    fn test_hann_apodization_center_is_one() {
        let dw = uniform_array(8, 3.0e-4);
        let z = 0.020;
        // Find element closest to x=0
        let center_idx = dw
            .config
            .element_positions
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let xc = dw.config.element_positions[center_idx];
        let w = dw.hann_apodization(xc, z, center_idx);
        assert!(
            (w - 1.0).abs() < 1e-12,
            "Hann weight at dist=0 must be 1.0, got {w:.6}"
        );
    }

    /// Elements outside the F-number cone receive zero weight.
    #[test]
    fn test_hann_apodization_out_of_aperture_is_zero() {
        let dw = uniform_array(8, 3.0e-4);
        let z = 0.001; // Very shallow → narrow aperture cone
                       // All elements are outside the D_half = (z+F)/(2*f_num) = (0.001+0.010)/3.0 = 3.67 mm
                       // Element 0 is at x ≈ −1.05 mm, which is within the cone
                       // Use an extreme lateral point that must be outside
        let x_far = 0.020; // 20 mm lateral, far from any element
        for elem in 0..dw.n_elements() {
            let xj = dw.config.element_positions[elem];
            let f = dw.config.virtual_source_depth;
            let d_half = (z + f) / (2.0 * dw.config.f_number);
            if (xj - x_far).abs() > d_half {
                let w = dw.hann_apodization(x_far, z, elem);
                assert!(
                    w == 0.0,
                    "Element {elem} outside aperture must have w=0, got {w}"
                );
            }
        }
    }

    /// transmit_delay_surface shape must be [n_elements × (nx·nz)].
    #[test]
    fn test_transmit_delay_surface_shape() {
        let dw = uniform_array(8, 3.0e-4);
        let x_px = Array1::linspace(-0.005, 0.005, 16);
        let z_px = Array1::linspace(0.005, 0.030, 32);
        let surf = dw.transmit_delay_surface(&x_px, &z_px).unwrap();
        assert_eq!(
            surf.dim(),
            (8, 16 * 32),
            "transmit_delay_surface shape mismatch"
        );
    }

    /// sta_delay_table shape must be [n_tx × n_rx × (nx·nz)].
    #[test]
    fn test_sta_delay_table_shape() {
        let dw = uniform_array(4, 3.0e-4);
        let x_px = Array1::linspace(-0.003, 0.003, 8);
        let z_px = Array1::linspace(0.005, 0.020, 16);
        let table = dw.sta_delay_table(&x_px, &z_px).unwrap();
        assert_eq!(
            table.dim(),
            (4, 4, 8 * 16),
            "sta_delay_table shape mismatch"
        );
    }

    /// Transmit delay is consistent between scalar method and delay table.
    #[test]
    fn test_transmit_delay_surface_matches_scalar() {
        let dw = uniform_array(4, 3.0e-4);
        let x_px = Array1::from_vec(vec![0.0, 0.002]);
        let z_px = Array1::from_vec(vec![0.010, 0.020]);
        let surf = dw.transmit_delay_surface(&x_px, &z_px).unwrap();

        // pixel at (ix=0, iz=1) → index = 1*2+0 = 2
        let tau_table = surf[[2, 2]]; // elem=2, pixel iz=1,ix=0
        let tau_scalar = dw.transmit_delay(x_px[0], z_px[1], 2).unwrap();
        assert!(
            (tau_table - tau_scalar).abs() < 1e-15,
            "Surface/scalar mismatch: table={tau_table:.6e} scalar={tau_scalar:.6e}"
        );
    }

    /// The `_` tilt_angle parameter in PlaneWave::beamforming_delays is a reserved
    /// variable to align the API — verify DivergingWave does NOT use it.
    /// At normal incidence (all center, z=20mm), STA delay must be 2z/c.
    #[test]
    fn test_monostatic_sta_delay_equals_round_trip() {
        let dw = uniform_array(9, 3.0e-4);
        let center = 4; // x=0
        let z = 0.020;
        // Monostatic: same element transmits and receives
        // τ_tx(0, z, 0) = z/c  (on-axis), τ_rx(0, z, 0) = z/c → total = 2z/c
        let tau = dw.sta_delay(0.0, z, center, center).unwrap();
        let expected = 2.0 * z / dw.config.sound_speed;
        assert!(
            (tau - expected).abs() < 1e-12,
            "Monostatic STA delay {tau:.6e} ≠ 2z/c = {expected:.6e}"
        );
    }

    /// Out-of-range element index returns an error.
    #[test]
    fn test_out_of_range_index_errors() {
        let dw = uniform_array(4, 3.0e-4);
        assert!(dw.transmit_delay(0.0, 0.01, 10).is_err());
        assert!(dw.receive_delay(0.0, 0.01, 10).is_err());
        assert!(dw.sta_delay(0.0, 0.01, 0, 10).is_err());
    }
}
