//! Diverging wave processor implementing virtual-source delay calculations.
//!
//! ## Virtual Source Model (Papadacci et al. 2014; Jensen et al. 2006)
//!
//! **Transmit delay** for element i at lateral position xᵢ, imaging point (x, z):
//! ```text
//!   τ_tx(x, z, i) = [sqrt((x − xᵢ)² + (z + F)²) − F] / c        (s)
//! ```
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

use super::config::DivergingWaveConfig;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2, Array3};

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
