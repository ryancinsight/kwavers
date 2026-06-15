use super::super::{KWaveArray, KwaveApodizationWindow};
use kwavers_math::signal::ApodizationType;

impl KWaveArray {
    /// Calculate focus delays `(s)` for each element to a target point
    /// (`delay = distance / c`).
    ///
    /// # Arguments
    /// * `focus_point` - Focus position `(x, y, z)` in metres
    #[must_use]
    pub fn get_focus_delays(&self, focus_point: (f64, f64, f64)) -> Vec<f64> {
        let c = self.sound_speed;
        self.get_element_positions()
            .iter()
            .map(|(ex, ey, ez)| {
                let dist = (ez - focus_point.2)
                    .mul_add(
                        ez - focus_point.2,
                        (ey - focus_point.1)
                            .mul_add(ey - focus_point.1, (ex - focus_point.0).powi(2)),
                    )
                    .sqrt();
                dist / c
            })
            .collect()
    }

    /// Calculate per-element time delays `(s)` for electronic focusing.
    ///
    /// # Algorithm — Time-Delay Focusing (Selfridge et al. 1980)
    ///
    /// `τᵢ = (d_max − dᵢ) / c` where `dᵢ = ‖pᵢ − f‖`. The element farthest
    /// from the focus fires first (zero delay); all others are delayed so
    /// wavefronts add coherently at the focus.
    ///
    /// All returned values are ≥ 0, and `min(τ) = 0`.
    ///
    /// Reference: Selfridge, A.R., Kino, G.S. & Khuri-Yakub, B.T. (1980).
    /// "A theory for the radiation pattern of a narrow-strip acoustic
    /// transducer." Appl. Phys. Lett. 37(1):35–36.
    #[must_use]
    pub fn get_element_delays(&self, focus_point: (f64, f64, f64)) -> Vec<f64> {
        let c = self.sound_speed;
        let positions = self.get_element_positions();
        let distances: Vec<f64> = positions
            .iter()
            .map(|(ex, ey, ez)| {
                (ez - focus_point.2)
                    .mul_add(
                        ez - focus_point.2,
                        (ey - focus_point.1)
                            .mul_add(ey - focus_point.1, (ex - focus_point.0).powi(2)),
                    )
                    .sqrt()
            })
            .collect();
        let d_max = distances.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        distances.iter().map(|&d| (d_max - d) / c).collect()
    }

    /// Compute per-element amplitude weights for beam apodization.
    ///
    /// # Algorithm — Discrete Window Functions (Harris 1978)
    ///
    /// For N elements indexed `i = 0, …, N−1`:
    /// - `Rectangular`: `wᵢ = 1.0`
    /// - `Hann`:        `wᵢ = 0.5·(1 − cos(2π·i/(N−1)))` if N > 1, else 1.0
    /// - `Hamming`:     `wᵢ = 0.54 − 0.46·cos(2π·i/(N−1))` if N > 1, else 1.0
    /// - `Blackman`:    `wᵢ = 0.42 − 0.5·cos(2π·i/(N−1)) + 0.08·cos(4π·i/(N−1))`
    /// - `Tukey(r)`:    tapered cosine, fraction `r ∈ [0, 1]` (rect at 0, Hann at 1)
    ///
    /// Weights delegate to the canonical window SSOT
    /// ([`kwavers_math::signal::ApodizationType`]) so the formulas live in one
    /// place; returns an empty vector when the array has no elements.
    ///
    /// Reference: Harris, F.J. (1978). Proc. IEEE 66(1):51–83.
    #[must_use]
    pub fn get_apodization(&self, window: KwaveApodizationWindow) -> Vec<f64> {
        let n = self.elements.len();
        if n == 0 {
            return Vec::new();
        }
        let apodization = match window {
            KwaveApodizationWindow::Rectangular => ApodizationType::Uniform,
            KwaveApodizationWindow::Hann => ApodizationType::Hanning,
            KwaveApodizationWindow::Hamming => ApodizationType::Hamming,
            KwaveApodizationWindow::Blackman => ApodizationType::Blackman,
            KwaveApodizationWindow::Tukey(r) => ApodizationType::Tukey { r },
        };
        apodization.weights(n)
    }
}
