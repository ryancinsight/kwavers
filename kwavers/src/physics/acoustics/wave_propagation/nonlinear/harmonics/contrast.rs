/// Contrast agent second-harmonic response [Pa].
///
/// ## Model
/// Near resonance a microbubble acts as a weakly nonlinear oscillator whose
/// second-harmonic emission scales as P² with a resonance enhancement factor:
///
/// ```text
/// R₂(f) = K × P² / sqrt((1 − Ω²)² + δ²Ω²)
/// ```
///
/// where Ω = f/f_r, δ = 0.1, and K = 1×10⁻⁶ Pa⁻¹.
///
/// ## References
/// - Church CC (1995). J. Acoust. Soc. Am. 97(3), 1510–1521.
/// - Goldberg BB et al. (1994). Radiology 190, 7–12.
#[must_use]
pub fn contrast_harmonic_response(pressure: f64, frequency: f64, bubble_resonance: f64) -> f64 {
    let omega_ratio = frequency / bubble_resonance;
    const DAMPING: f64 = 0.1;
    const BUBBLE_NONLINEARITY_SCALE: f64 = 1e-6;

    let resonance_enhancement =
        1.0 / ((1.0 - omega_ratio.powi(2)).powi(2) + DAMPING.powi(2) * omega_ratio.powi(2)).sqrt();
    BUBBLE_NONLINEARITY_SCALE * pressure.powi(2) * resonance_enhancement
}
