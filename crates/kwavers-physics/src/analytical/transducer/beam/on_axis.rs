use kwavers_core::constants::numerical::TWO_PI;

// ─── On-axis pressure ─────────────────────────────────────────────────────────

/// On-axis pressure magnitude of a baffled circular piston (O'Neil formula).
///
/// ```text
/// |p(z)| = 2·p₀·|sin(k/2·(√(z²+a²) − z))|
/// ```
///
/// # Arguments
/// * `z_arr` – on-axis distances from piston face [m] (must be > 0)
/// * `radius_m` – piston radius a [m]
/// * `freq_hz` – frequency [Hz]
/// * `p0_pa` – surface pressure amplitude [Pa]
/// * `c` – sound speed [m/s]
///
/// # Reference
/// O'Neil (1949), *J. Acoust. Soc. Am.* 21, 516.
#[must_use]
pub fn circular_piston_onaxis(
    z_arr: &[f64],
    radius_m: f64,
    freq_hz: f64,
    p0_pa: f64,
    c: f64,
) -> Vec<f64> {
    let k = TWO_PI * freq_hz / c;
    z_arr
        .iter()
        .map(|&z| {
            let r = (z * z + radius_m * radius_m).sqrt();
            let arg = k / 2.0 * (r - z);
            2.0 * p0_pa * arg.sin().abs()
        })
        .collect()
}

/// On-axis pressure magnitude of a focused spherical bowl (O'Neil 1949).
///
/// The bowl is a spherical cap of radius of curvature `R = F` (the focal
/// length) and aperture radius `a`. With the vertex at the origin and the
/// centre of curvature at the focus `z = F`, the cap edge sits at axial
/// coordinate `h = F − √(F² − a²)` (the sagitta) and radius `a`. The on-axis
/// pressure is the interference of the vertex ray (length `z`) with the edge
/// ray (length `d = √((z − h)² + a²)`):
///
/// ```text
/// |p(z)| = 2·p₀·|F/(F − z)|·|sin( (k/2)(d − z) )|,   d = √((z − h)² + a²)
/// ```
///
/// The prefactor `F/(F − z)` has a removable singularity at the geometric
/// focus `z = F`, where the L'Hôpital limit gives the focusing gain
/// `|p(F)|/p₀ = k·h = 2π h/λ` (≈ `π a²/(λ F)` for `a ≪ F`). Without this
/// prefactor the formula collapses to the unfocused piston result and exhibits
/// no focal peak.
///
/// # Arguments
/// * `z_arr` – axial positions from the vertex [m] (z > 0, in front of the bowl)
/// * `bowl_radius_m` – bowl aperture radius a [m] (must satisfy a ≤ F)
/// * `focal_length_m` – geometric focal length / radius of curvature F [m]
/// * `freq_hz` – frequency [Hz]
/// * `p0_pa` – source pressure [Pa]
/// * `c` – sound speed [m/s]
///
/// # Reference
/// O'Neil (1949), *J. Acoust. Soc. Am.* 21, 516, eq. (8); Cobbold,
/// *Foundations of Biomedical Ultrasound* (2007), §6.3.
#[must_use]
pub fn focused_bowl_onaxis(
    z_arr: &[f64],
    bowl_radius_m: f64,
    focal_length_m: f64,
    freq_hz: f64,
    p0_pa: f64,
    c: f64,
) -> Vec<f64> {
    let k = TWO_PI * freq_hz / c;
    let a = bowl_radius_m;
    let f = focal_length_m;
    // Sagitta of the cap; `max(0)` guards a ≥ F (cap ≥ hemisphere).
    let b = (f * f - a * a).max(0.0).sqrt();
    let h = f - b;
    let focal_gain = k * h; // |p(focus)|/p₀
    z_arr
        .iter()
        .map(|&z| {
            let denom = f - z;
            let d = ((z - h) * (z - h) + a * a).sqrt();
            let ratio = if denom.abs() < f * 1e-9 {
                focal_gain
            } else {
                2.0 * (f / denom).abs() * ((k / 2.0) * (d - z)).sin().abs()
            };
            p0_pa * ratio
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::numerical::TWO_PI;

    const C: f64 = 1500.0;
    const FREQ: f64 = 1.0e6;
    const A: f64 = 0.010; // 10 mm aperture radius
    const F: f64 = 0.064; // 64 mm focal length / radius of curvature

    #[test]
    fn focused_bowl_focal_gain_matches_oneil() {
        let k = TWO_PI * FREQ / C;
        let lambda = C / FREQ;
        let h = F - (F * F - A * A).sqrt(); // sagitta
        let expected_gain = k * h; // |p(F)|/p0 (O'Neil L'Hôpital limit)

        // Evaluated exactly at the focus, the prefactor singularity is replaced
        // by the closed-form focal value.
        let p = focused_bowl_onaxis(&[F], A, F, FREQ, 1.0, C);
        assert!(
            (p[0] - expected_gain).abs() < 1e-9,
            "focal value {} != k·h {expected_gain}",
            p[0]
        );

        // Paraxial focusing gain π a²/(λ F) holds for a ≪ F (agreement < 1%).
        let paraxial = std::f64::consts::PI * A * A / (lambda * F);
        assert!(
            (expected_gain - paraxial).abs() / paraxial < 0.01,
            "k·h {expected_gain} vs paraxial {paraxial}"
        );
        // Genuine focusing gain (> 1), not the unfocused ≤ 2 p0 piston envelope.
        assert!(p[0] > 3.0 && p[0] < 3.5, "gain {} out of range", p[0]);
    }

    #[test]
    fn focused_bowl_reduces_to_flat_piston_as_focus_diverges() {
        // As F → ∞ the sagitta h → 0, the prefactor F/(F−z) → 1, and the bowl
        // formula must reproduce the baffled-piston on-axis field exactly.
        let z: Vec<f64> = (1..50).map(|i| 0.002 * i as f64).collect();
        let huge_f = 1.0e6; // effectively unfocused
        let bowl = focused_bowl_onaxis(&z, A, huge_f, FREQ, 1.0, C);
        let piston = circular_piston_onaxis(&z, A, FREQ, 1.0, C);
        for (b, p) in bowl.iter().zip(&piston) {
            assert!((b - p).abs() < 1e-6, "bowl {b} != piston {p} at F→∞");
        }
    }

    #[test]
    fn focused_bowl_peak_at_focus_strong_focus() {
        // For a tightly focused bowl (low f-number, here F/2a = 1.67, gain ≈ 9.6),
        // the geometric focus is the global on-axis maximum.
        let (a, f): (f64, f64) = (0.015, 0.050);
        let k = TWO_PI * FREQ / C;
        let expected_gain = k * (f - (f * f - a * a).sqrt());
        let p_focus = focused_bowl_onaxis(&[f], a, f, FREQ, 1.0, C)[0];
        assert!(
            (p_focus - expected_gain).abs() < 1e-9,
            "focal value {p_focus} != k·h {expected_gain}"
        );

        let n = 2000usize;
        let (z0, z1) = (0.2 * f, 1.6 * f);
        let z: Vec<f64> = (0..n)
            .map(|i| z0 + (z1 - z0) * i as f64 / (n - 1) as f64)
            .collect();
        let p = focused_bowl_onaxis(&z, a, f, FREQ, 1.0, C);
        let (imax, &pmax) = p
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).expect("invariant: finite pressures"))
            .expect("invariant: non-empty profile");
        // The on-axis pressure maximum sits in the focal region but is shifted
        // toward the transducer (the documented pre-focal shift; O'Neil 1949,
        // Lucas & Muir 1982): z_max ≤ F and within 15% of F.
        assert!(
            z[imax] <= f + 1e-9 && (z[imax] - f).abs() / f < 0.15,
            "peak at z={} not in the pre-focal region near F={f}",
            z[imax]
        );
        // Focal peak exceeds both the pre-focal near field and the post-focal far field.
        let p_near = focused_bowl_onaxis(&[0.3 * f], a, f, FREQ, 1.0, C)[0];
        let p_far = focused_bowl_onaxis(&[1.5 * f], a, f, FREQ, 1.0, C)[0];
        assert!(pmax > p_near && pmax > p_far);
    }

    #[test]
    fn circular_piston_last_axial_maximum() {
        // The last (farthest) on-axis maximum of a baffled piston is at
        // z₀ = (a² − (λ/2)²)/λ, where the path difference equals λ/2 and
        // |p| = 2 p0 (full constructive interference).
        let lambda = C / FREQ;
        let z0 = (A * A - (lambda / 2.0).powi(2)) / lambda;
        let p = circular_piston_onaxis(&[z0], A, FREQ, 1.0, C);
        assert!((p[0] - 2.0).abs() < 1e-6, "last-max |p| {} != 2 p0", p[0]);

        // Beyond z₀ the field decays monotonically toward the far field.
        let beyond = circular_piston_onaxis(&[2.0 * z0, 4.0 * z0], A, FREQ, 1.0, C);
        assert!(beyond[0] < 2.0 && beyond[1] < beyond[0]);
    }
}
