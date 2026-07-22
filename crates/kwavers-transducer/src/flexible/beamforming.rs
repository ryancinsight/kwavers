//! Conformal beamforming for flexible (deformable) transducer arrays.
//!
//! A flexible array's value is that it can **refocus after it bends**: once the
//! tracked element positions move, the focusing law must be recomputed from the
//! deformed geometry. This module provides that geometry-driven beamformer plus
//! the coupling to the [`mems`](crate::mems) cell models that "populate" the
//! array — per-element transmit apodization from the CMUT flex-derating at the
//! locally-measured curvature.
//!
//! All functions take the element positions directly (an `n×3` view), so they
//! apply to any geometry — flat, curved, or arbitrarily deformed.
//!
//! # References
//! - Wang, Y., et al. (2018). "Conformable ultrasound arrays." (geometry-aware
//!   delay-and-sum focusing on deformed apertures).
//! - Khuri-Yakub & Oralkan (2011), CMUT flex-derating (`CmutCell::flex_gap_derating`).

use super::super::mems::CmutCell;
use leto::ArrayView2;

/// Conformal **delay-and-sum focusing** delays \`s` for a (possibly deformed)
/// array focusing at `focus` \`m`.
///
/// `τ_i = (d_max − d_i)/c` with `d_i = |focus − r_i|`. By construction every
/// element's emission arrives at the focus simultaneously:
/// `d_i/c + τ_i = d_max/c` for all `i`, so the farthest element fires first
/// (`τ = 0`) and the nearest is delayed the most. Delays are non-negative.
///
/// Returns an empty vector if `positions` is empty or `c ≤ 0`.
#[must_use]
pub fn focusing_delays(positions: &ArrayView2<f64>, focus: [f64; 3], c: f64) -> Vec<f64> {
    let n = positions.shape()[0];
    if n == 0 || c <= 0.0 {
        return Vec::new();
    }
    let distances: Vec<f64> = (0..n)
        .map(|i| {
            let dx = focus[0] - positions[[i, 0]];
            let dy = focus[1] - positions[[i, 1]];
            let dz = focus[2] - positions[[i, 2]];
            dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt()
        })
        .collect();
    let d_max = distances.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    distances.iter().map(|&d| (d_max - d) / c).collect()
}

/// Far-field **plane-wave steering** delays \`s` toward unit direction `dir`.
///
/// `τ_i = (p_max − r_i·d̂)/c` where `p_i = r_i·d̂` is the projection of element
/// `i` onto the steering direction. Wavefronts then align along `dir`; the
/// most-advanced element (largest projection) fires first (`τ = 0`).
/// `dir` is normalized internally. Returns empty if `positions` is empty,
/// `c ≤ 0`, or `dir` is the zero vector.
#[must_use]
pub fn steering_delays(positions: &ArrayView2<f64>, dir: [f64; 3], c: f64) -> Vec<f64> {
    let n = positions.shape()[0];
    let norm = dir[2]
        .mul_add(dir[2], dir[0].mul_add(dir[0], dir[1] * dir[1]))
        .sqrt();
    if n == 0 || c <= 0.0 || norm <= 0.0 {
        return Vec::new();
    }
    let u = [dir[0] / norm, dir[1] / norm, dir[2] / norm];
    let proj: Vec<f64> = (0..n)
        .map(|i| {
            positions[[i, 2]].mul_add(
                u[2],
                positions[[i, 0]].mul_add(u[0], positions[[i, 1]] * u[1]),
            )
        })
        .collect();
    let p_max = proj.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    proj.iter().map(|&p| (p_max - p) / c).collect()
}

/// Per-element local **Menger curvature** \[1/m] of the array centreline from
/// three consecutive element positions: `κ_i = 2|(r_i−r_{i-1})×(r_{i+1}−r_i)| /
/// (|r_i−r_{i-1}||r_{i+1}−r_i||r_{i+1}−r_{i-1}|)` (= `1/R_circumscribed`).
///
/// Endpoints copy their interior neighbour. Collinear (flat) triples give `0`.
/// Used to drive the CMUT flex-derating apodization below.
#[must_use]
pub fn per_element_curvature(positions: &ArrayView2<f64>) -> Vec<f64> {
    let n = positions.shape()[0];
    if n < 3 {
        return vec![0.0; n];
    }
    let row = |i: usize| [positions[[i, 0]], positions[[i, 1]], positions[[i, 2]]];
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let norm = |v: [f64; 3]| v[2].mul_add(v[2], v[0].mul_add(v[0], v[1] * v[1])).sqrt();
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1].mul_add(b[2], -(a[2] * b[1])),
            a[2].mul_add(b[0], -(a[0] * b[2])),
            a[0].mul_add(b[1], -(a[1] * b[0])),
        ]
    };

    let mut kappa = vec![0.0; n];
    for (i, cell) in kappa.iter_mut().enumerate().take(n - 1).skip(1) {
        let (pm, pi, pp) = (row(i - 1), row(i), row(i + 1));
        let a = sub(pi, pm);
        let b = sub(pp, pi);
        let chord = sub(pp, pm);
        let denom = norm(a) * norm(b) * norm(chord);
        *cell = if denom > 0.0 {
            2.0 * norm(cross(a, b)) / denom
        } else {
            0.0
        };
    }
    kappa[0] = kappa[1];
    kappa[n - 1] = kappa[n - 2];
    kappa
}

/// Per-element transmit **apodization** for a CMUT-populated conformal array.
///
/// Each element's weight is the flex-derated output `CmutCell::flex_gap_derating(κ_i)`
/// at its local curvature `κ_i` — wrapping the array over tissue perturbs the
/// sub-micron gap and cuts output where curvature is high (§33.8). Weights lie
/// in `(0, 1]` (a flat element, `κ=0`, keeps the full weight `1`); they are
/// returned un-normalized so the caller can normalize per its transmit budget.
#[must_use]
pub fn cmut_flex_apodization(curvatures: &[f64], cell: &CmutCell) -> Vec<f64> {
    curvatures
        .iter()
        .map(|&k| cell.flex_gap_derating(k))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use leto::Array2;

    /// Build an `n×3` position array from rows.
    fn positions(rows: &[[f64; 3]]) -> Array2<f64> {
        let mut a = Array2::zeros([rows.len(), 3]);
        for (i, r) in rows.iter().enumerate() {
            a[[i, 0]] = r[0];
            a[[i, 1]] = r[1];
            a[[i, 2]] = r[2];
        }
        a
    }

    fn distance(p: [f64; 3], q: [f64; 3]) -> f64 {
        let d = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
        d[2].mul_add(d[2], d[0].mul_add(d[0], d[1] * d[1])).sqrt()
    }

    const C: f64 = 1500.0;

    /// The defining property: focusing delays make every element arrive at the
    /// focus simultaneously — `d_i/c + τ_i` is identical for all elements — and
    /// delays are non-negative with at least one zero (the farthest element).
    #[test]
    fn focusing_delays_align_arrivals_at_focus() {
        // An arbitrarily *deformed* 5-element array (not flat).
        let pos = positions(&[
            [-2e-3, 0.0, 0.0],
            [-1e-3, 0.2e-3, 0.0],
            [0.0, 0.3e-3, 0.0],
            [1e-3, 0.2e-3, 0.0],
            [2e-3, 0.0, 0.0],
        ]);
        let focus = [0.0, 0.0, 10e-3];
        let tau = focusing_delays(&pos.view(), focus, C);
        assert_eq!(tau.len(), 5);

        let arrival: Vec<f64> = (0..5)
            .map(|i| {
                let r = [pos[[i, 0]], pos[[i, 1]], pos[[i, 2]]];
                distance(r, focus) / C + tau[i]
            })
            .collect();
        let a0 = arrival[0];
        for a in &arrival {
            assert!(
                (a - a0).abs() <= 1e-15,
                "arrivals must coincide: {a} vs {a0}"
            );
        }
        assert!(tau.iter().all(|&t| t >= 0.0));
        assert!(
            tau.iter().any(|&t| t.abs() < 1e-15),
            "farthest element has τ=0"
        );
    }

    /// Flat array, on-axis focus: delays are symmetric and peak at the centre
    /// (closest element ⇒ largest delay), zero at the edges (farthest).
    #[test]
    fn flat_array_on_axis_focus_is_symmetric_peaked_center() {
        let pos = positions(&[
            [-2e-3, 0.0, 0.0],
            [-1e-3, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1e-3, 0.0, 0.0],
            [2e-3, 0.0, 0.0],
        ]);
        let tau = focusing_delays(&pos.view(), [0.0, 0.0, 20e-3], C);
        assert!((tau[0] - tau[4]).abs() < 1e-15, "symmetric");
        assert!((tau[1] - tau[3]).abs() < 1e-15, "symmetric");
        assert!(tau[2] > tau[1] && tau[1] > tau[0], "peaked at centre");
        assert!(tau[0].abs() < 1e-15, "edges fire first");
    }

    /// Broadside steering (dir = +z) of a flat x-array gives equal (zero) delays;
    /// an oblique in-plane direction gives a monotone linear ramp.
    #[test]
    fn steering_delays_broadside_and_oblique() {
        let pos = positions(&[
            [0.0, 0.0, 0.0],
            [1e-3, 0.0, 0.0],
            [2e-3, 0.0, 0.0],
            [3e-3, 0.0, 0.0],
        ]);
        // Broadside (perpendicular to the array axis): all projections equal ⇒ τ=0.
        let broadside = steering_delays(&pos.view(), [0.0, 0.0, 1.0], C);
        assert!(broadside.iter().all(|&t| t.abs() < 1e-15));

        // Along +x: projections increase with index ⇒ delays decrease linearly.
        let along = steering_delays(&pos.view(), [1.0, 0.0, 0.0], C);
        let step = along[0] - along[1];
        assert!(step > 0.0);
        for i in 0..3 {
            assert!(
                (along[i] - along[i + 1] - step).abs() < 1e-15,
                "linear ramp"
            );
        }
        assert!(along[3].abs() < 1e-15, "leading element fires first");
    }

    /// Per-element curvature is zero for a collinear (flat) array and equals
    /// `1/R` for points on a circle of radius `R`.
    #[test]
    fn per_element_curvature_flat_zero_circle_inverse_radius() {
        let flat = positions(&[
            [-2e-3, 0.0, 0.0],
            [-1e-3, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1e-3, 0.0, 0.0],
            [2e-3, 0.0, 0.0],
        ]);
        for k in per_element_curvature(&flat.view()) {
            assert!(k.abs() < 1e-9, "flat array curvature must be 0, got {k}");
        }

        // Points on a circle of radius R in the x–z plane.
        let r = 5e-3;
        let mut rows = Vec::new();
        for j in -2..=2 {
            let theta = j as f64 * 0.1; // small arc
            rows.push([r * theta.sin(), 0.0, r * (1.0 - theta.cos())]);
        }
        let arc = positions(&rows);
        let kappa = per_element_curvature(&arc.view());
        // Interior points recover 1/R.
        for &k in &kappa[1..kappa.len() - 1] {
            assert!(
                (k - 1.0 / r).abs() <= 1e-3 * (1.0 / r),
                "circle κ≈1/R: {k} vs {}",
                1.0 / r
            );
        }
    }

    /// CMUT flex apodization: flat elements keep full weight 1; curved elements
    /// are derated (< 1) and tighter-gap cells are derated more.
    #[test]
    fn cmut_flex_apodization_derates_curved_elements() {
        let cell = CmutCell::silicon(60e-6, 2.0e-6, 0.2e-6).unwrap();
        let curvatures = [0.0, 1.0 / 2.0e-3, 1.0 / 1.0e-3]; // flat, 2 mm, 1 mm radius
        let w = cmut_flex_apodization(&curvatures, &cell);
        assert!((w[0] - 1.0).abs() < 1e-12, "flat element keeps full weight");
        assert!(
            w[1] < w[0] && w[2] < w[1],
            "tighter curvature ⇒ more derating"
        );
        assert!(w.iter().all(|&x| x > 0.0 && x <= 1.0));

        // A tighter-gap cell loses more output at the same curvature.
        let tight = CmutCell::silicon(60e-6, 2.0e-6, 0.1e-6).unwrap();
        let wt = cmut_flex_apodization(&curvatures, &tight);
        assert!(wt[2] < w[2], "tighter gap ⇒ more flex derating");
    }
}
