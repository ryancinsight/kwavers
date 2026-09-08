//! Value-semantic tests for the far-field patch SIR and its sparse-delta
//! evaluation. Bounds are derived from the trapezoid closed form, the
//! midpoint rule, and the sagitta of the far-field replacement.

use super::super::RectangularPistonSir;
use super::FarFieldRectangleSir;
use std::f64::consts::PI;
use std::num::NonZeroUsize;

const C: f64 = 1540.0;
/// A linear-array element: 0.3 mm wide, 5 mm tall.
const WX: f64 = 0.15e-3;
const WY: f64 = 2.5e-3;
const DT: f64 = 1.0 / 100.0e6;

fn patches(nx: usize, ny: usize) -> [NonZeroUsize; 2] {
    [
        NonZeroUsize::new(nx).expect("non-zero"),
        NonZeroUsize::new(ny).expect("non-zero"),
    ]
}

fn sir(nx: usize, ny: usize) -> FarFieldRectangleSir {
    FarFieldRectangleSir::new(WX, WY, patches(nx, ny), C).expect("valid element")
}

/// The far-field trapezoid of a single patch, written out from the closed
/// form independently of the implementation.
struct Trapezoid {
    t1: f64,
    t2: f64,
    t3: f64,
    t4: f64,
    slope: f64,
    height: f64,
    area: f64,
}

fn trapezoid(wx: f64, wy: f64, centre: [f64; 2], field: [f64; 3]) -> Trapezoid {
    let dx = field[0] - centre[0];
    let dy = field[1] - centre[1];
    let l = (dx * dx + dy * dy + field[2] * field[2]).sqrt();
    let a = wx * dx.abs() / (l * C);
    let b = wy * dy.abs() / (l * C);
    let (dt1, dt2) = (a.min(b), a.max(b));
    let area = wx * wy / (2.0 * PI * l);
    let height = area / dt2;
    let t1 = l / C - 0.5 * (dt1 + dt2);
    Trapezoid {
        t1,
        t2: t1 + dt1,
        t3: t1 + dt2,
        t4: t1 + dt1 + dt2,
        slope: height / dt1,
        height,
        area,
    }
}

impl Trapezoid {
    fn at(&self, t: f64) -> f64 {
        if t < self.t1 || t >= self.t4 {
            0.0
        } else if t < self.t2 {
            self.slope * (t - self.t1)
        } else if t < self.t3 {
            self.height
        } else {
            self.slope * (self.t4 - t)
        }
    }
}

/// ## Theorem
/// Sparse delta integration reproduces the trapezoid exactly at every node.
///
/// A linearly split delta keeps its zeroth and first moments, so the double
/// cumulative sum equals `w·(t_n − t_k)` at every node past it and zero
/// before; a trapezoid is four such ramps. The only discrepancy is rounding:
/// each node accumulates at most `n` terms of size `s·dt`, so the error is
/// below `n²·ε·s·dt` with `n ≤ 200` here, i.e. under `1e-11·s·dt`, against a
/// plateau `h_max = s·Δt₁` of tens of samples.
#[test]
fn single_patch_matches_the_trapezoid_at_every_node() {
    let s = sir(1, 1);
    let field = [10.0e-3, 3.0e-3, 30.0e-3];
    let reference = trapezoid(2.0 * WX, 2.0 * WY, [0.0, 0.0], field);
    let response = s.response(field[0], field[1], field[2], DT);

    assert!(
        !response.samples.is_empty(),
        "a 5 mm element seen 18° off axis spans many samples"
    );
    let span_nodes = ((reference.t4 - reference.t1) / DT).ceil();
    assert!(
        span_nodes > 20.0,
        "kernel must be wide enough to test ramps"
    );
    let bound = 1.0e-10 * reference.slope * DT;

    // Every node of the returned support.
    for (i, &sample) in response.samples.iter().enumerate() {
        let t = (response.first_sample + i) as f64 * DT + 0.5 * DT;
        let expected = reference.at(t);
        assert!(
            (sample - expected).abs() <= bound,
            "node {i} at t={t:.4e}: {sample:.6e} against trapezoid {expected:.6e}"
        );
    }
    // And nothing outside it: the node before the first sample and the node
    // after the last are outside `[t₁, t₄)`.
    let before = (response.first_sample as f64 - 0.5) * DT;
    let after = (response.first_sample + response.samples.len()) as f64 * DT + 0.5 * DT;
    assert_eq!(reference.at(before), 0.0);
    assert_eq!(reference.at(after), 0.0);
}

/// ## Theorem
/// On the patch axis the far-field SIR is `A·δ(t − l/c)`, so one bin carries
/// `A/dt` and no other carries anything.
#[test]
fn axial_patch_is_a_single_bin_delta() {
    let s = sir(1, 1);
    let z = 20.0e-3;
    let response = s.response(0.0, 0.0, z, DT);
    let area = 4.0 * WX * WY / (2.0 * PI * z);
    let bin = (z / C / DT).floor() as usize;

    assert_eq!(response.samples.len(), 1, "exactly one bin is occupied");
    assert_eq!(response.first_sample, bin);
    assert!(
        (response.samples[0] - area / DT).abs() <= 1.0e-12 * area / DT,
        "delta weight {:.6e} against A/dt = {:.6e}",
        response.samples[0],
        area / DT
    );
}

/// ## Theorem
/// In a symmetry plane the trapezoid is a rectangle of height `A/Δt₂`, and
/// the bin-average deposit preserves its area exactly.
#[test]
fn symmetry_plane_patch_is_an_area_preserving_rectangle() {
    let s = sir(1, 1);
    // In the x = 0 plane the 5 mm height sets the span: ~78 bins at 100 MHz.
    let field = [0.0, 5.0e-3, 20.0e-3];
    let reference = trapezoid(2.0 * WX, 2.0 * WY, [0.0, 0.0], field);
    let response = s.response(field[0], field[1], field[2], DT);

    let area: f64 = response.samples.iter().sum::<f64>() * DT;
    assert!(
        (area - reference.area).abs() <= 1.0e-12 * reference.area,
        "sampled area {area:.6e} against A = {:.6e}",
        reference.area
    );
    // Interior nodes (bins fully inside [t₁, t₃]) sit at the plateau.
    let interior: Vec<f64> = response
        .samples
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let bin = response.first_sample + i;
            bin as f64 * DT > reference.t1 && (bin + 1) as f64 * DT < reference.t3
        })
        .map(|(_, &v)| v)
        .collect();
    assert!(interior.len() > 5, "the rectangle spans several full bins");
    for v in interior {
        assert!(
            (v - reference.height).abs() <= 1.0e-12 * reference.height,
            "plateau {v:.6e} against A/Δt₂ = {:.6e}",
            reference.height
        );
    }
}

/// ## Theorem
/// The sampled area equals the Rayleigh integral `(1/2π)∫_S dS/R` within two
/// derived bounds.
///
/// `Σ_m A_m` is the composite midpoint rule for that integral over the patch
/// tiling, with error at most `S·(w_x² + w_y²)/(24π·R_min³)` since
/// `|∂²(1/R)| ≤ 2/R³`. Sampling each trapezoid at bin midpoints is exact on its
/// linear pieces and errs by at most `|Δs|·dt²/8` in each of the four kink
/// bins, i.e. `A_m·dt²/(2·Δt₁·Δt₂)` per patch. The reference uses a `512²`
/// tiling, whose own bound is a further `2⁻¹⁴` of the coarse one.
#[test]
fn sampled_area_is_the_rayleigh_integral_within_the_midpoint_bounds() {
    let (hw, nx, ny) = (1.0e-3, 4, 4);
    let s = FarFieldRectangleSir::new(hw, hw, patches(nx, ny), C).expect("valid");
    let field = [3.0e-3, 1.0e-3, 30.0e-3];
    let dt = 1.0e-9;
    let response = s.response(field[0], field[1], field[2], dt);
    let sampled: f64 = response.samples.iter().sum::<f64>() * dt;

    let rayleigh = |n: usize| -> f64 {
        let w = 2.0 * hw / n as f64;
        let mut total = 0.0;
        for ix in 0..n {
            for iy in 0..n {
                let cx = -hw + (ix as f64 + 0.5) * w;
                let cy = -hw + (iy as f64 + 0.5) * w;
                let r = ((field[0] - cx).powi(2) + (field[1] - cy).powi(2) + field[2] * field[2])
                    .sqrt();
                total += w * w / (2.0 * PI * r);
            }
        }
        total
    };
    let reference = rayleigh(512);
    let surface = 4.0 * hw * hw;
    let r_min = ((field[0] - hw).powi(2) + (field[1] - hw).powi(2) + field[2] * field[2]).sqrt();
    let midpoint_bound = |n: usize| {
        let w = 2.0 * hw / n as f64;
        surface * 2.0 * w * w / (24.0 * PI * r_min.powi(3))
    };
    // Kink bound summed over the patches from the closed form.
    let w = 2.0 * hw / nx as f64;
    let mut kink_bound = 0.0;
    for ix in 0..nx {
        for iy in 0..ny {
            let cx = -hw + (ix as f64 + 0.5) * w;
            let cy = -hw + (iy as f64 + 0.5) * w;
            let t = trapezoid(w, w, [cx, cy], field);
            kink_bound += t.area * dt * dt / (2.0 * (t.t2 - t.t1) * (t.t3 - t.t1));
        }
    }
    let bound = midpoint_bound(nx) + midpoint_bound(512) + kink_bound;
    assert!(
        (sampled - reference).abs() <= bound,
        "sampled area {sampled:.9e} against Rayleigh integral {reference:.9e}: |Δ| = {:.3e} exceeds the derived bound {bound:.3e}",
        (sampled - reference).abs()
    );
    // The bound is a bound, not a target: it must be small against the value.
    assert!(
        bound < 1.0e-2 * reference,
        "bound {bound:.3e} is not tight enough to mean anything"
    );
}

/// ## Theorem
/// The far-field support lies within the exact support up to the sagitta.
///
/// Replacing the wavefront's trace on the plane by a straight line drops a
/// distance term bounded by `(w_x² + w_y²)/(8l)` per patch, so the earliest
/// far-field onset and the latest end differ from the exact Lockwood–Willette
/// first and last arrivals by at most that over `c` at the nearest patch.
#[test]
fn support_lies_within_the_exact_support_by_the_sagitta_bound() {
    let s = sir(4, 16);
    let exact = RectangularPistonSir::new(WX, WY, C).expect("valid");
    let field = [10.0e-3, 3.0e-3, 30.0e-3];
    let [wx, wy] = s.patch_widths();
    // Nearest patch centre distance bounds every patch's `l` from below.
    let l_min = ((field[0] - WX).powi(2) + (field[1] - WY).powi(2) + field[2] * field[2]).sqrt();
    let sagitta_time = (wx * wx + wy * wy) / (8.0 * l_min * C);

    let first = s.first_arrival_time(field[0], field[1], field[2]);
    let last = s.last_arrival_time(field[0], field[1], field[2]);
    let exact_first = exact.first_arrival_time(field[0], field[1], field[2]);
    let exact_last = exact.last_arrival_time(field[0], field[1], field[2]);
    assert!(
        (first - exact_first).abs() <= sagitta_time,
        "onset {first:.6e} against exact {exact_first:.6e}, sagitta bound {sagitta_time:.3e}"
    );
    assert!(
        (last - exact_last).abs() <= sagitta_time,
        "end {last:.6e} against exact {exact_last:.6e}, sagitta bound {sagitta_time:.3e}"
    );
}

/// ## Theorem
/// The convolution integral factorizes: `Σ(h⊛h)·dt = (Σh·dt)²` exactly for
/// the discrete auto-convolution, and the round trip starts at twice the
/// one-way onset.
#[test]
fn round_trip_factorizes_and_starts_at_twice_the_onset() {
    let s = sir(1, 8);
    let field = [6.0e-3, 2.0e-3, 25.0e-3];
    let one_way = s.response(field[0], field[1], field[2], DT);
    let two_way = s.round_trip_response(field[0], field[1], field[2], DT);
    let area: f64 = one_way.samples.iter().sum::<f64>() * DT;
    let integral: f64 = two_way.samples.iter().sum::<f64>() * DT;
    assert!(
        (integral - area * area).abs() <= 1.0e-9 * area * area,
        "Σ(h⊛h)dt = {integral:.9e} against (Σh dt)² = {:.9e}",
        area * area
    );
    assert_eq!(two_way.first_sample, 2 * one_way.first_sample);
    assert_eq!(two_way.samples.len(), 2 * one_way.samples.len() - 1);
}

/// ## Theorem
/// Refining the tiling drives the far-field kernel toward the exact one.
///
/// The per-patch error is the sagitta, `O(w²)`, so each halving of the patch
/// widths must at least halve the `L1` distance to the Lockwood–Willette
/// kernel at the shared nodes (order one is the floor the derivation admits;
/// the measured ratios are close to four, order two, and are recorded on the
/// board). A misplaced or mis-signed trapezoid does not converge at all.
#[test]
fn refining_the_patches_converges_on_the_exact_kernel() {
    let exact = RectangularPistonSir::new(WX, WY, C).expect("valid");
    let field = [10.0e-3, 3.0e-3, 30.0e-3];
    let tilings = [(1, 4), (2, 8), (4, 16), (8, 32)];
    let distances: Vec<f64> = tilings
        .iter()
        .map(|&(nx, ny)| {
            let response = sir(nx, ny).response(field[0], field[1], field[2], DT);
            let mut l1 = 0.0;
            let mut area = 0.0;
            // Compare over the exact support plus the far-field one.
            let start = response
                .first_sample
                .min((exact.first_arrival_time(field[0], field[1], field[2]) / DT) as usize);
            let end = (response.first_sample + response.samples.len())
                .max((exact.last_arrival_time(field[0], field[1], field[2]) / DT) as usize + 2);
            for n in start..end {
                let t = (n as f64 + 0.5) * DT;
                let e = exact.evaluate(field[0], field[1], field[2], t);
                let f = n
                    .checked_sub(response.first_sample)
                    .and_then(|i| response.samples.get(i))
                    .copied()
                    .unwrap_or(0.0);
                l1 += (e - f).abs() * DT;
                area += e * DT;
            }
            l1 / area
        })
        .collect();
    for pair in distances.windows(2) {
        assert!(
            pair[1] <= 0.5 * pair[0],
            "each halving of the patches must at least halve the L1 distance, got {distances:?}"
        );
    }
}

/// ## Theorem
/// A corner before the first grid node still yields the trapezoid at every
/// node that exists.
///
/// The buffer is anchored at the earliest onset's floor node, signed, so the
/// rising pair splits and cancels as it would anywhere else; what the grid
/// cannot carry is the ramp before `t = dt/2`. Here the onset is negative —
/// the field point sits a millimetre from a 5 mm element — and nodes 0 to 205
/// lie on the plateau. The clamp this replaces annihilated the rising pair at
/// node 0 and returned one negative sample at node 206.
#[test]
fn a_corner_before_the_first_node_keeps_the_plateau() {
    let s = sir(1, 1);
    let field = [0.1e-3, 1.0e-3, 1.0e-3];
    let dt = 1.0e-8;
    let reference = trapezoid(2.0 * WX, 2.0 * WY, [0.0, 0.0], field);
    assert!(reference.t1 < 0.0, "the case must start before t = 0");
    let response = s.response(field[0], field[1], field[2], dt);

    assert_eq!(response.first_sample, 0, "node 0 is on the plateau");
    let bound = 1.0e-10 * reference.slope * dt;
    let mut plateau_nodes = 0;
    for (i, &sample) in response.samples.iter().enumerate() {
        let t = (i as f64 + 0.5) * dt;
        let expected = reference.at(t);
        assert!(
            (sample - expected).abs() <= bound,
            "node {i}: {sample:.6e} against trapezoid {expected:.6e}"
        );
        if expected == reference.height {
            plateau_nodes += 1;
        }
    }
    assert!(plateau_nodes >= 200, "got {plateau_nodes} plateau nodes");
}

/// A degenerate rectangle straddling `t = 0` keeps exactly the area it has
/// at `t ≥ 0`: the bins that exist take their overlap, nothing more. This
/// characterizes the bin-average deposit at the grid's edge; it is not a
/// regression guard — the clamp the plateau test above replaces already
/// clipped this case correctly.
#[test]
fn a_rectangle_straddling_the_first_node_keeps_its_representable_area() {
    let s = sir(1, 1);
    let field = [0.0, 0.5e-3, 0.1e-3];
    let dt = 1.0e-8;
    let reference = trapezoid(2.0 * WX, 2.0 * WY, [0.0, 0.0], field);
    assert!(reference.t1 < 0.0 && reference.t3 > 0.0, "straddles t = 0");
    let response = s.response(field[0], field[1], field[2], dt);
    let area: f64 = response.samples.iter().sum::<f64>() * dt;
    let expected = reference.height * (reference.t3 - reference.t1.max(0.0));
    assert!(
        (area - expected).abs() <= 1.0e-12 * expected,
        "area at t ≥ 0: {area:.6e} against {expected:.6e}"
    );
    assert_eq!(response.first_sample, 0);
}

/// A field point at a patch centre in the aperture plane has `l = 0`: no
/// finite corner time exists, and the response must say so with a non-finite
/// sample — the signal the phantom seam turns into an error — rather than
/// overflow the buffer arithmetic or return an empty kernel that would pass
/// as a point element.
#[test]
fn a_field_point_at_a_patch_centre_yields_a_non_finite_sample() {
    let s = sir(1, 1);
    let response = s.response(0.0, 0.0, 0.0, DT);
    assert_eq!(response.samples.len(), 1);
    assert!(response.samples[0].is_nan());
    let two_way = s.round_trip_response(0.0, 0.0, 0.0, DT);
    assert!(two_way.samples.iter().all(|v| v.is_nan()));
}

#[test]
fn far_field_number_is_the_sagitta_over_the_half_wavelength() {
    let s = sir(1, 16);
    let [_, wy] = s.patch_widths();
    let (l, f) = (30.0e-3, 5.0e6);
    let expected = wy * wy * f / (4.0 * l * C);
    let sagitta = wy * wy / (8.0 * l);
    let half_wavelength = C / (2.0 * f);
    assert!((s.far_field_number(l, f) - expected).abs() <= 1.0e-15);
    assert!((expected - sagitta / half_wavelength).abs() <= 1.0e-15 * expected);
}

#[test]
fn rejects_invalid_parameters() {
    assert!(FarFieldRectangleSir::new(0.0, WY, patches(1, 1), C).is_err());
    assert!(FarFieldRectangleSir::new(WX, -1.0, patches(1, 1), C).is_err());
    assert!(FarFieldRectangleSir::new(WX, WY, patches(1, 1), 0.0).is_err());
    assert!(FarFieldRectangleSir::new(WX, WY, patches(1, 1), f64::NAN).is_err());
}
