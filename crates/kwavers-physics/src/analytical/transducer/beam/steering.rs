use eunomia::Complex64;
use kwavers_math::special::bessel::j1;
use moirai_parallel::{map_collect_with, Adaptive};

// ─── Steering envelope (grating-lobe limited) ──────────────────────────────────

/// Aperiodic ("sparse") linear element positions — same aperture and element
/// count as a uniform array, but with the periodic grid broken by a
/// deterministic low-discrepancy dither.
///
/// The array lies along x, centred on the origin, spanning the full aperture
/// `aperture_m`; the two endpoint elements are anchored at `±aperture/2`, so
/// the physical aperture (hence the diffraction-limited main-lobe width) is
/// identical to the uniform array. Each interior element is displaced from its
/// grid position by a fraction `jitter_frac` of the element pitch using a
/// golden-ratio additive recurrence (`jitter_frac = 0` reproduces the uniform
/// layout; `≈0.7` strongly suppresses grating lobes). Destroying the spatial
/// periodicity scatters what would otherwise be a single coherent grating lobe
/// into a low, incoherent pedestal, so the same element count can be steered
/// further before any secondary lobe reaches the −6 dB safety limit. Only the
/// *placement* changes — frequency, aperture and element count are fixed — so
/// the comparison isolates the element-activation pattern.
///
/// # Arguments
/// * `n` – number of elements (matched to the uniform array)
/// * `aperture_m` – full aperture span D [m]
/// * `jitter_frac` – dither amplitude as a fraction of the element pitch
///
/// # Returns
/// Element x-positions [m], length `n`.
///
/// # Reference
/// Steinberg (1976) *Principles of Aperture and Array System Design* (thinned
/// aperiodic arrays and grating-lobe suppression).
#[must_use]
pub fn linear_array_aperiodic_positions(n: usize, aperture_m: f64, jitter_frac: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0.0];
    }
    // Golden-ratio additive recurrence → low-discrepancy dither in [0, 1).
    let golden = 0.5 * (5.0_f64.sqrt() - 1.0);
    let pitch = aperture_m / (n as f64 - 1.0);
    let half = 0.5 * aperture_m;
    (0..n)
        .map(|i| {
            if i == 0 {
                -half
            } else if i == n - 1 {
                half
            } else {
                let base = -half + pitch * (i as f64);
                let u = ((i as f64 + 1.0) * golden).fract();
                base + (u - 0.5) * jitter_frac * pitch
            }
        })
        .collect()
}

/// Baffled circular-piston element directivity `D(θ) = 2 J₁(ka·sinθ)/(ka·sinθ)`,
/// normalised so `D(0) = 1`. Avoids the per-call allocation of
/// [`circular_piston_directivity`] in the hot beam-pattern loops.
#[inline]
fn piston_directivity(theta: f64, ka: f64) -> f64 {
    let arg = ka * theta.sin();
    if arg.abs() < 1e-12 {
        1.0
    } else {
        2.0 * j1(arg) / arg
    }
}

/// Steered far-field beam pattern of a linear array.
///
/// The `N` elements lie along x at positions `elem_x` and radiate broadside
/// (normal `+z`); the array is phased to steer its main lobe to `steer_theta`
/// (measured from broadside). The far-field response at observation angle `θ`
/// is the product of the element directivity and the array factor,
/// ```text
/// P(θ) = D(θ) · | (1/N) Σ_i exp[ i·k·x_i·(sinθ − sin θ_s) ] |,
/// ```
/// where `D(θ) = 2 J₁(ka·sinθ)/(ka·sinθ)` is the baffled circular-piston
/// element factor with parameter `ka_elem = k·a_elem`. At `θ = θ_s` the array
/// factor is unity and `P = D(θ_s)` (the main lobe); coherent secondary peaks
/// where the array factor returns to unity at other angles are grating lobes.
///
/// # Arguments
/// * `elem_x` – element x-positions [m]
/// * `obs_theta` – observation angles [rad], from broadside
/// * `k` – wavenumber 2πf/c [rad/m]
/// * `steer_theta` – steering angle [rad], from broadside
/// * `ka_elem` – element directivity parameter k·a_elem
///
/// # Returns
/// Beam-pattern magnitude at each `obs_theta`.
///
/// # Reference
/// Van Trees (2002) *Optimum Array Processing*, §2.2; O'Neil (1949) (element
/// directivity).
#[must_use]
pub fn steered_beam_pattern_1d(
    elem_x: &[f64],
    obs_theta: &[f64],
    k: f64,
    steer_theta: f64,
    ka_elem: f64,
) -> Vec<f64> {
    let n = (elem_x.len() as f64).max(1.0);
    let sin_s = steer_theta.sin();
    obs_theta
        .iter()
        .map(|&th| {
            let dsin = th.sin() - sin_s;
            let mut acc = Complex64::new(0.0, 0.0);
            for &x in elem_x {
                acc += Complex64::from_polar(1.0, k * x * dsin);
            }
            piston_directivity(th, ka_elem) * (acc.norm() / n)
        })
        .collect()
}

/// Grating-lobe ratio versus steering angle — the basis of the *steering
/// envelope* at a fixed frequency.
///
/// For each steering angle `θ_s` the array is phased to that angle and its
/// beam pattern ([`steered_beam_pattern_1d`]) is searched for the strongest
/// lobe outside the main lobe (a `±mainlobe_halfwidth_rad` window about `θ_s`).
/// The returned value is the grating-lobe ratio
/// ```text
/// G(θ_s) = max_{|θ − θ_s| > Δ} P(θ) / P(θ_s),
/// ```
/// with `P(θ_s) = D(θ_s)` the main-lobe peak. The safe **steering envelope**
/// is the set `{ θ_s : G(θ_s) ≤ 0.5 }`, where no secondary lobe exceeds half
/// (−6 dB) the main-lobe pressure.
///
/// Holding frequency, aperture and element count fixed, a uniform array raises
/// a coherent grating lobe once `θ_s` exceeds the threshold set by its element
/// pitch — `G` jumps up — whereas an aperiodic array
/// ([`linear_array_aperiodic_positions`]) keeps `G` low over a much wider
/// steering range. The distinction is purely the element-activation pattern.
///
/// # Arguments
/// * `elem_x` – element x-positions [m]
/// * `steer_theta` – steering-angle grid [rad]
/// * `obs_theta` – observation-angle grid [rad] for the lobe search
/// * `k` – wavenumber 2πf/c [rad/m]
/// * `ka_elem` – element directivity parameter k·a_elem
/// * `mainlobe_halfwidth_rad` – half-width of the main-lobe exclusion window [rad]
///
/// # Returns
/// Grating-lobe ratio at each `steer_theta`.
///
/// # Reference
/// Steinberg (1976) *Principles of Aperture and Array System Design*; Pernot
/// et al. (2003) *Ultrasound Med. Biol.* 29(11):1559–1565 (electronic-steering
/// envelope and grating lobes of therapy arrays).
#[must_use]
pub fn steering_grating_lobe_ratio_1d(
    elem_x: &[f64],
    steer_theta: &[f64],
    obs_theta: &[f64],
    k: f64,
    ka_elem: f64,
    mainlobe_halfwidth_rad: f64,
) -> Vec<f64> {
    map_collect_with::<Adaptive, _, _, _>(steer_theta, |&ts| {
        let pat = steered_beam_pattern_1d(elem_x, obs_theta, k, ts, ka_elem);
        let main = piston_directivity(ts, ka_elem).max(1e-300);
        let mut secondary = 0.0_f64;
        for (i, &th) in obs_theta.iter().enumerate() {
            if (th - ts).abs() <= mainlobe_halfwidth_rad {
                continue; // inside the main lobe
            }
            if pat[i] > secondary {
                secondary = pat[i];
            }
        }
        secondary / main
    })
}

/// Safe steering half-angle — the largest steering excursion from broadside
/// over which the grating-lobe ratio stays at or below a safety threshold.
///
/// Starting from the steering angle closest to broadside (`θ_s = 0`), the safe
/// region is expanded outward to both sides while `G ≤ threshold`; the returned
/// value is the symmetric half-angle `min(θ_right, |θ_left|)` of that
/// contiguous run. With `threshold = 0.5` this is the −6 dB grating-lobe-safe
/// steering half-angle; the ratio of an aperiodic to a uniform half-angle
/// quantifies the steering-envelope expansion from sparse activation. Returns
/// `0` if broadside itself is unsafe.
///
/// # Arguments
/// * `steer_theta` – steering-angle grid [rad] (monotonically increasing)
/// * `glr` – grating-lobe ratio at each `steer_theta`
/// * `threshold` – grating-lobe safety threshold (e.g. 0.5)
///
/// # Returns
/// Safe steering half-angle [rad].
///
/// # Panics
/// Panics if `steer_theta` and `glr` differ in length.
#[must_use]
pub fn safe_steering_halfangle(steer_theta: &[f64], glr: &[f64], threshold: f64) -> f64 {
    assert_eq!(
        steer_theta.len(),
        glr.len(),
        "steer_theta and glr length mismatch"
    );
    if steer_theta.is_empty() {
        return 0.0;
    }
    // Index closest to broadside (θ = 0).
    let i0 = (0..steer_theta.len())
        .min_by(|&a, &b| {
            steer_theta[a]
                .abs()
                .partial_cmp(&steer_theta[b].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    if glr[i0] > threshold {
        return 0.0;
    }
    let mut hi = i0;
    while hi + 1 < glr.len() && glr[hi + 1] <= threshold {
        hi += 1;
    }
    let mut lo = i0;
    while lo > 0 && glr[lo - 1] <= threshold {
        lo -= 1;
    }
    steer_theta[hi].abs().min(steer_theta[lo].abs())
}
