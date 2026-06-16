//! Transducer array and beamforming physics for book chapters ch04, ch11.
//!
//! Covers: circular piston directivity, linear array factor, grating lobes,
//! apodization windows, delay laws, 2-D beam patterns, on-axis pressure
//! profiles, and bandlimited interpolation stencils.

pub mod array_factor;
pub mod beam;
pub mod interpolation;
pub mod optoacoustic;
pub mod steering;

pub use array_factor::{
    apodization_weights, beam_pattern_magnitude, circular_piston_directivity, grating_lobe_angles,
    linear_array_factor,
};
pub use beam::{
    beam_pattern_2d, beam_pattern_2d_magnitude, circular_piston_onaxis, delay_law_focus_2d,
    delay_law_focus_3d, delay_law_steer_2d, focused_bowl_element_positions_3d, focused_bowl_onaxis,
    focused_bowl_steered_pressure_profile, linear_array_aperiodic_positions,
    linear_array_positions, multi_focus_delay_laws_2d, multi_focus_field_magnitude_2d,
    near_field_distance, safe_steering_halfangle, steered_aperture_pressure_3d,
    steered_beam_pattern_1d, steering_focus_point, steering_grating_lobe_ratio_1d,
};
pub use interpolation::bli_stencil_weights;
pub use optoacoustic::{
    acoustic_resolution_lateral, f_number_from_na, fiber_tip_fluence, focused_aperture_gain,
    na_from_f_number, numerical_aperture_from_geometry, optoacoustic_array_focal_pressure,
    optoacoustic_center_frequency, soap_focal_gain,
};
pub use steering::electronic_steering_efficiency;

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;
    use std::f64::consts::PI;

    #[test]
    fn piston_directivity_on_axis() {
        let d = circular_piston_directivity(&[0.0], 5.0);
        assert!((d[0] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn array_factor_at_steering_angle_is_one() {
        let steer = 0.1_f64;
        let k = 2.0 * PI * 2.0 * MHZ_TO_HZ / SOUND_SPEED_WATER_SIM;
        let af = linear_array_factor(&[steer], k, 0.3e-3, 64, steer);
        assert!((af[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn apodization_uniform_sum() {
        let w = apodization_weights(64, "uniform");
        let s: f64 = w.iter().sum();
        assert!((s - 64.0).abs() < 1e-10);
    }

    #[test]
    fn apodization_tukey_tapers_to_zero_symmetrically() {
        // Regression: the prior inline Tukey reimplementation had a sign error
        // in its falling edge and returned 1.0 (not 0) at the last element.
        // A symmetric Tukey window tapers to 0 at both ends with a flat
        // interior. r = 0.25 ⇒ taper over the outer 0.125 fraction per side.
        let n = 64;
        let w = apodization_weights(n, "tukey25");
        assert!(w[0].abs() < 1e-12, "left endpoint {} != 0", w[0]);
        assert!(w[n - 1].abs() < 1e-12, "right endpoint {} != 0", w[n - 1]);
        // Mid-array is in the flat unit-gain region.
        assert!((w[n / 2] - 1.0).abs() < 1e-12, "centre {} != 1", w[n / 2]);
        // Symmetric about the array centre.
        for i in 0..n {
            assert!((w[i] - w[n - 1 - i]).abs() < 1e-12, "asymmetry at {i}");
        }
    }

    #[test]
    fn apodization_hann_matches_known_endpoints() {
        // Hann tapers to 0 at both ends; Hamming to 0.08; both peak at 1 mid-array.
        let hann = apodization_weights(65, "hann");
        assert!(hann[0].abs() < 1e-12 && hann[64].abs() < 1e-12);
        assert!((hann[32] - 1.0).abs() < 1e-12);
        let hamm = apodization_weights(65, "hamming");
        assert!((hamm[0] - 0.08).abs() < 1e-12 && (hamm[64] - 0.08).abs() < 1e-12);
    }

    #[test]
    fn bli_stencil_dc_preservation() {
        let ws = bli_stencil_weights(&[0.0, 0.25, 0.5, 0.75], 8);
        for w in &ws {
            let s: f64 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "sum={}", s);
        }
    }

    #[test]
    fn delay_law_max_is_zero() {
        // The element closest to the focus should have delay approaching 0
        let ex = vec![0.0];
        let ez = vec![0.0];
        let d = delay_law_focus_2d(&ex, &ez, 0.0, 0.0, SOUND_SPEED_WATER_SIM);
        assert!((d[0]).abs() < 1e-15);
    }

    #[test]
    fn linear_array_positions_centered() {
        let (x, z) = linear_array_positions(4, 1.0e-3);
        // 4 elements, pitch 1 mm → positions [-1.5, -0.5, 0.5, 1.5] mm
        assert!((x[0] + 1.5e-3).abs() < 1e-12);
        assert!((x[3] - 1.5e-3).abs() < 1e-12);
        // Centroid is zero.
        let sum: f64 = x.iter().sum();
        assert!(sum.abs() < 1e-12);
        assert!(z.iter().all(|&zi| zi == 0.0));
    }

    #[test]
    fn near_field_natural_focus_formula() {
        // D = 20 mm, f = 2 MHz, c = 1540 m/s → λ = 0.77 mm, N = D²/(4λ).
        let f = 2.0 * MHZ_TO_HZ;
        let c = SOUND_SPEED_WATER_SIM;
        let d = 20.0e-3;
        let n = near_field_distance(d, f, c);
        let lambda = c / f;
        let expected = d * d / (4.0 * lambda);
        assert!((n - expected).abs() < 1e-12);
        assert!(n > 0.0);
    }

    #[test]
    fn steering_focus_point_traces_natural_focus_arc() {
        // At fixed range R the focus stays on the arc of radius R.
        let r = 40.0e-3;
        for &theta in &[-0.5, 0.0, 0.3, 0.7] {
            let (xf, zf) = steering_focus_point(r, theta);
            let radius = (xf * xf + zf * zf).sqrt();
            assert!((radius - r).abs() < 1e-12, "radius drift at θ={theta}");
        }
        // On-axis steering → focus straight ahead.
        let (x0, z0) = steering_focus_point(r, 0.0);
        assert!(x0.abs() < 1e-12 && (z0 - r).abs() < 1e-12);
    }

    #[test]
    fn delay_law_steer_focuses_on_arc() {
        // A symmetric array steered on-axis must yield a symmetric delay law
        // peaking at the edges (focus straight ahead).
        let (ex, ez) = linear_array_positions(5, 1.0e-3);
        let c = SOUND_SPEED_WATER_SIM;
        let d = delay_law_steer_2d(&ex, &ez, 40.0e-3, 0.0, c);
        // Symmetry about the centre element.
        assert!((d[0] - d[4]).abs() < 1e-12);
        assert!((d[1] - d[3]).abs() < 1e-12);
        // Centre element is closest to the on-axis focus → largest delay; the
        // edge elements are farthest → smallest delay (=0).
        assert!(d[2] >= d[1] - 1e-15 && d[2] >= d[0] - 1e-15);
    }

    #[test]
    fn beam_pattern_magnitude_peaks_at_unity() {
        let k = 2.0 * PI * 2.0 * MHZ_TO_HZ / SOUND_SPEED_WATER_SIM;
        let theta: Vec<f64> = (0..361).map(|i| (i as f64 - 180.0) * PI / 360.0).collect();
        let mag = beam_pattern_magnitude(&theta, k, 0.385e-3, 64, 0.0, 0.5);
        let peak = mag.iter().cloned().fold(0.0_f64, f64::max);
        assert!((peak - 1.0).abs() < 1e-9);
        assert!(mag.iter().all(|&m| (0.0..=1.0 + 1e-9).contains(&m)));
    }

    #[test]
    fn multi_focus_delay_laws_shape_and_per_spot_minimum() {
        // 6 sub-spots, 8-element array → 6×8 row-major delay matrix.
        let (ex, ez) = linear_array_positions(8, 0.5e-3);
        let c = SOUND_SPEED_WATER_SIM;
        let n_spots = 6;
        let spot_x: Vec<f64> = (0..n_spots).map(|j| (j as f64 - 2.5) * 4.0e-3).collect();
        let spot_z = vec![40.0e-3; n_spots];
        let d = multi_focus_delay_laws_2d(&ex, &ez, &spot_x, &spot_z, c);
        assert_eq!(d.len(), n_spots * ex.len());
        // Each row is a valid focusing delay law: non-negative with min == 0.
        for j in 0..n_spots {
            let row = &d[j * ex.len()..(j + 1) * ex.len()];
            assert!(row.iter().all(|&t| t >= 0.0), "row {j} has negative delay");
            let rmin = row.iter().cloned().fold(f64::INFINITY, f64::min);
            assert!(rmin.abs() < 1e-15, "row {j} min delay = {rmin}, expected 0");
        }
    }

    #[test]
    fn focused_bowl_elements_lie_on_spherical_surface() {
        let focus = 30.0e-3;
        let roc = 30.0e-3;
        let elem = focused_bowl_element_positions_3d(3, 8, 10.0e-3, roc, focus);
        assert!(!elem.is_empty());
        for p in elem.chunks_exact(3) {
            let dx = p[0] - focus;
            let r = (dx * dx + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((r - roc).abs() < 1.0e-12, "surface radius {r}");
        }
    }

    #[test]
    fn steered_aperture_pressure_is_focus_normalized() {
        let elem = focused_bowl_element_positions_3d(4, 12, 10.0e-3, 30.0e-3, 30.0e-3);
        let focus = [30.0e-3, 1.0e-3, -0.5e-3];
        let delays = delay_law_focus_3d(&elem, focus, SOUND_SPEED_WATER_SIM);
        let weights = vec![1.0; delays.len()];
        let points = vec![
            focus[0],
            focus[1],
            focus[2],
            focus[0],
            focus[1] + 3.0e-3,
            focus[2],
        ];
        let p = steered_aperture_pressure_3d(
            &points,
            &elem,
            &weights,
            &delays,
            focus,
            1.0 * MHZ_TO_HZ,
            SOUND_SPEED_WATER_SIM,
            0.0,
            2.5e6,
        );
        assert_eq!(p.len(), 2);
        assert!((p[0] - 2.5e6).abs() / 2.5e6 < 1.0e-12);
        assert!(p[1].is_finite() && p[1] >= 0.0);
    }

    #[test]
    fn focused_bowl_profile_is_source_assembled_in_rust() {
        let r = [0.0, 1.0e-3, 2.0e-3];
        let profile = focused_bowl_steered_pressure_profile(
            &r,
            [30.0e-3, 0.0, 0.0],
            2.0e6,
            4,
            12,
            10.0e-3,
            30.0e-3,
            30.0e-3,
            1.0 * MHZ_TO_HZ,
            SOUND_SPEED_WATER_SIM,
            0.0,
        );
        assert_eq!(profile.len(), r.len());
        assert!((profile[0] - 1.0).abs() < 1.0e-12);
        assert!(profile.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn multi_focus_field_peaks_at_every_subspot() {
        // Phase-conjugation synthesis must place a focus at each sub-spot:
        // the field magnitude at each commanded sub-spot exceeds the field at a
        // control point midway between two sub-spots.
        let (ex, ez) = linear_array_positions(64, 0.385e-3);
        let c = SOUND_SPEED_WATER_SIM;
        let f = 2.0 * MHZ_TO_HZ;
        let spot_x = vec![-6.0e-3, 0.0, 6.0e-3];
        let spot_z = vec![40.0e-3, 40.0e-3, 40.0e-3];
        let amp = vec![1.0, 1.0, 1.0];

        // Evaluation grid: include the three sub-spots and two midpoints.
        let x_arr = vec![-6.0e-3, -3.0e-3, 0.0, 3.0e-3, 6.0e-3];
        let z_arr = vec![40.0e-3];
        let mag =
            multi_focus_field_magnitude_2d(&x_arr, &z_arr, &ex, &ez, &spot_x, &spot_z, &amp, f, c);
        // Layout is row-major NX×NZ with NZ=1 → mag[ix].
        let at_spot_left = mag[0];
        let at_mid_left = mag[1];
        let at_spot_centre = mag[2];
        let at_mid_right = mag[3];
        let at_spot_right = mag[4];
        assert!(at_spot_left > at_mid_left, "no focus at left sub-spot");
        assert!(at_spot_centre > at_mid_left, "centre below left midpoint");
        assert!(at_spot_centre > at_mid_right, "centre below right midpoint");
        assert!(at_spot_right > at_mid_right, "no focus at right sub-spot");
        // Field is peak-normalised to unit maximum.
        let peak = mag.iter().cloned().fold(0.0_f64, f64::max);
        assert!(peak <= 1.0 + 1e-9 && peak > 0.5);
    }

    #[test]
    fn aperiodic_layout_same_aperture_breaks_periodicity() {
        // The aperiodic layout keeps the same endpoints (hence the same
        // aperture) and the same element count, but the spacing is no longer
        // uniform.
        let n = 24;
        let aperture = 120.0e-3;
        let (xu, _) = linear_array_positions(n, aperture / (n as f64 - 1.0));
        let xa = linear_array_aperiodic_positions(n, aperture, 0.7);
        assert_eq!(xa.len(), n);
        // Endpoints anchored → identical aperture.
        assert!((xa[0] - xu[0]).abs() < 1e-12, "left endpoint moved");
        assert!(
            (xa[n - 1] - xu[n - 1]).abs() < 1e-12,
            "right endpoint moved"
        );
        // Spacing is uniform for the periodic array, non-uniform here.
        let gaps: Vec<f64> = xa.windows(2).map(|w| w[1] - w[0]).collect();
        let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let var = gaps.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / gaps.len() as f64;
        assert!(var > 1e-9, "aperiodic spacing variance too small: {var}");
    }

    #[test]
    fn beam_pattern_peaks_at_steering_angle() {
        // Steered to broadside, the beam pattern attains its maximum at the
        // steering angle, where the array factor is unity and the element
        // factor D(0) = 1.
        let n = 24;
        let aperture = 120.0e-3;
        let (x, _) = linear_array_positions(n, aperture / (n as f64 - 1.0));
        let c = SOUND_SPEED_WATER_SIM;
        let f = 0.22 * MHZ_TO_HZ;
        let k = 2.0 * PI * f / c;
        let ka = k * 3.0e-3;
        let nobs = 361usize;
        let obs: Vec<f64> = (0..nobs)
            .map(|i| -PI / 2.0 + PI * (i as f64) / ((nobs - 1) as f64))
            .collect();
        let pat = steered_beam_pattern_1d(&x, &obs, k, 0.0, ka);
        let peak = pat.iter().cloned().fold(0.0_f64, f64::max);
        // Broadside sample (centre) is the global peak ≈ 1.
        let centre = nobs / 2;
        assert!(
            (pat[centre] - peak).abs() < 1e-9,
            "steer angle not the peak"
        );
        assert!(
            (pat[centre] - 1.0).abs() < 1e-6,
            "peak != 1: {}",
            pat[centre]
        );
    }

    #[test]
    fn aperiodic_layout_suppresses_grating_lobes() {
        // At a fixed frequency, aperture and element count, steering a coarse-
        // pitch uniform array off-axis raises a coherent grating lobe; the
        // aperiodic layout scatters it, lowering the grating-lobe ratio. The
        // elements are small (broad directivity) so the element factor alone
        // does not suppress the grating lobe — only the layout does.
        let n = 48;
        let aperture = 270.0e-3;
        let c = SOUND_SPEED_WATER_SIM;
        let f = 0.22 * MHZ_TO_HZ;
        let k = 2.0 * PI * f / c;
        let ka = k * 0.8e-3;
        let (xu, _) = linear_array_positions(n, aperture / (n as f64 - 1.0));
        let xa = linear_array_aperiodic_positions(n, aperture, 1.0);

        // Observation grid and a single off-axis steer in the grating regime.
        let nobs = 1441usize;
        let obs: Vec<f64> = (0..nobs)
            .map(|i| -PI / 2.0 + PI * (i as f64) / ((nobs - 1) as f64))
            .collect();
        let steer = vec![30.0_f64.to_radians()];
        let halfwidth = 5.0_f64.to_radians();

        let glr_u = steering_grating_lobe_ratio_1d(&xu, &steer, &obs, k, ka, halfwidth)[0];
        let glr_a = steering_grating_lobe_ratio_1d(&xa, &steer, &obs, k, ka, halfwidth)[0];
        assert!(
            glr_a < glr_u,
            "aperiodic GLR {glr_a:.3} not below uniform GLR {glr_u:.3}"
        );
    }

    #[test]
    fn aperiodic_enlarges_safe_steering_envelope() {
        // The −6 dB grating-lobe-safe steering half-angle at a single
        // frequency is larger for the aperiodic (sparse) layout than for the
        // uniform (dense) one — the basis of Insightec's sparse 220 kHz
        // transducer enlarging the treatment envelope.
        let n = 48;
        let aperture = 270.0e-3;
        let c = SOUND_SPEED_WATER_SIM;
        let f = 0.22 * MHZ_TO_HZ;
        let k = 2.0 * PI * f / c;
        let ka = k * 0.8e-3;
        let (xu, _) = linear_array_positions(n, aperture / (n as f64 - 1.0));
        let xa = linear_array_aperiodic_positions(n, aperture, 1.0);

        let nobs = 1441usize;
        let obs: Vec<f64> = (0..nobs)
            .map(|i| -PI / 2.0 + PI * (i as f64) / ((nobs - 1) as f64))
            .collect();
        let nsteer = 121usize;
        let smax = 60.0_f64.to_radians();
        let steer: Vec<f64> = (0..nsteer)
            .map(|i| -smax + 2.0 * smax * (i as f64) / ((nsteer - 1) as f64))
            .collect();
        let halfwidth = 5.0_f64.to_radians();

        let glr_u = steering_grating_lobe_ratio_1d(&xu, &steer, &obs, k, ka, halfwidth);
        let glr_a = steering_grating_lobe_ratio_1d(&xa, &steer, &obs, k, ka, halfwidth);
        let ha_u = safe_steering_halfangle(&steer, &glr_u, 0.5);
        let ha_a = safe_steering_halfangle(&steer, &glr_a, 0.5);
        assert!(ha_u > 0.0 && ha_a > 0.0, "u={ha_u:.4} a={ha_a:.4}");
        assert!(
            ha_a > ha_u,
            "aperiodic safe half-angle {ha_a:.4} rad not larger than uniform {ha_u:.4} rad"
        );
    }
}
