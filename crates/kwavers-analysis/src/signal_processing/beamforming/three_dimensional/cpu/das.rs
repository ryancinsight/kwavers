//! CPU Delay-and-Sum (DAS) beamformer for 3D volumetric ultrasound.
//!
//! ## Theorem: Coherent DAS Image Formation
//!
//! Let M receive elements be located at positions {**r**_i}_{i=1}^{M}, and let a
//! voxel be located at **r**_v. For a plane-wave transmit event, element i
//! receives the echo from **r**_v with one-way time-of-flight
//!
//! ```text
//!   τ_i(r_v) = |r_v − r_i| / c
//! ```
//!
//! The DAS estimate of the back-scattered pressure at **r**_v from N_f compounded
//! frames is
//!
//! ```text
//!   I_DAS(r_v) = (1/N_f) Σ_{f=1}^{N_f} Σ_{i=1}^{M} w_i · x_i^f(n_i)
//! ```
//!
//! where `n_i = τ_i · f_s` (fractional delay, linearly interpolated),
//! w_i ∈ [0,1] is the apodization weight, and x_i^f is the i-th channel
//! RF signal from frame f.
//!
//! ## Proof of Coherence
//!
//! Under the far-field assumption |**r**_v − **r**_i| ≫ λ the wave from **r**_v
//! arrives at element i as a spherical wavefront with the stated delay. Applying
//! the inverse delay and summing over elements constructively interferes the
//! mainlobe while incoherently averaging sidelobes. The coherent gain is M (linear),
//! or 20 log10 M dB.  Apodization trades mainlobe width for sidelobe suppression.
//!
//! ## References
//! - Thomenius K.E. (1996): "Evolution of ultrasound beamformers."
//!   Proc. IEEE Ultrasonics Symposium, pp. 1615–1622.
//! - Jeong M.K., Kwon S. (2013): "A comparison study of beamforming techniques
//!   for 3D ultrasound." *J. Med. Ultrason.* 40, 395–408.

use leto::{Array3, Array4};
use moirai_parallel::{map_collect_index_with, Adaptive};

use crate::signal_processing::beamforming::three_dimensional::apodization::create_apodization_weights;
use crate::signal_processing::beamforming::three_dimensional::config::{
    Beamforming3dApodizationWindow, BeamformingConfig3D,
};
use kwavers_core::error::{KwaversError, KwaversResult};

/// Execute CPU delay-and-sum for a single 3D volume.
///
/// # Arguments
/// - `rf_data`     : RF array, shape `[frames, channels, samples, 1]`
/// - `config`      : Beamforming parameters
/// - `apodization` : Window function applied per-element
///
/// # Returns
/// 3D volume of DAS amplitude values, shape `config.volume_dims`,
/// coherently averaged over all frames (plane-wave compounding).
///
/// # Errors
/// Returns [`KwaversError::InvalidInput`] if the channel count does not match
/// `config.num_elements_3d`.
// Authoritative CPU DAS kernel: the active production path under default
// features and the differential-test reference baseline. Under `gpu`, the only
// non-test caller (`processing::algorithms`) is `cfg(not(gpu))`, so the lib
// build sees no caller; tests still use it. Keep it, silence dead_code on gpu.
#[cfg_attr(feature = "gpu", allow(dead_code))]
pub fn delay_and_sum_cpu(
    rf_data: &Array4<f32>,
    config: &BeamformingConfig3D,
    apodization: &Beamforming3dApodizationWindow,
) -> KwaversResult<Array3<f32>> {
    let [frames, channels, samples, _] = rf_data.shape();
    let (vol_x, vol_y, vol_z) = config.volume_dims;
    let (nel_x, nel_y, nel_z) = config.num_elements_3d;
    let expected_channels = nel_x * nel_y * nel_z;

    if channels != expected_channels {
        return Err(KwaversError::InvalidInput(format!(
            "DAS CPU: channel count {channels} ≠ element count {expected_channels} \
             ({nel_x}×{nel_y}×{nel_z})"
        )));
    }
    if samples == 0 {
        return Err(KwaversError::InvalidInput(
            "DAS CPU: RF data must have at least one sample".to_owned(),
        ));
    }

    let fs = config.sampling_frequency as f32;
    let c = config.sound_speed as f32;
    let (dx, dy, dz) = (
        config.voxel_spacing.0 as f32,
        config.voxel_spacing.1 as f32,
        config.voxel_spacing.2 as f32,
    );
    let (sx, sy, sz) = (
        config.element_spacing_3d.0 as f32,
        config.element_spacing_3d.1 as f32,
        config.element_spacing_3d.2 as f32,
    );

    // Element positions, centred at the array origin.
    // Element (ex, ey, ez) → physical (x, y, z).
    let elem_pos: Vec<[f32; 3]> = (0..nel_x)
        .flat_map(|ex| {
            (0..nel_y).flat_map(move |ey| {
                (0..nel_z).map(move |ez| {
                    [
                        (nel_x as f32 - 1.0).mul_add(-0.5, ex as f32) * sx,
                        (nel_y as f32 - 1.0).mul_add(-0.5, ey as f32) * sy,
                        (nel_z as f32 - 1.0).mul_add(-0.5, ez as f32) * sz,
                    ]
                })
            })
        })
        .collect();

    // Apodization weights — flat, indexed by channel (same order as elem_pos).
    let apod_3d = create_apodization_weights((nel_x, nel_y, nel_z), apodization);
    let apod_flat: Vec<f32> = apod_3d.iter().copied().collect();

    // Safe RF accessor: returns the linearly interpolated value at fractional
    // sample index `tau_s` for channel `ch` in `frame`.  Returns 0.0 if the
    // delay falls outside the recorded window (no wrap-around — echoes from
    // very shallow voxels or elements at the boundary are silently zeroed).
    let rf_get = |frame: usize, ch: usize, tau_s: f32| -> f32 {
        if tau_s < 0.0 {
            return 0.0;
        }
        let n0 = tau_s as usize;
        let alpha = tau_s - n0 as f32;
        if n0 + 1 >= samples {
            return 0.0;
        }
        let s0 = rf_data[[frame, ch, n0, 0]];
        let s1 = rf_data[[frame, ch, n0 + 1, 0]];
        alpha.mul_add(s1 - s0, s0)
    };

    // Schedule independent voxel evaluations through the Atlas provider seam.
    let n_voxels = vol_x * vol_y * vol_z;
    let flat: Vec<f32> = map_collect_index_with::<Adaptive, _, _>(n_voxels, |v_idx| {
        let vx = v_idx / (vol_y * vol_z);
        let vy = (v_idx / vol_z) % vol_y;
        let vz = v_idx % vol_z;

        // Voxel centre in physical coordinates (origin at array centre).
        let pv = [
            (vol_x as f32 - 1.0).mul_add(-0.5, vx as f32) * dx,
            (vol_y as f32 - 1.0).mul_add(-0.5, vy as f32) * dy,
            (vol_z as f32 - 1.0).mul_add(-0.5, vz as f32) * dz,
        ];

        let mut coherent_sum = 0.0_f32;

        for frame in 0..frames {
            for (ch, ep) in elem_pos.iter().enumerate() {
                // Euclidean distance from element to voxel.
                let dx_ = pv[0] - ep[0];
                let dy_ = pv[1] - ep[1];
                let dz_ = pv[2] - ep[2];
                let dist = (dx_ * dx_ + dy_ * dy_ + dz_ * dz_).sqrt();

                // One-way receive delay (plane-wave transmit assumed).
                let tau_s = dist / c * fs;
                coherent_sum += apod_flat[ch] * rf_get(frame, ch, tau_s);
            }
        }

        // Coherent compounding average.
        coherent_sum / frames.max(1) as f32
    });

    Array3::from_shape_vec((vol_x, vol_y, vol_z), flat)
        .map_err(|e| KwaversError::InvalidInput(format!("DAS CPU: output volume shape error: {e}")))
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal_processing::beamforming::three_dimensional::config::{
        Beamforming3dApodizationWindow, BeamformingConfig3D,
    };
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;
    use leto::Array4;

    fn make_config(
        nel: (usize, usize, usize),
        elem_spacing: (f64, f64, f64),
        vol: (usize, usize, usize),
        sound_speed: f64,
        sampling_frequency: f64,
    ) -> BeamformingConfig3D {
        BeamformingConfig3D {
            num_elements_3d: nel,
            element_spacing_3d: elem_spacing,
            volume_dims: vol,
            voxel_spacing: (1e-3, 1e-3, 1e-3),
            sound_speed,
            sampling_frequency,
            ..BeamformingConfig3D::default()
        }
    }

    /// ## Theorem: Zero-delay passthrough
    ///
    /// For a 1-element array with the element at the origin and a single voxel
    /// also at the origin, the one-way receive delay τ = dist(0,0) / c = 0.
    /// With Rectangular apodization (w = 1), the DAS output equals rf[0,0,0,0].
    ///
    /// ## Proof
    /// dist(origin, origin) = 0 → τ_s = 0 · fs / c = 0.
    /// `rf_get(0, 0, 0.0)` returns rf[0,0,0,0] (alpha = 0 → no interpolation).
    /// coherent_sum = 1.0 × rf[0,0,0,0].  Division by frames = 1 gives
    /// output = rf[0,0,0,0].
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn das_single_element_zero_delay_passthrough() {
        let config = make_config(
            (1, 1, 1),
            (1e-3, 1e-3, 1e-3),
            (1, 1, 1),
            SOUND_SPEED_WATER_SIM,
            MHZ_TO_HZ,
        );
        let mut rf = Array4::<f32>::zeros((1, 1, 4, 1));
        rf[[0, 0, 0, 0]] = 7.0;

        let vol =
            delay_and_sum_cpu(&rf, &config, &Beamforming3dApodizationWindow::Rectangular).unwrap();
        assert!(
            (vol[[0, 0, 0]] - 7.0_f32).abs() < 1e-5_f32,
            "DAS zero-delay passthrough: expected 7.0, got {}",
            vol[[0, 0, 0]]
        );
    }

    /// ## Theorem: Channel-count mismatch rejected
    ///
    /// `delay_and_sum_cpu` returns `KwaversError::InvalidInput` when the RF
    /// channel axis ≠ nel_x × nel_y × nel_z.
    ///
    /// ## Proof
    /// The guard at the top of `delay_and_sum_cpu` checks `channels ==
    /// expected_channels` and short-circuits with an error otherwise.
    /// # Panics
    /// - Panics with `"expected InvalidInput, got {other:?}"`.
    ///
    #[test]
    fn das_channel_mismatch_returns_error() {
        let config = make_config(
            (1, 1, 1), // expects 1 channel
            (1e-3, 1e-3, 1e-3),
            (1, 1, 1),
            SOUND_SPEED_WATER_SIM,
            MHZ_TO_HZ,
        );
        // RF supplies 5 channels; config expects 1.
        let rf = Array4::<f32>::zeros((1, 5, 4, 1));
        let result = delay_and_sum_cpu(&rf, &config, &Beamforming3dApodizationWindow::Rectangular);
        assert!(result.is_err(), "DAS must reject channel count mismatch");
        match result.unwrap_err() {
            KwaversError::InvalidInput(msg) => {
                assert!(
                    msg.contains("channel") || msg.contains("element"),
                    "error must reference channel/element mismatch; got: {msg}"
                );
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    /// ## Theorem: Linear coherent gain
    ///
    /// With M elements all co-located at the origin and a voxel at the origin,
    /// every element contributes sample[0] with zero delay.  The DAS output equals
    /// M × rf[0,ch,0,0] / frames = M (coherent gain M, linear in element count).
    ///
    /// ## Proof
    /// element_spacing = 0 → all element positions = (0,0,0).
    /// τ_i = dist(origin, origin) / c = 0 for all i.
    /// coherent_sum = Σ_{i=0}^{M-1} 1.0 × rf[0,i,0,0] = M × 1.0 = M.
    /// output = M / frames = M.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn das_coherent_gain_co_located_elements() {
        const M: usize = 4;
        let config = make_config(
            (1, M, 1),
            (0.0, 0.0, 0.0), // all elements co-located at origin
            (1, 1, 1),
            SOUND_SPEED_WATER_SIM,
            MHZ_TO_HZ,
        );
        let mut rf = Array4::<f32>::zeros((1, M, 4, 1));
        for ch in 0..M {
            rf[[0, ch, 0, 0]] = 1.0;
        }
        let vol =
            delay_and_sum_cpu(&rf, &config, &Beamforming3dApodizationWindow::Rectangular).unwrap();
        let expected = M as f32;
        assert!(
            (vol[[0, 0, 0]] - expected).abs() < 1e-5_f32,
            "DAS coherent gain: expected {expected}, got {}",
            vol[[0, 0, 0]]
        );
    }

    /// ## Theorem: Receive delay is geometrically correct
    ///
    /// For a 2-element z-axis array (nel_z = 2) with spacing sz, the receive
    /// delay to the origin voxel is τ = (sz/2) / c × fs.  Choosing sz such that
    /// τ is an exact integer validates that the fractional-delay interpolator
    /// reads from the correct sample and that the DAS output equals the coherent
    /// sum of two aligned pulses.
    ///
    /// ## Proof
    /// Element positions (centred): ez=0 → z = −sz/2; ez=1 → z = +sz/2.
    /// dist(origin, (0,0,−sz/2)) = sz/2.
    /// τ_s = (sz/2) / c × fs.  Set sz = 3 mm, c = 1500 m/s, fs = 1 MHz:
    /// τ_s = 1.5e-3 / 1500 × 1e6 = 1.0 (exact integer).
    /// RF: ch0[1] = ch1[1] = 5.0; all other samples = 0.
    /// DAS output = (5.0 + 5.0) / 1 frame = 10.0.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn das_receive_delay_is_geometrically_correct() {
        let sz = 3e-3_f64; // 3 mm element z-spacing
        let c = SOUND_SPEED_WATER_SIM;
        let fs = MHZ_TO_HZ;
        // Precondition: τ is exactly 1.0 sample.
        let tau_exact = 0.5 * sz / c * fs;
        assert!(
            (tau_exact - 1.0).abs() < 1e-12,
            "test precondition: τ must equal 1.0 sample; got {tau_exact}"
        );

        let config = make_config((1, 1, 2), (1e-3, 1e-3, sz), (1, 1, 1), c, fs);
        let mut rf = Array4::<f32>::zeros((1, 2, 4, 1));
        // Pulse at sample index 1 in both channels.
        rf[[0, 0, 1, 0]] = 5.0;
        rf[[0, 1, 1, 0]] = 5.0;

        let vol =
            delay_and_sum_cpu(&rf, &config, &Beamforming3dApodizationWindow::Rectangular).unwrap();
        assert!(
            (vol[[0, 0, 0]] - 10.0_f32).abs() < 1e-4_f32,
            "DAS delay geometry: expected 10.0, got {}",
            vol[[0, 0, 0]]
        );
    }
}
