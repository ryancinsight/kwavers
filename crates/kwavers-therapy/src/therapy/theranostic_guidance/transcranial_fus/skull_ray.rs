use leto::{Array1, Array2, Array3};

/// Per-element skull-path correction: `(phases_rad, delays_s, skull_lengths_m,
/// amplitude_weights)`, one value per transducer element.
type SkullPathCorrection = (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>);

use kwavers_core::constants::acoustic_parameters::DB_TO_NP;
use kwavers_core::constants::ct_acoustics::{
    DENSITY_SKULL_CORTICAL_RANGE, DENSITY_SKULL_MIN, HU_BONE_THRESHOLD, HU_SKULL_RANGE,
};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::constants::tissue_acoustics::{
    ACOUSTIC_ABSORPTION_BRAIN, ACOUSTIC_ABSORPTION_SKULL_MIN, ACOUSTIC_ABSORPTION_SKULL_RANGE,
    DENSITY_BRAIN,
};
use kwavers_core::error::{KwaversError, KwaversResult};

/// Map a HU value to (sound_speed_m_s, density_kg_m3, attenuation_np_m).
///
/// Below `HU_BONE_THRESHOLD` (300 HU) the voxel is treated as brain tissue;
/// above it, bone fraction linearly interpolates between brain and cortical-bone
/// properties following Aubry et al. (2003) J. Acoust. Soc. Am. 113(1):84–93.
pub fn acoustic_properties_from_hu(
    hu: f64,
    frequency_hz: f64,
    brain_c: f64,
    skull_c: f64,
) -> (f64, f64, f64) {
    if hu <= HU_BONE_THRESHOLD {
        let alpha_np_m = ACOUSTIC_ABSORPTION_BRAIN * 100.0 * DB_TO_NP * (frequency_hz / MHZ_TO_HZ);
        return (brain_c, DENSITY_BRAIN, alpha_np_m);
    }
    let bone_fraction = ((hu - HU_BONE_THRESHOLD) / HU_SKULL_RANGE).clamp(0.0, 1.0);
    let density = DENSITY_SKULL_MIN + DENSITY_SKULL_CORTICAL_RANGE * bone_fraction;
    let c = brain_c + (skull_c - brain_c) * bone_fraction;
    let alpha_db_cm_mhz =
        ACOUSTIC_ABSORPTION_SKULL_MIN + ACOUSTIC_ABSORPTION_SKULL_RANGE * bone_fraction;
    let alpha_np_m = alpha_db_cm_mhz * 100.0 * DB_TO_NP * (frequency_hz / MHZ_TO_HZ);
    (c, density, alpha_np_m)
}

/// Compute per-element skull-path phase correction by HU-based ray tracing.
///
/// # Returns
/// `(phases_rad, delays_s, skull_lengths_m, amplitude_weights)`
pub(super) struct SkullPathPhaseCorrectionInput<'a> {
    pub(super) ct_hu: &'a Array3<f64>,
    pub(super) spacing_m: [f64; 3],
    pub(super) target_index_xyz: [usize; 3],
    pub(super) element_positions: &'a Array2<f64>,
    pub(super) frequency_hz: f64,
    pub(super) brain_c: f64,
    pub(super) skull_c: f64,
    pub(super) skull_mask: &'a Array3<bool>,
    pub(super) samples_per_ray: usize,
}

pub(super) fn skull_path_phase_correction(
    input: SkullPathPhaseCorrectionInput<'_>,
) -> KwaversResult<SkullPathCorrection> {
    let SkullPathPhaseCorrectionInput {
        ct_hu,
        spacing_m,
        target_index_xyz,
        element_positions,
        frequency_hz,
        brain_c,
        skull_c,
        skull_mask,
        samples_per_ray,
    } = input;
    let [nx, ny, nz] = ct_hu.shape();
    if element_positions.shape()[1] != 3 {
        return Err(KwaversError::InvalidInput(
            "element_positions must have 3 columns".to_owned(),
        ));
    }
    let n_elem = element_positions.shape()[0];
    let target_m = [0.0_f64, 0.0_f64, 0.0_f64];
    let samples = samples_per_ray.max(2);
    let segment_m = |start: [f64; 3], end: [f64; 3]| -> f64 {
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let dz = end[2] - start[2];
        (dx * dx + dy * dy + dz * dz).sqrt() / (samples - 1) as f64
    };

    let mut delays = Array1::<f64>::zeros(n_elem);
    let mut skull_lengths = Array1::<f64>::zeros(n_elem);
    let mut amplitudes = Array1::<f64>::ones(n_elem);

    for elem_idx in 0..n_elem {
        let ex = element_positions[[elem_idx, 0]];
        let ey = element_positions[[elem_idx, 1]];
        let ez = element_positions[[elem_idx, 2]];
        let start = [ex, ey, ez];
        let end = target_m;
        let seg = segment_m(start, end);
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let dz = end[2] - start[2];

        let mut delay_s = 0.0_f64;
        let mut skull_len = 0.0_f64;
        let mut attenuation_np = 0.0_f64;
        let mut transmission = 1.0_f64;
        let z_brain = DENSITY_BRAIN * brain_c;
        let mut prev_impedance = z_brain;

        for sample_idx in 0..samples {
            let t = sample_idx as f64 / (samples - 1) as f64;
            let px = start[0] + t * dx;
            let py = start[1] + t * dy;
            let pz = start[2] + t * dz;

            let ix_f = px / spacing_m[0] + target_index_xyz[0] as f64;
            let iy_f = py / spacing_m[1] + target_index_xyz[1] as f64;
            let iz_f = pz / spacing_m[2] + target_index_xyz[2] as f64;
            let ix = ix_f.round() as isize;
            let iy = iy_f.round() as isize;
            let iz = iz_f.round() as isize;

            let (sound_speed, density, alpha, in_skull) = if ix >= 0
                && iy >= 0
                && iz >= 0
                && (ix as usize) < nx
                && (iy as usize) < ny
                && (iz as usize) < nz
            {
                let ix = ix as usize;
                let iy = iy as usize;
                let iz = iz as usize;
                let hu = ct_hu[[ix, iy, iz]];
                let in_sk = skull_mask[[ix, iy, iz]];
                let (c, rho, a) = acoustic_properties_from_hu(hu, frequency_hz, brain_c, skull_c);
                (c, rho, a, in_sk)
            } else {
                (brain_c, DENSITY_BRAIN, 0.0, false)
            };

            let impedance = density * sound_speed;
            let rel_diff =
                (impedance - prev_impedance).abs() / prev_impedance.max(impedance).max(1.0);
            if rel_diff > 0.05 {
                let intensity_t =
                    4.0 * prev_impedance * impedance / (prev_impedance + impedance).powi(2);
                transmission *= intensity_t.clamp(0.0, 1.0).sqrt();
            }
            delay_s += seg / sound_speed;
            attenuation_np += alpha * seg;
            if in_skull {
                skull_len += seg;
            }
            prev_impedance = impedance;
        }

        delays[elem_idx] = delay_s;
        skull_lengths[elem_idx] = skull_len;
        amplitudes[elem_idx] = transmission * (-attenuation_np).exp();
    }

    let mean_delay = delays.iter().sum::<f64>() / n_elem as f64;
    let phases = Array1::from_shape_fn(n_elem, |i| {
        let relative = delays[i] - mean_delay;
        let raw = -TWO_PI * frequency_hz * relative;
        raw.sin().atan2(raw.cos())
    });

    Ok((phases, delays, skull_lengths, amplitudes))
}
