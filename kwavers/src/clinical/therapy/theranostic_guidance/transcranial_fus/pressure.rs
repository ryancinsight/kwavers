use ndarray::{Array1, Array2, Array3};

use crate::core::constants::numerical::TWO_PI;
use crate::core::error::{KwaversError, KwaversResult};

/// Compute the Rayleigh-Sommerfeld pressure field on a 3-D grid.
///
/// Uses chunked evaluation to bound peak memory usage.
///
/// # Arguments
/// * `active` — bool mask selecting which elements contribute.
/// * `shape_nxnynz` — grid dimensions (nx, ny, nz).
/// * `spacing_m` — voxel spacing [m].
/// * `target_index_xyz` — grid index of the focus (origin of the coordinate system).
/// * `target_peak_pa` — desired peak pressure at the maximum point.
/// * `chunk_size` — number of grid points processed per iteration.
#[allow(clippy::too_many_arguments)]
pub fn rayleigh_pressure_field(
    element_positions: &Array2<f64>,
    phases_rad: &Array1<f64>,
    amplitude_weights: &Array1<f64>,
    active: &Array1<bool>,
    shape_nxnynz: [usize; 3],
    spacing_m: [f64; 3],
    target_index_xyz: [usize; 3],
    frequency_hz: f64,
    brain_c: f64,
    target_peak_pa: f64,
    chunk_size: usize,
) -> KwaversResult<Array3<f32>> {
    let pressure = rayleigh_pressure_field_unscaled(
        element_positions,
        phases_rad,
        amplitude_weights,
        active,
        shape_nxnynz,
        spacing_m,
        target_index_xyz,
        frequency_hz,
        brain_c,
        chunk_size,
    )?;
    let (_, scale) = pressure_peak_and_scale(&pressure, target_peak_pa)?;
    Ok(scale_pressure_field(pressure, scale))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rayleigh_pressure_field_unscaled(
    element_positions: &Array2<f64>,
    phases_rad: &Array1<f64>,
    amplitude_weights: &Array1<f64>,
    active: &Array1<bool>,
    shape_nxnynz: [usize; 3],
    spacing_m: [f64; 3],
    target_index_xyz: [usize; 3],
    frequency_hz: f64,
    brain_c: f64,
    chunk_size: usize,
) -> KwaversResult<Array3<f32>> {
    let n_elem = element_positions.nrows();
    if phases_rad.len() != n_elem || amplitude_weights.len() != n_elem || active.len() != n_elem {
        return Err(KwaversError::InvalidInput(
            "element_positions, phases_rad, amplitude_weights, and active must have matching length"
                .to_owned(),
        ));
    }
    let [nx, ny, nz] = shape_nxnynz;
    let total = nx * ny * nz;
    let k = TWO_PI * frequency_hz / brain_c;

    let weight_sum: f64 = (0..n_elem)
        .filter(|&i| active[i])
        .map(|i| amplitude_weights[i])
        .sum();
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "amplitude weight sum is non-positive; all elements may be inactive".to_owned(),
        ));
    }

    let active_x: Vec<f64> = (0..n_elem)
        .filter(|&i| active[i])
        .map(|i| element_positions[[i, 0]])
        .collect();
    let active_y: Vec<f64> = (0..n_elem)
        .filter(|&i| active[i])
        .map(|i| element_positions[[i, 1]])
        .collect();
    let active_z: Vec<f64> = (0..n_elem)
        .filter(|&i| active[i])
        .map(|i| element_positions[[i, 2]])
        .collect();
    let active_phase: Vec<f64> = (0..n_elem)
        .filter(|&i| active[i])
        .map(|i| phases_rad[i])
        .collect();
    let active_weight: Vec<f64> = (0..n_elem)
        .filter(|&i| active[i])
        .map(|i| amplitude_weights[i] / weight_sum)
        .collect();
    let n_active = active_x.len();

    let mut pressure_flat = vec![0.0_f32; total];
    let chunk = chunk_size.max(1);

    for chunk_start in (0..total).step_by(chunk) {
        let chunk_end = (chunk_start + chunk).min(total);
        for (offset, pressure_value) in pressure_flat[chunk_start..chunk_end].iter_mut().enumerate()
        {
            let flat_idx = chunk_start + offset;
            let ix = flat_idx / (ny * nz);
            let iy = (flat_idx % (ny * nz)) / nz;
            let iz = flat_idx % nz;
            let gx = (ix as f64 - target_index_xyz[0] as f64) * spacing_m[0];
            let gy = (iy as f64 - target_index_xyz[1] as f64) * spacing_m[1];
            let gz = (iz as f64 - target_index_xyz[2] as f64) * spacing_m[2];

            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for ei in 0..n_active {
                let dx = gx - active_x[ei];
                let dy = gy - active_y[ei];
                let dz = gz - active_z[ei];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1.0e-6);
                let phase = k * dist + active_phase[ei];
                let amplitude = active_weight[ei] / dist;
                re += amplitude * phase.cos();
                im += amplitude * phase.sin();
            }
            *pressure_value = ((re * re + im * im).sqrt()) as f32;
        }
    }

    let peak = pressure_flat.iter().copied().fold(0.0_f32, f32::max);
    if !peak.is_finite() || peak <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "Rayleigh field produced non-positive peak pressure".to_owned(),
        ));
    }

    Array3::from_shape_vec((nx, ny, nz), pressure_flat)
        .map_err(|e| KwaversError::InvalidInput(format!("shape error: {e}")))
}

pub(super) fn pressure_peak_and_scale(
    pressure_pa: &Array3<f32>,
    target_peak_pa: f64,
) -> KwaversResult<(f32, f32)> {
    if !target_peak_pa.is_finite() || target_peak_pa <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "target_peak_pa must be positive and finite".to_owned(),
        ));
    }
    let peak = pressure_pa.iter().copied().fold(0.0_f32, f32::max);
    if !peak.is_finite() || peak <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "Rayleigh field produced non-positive peak pressure".to_owned(),
        ));
    }
    Ok((peak, target_peak_pa as f32 / peak))
}

pub(super) fn scale_pressure_field(mut pressure_pa: Array3<f32>, scale: f32) -> Array3<f32> {
    for value in &mut pressure_pa {
        *value *= scale;
    }
    pressure_pa
}
