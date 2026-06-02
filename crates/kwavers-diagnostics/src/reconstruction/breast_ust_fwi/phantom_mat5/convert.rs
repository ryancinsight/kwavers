//! Published MRI-to-sound-speed mapping for Ali et al. 2025.

use super::BreastUstMriBreastSide;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::collections::VecDeque;

const MRI_EXTENT_X_MM: f64 = 340.0;
const MRI_EXTENT_Y_MM: f64 = 340.0;
const MRI_EXTENT_Z_MM: f64 = 158.4;
const TISSUE_MAX_SOUND_SPEED_M_S: f64 = 1750.0;
const BREAST_RADIUS_MAX_MM: f64 = 80.0;
/// Published MRI-to-sound-speed mapping lower bound for the left breast [m/s].
///
/// Ali et al. (2025) calibration constant for left-breast MRI intensity → sound-speed
/// linear mapping. The intensity range [min, max] within the breast mask is linearly
/// mapped to [c_min, TISSUE_MAX_SOUND_SPEED_M_S].
///
/// Reference: Ali R et al. (2025). *IEEE Trans. Med. Imaging* (in press).
const BREAST_LEFT_SOUND_SPEED_MIN_M_S: f64 = 1400.0;
/// Published MRI-to-sound-speed mapping lower bound for the right breast [m/s].
///
/// Calibration value for the right breast in the Ali et al. (2025) model;
/// slightly lower than left due to observed intensity offset between orientations.
///
/// Reference: Ali R et al. (2025). *IEEE Trans. Med. Imaging* (in press).
const BREAST_RIGHT_SOUND_SPEED_MIN_M_S: f64 = 1350.0;

pub(super) struct BreastMriSoundSpeedMapConfig {
    pub output_shape: [usize; 3],
    pub grid_spacing_m: f64,
    pub breast_side: BreastUstMriBreastSide,
    pub tissue_threshold: f64,
}

pub(super) fn mri_to_sound_speed(
    mri_dims: [usize; 3],
    mri_values: &[f64],
    config: BreastMriSoundSpeedMapConfig,
) -> KwaversResult<Array3<f64>> {
    let expected = mri_dims.iter().product::<usize>();
    if mri_values.len() != expected {
        return Err(KwaversError::DimensionMismatch(format!(
            "MRI value count {} does not match dims {:?}",
            mri_values.len(),
            mri_dims
        )));
    }
    let breast_seg = interpolate_rotated_mri(mri_dims, mri_values, &config)?;
    let mut tissue = threshold_and_fill(&breast_seg, config.tissue_threshold);
    apply_radius_mask(&mut tissue, config.output_shape, config.grid_spacing_m);
    let (min_intensity, max_intensity) = tissue_intensity_bounds(&breast_seg, &tissue)?;
    let c_min = match config.breast_side {
        BreastUstMriBreastSide::Left => BREAST_LEFT_SOUND_SPEED_MIN_M_S,
        BreastUstMriBreastSide::Right => BREAST_RIGHT_SOUND_SPEED_MIN_M_S,
    };
    let scale = (TISSUE_MAX_SOUND_SPEED_M_S - c_min) / (max_intensity - min_intensity);
    Ok(Array3::from_shape_fn(
        (
            config.output_shape[0],
            config.output_shape[1],
            config.output_shape[2],
        ),
        |(i, j, k)| {
            if tissue[[i, j, k]] {
                c_min + (breast_seg[[i, j, k]] - min_intensity) * scale
            } else {
                SOUND_SPEED_WATER_SIM
            }
        },
    ))
}

fn interpolate_rotated_mri(
    mri_dims: [usize; 3],
    mri_values: &[f64],
    config: &BreastMriSoundSpeedMapConfig,
) -> KwaversResult<Array3<f64>> {
    let transform = PublishedTransform::for_side(config.breast_side, mri_dims);
    let spacing_mm = config.grid_spacing_m * 1000.0;
    Ok(Array3::from_shape_fn(
        (
            config.output_shape[0],
            config.output_shape[1],
            config.output_shape[2],
        ),
        |(i, j, k)| {
            let x_mm = centered_coord(i, config.output_shape[0], spacing_mm);
            let y_mm = centered_coord(j, config.output_shape[1], spacing_mm);
            let z_mm = centered_coord(k, config.output_shape[2], spacing_mm);
            let [xr, yr, zr] = transform.rotate([x_mm, y_mm, z_mm]);
            cubic_sample_mri(
                mri_dims,
                mri_values,
                [xr + transform.xc, yr + transform.yc, zr + transform.zc],
            )
            .unwrap_or(0.0)
        },
    ))
}

fn centered_coord(index: usize, len: usize, spacing_mm: f64) -> f64 {
    (index as f64 - 0.5 * (len.saturating_sub(1)) as f64) * spacing_mm
}

struct PublishedTransform {
    matrix: [[f64; 3]; 3],
    xc: f64,
    yc: f64,
    zc: f64,
}

impl PublishedTransform {
    fn for_side(side: BreastUstMriBreastSide, dims: [usize; 3]) -> Self {
        let (theta, phi, center_idx) = match side {
            BreastUstMriBreastSide::Left => (100.0_f64, 90.0_f64, [330usize, 125usize, 75usize]),
            BreastUstMriBreastSide::Right => (80.0_f64, 80.0_f64, [120usize, 130usize, 70usize]),
        };
        let theta = theta.to_radians();
        let phi = phi.to_radians();
        let matrix = [
            [
                theta.cos() * phi.cos(),
                -theta.sin(),
                theta.cos() * phi.sin(),
            ],
            [
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ],
            [-phi.sin(), 0.0, phi.cos()],
        ];
        Self {
            matrix,
            xc: mri_axis_coord(center_idx[0] - 1, dims[0], MRI_EXTENT_X_MM),
            yc: mri_axis_coord(center_idx[1] - 1, dims[1], MRI_EXTENT_Y_MM),
            zc: mri_axis_coord(center_idx[2] - 1, dims[2], MRI_EXTENT_Z_MM),
        }
    }

    fn rotate(&self, point: [f64; 3]) -> [f64; 3] {
        [
            self.matrix[0][0] * point[0]
                + self.matrix[0][1] * point[1]
                + self.matrix[0][2] * point[2],
            self.matrix[1][0] * point[0]
                + self.matrix[1][1] * point[1]
                + self.matrix[1][2] * point[2],
            self.matrix[2][0] * point[0]
                + self.matrix[2][1] * point[1]
                + self.matrix[2][2] * point[2],
        ]
    }
}

fn mri_axis_coord(index: usize, len: usize, extent_mm: f64) -> f64 {
    let spacing = extent_mm / len as f64;
    (index as f64 - 0.5 * (len.saturating_sub(1)) as f64) * spacing
}

fn mri_axis_index(coord_mm: f64, len: usize, extent_mm: f64) -> f64 {
    let spacing = extent_mm / len as f64;
    coord_mm / spacing + 0.5 * (len.saturating_sub(1)) as f64
}

fn cubic_sample_mri(dims: [usize; 3], values: &[f64], point_mm: [f64; 3]) -> Option<f64> {
    let x = mri_axis_index(point_mm[0], dims[0], MRI_EXTENT_X_MM);
    let y = mri_axis_index(point_mm[1], dims[1], MRI_EXTENT_Y_MM);
    let z = mri_axis_index(point_mm[2], dims[2], MRI_EXTENT_Z_MM);
    if x < -1.0
        || y < -1.0
        || z < -1.0
        || x > dims[0] as f64
        || y > dims[1] as f64
        || z > dims[2] as f64
    {
        return None;
    }
    let ix = x.floor() as isize;
    let iy = y.floor() as isize;
    let iz = z.floor() as isize;
    let fx = x - ix as f64;
    let fy = y - iy as f64;
    let fz = z - iz as f64;
    let mut acc = 0.0;
    for dz in -1..=2 {
        let wz = cubic_weight(fz - dz as f64);
        for dy in -1..=2 {
            let wy = cubic_weight(fy - dy as f64);
            for dx in -1..=2 {
                let wx = cubic_weight(fx - dx as f64);
                acc += wx * wy * wz * mri_value(dims, values, ix + dx, iy + dy, iz + dz);
            }
        }
    }
    Some(acc.max(0.0))
}

fn cubic_weight(x: f64) -> f64 {
    let a = -0.5;
    let x = x.abs();
    if x <= 1.0 {
        (a + 2.0) * x.powi(3) - (a + 3.0) * x.powi(2) + 1.0
    } else if x < 2.0 {
        a * x.powi(3) - 5.0 * a * x.powi(2) + 8.0 * a * x - 4.0 * a
    } else {
        0.0
    }
}

fn mri_value(dims: [usize; 3], values: &[f64], x: isize, y: isize, z: isize) -> f64 {
    if x < 0 || y < 0 || z < 0 {
        return 0.0;
    }
    let (x, y, z) = (x as usize, y as usize, z as usize);
    if x >= dims[0] || y >= dims[1] || z >= dims[2] {
        return 0.0;
    }
    values[x + dims[0] * (y + dims[1] * z)]
}

fn threshold_and_fill(volume: &Array3<f64>, threshold: f64) -> Array3<bool> {
    let dims = volume.dim();
    let mut mask = Array3::from_shape_fn(dims, |idx| volume[idx] > threshold);
    for k in 0..dims.2 {
        fill_holes_in_slice(&mut mask, k);
    }
    mask
}

fn fill_holes_in_slice(mask: &mut Array3<bool>, k: usize) {
    let (nx, ny, _) = mask.dim();
    let mut exterior = vec![false; nx * ny];
    let mut queue = VecDeque::new();
    for i in 0..nx {
        enqueue_background(mask, &mut exterior, &mut queue, i, 0, k);
        enqueue_background(mask, &mut exterior, &mut queue, i, ny - 1, k);
    }
    for j in 0..ny {
        enqueue_background(mask, &mut exterior, &mut queue, 0, j, k);
        enqueue_background(mask, &mut exterior, &mut queue, nx - 1, j, k);
    }
    while let Some((i, j)) = queue.pop_front() {
        for (ni, nj) in neighbors(i, j, nx, ny) {
            enqueue_background(mask, &mut exterior, &mut queue, ni, nj, k);
        }
    }
    for i in 0..nx {
        for j in 0..ny {
            if !mask[[i, j, k]] && !exterior[i + nx * j] {
                mask[[i, j, k]] = true;
            }
        }
    }
}

fn enqueue_background(
    mask: &Array3<bool>,
    exterior: &mut [bool],
    queue: &mut VecDeque<(usize, usize)>,
    i: usize,
    j: usize,
    k: usize,
) {
    let (nx, _, _) = mask.dim();
    let idx = i + nx * j;
    if !mask[[i, j, k]] && !exterior[idx] {
        exterior[idx] = true;
        queue.push_back((i, j));
    }
}

fn neighbors(i: usize, j: usize, nx: usize, ny: usize) -> impl Iterator<Item = (usize, usize)> {
    let mut out = [(usize::MAX, usize::MAX); 4];
    let mut len = 0;
    if i > 0 {
        out[len] = (i - 1, j);
        len += 1;
    }
    if i + 1 < nx {
        out[len] = (i + 1, j);
        len += 1;
    }
    if j > 0 {
        out[len] = (i, j - 1);
        len += 1;
    }
    if j + 1 < ny {
        out[len] = (i, j + 1);
        len += 1;
    }
    out.into_iter().take(len)
}

fn apply_radius_mask(mask: &mut Array3<bool>, shape: [usize; 3], spacing_m: f64) {
    let spacing_mm = spacing_m * 1000.0;
    for i in 0..shape[0] {
        let x = centered_coord(i, shape[0], spacing_mm);
        for j in 0..shape[1] {
            let y = centered_coord(j, shape[1], spacing_mm);
            if x * x + y * y > BREAST_RADIUS_MAX_MM * BREAST_RADIUS_MAX_MM {
                for k in 0..shape[2] {
                    mask[[i, j, k]] = false;
                }
            }
        }
    }
}

fn tissue_intensity_bounds(volume: &Array3<f64>, mask: &Array3<bool>) -> KwaversResult<(f64, f64)> {
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    for (&value, &is_tissue) in volume.iter().zip(mask.iter()) {
        if is_tissue {
            min_value = min_value.min(value);
            max_value = max_value.max(value);
        }
    }
    if !min_value.is_finite() || !max_value.is_finite() || max_value <= min_value {
        return Err(KwaversError::InvalidInput(
            "MRI tissue segmentation has degenerate intensity bounds".to_owned(),
        ));
    }
    Ok((min_value, max_value))
}
