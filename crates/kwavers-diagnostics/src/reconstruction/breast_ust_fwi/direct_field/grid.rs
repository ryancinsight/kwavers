use kwavers_core::constants::numerical::FOUR_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_transducer::transducers::ElementPosition;
use ndarray::Array3;
use kwavers_math::fft::Complex64;
use std::f64::consts::TAU;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct GridShape {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl GridShape {
    pub(super) fn new(dimensions: (usize, usize, usize)) -> KwaversResult<Self> {
        let (nx, ny, nz) = dimensions;
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(KwaversError::InvalidInput(format!(
                "grid dimensions must be positive, got {dimensions:?}"
            )));
        }
        Ok(Self { nx, ny, nz })
    }

    #[inline]
    pub(super) fn dimensions(self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }
}

pub(super) fn validate_sampling(
    sound_speed_m_s: f64,
    spacing_m: f64,
    time_step_s: f64,
) -> KwaversResult<()> {
    if !sound_speed_m_s.is_finite() || sound_speed_m_s <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "sound_speed_m_s must be positive and finite, got {sound_speed_m_s}"
        )));
    }
    if !spacing_m.is_finite() || spacing_m <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "spacing_m must be positive and finite, got {spacing_m}"
        )));
    }
    if !time_step_s.is_finite() || time_step_s <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "time_step_s must be positive and finite, got {time_step_s}"
        )));
    }
    Ok(())
}

pub(super) fn point_to_grid_index(
    shape: GridShape,
    spacing_m: f64,
    point: ElementPosition,
) -> KwaversResult<(usize, usize, usize)> {
    let center = [
        0.5 * (shape.nx - 1) as f64 * spacing_m,
        0.5 * (shape.ny - 1) as f64 * spacing_m,
        0.5 * (shape.nz - 1) as f64 * spacing_m,
    ];
    let coordinate = [
        center[0] + point.x_m,
        center[1] + point.y_m,
        center[2] + point.z_m,
    ];
    let maximum = [
        (shape.nx - 1) as f64 * spacing_m,
        (shape.ny - 1) as f64 * spacing_m,
        (shape.nz - 1) as f64 * spacing_m,
    ];
    for axis in 0..3 {
        if coordinate[axis] < 0.0 || coordinate[axis] > maximum[axis] {
            return Err(KwaversError::InvalidInput(format!(
                "ring point {:?} maps outside centered PSTD grid bounds {:?}",
                point, maximum
            )));
        }
    }
    Ok((
        (coordinate[0] / spacing_m).round() as usize,
        (coordinate[1] / spacing_m).round() as usize,
        (coordinate[2] / spacing_m).round() as usize,
    ))
}

pub(super) fn grid_index_to_point(
    shape: GridShape,
    spacing_m: f64,
    (ix, iy, iz): (usize, usize, usize),
) -> ElementPosition {
    let center = [
        0.5 * (shape.nx - 1) as f64,
        0.5 * (shape.ny - 1) as f64,
        0.5 * (shape.nz - 1) as f64,
    ];
    ElementPosition {
        x_m: (ix as f64 - center[0]) * spacing_m,
        y_m: (iy as f64 - center[1]) * spacing_m,
        z_m: (iz as f64 - center[2]) * spacing_m,
    }
}

pub(super) fn source_mask(
    shape: GridShape,
    source_indices: &[(usize, usize, usize)],
) -> KwaversResult<Array3<Complex64>> {
    if source_indices.is_empty() {
        return Err(KwaversError::InvalidInput(
            "at least one source index is required".into(),
        ));
    }
    let mut mask = Array3::<Complex64>::zeros(shape.dimensions());
    for &(ix, iy, iz) in source_indices {
        if ix >= shape.nx || iy >= shape.ny || iz >= shape.nz {
            return Err(KwaversError::InvalidInput(format!(
                "source index {:?} lies outside grid {:?}",
                (ix, iy, iz),
                shape.dimensions()
            )));
        }
        mask[[ix, iy, iz]] += Complex64::new(1.0, 0.0);
    }
    Ok(mask)
}

#[inline]
pub(super) fn angular_mode(index: usize, count: usize, spacing_m: f64) -> f64 {
    let signed_index = if index <= count / 2 {
        index as f64
    } else {
        index as f64 - count as f64
    };
    TAU * signed_index / (count as f64 * spacing_m)
}

#[inline]
pub(super) fn wavenumber_magnitude(
    shape: GridShape,
    spacing_m: f64,
    ix: usize,
    iy: usize,
    iz: usize,
) -> f64 {
    let kx = angular_mode(ix, shape.nx, spacing_m);
    let ky = angular_mode(iy, shape.ny, spacing_m);
    let kz = angular_mode(iz, shape.nz, spacing_m);
    (kx * kx + ky * ky + kz * kz).sqrt()
}

#[inline]
pub(super) fn outgoing_green(
    source: ElementPosition,
    receiver: ElementPosition,
    wavenumber_rad_per_m: f64,
    min_distance_m: f64,
) -> Complex64 {
    let distance = distance_m(source, receiver).max(min_distance_m);
    let phase = wavenumber_rad_per_m * distance;
    Complex64::new(phase.cos(), phase.sin()) / (FOUR_PI * distance)
}

#[inline]
pub(super) fn distance_m(a: ElementPosition, b: ElementPosition) -> f64 {
    let dx = a.x_m - b.x_m;
    let dy = a.y_m - b.y_m;
    let dz = a.z_m - b.z_m;
    (dx * dx + dy * dy + dz * dz).sqrt()
}
