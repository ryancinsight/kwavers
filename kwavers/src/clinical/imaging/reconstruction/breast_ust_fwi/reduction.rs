//! Reduced-domain preparation for Ali 2025 breast-FWI probes.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

#[derive(Clone, Debug, PartialEq)]
pub struct BreastUstReducedPhantom {
    pub sound_speed_m_s: Array3<f64>,
    pub initial_sound_speed_m_s: Array3<f64>,
    pub original_shape: (usize, usize, usize),
    pub reduced_shape: (usize, usize, usize),
    pub source_spacing_m: f64,
    pub effective_spacing_m: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreastUstReducedArrayGeometry {
    pub diameter_m: f64,
    pub row_spacing_m: f64,
}

pub fn prepare_reduced_breast_ust_phantom(
    sound_speed_m_s: &Array3<f64>,
    source_spacing_m: f64,
    max_shape: (usize, usize, usize),
    decimation: usize,
) -> KwaversResult<BreastUstReducedPhantom> {
    validate_volume(sound_speed_m_s)?;
    validate_shape(max_shape, "max_shape")?;
    if !source_spacing_m.is_finite() || source_spacing_m <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "source_spacing_m must be positive and finite, got {source_spacing_m}"
        )));
    }
    if decimation == 0 {
        return Err(KwaversError::InvalidInput(
            "decimation factor must be positive".into(),
        ));
    }

    let original_shape = sound_speed_m_s.dim();
    let decimated_shape = (
        original_shape.0.div_ceil(decimation),
        original_shape.1.div_ceil(decimation),
        original_shape.2.div_ceil(decimation),
    );
    let reduced_shape = (
        usize::min(decimated_shape.0, max_shape.0),
        usize::min(decimated_shape.1, max_shape.1),
        usize::min(decimated_shape.2, max_shape.2),
    );
    let crop_start = (
        (decimated_shape.0 - reduced_shape.0) / 2,
        (decimated_shape.1 - reduced_shape.1) / 2,
        (decimated_shape.2 - reduced_shape.2) / 2,
    );
    let reduced = Array3::from_shape_fn(reduced_shape, |(x, y, z)| {
        sound_speed_m_s[[
            (crop_start.0 + x) * decimation,
            (crop_start.1 + y) * decimation,
            (crop_start.2 + z) * decimation,
        ]]
    });
    let median = median_sound_speed(&reduced)?;
    let initial = Array3::from_elem(reduced_shape, median);

    Ok(BreastUstReducedPhantom {
        sound_speed_m_s: reduced,
        initial_sound_speed_m_s: initial,
        original_shape,
        reduced_shape,
        source_spacing_m,
        effective_spacing_m: source_spacing_m * decimation as f64,
    })
}

pub fn derive_reduced_breast_ust_array_geometry(
    shape: (usize, usize, usize),
    spacing_m: f64,
    rows: usize,
    diameter_m: Option<f64>,
    row_spacing_m: Option<f64>,
) -> KwaversResult<BreastUstReducedArrayGeometry> {
    validate_shape(shape, "shape")?;
    if rows == 0 {
        return Err(KwaversError::InvalidInput("rows must be positive".into()));
    }
    if !spacing_m.is_finite() || spacing_m <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "spacing_m must be positive and finite, got {spacing_m}"
        )));
    }
    let lateral_extent = (usize::min(shape.0, shape.1) - 1) as f64 * spacing_m;
    if lateral_extent <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "lateral grid extent is degenerate for shape {shape:?}"
        )));
    }
    let diameter = diameter_m.unwrap_or(0.80 * lateral_extent);
    if !diameter.is_finite() || diameter <= 0.0 || diameter > lateral_extent {
        return Err(KwaversError::InvalidInput(
            "diameter_m must be positive, finite, and no larger than the reduced lateral grid extent"
                .into(),
        ));
    }
    let row_spacing = match row_spacing_m {
        Some(value) => value,
        None if rows == 1 => 0.0,
        None => {
            let axial_extent = (shape.2 - 1) as f64 * spacing_m;
            f64::min(0.0024, 0.80 * axial_extent / (rows - 1) as f64)
        }
    };
    if !row_spacing.is_finite() || row_spacing < 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "row_spacing_m must be finite and nonnegative, got {row_spacing}"
        )));
    }
    Ok(BreastUstReducedArrayGeometry {
        diameter_m: diameter,
        row_spacing_m: row_spacing,
    })
}

fn median_sound_speed(volume: &Array3<f64>) -> KwaversResult<f64> {
    let mut values = volume.iter().copied().collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    let median = if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    };
    if !median.is_finite() || median <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "median sound speed is outside the physical domain".into(),
        ));
    }
    Ok(median)
}

fn validate_volume(volume: &Array3<f64>) -> KwaversResult<()> {
    if volume.is_empty() {
        return Err(KwaversError::InvalidInput(
            "volume must not be empty".into(),
        ));
    }
    for &value in volume {
        if !value.is_finite() {
            return Err(KwaversError::InvalidInput(
                "volume contains nonfinite sound-speed values".into(),
            ));
        }
        if value <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "sound-speed volume must be strictly positive".into(),
            ));
        }
    }
    Ok(())
}

fn validate_shape(shape: (usize, usize, usize), name: &str) -> KwaversResult<()> {
    if shape.0 == 0 || shape.1 == 0 || shape.2 == 0 {
        return Err(KwaversError::InvalidInput(format!(
            "{name} entries must be positive, got {shape:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduced_phantom_matches_decimate_then_center_crop_contract() {
        let source = Array3::from_shape_fn((6, 8, 4), |(x, y, z)| {
            1400.0 + (x * 8 * 4 + y * 4 + z) as f64
        });

        let reduced =
            prepare_reduced_breast_ust_phantom(&source, 5.0e-4, (2, 2, 2), 2).expect("reduced");

        assert_eq!(reduced.original_shape, (6, 8, 4));
        assert_eq!(reduced.reduced_shape, (2, 2, 2));
        assert_eq!(reduced.source_spacing_m, 5.0e-4);
        assert_eq!(reduced.effective_spacing_m, 1.0e-3);
        assert_eq!(reduced.sound_speed_m_s[[0, 0, 0]], source[[0, 2, 0]]);
        assert_eq!(reduced.sound_speed_m_s[[1, 1, 1]], source[[2, 4, 2]]);
        assert!(reduced
            .initial_sound_speed_m_s
            .iter()
            .all(|value| *value == 1445.0));
    }

    #[test]
    fn reduced_array_geometry_matches_probe_policy() {
        let geometry = derive_reduced_breast_ust_array_geometry((10, 12, 4), 1.0e-3, 2, None, None)
            .expect("geometry");

        assert!((geometry.diameter_m - 0.80 * 9.0e-3).abs() <= 1.0e-15);
        assert_eq!(geometry.row_spacing_m, 0.0024);
        assert!(
            derive_reduced_breast_ust_array_geometry((10, 10, 4), 1.0e-3, 1, Some(0.1), None,)
                .is_err()
        );
    }
}
