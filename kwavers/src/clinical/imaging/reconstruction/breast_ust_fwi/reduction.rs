//! Reduced-domain preparation for Ali 2025 breast-FWI probes.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

const DEFAULT_DIAMETER_LATERAL_EXTENT_FRACTION: f64 = 0.80;
const ALI_2025_ROW_SPACING_M: f64 = 0.0024;
const REDUCED_ARRAY_AXIAL_EXTENT_FRACTION: f64 = 0.80;
const INTERIOR_Z_MARGIN_CELLS: usize = 1;

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
    /// Ring diameter constrained to the reduced lateral field of view.
    pub diameter_m: f64,
    /// Axial spacing between ring rows.
    pub row_spacing_m: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BreastUstReducedArrayRowPolicy {
    /// One central ring for smoke-scale pipeline checks.
    SmokeSingleRing,
    /// Caller supplied row count and optional row spacing.
    Explicit,
    /// One row per interior z-slice, leaving one grid-cell margin at each end.
    Table1ParityInteriorCoverage,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreastUstReducedArrayPlan {
    /// Number of ring rows selected for the reduced acquisition.
    pub rows: usize,
    /// Metric geometry associated with the selected row policy.
    pub geometry: BreastUstReducedArrayGeometry,
    /// Policy that produced the row count and spacing.
    pub row_policy: BreastUstReducedArrayRowPolicy,
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
    let diameter = diameter_m.unwrap_or(DEFAULT_DIAMETER_LATERAL_EXTENT_FRACTION * lateral_extent);
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
            f64::min(
                ALI_2025_ROW_SPACING_M,
                REDUCED_ARRAY_AXIAL_EXTENT_FRACTION * axial_extent / (rows - 1) as f64,
            )
        }
    };
    if !row_spacing.is_finite() || row_spacing < 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "row_spacing_m must be finite and nonnegative, got {row_spacing}"
        )));
    }
    validate_reduced_array_row_span(shape, spacing_m, rows, row_spacing)?;
    Ok(BreastUstReducedArrayGeometry {
        diameter_m: diameter,
        row_spacing_m: row_spacing,
    })
}

/// Derive the reduced multi-row ring acquisition plan.
///
/// The Table 1 parity policy maps each interior z-slice to one ring row and
/// reserves one cell at both axial boundaries. For a reduced grid with `nz`
/// cells, this gives `rows = nz - 2`; the default row spacing is exactly the
/// reduced voxel spacing, so the row span is `(nz - 3) * spacing` and stays
/// inside the physical axial extent `(nz - 1) * spacing`.
///
/// # Errors
/// Returns an error when the selected policy lacks required inputs, receives
/// incompatible explicit inputs, or produces geometry outside the reduced
/// finite domain.
pub fn derive_reduced_breast_ust_array_plan(
    shape: (usize, usize, usize),
    spacing_m: f64,
    row_policy: BreastUstReducedArrayRowPolicy,
    rows: Option<usize>,
    diameter_m: Option<f64>,
    row_spacing_m: Option<f64>,
) -> KwaversResult<BreastUstReducedArrayPlan> {
    let derived_rows = match row_policy {
        BreastUstReducedArrayRowPolicy::SmokeSingleRing => {
            if rows.is_some() {
                return Err(KwaversError::InvalidInput(
                    "smoke single-ring policy does not accept explicit rows".into(),
                ));
            }
            1
        }
        BreastUstReducedArrayRowPolicy::Explicit => rows.ok_or_else(|| {
            KwaversError::InvalidInput("explicit row policy requires rows".into())
        })?,
        BreastUstReducedArrayRowPolicy::Table1ParityInteriorCoverage => {
            if rows.is_some() {
                return Err(KwaversError::InvalidInput(
                    "Table 1 parity row policy derives rows from the reduced z-extent".into(),
                ));
            }
            shape
                .2
                .checked_sub(2 * INTERIOR_Z_MARGIN_CELLS)
                .filter(|interior| *interior > 0)
                .ok_or_else(|| {
                    KwaversError::InvalidInput(format!(
                        "Table 1 parity row policy requires at least {} z-cells, got {}",
                        2 * INTERIOR_Z_MARGIN_CELLS + 1,
                        shape.2
                    ))
                })?
        }
    };
    let derived_row_spacing = match row_policy {
        BreastUstReducedArrayRowPolicy::Table1ParityInteriorCoverage => {
            row_spacing_m.or(Some(spacing_m))
        }
        _ => row_spacing_m,
    };
    let geometry = derive_reduced_breast_ust_array_geometry(
        shape,
        spacing_m,
        derived_rows,
        diameter_m,
        derived_row_spacing,
    )?;
    Ok(BreastUstReducedArrayPlan {
        rows: derived_rows,
        geometry,
        row_policy,
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

fn validate_reduced_array_row_span(
    shape: (usize, usize, usize),
    spacing_m: f64,
    rows: usize,
    row_spacing_m: f64,
) -> KwaversResult<()> {
    if rows <= 1 {
        return Ok(());
    }
    if row_spacing_m <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "multi-row reduced array requires positive row_spacing_m".into(),
        ));
    }
    let axial_extent = (shape.2 - 1) as f64 * spacing_m;
    let row_span = (rows - 1) as f64 * row_spacing_m;
    if row_span > axial_extent {
        return Err(KwaversError::InvalidInput(format!(
            "row span {row_span} m exceeds reduced axial grid extent {axial_extent} m"
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

        assert!(
            (geometry.diameter_m - DEFAULT_DIAMETER_LATERAL_EXTENT_FRACTION * 9.0e-3).abs()
                <= 1.0e-15
        );
        assert_eq!(geometry.row_spacing_m, ALI_2025_ROW_SPACING_M);
        assert!(
            derive_reduced_breast_ust_array_geometry((10, 10, 4), 1.0e-3, 1, Some(0.1), None,)
                .is_err()
        );
    }

    #[test]
    fn reduced_array_plan_derives_table1_interior_rows() {
        let plan = derive_reduced_breast_ust_array_plan(
            (24, 24, 12),
            3.2e-3,
            BreastUstReducedArrayRowPolicy::Table1ParityInteriorCoverage,
            None,
            None,
            None,
        )
        .expect("plan");

        assert_eq!(plan.rows, 10);
        assert_eq!(
            plan.row_policy,
            BreastUstReducedArrayRowPolicy::Table1ParityInteriorCoverage
        );
        assert_eq!(plan.geometry.row_spacing_m, 3.2e-3);
        assert!((plan.rows - 1) as f64 * plan.geometry.row_spacing_m < 11.0 * 3.2e-3);
    }

    #[test]
    fn reduced_array_plan_rejects_invalid_policy_domains() {
        assert!(derive_reduced_breast_ust_array_plan(
            (8, 8, 2),
            1.0e-3,
            BreastUstReducedArrayRowPolicy::Table1ParityInteriorCoverage,
            None,
            None,
            None,
        )
        .is_err());
        assert!(derive_reduced_breast_ust_array_plan(
            (8, 8, 4),
            1.0e-3,
            BreastUstReducedArrayRowPolicy::Explicit,
            None,
            None,
            None,
        )
        .is_err());
        assert!(
            derive_reduced_breast_ust_array_geometry((8, 8, 4), 1.0e-3, 4, None, Some(2.0e-3),)
                .is_err()
        );
    }
}
