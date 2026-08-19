//! Scan conversion: polar beam space → Cartesian display via the ritk-image seam.
//!
//! Scan conversion is a resample from a curvilinear acquisition onto a Cartesian
//! raster. Coordinate mapping delegates to `ritk_spatial::CurvilinearArray`
//! through the `ritk_image::Image::physical_point_to_continuous_index` seam
//! (atlas ADR 0048), so no bespoke polar arithmetic lives in this module.
//!
//! # Entry point
//!
//! [`scan_convert`] is the only public API. The former `ScanConverter` struct
//! has been deleted; callers that stored it should call [`scan_convert`]
//! directly (atlas ADR 0048, US-023-A6).
//!
//! # Geometry
//!
//! Beams fan out from an apex. A Cartesian pixel at `(x, z)` maps back to a
//! fractional `(beam, sample)` index through the `CurvilinearArray` geometry.
//! Bilinear interpolation on the beam grid follows; pixels outside the acquired
//! fan stay zero.
//!
//! # Reference
//! - Szabo, T. L. (2014). *Diagnostic Ultrasound Imaging: Inside Out* (2nd ed.),
//!   §10.4 (scan conversion). Academic Press.

use aequitas::systems::si::quantities::{Angle, Length};
use aequitas::systems::si::units::{Meter, Radian};
use coeus_core::SequentialBackend;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array2, ArrayView2};
use ritk_image::Image;
use ritk_spatial::{CoordinateMap, Direction, Point, Spacing};

/// Polar acquisition geometry: uniformly-spaced beams, uniform range sampling.
#[derive(Debug, Clone, Copy)]
pub struct ScanGeometry {
    /// First beam angle (from the axial axis; negative = left of center).
    pub angle_min: Angle<f64>,
    /// Angular spacing between beams.
    pub angle_step: Angle<f64>,
    /// Apex-to-first-sample radius (zero for a sector phased array).
    pub radius_offset: Length<f64>,
    /// Range sampling step.
    pub range_step: Length<f64>,
}

/// Output Cartesian raster specification.
#[derive(Debug, Clone, Copy)]
pub struct CartesianGrid {
    /// Image width in pixels (lateral, `x`).
    pub width: usize,
    /// Image height in pixels (axial, `z`).
    pub height: usize,
    /// Lateral extent `[x_min, x_max]`.
    pub x_range: (Length<f64>, Length<f64>),
    /// Axial extent `[z_min, z_max]`.
    pub z_range: (Length<f64>, Length<f64>),
}

/// Convert polar `beam_data` `[n_lines, n_samples]` to a Cartesian image
/// `[height, width]` (row-major, row = axial `z`, column = lateral `x`).
///
/// Coordinate mapping routes through `ritk_image::Image::physical_point_to_continuous_index`
/// with a `CoordinateMap::CurvilinearArray` attached. Each Cartesian pixel's
/// world coordinates `(x, z)` are mapped to fractional `(beam, sample)` indices
/// by the seam; bilinear interpolation follows. Pixels outside the acquired fan
/// stay zero.
///
/// # Errors
///
/// Returns `KwaversError::InvalidInput` when geometry parameters are
/// out-of-range, the Cartesian grid is degenerate, or `beam_data` has fewer
/// than two beams or two samples.
pub fn scan_convert(
    beam_data: ArrayView2<f64>,
    geometry: ScanGeometry,
    grid: CartesianGrid,
) -> KwaversResult<Array2<f64>> {
    let angle_min = geometry.angle_min.in_unit::<Radian>();
    let angle_step = geometry.angle_step.in_unit::<Radian>();
    let radius_offset = geometry.radius_offset.in_unit::<Meter>();
    let range_step = geometry.range_step.in_unit::<Meter>();
    let x_min = grid.x_range.0.in_unit::<Meter>();
    let x_max = grid.x_range.1.in_unit::<Meter>();
    let z_min = grid.z_range.0.in_unit::<Meter>();
    let z_max = grid.z_range.1.in_unit::<Meter>();
    if !angle_min.is_finite()
        || !angle_step.is_finite()
        || !radius_offset.is_finite()
        || !range_step.is_finite()
        || !x_min.is_finite()
        || !x_max.is_finite()
        || !z_min.is_finite()
        || !z_max.is_finite()
        || angle_step <= 0.0
        || radius_offset < 0.0
        || range_step <= 0.0
        || x_min >= x_max
        || z_min >= z_max
    {
        return Err(KwaversError::InvalidInput(
            "scan geometry and Cartesian extents must be finite and ordered; range and angle steps must be positive".to_owned(),
        ));
    }
    if grid.width < 2 || grid.height < 2 {
        return Err(KwaversError::InvalidInput(
            "Cartesian grid must be at least 2×2".to_owned(),
        ));
    }
    let [n_lines, n_samples] = beam_data.shape();
    if n_lines < 2 || n_samples < 2 {
        return Err(KwaversError::InvalidInput(
            "scan conversion needs at least 2 beams and 2 samples".to_owned(),
        ));
    }
    // Build the acquisition geometry and attach it to a sentinel Image. The
    // sentinel carries no meaningful pixel data; it acts as the
    // coordinate-map host so that `physical_point_to_continuous_index`
    // dispatches through the `CoordinateMap` seam (atlas ADR 0048).
    let fan = ritk_spatial::CurvilinearArray::try_new(range_step, radius_offset, angle_step, angle_min)
        .map_err(|e| KwaversError::InvalidInput(e.to_string()))?;
    let sentinel: Image<f32, SequentialBackend, 2> = Image::from_flat(
        vec![0.0_f32; n_lines * n_samples],
        [n_lines, n_samples],
        Point::new([0.0_f64, 0.0_f64]),
        Spacing::new([1.0_f64, 1.0_f64]),
        Direction::identity(),
    )
    .map_err(|e| KwaversError::InvalidInput(e.to_string()))?
    .with_coordinate_map(CoordinateMap::CurvilinearArray(fan))
    .map_err(|e| KwaversError::InvalidInput(e.to_string()))?;

    // Resample: for each Cartesian pixel, map (z, x) world coordinates through
    // the seam to fractional (beam, sample) indices and bilinearly interpolate.
    //
    // Convention in physical_point_to_continuous_index for D=2:
    //   point[0] → passed as the axial  (second) arg of index_from_cartesian
    //   point[1] → passed as the lateral (first)  arg of index_from_cartesian
    // Returned index: idx[0] = beam line, idx[1] = range sample.
    let dx = (x_max - x_min) / (grid.width - 1) as f64;
    let dz = (z_max - z_min) / (grid.height - 1) as f64;
    let mut output = Array2::zeros((grid.height, grid.width));
    for row in 0..grid.height {
        let z = z_min + row as f64 * dz;
        for col in 0..grid.width {
            let x = x_min + col as f64 * dx;
            let Ok(idx) = sentinel.physical_point_to_continuous_index(&Point::new([z, x])) else {
                continue;
            };
            // idx[0] = beam (line), idx[1] = range (sample)
            if let Some(v) = bilinear(beam_data, idx[0], idx[1]) {
                output[[row, col]] = v;
            }
        }
    }
    Ok(output)
}

fn bilinear(beam: ArrayView2<f64>, line: f64, sample: f64) -> Option<f64> {
    let [n_lines, n_samples] = beam.shape();
    if line < 0.0 || sample < 0.0 {
        return None;
    }
    let l0 = line.floor() as usize;
    let s0 = sample.floor() as usize;
    if l0 + 1 >= n_lines || s0 + 1 >= n_samples {
        return None;
    }
    let fl = line - l0 as f64;
    let fs = sample - s0 as f64;
    Some(
        beam[[l0, s0]] * (1.0 - fl) * (1.0 - fs)
            + beam[[l0 + 1, s0]] * fl * (1.0 - fs)
            + beam[[l0, s0 + 1]] * (1.0 - fl) * fs
            + beam[[l0 + 1, s0 + 1]] * fl * fs,
    )
}
