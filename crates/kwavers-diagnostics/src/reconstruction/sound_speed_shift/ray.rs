//! Exact straight-segment intersection lengths for square imaging pixels.

use kwavers_solver::inverse::same_aperture::PlanarPoint;

const PARAM_EPS: f64 = 1.0e-12;

#[cfg(test)]
#[must_use]
pub(super) fn segment_cell_length(
    transmitter: PlanarPoint,
    receiver: PlanarPoint,
    ix: usize,
    iy: usize,
    shape: (usize, usize),
    spacing_m: f64,
) -> f64 {
    let (xmin, xmax, ymin, ymax) = cell_bounds(ix, iy, shape, spacing_m);
    let dx = receiver.x_m - transmitter.x_m;
    let dy = receiver.y_m - transmitter.y_m;
    let length = dx.hypot(dy);
    if length <= f64::EPSILON {
        return 0.0;
    }

    let mut t_min = 0.0;
    let mut t_max = 1.0;
    if !clip_axis(transmitter.x_m, dx, xmin, xmax, &mut t_min, &mut t_max) {
        return 0.0;
    }
    if !clip_axis(transmitter.y_m, dy, ymin, ymax, &mut t_min, &mut t_max) {
        return 0.0;
    }
    if t_max <= t_min {
        return 0.0;
    }
    (t_max - t_min) * length
}

#[must_use]
pub(super) fn segment_grid_lengths(
    transmitter: PlanarPoint,
    receiver: PlanarPoint,
    shape: (usize, usize),
    spacing_m: f64,
) -> Vec<((usize, usize), f64)> {
    let (nx, ny) = shape;
    let dx = receiver.x_m - transmitter.x_m;
    let dy = receiver.y_m - transmitter.y_m;
    let length = dx.hypot(dy);
    if nx == 0 || ny == 0 || length <= f64::EPSILON {
        return Vec::new();
    }

    let (xmin, xmax, ymin, ymax) = grid_bounds(shape, spacing_m);
    let mut t_min = 0.0;
    let mut t_max = 1.0;
    if !clip_axis(transmitter.x_m, dx, xmin, xmax, &mut t_min, &mut t_max) {
        return Vec::new();
    }
    if !clip_axis(transmitter.y_m, dy, ymin, ymax, &mut t_min, &mut t_max) {
        return Vec::new();
    }
    if t_max <= t_min {
        return Vec::new();
    }

    let mut cuts = vec![t_min, t_max];
    append_axis_cuts(
        &mut cuts,
        transmitter.x_m,
        dx,
        AxisCutSpec {
            lower: xmin,
            cells: nx,
            spacing_m,
            t_min,
            t_max,
        },
    );
    append_axis_cuts(
        &mut cuts,
        transmitter.y_m,
        dy,
        AxisCutSpec {
            lower: ymin,
            cells: ny,
            spacing_m,
            t_min,
            t_max,
        },
    );
    cuts.sort_by(|a, b| a.total_cmp(b));
    cuts.dedup_by(|a, b| (*a - *b).abs() <= PARAM_EPS);

    let mut out = Vec::with_capacity(cuts.len().saturating_sub(1));
    for interval in cuts.windows(2) {
        let enter = interval[0];
        let exit = interval[1];
        if exit - enter <= PARAM_EPS {
            continue;
        }
        let midpoint = 0.5 * (enter + exit);
        let x = transmitter.x_m + dx * midpoint;
        let y = transmitter.y_m + dy * midpoint;
        if let Some((ix, iy)) = cell_index(x, y, shape, spacing_m) {
            let segment = (exit - enter) * length;
            if let Some((last_idx, last_length)) = out.last_mut() {
                if *last_idx == (ix, iy) {
                    *last_length += segment;
                    continue;
                }
            }
            out.push(((ix, iy), segment));
        }
    }
    out
}

fn clip_axis(
    start: f64,
    delta: f64,
    lower: f64,
    upper: f64,
    t_min: &mut f64,
    t_max: &mut f64,
) -> bool {
    if delta.abs() <= f64::EPSILON {
        return start >= lower && start <= upper;
    }

    let inv = 1.0 / delta;
    let mut enter = (lower - start) * inv;
    let mut exit = (upper - start) * inv;
    if enter > exit {
        std::mem::swap(&mut enter, &mut exit);
    }
    *t_min = (*t_min).max(enter);
    *t_max = (*t_max).min(exit);
    *t_min <= *t_max
}

#[derive(Clone, Copy, Debug)]
struct AxisCutSpec {
    lower: f64,
    cells: usize,
    spacing_m: f64,
    t_min: f64,
    t_max: f64,
}

fn append_axis_cuts(cuts: &mut Vec<f64>, start: f64, delta: f64, spec: AxisCutSpec) {
    if delta.abs() <= f64::EPSILON {
        return;
    }
    for boundary in 1..spec.cells {
        let coordinate = spec.lower + boundary as f64 * spec.spacing_m;
        let t = (coordinate - start) / delta;
        if t > spec.t_min + PARAM_EPS && t < spec.t_max - PARAM_EPS {
            cuts.push(t);
        }
    }
}

fn cell_index(x_m: f64, y_m: f64, shape: (usize, usize), spacing_m: f64) -> Option<(usize, usize)> {
    let (nx, ny) = shape;
    let (xmin, xmax, ymin, ymax) = grid_bounds(shape, spacing_m);
    if x_m < xmin - PARAM_EPS
        || x_m > xmax + PARAM_EPS
        || y_m < ymin - PARAM_EPS
        || y_m > ymax + PARAM_EPS
    {
        return None;
    }

    let ix = (((x_m - xmin) / spacing_m).floor() as usize).min(nx - 1);
    let iy = (((y_m - ymin) / spacing_m).floor() as usize).min(ny - 1);
    Some((ix, iy))
}

#[cfg(test)]
fn cell_bounds(
    ix: usize,
    iy: usize,
    shape: (usize, usize),
    spacing_m: f64,
) -> (f64, f64, f64, f64) {
    let (xmin, _, ymin, _) = grid_bounds(shape, spacing_m);
    (
        xmin + ix as f64 * spacing_m,
        xmin + (ix + 1) as f64 * spacing_m,
        ymin + iy as f64 * spacing_m,
        ymin + (iy + 1) as f64 * spacing_m,
    )
}

pub(super) fn cell_center(
    ix: usize,
    iy: usize,
    shape: (usize, usize),
    spacing_m: f64,
) -> PlanarPoint {
    let (xmin, _, ymin, _) = grid_bounds(shape, spacing_m);
    PlanarPoint {
        x_m: xmin + (ix as f64 + 0.5) * spacing_m,
        y_m: ymin + (iy as f64 + 0.5) * spacing_m,
    }
}

pub(super) fn grid_bounds(shape: (usize, usize), spacing_m: f64) -> (f64, f64, f64, f64) {
    let (nx, ny) = shape;
    let x_extent = nx as f64 * spacing_m;
    let y_extent = ny as f64 * spacing_m;
    (
        -0.5 * x_extent,
        0.5 * x_extent,
        -0.5 * y_extent,
        0.5 * y_extent,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_horizontal_segment_intersects_one_pixel_width() {
        let length = segment_cell_length(
            PlanarPoint {
                x_m: -0.002,
                y_m: 0.0,
            },
            PlanarPoint {
                x_m: 0.002,
                y_m: 0.0,
            },
            1,
            1,
            (3, 3),
            0.001,
        );

        assert!((length - 0.001).abs() <= 1.0e-15);
    }

    #[test]
    fn grid_traversal_matches_per_cell_oracle_for_oblique_ray() {
        let transmitter = PlanarPoint {
            x_m: -0.003,
            y_m: -0.0017,
        };
        let receiver = PlanarPoint {
            x_m: 0.003,
            y_m: 0.0013,
        };
        let shape = (5, 4);
        let spacing = 0.001;
        let mut traversed = segment_grid_lengths(transmitter, receiver, shape, spacing);
        let mut expected = Vec::new();
        for ix in 0..shape.0 {
            for iy in 0..shape.1 {
                let length = segment_cell_length(transmitter, receiver, ix, iy, shape, spacing);
                if length > 0.0 {
                    expected.push(((ix, iy), length));
                }
            }
        }
        traversed.sort_by_key(|(idx, _)| *idx);

        assert_eq!(
            traversed.iter().map(|(idx, _)| *idx).collect::<Vec<_>>(),
            expected.iter().map(|(idx, _)| *idx).collect::<Vec<_>>()
        );
        for ((actual_idx, actual), (expected_idx, expected)) in
            traversed.iter().zip(expected.iter())
        {
            assert_eq!(actual_idx, expected_idx);
            assert!(
                (*actual - *expected).abs() <= 1.0e-14,
                "idx={actual_idx:?}, actual={actual:.12e}, expected={expected:.12e}"
            );
        }
    }

    #[test]
    fn grid_traversal_preserves_clipped_path_length() {
        let transmitter = PlanarPoint {
            x_m: -0.004,
            y_m: -0.002,
        };
        let receiver = PlanarPoint {
            x_m: 0.004,
            y_m: 0.002,
        };
        let traversed = segment_grid_lengths(transmitter, receiver, (4, 4), 0.001);
        let total = traversed.iter().map(|(_, length)| *length).sum::<f64>();
        let expected = 0.004f64.hypot(0.002);

        assert!(
            (total - expected).abs() <= 1.0e-14,
            "total={total:.12e}, expected={expected:.12e}"
        );
    }
}
