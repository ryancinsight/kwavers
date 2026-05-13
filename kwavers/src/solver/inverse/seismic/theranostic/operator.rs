//! Matrix-free-style finite-frequency operators stored as row-major blocks.

use ndarray::Array2;

use super::config::{TheranosticFwiConfig, C_REF_M_S};
use super::geometry::{receiver_index, DeviceLayout, Point2};
use super::medium::PreparedTheranosticSlice;

#[derive(Clone, Debug)]
pub struct ActiveGrid {
    pub indices: Vec<(usize, usize)>,
    pub points_m: Vec<Point2>,
}

#[derive(Clone, Debug)]
pub struct RowMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl RowMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn row_mut(&mut self, row: usize) -> &mut [f32] {
        let start = row * self.cols;
        &mut self.data[start..start + self.cols]
    }

    pub fn matvec(&self, x: &[f32], out: &mut [f32]) {
        for (row, out_value) in out.iter_mut().enumerate().take(self.rows) {
            let block = &self.data[row * self.cols..(row + 1) * self.cols];
            *out_value = dot(block, x);
        }
    }

    pub fn t_matvec(&self, y: &[f32], out: &mut [f32]) {
        out.fill(0.0);
        for (row, y_value) in y.iter().copied().enumerate().take(self.rows) {
            let block = &self.data[row * self.cols..(row + 1) * self.cols];
            for (dst, value) in out.iter_mut().zip(block.iter()) {
                *dst += y_value * *value;
            }
        }
    }

    pub fn normal_diag(&self) -> Vec<f32> {
        let mut diag = vec![0.0; self.cols];
        for row in 0..self.rows {
            let block = &self.data[row * self.cols..(row + 1) * self.cols];
            for (dst, value) in diag.iter_mut().zip(block.iter()) {
                *dst += *value * *value;
            }
        }
        diag
    }
}

pub fn active_grid(mask: &Array2<bool>, spacing_m: f64) -> ActiveGrid {
    let (nx, ny) = mask.dim();
    let cx = (nx - 1) as f64 * 0.5;
    let cy = (ny - 1) as f64 * 0.5;
    let mut indices = Vec::new();
    let mut points_m = Vec::new();
    for ((ix, iy), active) in mask.indexed_iter() {
        if *active {
            indices.push((ix, iy));
            points_m.push(Point2 {
                x_m: (ix as f64 - cx) * spacing_m,
                y_m: (iy as f64 - cy) * spacing_m,
            });
        }
    }
    ActiveGrid { indices, points_m }
}

pub fn vector_from_image(image: &Array2<f64>, active: &ActiveGrid) -> Vec<f32> {
    active
        .indices
        .iter()
        .map(|(ix, iy)| image[[*ix, *iy]] as f32)
        .collect()
}

pub fn image_from_vector(
    values: &[f32],
    active: &ActiveGrid,
    shape: (usize, usize),
) -> Array2<f64> {
    let mut image = Array2::<f64>::zeros(shape);
    for ((ix, iy), value) in active.indices.iter().zip(values.iter()) {
        image[[*ix, *iy]] = f64::from(*value);
    }
    image
}

pub fn build_fundamental_matrix(
    prepared: &PreparedTheranosticSlice,
    layout: &DeviceLayout,
    active: &ActiveGrid,
    config: &TheranosticFwiConfig,
) -> RowMatrix {
    build_pitch_catch_matrix(prepared, layout, active, config, Channel::Fundamental)
}

pub fn build_harmonic_matrix(
    prepared: &PreparedTheranosticSlice,
    layout: &DeviceLayout,
    active: &ActiveGrid,
    config: &TheranosticFwiConfig,
) -> RowMatrix {
    build_pitch_catch_matrix(prepared, layout, active, config, Channel::SecondHarmonic)
}

pub fn build_ultraharmonic_matrix(
    prepared: &PreparedTheranosticSlice,
    layout: &DeviceLayout,
    active: &ActiveGrid,
    config: &TheranosticFwiConfig,
) -> RowMatrix {
    build_pitch_catch_matrix(prepared, layout, active, config, Channel::Ultraharmonic)
}

pub fn build_passive_matrix(
    prepared: &PreparedTheranosticSlice,
    layout: &DeviceLayout,
    active: &ActiveGrid,
    config: &TheranosticFwiConfig,
) -> RowMatrix {
    let receiver_count = layout.therapy_elements.len() + layout.imaging_receivers.len();
    let rows = 2 * receiver_count * config.frequencies_hz.len();
    let mut matrix = RowMatrix::zeros(rows, active.points_m.len());
    let mut row = 0;
    for frequency_hz in &config.frequencies_hz {
        let k = std::f64::consts::TAU * (0.5 * *frequency_hz) / C_REF_M_S;
        let frequency_mhz = 0.5 * *frequency_hz * 1.0e-6;
        for receiver in layout
            .therapy_elements
            .iter()
            .chain(layout.imaging_receivers.iter())
        {
            fill_passive_row(
                matrix.row_mut(row),
                prepared,
                active,
                *receiver,
                k,
                frequency_mhz,
                false,
            );
            row += 1;
            fill_passive_row(
                matrix.row_mut(row),
                prepared,
                active,
                *receiver,
                k,
                frequency_mhz,
                true,
            );
            row += 1;
        }
    }
    matrix
}

pub fn exposure_map(
    prepared: &PreparedTheranosticSlice,
    layout: &DeviceLayout,
    config: &TheranosticFwiConfig,
) -> Array2<f64> {
    let (nx, ny) = prepared.ct_hu.dim();
    let cx = (nx - 1) as f64 * 0.5;
    let cy = (ny - 1) as f64 * 0.5;
    let frequency = config.frequencies_hz.last().copied().unwrap_or(500_000.0);
    let k = std::f64::consts::TAU * frequency / C_REF_M_S;
    let mut field = Array2::<f64>::zeros((nx, ny));
    for ix in 0..nx {
        for iy in 0..ny {
            if !prepared.body_mask[[ix, iy]] {
                continue;
            }
            let point = Point2 {
                x_m: (ix as f64 - cx) * prepared.spacing_m,
                y_m: (iy as f64 - cy) * prepared.spacing_m,
            };
            let mut pressure = 0.0;
            for source in &layout.therapy_elements {
                let d = distance(point, *source).max(prepared.spacing_m);
                let df = distance(layout.focus_m, *source).max(prepared.spacing_m);
                pressure += (k * (d - df)).cos() / d.sqrt();
            }
            field[[ix, iy]] = pressure.abs();
        }
    }
    let normalized = normalize_positive(&field, &prepared.body_mask);
    normalized.mapv(|value| value * config.source_pressure_pa)
}

enum Channel {
    Fundamental,
    SecondHarmonic,
    Ultraharmonic,
}

fn build_pitch_catch_matrix(
    prepared: &PreparedTheranosticSlice,
    layout: &DeviceLayout,
    active: &ActiveGrid,
    config: &TheranosticFwiConfig,
    channel: Channel,
) -> RowMatrix {
    let rows =
        layout.therapy_elements.len() * config.receiver_offsets.len() * config.frequencies_hz.len();
    let mut matrix = RowMatrix::zeros(rows, active.points_m.len());
    let mut row = 0;
    for frequency_hz in &config.frequencies_hz {
        let harmonic = match channel {
            Channel::Fundamental => 1.0,
            Channel::SecondHarmonic => 2.0,
            Channel::Ultraharmonic => 1.5,
        };
        let k = std::f64::consts::TAU * *frequency_hz * harmonic / C_REF_M_S;
        let frequency_mhz = *frequency_hz * 1.0e-6 * harmonic;
        for source_idx in 0..layout.therapy_elements.len() {
            let source = layout.therapy_elements[source_idx];
            for offset in &config.receiver_offsets {
                let receiver = layout.therapy_elements
                    [receiver_index(source_idx, *offset, layout.therapy_elements.len())];
                fill_pitch_catch_row(
                    matrix.row_mut(row),
                    prepared,
                    active,
                    source,
                    receiver,
                    k,
                    frequency_mhz,
                    harmonic,
                );
                row += 1;
            }
        }
    }
    matrix
}

fn fill_pitch_catch_row(
    row: &mut [f32],
    prepared: &PreparedTheranosticSlice,
    active: &ActiveGrid,
    source: Point2,
    receiver: Point2,
    k: f64,
    frequency_mhz: f64,
    harmonic: f64,
) {
    for (idx, point) in active.points_m.iter().copied().enumerate() {
        let ds = distance(point, source).max(prepared.spacing_m);
        let dr = distance(point, receiver).max(prepared.spacing_m);
        let (ix, iy) = active.indices[idx];
        let alpha = prepared.attenuation_np_per_m_mhz[[ix, iy]];
        let attenuation = (-alpha * frequency_mhz * (ds + dr)).exp();
        let nonlinear = if harmonic > 1.0 {
            0.18 * (ds + dr)
        } else {
            1.0
        };
        row[idx] = (prepared.spacing_m
            * prepared.spacing_m
            * nonlinear
            * attenuation
            * (k * (ds + dr)).cos()
            / (ds * dr).sqrt()) as f32;
    }
    normalize_row(row);
}

fn fill_passive_row(
    row: &mut [f32],
    prepared: &PreparedTheranosticSlice,
    active: &ActiveGrid,
    receiver: Point2,
    k: f64,
    frequency_mhz: f64,
    sine_phase: bool,
) {
    for (idx, point) in active.points_m.iter().copied().enumerate() {
        let dr = distance(point, receiver).max(prepared.spacing_m);
        let (ix, iy) = active.indices[idx];
        let alpha = prepared.attenuation_np_per_m_mhz[[ix, iy]];
        let phase = if sine_phase {
            (k * dr).sin()
        } else {
            (k * dr).cos()
        };
        row[idx] =
            (prepared.spacing_m * prepared.spacing_m * (-alpha * frequency_mhz * dr).exp() * phase
                / dr.sqrt()) as f32;
    }
    normalize_row(row);
}

pub fn normalize_positive(image: &Array2<f64>, mask: &Array2<bool>) -> Array2<f64> {
    let mut max_value = 0.0;
    for (value, active) in image.iter().zip(mask.iter()) {
        if *active {
            max_value = f64::max(max_value, value.abs());
        }
    }
    if max_value <= 0.0 {
        return Array2::<f64>::zeros(image.dim());
    }
    Array2::from_shape_fn(image.dim(), |idx| {
        if mask[idx] {
            (image[idx] / max_value).clamp(0.0, 1.0)
        } else {
            0.0
        }
    })
}

fn normalize_row(row: &mut [f32]) {
    let norm = dot(row, row).sqrt().max(1.0e-12);
    for value in row.iter_mut() {
        *value /= norm;
    }
}

fn distance(a: Point2, b: Point2) -> f64 {
    (a.x_m - b.x_m).hypot(a.y_m - b.y_m)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
