//! Matrix-free finite-frequency same-aperture operators.

use super::active_grid::{ActiveGrid, PlanarPoint};
use super::finite_frequency::{SameApertureMedium, SameApertureSettings, C_REF_M_S};
use super::linear_operator::LinearOperator;
use super::row_matrix::RowMatrix;

#[derive(Clone, Debug)]
pub struct FiniteFrequencyOperator<'a> {
    medium: SameApertureMedium<'a>,
    active: &'a ActiveGrid,
    kind: OperatorKind<'a>,
    rows: usize,
    row_norms: Vec<f32>,
}

#[derive(Clone, Debug)]
enum OperatorKind<'a> {
    PitchCatch {
        therapy_elements: &'a [PlanarPoint],
        settings: SameApertureSettings<'a>,
        harmonic: f64,
    },
    Passive {
        therapy_elements: &'a [PlanarPoint],
        imaging_receivers: &'a [PlanarPoint],
        frequencies_hz: &'a [f64],
    },
}

#[derive(Clone, Copy, Debug)]
struct PitchCatchRow {
    source: PlanarPoint,
    receiver: PlanarPoint,
    k: f64,
    frequency_mhz: f64,
    harmonic: f64,
}

#[derive(Clone, Copy, Debug)]
struct PassiveRow {
    receiver: PlanarPoint,
    k: f64,
    frequency_mhz: f64,
    sine_phase: bool,
}

impl<'a> FiniteFrequencyOperator<'a> {
    #[must_use]
    pub fn fundamental(
        medium: SameApertureMedium<'a>,
        therapy_elements: &'a [PlanarPoint],
        active: &'a ActiveGrid,
        settings: SameApertureSettings<'a>,
    ) -> Self {
        Self::pitch_catch(medium, therapy_elements, active, settings, 1.0)
    }

    #[must_use]
    pub fn harmonic(
        medium: SameApertureMedium<'a>,
        therapy_elements: &'a [PlanarPoint],
        active: &'a ActiveGrid,
        settings: SameApertureSettings<'a>,
    ) -> Self {
        Self::pitch_catch(medium, therapy_elements, active, settings, 2.0)
    }

    #[must_use]
    pub fn ultraharmonic(
        medium: SameApertureMedium<'a>,
        therapy_elements: &'a [PlanarPoint],
        active: &'a ActiveGrid,
        settings: SameApertureSettings<'a>,
    ) -> Self {
        Self::pitch_catch(medium, therapy_elements, active, settings, 1.5)
    }

    #[must_use]
    pub fn passive(
        medium: SameApertureMedium<'a>,
        therapy_elements: &'a [PlanarPoint],
        imaging_receivers: &'a [PlanarPoint],
        active: &'a ActiveGrid,
        frequencies_hz: &'a [f64],
    ) -> Self {
        let receiver_count = therapy_elements.len() + imaging_receivers.len();
        let rows = 2 * receiver_count * frequencies_hz.len();
        let mut operator = Self {
            medium,
            active,
            kind: OperatorKind::Passive {
                therapy_elements,
                imaging_receivers,
                frequencies_hz,
            },
            rows,
            row_norms: Vec::new(),
        };
        operator.row_norms = operator.compute_row_norms();
        operator
    }

    #[must_use]
    pub fn materialize(&self) -> RowMatrix {
        let mut matrix = RowMatrix::zeros(self.rows, self.active.len());
        for row in 0..self.rows {
            let norm = self.row_norms[row];
            for col in 0..self.active.len() {
                matrix.row_mut(row)[col] = self.unscaled_value(row, col) / norm;
            }
        }
        matrix
    }

    fn pitch_catch(
        medium: SameApertureMedium<'a>,
        therapy_elements: &'a [PlanarPoint],
        active: &'a ActiveGrid,
        settings: SameApertureSettings<'a>,
        harmonic: f64,
    ) -> Self {
        let rows = therapy_elements.len()
            * settings.receiver_offsets.len()
            * settings.frequencies_hz.len();
        let mut operator = Self {
            medium,
            active,
            kind: OperatorKind::PitchCatch {
                therapy_elements,
                settings,
                harmonic,
            },
            rows,
            row_norms: Vec::new(),
        };
        operator.row_norms = operator.compute_row_norms();
        operator
    }

    fn compute_row_norms(&self) -> Vec<f32> {
        (0..self.rows)
            .map(|row| {
                let mut norm_sq = 0.0_f32;
                for col in 0..self.active.len() {
                    let value = self.unscaled_value(row, col);
                    norm_sq += value * value;
                }
                norm_sq.sqrt().max(f32::EPSILON)
            })
            .collect()
    }

    fn unscaled_value(&self, row: usize, col: usize) -> f32 {
        match &self.kind {
            OperatorKind::PitchCatch { .. } => {
                self.pitch_catch_value(self.pitch_catch_row(row), col)
            }
            OperatorKind::Passive { .. } => self.passive_value(self.passive_row(row), col),
        }
    }

    fn pitch_catch_row(&self, row: usize) -> PitchCatchRow {
        let OperatorKind::PitchCatch {
            therapy_elements,
            settings,
            harmonic,
        } = &self.kind
        else {
            unreachable!("pitch-catch row requested from passive operator")
        };
        let per_frequency = therapy_elements.len() * settings.receiver_offsets.len();
        let frequency_idx = row / per_frequency;
        let within_frequency = row % per_frequency;
        let source_idx = within_frequency / settings.receiver_offsets.len();
        let offset_idx = within_frequency % settings.receiver_offsets.len();
        let source = therapy_elements[source_idx];
        let receiver = therapy_elements[receiver_index(
            source_idx,
            settings.receiver_offsets[offset_idx],
            therapy_elements.len(),
        )];
        let frequency_hz = settings.frequencies_hz[frequency_idx];
        PitchCatchRow {
            source,
            receiver,
            k: std::f64::consts::TAU * frequency_hz * *harmonic / C_REF_M_S,
            frequency_mhz: frequency_hz * 1.0e-6 * *harmonic,
            harmonic: *harmonic,
        }
    }

    fn passive_row(&self, row: usize) -> PassiveRow {
        let OperatorKind::Passive {
            therapy_elements,
            imaging_receivers,
            frequencies_hz,
        } = &self.kind
        else {
            unreachable!("passive row requested from pitch-catch operator")
        };
        let receiver_count = therapy_elements.len() + imaging_receivers.len();
        let per_frequency = 2 * receiver_count;
        let frequency_idx = row / per_frequency;
        let within_frequency = row % per_frequency;
        let receiver_idx = within_frequency / 2;
        let receiver = if receiver_idx < therapy_elements.len() {
            therapy_elements[receiver_idx]
        } else {
            imaging_receivers[receiver_idx - therapy_elements.len()]
        };
        let frequency_hz = frequencies_hz[frequency_idx];
        PassiveRow {
            receiver,
            k: std::f64::consts::TAU * (0.5 * frequency_hz) / C_REF_M_S,
            frequency_mhz: 0.5 * frequency_hz * 1.0e-6,
            sine_phase: within_frequency % 2 == 1,
        }
    }

    fn pitch_catch_value(&self, row: PitchCatchRow, col: usize) -> f32 {
        let point = self.active.points_m[col];
        let ds = distance(point, row.source).max(self.medium.spacing_m);
        let dr = distance(point, row.receiver).max(self.medium.spacing_m);
        let path_m = ds + dr;
        let (ix, iy) = self.active.indices[col];
        let alpha = self.medium.attenuation_np_per_m_mhz[[ix, iy]];
        let attenuation = (-alpha * row.frequency_mhz * path_m).exp();
        let nonlinear_path_weight = if row.harmonic > 1.0 { path_m } else { 1.0 };
        (self.medium.spacing_m
            * self.medium.spacing_m
            * nonlinear_path_weight
            * attenuation
            * (row.k * path_m).cos()
            / (ds * dr).sqrt()) as f32
    }

    fn passive_value(&self, row: PassiveRow, col: usize) -> f32 {
        let point = self.active.points_m[col];
        let dr = distance(point, row.receiver).max(self.medium.spacing_m);
        let (ix, iy) = self.active.indices[col];
        let alpha = self.medium.attenuation_np_per_m_mhz[[ix, iy]];
        let phase = if row.sine_phase {
            (row.k * dr).sin()
        } else {
            (row.k * dr).cos()
        };
        (self.medium.spacing_m
            * self.medium.spacing_m
            * (-alpha * row.frequency_mhz * dr).exp()
            * phase
            / dr.sqrt()) as f32
    }
}

impl LinearOperator for FiniteFrequencyOperator<'_> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.active.len()
    }

    fn matvec(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.cols());
        debug_assert_eq!(out.len(), self.rows());
        for (row, dst) in out.iter_mut().enumerate().take(self.rows) {
            let mut sum = 0.0_f32;
            let inv_norm = 1.0 / self.row_norms[row];
            for (col, x_value) in x.iter().copied().enumerate().take(self.cols()) {
                sum += self.unscaled_value(row, col) * inv_norm * x_value;
            }
            *dst = sum;
        }
    }

    fn t_matvec(&self, y: &[f32], out: &mut [f32]) {
        debug_assert_eq!(y.len(), self.rows());
        debug_assert_eq!(out.len(), self.cols());
        out.fill(0.0);
        for (row, y_value) in y.iter().copied().enumerate().take(self.rows) {
            let scale = y_value / self.row_norms[row];
            for (col, dst) in out.iter_mut().enumerate().take(self.cols()) {
                *dst += scale * self.unscaled_value(row, col);
            }
        }
    }

    fn normal_diag(&self) -> Vec<f32> {
        let mut diag = vec![0.0; self.cols()];
        for row in 0..self.rows {
            let inv_norm = 1.0 / self.row_norms[row];
            for (col, dst) in diag.iter_mut().enumerate().take(self.cols()) {
                let value = self.unscaled_value(row, col) * inv_norm;
                *dst += value * value;
            }
        }
        diag
    }

    fn storage_values(&self) -> usize {
        self.row_norms.len()
    }
}

fn receiver_index(source: usize, offset: usize, count: usize) -> usize {
    (source + offset) % count
}

fn distance(a: PlanarPoint, b: PlanarPoint) -> f64 {
    (a.x_m - b.x_m).hypot(a.y_m - b.y_m)
}
