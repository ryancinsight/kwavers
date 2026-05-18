//! Focused-transmit sequence selection for theranostic acquisition studies.
//!
//! # Contract
//!
//! The scheduler owns only the ordered transmit subset. It does not construct a
//! second imaging pipeline. Given a fixed CT-derived layout `E` and budget `k`,
//! it returns `S_k subset E`; downstream operators consume `S_k` through their
//! existing finite-frequency row construction.

use std::cmp::Ordering;

use ndarray::Array2;

use crate::core::error::{KwaversError, KwaversResult};

use super::geometry::{DeviceLayout, Point2};
use super::medium::PreparedTheranosticSlice;

pub const TRANSMIT_SCHEDULE_MODEL: &str =
    "focused_transmit_sequence_subset_target_sensitivity_greedy";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransmitScheduleStrategy {
    Full,
    Uniform,
    PatientAdaptive,
}

impl TransmitScheduleStrategy {
    pub fn from_name(name: &str) -> KwaversResult<Self> {
        match name.to_ascii_lowercase().replace('-', "_").as_str() {
            "full" => Ok(Self::Full),
            "uniform" | "equispaced" | "equi_spaced" => Ok(Self::Uniform),
            "patient_adaptive" | "adaptive" | "target_adaptive" => Ok(Self::PatientAdaptive),
            other => Err(KwaversError::InvalidInput(format!(
                "unsupported transmit_schedule_strategy '{other}', expected full, uniform, or patient_adaptive"
            ))),
        }
    }

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Uniform => "uniform",
            Self::PatientAdaptive => "patient_adaptive",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransmitScheduleConfig {
    pub strategy: TransmitScheduleStrategy,
    pub budget: Option<usize>,
}

impl TransmitScheduleConfig {
    #[must_use]
    pub fn full() -> Self {
        Self {
            strategy: TransmitScheduleStrategy::Full,
            budget: None,
        }
    }

    pub fn validate(&self, element_count: usize) -> KwaversResult<()> {
        if element_count == 0 {
            return Err(KwaversError::InvalidInput(
                "transmit schedule requires at least one therapy element".to_owned(),
            ));
        }
        if let Some(budget) = self.budget {
            if budget == 0 || budget > element_count {
                return Err(KwaversError::InvalidInput(format!(
                    "transmit_budget must lie in 1..={element_count}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct TransmitScheduleResult {
    pub strategy: TransmitScheduleStrategy,
    pub requested_budget: Option<usize>,
    pub total_element_count: usize,
    pub active_indices: Vec<usize>,
    pub source_elements: Vec<Point2>,
}

impl TransmitScheduleResult {
    #[must_use]
    pub fn effective_budget(&self) -> usize {
        self.active_indices.len()
    }

    #[must_use]
    pub fn budget_fraction(&self) -> f64 {
        if self.total_element_count == 0 {
            0.0
        } else {
            self.effective_budget() as f64 / self.total_element_count as f64
        }
    }
}

pub fn select_transmit_schedule(
    layout: &DeviceLayout,
    prepared: &PreparedTheranosticSlice,
    config: TransmitScheduleConfig,
) -> KwaversResult<TransmitScheduleResult> {
    let element_count = layout.therapy_elements.len();
    config.validate(element_count)?;
    let budget = config.budget.unwrap_or(element_count);
    let active_indices = match config.strategy {
        TransmitScheduleStrategy::Full => (0..element_count).collect(),
        TransmitScheduleStrategy::Uniform => uniform_indices(element_count, budget),
        TransmitScheduleStrategy::PatientAdaptive => {
            patient_adaptive_indices(layout, prepared, budget)
        }
    };
    let source_elements = active_indices
        .iter()
        .map(|&idx| layout.therapy_elements[idx])
        .collect();
    Ok(TransmitScheduleResult {
        strategy: config.strategy,
        requested_budget: config.budget,
        total_element_count: element_count,
        active_indices,
        source_elements,
    })
}

fn uniform_indices(element_count: usize, budget: usize) -> Vec<usize> {
    if budget >= element_count {
        return (0..element_count).collect();
    }
    (0..budget)
        .map(|rank| rank * element_count / budget)
        .collect()
}

fn patient_adaptive_indices(
    layout: &DeviceLayout,
    prepared: &PreparedTheranosticSlice,
    budget: usize,
) -> Vec<usize> {
    let element_count = layout.therapy_elements.len();
    if budget >= element_count {
        return (0..element_count).collect();
    }
    let target_points = mask_points(&prepared.target_mask, prepared.spacing_m);
    if target_points.is_empty() {
        return uniform_indices(element_count, budget);
    }
    let sensitivity =
        target_sensitivity(&layout.therapy_elements, &target_points, prepared.spacing_m);
    let mut selected = vec![argmax(&sensitivity)];
    while selected.len() < budget {
        let mut best_idx = 0;
        let mut best_value = f64::NEG_INFINITY;
        for idx in 0..element_count {
            if selected.contains(&idx) {
                continue;
            }
            let separation =
                nearest_selected_distance(layout.therapy_elements[idx], &selected, layout);
            let value = sensitivity[idx] * separation;
            let ordering = value.total_cmp(&best_value);
            if ordering == Ordering::Greater || (ordering == Ordering::Equal && idx < best_idx) {
                best_idx = idx;
                best_value = value;
            }
        }
        selected.push(best_idx);
    }
    selected
}

fn target_sensitivity(elements: &[Point2], target_points: &[Point2], spacing_m: f64) -> Vec<f64> {
    let min_distance_sq = spacing_m * spacing_m;
    elements
        .iter()
        .map(|&element| {
            target_points
                .iter()
                .map(|&target| 1.0 / distance_sq(element, target).max(min_distance_sq))
                .sum::<f64>()
                / target_points.len() as f64
        })
        .collect()
}

fn mask_points(mask: &Array2<bool>, spacing_m: f64) -> Vec<Point2> {
    let (nx, ny) = mask.dim();
    let center_x = 0.5 * (nx.saturating_sub(1)) as f64;
    let center_y = 0.5 * (ny.saturating_sub(1)) as f64;
    mask.indexed_iter()
        .filter_map(|((ix, iy), active)| {
            active.then_some(Point2 {
                x_m: (ix as f64 - center_x) * spacing_m,
                y_m: (iy as f64 - center_y) * spacing_m,
            })
        })
        .collect()
}

fn nearest_selected_distance(candidate: Point2, selected: &[usize], layout: &DeviceLayout) -> f64 {
    selected
        .iter()
        .map(|&idx| distance(candidate, layout.therapy_elements[idx]))
        .fold(f64::INFINITY, f64::min)
}

fn argmax(values: &[f64]) -> usize {
    let mut best_idx = 0;
    let mut best_value = f64::NEG_INFINITY;
    for (idx, &value) in values.iter().enumerate() {
        if value > best_value {
            best_idx = idx;
            best_value = value;
        }
    }
    best_idx
}

fn distance(a: Point2, b: Point2) -> f64 {
    distance_sq(a, b).sqrt()
}

fn distance_sq(a: Point2, b: Point2) -> f64 {
    let dx = a.x_m - b.x_m;
    let dy = a.y_m - b.y_m;
    dx * dx + dy * dy
}
