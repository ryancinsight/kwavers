pub mod physics;

mod metrics;
mod plot;
mod simulation;

pub use metrics::{write_metrics_csv, write_profiles_csv, PairwiseMetric, SolverMetric};
pub use plot::write_plot;

use anyhow::Result;
use leto::Array2;
use std::time::Duration;

pub const OUT_DIR: &str = "target/focused_water_tank";

#[derive(Debug, Clone)]
pub struct SolverField {
    pub name: &'static str,
    pub normalized_peak: Array2<f64>,
    pub elapsed: Duration,
}

#[derive(Debug, Clone)]
pub struct AxialField {
    pub name: &'static str,
    pub normalized_peak: Vec<f64>,
    pub elapsed: Duration,
}

#[derive(Debug, Clone)]
pub struct ProfileSet {
    pub name: &'static str,
    pub axial: Vec<(f64, f64)>,
    pub lateral: Vec<(f64, f64)>,
}

#[derive(Debug, Clone)]
pub struct WaterTankOutput {
    pub solver_fields: Vec<SolverField>,
    pub axial_fields: Vec<AxialField>,
    pub solver_metrics: Vec<SolverMetric>,
    pub pairwise_metrics: Vec<PairwiseMetric>,
    pub axial_pairwise_metrics: Vec<PairwiseMetric>,
    pub profiles: Vec<ProfileSet>,
}

pub fn run_comparison() -> Result<WaterTankOutput> {
    let mut fields = simulation::run_solver_fields()?;
    fields.push(SolverField {
        name: "Analytic",
        normalized_peak: metrics::normalize_map(&physics::analytical_peak_map()),
        elapsed: Duration::ZERO,
    });

    let solver_metrics = fields
        .iter()
        .map(|field| metrics::solver_metric(field.name, &field.normalized_peak, field.elapsed))
        .collect();
    let pairwise_metrics = metrics::pairwise_metrics(&fields);
    let mut axial_fields = metrics::axial_fields_from_maps(&fields);
    axial_fields.push(simulation::run_dg_axial_field()?);
    let axial_pairwise_metrics = metrics::axial_pairwise_metrics(&axial_fields);
    let profiles = metrics::profile_sets(&fields, &axial_fields);

    Ok(WaterTankOutput {
        solver_fields: fields,
        axial_fields,
        solver_metrics,
        pairwise_metrics,
        axial_pairwise_metrics,
        profiles,
    })
}
