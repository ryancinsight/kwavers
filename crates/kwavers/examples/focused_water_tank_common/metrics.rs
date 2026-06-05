use super::physics;
use super::{AxialField, ProfileSet, SolverField, WaterTankOutput};
use anyhow::Result;
use ndarray::Array2;
use std::fs::File;
use std::io::Write;
use std::ops::Range;
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct SolverMetric {
    pub name: &'static str,
    pub focus_error_mm: f64,
    pub peak_x_mm: f64,
    pub peak_y_mm: f64,
    pub focus_value: f64,
    pub focus_to_offaxis: f64,
    pub lateral_fwhm_mm: f64,
    pub axial_fwhm_mm: f64,
    pub elapsed: Duration,
}

#[derive(Debug, Clone)]
pub struct PairwiseMetric {
    pub lhs: &'static str,
    pub rhs: &'static str,
    pub normalized_l2: f64,
    pub correlation: f64,
}

pub fn normalize_map(map: &Array2<f64>) -> Array2<f64> {
    let max_value = region_max(map).max(f64::EPSILON);
    map.mapv(|value| value / max_value)
}

pub fn solver_metric(
    name: &'static str,
    normalized_map: &Array2<f64>,
    elapsed: Duration,
) -> SolverMetric {
    let (peak_x, peak_y, _) = peak_location(normalized_map);
    let (focus_x_mm, focus_y_mm) = physics::focus_mm();
    let peak_x_mm = peak_x as f64 * physics::DX * 1.0e3;
    let peak_y_mm = peak_y as f64 * physics::DX * 1.0e3;
    let focus_error_mm =
        ((peak_x_mm - focus_x_mm).powi(2) + (peak_y_mm - focus_y_mm).powi(2)).sqrt();

    SolverMetric {
        name,
        focus_error_mm,
        peak_x_mm,
        peak_y_mm,
        focus_value: normalized_map[[physics::FOCUS_X, physics::FOCUS_Y]],
        focus_to_offaxis: focus_to_offaxis(normalized_map),
        lateral_fwhm_mm: fwhm_lateral(normalized_map),
        axial_fwhm_mm: fwhm_axial(normalized_map),
        elapsed,
    }
}

pub fn pairwise_metrics(fields: &[SolverField]) -> Vec<PairwiseMetric> {
    let mut out = Vec::new();
    for lhs in 0..fields.len() {
        for rhs in lhs + 1..fields.len() {
            out.push(pairwise_metric(&fields[lhs], &fields[rhs]));
        }
    }
    out
}

pub fn axial_fields_from_maps(fields: &[SolverField]) -> Vec<AxialField> {
    fields
        .iter()
        .map(|field| AxialField {
            name: field.name,
            normalized_peak: axial_values(&field.normalized_peak),
            elapsed: field.elapsed,
        })
        .collect()
}

pub fn axial_pairwise_metrics(fields: &[AxialField]) -> Vec<PairwiseMetric> {
    let mut out = Vec::new();
    for lhs in 0..fields.len() {
        for rhs in lhs + 1..fields.len() {
            out.push(axial_pairwise_metric(&fields[lhs], &fields[rhs]));
        }
    }
    out
}

pub fn profile_sets(fields: &[SolverField], axial_fields: &[AxialField]) -> Vec<ProfileSet> {
    let mut profiles = fields
        .iter()
        .map(|field| ProfileSet {
            name: field.name,
            axial: axial_profile_from_values(&axial_values(&field.normalized_peak)),
            lateral: lateral_profile(&field.normalized_peak),
        })
        .collect::<Vec<_>>();

    for field in axial_fields {
        if fields.iter().any(|map_field| map_field.name == field.name) {
            continue;
        }
        profiles.push(ProfileSet {
            name: field.name,
            axial: axial_profile_from_values(&field.normalized_peak),
            lateral: Vec::new(),
        });
    }

    profiles
}

pub fn write_metrics_csv(path: &Path, output: &WaterTankOutput) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "kind,name,lhs,rhs,metric,value")?;
    for metric in &output.solver_metrics {
        writeln!(
            file,
            "solver,{name},,,focus_error_mm,{:.12e}",
            metric.focus_error_mm,
            name = metric.name
        )?;
        writeln!(
            file,
            "solver,{name},,,peak_x_mm,{:.12e}",
            metric.peak_x_mm,
            name = metric.name
        )?;
        writeln!(
            file,
            "solver,{name},,,peak_y_mm,{:.12e}",
            metric.peak_y_mm,
            name = metric.name
        )?;
        writeln!(
            file,
            "solver,{name},,,focus_value,{:.12e}",
            metric.focus_value,
            name = metric.name
        )?;
        writeln!(
            file,
            "solver,{name},,,focus_to_offaxis,{:.12e}",
            metric.focus_to_offaxis,
            name = metric.name
        )?;
        writeln!(
            file,
            "solver,{name},,,lateral_fwhm_mm,{:.12e}",
            metric.lateral_fwhm_mm,
            name = metric.name
        )?;
        writeln!(
            file,
            "solver,{name},,,axial_fwhm_mm,{:.12e}",
            metric.axial_fwhm_mm,
            name = metric.name
        )?;
        writeln!(
            file,
            "solver,{name},,,elapsed_ms,{:.12e}",
            metric.elapsed.as_secs_f64() * 1.0e3,
            name = metric.name
        )?;
    }
    for field in &output.axial_fields {
        let peak_y = axial_peak_location(&field.normalized_peak);
        let peak_y_mm = peak_y as f64 * physics::DX * 1.0e3;
        let focus_y_mm = physics::FOCUS_Y as f64 * physics::DX * 1.0e3;
        writeln!(
            file,
            "axial_solver,{name},,,focus_error_mm,{:.12e}",
            (peak_y_mm - focus_y_mm).abs(),
            name = field.name
        )?;
        writeln!(
            file,
            "axial_solver,{name},,,peak_y_mm,{:.12e}",
            peak_y_mm,
            name = field.name
        )?;
        writeln!(
            file,
            "axial_solver,{name},,,focus_value,{:.12e}",
            field.normalized_peak[physics::FOCUS_Y],
            name = field.name
        )?;
        writeln!(
            file,
            "axial_solver,{name},,,axial_fwhm_mm,{:.12e}",
            fwhm_around(&field.normalized_peak, physics::FOCUS_Y) * physics::DX * 1.0e3,
            name = field.name
        )?;
        writeln!(
            file,
            "axial_solver,{name},,,elapsed_ms,{:.12e}",
            field.elapsed.as_secs_f64() * 1.0e3,
            name = field.name
        )?;
    }
    for metric in &output.pairwise_metrics {
        writeln!(
            file,
            "pair,,{},{},normalized_l2,{:.12e}",
            metric.lhs, metric.rhs, metric.normalized_l2
        )?;
        writeln!(
            file,
            "pair,,{},{},correlation,{:.12e}",
            metric.lhs, metric.rhs, metric.correlation
        )?;
    }
    for metric in &output.axial_pairwise_metrics {
        writeln!(
            file,
            "axial_pair,,{},{},normalized_l2,{:.12e}",
            metric.lhs, metric.rhs, metric.normalized_l2
        )?;
        writeln!(
            file,
            "axial_pair,,{},{},correlation,{:.12e}",
            metric.lhs, metric.rhs, metric.correlation
        )?;
    }
    Ok(())
}

pub fn write_profiles_csv(path: &Path, output: &WaterTankOutput) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "field,axis,coordinate_mm,normalized_peak")?;
    for profile in &output.profiles {
        for &(coordinate, value) in &profile.axial {
            writeln!(
                file,
                "{},axial,{coordinate:.12e},{value:.12e}",
                profile.name
            )?;
        }
        for &(coordinate, value) in &profile.lateral {
            writeln!(
                file,
                "{},lateral,{coordinate:.12e},{value:.12e}",
                profile.name
            )?;
        }
    }
    Ok(())
}

fn pairwise_metric(lhs: &SolverField, rhs: &SolverField) -> PairwiseMetric {
    let xr = x_range();
    let yr = y_range();
    let mut diff_sq = 0.0;
    let mut lhs_sq = 0.0;
    let mut rhs_sq = 0.0;
    let mut lhs_sum = 0.0;
    let mut rhs_sum = 0.0;
    let mut cross_sum = 0.0;
    let mut count = 0.0;

    for i in xr {
        for j in yr.clone() {
            let a = lhs.normalized_peak[[i, j]];
            let b = rhs.normalized_peak[[i, j]];
            let diff = a - b;
            diff_sq += diff * diff;
            lhs_sq += a * a;
            rhs_sq += b * b;
            lhs_sum += a;
            rhs_sum += b;
            cross_sum += a * b;
            count += 1.0;
        }
    }

    let lhs_var = (lhs_sq - lhs_sum * lhs_sum / count).max(0.0);
    let rhs_var = (rhs_sq - rhs_sum * rhs_sum / count).max(0.0);
    let covariance = cross_sum - lhs_sum * rhs_sum / count;
    let correlation = if lhs_var > 0.0 && rhs_var > 0.0 {
        covariance / (lhs_var * rhs_var).sqrt()
    } else {
        1.0
    };

    PairwiseMetric {
        lhs: lhs.name,
        rhs: rhs.name,
        normalized_l2: diff_sq.sqrt() / lhs_sq.sqrt().max(rhs_sq.sqrt()).max(f64::EPSILON),
        correlation,
    }
}

fn axial_pairwise_metric(lhs: &AxialField, rhs: &AxialField) -> PairwiseMetric {
    let yr = y_range();
    let mut diff_sq = 0.0;
    let mut lhs_sq = 0.0;
    let mut rhs_sq = 0.0;
    let mut lhs_sum = 0.0;
    let mut rhs_sum = 0.0;
    let mut cross_sum = 0.0;
    let mut count = 0.0;

    for j in yr {
        let a = lhs.normalized_peak[j];
        let b = rhs.normalized_peak[j];
        let diff = a - b;
        diff_sq += diff * diff;
        lhs_sq += a * a;
        rhs_sq += b * b;
        lhs_sum += a;
        rhs_sum += b;
        cross_sum += a * b;
        count += 1.0;
    }

    let lhs_var = (lhs_sq - lhs_sum * lhs_sum / count).max(0.0);
    let rhs_var = (rhs_sq - rhs_sum * rhs_sum / count).max(0.0);
    let covariance = cross_sum - lhs_sum * rhs_sum / count;
    let correlation = if lhs_var > 0.0 && rhs_var > 0.0 {
        covariance / (lhs_var * rhs_var).sqrt()
    } else {
        1.0
    };

    PairwiseMetric {
        lhs: lhs.name,
        rhs: rhs.name,
        normalized_l2: diff_sq.sqrt() / lhs_sq.sqrt().max(rhs_sq.sqrt()).max(f64::EPSILON),
        correlation,
    }
}

fn peak_location(map: &Array2<f64>) -> (usize, usize, f64) {
    let mut best = (0, 0, f64::NEG_INFINITY);
    for i in x_range() {
        for j in y_range() {
            let value = map[[i, j]];
            if value > best.2 {
                best = (i, j, value);
            }
        }
    }
    best
}

fn axial_peak_location(values: &[f64]) -> usize {
    let mut best = (0, f64::NEG_INFINITY);
    for j in y_range() {
        let value = values[j];
        if value > best.1 {
            best = (j, value);
        }
    }
    best.0
}

fn region_max(map: &Array2<f64>) -> f64 {
    let mut best = 0.0;
    for i in x_range() {
        for j in y_range() {
            best = f64::max(best, map[[i, j]].abs());
        }
    }
    best
}

fn focus_to_offaxis(map: &Array2<f64>) -> f64 {
    let focus = map[[physics::FOCUS_X, physics::FOCUS_Y]];
    let exclusion = physics::WAVELENGTH_CELLS.round() as usize;
    let mut off_axis = 0.0;
    for i in x_range() {
        if i.abs_diff(physics::FOCUS_X) > exclusion {
            off_axis = f64::max(off_axis, map[[i, physics::FOCUS_Y]]);
        }
    }
    focus / off_axis.max(f64::EPSILON)
}

fn fwhm_lateral(map: &Array2<f64>) -> f64 {
    let values = (0..physics::NX)
        .map(|i| map[[i, physics::FOCUS_Y]])
        .collect::<Vec<_>>();
    fwhm_around(&values, physics::FOCUS_X) * physics::DX * 1.0e3
}

fn fwhm_axial(map: &Array2<f64>) -> f64 {
    let values = (0..physics::NY)
        .map(|j| map[[physics::FOCUS_X, j]])
        .collect::<Vec<_>>();
    fwhm_around(&values, physics::FOCUS_Y) * physics::DX * 1.0e3
}

fn fwhm_around(values: &[f64], center: usize) -> f64 {
    if values.is_empty() || center >= values.len() {
        return 0.0;
    }
    let threshold = 0.5 * values[center].max(f64::EPSILON);
    let mut first = center;
    while first > 0 && values[first - 1] >= threshold {
        first -= 1;
    }
    let mut last = center;
    while last + 1 < values.len() && values[last + 1] >= threshold {
        last += 1;
    }
    (last - first + 1) as f64
}

fn axial_values(map: &Array2<f64>) -> Vec<f64> {
    (0..physics::NY)
        .map(|j| map[[physics::FOCUS_X, j]])
        .collect()
}

fn axial_profile_from_values(values: &[f64]) -> Vec<(f64, f64)> {
    values
        .iter()
        .enumerate()
        .map(|(j, &value)| {
            (
                (j as f64 - physics::FOCUS_Y as f64) * physics::DX * 1.0e3,
                value,
            )
        })
        .collect()
}

fn lateral_profile(map: &Array2<f64>) -> Vec<(f64, f64)> {
    (0..physics::NX)
        .map(|i| {
            let coordinate = (i as f64 - physics::FOCUS_X as f64) * physics::DX * 1.0e3;
            (coordinate, map[[i, physics::FOCUS_Y]])
        })
        .collect()
}

fn x_range() -> Range<usize> {
    physics::PML..(physics::NX - physics::PML)
}

fn y_range() -> Range<usize> {
    let start = physics::FOCUS_Y
        .saturating_sub(physics::FOCAL_WINDOW_CELLS)
        .max(physics::SOURCE_Y + 4);
    let end = (physics::FOCUS_Y + physics::FOCAL_WINDOW_CELLS + 1).min(physics::NY - physics::PML);
    start..end
}
