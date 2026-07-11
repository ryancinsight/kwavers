//! Focused ultrasound water-tank comparison with plots.
//!
//! Outputs:
//! - `target/focused_water_tank/focused_water_tank.png`
//! - `target/focused_water_tank/focused_water_tank_metrics.csv`
//! - `target/focused_water_tank/focused_water_tank_profiles.csv`
//!
//! The fixture drives a phased line aperture in homogeneous water and compares
//! gated peak-pressure maps from FDTD+CPML, PSTD+CPML, DG-2D, and DG-3D
//! against the analytical focused-array phase law. DG also includes the legacy
//! 1-D axial profile diagnostic to keep line-based acoustic RHS regressions
//! visible beside the tensor-product maps.
//!
//! References:
//! - O'Neil (1949): focused circular piston pressure fields.
//! - Yee (1966): staggered-grid finite differences.
//! - Treeby and Cox (2010): k-space pseudospectral acoustic modeling.
//! - Liu (1997): pseudospectral time-domain acoustic propagation.
//! - Hesthaven and Warburton (2008): nodal discontinuous Galerkin methods.
//! - Cockburn and Shu (2001): Runge-Kutta DG stability and conservation.

#[path = "focused_water_tank_common/mod.rs"]
mod focused_water_tank_common;

use anyhow::Result;
use focused_water_tank_common::{
    run_comparison, write_metrics_csv, write_plot, write_profiles_csv, OUT_DIR,
};
use std::fs;
use std::path::PathBuf;

fn main() -> Result<()> {
    let output = run_comparison()?;
    let out_dir = PathBuf::from(OUT_DIR);
    fs::create_dir_all(&out_dir)?;

    let png_path = out_dir.join("focused_water_tank.png");
    let metrics_path = out_dir.join("focused_water_tank_metrics.csv");
    let profiles_path = out_dir.join("focused_water_tank_profiles.csv");

    write_plot(&png_path, &output)?;
    write_metrics_csv(&metrics_path, &output)?;
    write_profiles_csv(&profiles_path, &output)?;

    println!("Focused ultrasound water-tank comparison");
    println!("png: {}", png_path.display());
    println!("metrics: {}", metrics_path.display());
    println!("profiles: {}", profiles_path.display());
    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "field", "focus_mm", "peak_x", "peak_y", "lat_fwhm", "ax_fwhm"
    );
    for metric in &output.solver_metrics {
        println!(
            "{:<10} {:>12.4} {:>12.3} {:>12.3} {:>12.4} {:>12.4}",
            metric.name,
            metric.focus_error_mm,
            metric.peak_x_mm,
            metric.peak_y_mm,
            metric.lateral_fwhm_mm,
            metric.axial_fwhm_mm
        );
    }
    println!();
    println!("{:<20} {:>14} {:>14}", "pair", "norm_l2", "corr");
    for pair in &output.pairwise_metrics {
        println!(
            "{:<20} {:>14.6e} {:>14.6}",
            format!("{} vs {}", pair.lhs, pair.rhs),
            pair.normalized_l2,
            pair.correlation
        );
    }
    println!();
    println!("{:<20} {:>14} {:>14}", "axial pair", "norm_l2", "corr");
    for pair in &output.axial_pairwise_metrics {
        println!(
            "{:<20} {:>14.6e} {:>14.6}",
            format!("{} vs {}", pair.lhs, pair.rhs),
            pair.normalized_l2,
            pair.correlation
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::focused_water_tank_common::physics;
    use super::*;

    #[test]
    fn focused_array_phase_law_aligns_at_target() {
        let residual = physics::focus_phase_residual_rad();
        assert!(
            residual < 1.0e-12,
            "phase residual at focus must be roundoff-bounded, got {residual:e}"
        );
    }

    #[test]
    fn focused_source_spans_the_thin_slab_in_fortran_row_order() {
        let source = physics::focused_source();
        let mask = source.p_mask.as_ref().expect("pressure mask required");
        let signal = source.p_signal.as_ref().expect("pressure signal required");
        let elements = physics::elements();
        let expected_rows = elements.len() * physics::NZ;

        let mask_sum: f64 = mask.iter().sum();
        assert_eq!(mask_sum, expected_rows as f64);
        assert_eq!(signal.shape(), [expected_rows, physics::NT]);

        for z in 0..physics::NZ {
            for (element_index, element) in elements.iter().enumerate() {
                assert_eq!(mask[[element.x, element.y, z]], 1.0);
                let row = z * elements.len() + element_index;
                let reference_row = element_index;
                let row_view = signal.index_axis::<1>(0, row).expect("signal row");
                let reference_row_view = signal
                    .index_axis::<1>(0, reference_row)
                    .expect("reference row");
                let row_error = row_view
                    .iter()
                    .zip(reference_row_view.iter())
                    .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                    .fold(0.0, f64::max);
                assert!(
                    row_error < 1.0e-12,
                    "z-invariant source row {row} must match row {reference_row}, got {row_error:e}"
                );
            }
        }
    }

    #[test]
    fn water_tank_comparison_writes_nonempty_artifacts() {
        let output = run_comparison().unwrap();
        assert_eq!(output.solver_fields.len(), 6);
        assert_eq!(output.axial_fields.len(), 7);
        assert!(output
            .solver_metrics
            .iter()
            .all(|metric| metric.focus_error_mm.is_finite()));
        assert!(output
            .pairwise_metrics
            .iter()
            .all(|metric| metric.normalized_l2.is_finite() && metric.correlation.is_finite()));
        assert!(output
            .axial_pairwise_metrics
            .iter()
            .all(|metric| { metric.normalized_l2.is_finite() && metric.correlation.is_finite() }));
        let dg_axial = output
            .axial_fields
            .iter()
            .find(|field| field.name == "DG-1D axial")
            .expect("DG axial field must be present");
        assert_eq!(dg_axial.normalized_peak.len(), physics::NY);
        assert!(dg_axial.normalized_peak.iter().any(|&value| value > 0.0));
        for name in ["DG-2D", "DG-3D"] {
            let field = output
                .solver_fields
                .iter()
                .find(|field| field.name == name)
                .expect("tensor DG field must be present");
            assert_eq!(field.normalized_peak.shape(), [physics::NX, physics::NY]);
            assert!(field.normalized_peak.iter().all(|value| value.is_finite()));
            assert!(field.normalized_peak.iter().any(|&value| value > 0.0));
        }

        let out_dir = PathBuf::from("target/focused_water_tank_tests");
        fs::create_dir_all(&out_dir).unwrap();
        let png_path = out_dir.join("focused_water_tank.png");
        let metrics_path = out_dir.join("focused_water_tank_metrics.csv");
        let profiles_path = out_dir.join("focused_water_tank_profiles.csv");

        write_plot(&png_path, &output).unwrap();
        write_metrics_csv(&metrics_path, &output).unwrap();
        write_profiles_csv(&profiles_path, &output).unwrap();

        assert!(fs::metadata(png_path).unwrap().len() > 16_384);
        assert!(fs::metadata(metrics_path).unwrap().len() > 256);
        assert!(fs::metadata(profiles_path).unwrap().len() > 512);
    }
}
