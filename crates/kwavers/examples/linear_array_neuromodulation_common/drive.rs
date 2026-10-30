//! Drive stages: the array's analytical gain and the drive voltage each focal
//! target requires.

use super::metrics::{self, mechanical_index};
use super::{
    analytic, plots, LinearArraySpec, ARRAY_APERTURE_M, ARRAY_ELEVATION_M, FOCUS_FOCI_MM,
    FREQUENCY_HZ, TARGET_DRIVE_VP, TARGET_PRESSURE_PA,
};
use anyhow::Result;
use eunomia::FloatElement;
use plotters::prelude::{BLACK, BLUE, RED};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

/// Analytical gains at 60, 70 and 80 mm, the axial profile and the focal-plane
/// cuts; returns the 70 mm gain.
///
/// # Errors
/// Returns an error when the oracle, a plot, or an output file fails.
pub fn stage_gain(out: &Path) -> Result<f64> {
    let array = LinearArraySpec::at(ARRAY_APERTURE_M, ARRAY_ELEVATION_M, [0.0, 0.0, 0.0]);
    println!(
        "array: {} elements at pitch {:.4} mm, aperture {:.1} x {:.1} mm",
        array.element_count(),
        array.pitch_m * 1.0e3,
        ARRAY_APERTURE_M * 1.0e3,
        ARRAY_ELEVATION_M * 1.0e3
    );
    let mut csv = String::from("focal_mm,gain_pa_per_pa,surface_kpa_for_120kpa,mi_at_120kpa\n");
    let mut gain_70 = f64::NAN;
    for focal_mm in [60.0_f64, 70.0, 80.0] {
        let focus = [0.0, 0.0, focal_mm * 1.0e-3];
        let gain = analytic::focal_amplitude(&array, focus, 1.0)?;
        if (focal_mm - 70.0).abs() < f64::EPSILON {
            gain_70 = gain;
        }
        let required = metrics::required_surface_pa(TARGET_PRESSURE_PA, gain);
        writeln!(
            csv,
            "{focal_mm},{gain:.6},{:.2},{:.4}",
            required * 1.0e-3,
            mechanical_index(TARGET_PRESSURE_PA, FREQUENCY_HZ)
        )?;
        println!(
            "  focus {focal_mm} mm: gain {gain:.4} Pa/Pa, surface for 120 kPa {:.1} kPa",
            required * 1.0e-3
        );
    }
    fs::write(out.join("array_gain.csv"), csv)?;

    // Axial gain along the broadside axis, oracle only, with attenuation.
    let depths: Vec<f64> = (0..=400)
        .map(|index| 30.0 + 0.15 * f64::from_count(index))
        .collect();
    let points: Vec<[f64; 3]> = depths.iter().map(|&z| [0.0, 0.0, z * 1.0e-3]).collect();
    let apertures = analytic::element_apertures(&array, [0.0, 0.0, 0.07], 1.0)?;
    let gains = analytic::amplitudes_at(&points, &apertures)?;
    plots::write_profiles(
        &out.join("fig4_oracle_axial_gain.png"),
        "Array axial gain (oracle, 70 mm focus, broadside)",
        "depth z (mm)",
        "pressure gain (Pa/Pa)",
        &depths,
        &[("focal gain", BLACK, gains)],
    )?;

    // Focal-plane cuts at each focus. The 1.5 mm elevation is unfocused and
    // diffracts to a half-maximum width of about 40 mm, wider than the full-wave
    // grid, so its extent comes from the oracle.
    let offsets_mm: Vec<f64> = (0..=800)
        .map(|index| -40.0 + 0.1 * f64::from_count(index))
        .collect();
    let width_mm = |profile: &[f64]| metrics::fwhm(profile, 0.1e-3).unwrap_or(f64::NAN) * 1.0e3;
    let mut cuts = String::from("focal_mm,offset_mm,lateral_gain,elevation_gain\n");
    for focal_mm in FOCUS_FOCI_MM {
        let depth_m = focal_mm * 1.0e-3;
        let focused = analytic::element_apertures(&array, [0.0, 0.0, depth_m], 1.0)?;
        let cut = |axis: usize| -> Result<Vec<f64>> {
            let points: Vec<[f64; 3]> = offsets_mm
                .iter()
                .map(|&offset| {
                    let mut point = [0.0, 0.0, depth_m];
                    point[axis] = offset * 1.0e-3;
                    point
                })
                .collect();
            analytic::amplitudes_at(&points, &focused)
        };
        let lateral = cut(0)?;
        let elevation = cut(1)?;
        println!(
            "  {focal_mm} mm focal spot (oracle): FWHM lateral {:.2} mm, elevation {:.1} mm",
            width_mm(&lateral),
            width_mm(&elevation)
        );
        for ((offset, lat), elev) in offsets_mm.iter().zip(&lateral).zip(&elevation) {
            writeln!(cuts, "{focal_mm},{offset:.1},{lat:.6},{elev:.6}")?;
        }
        if (focal_mm - 70.0).abs() < f64::EPSILON {
            plots::write_profiles(
                &out.join("fig7_focal_spot_70mm.png"),
                "Focal spot at 70 mm (oracle): narrow along the array, wide in elevation",
                "offset from axis (mm)",
                "pressure gain (Pa/Pa)",
                &offsets_mm,
                &[
                    ("along the array", BLUE, lateral),
                    ("elevation", RED, elevation),
                ],
            )?;
        }
    }
    fs::write(out.join("focal_plane.csv"), cuts)?;
    Ok(gain_70)
}

/// Drive-voltage scan of the surface sensitivity each focal target requires.
///
/// # Errors
/// Returns an error when a plot or an output file fails.
pub fn stage_vp(out: &Path, gain_70: f64) -> Result<()> {
    let implied_sensitivity = TARGET_PRESSURE_PA / (gain_70 * TARGET_DRIVE_VP);
    println!(
        "implied surface sensitivity of the 120 kPa target at 70 mm and 30 Vp: {implied_sensitivity:.1} Pa/V"
    );
    let vps: Vec<f64> = (0..=27)
        .map(|index| 30.0 + 10.0 * f64::from_count(index))
        .collect();
    let mut csv = String::from("vp_volts,target_kpa,required_sensitivity_pa_per_v,focal_kpa_at_implied_sensitivity,surface_kpa,surface_shock_distance_mm\n");
    let targets = [120.0e3_f64, 500.0e3, 1.0e6, 2.0e6];
    let mut series = Vec::new();
    for &target in &targets {
        let mut required = Vec::new();
        for &vp in &vps {
            let sensitivity = target / (gain_70 * vp);
            required.push(sensitivity);
        }
        series.push(required);
    }
    for (row, &vp) in vps.iter().enumerate() {
        for (column, &target) in targets.iter().enumerate() {
            // Distortion accrues where the pressure is highest, at the array face.
            let surface = implied_sensitivity * vp;
            writeln!(
                csv,
                "{vp:.1},{:.0},{:.2},{:.1},{:.1},{:.2}",
                target * 1.0e-3,
                series[column][row],
                gain_70 * surface * 1.0e-3,
                surface * 1.0e-3,
                metrics::shock_distance_m(surface, FREQUENCY_HZ) * 1.0e3
            )?;
        }
    }
    fs::write(out.join("vp_scan.csv"), csv)?;

    let labels = ["120 kPa", "500 kPa", "1 MPa", "2 MPa"];
    let colours = [BLUE, RED, BLACK, plotters::prelude::GREEN];
    let named: Vec<(&str, plotters::style::RGBColor, Vec<f64>)> = labels
        .iter()
        .zip(colours)
        .zip(series)
        .map(|((&label, colour), values)| (label, colour, values))
        .collect();
    plots::write_profiles(
        &out.join("fig5_vp_required_sensitivity.png"),
        "Surface sensitivity needed for each focal target, array at 70 mm",
        "drive voltage Vp (V)",
        "required surface pressure per volt (Pa/V)",
        &vps,
        &named,
    )?;

    // Linear focal pressure at the target-implied sensitivity, with the targets.
    let focal_kpa: Vec<f64> = vps
        .iter()
        .map(|&vp| gain_70 * implied_sensitivity * vp * 1.0e-3)
        .collect();
    let constant = |kpa: f64| vec![kpa; vps.len()];
    plots::write_profiles(
        &out.join("fig6_vp_focal_pressure.png"),
        "Focal pressure at 70 mm vs drive, target-implied sensitivity (linear)",
        "drive voltage Vp (V)",
        "focal pressure (kPa)",
        &vps,
        &[
            ("focal pressure", BLACK, focal_kpa),
            ("120 kPa", BLUE, constant(120.0)),
            ("500 kPa", RED, constant(500.0)),
            ("1 MPa", plotters::prelude::GREEN, constant(1000.0)),
        ],
    )?;
    Ok(())
}
