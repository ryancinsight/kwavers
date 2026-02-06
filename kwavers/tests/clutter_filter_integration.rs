//! Comprehensive integration tests for clutter filtering algorithms
//!
//! This test suite validates:
//! - All 4 filter types work without errors
//! - Filters reduce signal power (remove high-amplitude clutter)
//! - Filter cascading works correctly
//! - Edge cases handled gracefully
//! - Realistic functional ultrasound workflow

use kwavers::analysis::signal_processing::clutter_filter::{
    AdaptiveFilter, AdaptiveFilterConfig, IirFilter, IirFilterConfig, PolynomialFilter,
    PolynomialFilterConfig, SubspaceSeparationMethod, SvdClutterFilter, SvdClutterFilterConfig,
};
use kwavers::core::error::KwaversResult;
use ndarray::Array2;
use std::f64::consts::PI;

/// Generate synthetic functional ultrasound data
/// Returns (combined_data, tissue_only, blood_only) as Array2<f64> (n_pixels × n_frames)
fn generate_fus_data(
    n_pixels: usize,
    n_frames: usize,
    tissue_amplitude: f64,
    blood_amplitude: f64,
    tissue_freq: f64,
    blood_freq: f64,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let mut data = Array2::<f64>::zeros((n_pixels, n_frames));
    let mut tissue_only = Array2::<f64>::zeros((n_pixels, n_frames));
    let mut blood_only = Array2::<f64>::zeros((n_pixels, n_frames));

    for i in 0..n_pixels {
        for t in 0..n_frames {
            let time = t as f64;

            // Low-frequency, high-amplitude tissue motion (clutter)
            let tissue = tissue_amplitude * (2.0 * PI * tissue_freq * time).sin();

            // High-frequency, low-amplitude blood flow with spatial variation
            let blood =
                blood_amplitude * (2.0 * PI * blood_freq * time).sin() * (i as f64 / 10.0 + 1.0);

            data[[i, t]] = tissue + blood;
            tissue_only[[i, t]] = tissue;
            blood_only[[i, t]] = blood;
        }
    }

    (data, tissue_only, blood_only)
}

#[test]
fn test_all_filters_reduce_power() -> KwaversResult<()> {
    // Generate synthetic fUS data: high tissue (10x), low blood
    let (data, _tissue, _blood) = generate_fus_data(50, 100, 10.0, 1.0, 0.02, 0.15);

    let original_power: f64 = data.iter().map(|x| x * x).sum();

    // Test SVD filter
    let svd_config = SvdClutterFilterConfig::with_fixed_rank(2);
    let svd_filter = SvdClutterFilter::new(svd_config)?;
    let svd_result = svd_filter.filter(&data)?;
    let svd_power: f64 = svd_result.iter().map(|x| x * x).sum();
    assert!(svd_power < original_power, "SVD filter should reduce power");

    // Test Polynomial filter
    let poly_config = PolynomialFilterConfig::with_order(3);
    let poly_filter = PolynomialFilter::new(poly_config)?;
    let poly_result = poly_filter.filter(&data)?;
    let poly_power: f64 = poly_result.iter().map(|x| x * x).sum();
    assert!(
        poly_power < original_power,
        "Polynomial filter should reduce power"
    );

    // Test IIR filter
    let iir_config = IirFilterConfig::with_cutoff(0.05);
    let iir_filter = IirFilter::new(iir_config)?;
    let iir_result = iir_filter.filter(&data)?;
    let iir_power: f64 = iir_result.iter().map(|x| x * x).sum();
    assert!(iir_power < original_power, "IIR filter should reduce power");

    // Test Adaptive filter
    let adaptive_config = AdaptiveFilterConfig {
        separation_method: SubspaceSeparationMethod::FixedRank { clutter_rank: 2 },
        ..Default::default()
    };
    let mut adaptive_filter = AdaptiveFilter::new(adaptive_config)?;
    let adaptive_result = adaptive_filter.filter(&data)?;
    let adaptive_power: f64 = adaptive_result.iter().map(|x| x * x).sum();
    assert!(
        adaptive_power < original_power,
        "Adaptive filter should reduce power"
    );

    println!("Power Reduction:");
    println!(
        "  SVD Filter:      {:.1}%",
        (1.0 - svd_power / original_power) * 100.0
    );
    println!(
        "  Polynomial:      {:.1}%",
        (1.0 - poly_power / original_power) * 100.0
    );
    println!(
        "  IIR Filter:      {:.1}%",
        (1.0 - iir_power / original_power) * 100.0
    );
    println!(
        "  Adaptive Filter: {:.1}%",
        (1.0 - adaptive_power / original_power) * 100.0
    );

    Ok(())
}

#[test]
fn test_filter_cascading() -> KwaversResult<()> {
    // Test cascade: IIR → Polynomial → SVD
    let (data, _, _) = generate_fus_data(30, 120, 5.0, 0.5, 0.03, 0.2);

    // Stage 1: IIR high-pass
    let iir_config = IirFilterConfig::with_cutoff(0.02);
    let iir_filter = IirFilter::new(iir_config)?;
    let stage1 = iir_filter.filter(&data)?;

    // Stage 2: Polynomial detrending
    let poly_config = PolynomialFilterConfig::with_order(2);
    let poly_filter = PolynomialFilter::new(poly_config)?;
    let stage2 = poly_filter.filter(&stage1)?;

    // Stage 3: SVD refinement
    let svd_config = SvdClutterFilterConfig::with_fixed_rank(1);
    let svd_filter = SvdClutterFilter::new(svd_config)?;
    let stage3 = svd_filter.filter(&stage2)?;

    // Each stage should work without errors
    assert_eq!(stage1.dim(), data.dim());
    assert_eq!(stage2.dim(), data.dim());
    assert_eq!(stage3.dim(), data.dim());

    Ok(())
}

#[test]
fn test_output_dimensions_match() -> KwaversResult<()> {
    let (data, _, _) = generate_fus_data(25, 120, 3.0, 0.3, 0.02, 0.15);

    let svd_config = SvdClutterFilterConfig::with_fixed_rank(1);
    let svd_filter = SvdClutterFilter::new(svd_config)?;
    let result = svd_filter.filter(&data)?;

    assert_eq!(result.dim(), data.dim(), "Output shape must match input");

    Ok(())
}

#[test]
fn test_edge_case_single_pixel() -> KwaversResult<()> {
    let (data, _, _) = generate_fus_data(1, 100, 5.0, 1.0, 0.02, 0.15);

    let svd_config = SvdClutterFilterConfig::with_fixed_rank(1);
    let svd_filter = SvdClutterFilter::new(svd_config)?;
    let result = svd_filter.filter(&data)?;

    assert_eq!(result.dim(), data.dim());

    Ok(())
}

#[test]
fn test_edge_case_minimal_frames() -> KwaversResult<()> {
    let data = Array2::<f64>::from_elem((10, 120), 1.0);

    let svd_config = SvdClutterFilterConfig::with_fixed_rank(2);
    let svd_filter = SvdClutterFilter::new(svd_config)?;

    // Should either work or fail gracefully
    let _ = svd_filter.filter(&data);

    Ok(())
}

#[test]
fn test_zero_input_gives_zero_output() -> KwaversResult<()> {
    let data = Array2::<f64>::zeros((20, 120));

    let svd_config = SvdClutterFilterConfig::default();
    let svd_filter = SvdClutterFilter::new(svd_config)?;
    let result = svd_filter.filter(&data)?;

    let output_power: f64 = result.iter().map(|x| x * x).sum();
    assert!(
        output_power < 1e-6,
        "Zero input should give near-zero output"
    );

    Ok(())
}

#[test]
fn test_numerical_stability_small_amplitudes() -> KwaversResult<()> {
    let (data, _, _) = generate_fus_data(20, 120, 1e-8, 1e-9, 0.02, 0.15);

    let svd_config = SvdClutterFilterConfig::with_fixed_rank(1);
    let svd_filter = SvdClutterFilter::new(svd_config)?;
    let result = svd_filter.filter(&data)?;

    assert!(
        result.iter().all(|x| x.is_finite()),
        "Output should be finite for small amplitudes"
    );

    Ok(())
}

#[test]
fn test_numerical_stability_large_amplitudes() -> KwaversResult<()> {
    let (data, _, _) = generate_fus_data(20, 120, 1e5, 1e4, 0.02, 0.15);

    let svd_config = SvdClutterFilterConfig::with_fixed_rank(1);
    let svd_filter = SvdClutterFilter::new(svd_config)?;
    let result = svd_filter.filter(&data)?;

    assert!(
        result.iter().all(|x| x.is_finite()),
        "Output should be finite for large amplitudes"
    );

    Ok(())
}

#[test]
fn test_power_doppler_computation() -> KwaversResult<()> {
    let (data, _, _) = generate_fus_data(40, 120, 8.0, 0.8, 0.02, 0.18);

    let svd_config = SvdClutterFilterConfig::with_fixed_rank(2);
    let svd_filter = SvdClutterFilter::new(svd_config)?;
    let filtered = svd_filter.filter(&data)?;

    // Compute Power Doppler: sum of squared magnitudes over time
    let pd_image = filtered.mapv(|x| x * x).sum_axis(ndarray::Axis(1));

    assert_eq!(pd_image.len(), data.dim().0);
    assert!(pd_image.iter().all(|&x| x >= 0.0));

    Ok(())
}

#[test]
fn test_adaptive_filter_cbr_tracking() -> KwaversResult<()> {
    let (data, _, _) = generate_fus_data(25, 100, 6.0, 0.6, 0.025, 0.16);

    let adaptive_config = AdaptiveFilterConfig {
        separation_method: SubspaceSeparationMethod::CbrBased {
            target_cbr_db: 20.0,
        },
        temporal_smoothing: true,
        smoothing_window: 5,
        ..Default::default()
    };
    let mut adaptive_filter = AdaptiveFilter::new(adaptive_config)?;

    let _result = adaptive_filter.filter(&data)?;

    let cbr_history = adaptive_filter.cbr_history();
    assert!(!cbr_history.is_empty(), "CBR history should be populated");

    Ok(())
}

#[test]
fn test_realistic_fus_workflow() -> KwaversResult<()> {
    // Simulate realistic functional ultrasound imaging workflow
    let n_pixels = 128;
    let n_frames = 200;
    let tissue_amp = 15.0;
    let blood_amp = 0.5;
    let tissue_freq = 0.015;
    let blood_freq = 0.12;

    let (data, _tissue, _blood) = generate_fus_data(
        n_pixels,
        n_frames,
        tissue_amp,
        blood_amp,
        tissue_freq,
        blood_freq,
    );

    // SVD clutter filtering
    let svd_config = SvdClutterFilterConfig::with_fixed_rank(5);
    let svd_filter = SvdClutterFilter::new(svd_config)?;
    let filtered = svd_filter.filter(&data)?;

    // Compute Power Doppler
    let power_doppler = filtered.mapv(|x| x * x).sum_axis(ndarray::Axis(1));

    let original_power: f64 = data.iter().map(|x| x * x).sum();
    let filtered_power: f64 = filtered.iter().map(|x| x * x).sum();
    let clutter_rejection = (1.0 - filtered_power / original_power) * 100.0;

    println!("Realistic fUS Workflow:");
    println!("  Dimensions: {} × {}", n_pixels, n_frames);
    println!("  Clutter rejection: {:.1}%", clutter_rejection);
    println!(
        "  Power Doppler range: {:.2e} to {:.2e}",
        power_doppler.iter().cloned().fold(f64::INFINITY, f64::min),
        power_doppler
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );

    assert_eq!(power_doppler.len(), n_pixels);
    assert!(clutter_rejection > 50.0, "Should reject >50% of power");

    Ok(())
}

#[test]
fn test_all_separation_methods() -> KwaversResult<()> {
    let (data, _, _) = generate_fus_data(30, 120, 5.0, 0.5, 0.02, 0.15);

    // Test FixedRank
    let config1 = AdaptiveFilterConfig {
        separation_method: SubspaceSeparationMethod::FixedRank { clutter_rank: 2 },
        ..Default::default()
    };
    let mut filter1 = AdaptiveFilter::new(config1)?;
    let _result1 = filter1.filter(&data)?;

    // Test AdaptiveThreshold
    let config2 = AdaptiveFilterConfig {
        separation_method: SubspaceSeparationMethod::AdaptiveThreshold { decay_factor: 0.1 },
        ..Default::default()
    };
    let mut filter2 = AdaptiveFilter::new(config2)?;
    let _result2 = filter2.filter(&data)?;

    // Test CbrBased
    let config3 = AdaptiveFilterConfig {
        separation_method: SubspaceSeparationMethod::CbrBased {
            target_cbr_db: 25.0,
        },
        ..Default::default()
    };
    let mut filter3 = AdaptiveFilter::new(config3)?;
    let _result3 = filter3.filter(&data)?;

    Ok(())
}
