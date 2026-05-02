use ndarray::Array3;

use crate::core::error::KwaversResult;

use super::BurnDasBeamformer;

/// CPU convenience function for Burn-based beamforming
///
/// Uses NdArray backend for CPU-only operation without GPU dependencies.
pub fn beamform_cpu(
    rf_data: &Array3<f64>,
    sensor_positions: &ndarray::Array2<f64>,
    focal_points: &ndarray::Array2<f64>,
    apodization: Option<&[f64]>,
    sampling_rate: f64,
    sound_speed: f64,
) -> KwaversResult<Array3<f64>> {
    use burn::backend::NdArray;
    let device = Default::default();
    let beamformer: BurnDasBeamformer<NdArray> = BurnDasBeamformer::new(device);
    beamformer.beamform(
        rf_data,
        sensor_positions,
        focal_points,
        apodization,
        sampling_rate,
        sound_speed,
    )
}
