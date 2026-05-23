use crate::clinical::therapy::theranostic_guidance::nonlinear3d::types::{GridIndex, SourceDomain};
use crate::clinical::therapy::theranostic_guidance::{AnatomyKind, Point3};
use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};

use super::calibration::{calibrated_source_scale, SourceCalibrationInput};
use super::{
    flat_index, forward_with_schedule, time_schedule, ForwardInput, Nonlinear3dAperture,
    Nonlinear3dConfig, SourceEncoding,
};

#[test]
fn source_calibration_never_overdrives_configured_transducer_pressure() {
    let n = 12;
    let cells = n * n * n;
    let spacing_m = 1.0e-3;
    let speed = vec![SOUND_SPEED_WATER_SIM; cells];
    let density = vec![DENSITY_WATER_NOMINAL; cells];
    let attenuation = vec![0.0; cells];
    let attenuation_y = vec![1.0; cells];
    let beta = vec![0.0; cells];
    let focus = GridIndex { x: 6, y: 6, z: 8 };
    let mut target = vec![false; cells];
    target[flat_index(focus, n)] = true;
    let aperture = Nonlinear3dAperture {
        sources: vec![
            GridIndex { x: 3, y: 3, z: 2 },
            GridIndex { x: 8, y: 3, z: 2 },
            GridIndex { x: 3, y: 8, z: 2 },
            GridIndex { x: 8, y: 8, z: 2 },
        ],
        receivers: vec![focus],
        therapy_points_m: vec![Point3 {
            x_m: 0.0,
            y_m: 0.0,
            z_m: 0.0,
        }],
        receiver_points_m: vec![Point3 {
            x_m: 0.0,
            y_m: 0.0,
            z_m: 0.0,
        }],
        model_name: "calibration_test".to_owned(),
        source_domain: SourceDomain::TissueBoundary,
        focus,
    };
    let mut config = Nonlinear3dConfig::new(AnatomyKind::Liver);
    config.frequency_hz = 300_000.0;
    config.source_pressure_pa = 2.0e5;
    config.cycles = 2.0;
    config.cfl = 0.4;
    let schedule = time_schedule(&speed, n, spacing_m, &config);
    let body = vec![true; cells];
    let source_scale = calibrated_source_scale(SourceCalibrationInput {
        background_speed: &speed,
        density: &density,
        attenuation_alpha0: &attenuation,
        attenuation_y: &attenuation_y,
        body: &body,
        target: &target,
        n,
        spacing_m,
        aperture: &aperture,
        config: &config,
        schedule,
    });
    let result = forward_with_schedule(ForwardInput {
        speed: &speed,
        density: &density,
        beta: &beta,
        attenuation_np_per_m_mhz: Some(&attenuation),
        attenuation_power_law_y: Some(&attenuation_y),
        source_body_mask: Some(&body),
        n,
        spacing_m,
        aperture: &aperture,
        config: &config,
        schedule,
        encoding: SourceEncoding { index: 0, count: 1 },
        source_scale,
        retain_history: false,
    });
    let calibrated_peak = result.peak_pressure[flat_index(focus, n)];

    assert!(
        source_scale <= 1.0,
        "source calibration may attenuate but must not amplify a configured transducer drive; got scale={source_scale}"
    );
    assert!(
        calibrated_peak <= config.source_pressure_pa * (1.0 + 1.0e-12),
        "calibrated target peak {calibrated_peak} Pa must not exceed configured drive {} Pa",
        config.source_pressure_pa
    );
}
