use super::config::KalmanFilterConfig;
use super::filter::BayesianFilter;

fn default_filter() -> BayesianFilter {
    BayesianFilter::new(&KalmanFilterConfig::default()).unwrap()
}

#[test]
fn test_bayesian_filter_creation() {
    assert!(BayesianFilter::new(&KalmanFilterConfig::default()).is_ok());
}

/// **Test: predict propagates position by velocity × dt.**
///
/// With initial state x=1 m, vx=1 m/s and dt=1 s: x_new = 2 m exactly.
#[test]
fn test_bayesian_filter_predict() {
    let mut filter = default_filter();
    filter.state[0] = 1.0;
    filter.state[3] = 1.0;
    filter.predict(1.0).unwrap();
    assert!(
        (filter.state[0] - 2.0).abs() < 1e-12,
        "x after predict: {}",
        filter.state[0]
    );
}

/// **Test: update moves estimate toward measurement.**
///
/// Starting at origin with measurement z = [1, 0, 0],
/// the state x estimate must become positive.
#[test]
fn test_bayesian_filter_update() {
    let mut filter = default_filter();
    filter.update(&[1.0, 0.0, 0.0]).unwrap();
    assert!(
        filter.get_state()[0] > 0.0,
        "EKF must move toward measurement"
    );
}

/// **Test: covariance decreases with repeated identical measurements.**
///
/// 20 EKF updates must strictly reduce P[0,0] from the initial value.
/// Reference: Bar-Shalom (2001) §6.2.
#[test]
fn test_ekf_covariance_decreases() {
    let cfg = KalmanFilterConfig::default()
        .with_measurement_noise(1e-6)
        .with_initial_uncertainty(1.0);
    let mut filter = BayesianFilter::new(&cfg).unwrap();
    let initial_sigma_x = filter.covariance[0];

    for _ in 0..20 {
        filter.update(&[1.0, 0.0, 0.0]).unwrap();
    }

    let final_sigma_x = filter.covariance[0];
    assert!(
        final_sigma_x < initial_sigma_x,
        "P[0,0] must decrease after 20 measurements: initial={initial_sigma_x:.6}, final={final_sigma_x:.6}"
    );
}

/// **Test: EKF converges to stationary target after repeated measurements.**
///
/// After 50 updates toward z = [2, 3, 1], estimated position must be
/// within 0.01 m of the true target in all axes.
#[test]
fn test_ekf_converges_to_stationary_target() {
    let cfg = KalmanFilterConfig::default()
        .with_measurement_noise(1e-6)
        .with_initial_uncertainty(0.1);
    let mut filter = BayesianFilter::new(&cfg).unwrap();
    let target = [2.0, 3.0, 1.0];

    for _ in 0..50 {
        filter.update(&target).unwrap();
    }

    let est = filter.get_state();
    for (i, (&e, &t)) in est.iter().zip(target.iter()).enumerate() {
        assert!(
            (e - t).abs() < 0.01,
            "axis {i}: estimate {e:.6} m, target {t:.6} m (error {:.6} m > 0.01 m)",
            (e - t).abs()
        );
    }
}

/// **Test: position uncertainty is independent per axis (J-form preserves PSD).**
///
/// All diagonal elements P[i,i] must remain non-negative after updates.
#[test]
fn test_ekf_covariance_psd_preserved() {
    let cfg = KalmanFilterConfig::default().with_measurement_noise(0.001);
    let mut filter = BayesianFilter::new(&cfg).unwrap();

    for _ in 0..10 {
        filter.update(&[1.0, 1.0, 0.0]).unwrap();
    }

    for i in 0..6 {
        assert!(
            filter.covariance[i * 6 + i] >= 0.0,
            "P[{i},{i}] = {:.3e} must be non-negative (Joseph form)",
            filter.covariance[i * 6 + i]
        );
    }
}

/// **Test: predict grows covariance (process noise)**
///
/// After a predict step with dt>0, P[0,0] must be strictly larger than before.
#[test]
fn test_predict_increases_covariance() {
    let mut filter = default_filter();
    let p0 = filter.covariance[0];
    filter.predict(0.1).unwrap();
    let p1 = filter.covariance[0];
    assert!(
        p1 > p0,
        "P[0,0] must increase after predict: before={p0:.6}, after={p1:.6}"
    );
}

#[test]
fn test_kalman_filter_config_builder() {
    let config = KalmanFilterConfig::default()
        .with_process_noise(0.05)
        .with_measurement_noise(0.002);
    assert_eq!(config.process_noise, 0.05);
    assert_eq!(config.measurement_noise, 0.002);
}
