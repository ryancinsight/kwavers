use super::*;

#[test]
fn latency_stats_calculation() {
    let mut stats = RecoveryLatencyStats::new();
    stats.add_sample(Duration::from_millis(10));
    stats.add_sample(Duration::from_millis(20));
    stats.add_sample(Duration::from_millis(30));

    assert_eq!(stats.mean().as_millis(), 20);
    assert!(stats.std_dev().as_millis() > 0);
    assert_eq!(stats.n, 3);
    assert_eq!(stats.min().as_millis(), 10);
    assert_eq!(stats.max().as_millis(), 30);
}

#[test]
fn recovery_stats_success_rate() {
    let mut stats = RecoveryStats::default();
    stats.record_trial(true, Duration::from_millis(10), true);
    stats.record_trial(true, Duration::from_millis(15), true);
    stats.record_trial(false, Duration::from_millis(20), true);

    let success_rate = stats.success_rate();
    assert!((success_rate - 2.0 / 3.0).abs() < 1e-10);

    let failure_rate = stats.failure_rate();
    assert!((failure_rate - 1.0 / 3.0).abs() < 1e-10);
}

#[test]
fn stability_criteria() {
    let stable = StabilityReport {
        total_steps: 1_000_000,
        faults_injected: 10,
        successful_recoveries: 10,
        mtbf_seconds: 1000.0,
        mttr_microseconds: 10_000.0,
        energy_violations: 0,
        cfl_violations: 0,
        convergence_failures: 0,
        state_integrity: 1.0,
        throughput_cv: 0.02,
        memory_trend: 0.0,
        stable_percentage: 100.0,
    };
    assert!(stable.is_stable());
    assert!(stable.stability_score() > 0.9);

    let unstable = StabilityReport {
        total_steps: 1_000_000,
        faults_injected: 1000,
        successful_recoveries: 500,
        mtbf_seconds: 10.0,
        mttr_microseconds: 500_000.0,
        energy_violations: 100,
        cfl_violations: 50,
        convergence_failures: 20,
        state_integrity: 0.5,
        throughput_cv: 0.2,
        memory_trend: 100.0,
        stable_percentage: 50.0,
    };
    assert!(!unstable.is_stable());
    assert!(unstable.stability_score() < 0.5);
}

#[test]
fn fairness_index_calculation() {
    let fair = vec![100.0, 100.0, 100.0];
    assert!((ContentionReport::calculate_fairness(&fair) - 1.0).abs() < 1e-10);

    let unfair = vec![100.0, 50.0, 50.0];
    let fairness = ContentionReport::calculate_fairness(&unfair);
    assert!(fairness < 1.0);
    assert!(fairness > 0.0);

    let min_fair = vec![100.0, 0.0, 0.0];
    let min_fairness = ContentionReport::calculate_fairness(&min_fair);
    assert!(min_fairness >= 0.33 && min_fairness <= 0.34);
}

#[test]
fn confidence_interval_calculation() {
    let mut stats = RecoveryStats::default();

    for _ in 0..950 {
        stats.record_trial(true, Duration::from_millis(10), true);
    }
    for _ in 0..50 {
        stats.record_trial(false, Duration::from_millis(10), true);
    }

    let (lower, upper) = stats.confidence_interval();
    assert!(lower < 0.95);
    assert!(upper > 0.95);
    assert!(lower > 0.90);

    assert!(lower >= 0.0 && lower <= 1.0);
    assert!(upper >= 0.0 && upper <= 1.0);
    assert!(lower <= upper);
}

#[test]
fn wilson_interval_properties() {
    let mut all_success = RecoveryStats::default();
    for _ in 0..100 {
        all_success.record_trial(true, Duration::from_millis(10), true);
    }
    let (lo, _hi) = all_success.confidence_interval();
    assert!(lo > 0.95);

    let mut all_failure = RecoveryStats::default();
    for _ in 0..100 {
        all_failure.record_trial(false, Duration::from_millis(10), true);
    }
    let (lo, hi) = all_failure.confidence_interval();
    assert!(hi < 0.05);
    assert_eq!(lo, 0.0);
}

#[test]
fn stats_merge() {
    use crate::core::fault_injection::scenario::{FaultInjectionScenario, InjectionTiming};

    let mut stats1 = RecoveryStats::new(FaultInjectionScenario::GpuOomSudden {
        allocation_size_bytes: 1024,
        timing: InjectionTiming::Immediate,
    });
    stats1.record_trial(true, Duration::from_millis(10), true);

    let mut stats2 = RecoveryStats::default();
    stats2.record_trial(false, Duration::from_millis(20), true);

    stats1.merge(&stats2);

    assert_eq!(stats1.distribution.total_attempts, 2);
    assert_eq!(stats1.distribution.successful_recoveries, 1);
    assert_eq!(stats1.distribution.failed_recoveries, 1);
}
