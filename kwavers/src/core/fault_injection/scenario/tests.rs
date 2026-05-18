use super::*;

#[test]
fn scenario_recovery_expectations() {
    let oom_gradual = FaultInjectionScenario::GpuOomGradual {
        leak_rate_bytes: 1024,
        total_leak_bytes: 10240,
        timing: InjectionTiming::Immediate,
    };
    assert_eq!(
        oom_gradual.recovery_expectation(),
        RecoveryExpectation::Automatic
    );

    let deadlock = FaultInjectionScenario::Deadlock {
        resource_count: 2,
        timing: InjectionTiming::AtStep(100),
    };
    assert_eq!(deadlock.recovery_expectation(), RecoveryExpectation::Terminal);
}

#[test]
fn scenario_requires_gpu() {
    let gpu_oom = FaultInjectionScenario::GpuOomSudden {
        allocation_size_bytes: 1024,
        timing: InjectionTiming::Immediate,
    };
    assert!(gpu_oom.requires_gpu());

    let cpu_starve = FaultInjectionScenario::CpuStarvation {
        load_factor: 0.9,
        duration_ms: 1000,
        timing: InjectionTiming::Immediate,
    };
    assert!(!cpu_starve.requires_gpu());
}

#[test]
fn scenario_categories() {
    let scenarios = vec![
        (
            FaultInjectionScenario::GpuOomSudden {
                allocation_size_bytes: 1024,
                timing: InjectionTiming::Immediate,
            },
            "memory",
        ),
        (
            FaultInjectionScenario::CflViolation {
                overshoot_factor: 1.5,
                timing: InjectionTiming::Immediate,
            },
            "numerical",
        ),
        (
            FaultInjectionScenario::GpuDeviceLost {
                timing: InjectionTiming::Immediate,
                recovery_time_ms: 100,
            },
            "device",
        ),
    ];

    for (scenario, expected_category) in scenarios {
        assert_eq!(scenario.category(), expected_category);
    }
}
