// Fault scenario definitions for injection testing
//
// Each scenario represents a specific failure mode that can be injected
// at controlled points during simulation execution.
//
// # Mathematical Coverage Model
//
// For a system with n failure modes F = {f₁, f₂, ..., fₙ}, coverage C is:
// ```text
// C = |⋃(i=1 to n) Sᵢ| / n
// ```
// where Sᵢ is the set of successful test scenarios for failure mode fᵢ.
//
// Target: C ≥ 0.95 (95% coverage of all failure modes)

use serde::{Deserialize, Serialize};
use std::fmt;

#[cfg(test)]
mod tests;

/// Fault injection timing strategy
///
/// Controls when faults are injected during operation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InjectionTiming {
    /// Inject at a specific step number
    AtStep(usize),
    /// Inject after a duration
    AfterDuration(std::time::Duration),
    /// Inject based on memory usage threshold
    MemoryThreshold(f64), // fraction of available
    /// Inject randomly within a range
    RandomRange(usize, usize),
    /// Inject immediately on next operation
    Immediate,
    /// Gradual degradation pattern
    Gradual {
        steps: usize,
        intensity: f64,
    },
}

/// Expected recovery outcome for a scenario
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RecoveryExpectation {
    /// Should recover automatically
    Automatic,
    /// May require manual intervention
    Manual,
    /// Unrecoverable, should fail gracefully
    Terminal,
    /// Degraded operation acceptable
    Degraded,
}

/// Comprehensive fault scenarios covering all error modes
///
/// ## Coverage Matrix
///
/// | Category | Scenarios | GPU | CPU | Recovery Rate Target |
/// |----------|-----------|-----|-----|---------------------|
/// | Memory | OOM Gradual, OOM Sudden | ✓ | ✓ | ≥95% |
/// | Numerical | CFL Violation, Divergence | ✓ | ✓ | ≥98% |
/// | Convergence | Solver Failure, Ill-conditioned | ✓ | ✓ | ≥90% |
/// | Device | GPU Lost, Timeout | ✓ | - | ≥99% |
/// | System | Thread Panic, Resource Exhaustion | ✓ | ✓ | ≥95% |
/// | Cascade | Multi-fault sequences | ✓ | ✓ | ≥85% |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FaultInjectionScenario {
    // Memory exhaustion scenarios
    /// Gradual memory exhaustion (slow leak)
    GpuOomGradual {
        leak_rate_bytes: usize,
        total_leak_bytes: usize,
        timing: InjectionTiming,
    },
    /// Sudden memory exhaustion (immediate allocation failure)
    GpuOomSudden {
        allocation_size_bytes: usize,
        timing: InjectionTiming,
    },
    /// Memory fragmentation pattern
    MemoryFragmentation {
        block_count: usize,
        block_size: usize,
        timing: InjectionTiming,
    },

    // Numerical stability scenarios
    /// CFL condition violation (timestep too large)
    CflViolation {
        overshoot_factor: f64, // > 1.0
        timing: InjectionTiming,
    },
    /// Numerical divergence (unstable solution)
    NumericalDivergence {
        growth_rate: f64,
        timing: InjectionTiming,
    },
    /// Conservation law violation
    ConservationViolation {
        quantity: String,
        violation_amount: f64,
        timing: InjectionTiming,
    },

    // Convergence scenarios
    /// Iterative solver convergence failure
    ConvergenceFailure {
        solver_name: String,
        target_residual: f64,
        timing: InjectionTiming,
    },
    /// Ill-conditioned system
    IllConditioned {
        condition_number: f64, // large value
        timing: InjectionTiming,
    },
    /// Stiff problem requiring small timesteps
    StiffProblem {
        stiffness_ratio: f64,
        timing: InjectionTiming,
    },

    // Device/ hardware scenarios
    /// GPU device lost (driver/hardware failure)
    GpuDeviceLost {
        timing: InjectionTiming,
        recovery_time_ms: u64,
    },
    /// GPU timeout (kernel execution too long)
    GpuTimeout {
        timeout_ms: u64,
        timing: InjectionTiming,
    },
    /// PCIe bus error
    PcieError {
        error_rate: f64,
        timing: InjectionTiming,
    },

    // System resource scenarios
    /// Thread pool exhaustion
    ThreadExhaustion {
        thread_count: usize,
        timing: InjectionTiming,
    },
    /// File descriptor exhaustion
    FdExhaustion {
        fd_count: usize,
        timing: InjectionTiming,
    },
    /// CPU starvation (high load)
    CpuStarvation {
        load_factor: f64,
        duration_ms: u64,
        timing: InjectionTiming,
    },

    // Concurrency scenarios
    /// Race condition trigger
    RaceCondition {
        contention_points: Vec<usize>,
        timing: InjectionTiming,
    },
    /// Deadlock scenario
    Deadlock {
        resource_count: usize,
        timing: InjectionTiming,
    },
    /// Priority inversion
    PriorityInversion {
        low_priority_hold_ms: u64,
        timing: InjectionTiming,
    },

    // Cascading failure scenarios
    /// Sequenced fault chain
    CascadingSequence {
        sequence: Vec<Box<FaultInjectionScenario>>,
        delay_ms: u64,
    },
    /// Fault during recovery from previous fault
    RecoveryFault {
        primary: Box<FaultInjectionScenario>,
        secondary: Box<FaultInjectionScenario>,
        delay_ms: u64,
    },

    /// Custom fault scenario
    Custom {
        name: String,
        description: String,
        recovery_expectation: RecoveryExpectation,
    },
}

impl FaultInjectionScenario {
    /// Get the expected recovery outcome for this scenario
    pub fn recovery_expectation(&self) -> RecoveryExpectation {
        match self {
            // Memory scenarios - should recover with fallback
            Self::GpuOomGradual { .. } => RecoveryExpectation::Automatic,
            Self::GpuOomSudden { .. } => RecoveryExpectation::Automatic,
            Self::MemoryFragmentation { .. } => RecoveryExpectation::Degraded,

            // Numerical scenarios - should auto-correct
            Self::CflViolation { .. } => RecoveryExpectation::Automatic,
            Self::NumericalDivergence { .. } => RecoveryExpectation::Automatic,
            Self::ConservationViolation { .. } => RecoveryExpectation::Manual,

            // Convergence scenarios - may require solver switch
            Self::ConvergenceFailure { .. } => RecoveryExpectation::Automatic,
            Self::IllConditioned { .. } => RecoveryExpectation::Degraded,
            Self::StiffProblem { .. } => RecoveryExpectation::Degraded,

            // Device scenarios - depends on driver state
            Self::GpuDeviceLost { .. } => RecoveryExpectation::Automatic,
            Self::GpuTimeout { .. } => RecoveryExpectation::Automatic,
            Self::PcieError { .. } => RecoveryExpectation::Manual,

            // System scenarios - usually recoverable
            Self::ThreadExhaustion { .. } => RecoveryExpectation::Automatic,
            Self::FdExhaustion { .. } => RecoveryExpectation::Manual,
            Self::CpuStarvation { .. } => RecoveryExpectation::Automatic,

            // Concurrency scenarios - may be brittle
            Self::RaceCondition { .. } => RecoveryExpectation::Degraded,
            Self::Deadlock { .. } => RecoveryExpectation::Terminal,
            Self::PriorityInversion { .. } => RecoveryExpectation::Automatic,

            // Cascading scenarios - challenging
            Self::CascadingSequence { .. } => RecoveryExpectation::Degraded,
            Self::RecoveryFault { .. } => RecoveryExpectation::Terminal,

            // Custom scenarios - user-defined
            Self::Custom {
                recovery_expectation,
                ..
            } => *recovery_expectation,
        }
    }

    /// Get the injection timing for this scenario
    pub fn timing(&self) -> InjectionTiming {
        match self {
            Self::GpuOomGradual { timing, .. } => *timing,
            Self::GpuOomSudden { timing, .. } => *timing,
            Self::MemoryFragmentation { timing, .. } => *timing,
            Self::CflViolation { timing, .. } => *timing,
            Self::NumericalDivergence { timing, .. } => *timing,
            Self::ConservationViolation { timing, .. } => *timing,
            Self::ConvergenceFailure { timing, .. } => *timing,
            Self::IllConditioned { timing, .. } => *timing,
            Self::StiffProblem { timing, .. } => *timing,
            Self::GpuDeviceLost { timing, .. } => *timing,
            Self::GpuTimeout { timing, .. } => *timing,
            Self::PcieError { timing, .. } => *timing,
            Self::ThreadExhaustion { timing, .. } => *timing,
            Self::FdExhaustion { timing, .. } => *timing,
            Self::CpuStarvation { timing, .. } => *timing,
            Self::RaceCondition { timing, .. } => *timing,
            Self::Deadlock { timing, .. } => *timing,
            Self::PriorityInversion { timing, .. } => *timing,
            Self::CascadingSequence { sequence, .. } => sequence
                .first()
                .map(|s| s.timing())
                .unwrap_or(InjectionTiming::Immediate),
            Self::RecoveryFault { primary, .. } => primary.timing(),
            Self::Custom { .. } => InjectionTiming::Immediate,
        }
    }

    /// Check if this scenario requires GPU
    pub fn requires_gpu(&self) -> bool {
        matches!(
            self,
            Self::GpuOomGradual { .. }
                | Self::GpuOomSudden { .. }
                | Self::GpuDeviceLost { .. }
                | Self::GpuTimeout { .. }
        )
    }

    /// Get category name for telemetry
    pub fn category(&self) -> &'static str {
        match self {
            Self::GpuOomGradual { .. }
            | Self::GpuOomSudden { .. }
            | Self::MemoryFragmentation { .. } => "memory",
            Self::CflViolation { .. }
            | Self::NumericalDivergence { .. }
            | Self::ConservationViolation { .. } => "numerical",
            Self::ConvergenceFailure { .. }
            | Self::IllConditioned { .. }
            | Self::StiffProblem { .. } => "convergence",
            Self::GpuDeviceLost { .. } | Self::GpuTimeout { .. } | Self::PcieError { .. } => {
                "device"
            }
            Self::ThreadExhaustion { .. }
            | Self::FdExhaustion { .. }
            | Self::CpuStarvation { .. } => "system",
            Self::RaceCondition { .. } | Self::Deadlock { .. } | Self::PriorityInversion { .. } => {
                "concurrency"
            }
            Self::CascadingSequence { .. } | Self::RecoveryFault { .. } => "cascade",
            Self::Custom { .. } => "custom",
        }
    }
}

impl fmt::Display for FaultInjectionScenario {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::GpuOomGradual { .. } => "GPU OOM (Gradual)",
            Self::GpuOomSudden { .. } => "GPU OOM (Sudden)",
            Self::MemoryFragmentation { .. } => "Memory Fragmentation",
            Self::CflViolation { overshoot_factor, .. } => {
                return write!(f, "CFL Violation ({:.2}x)", overshoot_factor)
            }
            Self::NumericalDivergence { .. } => "Numerical Divergence",
            Self::ConservationViolation { quantity, .. } => {
                return write!(f, "Conservation Violation ({})", quantity)
            }
            Self::ConvergenceFailure { solver_name, .. } => {
                return write!(f, "Convergence Failure ({})", solver_name)
            }
            Self::IllConditioned { .. } => "Ill-conditioned System",
            Self::StiffProblem { .. } => "Stiff Problem",
            Self::GpuDeviceLost { .. } => "GPU Device Lost",
            Self::GpuTimeout { .. } => "GPU Timeout",
            Self::PcieError { .. } => "PCIe Error",
            Self::ThreadExhaustion { .. } => "Thread Exhaustion",
            Self::FdExhaustion { .. } => "FD Exhaustion",
            Self::CpuStarvation { .. } => "CPU Starvation",
            Self::RaceCondition { .. } => "Race Condition",
            Self::Deadlock { .. } => "Deadlock",
            Self::PriorityInversion { .. } => "Priority Inversion",
            Self::CascadingSequence { sequence, .. } => {
                return write!(f, "Cascading Sequence ({} faults)", sequence.len())
            }
            Self::RecoveryFault { .. } => "Fault During Recovery",
            Self::Custom { name, .. } => return write!(f, "Custom: {}", name),
        };
        write!(f, "{}", name)
    }
}
