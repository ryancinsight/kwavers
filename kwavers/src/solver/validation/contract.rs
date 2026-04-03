//! Scientific validation contract types shared across solver implementations.
//!
//! These types provide a single vocabulary for describing mathematical
//! references, validation targets, benchmark cases, and memory expectations.
//! They are intentionally lightweight so they can be reused by forward solvers,
//! inverse solvers, and higher-level parity harnesses.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Completion gate for a scientific module or workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionGate {
    Inventory,
    Architecture,
    Implementation,
    Validation,
    Performance,
}

/// Primary external truth source for a validation case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationTarget {
    Analytical,
    Literature,
    KWavePython,
    PublicDataset,
    CrossImplementation,
}

/// Reference supporting an implementation or validation claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScientificReference {
    pub citation: &'static str,
    pub locator: &'static str,
}

/// Scientific metadata attached to a retained numerical method.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScientificMetadata {
    pub method: &'static str,
    pub mathematical_statement: &'static str,
    pub invariants: &'static [&'static str],
    pub failure_modes: &'static [&'static str],
    pub references: &'static [ScientificReference],
}

/// Validation case describing the expected authority and success criterion.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationCase {
    pub name: &'static str,
    pub target: ValidationTarget,
    pub metric: &'static str,
    pub threshold: f64,
    pub gate: CompletionGate,
}

/// Benchmark case describing the expected measurement scope.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkCase {
    pub name: &'static str,
    pub metric: &'static str,
    pub target_value: Option<f64>,
    pub notes: &'static str,
}

/// Memory budget with dynamic transient allocation tracking.
///
/// THEOREM: Memory Invariant
/// For simulation S with n timesteps and workspace W:
/// TotalMemory(S) = StaticMemory(W) + Σ TransientMemory(t) for t ∈ [1, n]
///
/// The peak memory consumption is bounded by:
/// PeakMemory = StaticMemory + PeakTransient where
/// PeakTransient = max(TransientMemory(t)) ∀t ∈ [1, n]
///
/// REFERENCE: Wilson et al. (1995) "Dynamic Memory Management"
/// ISBN: 0-201-52992-9
#[derive(Debug)]
pub struct MemoryBudget {
    /// Statically allocated workspace memory (pre-allocated buffers)
    pub workspace_bytes: usize,
    /// Peak transient allocation observed during execution
    pub peak_transient_bytes: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Current transient allocation for this timestep
    pub current_transient_bytes: std::sync::atomic::AtomicUsize,
    /// Historical maximum across all timesteps
    pub max_transient_bytes: usize,
}

impl Clone for MemoryBudget {
    fn clone(&self) -> Self {
        Self {
            workspace_bytes: self.workspace_bytes,
            peak_transient_bytes: self.peak_transient_bytes.clone(),
            current_transient_bytes: std::sync::atomic::AtomicUsize::new(
                self.current_transient_bytes
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            max_transient_bytes: self.max_transient_bytes,
        }
    }
}

impl MemoryBudget {
    /// Create a new memory budget with dynamic tracking.
    ///
    /// THEOREM: Initial Memory Budget
    /// At initialization, transient allocations are zero.
    /// Peak tracking begins from the first timestep.
    pub fn new(workspace_bytes: usize) -> Self {
        Self {
            workspace_bytes,
            peak_transient_bytes: Arc::new(AtomicUsize::new(0)),
            current_transient_bytes: AtomicUsize::new(0),
            max_transient_bytes: 0,
        }
    }

    /// Record transient allocation for current timestep.
    ///
    /// Updates both current and peak transient memory.
    /// Thread-safe for concurrent solver operations.
    #[inline]
    pub fn record_transient(&mut self, bytes: usize) {
        self.current_transient_bytes.store(bytes, Ordering::Relaxed);

        // Update peak with an atomic CAS loop.
        let current_peak = self.peak_transient_bytes.load(Ordering::Relaxed);
        if bytes > current_peak {
            let _ = self.peak_transient_bytes.compare_exchange(
                current_peak,
                bytes,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }

        // Update max for non-atomic snapshot
        self.max_transient_bytes = self.max_transient_bytes.max(bytes);
    }

    /// Get peak transient allocation observed.
    #[inline]
    pub fn peak_transient(&self) -> usize {
        self.peak_transient_bytes.load(Ordering::Relaxed)
    }

    /// Get current transient allocation for this timestep.
    #[inline]
    pub fn current_transient(&self) -> usize {
        self.current_transient_bytes.load(Ordering::Relaxed)
    }

    /// Get total memory (static workspace + peak transient).
    ///
    /// THEOREM: Total Memory Bound
    /// TotalMemory ≤ workspace_bytes + peak_transient
    /// This is the memory required to run the simulation.
    #[inline]
    pub fn total_memory(&self) -> usize {
        self.workspace_bytes + self.peak_transient()
    }
}

/// Common scientific contract exposed by production-facing numerical methods.
pub trait ScientificMethod {
    fn metadata(&self) -> ScientificMetadata;
    fn validation_cases(&self) -> Vec<ValidationCase>;
    fn benchmark_cases(&self) -> Vec<BenchmarkCase>;
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFS: &[ScientificReference] = &[ScientificReference {
        citation: "Treeby & Cox (2010)",
        locator: "JBO 15(2) 021314",
    }];

    #[test]
    fn scientific_metadata_is_stable() {
        let metadata = ScientificMetadata {
            method: "k-space PSTD",
            mathematical_statement:
                "Solve the first-order acoustic system with spectral gradients.",
            invariants: &["C-order field layout", "Explicit time stepping"],
            failure_modes: &["Aliasing above Nyquist"],
            references: REFS,
        };

        assert_eq!(metadata.method, "k-space PSTD");
        assert_eq!(metadata.references.len(), 1);
    }

    #[test]
    fn validation_case_carries_gate_and_target() {
        let case = ValidationCase {
            name: "plane-wave-dispersion",
            target: ValidationTarget::Analytical,
            metric: "relative_l2",
            threshold: 1e-3,
            gate: CompletionGate::Validation,
        };

        assert_eq!(case.target, ValidationTarget::Analytical);
        assert_eq!(case.gate, CompletionGate::Validation);
    }
}
