use super::GpuErrorType;
use crate::core::error::{KwaversError, SystemError};
use std::sync::Mutex;

// ── Fault Injector ───────────────────────────────────────────────────────────
//
// Theorem: Deterministic Fault Injection (Avizienis et al. 2004)
// For seed s and failure probability p, the sequence of injected faults is
// uniquely determined by s and p — enabling exact test reproducibility.
//
// Reference: Avizienis, A. et al. (2004). "Basic concepts and taxonomy of
// dependable and secure computing." IEEE Trans. Dependable Secure Comput.
// 1(1), 11–33. DOI: 10.1109/TDSC.2004.2

/// Deterministic fault injector for GPU stress testing.
///
/// Uses a seeded PRNG to inject GPU faults at a controlled probability.
/// Reproducibility is guaranteed: the same seed and probability always
/// produce the same sequence of failures (Law of Large Numbers ensures
/// the observed rate converges to `failure_probability` over ≥1000 trials).
///
/// # Example (stress test)
/// ```
/// use kwavers::gpu::recovery::{FaultInjector, GpuErrorType};
///
/// let injector = FaultInjector::new(0.001, GpuErrorType::OutOfMemory, 42);
/// for step in 0..10_000 {
///     if injector.should_inject() {
///         // handle simulated OOM
///     }
/// }
/// ```
#[derive(Debug)]
pub struct FaultInjector {
    /// Probability of injecting a fault on each call to `should_inject()`.
    /// Must be in `[0.0, 1.0]`.
    pub failure_probability: f64,
    /// The type of GPU error to produce when injecting a fault.
    pub failure_type: GpuErrorType,
    /// Thread-safe inner PRNG state (StdRng for deterministic reproducibility).
    ///
    /// `Mutex<rand::rngs::StdRng>` allows shared access from multiple threads
    /// without poisoning the generator (each lock acquisition is O(1) and very brief).
    rng: Mutex<rand::rngs::StdRng>,
}

impl FaultInjector {
    /// Create a new fault injector with a deterministic seed.
    ///
    /// # Parameters
    /// - `failure_probability` — fraction of calls to `should_inject()` that return `true`.
    ///   Clamped to `[0.0, 1.0]`.
    /// - `failure_type` — which GPU error variant to produce via `make_error()`.
    /// - `rng_seed` — initial PRNG state; same seed ⇒ same injection sequence.
    pub fn new(failure_probability: f64, failure_type: GpuErrorType, rng_seed: u64) -> Self {
        use rand::SeedableRng;
        Self {
            failure_probability: failure_probability.clamp(0.0, 1.0),
            failure_type,
            rng: Mutex::new(rand::rngs::StdRng::seed_from_u64(rng_seed)),
        }
    }

    /// Return `true` with probability `self.failure_probability`.
    ///
    /// Thread-safe: acquires the PRNG mutex for each call. Throughput is
    /// adequate for per-step fault injection at simulation step rates (< 10 MHz).
    /// # Panics
    /// - Panics if `FaultInjector RNG mutex poisoned`.
    ///
    pub fn should_inject(&self) -> bool {
        use rand::Rng;
        let mut rng = self.rng.lock().expect("FaultInjector RNG mutex poisoned");
        rng.gen::<f64>() < self.failure_probability
    }

    /// Construct the `KwaversError` that corresponds to this injector's `failure_type`.
    ///
    /// The returned error passes the `can_handle()` check on the matching recovery strategy.
    pub fn make_error(&self) -> KwaversError {
        match self.failure_type {
            GpuErrorType::OutOfMemory => KwaversError::System(SystemError::ResourceExhausted {
                resource: "GPU memory".to_string(),
                reason: "OOM (injected)".to_string(),
            }),
            GpuErrorType::DeviceLost => KwaversError::System(SystemError::ResourceUnavailable {
                resource: "GPU device_lost (injected)".to_string(),
            }),
            GpuErrorType::Timeout => KwaversError::System(SystemError::ResourceExhausted {
                resource: "GPU timeout".to_string(),
                reason: "Timeout (injected)".to_string(),
            }),
        }
    }
}
