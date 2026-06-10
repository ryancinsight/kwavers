// Canonical photoacoustic simulation orchestration.

mod orchestrator;
mod runner;
// The `vertical` slice carries deliberate benchmark/validation *descriptor*
// scaffolding (e.g. `AcousticValidationCase`, `OpticalBenchmarkCase`) that
// documents the canonical per-stage cases. The consuming benchmark/validation
// *runner* is not yet wired, so clippy 1.95 flags the descriptor fields as
// never-read. The descriptors are intended design, not stale debt; allow
// dead_code on the slice until the runner lands rather than deleting the
// in-progress contract. Tracked for completion in the photoacoustics vertical.
#[allow(dead_code)]
mod vertical;

pub use orchestrator::PhotoacousticOrchestrator;
pub use runner::PhotoacousticRunner;
