# Backlog / Strategy

## Architectural Enhancements
- Restructure into clean Domain/Application/Infrastructure/Presentation bounded contexts.
- Ensure dependency flows are strictly unidirectional (Domain -> App -> Infra/Presentation).
- Review all modules (core, physics, math, domains, simulation, clinical, analysis, solvers).
- BURN crate integration for optimized GPU support.
- Autodiff/PINN implementations for neural network-based physics solving.

## Validation Goals
- Implement automated test scenarios comparing `pykwavers` outputs natively against `k-wave-python` identical scenarios.
- Quantitatively verify sources, signals, grids, sensors, and solvers.

## Technical Debt Prevention
- Proactively locate and discard deprecated or duplicate methods, replacing them strictly with unified accessors.
- Remove outdated benchmarking, test data, and logs upon obsolescence.
