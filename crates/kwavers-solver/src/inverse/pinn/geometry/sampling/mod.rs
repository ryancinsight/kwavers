//! PINN collocation strategy and statically dispatched sampler.

mod sampler;
mod strategy;
#[cfg(test)]
mod tests;

pub use sampler::CollocationSampler;
pub use strategy::CollocationSamplingStrategy;
