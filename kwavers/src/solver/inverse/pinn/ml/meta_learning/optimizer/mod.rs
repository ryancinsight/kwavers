//! Meta-optimizer for outer-loop updates and learning rate schedules.

pub mod lr_schedule;
pub mod meta_optimizer;
#[cfg(test)]
mod tests;

pub use lr_schedule::LearningRateSchedule;
pub use meta_optimizer::{MetaOptimizer, MetaOptimizerMode};
