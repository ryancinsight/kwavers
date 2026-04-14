use parking_lot::{Mutex, RwLock};
use std::collections::VecDeque;
use std::time::Instant;
use tracing::{info, trace, warn};

/// Budget manager for latency-constrained execution.
pub struct LatencyBudget {
    /// Total budget per frame.
    total_budget_ms: f64,
    /// Allocations per category.
    allocations: RwLock<Vec<(&'static str, f64)>>,
    /// Measured latencies per category.
    measurements: Mutex<VecDeque<(Instant, &'static str, f64)>>,
    /// Whether budget is enforced (strict mode).
    strict: bool,
}

impl LatencyBudget {
    /// Create a new latency budget.
    pub fn new(total_budget_ms: f64) -> Self {
        Self {
            total_budget_ms,
            allocations: RwLock::new(Vec::new()),
            measurements: Mutex::new(VecDeque::with_capacity(100)),
            strict: true,
        }
    }

    /// Allocate budget to a category.
    pub fn allocate(&self, category: &'static str, budget_ms: f64) {
        let mut allocations = self.allocations.write();
        // Check total doesn't exceed budget
        let current_total: f64 = allocations.iter().map(|(_, b)| b).sum();
        if current_total + budget_ms > self.total_budget_ms {
            warn!(
                category,
                requested = budget_ms,
                current_total,
                total = self.total_budget_ms,
                "Budget allocation exceeds total"
            );
        }
        allocations.push((category, budget_ms));
        info!(
            category,
            budget_ms,
            total = self.total_budget_ms,
            "Allocated latency budget"
        );
    }

    /// Check if a measurement is within budget for its category.
    pub fn check(&self, category: &'static str, actual_ms: f64) -> bool {
        let allocations = self.allocations.read();
        let budget = allocations
            .iter()
            .find(|(c, _)| *c == category)
            .map(|(_, b)| *b)
            .unwrap_or(self.total_budget_ms);

        let within = actual_ms <= budget;
        if !within {
            warn!(
                category,
                actual_ms,
                budget_ms = budget,
                overrun_ms = actual_ms - budget,
                "Latency budget exceeded"
            );
        } else {
            trace!(
                category,
                actual_ms,
                budget_ms = budget,
                remaining_ms = budget - actual_ms,
                "Within latency budget"
            );
        }

        // Record measurement
        let mut measurements = self.measurements.lock();
        if measurements.len() >= 100 {
            measurements.pop_front();
        }
        measurements.push_back((Instant::now(), category, actual_ms));

        within
    }

    /// Get remaining budget for a category.
    pub fn remaining(&self, category: &'static str, consumed_ms: f64) -> f64 {
        let allocations = self.allocations.read();
        let budget = allocations
            .iter()
            .find(|(c, _)| *c == category)
            .map(|(_, b)| *b)
            .unwrap_or(self.total_budget_ms);
        (budget - consumed_ms).max(0.0)
    }

    /// Get total utilization percentage.
    pub fn utilization_percent(&self) -> f64 {
        let measurements = self.measurements.lock();
        let recent: Vec<_> = measurements
            .iter()
            .rev()
            .take(10)
            .map(|(_, _, lat)| *lat)
            .collect();

        if recent.is_empty() {
            return 0.0;
        }

        let avg_latency: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        (avg_latency / self.total_budget_ms * 100.0).min(100.0)
    }

    /// Get budget summary.
    pub fn summary(&self) -> String {
        let allocations = self.allocations.read();
        let measurements = self.measurements.lock();

        let mut summary = format!(
            "Latency Budget: {:.1}ms total, {:.1}% utilized\n",
            self.total_budget_ms,
            self.utilization_percent()
        );

        for (category, budget) in allocations.iter() {
            let avg: f64 = measurements
                .iter()
                .filter(|(_, c, _)| *c == *category)
                .map(|(_, _, lat)| *lat)
                .sum::<f64>()
                / measurements
                    .iter()
                    .filter(|(_, c, _)| *c == *category)
                    .count()
                    .max(1) as f64;

            summary.push_str(&format!(
                " {}: target={:.1}ms, avg={:.1}ms\n",
                category, budget, avg
            ));
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_budget_allocation() {
        let budget = LatencyBudget::new(16.7);
        budget.allocate("extract", 5.0);
        budget.allocate("encode", 5.0);

        let allocations = budget.allocations.read();
        assert_eq!(allocations.len(), 2);
        assert!(allocations.iter().any(|(n, _)| *n == "extract"));
        assert!(allocations.iter().any(|(n, _)| *n == "encode"));
    }

    #[test]
    fn test_latency_budget_check() {
        let budget = LatencyBudget::new(16.7);
        budget.allocate("test", 10.0);

        assert!(budget.check("test", 5.0)); // Within budget
        assert!(!budget.check("test", 15.0)); // Exceeds budget
    }

    #[test]
    fn test_latency_budget_remaining() {
        let budget = LatencyBudget::new(16.7);
        budget.allocate("test", 10.0);

        assert_eq!(budget.remaining("test", 5.0), 5.0);
        assert_eq!(budget.remaining("test", 12.0), 0.0); // Over budget
    }
}
