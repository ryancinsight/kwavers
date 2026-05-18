use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::debug;

use crate::core::error::KwaversError;

use super::ids::current_epoch_seconds;
use super::severity::{all_error_type_names, error_type_name, ErrorSeverity};

const BUCKET_RESETTING: u64 = u64::MAX;

#[derive(Debug)]
struct SlidingBucket {
    epoch_second: AtomicU64,
    count: AtomicU64,
}

impl Default for SlidingBucket {
    fn default() -> Self {
        Self {
            epoch_second: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }
}

/// Prometheus-style metric types for compatibility with existing exports.
#[derive(Debug)]
pub enum MetricType {
    Counter {
        name: String,
        help: String,
        value: AtomicU64,
        labels: HashMap<String, String>,
    },
    Gauge {
        name: String,
        help: String,
        value: AtomicU64,
        labels: HashMap<String, String>,
    },
    Histogram {
        name: String,
        help: String,
        buckets: Vec<f64>,
        counts: Vec<AtomicU64>,
        sum: AtomicU64,
        labels: HashMap<String, String>,
    },
}

#[derive(Debug)]
pub(crate) struct SlidingRateWindow<const N: usize> {
    buckets: [SlidingBucket; N],
}

impl<const N: usize> Default for SlidingRateWindow<N> {
    fn default() -> Self {
        Self {
            buckets: std::array::from_fn(|_| SlidingBucket::default()),
        }
    }
}

impl<const N: usize> SlidingRateWindow<N> {
    pub(crate) fn record_at(&self, epoch_second: u64) {
        let bucket = &self.buckets[(epoch_second as usize) % N];

        loop {
            let seen_epoch = bucket.epoch_second.load(Ordering::Acquire);
            if seen_epoch == epoch_second {
                bucket.count.fetch_add(1, Ordering::Relaxed);
                break;
            }

            if seen_epoch == BUCKET_RESETTING {
                std::hint::spin_loop();
                continue;
            }

            if bucket
                .epoch_second
                .compare_exchange(
                    seen_epoch,
                    BUCKET_RESETTING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                bucket.count.store(0, Ordering::Relaxed);
                bucket.epoch_second.store(epoch_second, Ordering::Release);
                bucket.count.fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }

    #[must_use]
    pub(crate) fn rate_per_minute(&self, lookback_seconds: u64) -> f64 {
        self.rate_per_minute_at(current_epoch_seconds(), lookback_seconds)
    }

    #[must_use]
    pub(crate) fn rate_per_minute_at(&self, now_second: u64, lookback_seconds: u64) -> f64 {
        let window = lookback_seconds.clamp(1, N as u64);
        let start = now_second.saturating_sub(window.saturating_sub(1));
        let total: u64 = self
            .buckets
            .iter()
            .map(|bucket| {
                let epoch = bucket.epoch_second.load(Ordering::Acquire);
                if (start..=now_second).contains(&epoch) {
                    bucket.count.load(Ordering::Relaxed)
                } else {
                    0
                }
            })
            .sum();

        total as f64 * 60.0 / window as f64
    }
}

/// Error metrics registry.
#[derive(Debug)]
pub struct TelemetryErrorCounts {
    errors_total: HashMap<String, AtomicU64>,
    errors_by_type: HashMap<&'static str, AtomicU64>,
    error_rates: HashMap<ErrorSeverity, SlidingRateWindow<60>>,
    recovery_attempts_total: AtomicU64,
    recovery_successes_total: AtomicU64,
    recovery_success_rate: AtomicU64,
    last_error_timestamp: AtomicU64,
    threshold_breaches: HashMap<ErrorSeverity, AtomicU64>,
}

impl Default for TelemetryErrorCounts {
    fn default() -> Self {
        let errors_total = ErrorSeverity::all()
            .into_iter()
            .map(|severity| (severity.as_str().to_string(), AtomicU64::new(0)))
            .collect();

        let errors_by_type = all_error_type_names()
            .iter()
            .copied()
            .map(|error_type| (error_type, AtomicU64::new(0)))
            .collect();

        let error_rates = ErrorSeverity::all()
            .into_iter()
            .map(|severity| (severity, SlidingRateWindow::<60>::default()))
            .collect();

        let threshold_breaches = ErrorSeverity::all()
            .into_iter()
            .map(|severity| (severity, AtomicU64::new(0)))
            .collect();

        Self {
            errors_total,
            errors_by_type,
            error_rates,
            recovery_attempts_total: AtomicU64::new(0),
            recovery_successes_total: AtomicU64::new(0),
            recovery_success_rate: AtomicU64::new(1.0_f64.to_bits()),
            last_error_timestamp: AtomicU64::new(0),
            threshold_breaches,
        }
    }
}

impl TelemetryErrorCounts {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_error(&self, error: &KwaversError) {
        let severity = ErrorSeverity::from(error);
        self.record_error_with_time(error, current_epoch_seconds());
        debug!(
            severity = severity.as_str(),
            error_type = error_type_name(error),
            "Error recorded in metrics"
        );
    }

    pub(crate) fn record_error_with_time(&self, error: &KwaversError, epoch_second: u64) {
        let severity = ErrorSeverity::from(error);

        if let Some(counter) = self.errors_total.get(severity.as_str()) {
            counter.fetch_add(1, Ordering::Relaxed);
        }

        let error_type = error_type_name(error);
        if let Some(counter) = self.errors_by_type.get(error_type) {
            counter.fetch_add(1, Ordering::Relaxed);
        }

        if let Some(window) = self.error_rates.get(&severity) {
            window.record_at(epoch_second);
        }

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.last_error_timestamp.store(now_ms, Ordering::Relaxed);
    }

    pub fn record_recovery_attempt(&self) {
        self.recovery_attempts_total.fetch_add(1, Ordering::Relaxed);
        self.update_success_rate();
    }

    pub fn record_recovery_success(&self) {
        self.recovery_successes_total
            .fetch_add(1, Ordering::Relaxed);
        self.update_success_rate();
    }

    fn update_success_rate(&self) {
        let total = self.recovery_attempts_total.load(Ordering::Relaxed);
        let success = self.recovery_successes_total.load(Ordering::Relaxed);
        if total > 0 {
            let rate = success as f64 / total as f64;
            self.recovery_success_rate
                .store(rate.to_bits(), Ordering::Relaxed);
        }
    }

    pub fn record_threshold_breach(&self, severity: ErrorSeverity) {
        if let Some(counter) = self.threshold_breaches.get(&severity) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[must_use]
    pub fn error_count(&self, severity: &str) -> u64 {
        self.errors_total
            .get(severity)
            .map(|counter| counter.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    #[must_use]
    pub fn total_errors(&self) -> u64 {
        self.errors_total
            .values()
            .map(|counter| counter.load(Ordering::Relaxed))
            .sum()
    }

    #[must_use]
    pub fn error_count_by_type(&self, error_type: &'static str) -> u64 {
        self.errors_by_type
            .get(error_type)
            .map(|counter| counter.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    #[must_use]
    pub fn recovery_rate(&self) -> f64 {
        f64::from_bits(self.recovery_success_rate.load(Ordering::Relaxed))
    }

    #[must_use]
    pub fn error_rate_per_minute(&self, severity: ErrorSeverity, window_seconds: u64) -> f64 {
        self.error_rates
            .get(&severity)
            .map(|window| window.rate_per_minute(window_seconds))
            .unwrap_or(0.0)
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn error_rate_per_minute_at(
        &self,
        severity: ErrorSeverity,
        now_second: u64,
        window_seconds: u64,
    ) -> f64 {
        self.error_rates
            .get(&severity)
            .map(|window| window.rate_per_minute_at(now_second, window_seconds))
            .unwrap_or(0.0)
    }

    #[must_use]
    pub fn export_prometheus(&self) -> String {
        let mut output = String::with_capacity(1400);
        output.push_str("# HELP kwavers_errors_total Total errors by severity\n");
        output.push_str("# TYPE kwavers_errors_total counter\n");
        for severity in ErrorSeverity::all() {
            let value = self.error_count(severity.as_str());
            output.push_str(&format!(
                "kwavers_errors_total{{severity=\"{}\"}} {}\n",
                severity.as_str(),
                value
            ));
        }

        output.push_str("# HELP kwavers_errors_by_type_total Total errors by error type\n");
        output.push_str("# TYPE kwavers_errors_by_type_total counter\n");
        for error_type in all_error_type_names() {
            let value = self.error_count_by_type(error_type);
            output.push_str(&format!(
                "kwavers_errors_by_type_total{{error_type=\"{}\"}} {}\n",
                error_type, value
            ));
        }

        output.push_str("# HELP kwavers_recovery_attempts_total Total recovery attempts\n");
        output.push_str("# TYPE kwavers_recovery_attempts_total counter\n");
        output.push_str(&format!(
            "kwavers_recovery_attempts_total {}\n",
            self.recovery_attempts_total.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP kwavers_recovery_success_rate Current recovery success rate\n");
        output.push_str("# TYPE kwavers_recovery_success_rate gauge\n");
        output.push_str(&format!(
            "kwavers_recovery_success_rate {:.4}\n",
            self.recovery_rate()
        ));

        output.push_str("# HELP kwavers_threshold_breaches_total Alert threshold breaches\n");
        output.push_str("# TYPE kwavers_threshold_breaches_total counter\n");
        for severity in ErrorSeverity::all() {
            let value = self
                .threshold_breaches
                .get(&severity)
                .map(|counter| counter.load(Ordering::Relaxed))
                .unwrap_or(0);
            output.push_str(&format!(
                "kwavers_threshold_breaches_total{{severity=\"{}\"}} {}\n",
                severity.as_str(),
                value
            ));
        }

        output.push_str("# HELP kwavers_last_error_timestamp_ms Last error timestamp in milliseconds since epoch\n");
        output.push_str("# TYPE kwavers_last_error_timestamp_ms gauge\n");
        output.push_str(&format!(
            "kwavers_last_error_timestamp_ms {}\n",
            self.last_error_timestamp.load(Ordering::Relaxed)
        ));

        output
    }
}
