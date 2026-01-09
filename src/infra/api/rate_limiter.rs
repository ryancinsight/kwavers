//! Production-ready rate limiter using token bucket algorithm
//!
//! Implements per-user, per-endpoint rate limiting with proper concurrency controls.
//! Uses token bucket algorithm for smooth rate limiting with burst capacity.
//!
//! References:
//! - Token Bucket Algorithm: https://en.wikipedia.org/wiki/Token_bucket
//! - Rate Limiting Patterns: https://stripe.com/blog/rate-limiters

use crate::infra::api::{APIError, APIErrorType, RateLimitInfo};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for rate limiting
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per window
    pub requests_per_window: u32,
    /// Window duration in seconds
    pub window_seconds: u64,
    /// Burst capacity (additional tokens allowed)
    pub burst_capacity: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_window: 100,
            window_seconds: 60, // 1 minute
            burst_capacity: 20,
        }
    }
}

/// Per-user rate limit state
#[derive(Debug, Clone)]
struct UserRateLimit {
    /// Tokens available
    tokens: f64,
    /// Last refill timestamp
    last_refill: Instant,
    /// Reset time for current window
    reset_time: chrono::DateTime<chrono::Utc>,
}

/// Production-ready rate limiter using token bucket algorithm
#[derive(Clone, Debug)]
pub struct RateLimiter {
    /// Rate limit configurations per endpoint
    configs: HashMap<String, RateLimitConfig>,
    /// Per-user rate limit states
    states: Arc<RwLock<HashMap<String, UserRateLimit>>>,
    /// Default configuration for unspecified endpoints
    default_config: RateLimitConfig,
}

impl RateLimiter {
    /// Create a new rate limiter with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
            states: Arc::new(RwLock::new(HashMap::new())),
            default_config: RateLimitConfig::default(),
        }
    }

    /// Create rate limiter with custom configuration
    #[must_use]
    pub fn with_config(default_config: RateLimitConfig) -> Self {
        Self {
            configs: HashMap::new(),
            states: Arc::new(RwLock::new(HashMap::new())),
            default_config,
        }
    }

    /// Add endpoint-specific rate limit configuration
    pub fn add_endpoint_config(&mut self, endpoint: &str, config: RateLimitConfig) {
        self.configs.insert(endpoint.to_string(), config);
    }

    /// Check if request is within rate limits
    pub async fn check_limit(&self, user_id: &str, endpoint: &str) -> Result<(), APIError> {
        let config = self.configs.get(endpoint).unwrap_or(&self.default_config);
        let user_key = format!("{}:{}", user_id, endpoint);

        let mut states = self.states.write();
        let now = Instant::now();

        let user_limit = states
            .entry(user_key.clone())
            .or_insert_with(|| UserRateLimit {
                tokens: config.requests_per_window as f64,
                last_refill: now,
                reset_time: chrono::Utc::now()
                    + chrono::Duration::seconds(config.window_seconds as i64),
            });

        // Refill tokens based on elapsed time
        let elapsed = now.duration_since(user_limit.last_refill);
        let refill_rate = config.requests_per_window as f64 / config.window_seconds as f64;
        let tokens_to_add = elapsed.as_secs_f64() * refill_rate;

        user_limit.tokens = (user_limit.tokens + tokens_to_add)
            .min((config.requests_per_window + config.burst_capacity) as f64);
        user_limit.last_refill = now;

        // Check if request can be allowed
        if user_limit.tokens >= 1.0 {
            user_limit.tokens -= 1.0;
            Ok(())
        } else {
            // Calculate reset time
            let tokens_needed = 1.0 - user_limit.tokens;
            let wait_seconds = tokens_needed / refill_rate;
            user_limit.reset_time =
                chrono::Utc::now() + chrono::Duration::seconds(wait_seconds as i64);

            Err(APIError {
                error: APIErrorType::RateLimitExceeded,
                message: format!(
                    "Rate limit exceeded for endpoint '{}'. Try again in {:.1}s",
                    endpoint, wait_seconds
                ),
                details: Some(HashMap::from([
                    ("endpoint".to_string(), serde_json::json!(endpoint)),
                    ("user_id".to_string(), serde_json::json!(user_id)),
                    (
                        "reset_in_seconds".to_string(),
                        serde_json::json!(wait_seconds),
                    ),
                ])),
            })
        }
    }

    /// Get current rate limit information for user and endpoint
    #[must_use]
    pub fn get_limit_info(&self, user_id: &str, endpoint: &str) -> RateLimitInfo {
        let config = self.configs.get(endpoint).unwrap_or(&self.default_config);
        let user_key = format!("{}:{}", user_id, endpoint);

        let states = self.states.read();
        let now = Instant::now();

        if let Some(user_limit) = states.get(&user_key) {
            // Calculate remaining tokens
            let elapsed = now.duration_since(user_limit.last_refill);
            let refill_rate = config.requests_per_window as f64 / config.window_seconds as f64;
            let tokens_to_add = elapsed.as_secs_f64() * refill_rate;
            let current_tokens = (user_limit.tokens + tokens_to_add)
                .min((config.requests_per_window + config.burst_capacity) as f64);

            let remaining = current_tokens.floor() as u32;
            let reset_time = if remaining == 0 {
                let tokens_needed = 1.0 - current_tokens;
                let wait_seconds = tokens_needed / refill_rate;
                chrono::Utc::now() + chrono::Duration::seconds(wait_seconds as i64)
            } else {
                user_limit.reset_time
            };

            RateLimitInfo {
                limit: config.requests_per_window as usize,
                remaining: remaining as usize,
                reset_time,
            }
        } else {
            // No previous requests, return full limit
            RateLimitInfo {
                limit: config.requests_per_window as usize,
                remaining: config.requests_per_window as usize,
                reset_time: chrono::Utc::now()
                    + chrono::Duration::seconds(config.window_seconds as i64),
            }
        }
    }

    /// Clean up expired rate limit states (for memory management)
    pub fn cleanup_expired(&self) {
        let mut states = self.states.write();
        let now = chrono::Utc::now();
        let expired_keys: Vec<String> = states
            .iter()
            .filter(|(_, limit)| now > limit.reset_time)
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_keys {
            states.remove(&key);
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_rate_limiter_basic() {
        let limiter = RateLimiter::new();

        // Should allow initial requests
        assert!(limiter.check_limit("user1", "/api/test").await.is_ok());
        assert!(limiter.check_limit("user1", "/api/test").await.is_ok());
    }

    #[tokio::test]
    async fn test_rate_limiter_exhaustion() {
        let config = RateLimitConfig {
            requests_per_window: 2,
            window_seconds: 1,
            burst_capacity: 0,
        };
        let limiter = RateLimiter::with_config(config);

        // Use up the limit
        assert!(limiter.check_limit("user1", "/api/test").await.is_ok());
        assert!(limiter.check_limit("user1", "/api/test").await.is_ok());

        // Should be rate limited
        assert!(limiter.check_limit("user1", "/api/test").await.is_err());
    }

    #[tokio::test]
    async fn test_rate_limiter_recovery() {
        let config = RateLimitConfig {
            requests_per_window: 1,
            window_seconds: 1,
            burst_capacity: 0,
        };
        let limiter = RateLimiter::with_config(config);

        // Use up the limit
        assert!(limiter.check_limit("user1", "/api/test").await.is_ok());
        assert!(limiter.check_limit("user1", "/api/test").await.is_err());

        // Wait for recovery
        sleep(Duration::from_secs(2)).await;

        // Should allow again
        assert!(limiter.check_limit("user1", "/api/test").await.is_ok());
    }

    #[test]
    fn test_rate_limit_info() {
        let limiter = RateLimiter::new();
        let info = limiter.get_limit_info("user1", "/api/test");

        assert_eq!(info.limit, 100);
        assert_eq!(info.remaining, 100);
    }

    #[test]
    fn test_endpoint_specific_config() {
        let mut limiter = RateLimiter::new();
        let strict_config = RateLimitConfig {
            requests_per_window: 5,
            window_seconds: 60,
            burst_capacity: 0,
        };

        limiter.add_endpoint_config("/api/admin", strict_config);

        let admin_info = limiter.get_limit_info("user1", "/api/admin");
        let regular_info = limiter.get_limit_info("user1", "/api/test");

        assert_eq!(admin_info.limit, 5);
        assert_eq!(regular_info.limit, 100);
    }
}
