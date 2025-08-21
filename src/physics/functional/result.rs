//! Monadic Operations for Result Types
//!
//! This module provides monadic operations for Result types in physics calculations,
//! enabling elegant error handling and composition of fallible operations.

/// Monadic operations for Result types in physics calculations
pub trait ResultOps<T, E> {
    /// Map over successful values
    fn map_ok<F, U>(self, f: F) -> Result<U, E>
    where
        F: FnOnce(T) -> U;

    /// Flat map for chaining operations
    fn and_then_ok<F, U>(self, f: F) -> Result<U, E>
    where
        F: FnOnce(T) -> Result<U, E>;

    /// Apply a fallback on error
    fn or_else_err<F>(self, f: F) -> Result<T, E>
    where
        F: FnOnce(E) -> Result<T, E>;

    /// Transform errors while preserving success
    fn map_err_preserve<F, E2>(self, f: F) -> Result<T, E2>
    where
        F: FnOnce(E) -> E2;

    /// Combine two Results with a binary operation
    fn zip_with<U, V, F>(self, other: Result<U, E>, f: F) -> Result<V, E>
    where
        F: FnOnce(T, U) -> V,
        E: Clone;

    /// Apply a function to the contained value if Ok, otherwise return the default
    fn map_or<U, F>(self, default: U, f: F) -> U
    where
        F: FnOnce(T) -> U;

    /// Apply a function to the contained value if Ok, otherwise compute a default
    fn map_or_else<U, D, F>(self, default: D, f: F) -> U
    where
        D: FnOnce(E) -> U,
        F: FnOnce(T) -> U;
}

impl<T, E> ResultOps<T, E> for Result<T, E> {
    fn map_ok<F, U>(self, f: F) -> Result<U, E>
    where
        F: FnOnce(T) -> U,
    {
        self.map(f)
    }

    fn and_then_ok<F, U>(self, f: F) -> Result<U, E>
    where
        F: FnOnce(T) -> Result<U, E>,
    {
        self.and_then(f)
    }

    fn or_else_err<F>(self, f: F) -> Result<T, E>
    where
        F: FnOnce(E) -> Result<T, E>,
    {
        self.or_else(f)
    }

    fn map_err_preserve<F, E2>(self, f: F) -> Result<T, E2>
    where
        F: FnOnce(E) -> E2,
    {
        self.map_err(f)
    }

    fn zip_with<U, V, F>(self, other: Result<U, E>, f: F) -> Result<V, E>
    where
        F: FnOnce(T, U) -> V,
        E: Clone,
    {
        match (self, other) {
            (Ok(a), Ok(b)) => Ok(f(a, b)),
            (Err(e), _) => Err(e),
            (_, Err(e)) => Err(e),
        }
    }

    fn map_or<U, F>(self, default: U, f: F) -> U
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Ok(value) => f(value),
            Err(_) => default,
        }
    }

    fn map_or_else<U, D, F>(self, default: D, f: F) -> U
    where
        D: FnOnce(E) -> U,
        F: FnOnce(T) -> U,
    {
        match self {
            Ok(value) => f(value),
            Err(error) => default(error),
        }
    }
}

/// Collect a vector of Results into a Result of a vector
pub fn collect_results<T, E>(results: Vec<Result<T, E>>) -> Result<Vec<T>, Vec<E>> {
    let mut successes = Vec::new();
    let mut errors = Vec::new();

    for result in results {
        match result {
            Ok(value) => successes.push(value),
            Err(error) => errors.push(error),
        }
    }

    if errors.is_empty() {
        Ok(successes)
    } else {
        Err(errors)
    }
}

/// Partition results into successes and failures
pub fn partition_results<T, E>(results: Vec<Result<T, E>>) -> (Vec<T>, Vec<E>) {
    let mut successes = Vec::new();
    let mut errors = Vec::new();

    for result in results {
        match result {
            Ok(value) => successes.push(value),
            Err(error) => errors.push(error),
        }
    }

    (successes, errors)
}

/// Try to apply a function to each element, collecting all results
pub fn try_map<T, U, E, F>(items: Vec<T>, f: F) -> Result<Vec<U>, E>
where
    F: Fn(T) -> Result<U, E>,
{
    items.into_iter().map(f).collect()
}

/// Apply a function to each element, ignoring errors
pub fn filter_map_ok<T, U, E, F>(results: Vec<Result<T, E>>, f: F) -> Vec<U>
where
    F: Fn(T) -> U,
{
    results
        .into_iter()
        .filter_map(|result| result.map(&f).ok())
        .collect()
}

/// Chain multiple fallible operations
pub fn chain_operations<T, E>(
    initial: Result<T, E>,
    operations: Vec<Box<dyn Fn(T) -> Result<T, E>>>,
) -> Result<T, E> {
    operations
        .into_iter()
        .fold(initial, |acc, op| acc.and_then(op))
}

/// Retry an operation a specified number of times
pub fn retry<T, E, F>(mut operation: F, max_attempts: usize) -> Result<T, E>
where
    F: FnMut() -> Result<T, E>,
{
    let mut last_error = None;

    for _ in 0..max_attempts {
        match operation() {
            Ok(value) => return Ok(value),
            Err(error) => last_error = Some(error),
        }
    }

    Err(last_error.expect("Should have at least one error after retries"))
}

/// Timeout wrapper for operations (conceptual - would need async for real implementation)
pub fn with_fallback<T, E, F, G>(primary: F, fallback: G) -> impl Fn() -> Result<T, E>
where
    F: Fn() -> Result<T, E>,
    G: Fn() -> Result<T, E>,
{
    move || primary().or_else(|_| fallback())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_ops() {
        let result: Result<i32, &str> = Ok(42);
        let doubled = result.map_ok(|x| x * 2);
        assert_eq!(doubled, Ok(84));

        let error_result: Result<i32, &str> = Err("error");
        let fallback = error_result.or_else_err(|_| Ok(100));
        assert_eq!(fallback, Ok(100));
    }

    #[test]
    fn test_zip_with() {
        let a: Result<i32, &str> = Ok(10);
        let b: Result<i32, &str> = Ok(20);
        let combined = a.zip_with(b, |x, y| x + y);
        assert_eq!(combined, Ok(30));

        let error_case: Result<i32, &str> = Err("error");
        let combined_error = error_case.zip_with(Ok(20), |x, y| x + y);
        assert!(combined_error.is_err());
    }

    #[test]
    fn test_collect_results() {
        let results: Vec<Result<i32, String>> = vec![Ok(1), Ok(2), Ok(3)];
        let collected = collect_results(results);
        assert_eq!(collected, Ok(vec![1, 2, 3]));

        let mixed_results = vec![Ok(1), Err("error"), Ok(3)];
        let collected_mixed = collect_results(mixed_results);
        assert!(collected_mixed.is_err());
    }

    #[test]
    fn test_partition_results() {
        let results = vec![Ok(1), Err("error1"), Ok(3), Err("error2")];
        let (successes, errors) = partition_results(results);
        assert_eq!(successes, vec![1, 3]);
        assert_eq!(errors, vec!["error1", "error2"]);
    }

    #[test]
    fn test_retry() {
        let mut attempt_count = 0;
        let result = retry(
            || {
                attempt_count += 1;
                if attempt_count < 3 {
                    Err("failed")
                } else {
                    Ok(42)
                }
            },
            5,
        );
        assert_eq!(result, Ok(42));
        assert_eq!(attempt_count, 3);
    }
}
