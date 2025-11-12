// localization/multilateration/core.rs - Generalized multilateration (LS/WLS/ML)

use crate::sensor::localization::{Position, SensorArray, TrilaterationSolver};
use crate::error::KwaversResult;

/// Multilateration methods for range-based localization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MultilaterationMethod {
    LeastSquares,
    WeightedLeastSquares,
    MaximumLikelihood,
}

/// Solver for range-based multilateration across N≥3 sensors
#[derive(Debug)]
pub struct MultilaterationSolver {
    method: MultilaterationMethod,
}

impl MultilaterationSolver {
    /// Create a multilateration solver
    #[must_use]
    pub fn new(method: MultilaterationMethod) -> Self {
        Self { method }
    }

    /// Solve given per-sensor ranges (meters) using selected method.
    pub fn solve_ranges(&self, ranges: &[f64], array: &SensorArray) -> KwaversResult<Position> {
        // Input validation: finite, non-negative ranges
        if !ranges.iter().all(|&r| r.is_finite() && r >= 0.0) {
            return Err(crate::error::KwaversError::InvalidInput(
                "Ranges must be finite and non-negative".to_string(),
            ));
        }
        if ranges.len() < 3 || array.num_sensors() < 3 {
            return Err(crate::error::KwaversError::InvalidInput(
                "Need at least 3 ranges and sensors for 3D multilateration".to_string(),
            ));
        }

        match self.method {
            MultilaterationMethod::LeastSquares => self.gauss_newton(ranges, None, array),
            MultilaterationMethod::WeightedLeastSquares => {
                // If no explicit weights provided, default to equal weights.
                self.gauss_newton(ranges, None, array)
            }
            MultilaterationMethod::MaximumLikelihood => {
                // ML under i.i.d. Gaussian noise reduces to LS objective.
                self.gauss_newton(ranges, None, array)
            }
        }
    }

    /// Weighted solve: weights are per-measurement inverse-variance (higher = more trusted).
    pub fn solve_ranges_weighted(
        &self,
        ranges: &[f64],
        weights: &[f64],
        array: &SensorArray,
    ) -> KwaversResult<Position> {
        // Input validation: ranges and weights must be finite; weights non-negative (inverse variances)
        if !ranges.iter().all(|&r| r.is_finite() && r >= 0.0) {
            return Err(crate::error::KwaversError::InvalidInput(
                "Ranges must be finite and non-negative".to_string(),
            ));
        }
        if !weights.iter().all(|&w| w.is_finite() && w >= 0.0) {
            return Err(crate::error::KwaversError::InvalidInput(
                "Weights must be finite and non-negative".to_string(),
            ));
        }
        if ranges.len() != weights.len() {
            return Err(crate::error::KwaversError::InvalidInput(
                "Weights length must match ranges length".to_string(),
            ));
        }
        if ranges.len() < 3 || array.num_sensors() < 3 {
            return Err(crate::error::KwaversError::InvalidInput(
                "Need at least 3 ranges and sensors for 3D multilateration".to_string(),
            ));
        }

        self.gauss_newton(ranges, Some(weights), array)
    }

    /// Gauss-Newton iterative solver for range residuals.
    ///
    /// Numerical stability: solves normal equations `(J^T J) Δx = -J^T r` using
    /// Cholesky factorization with Levenberg–Marquardt diagonal damping when needed.
    /// This avoids adjugate/explicit inversion instability near singular or ill-conditioned
    /// `J^T J` and ensures positive-definite augmentation when rank-deficient.
    fn gauss_newton(
        &self,
        ranges: &[f64],
        weights: Option<&[f64]>,
        array: &SensorArray,
    ) -> KwaversResult<Position> {
        let n = ranges.len().min(array.num_sensors());
        if n < 3 {
            return Err(crate::error::KwaversError::InvalidInput(
                "Insufficient measurements".to_string(),
            ));
        }

        let mut x = array.centroid();

        const MAX_ITER: usize = 50;
        const TOL: f64 = 1e-6;
        let mut lambda: f64 = 0.0; // LM damping parameter; adaptively increased on failure

        for _iter in 0..MAX_ITER {
            // Build residuals and Jacobian at current estimate
            let mut residuals = Vec::with_capacity(n);
            let mut jac = Vec::with_capacity(n);

            for i in 0..n {
                let pi = array.get_sensor_position(i);
                let di = x.distance_to(pi);
                let ri = ranges[i];

                // Residual r_i = d(x, p_i) - r_i
                let mut r = di - ri;
                let mut jrow = if di > 1e-10 {
                    [
                        (x.x - pi.x) / di,
                        (x.y - pi.y) / di,
                        (x.z - pi.z) / di,
                    ]
                } else {
                    [0.0, 0.0, 0.0]
                };

                if let Some(w) = weights {
                    let wi = w[i].max(0.0);
                    let sqrt_wi = wi.sqrt();
                    r *= sqrt_wi;
                    jrow[0] *= sqrt_wi;
                    jrow[1] *= sqrt_wi;
                    jrow[2] *= sqrt_wi;
                }

                residuals.push(r);
                jac.push(jrow);
            }

            // Normal equations: (J^T J) Δx = -J^T r
            let jtj = compute_jtj(&jac);
            let jtr = compute_jtr(&jac, &residuals);
            let rhs = [-jtr[0], -jtr[1], -jtr[2]];

            // Current cost
            let cost0 = compute_cost(array, ranges, weights, x);

            // Try LM damping escalations to ensure descent
            let mut accepted = false;
            for _attempt in 0..3 {
                let mut jtj_damped = jtj;
                if lambda > 0.0 {
                    jtj_damped[0][0] += lambda;
                    jtj_damped[1][1] += lambda;
                    jtj_damped[2][2] += lambda;
                }

                let delta = match solve_spd_3x3_cholesky(jtj_damped, rhs) {
                    Ok(d) => d,
                    Err(_) => {
                        // Increase damping and retry
                        let trace = jtj[0][0] + jtj[1][1] + jtj[2][2];
                        lambda = if lambda == 0.0 {
                            (1e-8 * trace).max(1e-12)
                        } else {
                            (lambda * 10.0).min(1e6)
                        };
                        continue;
                    }
                };

                let x_try = Position::new(x.x + delta[0], x.y + delta[1], x.z + delta[2]);
                let cost1 = compute_cost(array, ranges, weights, x_try);
                if cost1.is_finite() && cost1 < cost0 {
                    x = x_try;
                    // Modest damping reduction after successful step
                    if lambda > 0.0 { lambda *= 0.3; }
                    let norm = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
                    if norm < TOL { return Ok(x); }
                    accepted = true;
                    break;
                } else {
                    // Escalate damping and retry
                    let trace = jtj[0][0] + jtj[1][1] + jtj[2][2];
                    lambda = if lambda == 0.0 {
                        (1e-8 * trace).max(1e-12)
                    } else {
                        (lambda * 10.0).min(1e6)
                    };
                }
            }

            if !accepted { break; }
        }

        Ok(x)
    }
}

fn compute_jtj(jacobian: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let mut jtj = [[0.0; 3]; 3];
    for row in jacobian {
        for i in 0..3 {
            for j in 0..3 {
                jtj[i][j] += row[i] * row[j];
            }
        }
    }
    jtj
}

fn compute_jtr(jacobian: &[[f64; 3]], residuals: &[f64]) -> [f64; 3] {
    let mut jtr = [0.0; 3];
    for (row, &r) in jacobian.iter().zip(residuals.iter()) {
        for i in 0..3 {
            jtr[i] += row[i] * r;
        }
    }
    jtr
}

fn compute_cost(array: &SensorArray, ranges: &[f64], weights: Option<&[f64]>, x: Position) -> f64 {
    let n = ranges.len().min(array.num_sensors());
    let mut cost = 0.0;
    for i in 0..n {
        let pi = array.get_sensor_position(i);
        let di = x.distance_to(pi);
        let ri = ranges[i];
        let mut r = di - ri;
        if let Some(w) = weights {
            let wi = w[i].max(0.0);
            let sqrt_wi = wi.sqrt();
            r *= sqrt_wi;
        }
        cost += r * r;
    }
    cost
}

/// Solve 3x3 symmetric positive-definite system using Cholesky (LL^T) factorization.
/// Returns Err if factorization fails due to non-positive definiteness.
fn solve_spd_3x3_cholesky(a: [[f64; 3]; 3], b: [f64; 3]) -> Result<[f64; 3], ()> {
    // Ensure symmetry numerically (guard against tiny asymmetries)
    let a = [
        [a[0][0], 0.5 * (a[0][1] + a[1][0]), 0.5 * (a[0][2] + a[2][0])],
        [0.5 * (a[1][0] + a[0][1]), a[1][1], 0.5 * (a[1][2] + a[2][1])],
        [0.5 * (a[2][0] + a[0][2]), 0.5 * (a[2][1] + a[1][2]), a[2][2]],
    ];

    // Cholesky decomposition: a = L L^T, with L lower-triangular
    let mut l = [[0.0f64; 3]; 3];

    // l00
    if a[0][0] <= 0.0 { return Err(()); }
    l[0][0] = a[0][0].sqrt();
    // l10, l20
    l[1][0] = a[1][0] / l[0][0];
    l[2][0] = a[2][0] / l[0][0];

    // l11
    let d11 = a[1][1] - l[1][0] * l[1][0];
    if d11 <= 0.0 { return Err(()); }
    l[1][1] = d11.sqrt();
    // l21
    l[2][1] = (a[2][1] - l[2][0] * l[1][0]) / l[1][1];

    // l22
    let d22 = a[2][2] - l[2][0] * l[2][0] - l[2][1] * l[2][1];
    if d22 <= 0.0 { return Err(()); }
    l[2][2] = d22.sqrt();

    // Forward solve: L y = b
    let y0 = b[0] / l[0][0];
    let y1 = (b[1] - l[1][0] * y0) / l[1][1];
    let y2 = (b[2] - l[2][0] * y0 - l[2][1] * y1) / l[2][2];

    // Backward solve: L^T x = y
    let x2 = y2 / l[2][2];
    let x1 = (y1 - l[2][1] * x2) / l[1][1];
    let x0 = (y0 - l[1][0] * x1 - l[2][0] * x2) / l[0][0];

    Ok([x0, x1, x2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensor::localization::array::{ArrayGeometry, Sensor};

    fn make_array() -> SensorArray {
        let sensors = vec![
            Sensor::new(0, Position::new(0.0, 0.0, 0.0)),
            Sensor::new(1, Position::new(1.0, 0.0, 0.0)),
            Sensor::new(2, Position::new(0.0, 1.0, 0.0)),
            Sensor::new(3, Position::new(0.0, 0.0, 1.0)),
        ];
        SensorArray::new(sensors, 1500.0, ArrayGeometry::Arbitrary)
    }

    #[test]
    fn test_multilateration_ls() {
        let array = make_array();
        let source = Position::new(0.3, 0.25, 0.2);
        let ranges: Vec<f64> = (0..array.num_sensors())
            .map(|i| array.get_sensor_position(i).distance_to(&source))
            .collect();

        let solver = MultilaterationSolver::new(MultilaterationMethod::LeastSquares);
        let pos = solver.solve_ranges(&ranges, &array).expect("LS solve failed");
        let err = pos.distance_to(&source);
        assert!(err < 0.05, "LS multilateration error too large: {err}");
    }

    #[test]
    fn test_multilateration_wls() {
        let array = make_array();
        let source = Position::new(0.6, -0.2, 0.15);
        let ranges: Vec<f64> = (0..array.num_sensors())
            .map(|i| array.get_sensor_position(i).distance_to(&source))
            .collect();
        let weights = vec![1.0, 4.0, 1.0, 0.5]; // sensor 1 more trusted, sensor 3 less

        let solver = MultilaterationSolver::new(MultilaterationMethod::WeightedLeastSquares);
        let pos = solver
            .solve_ranges_weighted(&ranges, &weights, &array)
            .expect("WLS solve failed");
        let err = pos.distance_to(&source);
        assert!(err < 0.05, "WLS multilateration error too large: {err}");
    }

    #[test]
    fn test_multilateration_ml() {
        let array = make_array();
        let source = Position::new(-0.1, 0.3, 0.4);
        let ranges: Vec<f64> = (0..array.num_sensors())
            .map(|i| array.get_sensor_position(i).distance_to(&source))
            .collect();

        let solver = MultilaterationSolver::new(MultilaterationMethod::MaximumLikelihood);
        let pos = solver.solve_ranges(&ranges, &array).expect("ML solve failed");
        let err = pos.distance_to(&source);
        assert!(err < 0.05, "ML multilateration error too large: {err}");
    }

    #[test]
    fn test_multilateration_ill_conditioned_cholesky_lm() {
        // Construct nearly collinear sensor geometry to induce ill-conditioned J^T J
        let sensors = vec![
            Sensor::new(0, Position::new(0.0, 0.0, 0.0)),
            Sensor::new(1, Position::new(1e-6, 0.0, 0.0)),
            Sensor::new(2, Position::new(2e-6, 0.0, 0.0)),
            Sensor::new(3, Position::new(0.0, 1e-6, 0.0)),
        ];
        let array = SensorArray::new(sensors, 1500.0, ArrayGeometry::Arbitrary);
        let source = Position::new(0.1, 0.0, 0.0);
        let ranges: Vec<f64> = (0..array.num_sensors())
            .map(|i| array.get_sensor_position(i).distance_to(&source))
            .collect();

        let solver = MultilaterationSolver::new(MultilaterationMethod::LeastSquares);
        let pos = solver.solve_ranges(&ranges, &array).expect("Solve failed");
        // Validate finite solution and bounded error
        assert!(pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite());
        let err = pos.distance_to(&source);
        assert!(err < 0.2, "Ill-conditioned solve error too large: {err}");
    }

    use proptest::prelude::*;
    proptest! {
        #[test]
        fn proptest_multilateration_random_arrays(
            sx in -0.5f64..0.5,
            sy in -0.5f64..0.5,
            sz in -0.5f64..0.5,
            ax0 in -1.0f64..1.0, ay0 in -1.0f64..1.0, az0 in -1.0f64..1.0,
            ax1 in -1.0f64..1.0, ay1 in -1.0f64..1.0, az1 in -1.0f64..1.0,
            ax2 in -1.0f64..1.0, ay2 in -1.0f64..1.0, az2 in -1.0f64..1.0,
            ax3 in -1.0f64..1.0, ay3 in -1.0f64..1.0, az3 in -1.0f64..1.0,
        ) {
            // Avoid degenerate arrays by adding small jitter and ensuring sensors are distinct
            let sensors = vec![
                // Ensure non-coplanarity by injecting z-offsets
                Sensor::new(0, Position::new(ax0, ay0, az0 + 0.015)),
                Sensor::new(1, Position::new(ax1 + 1e-3, ay1 - 1e-3, az1 + 0.025)),
                Sensor::new(2, Position::new(ax2 - 2e-3, ay2 + 2e-3, az2 + 0.035)),
                Sensor::new(3, Position::new(ax3 + 3e-3, ay3 + 1e-3, az3 - 0.045)),
            ];
            let array = SensorArray::new(sensors, 1500.0, ArrayGeometry::Arbitrary);
            let source = Position::new(sx, sy, sz);
            let ranges: Vec<f64> = (0..array.num_sensors())
                .map(|i| array.get_sensor_position(i).distance_to(&source))
                .collect();

            // Baseline cost at centroid
            let x0 = array.centroid();
            let cost0: f64 = (0..array.num_sensors()).map(|i|{
                let di = x0.distance_to(&array.get_sensor_position(i));
                let ri = ranges[i];
                let r = di - ri;
                r*r
            }).sum();

            let solver = MultilaterationSolver::new(MultilaterationMethod::LeastSquares);
            let pos = solver.solve_ranges(&ranges, &array).expect("Solve failed");
            let cost1: f64 = (0..array.num_sensors()).map(|i|{
                let di = pos.distance_to(&array.get_sensor_position(i));
                let ri = ranges[i];
                let r = di - ri;
                r*r
            }).sum();

            // Property: Gauss–Newton with LM damping should not increase cost
            prop_assert!(cost1.is_finite() && cost0.is_finite() && cost1 <= cost0);
        }
    }

    #[test]
    fn test_multilateration_adaptive_three() {
        let array = make_array();
        let source = Position::new(0.35, 0.2, 0.15);
        let ranges: Vec<f64> = (0..3)
            .map(|i| array.get_sensor_position(i).distance_to(&source))
            .collect();

        let solver = MultilaterationSolver::new(MultilaterationMethod::LeastSquares);
        let pos = solver
            .solve_adaptive(&ranges, None, &array)
            .expect("Adaptive trilateration failed");
        let err = pos.distance_to(&source);
        assert!(err < 1e-6, "Adaptive trilateration error too large: {err}");
    }

    #[test]
    fn test_multilateration_adaptive_four() {
        let array = make_array();
        let source = Position::new(-0.2, 0.4, 0.25);
        let ranges: Vec<f64> = (0..array.num_sensors())
            .map(|i| array.get_sensor_position(i).distance_to(&source))
            .collect();

        let solver = MultilaterationSolver::new(MultilaterationMethod::LeastSquares);
        let pos = solver
            .solve_adaptive(&ranges, None, &array)
            .expect("Adaptive LS failed");
        let err = pos.distance_to(&source);
        assert!(err < 0.05, "Adaptive LS error too large: {err}");
    }
}

impl MultilaterationSolver {
    /// Adaptive solve: if exactly three ranges are provided, use trilateration closed-form.
    /// If four or more ranges, use LS/WLS/ML according to `self.method` and optional weights.
    pub fn solve_adaptive(
        &self,
        ranges: &[f64],
        weights: Option<&[f64]>,
        array: &SensorArray,
    ) -> KwaversResult<Position> {
        if ranges.len() == 3 {
            // Use first three sensors by convention; ranges must correspond to indices 0..2
            let tri = TrilaterationSolver::new(array);
            let res = tri.solve_three([ranges[0], ranges[1], ranges[2]], [0, 1, 2])?;
            return Ok(Position::from_array(res.position));
        }

        match (self.method, weights) {
            (MultilaterationMethod::WeightedLeastSquares, Some(w)) => {
                self.solve_ranges_weighted(ranges, w, array)
            }
            _ => self.solve_ranges(ranges, array),
        }
    }
}
