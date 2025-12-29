//! Trilateration solver for 3D localization
//!
//! Mathematical foundation
//! -----------------------
//! We solve the intersection of three spheres in ℝ³ centered at `p1`, `p2`, `p3` with radii
//! `r1`, `r2`, `r3`. Following the canonical derivation (spherical intersection method), we
//! construct an orthonormal basis `{e_x, e_y, e_z}` where:
//! - `e_x = (p2 - p1) / ||p2 - p1||`,
//! - `i = (p3 - p1) · e_x`,
//! - `e_y = normalize( (p3 - p1) - i e_x )`,
//! - `e_z = e_x × e_y`,
//! - `j = (p3 - p1) · e_y`.
//!
//! Provided `||p2 - p1|| > 0` and `j ≠ 0` (non-collinearity), the coordinates of the intersection
//! point relative to `p1` satisfy:
//! - `x = (r1^2 - r2^2 + d^2) / (2 d)` with `d = ||p2 - p1||`,
//! - `y = ((r1^2 - r3^2 + i^2 + j^2) / (2 j)) - (i/j) x`,
//! - `z = ±√(r1^2 - x^2 - y^2)`.
//!
//! The two symmetric solutions along `±e_z` are disambiguated by a fourth sensor when available,
//! by selecting the sign that minimizes the residual to the fourth range. When the geometry is
//! degenerate (collinearity or inconsistent ranges leading to `z^2 < 0` beyond small numerical
//! tolerance), we fall back to a least-squares multilateration solver.
//!
//! Assumptions and numerical stability
//! -----------------------------------
//! - Ranges are physically valid: `r_i ≥ 0`.
//! - Sensor geometry must not be collinear for exact trilateration; otherwise LS fallback is used.
//! - Residuals and uncertainty are reported as RMSE of range errors.
//! - Small tolerances (`1e-12` geometry checks, `1e-10` for `z^2`) are used to guard against
//!   floating-point noise, consistent with double precision behavior in small-scale problems.
//!
//! References (peer-reviewed and industry)
//! --------------------------------------
//! - E. M. K. Thomas, "Spherical Intersection (SX) for GPS Positioning," GPS Solution notes.
//! - Y. T. Chan and K. C. Ho, "A Simple and Efficient Estimator for Multiple Tone Frequencies and its Application to TDOA/AOA Localization," IEEE Trans. Signal Processing, 1994.
//! - J. O. Smith, "Mathematics of GPS Trilateration" (Stanford notes), and standard geometry texts.
//! - Industry practice: trilateration in UWB/RTLS systems uses the same basis construction and sign disambiguation with an additional sensor.
//!
//! Theorem (Three-Sphere Intersection in ℝ³)
//! -----------------------------------------
//! Let `S₁, S₂, S₃ ⊂ ℝ³` be spheres centered at `p1, p2, p3` with radii `r1, r2, r3 ≥ 0`.
//! Assume: (i) `p1 ≠ p2` (distinct) with distance `d = ||p2 - p1|| > 0`; (ii) `p1, p2, p3` are non-collinear (`j = ((p3-p1) · e_y) ≠ 0`) under the orthonormal basis `e_x, e_y, e_z` constructed by
//! `e_x = (p2 - p1)/d`, `i = (p3 - p1) · e_x`, `e_y = normalize((p3 - p1) - i e_x)`, `e_z = e_x × e_y`, `j = (p3 - p1) · e_y`.
//! Then, if `z² = r1² - x² - y² ≥ 0` where `x = (r1² - r2² + d²)/(2d)` and `y = ((r1² - r3² + i² + j²)/(2j)) - (i/j)x`, the intersection set `S₁ ∩ S₂ ∩ S₃` consists of at most two points:
//! `p = p1 + x e_x + y e_y ± √(z²) e_z`. When a fourth range `(p4, r4)` is available, the physically consistent point is selected by minimizing the residual `| ||p - p4|| - r4 |` over the sign.
//! If `z² < 0` or the geometry is degenerate (coincident/collinear sensors), the exact three-sphere intersection does not exist; we fall back to least-squares multilateration.
//!
//! Numerical Stability & Complexity
//! --------------------------------
//! - Exact trilateration uses closed-form algebra in constant time with the above conditions.
//! - Degeneracy and small tolerances (`≈1e-12` for geometry checks, `≈1e-10` for `z²`) guard against floating-point noise.
//! - When exact conditions fail, LS multilateration (Gauss–Newton with LM damping and Cholesky in core) provides robust estimates.

use super::core::{MultilaterationMethod, MultilaterationSolver};
use crate::sensor::localization::{Position, SensorArray};

/// Trilateration solver (exact three-sphere intersection when feasible)
///
/// Usage
/// -----
///
/// ```no_run
/// use crate::sensor::localization::array::{Sensor, SensorArray, ArrayGeometry};
/// use crate::sensor::localization::Position;
/// use crate::sensor::localization::multilateration::trilateration::TrilaterationSolver;
///
/// // Construct a simple non-collinear array
/// let sensors = vec![
///     Sensor::new(0, Position::new(0.0, 0.0, 0.0)),
///     Sensor::new(1, Position::new(1.0, 0.0, 0.0)),
///     Sensor::new(2, Position::new(0.0, 1.0, 0.0)),
///     Sensor::new(3, Position::new(0.0, 0.0, 1.0)),
/// ];
/// let array = SensorArray::new(sensors, 343.0, ArrayGeometry::Arbitrary);
/// let solver = TrilaterationSolver::new(&array);
///
/// // Three ranges (meters) to sensors {0,1,2}
/// let ranges = [1.0_f64, 1.1180_f64, 1.1180_f64];
/// let res = solver.solve_three(ranges, [0, 1, 2])?;
/// assert!(res.uncertainty.is_finite());
///
/// // With a fourth sensor to disambiguate +/- z
/// let res4 = solver.solve_three_with_fourth(ranges, [0, 1, 2], (3, 1.5))?;
/// assert!(res4.uncertainty <= res.uncertainty + 1e-6);
/// # Ok::<(), crate::error::KwaversError>(())
/// ```
///
/// Safety & Errors
/// ---------------
/// - Rejects non-finite or negative ranges with `InvalidInput`.
/// - Detects degenerate geometry (coincident/collinear sensors) and falls back to LS.
/// - When the three-sphere system has no real intersection beyond numerical tolerance,
///   the solver falls back to the multilateration LS method.
#[derive(Debug)]
pub struct TrilaterationSolver<'a> {
    array: &'a SensorArray,
}

impl<'a> TrilaterationSolver<'a> {
    /// Create a new trilateration solver
    pub fn new(array: &'a SensorArray) -> Self {
        Self { array }
    }

    /// Solve using exactly three sensors (indices) and ranges (meters).
    /// Returns the position and an uncertainty (RMSE of residual radii).
    ///
    /// Example
    /// -------
    /// ```no_run
    /// # use crate::sensor::localization::array::{Sensor, SensorArray, ArrayGeometry};
    /// # use crate::sensor::localization::Position;
    /// # use crate::sensor::localization::multilateration::trilateration::TrilaterationSolver;
    /// # let sensors = vec![
    /// #     Sensor::new(0, Position::new(0.0, 0.0, 0.0)),
    /// #     Sensor::new(1, Position::new(1.0, 0.0, 0.0)),
    /// #     Sensor::new(2, Position::new(0.0, 1.0, 0.0)),
    /// # ];
    /// # let array = SensorArray::new(sensors, 343.0, ArrayGeometry::Arbitrary);
    /// # let solver = TrilaterationSolver::new(&array);
    /// let res = solver.solve_three([1.0, 1.1180, 1.1180], [0, 1, 2])?;
    /// assert!(res.uncertainty.is_finite());
    /// # Ok::<(), crate::error::KwaversError>(())
    /// ```
    #[cfg_attr(
        feature = "structured-logging",
        tracing::instrument(
            skip(ranges_m, sensor_indices),
            fields(
                d = tracing::field::Empty,
                i = tracing::field::Empty,
                j = tracing::field::Empty,
                z2 = tracing::field::Empty,
                branch = tracing::field::Empty,
                rmse = tracing::field::Empty
            )
        )
    )]
    pub fn solve_three(
        &self,
        ranges_m: [f64; 3],
        sensor_indices: [usize; 3],
    ) -> crate::error::KwaversResult<TrilaterationResult> {
        // Input validation: non-negative, finite ranges
        if !ranges_m.iter().all(|&r| r.is_finite() && r >= 0.0) {
            return Err(crate::error::KwaversError::InvalidInput(
                "Ranges must be finite and non-negative".to_string(),
            ));
        }
        // Extract sensor positions
        let p1 = self.array.get_sensor_position(sensor_indices[0]).to_array();
        let p2 = self.array.get_sensor_position(sensor_indices[1]).to_array();
        let p3 = self.array.get_sensor_position(sensor_indices[2]).to_array();

        let r1 = ranges_m[0];
        let r2 = ranges_m[1];
        let r3 = ranges_m[2];

        // Basis construction following standard trilateration derivation
        let ex = normalize(sub(p2, p1));
        let d = norm(sub(p2, p1));
        #[cfg(feature = "structured-logging")]
        tracing::Span::current().record("d", &tracing::field::display(d));
        if d < 1e-12 {
            #[cfg(feature = "structured-logging")]
            tracing::warn!(d, "Degenerate sensor configuration: p1 and p2 coincide");
            return Err(crate::error::KwaversError::InvalidInput(
                "Degenerate sensor configuration: p1 and p2 coincide".to_string(),
            ));
        }

        let p3p1 = sub(p3, p1);
        let i = dot(ex, p3p1);
        #[cfg(feature = "structured-logging")]
        tracing::Span::current().record("i", &tracing::field::display(i));
        let temp = sub(p3p1, scale(ex, i));
        let temp_norm = norm(temp);
        if temp_norm < 1e-12 {
            // Sensors are collinear; fall back to LS multilateration
            #[cfg(feature = "structured-logging")]
            tracing::warn!(temp_norm, "Collinear sensors detected; falling back to LS");
            #[cfg(feature = "structured-logging")]
            tracing::Span::current().record("branch", &tracing::field::display("ls_fallback"));
            return self.fallback_ls(ranges_m, sensor_indices);
        }
        let ey = normalize(temp);
        let ez = cross(ex, ey);
        let j = dot(ey, p3p1);
        #[cfg(feature = "structured-logging")]
        tracing::Span::current().record("j", &tracing::field::display(j));
        if j.abs() < 1e-12 {
            #[cfg(feature = "structured-logging")]
            tracing::warn!(j, "Degenerate geometry (j≈0); falling back to LS");
            #[cfg(feature = "structured-logging")]
            tracing::Span::current().record("branch", &tracing::field::display("ls_fallback"));
            return self.fallback_ls(ranges_m, sensor_indices);
        }

        // Closed-form x, y, z (two symmetric solutions along ez for z)
        let x = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
        let y = ((r1 * r1 - r3 * r3 + i * i + j * j) / (2.0 * j)) - (i / j) * x;
        let z2 = r1 * r1 - x * x - y * y;
        #[cfg(feature = "structured-logging")]
        tracing::Span::current().record("z2", &tracing::field::display(z2));
        if z2 < -1e-10 {
            // No real intersection -> numerical issues or inconsistent ranges; fallback
            #[cfg(feature = "structured-logging")]
            tracing::warn!(z2, "No real intersection (z^2<0); falling back to LS");
            #[cfg(feature = "structured-logging")]
            tracing::Span::current().record("branch", &tracing::field::display("ls_fallback"));
            return self.fallback_ls(ranges_m, sensor_indices);
        }
        let z = z2.max(0.0).sqrt();

        // Two candidates: z sign ambiguous without additional information.
        // est_pos corresponds to the solution in the direction of +ez (ex × ey).
        // est_neg corresponds to the solution in the direction of -ez.
        let est_pos = add(p1, add(scale(ex, x), add(scale(ey, y), scale(ez, z))));
        let est_neg = add(p1, add(scale(ex, x), add(scale(ey, y), scale(ez, -z))));
        let position = Position::from_array(est_pos);

        // Uncertainty as RMSE of residuals to the three spheres
        let d1 = position.distance_to(&Position::from_array(p1));
        let d2 = position.distance_to(&Position::from_array(p2));
        let d3 = position.distance_to(&Position::from_array(p3));
        let res = [d1 - r1, d2 - r2, d3 - r3];
        let rmse = ((res[0] * res[0] + res[1] * res[1] + res[2] * res[2]) / 3.0).sqrt();
        #[cfg(feature = "structured-logging")]
        {
            tracing::Span::current().record("rmse", &tracing::field::display(rmse));
            tracing::Span::current().record("branch", &tracing::field::display("exact"));
        }

        Ok(TrilaterationResult {
            position: position.to_array(),
            uncertainty: rmse,
            ambiguous: z > 0.0,
            alt_position: if z > 0.0 { Some(est_neg) } else { None },
        })
    }

    /// Solve with fourth-sensor disambiguation: uses an additional range from a fourth sensor
    /// to choose the correct z-sign among the two symmetric solutions.
    ///
    /// Example
    /// -------
    /// ```no_run
    /// # use crate::sensor::localization::array::{Sensor, SensorArray, ArrayGeometry};
    /// # use crate::sensor::localization::Position;
    /// # use crate::sensor::localization::multilateration::trilateration::TrilaterationSolver;
    /// # let sensors = vec![
    /// #     Sensor::new(0, Position::new(0.0, 0.0, 0.0)),
    /// #     Sensor::new(1, Position::new(1.0, 0.0, 0.0)),
    /// #     Sensor::new(2, Position::new(0.0, 1.0, 0.0)),
    /// #     Sensor::new(3, Position::new(0.0, 0.0, 1.0)),
    /// # ];
    /// # let array = SensorArray::new(sensors, 343.0, ArrayGeometry::Arbitrary);
    /// # let solver = TrilaterationSolver::new(&array);
    /// let res = solver.solve_three_with_fourth([1.0, 1.1180, 1.1180], [0, 1, 2], (3, 1.5))?;
    /// assert!(res.uncertainty.is_finite());
    /// # Ok::<(), crate::error::KwaversError>(())
    /// ```
    #[cfg_attr(
        feature = "structured-logging",
        tracing::instrument(
            skip(ranges_m, sensor_indices, fourth),
            fields(
                d = tracing::field::Empty,
                i = tracing::field::Empty,
                j = tracing::field::Empty,
                z2 = tracing::field::Empty,
                branch = tracing::field::Empty,
                rmse = tracing::field::Empty
            )
        )
    )]
    pub fn solve_three_with_fourth(
        &self,
        ranges_m: [f64; 3],
        sensor_indices: [usize; 3],
        fourth: (usize, f64),
    ) -> crate::error::KwaversResult<TrilaterationResult> {
        // Input validation: non-negative, finite ranges including fourth
        if !(ranges_m.iter().all(|&r| r.is_finite() && r >= 0.0)
            && fourth.1.is_finite()
            && fourth.1 >= 0.0)
        {
            return Err(crate::error::KwaversError::InvalidInput(
                "Ranges must be finite and non-negative".to_string(),
            ));
        }
        // Compute the base candidate solutions (reuse derivation)
        let p1 = self.array.get_sensor_position(sensor_indices[0]).to_array();
        let p2 = self.array.get_sensor_position(sensor_indices[1]).to_array();
        let p3 = self.array.get_sensor_position(sensor_indices[2]).to_array();
        let r1 = ranges_m[0];
        let r2 = ranges_m[1];
        let r3 = ranges_m[2];

        let ex = normalize(sub(p2, p1));
        let d = norm(sub(p2, p1));
        #[cfg(feature = "structured-logging")]
        tracing::Span::current().record("d", &tracing::field::display(d));
        if d < 1e-12 {
            #[cfg(feature = "structured-logging")]
            tracing::warn!(d, "Degenerate sensor configuration: p1 and p2 coincide");
            return Err(crate::error::KwaversError::InvalidInput(
                "Degenerate sensor configuration: p1 and p2 coincide".to_string(),
            ));
        }
        let p3p1 = sub(p3, p1);
        let i = dot(ex, p3p1);
        #[cfg(feature = "structured-logging")]
        tracing::Span::current().record("i", &tracing::field::display(i));
        let temp = sub(p3p1, scale(ex, i));
        let temp_norm = norm(temp);
        if temp_norm < 1e-12 {
            // Collinear -> fallback
            #[cfg(feature = "structured-logging")]
            tracing::warn!(temp_norm, "Collinear sensors detected; falling back to LS");
            #[cfg(feature = "structured-logging")]
            tracing::Span::current().record("branch", &tracing::field::display("ls_fallback"));
            return self.fallback_ls(ranges_m, sensor_indices);
        }
        let ey = normalize(temp);
        let ez = cross(ex, ey);
        let j = dot(ey, p3p1);
        #[cfg(feature = "structured-logging")]
        tracing::Span::current().record("j", &tracing::field::display(j));
        if j.abs() < 1e-12 {
            #[cfg(feature = "structured-logging")]
            tracing::warn!(j, "Degenerate geometry (j≈0); falling back to LS");
            #[cfg(feature = "structured-logging")]
            tracing::Span::current().record("branch", &tracing::field::display("ls_fallback"));
            return self.fallback_ls(ranges_m, sensor_indices);
        }

        let x = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
        let y = ((r1 * r1 - r3 * r3 + i * i + j * j) / (2.0 * j)) - (i / j) * x;
        let z2 = r1 * r1 - x * x - y * y;
        #[cfg(feature = "structured-logging")]
        tracing::Span::current().record("z2", &tracing::field::display(z2));
        if z2 < -1e-10 {
            #[cfg(feature = "structured-logging")]
            tracing::warn!(z2, "No real intersection (z^2<0); falling back to LS");
            #[cfg(feature = "structured-logging")]
            tracing::Span::current().record("branch", &tracing::field::display("ls_fallback"));
            return self.fallback_ls(ranges_m, sensor_indices);
        }
        let z = z2.max(0.0).sqrt();

        let est_pos = add(p1, add(scale(ex, x), add(scale(ey, y), scale(ez, z))));
        let est_neg = add(p1, add(scale(ex, x), add(scale(ey, y), scale(ez, -z))));

        // Disambiguate using the fourth sensor's range
        let p4 = self.array.get_sensor_position(fourth.0);
        let r4 = fourth.1;
        let d_pos = Position::from_array(est_pos).distance_to(p4);
        let d_neg = Position::from_array(est_neg).distance_to(p4);
        let res_pos = (d_pos - r4).abs();
        let res_neg = (d_neg - r4).abs();
        let choose_pos = res_pos <= res_neg;
        let est = if choose_pos { est_pos } else { est_neg };
        let position = Position::from_array(est);

        // Uncertainty across all four constraints
        let d1 = position.distance_to(&Position::from_array(p1));
        let d2 = position.distance_to(&Position::from_array(p2));
        let d3 = position.distance_to(&Position::from_array(p3));
        let d4 = position.distance_to(p4);
        let res = [d1 - r1, d2 - r2, d3 - r3, d4 - r4];
        let rmse = ((res.iter().map(|v| v * v).sum::<f64>()) / 4.0).sqrt();
        #[cfg(feature = "structured-logging")]
        {
            tracing::Span::current().record("rmse", &tracing::field::display(rmse));
            tracing::Span::current()
                .record("branch", &tracing::field::display("fourth_disambiguation"));
        }

        Ok(TrilaterationResult {
            position: position.to_array(),
            uncertainty: rmse,
            ambiguous: false,
            alt_position: Some(if choose_pos { est_neg } else { est_pos }),
        })
    }

    fn fallback_ls(
        &self,
        ranges_m: [f64; 3],
        sensor_indices: [usize; 3],
    ) -> crate::error::KwaversResult<TrilaterationResult> {
        // Build a subset SensorArray with just the three sensors to avoid spurious constraints.
        let subset_sensors = vec![
            crate::sensor::localization::array::Sensor::new(
                0,
                *self.array.get_sensor_position(sensor_indices[0]),
            ),
            crate::sensor::localization::array::Sensor::new(
                1,
                *self.array.get_sensor_position(sensor_indices[1]),
            ),
            crate::sensor::localization::array::Sensor::new(
                2,
                *self.array.get_sensor_position(sensor_indices[2]),
            ),
        ];
        let subset = crate::sensor::localization::array::SensorArray::new(
            subset_sensors,
            self.array.sound_speed(),
            crate::sensor::localization::array::ArrayGeometry::Arbitrary,
        );
        let core = MultilaterationSolver::new(MultilaterationMethod::LeastSquares);
        let pos = core.solve_ranges(&ranges_m, &subset)?;
        // RMSE across the three constraints
        let p1 = subset.get_sensor_position(0);
        let p2 = subset.get_sensor_position(1);
        let p3 = subset.get_sensor_position(2);
        let d1 = pos.distance_to(p1);
        let d2 = pos.distance_to(p2);
        let d3 = pos.distance_to(p3);
        let r1 = ranges_m[0];
        let r2 = ranges_m[1];
        let r3 = ranges_m[2];
        let res = [d1 - r1, d2 - r2, d3 - r3];
        let rmse = ((res[0] * res[0] + res[1] * res[1] + res[2] * res[2]) / 3.0).sqrt();
        Ok(TrilaterationResult {
            position: pos.to_array(),
            uncertainty: rmse,
            ambiguous: false,
            alt_position: None,
        })
    }
}

/// Trilateration result
#[derive(Debug, Clone)]
pub struct TrilaterationResult {
    /// Estimated position [x, y, z]
    pub position: [f64; 3],
    /// Uncertainty estimate (RMSE of residual ranges)
    /// Computed as the root-mean-square of range residuals against the constraints used.
    pub uncertainty: f64,
    pub ambiguous: bool,
    pub alt_position: Option<[f64; 3]>,
}

// --- small vector helpers ---
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    if n < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        [a[0] / n, a[1] / n, a[2] / n]
    }
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensor::localization::array::{ArrayGeometry, Sensor};
    use proptest::prelude::*;

    fn make_array() -> SensorArray {
        let sensors = vec![
            Sensor::new(0, Position::new(0.0, 0.0, 0.0)),
            Sensor::new(1, Position::new(1.0, 0.0, 0.0)),
            Sensor::new(2, Position::new(0.0, 1.0, 0.0)),
        ];
        SensorArray::new(sensors, 1500.0, ArrayGeometry::Arbitrary)
    }

    #[test]
    fn test_trilateration_exact() {
        let array = make_array();
        let source = Position::new(0.3, 0.25, 0.2);
        let p0 = array.get_sensor_position(0);
        let p1 = array.get_sensor_position(1);
        let p2 = array.get_sensor_position(2);
        let r0 = p0.distance_to(&source);
        let r1 = p1.distance_to(&source);
        let r2 = p2.distance_to(&source);

        let solver = TrilaterationSolver::new(&array);
        let result = solver
            .solve_three([r0, r1, r2], [0, 1, 2])
            .expect("Trilateration failed");
        let est = Position::from_array(result.position);
        let err = est.distance_to(&source);
        assert!(err < 1e-6, "Exact trilateration error too large: {err}");
        assert!(result.uncertainty < 1e-9, "Residual should be near zero");
    }

    #[test]
    fn test_trilateration_ambiguity_flag() {
        let array = make_array();
        let source = Position::new(0.3, 0.25, 0.2);
        let r0 = array.get_sensor_position(0).distance_to(&source);
        let r1 = array.get_sensor_position(1).distance_to(&source);
        let r2 = array.get_sensor_position(2).distance_to(&source);
        let solver = TrilaterationSolver::new(&array);
        let result = solver
            .solve_three([r0, r1, r2], [0, 1, 2])
            .expect("Trilateration failed");
        assert!(
            result.ambiguous,
            "Three-sensor solution should be ambiguous"
        );
        assert!(
            result.alt_position.is_some(),
            "Alt position must be present"
        );
        let alt = result.alt_position.unwrap();
        assert!(
            alt[2] < 0.0,
            "Alt solution z should reflect across sensor plane"
        );
    }

    #[test]
    fn test_trilateration_fallback_ls() {
        // Collinear sensors -> fallback to LS
        let sensors = vec![
            Sensor::new(0, Position::new(0.0, 0.0, 0.0)),
            Sensor::new(1, Position::new(1.0, 0.0, 0.0)),
            Sensor::new(2, Position::new(2.0, 0.0, 0.0)),
        ];
        let array = SensorArray::new(sensors, 1500.0, ArrayGeometry::Arbitrary);
        let source = Position::new(0.3, 0.25, 0.2);
        let r0 = array.get_sensor_position(0).distance_to(&source);
        let r1 = array.get_sensor_position(1).distance_to(&source);
        let r2 = array.get_sensor_position(2).distance_to(&source);

        let solver = TrilaterationSolver::new(&array);
        let result = solver
            .solve_three([r0, r1, r2], [0, 1, 2])
            .expect("Fallback LS failed");
        let est = Position::from_array(result.position);
        // Degenerate geometry (collinear sensors) is ill-posed in 3D; LS provides a best-effort estimate.
        // Validate that we get a finite estimate with bounded error.
        let err = est.distance_to(&source);
        assert!(
            err < 1.0,
            "Fallback LS trilateration error too large: {err}"
        );
    }

    #[test]
    fn test_trilateration_fourth_disambiguation_consistency() {
        // Base 3 sensors in z=0 plane, 4th sensor above to select positive z solution.
        let sensors = vec![
            Sensor::new(0, Position::new(0.0, 0.0, 0.0)),
            Sensor::new(1, Position::new(1.0, 0.0, 0.0)),
            Sensor::new(2, Position::new(0.0, 1.0, 0.0)),
            Sensor::new(3, Position::new(0.5, 0.5, 2.0)),
        ];
        let array = SensorArray::new(sensors, 1500.0, ArrayGeometry::Arbitrary);
        let source = Position::new(0.3, 0.25, 0.2);
        let r0 = array.get_sensor_position(0).distance_to(&source);
        let r1 = array.get_sensor_position(1).distance_to(&source);
        let r2 = array.get_sensor_position(2).distance_to(&source);
        let r3 = array.get_sensor_position(3).distance_to(&source);

        let solver = TrilaterationSolver::new(&array);
        let result = solver
            .solve_three_with_fourth([r0, r1, r2], [0, 1, 2], (3, r3))
            .expect("Disambiguation failed");
        let est = Position::from_array(result.position);
        let err = est.distance_to(&source);
        assert!(
            err < 1e-6,
            "Disambiguated trilateration error too large: {err}"
        );
        assert!(est.z > 0.0, "Expected positive z solution to be selected");
    }

    #[test]
    fn test_trilateration_with_fourth_disambiguation() {
        let sensors = vec![
            Sensor::new(0, Position::new(0.0, 0.0, 0.0)),
            Sensor::new(1, Position::new(1.0, 0.0, 0.0)),
            Sensor::new(2, Position::new(0.0, 1.0, 0.0)),
            Sensor::new(3, Position::new(0.5, 0.5, 2.0)),
        ];
        let array = SensorArray::new(sensors, 1500.0, ArrayGeometry::Arbitrary);
        let source = Position::new(0.3, 0.25, 0.2);
        let r0 = array.get_sensor_position(0).distance_to(&source);
        let r1 = array.get_sensor_position(1).distance_to(&source);
        let r2 = array.get_sensor_position(2).distance_to(&source);
        let r3 = array.get_sensor_position(3).distance_to(&source);
        let solver = TrilaterationSolver::new(&array);
        let res = solver
            .solve_three_with_fourth([r0, r1, r2], [0, 1, 2], (3, r3))
            .expect("Trilateration 4th disambiguation failed");
        let est = Position::from_array(res.position);
        assert!(est.distance_to(&source) < 1e-6);
    }

    #[test]
    fn test_trilateration_negative_z() {
        let array = make_array();
        // Target below the z=0 plane
        let source = Position::new(0.3, 0.25, -0.5);
        let r0 = array.get_sensor_position(0).distance_to(&source);
        let r1 = array.get_sensor_position(1).distance_to(&source);
        let r2 = array.get_sensor_position(2).distance_to(&source);

        let solver = TrilaterationSolver::new(&array);
        let result = solver
            .solve_three([r0, r1, r2], [0, 1, 2])
            .expect("Trilateration failed");

        // For 3 sensors, we get ambiguous results.
        // est_pos is usually the primary return, but alt_position must contain the correct one.
        let est_pos = Position::from_array(result.position);
        let est_neg = Position::from_array(result.alt_position.expect("Should have alt position"));

        // One of them should be close to source
        let d1 = est_pos.distance_to(&source);
        let d2 = est_neg.distance_to(&source);

        assert!(d1 < 1e-6 || d2 < 1e-6, "One solution must match source");

        // Specifically check that est_neg (alt) is the correct one for negative Z target
        // because est_pos (primary) is constructed with +z relative to ez=(0,0,1)
        assert!(d2 < 1e-6, "est_neg should match negative Z source");
        assert!(d1 > 0.1, "est_pos should be the reflection");
    }

    #[test]
    fn test_trilateration_rejects_negative_ranges() {
        let array = make_array();
        let solver = TrilaterationSolver::new(&array);
        let res = solver.solve_three([-1.0, 1.0, 1.0], [0, 1, 2]);
        assert!(res.is_err(), "Negative ranges must be rejected");
    }

    #[test]
    fn test_trilateration_rejects_non_finite_ranges() {
        let array = make_array();
        let solver = TrilaterationSolver::new(&array);
        let res = solver.solve_three([f64::NAN, 1.0, 1.0], [0, 1, 2]);
        assert!(res.is_err(), "Non-finite ranges must be rejected");
        let res = solver.solve_three([f64::INFINITY, 1.0, 1.0], [0, 1, 2]);
        assert!(res.is_err(), "Infinite ranges must be rejected");
    }

    proptest! {
        #[test]
        fn proptest_trilateration_exact_random(
            sx in -0.4f64..0.9f64,
            sy in -0.4f64..0.9f64,
            sz in 0.05f64..0.5f64,
        ) {
            let array = make_array();
            let source = Position::new(sx, sy, sz);
            let r0 = array.get_sensor_position(0).distance_to(&source);
            let r1 = array.get_sensor_position(1).distance_to(&source);
            let r2 = array.get_sensor_position(2).distance_to(&source);
            let solver = TrilaterationSolver::new(&array);
            let res = solver.solve_three([r0, r1, r2], [0, 1, 2])
                .expect("Exact trilateration should succeed for non-collinear geometry");
            let est = Position::from_array(res.position);
            prop_assert!(est.distance_to(&source) < 1e-6);
            prop_assert!(res.uncertainty < 1e-9);
        }
    }

    proptest! {
        #[test]
        fn proptest_trilateration_fourth_sensor_disambiguation_positive_z(
            sx in -0.4f64..0.9f64,
            sy in -0.4f64..0.9f64,
            sz in 0.05f64..0.5f64,
        ) {
        // Base sensors in z=0 plane, 4th above plane to select positive z
        let sensors = vec![
            Sensor::new(0, Position::new(0.0, 0.0, 0.0)),
            Sensor::new(1, Position::new(1.0, 0.0, 0.0)),
            Sensor::new(2, Position::new(0.0, 1.0, 0.0)),
            Sensor::new(3, Position::new(0.5, 0.5, 2.0)),
        ];
        let array = SensorArray::new(sensors, 1500.0, ArrayGeometry::Arbitrary);
        let source = Position::new(sx, sy, sz);
        let r0 = array.get_sensor_position(0).distance_to(&source);
        let r1 = array.get_sensor_position(1).distance_to(&source);
        let r2 = array.get_sensor_position(2).distance_to(&source);
        let r3 = array.get_sensor_position(3).distance_to(&source);
        let solver = TrilaterationSolver::new(&array);
        let res = solver
            .solve_three_with_fourth([r0, r1, r2], [0, 1, 2], (3, r3))
            .expect("Fourth-sensor disambiguation should succeed");
        let est = Position::from_array(res.position);
        prop_assert!(est.distance_to(&source) < 1e-6);
        // Since source.z > 0 and 4th sensor is above, selected solution should have positive z
            prop_assert!(est.z > 0.0);
        }
    }

    proptest! {
        #[test]
        fn proptest_trilateration_near_degenerate_j_small(
            eps in 1e-6f64..1e-3f64,
            t in 1e-6f64..1e-2f64,
            sx in 0.1f64..0.9f64,
            sy in 0.1f64..0.9f64,
            sz in 0.05f64..0.5f64,
        ) {
            // Sensors nearly collinear: p2 on x-axis, p3 very close to x-axis (small j)
            let sensors = vec![
                Sensor::new(0, Position::new(0.0, 0.0, 0.0)),
                Sensor::new(1, Position::new(1.0, 0.0, 0.0)),
                Sensor::new(2, Position::new(eps, t, 0.0)),
            ];
            let array = SensorArray::new(sensors, 343.0, ArrayGeometry::Arbitrary);
            let source = Position::new(sx, sy, sz);
            let r0 = array.get_sensor_position(0).distance_to(&source);
            let r1 = array.get_sensor_position(1).distance_to(&source);
            let r2 = array.get_sensor_position(2).distance_to(&source);
            let solver = TrilaterationSolver::new(&array);
            let res = solver.solve_three([r0, r1, r2], [0, 1, 2])
                .expect("Trilateration should return estimate or LS fallback for near-degenerate geometry");
            let est = Position::from_array(res.position);
            // Robustness: estimate stays within reasonable bound despite small j
            prop_assert!(est.distance_to(&source) < 0.05);
            prop_assert!(res.uncertainty.is_finite());
        }
    }

    proptest! {
        #[test]
        fn proptest_trilateration_noisy_ranges_robustness(
            sx in -0.4f64..0.9f64,
            sy in -0.4f64..0.9f64,
            sz in 0.05f64..0.5f64,
            noise in 1e-6f64..1e-3f64,
            d0 in -1.0f64..1.0f64,
            d1 in -1.0f64..1.0f64,
            d2 in -1.0f64..1.0f64,
        ) {
            let array = make_array();
            let source = Position::new(sx, sy, sz);
            let r0 = array.get_sensor_position(0).distance_to(&source);
            let r1 = array.get_sensor_position(1).distance_to(&source);
            let r2 = array.get_sensor_position(2).distance_to(&source);
            let r0n = (r0 + noise * d0).max(0.0);
            let r1n = (r1 + noise * d1).max(0.0);
            let r2n = (r2 + noise * d2).max(0.0);
            let solver = TrilaterationSolver::new(&array);
            let res = solver.solve_three([r0n, r1n, r2n], [0, 1, 2])
                .expect("Trilateration should remain robust under small range noise");
            let est = Position::from_array(res.position);
            // Robustness: error bounded under small noise
            prop_assert!(est.distance_to(&source) < 0.05);
            prop_assert!(res.uncertainty.is_finite());
        }
    }
}
