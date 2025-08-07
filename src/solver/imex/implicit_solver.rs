//! Implicit solvers for IMEX schemes

use ndarray::{Array3, Zip};
use crate::error::{KwaversResult, KwaversError};
use std::fmt::Debug;

/// Trait for implicit solvers
pub trait ImplicitSolver: Debug + Send + Sync {
    /// Solve the implicit equation: F(y) = 0
    fn solve<F>(
        &self,
        initial_guess: &Array3<f64>,
        residual_fn: F,
    ) -> KwaversResult<Array3<f64>>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>;
    
    /// Get solver tolerance
    fn tolerance(&self) -> f64;
    
    /// Get maximum iterations
    fn max_iterations(&self) -> usize;
}

/// Linear solver for implicit equations
#[derive(Debug, Clone)]
pub struct LinearSolver {
    tolerance: f64,
    max_iterations: usize,
}

impl Default for LinearSolver {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            max_iterations: 100,
        }
    }
}

impl LinearSolver {
    /// Create a new linear solver
    pub fn new(tolerance: f64, max_iterations: usize) -> Self {
        Self {
            tolerance,
            max_iterations,
        }
    }
}

impl ImplicitSolver for LinearSolver {
    fn solve<F>(
        &self,
        initial_guess: &Array3<f64>,
        residual_fn: F,
    ) -> KwaversResult<Array3<f64>>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        // For linear problems, we can use a simple fixed-point iteration
        // This assumes the problem can be written as y = G(y)
        let mut solution = initial_guess.clone();
        let mut last_norm = f64::INFINITY;
        
        for iter in 0..self.max_iterations {
            let residual = residual_fn(&solution)?;
            
            // Check convergence
            let norm = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
            last_norm = norm;
            if norm < self.tolerance {
                return Ok(solution);
            }
            
            // Update solution (assuming residual = y - G(y))
            Zip::from(&mut solution)
                .and(&residual)
                .for_each(|s, &r| *s -= r);
        }
        
        Err(KwaversError::Physics(crate::error::PhysicsError::ConvergenceFailure {
            iteration: self.max_iterations,
            max_iterations: self.max_iterations,
            residual: last_norm,
            tolerance: self.tolerance,
        }))
    }
    
    fn tolerance(&self) -> f64 {
        self.tolerance
    }
    
    fn max_iterations(&self) -> usize {
        self.max_iterations
    }
}

/// Newton solver for nonlinear implicit equations
#[derive(Debug, Clone)]
pub struct NewtonSolver {
    tolerance: f64,
    max_iterations: usize,
    jacobian_epsilon: f64,
    line_search: bool,
}

impl Default for NewtonSolver {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            max_iterations: 50,
            jacobian_epsilon: 1e-7,
            line_search: true,
        }
    }
}

impl NewtonSolver {
    /// Create a new Newton solver
    pub fn new(tolerance: f64, max_iterations: usize) -> Self {
        Self {
            tolerance,
            max_iterations,
            jacobian_epsilon: 1e-7,
            line_search: true,
        }
    }
    
    /// Set whether to use line search
    pub fn with_line_search(mut self, line_search: bool) -> Self {
        self.line_search = line_search;
        self
    }
    
    /// Approximate Jacobian-vector product using finite differences
    fn jacobian_vector_product<F>(
        &self,
        y: &Array3<f64>,
        v: &Array3<f64>,
        residual_fn: &F,
    ) -> KwaversResult<Array3<f64>>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        let mut y_perturbed = y.clone();
        Zip::from(&mut y_perturbed)
            .and(v)
            .for_each(|yp, &vi| *yp += self.jacobian_epsilon * vi);
        
        let f_perturbed = residual_fn(&y_perturbed)?;
        let f_base = residual_fn(y)?;
        
        let mut jv = Array3::zeros(y.dim());
        Zip::from(&mut jv)
            .and(&f_perturbed)
            .and(&f_base)
            .for_each(|j, &fp, &fb| {
                *j = (fp - fb) / self.jacobian_epsilon;
            });
        
        Ok(jv)
    }
    
    /// Solve linear system using GMRES-like iteration
    fn solve_linear_system<F>(
        &self,
        y: &Array3<f64>,
        residual: &Array3<f64>,
        residual_fn: &F,
    ) -> KwaversResult<Array3<f64>>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        // Simplified conjugate gradient for the Newton system
        // J * delta = -residual
        let mut delta = Array3::zeros(y.dim());
        let mut r = -residual;
        let mut p = r.clone();
        
        for _ in 0..20 { // Inner iterations
            let jp = self.jacobian_vector_product(y, &p, residual_fn)?;
            
            let r_norm_sq: f64 = r.iter().map(|&x| x * x).sum();
            let alpha = r_norm_sq / p.iter().zip(&jp).map(|(&pi, &jpi)| pi * jpi).sum::<f64>();
            
            Zip::from(&mut delta)
                .and(&p)
                .for_each(|d, &pi| *d += alpha * pi);
            
            Zip::from(&mut r)
                .and(&jp)
                .for_each(|ri, &jpi| *ri -= alpha * jpi);
            
            let r_norm_sq_updated: f64 = r.iter().map(|&x| x * x).sum();
            if r_norm_sq_updated.sqrt() < self.tolerance {
                break;
            }
            
            let beta = r_norm_sq_updated / r_norm_sq;
            Zip::from(&mut p)
                .and(&r)
                .for_each(|pi, &ri| *pi = ri + beta * *pi);
        }
        
        Ok(delta)
    }
}

impl ImplicitSolver for NewtonSolver {
    fn solve<F>(
        &self,
        initial_guess: &Array3<f64>,
        residual_fn: F,
    ) -> KwaversResult<Array3<f64>>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        let mut solution = initial_guess.clone();
        let mut last_norm = f64::INFINITY;
        
        for iter in 0..self.max_iterations {
            let residual = residual_fn(&solution)?;
            
            // Check convergence
            let norm = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
            last_norm = norm;
            if norm < self.tolerance {
                return Ok(solution);
            }
            
            // Solve for Newton step
            let delta = self.solve_linear_system(&solution, &residual, &residual_fn)?;
            
            // Line search if enabled
            let mut alpha = 1.0;
            if self.line_search {
                let mut test_solution = solution.clone();
                for _ in 0..10 {
                    Zip::from(&mut test_solution)
                        .and(&solution)
                        .and(&delta)
                        .for_each(|t, &s, &d| *t = s + alpha * d);
                    
                    let test_residual = residual_fn(&test_solution)?;
                    let test_norm = test_residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
                    
                    if test_norm < norm {
                        break;
                    }
                    alpha *= 0.5;
                }
            }
            
            // Update solution
            Zip::from(&mut solution)
                .and(&delta)
                .for_each(|s, &d| *s += alpha * d);
        }
        
        Err(KwaversError::Physics(crate::error::PhysicsError::ConvergenceFailure {
            iteration: self.max_iterations,
            max_iterations: self.max_iterations,
            residual: last_norm,
            tolerance: self.tolerance,
        }))
    }
    
    fn tolerance(&self) -> f64 {
        self.tolerance
    }
    
    fn max_iterations(&self) -> usize {
        self.max_iterations
    }
}