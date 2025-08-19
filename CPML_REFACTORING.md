# Critical CPML Architecture Refactoring

## Executive Summary

The current CPML implementation has fundamental architectural flaws that violate physics principles and software design best practices. These must be fixed immediately.

## Issue 1: CPML is NOT a Simple Boundary Condition ❌

### Problem
The `Boundary` trait assumes boundary conditions are simple operations applied to fields:
```rust
trait Boundary {
    fn apply_acoustic(&mut self, field: ArrayViewMut3<f64>, ...);
}
```

But CPML is a **solver component** that:
- Modifies the wave equation itself within boundary regions
- Operates on field **derivatives**, not fields
- Maintains internal memory variables
- Must be integrated into the time-stepping loop

### Solution
Remove CPML from the Boundary trait hierarchy entirely:

```rust
// REMOVE any impl Boundary for CPMLBoundary

// In solver:
pub struct FdtdSolver {
    // NOT: boundary: Box<dyn Boundary>
    // BUT: 
    cpml: Option<CPMLBoundary>,  // CPML is a solver component
    simple_boundary: Option<Box<dyn Boundary>>,  // For actual simple boundaries
}
```

## Issue 2: Fragile Initialization with dt/sound_speed ❌

### Problem
Current constructor requires user to pass `dt` and `sound_speed`:
```rust
CPMLBoundary::new(config, grid, dt, sound_speed)  // FRAGILE!
```

If solver later uses different values → reflections!

### Solution
Two-phase initialization:

```rust
impl CPMLBoundary {
    // Phase 1: Basic setup (user calls this)
    pub fn new(config: CPMLConfig, grid: &Grid) -> KwaversResult<Self> {
        // Only validate config and allocate arrays
        // DO NOT compute profiles yet
    }
    
    // Phase 2: Solver finalizes (solver calls this)
    pub fn finalize(&mut self, dt: f64, sound_speed: f64) -> KwaversResult<()> {
        self.dt = dt;
        self.compute_profiles(sound_speed)?;
        Ok(())
    }
}
```

## Issue 3: 12-Parameter Static Function ❌

### Problem
```rust
fn compute_profile_for_dimension(
    n: usize, thickness: f64, m: f64, sigma_max: f64,
    dx: f64, config: &CPMLConfig, dt: f64,
    sigma: &mut [f64], kappa: &mut [f64],
    inv_kappa: &mut [f64], alpha: &mut [f64],
    b: &mut [f64], c: &mut [f64],
) { /* ... */ }
```

This is a code smell indicating poor encapsulation.

### Solution
Convert to instance method:

```rust
impl CPMLBoundary {
    fn compute_profile_for_dimension(&mut self, n: usize, dx: f64, dim: usize) {
        let (sigma, kappa, inv_kappa, alpha, b, c) = match dim {
            0 => (&mut self.sigma_x, &mut self.kappa_x, ...),
            1 => (&mut self.sigma_y, &mut self.kappa_y, ...),
            2 => (&mut self.sigma_z, &mut self.kappa_z, ...),
            _ => unreachable!(),
        };
        
        // Access self.config, self.dt directly
        // No need to pass 12 parameters!
    }
}
```

## Issue 4: Duplicated Code in Memory Updates ❌

### Problem
Both `update_acoustic_memory` and `apply_cpml_gradient` have identical logic repeated 3 times:

```rust
match component {
    0 => { /* nearly identical code */ }
    1 => { /* nearly identical code */ }  
    2 => { /* nearly identical code */ }
}
```

### Solution
Unify the logic:

```rust
pub fn update_acoustic_memory(&mut self, pressure_grad: &Array3<f64>, component: usize) {
    let mut psi = self.psi_acoustic.index_axis_mut(Axis(0), component);
    
    // Select profiles ONCE
    let (b_profile, c_profile) = match component {
        0 => (&self.b_x, &self.c_x),
        1 => (&self.b_y, &self.c_y),
        2 => (&self.b_z, &self.c_z),
        _ => unreachable!(),
    };
    
    // Single unified loop
    Zip::indexed(&mut psi)
        .and(pressure_grad)
        .for_each(|(i, j, k), psi_val, &grad| {
            let idx = match component {
                0 => i,
                1 => j,
                2 => k,
                _ => unreachable!(),
            };
            *psi_val = b_profile[idx] * *psi_val + c_profile[idx] * grad;
        });
}
```

## Physics Validation

These changes are **physics-preserving**:
1. CPML equations remain unchanged
2. Memory variable updates are identical
3. Profile computations are the same

But the architecture now:
- Correctly represents CPML as a solver component
- Ensures dt/sound_speed consistency
- Eliminates code duplication
- Improves maintainability

## Impact Assessment

- **Correctness**: ✅ Improves by ensuring consistent parameters
- **Performance**: ✅ Neutral (same computations, better cache locality)
- **Maintainability**: ✅ Significantly improved
- **API**: ⚠️ Breaking change (but necessary)

## Implementation Priority

1. **IMMEDIATE**: Fix dt/sound_speed initialization (prevents physics errors)
2. **HIGH**: Decouple from Boundary trait (architectural correctness)
3. **MEDIUM**: Refactor static method (code quality)
4. **MEDIUM**: Remove duplication (maintainability)

This is not optional refactoring - these are critical fixes for production readiness.