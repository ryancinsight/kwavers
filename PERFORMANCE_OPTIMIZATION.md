# Performance Optimization: Analytical Tests

## Issue Identified
In `src/physics/analytical_tests.rs`, there were two instances where `to_owned()` was being called inside loops, creating unnecessary memory allocations on every iteration.

## Problem Code

### Instance 1 (Line 222)
```rust
for step in 0..num_steps {
    let t = step as f64 * dt;
    let pressure_view = fields.index_axis(Axis(0), 0).to_owned(); // INEFFICIENT!
    solver.update_wave(&mut fields, &pressure_view, &source, &grid, &medium, dt, t);
}
```

### Instance 2 (Line 303)
```rust
for step in 0..num_steps {
    let t = step as f64 * dt;
    let pressure_view = fields.index_axis(Axis(0), 0).to_owned(); // INEFFICIENT!
    solver.update_wave(&mut fields, &pressure_view, &source, &grid, &medium, dt, t);
}
```

## Solution Applied

Allocated `pressure_view` once before the loop and used `.assign()` to update it in each iteration:

### Optimized Code
```rust
let mut pressure_view = fields.index_axis(Axis(0), 0).to_owned();

for step in 0..num_steps {
    let t = step as f64 * dt;
    pressure_view.assign(&fields.index_axis(Axis(0), 0)); // EFFICIENT!
    solver.update_wave(&mut fields, &pressure_view, &source, &grid, &medium, dt, t);
}
```

## Performance Impact

### Before
- **Memory Allocations**: `num_steps` allocations (100-1000+ per test)
- **Memory Traffic**: High due to repeated allocations and deallocations
- **Cache Performance**: Poor due to constant memory churn

### After
- **Memory Allocations**: 1 allocation per test
- **Memory Traffic**: Minimal, only updating existing memory
- **Cache Performance**: Better locality of reference

## Benefits

1. **Reduced Memory Pressure**: Eliminates hundreds of unnecessary allocations
2. **Better Cache Utilization**: Reuses the same memory location
3. **Faster Test Execution**: Less time spent in memory management
4. **Follows DRY Principle**: Don't repeat allocations unnecessarily

## Design Principles Applied

- **DRY (Don't Repeat Yourself)**: Avoid repeated allocations
- **KISS (Keep It Simple)**: Simple optimization with significant impact
- **Performance-Conscious Design**: Efficient memory usage in tight loops
- **Clean Code**: More explicit about memory lifecycle