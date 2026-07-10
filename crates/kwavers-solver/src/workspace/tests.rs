use super::*;
use eunomia::Complex;
use kwavers_grid::Grid;
use leto::Array3;

/// Build an owned f-contiguous (column-major) `Array3`, the leto-native
/// analogue of ndarray's `from_shape_fn(shape.f(), …)`: logical element
/// `[i, j, k]` holds `f([i, j, k])`, but the physical layout is non-C-contiguous,
/// so `as_slice()` returns `None`, exercising the logical-iterator path.
fn from_shape_fn_fortran<F>(shape: [usize; 3], mut f: F) -> Array3<f64>
where
    F: FnMut([usize; 3]) -> f64,
{
    let layout = leto::Layout::f_contiguous(shape).expect("f-contiguous layout");
    let [d0, d1, d2] = shape;
    let mut data = vec![0.0_f64; d0 * d1 * d2];
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                data[i + j * d0 + k * d0 * d1] = f([i, j, k]);
            }
        }
    }
    leto::Array::new(layout, leto::VecStorage::new(data)).expect("valid f-contiguous array")
}

#[test]
fn test_workspace_creation() {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
    let workspace = SolverWorkspace::new(&grid);

    assert_eq!(workspace.fft_buffer.shape(), [64, 64, 64]);
    assert_eq!(workspace.real_buffer.shape(), [64, 64, 64]);
    assert!(workspace.validate_shape(&grid));
}

#[test]
fn test_workspace_pool() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let pool = WorkspacePool::new(grid, 2);

    assert_eq!(pool.size(), 2);

    let mut guard = pool.acquire().unwrap();
    assert_eq!(pool.size(), 1);

    guard.get_mut().real_buffer.fill(1.0);

    drop(guard);
    assert_eq!(pool.size(), 2);
}

#[test]
fn test_inplace_operations() {
    use inplace_ops::*;

    let mut a = Array3::from_elem((10, 10, 10), 1.0);
    let b = Array3::from_elem((10, 10, 10), 2.0);
    let c = Array3::from_elem((10, 10, 10), 3.0);

    add_inplace(&mut a, &b);
    assert!(a.iter().all(|&value| value == 3.0));

    sub_inplace(&mut a, &b);
    assert!(a.iter().all(|&value| value == 1.0));

    scale_inplace(&mut a, 2.0);
    assert!(a.iter().all(|&value| value == 2.0));

    apply_inplace(&mut a, |value| value + 4.0);
    assert!(a.iter().all(|&value| value == 6.0));

    // a = a * b + c = 6 * 2 + 3 = 15
    fma_inplace(&mut a, &b, &c);
    assert!(a.iter().all(|&value| value == 15.0));
}

#[test]
fn inplace_operations_preserve_logical_order_for_nonstandard_layouts() {
    use inplace_ops::*;

    let shape = (2, 3, 4);
    let mut add_target = Array3::from_shape_fn(shape, |[i, j, k]| (100 * i + 10 * j + k) as f64);
    let add_input =
        from_shape_fn_fortran([shape.0, shape.1, shape.2], |[i, j, k]| {
            (1000 + 100 * i + 10 * j + k) as f64
        });

    assert!(
        add_input.as_slice().is_none(),
        "test invariant: input must force logical-iterator fallback"
    );
    add_inplace(&mut add_target, &add_input);

    assert_eq!(
        add_target,
        Array3::from_shape_fn(shape, |[i, j, k]| {
            (100 * i + 10 * j + k) as f64 + (1000 + 100 * i + 10 * j + k) as f64
        })
    );

    let mut sub_target = Array3::from_shape_fn(shape, |[i, j, k]| (100 * i + 10 * j + k) as f64);
    sub_inplace(&mut sub_target, &add_input);
    assert_eq!(
        sub_target,
        Array3::from_shape_fn(shape, |[i, j, k]| {
            (100 * i + 10 * j + k) as f64 - (1000 + 100 * i + 10 * j + k) as f64
        })
    );

    let mut fma_target =
        Array3::from_shape_fn(shape, |[i, j, k]| 1.0 + (100 * i + 10 * j + k) as f64);
    let multiplier =
        from_shape_fn_fortran([shape.0, shape.1, shape.2], |[i, j, k]| 2.0 + (i + j + k) as f64);
    let addend = Array3::from_shape_fn(shape, |[i, j, k]| (i * j + k) as f64);
    fma_inplace(&mut fma_target, &multiplier, &addend);
    assert_eq!(
        fma_target,
        Array3::from_shape_fn(shape, |[i, j, k]| {
            (1.0 + (100 * i + 10 * j + k) as f64)
                .mul_add(2.0 + (i + j + k) as f64, (i * j + k) as f64)
        })
    );
}

#[test]
fn scratch_arena_memory_bytes_matches_memory_usage() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let ws = SolverWorkspace::new(&grid);
    assert_eq!(ws.memory_bytes(), ws.memory_usage());
    let n = 8 * 8 * 8;
    let expected = n * 16 + 3 * n * 8;
    assert_eq!(ws.memory_bytes(), expected);
}

#[test]
fn scratch_arena_clear_zeros_solver_workspace() {
    let grid = Grid::new(4, 4, 4, 1e-3, 1e-3, 1e-3).unwrap();
    let mut ws = SolverWorkspace::new(&grid);
    ws.fft_buffer.fill(Complex::new(1.0, 2.0));
    ws.real_buffer.fill(3.0);
    ws.k_space_buffer.fill(4.0);
    ws.temp_buffer.fill(5.0);

    ScratchArena::clear(&mut ws);

    assert!(
        ws.fft_buffer.iter().all(|c| c.re == 0.0 && c.im == 0.0),
        "fft_buffer not zeroed"
    );
    assert!(
        ws.real_buffer.iter().all(|&v| v == 0.0),
        "real_buffer not zeroed"
    );
    assert!(
        ws.k_space_buffer.iter().all(|&v| v == 0.0),
        "k_space_buffer not zeroed"
    );
    assert!(
        ws.temp_buffer.iter().all(|&v| v == 0.0),
        "temp_buffer not zeroed"
    );
}

#[test]
fn scratch_arena_memory_bytes_stable_after_clear() {
    let grid = Grid::new(6, 6, 6, 1e-3, 1e-3, 1e-3).unwrap();
    let mut ws = SolverWorkspace::new(&grid);
    let before = ws.memory_bytes();
    ws.real_buffer.fill(99.0);
    ScratchArena::clear(&mut ws);
    assert_eq!(ws.memory_bytes(), before);
}
