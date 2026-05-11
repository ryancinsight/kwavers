use super::*;
use crate::domain::grid::Grid;
use ndarray::Array3;
use num_complex::Complex;

#[test]
fn test_workspace_creation() {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
    let workspace = SolverWorkspace::new(&grid);

    assert_eq!(workspace.fft_buffer.shape(), &[64, 64, 64]);
    assert_eq!(workspace.real_buffer.shape(), &[64, 64, 64]);
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
    assert_eq!(a[[0, 0, 0]], 3.0);

    scale_inplace(&mut a, 2.0);
    assert_eq!(a[[0, 0, 0]], 6.0);

    // a = a * b + c = 6 * 2 + 3 = 15
    fma_inplace(&mut a, &b, &c);
    assert_eq!(a[[0, 0, 0]], 15.0);
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
    assert!(ws.real_buffer.iter().all(|&v| v == 0.0), "real_buffer not zeroed");
    assert!(
        ws.k_space_buffer.iter().all(|&v| v == 0.0),
        "k_space_buffer not zeroed"
    );
    assert!(ws.temp_buffer.iter().all(|&v| v == 0.0), "temp_buffer not zeroed");
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
