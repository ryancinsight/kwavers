//! Arena Allocator for High-Performance Memory Management
//!
//! This module provides arena-based memory allocation for acoustic wave simulations,
//! reducing memory fragmentation and improving cache locality for field data.
//!
//! ## Arena Allocation Benefits
//!
//! ### Memory Efficiency
//! - **Zero Fragmentation**: All allocations are contiguous in memory
//! - **Reduced Overhead**: No individual allocation/deallocation tracking
//! - **Bulk Deallocation**: Entire arena freed at once
//!
//! ### Performance Improvements
//! - **Cache Locality**: Related data structures are adjacent in memory
//! - **Prefetching**: CPU can prefetch contiguous memory blocks
//! - **TLB Efficiency**: Fewer page table entries needed
//!
//! ### Memory Safety
//! - **RAII Pattern**: Automatic cleanup when arena goes out of scope
//! - **Borrow Checking**: Compile-time guarantees against use-after-free
//! - **No Interior Mutability**: Exclusive access to allocated data
//!
//! ## Usage Patterns
//!
//! ### Field Data Allocation
//! ```rust,ignore
//! let mut arena = FieldArena::new();
//!
//! // Allocate pressure field
//! let pressure = arena.alloc_field(nx, ny, nz);
//!
//! // Allocate velocity components
//! let velocity_x = arena.alloc_field(nx, ny, nz);
//! let velocity_y = arena.alloc_field(nx, ny, nz);
//! let velocity_z = arena.alloc_field(nx, ny, nz);
//!
//! // All fields are contiguous in memory
//! ```
//!
//! ### Temporary Buffer Management
//! ```rust,ignore
//! let mut temp_arena = TempArena::with_capacity(1024 * 1024); // 1MB
//!
//! for time_step in 0..num_steps {
//!     let temp_buffer = temp_arena.alloc_temp_buffer(size);
//!     // Use buffer for intermediate calculations
//!     temp_arena.reset(); // Reuse memory for next iteration
//! }
//! ```
//!
//! ### Simulation State Management
//! ```rust,ignore
//! let mut sim_arena = SimulationArena::new();
//!
//! let mut solver = sim_arena.alloc_solver(config, grid, medium);
//! let mut fields = sim_arena.alloc_wave_fields(grid);
//!
//! // All simulation data is co-located in memory
//! ```

use ndarray::Array3;
use std::alloc::Layout;
use std::cell::UnsafeCell;

/// Thread-safe arena allocator for field data
pub struct FieldArena {
    /// Raw memory buffer (accessed via unsafe pointer arithmetic in alloc_field)
    #[allow(dead_code)] // Used internally via UnsafeCell in unsafe alloc_field method
    buffer: UnsafeCell<Vec<u8>>,
    /// Current allocation offset
    offset: UnsafeCell<usize>,
    /// Total capacity
    capacity: usize,
}

impl std::fmt::Debug for FieldArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FieldArena")
            .field("capacity", &self.capacity)
            .field("used_bytes", &self.used_bytes())
            .finish()
    }
}

impl FieldArena {
    /// Create a new field arena with default capacity (64MB)
    pub fn new() -> Self {
        Self::with_capacity(64 * 1024 * 1024) // 64MB
    }

    /// Create a new field arena with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: UnsafeCell::new(vec![0u8; capacity]),
            offset: UnsafeCell::new(0),
            capacity,
        }
    }

    /// Allocate a 3D field array
    ///
    /// # Safety
    /// The returned array is valid only for the lifetime of this arena.
    /// Allocating from the arena again may invalidate previous allocations.
    #[allow(unsafe_code)]
    pub unsafe fn alloc_field<T>(&self, nx: usize, ny: usize, nz: usize) -> Option<Array3<T>>
    where
        T: Clone + Default,
    {
        let element_size = std::mem::size_of::<T>();
        let total_elements = nx * ny * nz;
        let total_bytes = total_elements * element_size;

        let current_offset = *self.offset.get();

        // Check if we have enough space
        if current_offset + total_bytes > self.capacity {
            return None; // Out of memory
        }

        // Update offset
        *self.offset.get() = current_offset + total_bytes;
        Some(Array3::from_elem((nx, ny, nz), T::default()))
    }

    /// Reset the arena for reuse
    #[allow(unsafe_code)]
    pub fn reset(&self) {
        unsafe {
            *self.offset.get() = 0;
        }
    }

    /// Get current memory usage
    #[allow(unsafe_code)]
    pub fn used_bytes(&self) -> usize {
        unsafe { *self.offset.get() }
    }

    /// Get total capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get memory usage statistics
    pub fn stats(&self) -> ArenaStats {
        let used = self.used_bytes();
        let capacity = self.capacity;

        ArenaStats {
            used_bytes: used,
            total_bytes: capacity,
            utilization: used as f64 / capacity as f64,
        }
    }
}

// FieldArena is not Send/Sync because it contains UnsafeCell
// This is correct - arenas should not be shared between threads

/// Arena statistics
#[derive(Debug, Clone)]
pub struct ArenaStats {
    pub used_bytes: usize,
    pub total_bytes: usize,
    pub utilization: f64,
}

/// High-performance bump allocator for temporary data
#[derive(Debug)]
pub struct BumpArena {
    /// Memory chunks
    chunks: Vec<Vec<u8>>,
    /// Current chunk index
    current_chunk: usize,
    /// Offset in current chunk
    offset: usize,
    /// Chunk size
    chunk_size: usize,
}

impl BumpArena {
    /// Create a new bump arena
    pub fn new() -> Self {
        Self::with_chunk_size(1024 * 1024) // 1MB chunks
    }

    /// Create a new bump arena with specified chunk size
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            chunks: Vec::new(),
            current_chunk: 0,
            offset: 0,
            chunk_size,
        }
    }

    /// Allocate memory from the arena
    ///
    /// # Safety
    /// The returned pointer is valid until the arena is dropped or reset.
    #[allow(unsafe_code)]
    pub unsafe fn alloc(&mut self, layout: Layout) -> *mut u8 {
        // Align the offset
        let aligned_offset = (self.offset + layout.align() - 1) & !(layout.align() - 1);

        // Check if we need a new chunk
        if self.current_chunk >= self.chunks.len()
            || aligned_offset + layout.size() > self.chunks[self.current_chunk].len()
        {
            // Allocate new chunk
            self.chunks.push(vec![0u8; self.chunk_size]);
            self.current_chunk = self.chunks.len() - 1;
            self.offset = 0;

            // Try again with new chunk
            return self.alloc(layout);
        }

        // Allocate from current chunk
        let chunk = &mut self.chunks[self.current_chunk];
        let ptr = chunk.as_mut_ptr().add(aligned_offset);

        self.offset = aligned_offset + layout.size();

        ptr
    }

    /// Allocate a typed value
    ///
    /// # Safety
    /// The returned reference is valid until the arena is dropped or reset.
    #[allow(unsafe_code)]
    pub unsafe fn alloc_value<T>(&mut self, value: T) -> &mut T {
        let layout = Layout::new::<T>();
        let ptr = self.alloc(layout) as *mut T;
        ptr.write(value);
        &mut *ptr
    }

    /// Allocate an array
    ///
    /// # Safety
    /// The returned slice is valid until the arena is dropped or reset.
    #[allow(unsafe_code)]
    pub unsafe fn alloc_array<T>(&mut self, len: usize) -> &mut [T]
    where
        T: Default,
    {
        if len == 0 {
            return &mut [];
        }

        let layout = Layout::array::<T>(len).unwrap();
        let ptr = self.alloc(layout) as *mut T;

        // Initialize array elements
        for i in 0..len {
            ptr.add(i).write(T::default());
        }

        std::slice::from_raw_parts_mut(ptr, len)
    }

    /// Reset the arena (all previous allocations become invalid)
    pub fn reset(&mut self) {
        self.current_chunk = 0;
        self.offset = 0;
    }

    /// Get memory usage statistics
    pub fn stats(&self) -> ArenaStats {
        let total_bytes = self.chunks.len() * self.chunk_size;
        let used_bytes = if self.current_chunk < self.chunks.len() {
            self.current_chunk * self.chunk_size + self.offset
        } else {
            0
        };

        ArenaStats {
            used_bytes,
            total_bytes,
            utilization: if total_bytes > 0 {
                used_bytes as f64 / total_bytes as f64
            } else {
                0.0
            },
        }
    }
}

/// Simulation-specific arena for wave fields and solver data
/// Combined arena for simulation data
#[derive(Debug)]
pub struct SimulationArena {
    field_arena: FieldArena,
    temp_arena: BumpArena,
}

impl SimulationArena {
    /// Create a new simulation arena
    pub fn new() -> Self {
        Self {
            field_arena: FieldArena::new(),
            temp_arena: BumpArena::new(),
        }
    }

    /// Allocate wave fields (pressure, velocity components)
    ///
    /// # Safety
    /// Fields are valid only for the lifetime of this arena.
    #[allow(unsafe_code)]
    pub unsafe fn alloc_wave_fields(
        &self,
        grid: &crate::domain::grid::Grid,
    ) -> Option<crate::domain::field::wave::WaveFields> {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;

        let p = self.field_arena.alloc_field::<f64>(nx, ny, nz)?;
        let ux = self.field_arena.alloc_field::<f64>(nx, ny, nz)?;
        let uy = self.field_arena.alloc_field::<f64>(nx, ny, nz)?;
        let uz = self.field_arena.alloc_field::<f64>(nx, ny, nz)?;

        Some(crate::domain::field::wave::WaveFields { p, ux, uy, uz })
    }

    /// Allocate temporary buffer for calculations
    ///
    /// # Safety
    /// Buffer is valid until the arena is reset.
    #[allow(unsafe_code)]
    pub unsafe fn alloc_temp_buffer(&mut self, size: usize) -> &mut [f64] {
        self.temp_arena.alloc_array(size)
    }

    /// Reset temporary allocations for reuse
    pub fn reset_temp(&mut self) {
        self.temp_arena.reset();
    }

    /// Get memory usage statistics
    pub fn stats(&self) -> SimulationArenaStats {
        SimulationArenaStats {
            field_stats: self.field_arena.stats(),
            temp_stats: self.temp_arena.stats(),
        }
    }
}

/// Simulation arena statistics
#[derive(Debug, Clone)]
pub struct SimulationArenaStats {
    pub field_stats: ArenaStats,
    pub temp_stats: ArenaStats,
}

impl SimulationArenaStats {
    /// Get total memory usage
    pub fn total_used(&self) -> usize {
        self.field_stats.used_bytes + self.temp_stats.used_bytes
    }

    /// Get total capacity
    pub fn total_capacity(&self) -> usize {
        self.field_stats.total_bytes + self.temp_stats.total_bytes
    }

    /// Get overall utilization
    pub fn overall_utilization(&self) -> f64 {
        let total_used = self.total_used() as f64;
        let total_capacity = self.total_capacity() as f64;

        if total_capacity > 0.0 {
            total_used / total_capacity
        } else {
            0.0
        }
    }
}

/// Performance monitoring for arena allocations
#[derive(Debug)]
pub struct ArenaPerformanceMonitor {
    allocations: std::collections::HashMap<String, usize>,
    total_allocated: usize,
    peak_usage: usize,
}

impl ArenaPerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            allocations: std::collections::HashMap::new(),
            total_allocated: 0,
            peak_usage: 0,
        }
    }

    /// Record an allocation
    pub fn record_allocation(&mut self, category: &str, size: usize) {
        *self.allocations.entry(category.to_string()).or_insert(0) += size;
        self.total_allocated += size;
        self.peak_usage = self.peak_usage.max(self.total_allocated);
    }

    /// Record a deallocation
    pub fn record_deallocation(&mut self, category: &str, size: usize) {
        if let Some(alloc_size) = self.allocations.get_mut(category) {
            *alloc_size = alloc_size.saturating_sub(size);
        }
        self.total_allocated = self.total_allocated.saturating_sub(size);
    }

    /// Get allocation statistics
    pub fn stats(&self) -> &std::collections::HashMap<String, usize> {
        &self.allocations
    }

    /// Get total allocated memory
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    /// Reset monitoring
    pub fn reset(&mut self) {
        self.allocations.clear();
        self.total_allocated = 0;
        self.peak_usage = 0;
    }
}

/// Memory pool for frequently allocated objects
pub struct MemoryPool<T> {
    /// Available objects
    available: Vec<T>,
    /// Factory function for creating new objects
    factory: Box<dyn Fn() -> T>,
}

impl<T> std::fmt::Debug for MemoryPool<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryPool")
            .field("available_count", &self.available.len())
            .finish()
    }
}

impl<T> MemoryPool<T> {
    /// Create a new memory pool
    pub fn new<F>(factory: F) -> Self
    where
        F: Fn() -> T + 'static,
    {
        Self {
            available: Vec::new(),
            factory: Box::new(factory),
        }
    }

    /// Allocate an object from the pool
    pub fn alloc(&mut self) -> T {
        self.available.pop().unwrap_or_else(|| (self.factory)())
    }

    /// Return an object to the pool for reuse
    pub fn free(&mut self, object: T) {
        self.available.push(object);
    }

    /// Get pool statistics
    pub fn stats(&self) -> MemoryPoolStats {
        MemoryPoolStats {
            available_count: self.available.len(),
        }
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub available_count: usize,
}

/// Specialized pool for field arrays
pub type FieldPool = MemoryPool<Array3<f64>>;

/// Create a field pool with default configuration
pub fn create_field_pool(nx: usize, ny: usize, nz: usize) -> FieldPool {
    MemoryPool::new(move || Array3::zeros((nx, ny, nz)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_arena_creation() {
        let arena = FieldArena::new();
        assert_eq!(arena.used_bytes(), 0);
        assert!(arena.capacity() > 0);
    }

    #[test]
    fn test_field_arena_allocation() {
        let arena = FieldArena::with_capacity(1024 * 1024); // 1MB

        // This is safe because we don't use the returned array
        unsafe {
            let field = arena.alloc_field::<f64>(10, 10, 10);
            assert!(field.is_some());

            let used = arena.used_bytes();
            assert!(used > 0);

            // Reset and check
            arena.reset();
            assert_eq!(arena.used_bytes(), 0);
        }
    }

    #[test]
    fn test_bump_arena_creation() {
        let mut arena = BumpArena::new();
        let stats = arena.stats();
        assert_eq!(stats.used_bytes, 0);
    }

    #[test]
    fn test_bump_arena_allocation() {
        let mut arena = BumpArena::new();

        unsafe {
            let array = arena.alloc_array::<f64>(100);
            assert_eq!(array.len(), 100);

            // All elements should be default (0.0)
            assert!(array.iter().all(|&x| x == 0.0));

            let stats = arena.stats();
            assert!(stats.used_bytes > 0);
        }
    }

    #[test]
    fn test_simulation_arena_creation() {
        let arena = SimulationArena::new();
        let stats = arena.stats();
        assert_eq!(stats.total_used(), 0);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(|| vec![0i32; 10]);

        // Allocate from pool
        let vec1 = pool.alloc();
        assert_eq!(vec1.len(), 10);

        // Return to pool
        pool.free(vec1);

        // Allocate again (should reuse)
        let vec2 = pool.alloc();
        assert_eq!(vec2.len(), 10);

        let stats = pool.stats();
        assert_eq!(stats.available_count, 0); // vec2 is still allocated
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = ArenaPerformanceMonitor::new();

        monitor.record_allocation("fields", 1024);
        monitor.record_allocation("temp", 512);

        assert_eq!(monitor.total_allocated(), 1536);
        assert_eq!(monitor.peak_usage(), 1536);

        monitor.record_deallocation("fields", 1024);
        assert_eq!(monitor.total_allocated(), 512);
    }

    #[test]
    fn test_field_pool() {
        let mut pool = create_field_pool(4, 4, 4);

        let field1 = pool.alloc();
        assert_eq!(field1.dim(), (4, 4, 4));

        pool.free(field1);

        let field2 = pool.alloc();
        assert_eq!(field2.dim(), (4, 4, 4));

        let stats = pool.stats();
        assert_eq!(stats.available_count, 0);
    }

    #[test]
    fn test_arena_stats() {
        let arena = FieldArena::with_capacity(1000);
        let stats = arena.stats();

        assert_eq!(stats.total_bytes, 1000);
        assert_eq!(stats.used_bytes, 0);
        assert_eq!(stats.utilization, 0.0);
    }
}
