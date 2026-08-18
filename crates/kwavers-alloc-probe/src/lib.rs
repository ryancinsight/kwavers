//! Thread-scoped allocation counting for allocation-contract tests.
//!
//! A process-global counting allocator (the previous `stats_alloc` regions)
//! observes every thread in the process, so under full-suite load a
//! measurement window occasionally records harness or peer-test allocations
//! and an exact-count assertion flakes (ATLAS-KWAVERS-ALLOC-TEST-RACE-1).
//! This probe counts only on threads that opened a [`Window`], confining the
//! measurement to the test's own effects while forwarding every request to
//! the system allocator unchanged.
//!
//! Usage, in a test binary:
//!
//! ```ignore
//! #[global_allocator]
//! static GLOBAL: kwavers_alloc_probe::ThreadScopedAllocator =
//!     kwavers_alloc_probe::ThreadScopedAllocator;
//!
//! let window = kwavers_alloc_probe::Window::open();
//! let value = build_the_thing();
//! let change = window.change();
//! assert_eq!(change.allocations, 1);
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

// `missing_const_for_thread_local` is a false positive here: the initializer
// below is already a `const` block, but the lint matches a node of the
// `thread_local!` expansion instead. The diagnostic is host-triple dependent —
// it fires on `windows-gnu` and stays quiet on `windows-msvc` under the same
// Clippy build — so an unconditional suppression is wrong on one host or the
// other: unfulfilled on msvc, load-bearing on gnu. Gating on the host that
// actually emits it keeps `expect` fulfilled where it applies and absent where
// it does not, so the ratchet still expires when upstream fixes the match.
thread_local! {
    /// Number of open measurement windows on the current thread.
    ///
    /// A depth counter rather than a flag so nested windows compose; it is
    /// const-initialized so reading it inside the allocator never allocates.
    #[cfg_attr(
        all(windows, target_env = "gnu"),
        expect(
            clippy::missing_const_for_thread_local,
            reason = "the initializer is already a const block; the lint matches the thread_local! expansion, and does so only on the windows-gnu host"
        )
    )]
    static OPEN_WINDOWS: Cell<u32> = const { Cell::new(0) };
}

/// Whether the current thread has any open measurement window.
fn measuring() -> bool {
    OPEN_WINDOWS.with(Cell::get) > 0
}

/// Allocation events observed on measuring threads.
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
/// Reallocation events observed on measuring threads.
static REALLOCATIONS: AtomicU64 = AtomicU64::new(0);

/// A [`System`]-forwarding allocator that counts only on threads with an
/// open [`Window`].
///
/// Install as the test binary's `#[global_allocator]`.
pub struct ThreadScopedAllocator;

// SAFETY: every method forwards verbatim to `System`, which upholds the
// `GlobalAlloc` contract; the counting side effect touches only atomics and
// a const-initialized thread-local, neither of which allocates or panics.
unsafe impl GlobalAlloc for ThreadScopedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if measuring() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller upholds `alloc`'s contract; forwarded verbatim.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller upholds `dealloc`'s contract; forwarded verbatim.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if measuring() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller upholds `alloc_zeroed`'s contract; forwarded
        // verbatim.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if measuring() {
            REALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller upholds `realloc`'s contract; forwarded verbatim.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// Counter deltas over one measurement window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Change {
    /// `alloc`/`alloc_zeroed` calls on the measuring thread.
    pub allocations: u64,
    /// `realloc` calls on the measuring thread.
    pub reallocations: u64,
}

/// An open measurement window on the current thread.
///
/// Only allocator traffic from threads holding an open window is counted,
/// so one window per test thread yields a measurement confined to that
/// test's effects. Closing (dropping) the window stops counting.
pub struct Window {
    start_allocations: u64,
    start_reallocations: u64,
}

impl Window {
    /// Open a window on the current thread and begin counting.
    #[must_use]
    pub fn open() -> Self {
        OPEN_WINDOWS.with(|depth| depth.set(depth.get() + 1));
        Self {
            start_allocations: ALLOCATIONS.load(Ordering::Relaxed),
            start_reallocations: REALLOCATIONS.load(Ordering::Relaxed),
        }
    }

    /// Counter deltas since the window opened.
    #[must_use]
    pub fn change(&self) -> Change {
        Change {
            allocations: ALLOCATIONS.load(Ordering::Relaxed) - self.start_allocations,
            reallocations: REALLOCATIONS.load(Ordering::Relaxed) - self.start_reallocations,
        }
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        OPEN_WINDOWS.with(|depth| depth.set(depth.get() - 1));
    }
}
