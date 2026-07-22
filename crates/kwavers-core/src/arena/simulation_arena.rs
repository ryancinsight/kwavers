// Simulation arena — thread-local arena for concurrent field management.
//
// [`ThreadLocalArena`] wraps a [`FieldArena`] in `Rc<RefCell<_>>` so that
// multiple handles can share ownership within a single thread.
// [`ThreadLocalFieldGuard`] provides RAII slot release when the guard is
// dropped, making it safe to hold field borrows across function boundaries
// without lifetime parameters on the caller.

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use crate::error::{KwaversError, KwaversResult};

use super::field_arena::{ArenaConfig, ArenaStats, FieldArena};

// ─── ThreadLocalFieldGuard ────────────────────────────────────────────────────

/// RAII guard for a field slot allocated from a [`ThreadLocalArena`].
///
/// The slot is automatically marked free when this guard is dropped.
#[allow(missing_debug_implementations)]
pub struct ThreadLocalFieldGuard {
    /// Weak reference prevents the guard from extending arena lifetime.
    arena: Weak<RefCell<FieldArena>>,
    field_index: usize,
}

impl ThreadLocalFieldGuard {
    /// Obtain a mutable slice to the allocated field.
    ///
    /// Returns `None` if the arena has been freed (weak pointer dangled).
    #[allow(clippy::bind_instead_of_map)]
    pub fn field(&mut self) -> Option<&mut [f64]> {
        self.arena.upgrade().and_then(|arena| {
            let arena = arena.borrow_mut();
            let offset = self.field_index * arena.config.field_size * arena.config.element_size;

            // SAFETY: `field_index < max_fields` (enforced by allocation bitmap).
            // `offset + field_size * element_size ≤ layout.size()`.
            // `RefCell::borrow_mut` guarantees exclusive access.
            let field_ptr = unsafe { arena.memory.as_ptr().add(offset) as *mut f64 };
            Some(unsafe { std::slice::from_raw_parts_mut(field_ptr, arena.config.field_size) })
        })
    }
}

impl Drop for ThreadLocalFieldGuard {
    fn drop(&mut self) {
        if let Some(arena) = self.arena.upgrade() {
            let arena = arena.borrow_mut();
            let mut state = arena.allocation_state.borrow_mut();
            if self.field_index < state.allocated.len() {
                state.allocated[self.field_index] = false;
                state.allocated_count = state.allocated_count.saturating_sub(1);
            }
        }
    }
}

// ─── ThreadLocalArena ─────────────────────────────────────────────────────────

/// Thread-local arena that allows multiple shared handles within one thread.
///
/// Uses `Rc<RefCell<FieldArena>>` so that handles (e.g. guards) can outlive
/// the immediate scope while still receiving automatic cleanup.
#[derive(Clone)]
#[allow(missing_debug_implementations)]
pub struct ThreadLocalArena {
    arena: Rc<RefCell<FieldArena>>,
}

impl ThreadLocalArena {
    /// Create a thread-local arena from the given configuration.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn new(config: ArenaConfig) -> KwaversResult<Self> {
        let arena = Rc::new(RefCell::new(FieldArena::new(config)?));
        Ok(Self { arena })
    }

    /// Allocate a field slot and return an RAII guard.
    ///
    /// The slot is released automatically when the guard is dropped.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn allocate_field(&self) -> KwaversResult<ThreadLocalFieldGuard> {
        let field_index = {
            let arena = self.arena.borrow_mut();
            let mut state = arena.allocation_state.borrow_mut();

            let slot = state
                .allocated
                .iter()
                .position(|&in_use| !in_use)
                .ok_or_else(|| {
                    KwaversError::System(crate::error::SystemError::ResourceUnavailable {
                        resource: "arena field slot".to_owned(),
                    })
                })?;

            state.allocated[slot] = true;
            state.allocated_count += 1;
            slot
        };

        Ok(ThreadLocalFieldGuard {
            arena: Rc::downgrade(&self.arena),
            field_index,
        })
    }

    /// Snapshot of current allocation statistics.
    #[must_use]
    pub fn stats(&self) -> ArenaStats {
        self.arena.borrow().stats()
    }
}
