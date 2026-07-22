//! Sparse exact checkpoints for Westervelt adjoint replay.
//!
//! A second-order Westervelt update with the nonlinear term evaluated from
//! `p`N``, `p[n-1]`, and `p[n-2]` is restarted exactly when a checkpoint stores
//! those three states. Replaying one interval from that state produces the same
//! pressure samples as a dense forward history, up to deterministic floating
//! point operation order.

#[derive(Clone, Debug)]
pub(super) struct ForwardHistory {
    cells: usize,
    steps: usize,
    interval: usize,
    checkpoints: Vec<ForwardCheckpoint>,
}

#[derive(Clone, Debug)]
pub(super) struct ForwardCheckpoint {
    pub(super) state_step: usize,
    pub(super) older: Vec<f64>,
    pub(super) previous: Vec<f64>,
    pub(super) current: Vec<f64>,
}

#[derive(Clone, Debug)]
pub(super) struct HistorySegment {
    pub(super) start_step: usize,
    pub(super) end_step: usize,
    pub(super) older_at_start: Vec<f64>,
    pub(super) previous_at_start: Vec<f64>,
    states: Vec<f64>,
    cells: usize,
}

#[derive(Clone, Debug)]
pub(super) struct HistoryReplayWorkspace {
    pub(super) older: Vec<f64>,
    pub(super) previous: Vec<f64>,
    pub(super) current: Vec<f64>,
    pub(super) next: Vec<f64>,
    pub(super) segment: HistorySegment,
}

impl ForwardHistory {
    #[must_use]
    pub(super) fn new(
        cells: usize,
        steps: usize,
        interval: usize,
        older: &[f64],
        previous: &[f64],
        current: &[f64],
    ) -> Self {
        let interval = interval.max(1);
        let mut history = Self {
            cells,
            steps,
            interval,
            checkpoints: Vec::with_capacity(steps / interval + 1),
        };
        history.store_checkpoint(0, older, previous, current);
        history
    }

    pub(super) fn store_if_boundary(
        &mut self,
        state_step: usize,
        older: &[f64],
        previous: &[f64],
        current: &[f64],
    ) {
        if state_step < self.steps && state_step.is_multiple_of(self.interval) {
            self.store_checkpoint(state_step, older, previous, current);
        }
    }

    #[must_use]
    pub(super) fn cells(&self) -> usize {
        self.cells
    }

    #[must_use]
    pub(super) fn steps(&self) -> usize {
        self.steps
    }

    #[must_use]
    pub(super) fn interval(&self) -> usize {
        self.interval
    }

    #[must_use]
    #[cfg(test)]
    pub(super) fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    #[must_use]
    pub(super) fn segment_bounds_for_step(&self, step: usize) -> (usize, usize) {
        debug_assert!(step < self.steps);
        let start = (step / self.interval) * self.interval;
        let end = (start + self.interval).min(self.steps);
        (start, end)
    }

    #[must_use]
    pub(super) fn checkpoint_at(&self, state_step: usize) -> &ForwardCheckpoint {
        let index = state_step / self.interval;
        let checkpoint = &self.checkpoints[index];
        debug_assert_eq!(checkpoint.state_step, state_step);
        checkpoint
    }

    fn store_checkpoint(
        &mut self,
        state_step: usize,
        older: &[f64],
        previous: &[f64],
        current: &[f64],
    ) {
        debug_assert_eq!(older.len(), self.cells);
        debug_assert_eq!(previous.len(), self.cells);
        debug_assert_eq!(current.len(), self.cells);
        self.checkpoints.push(ForwardCheckpoint {
            state_step,
            older: older.to_vec(),
            previous: previous.to_vec(),
            current: current.to_vec(),
        });
    }
}

impl HistorySegment {
    #[must_use]
    pub(super) fn with_capacity(cells: usize, interval: usize) -> Self {
        Self {
            start_step: 0,
            end_step: 0,
            older_at_start: vec![0.0; cells],
            previous_at_start: vec![0.0; cells],
            states: vec![0.0; (interval + 1) * cells],
            cells,
        }
    }

    pub(super) fn reset(
        &mut self,
        start_step: usize,
        end_step: usize,
        older_at_start: &[f64],
        previous_at_start: &[f64],
        current_at_start: &[f64],
    ) {
        debug_assert!(end_step >= start_step);
        debug_assert_eq!(older_at_start.len(), self.cells);
        debug_assert_eq!(previous_at_start.len(), self.cells);
        debug_assert_eq!(current_at_start.len(), self.cells);
        let needed = (end_step - start_step + 1) * self.cells;
        if self.states.len() < needed {
            self.states.resize(needed, 0.0);
        }
        self.start_step = start_step;
        self.end_step = end_step;
        self.older_at_start.copy_from_slice(older_at_start);
        self.previous_at_start.copy_from_slice(previous_at_start);
        self.states[..self.cells].copy_from_slice(current_at_start);
    }

    pub(super) fn set_state(&mut self, state_step: usize, state: &[f64]) {
        debug_assert!(state_step >= self.start_step && state_step <= self.end_step);
        debug_assert_eq!(state.len(), self.cells);
        let offset = (state_step - self.start_step) * self.cells;
        self.states[offset..offset + self.cells].copy_from_slice(state);
    }

    #[must_use]
    pub(super) fn state(&self, state_step: usize) -> &[f64] {
        debug_assert!(state_step >= self.start_step && state_step <= self.end_step);
        let offset = (state_step - self.start_step) * self.cells;
        &self.states[offset..offset + self.cells]
    }

    #[must_use]
    pub(super) fn previous_for_step(&self, step: usize) -> &[f64] {
        if step == self.start_step {
            &self.previous_at_start
        } else {
            self.state(step - 1)
        }
    }
}

impl HistoryReplayWorkspace {
    #[must_use]
    pub(super) fn new(cells: usize, interval: usize) -> Self {
        Self {
            older: vec![0.0; cells],
            previous: vec![0.0; cells],
            current: vec![0.0; cells],
            next: vec![0.0; cells],
            segment: HistorySegment::with_capacity(cells, interval),
        }
    }

    pub(super) fn reset_from_checkpoint(
        &mut self,
        start_step: usize,
        end_step: usize,
        checkpoint: &ForwardCheckpoint,
    ) {
        self.older.copy_from_slice(&checkpoint.older);
        self.previous.copy_from_slice(&checkpoint.previous);
        self.current.copy_from_slice(&checkpoint.current);
        self.segment.reset(
            start_step,
            end_step,
            &checkpoint.older,
            &checkpoint.previous,
            &checkpoint.current,
        );
    }

    #[must_use]
    pub(super) fn segment(&self) -> &HistorySegment {
        &self.segment
    }
}
