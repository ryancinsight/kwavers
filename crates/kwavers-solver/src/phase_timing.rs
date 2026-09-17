//! Wall-clock attribution for the ignored per-solver phase-split probes.
//!
//! A probe times a solver step against the phases it contains. The quantities
//! of interest are differences (an update minus the sweeps it contains), so
//! arms that are subtracted must be timed alternately inside one loop: timing
//! them in separate loops attributes any drift in the host load to whichever
//! arm ran while it drifted, which on a host carrying peer builds took one
//! such difference below zero.

use std::time::{Duration, Instant};

/// One arm's mean and fastest repeat, in microseconds.
///
/// The mean is what adds across phases; the fastest repeat is what survives a
/// busy host. A peer build inflates a mean by whatever share of the loop it
/// stole, but it cannot make any single repeat faster, so comparing two
/// revisions by their fastest repeats reads the code where the means read the
/// machine.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Phase {
    pub(crate) mean: f64,
    pub(crate) fastest: f64,
}

/// How many times each arm runs before and while it is timed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PhaseTimer {
    /// Timed repeats per arm.
    pub(crate) repeats: usize,
    /// Untimed repeats first, so plans, caches and task pools are warm.
    pub(crate) warm: usize,
}

impl PhaseTimer {
    /// Time `phase` alone.
    pub(crate) fn single<S>(self, state: &mut S, mut phase: impl FnMut(&mut S)) -> Phase {
        for _ in 0..self.warm {
            phase(state);
        }
        let mut arm = Arm::default();
        for _ in 0..self.repeats {
            arm.run(state, &mut phase);
        }
        arm.phase(self.repeats)
    }

    /// Time `first` and `second` alternately inside one loop.
    pub(crate) fn pair<S>(
        self,
        state: &mut S,
        mut first: impl FnMut(&mut S),
        mut second: impl FnMut(&mut S),
    ) -> (Phase, Phase) {
        for _ in 0..self.warm {
            first(state);
            second(state);
        }
        let (mut first_arm, mut second_arm) = (Arm::default(), Arm::default());
        for _ in 0..self.repeats {
            first_arm.run(state, &mut first);
            second_arm.run(state, &mut second);
        }
        (
            first_arm.phase(self.repeats),
            second_arm.phase(self.repeats),
        )
    }
}

/// Running total and fastest repeat of one arm.
struct Arm {
    total: Duration,
    fastest: Duration,
}

impl Default for Arm {
    fn default() -> Self {
        Self {
            total: Duration::ZERO,
            fastest: Duration::MAX,
        }
    }
}

impl Arm {
    fn run<S>(&mut self, state: &mut S, phase: &mut impl FnMut(&mut S)) {
        let start = Instant::now();
        phase(state);
        let elapsed = start.elapsed();
        self.total += elapsed;
        self.fastest = self.fastest.min(elapsed);
    }

    fn phase(&self, repeats: usize) -> Phase {
        let micros = |d: Duration| d.as_secs_f64() * 1.0e6;
        Phase {
            mean: micros(self.total) / repeats as f64,
            fastest: micros(self.fastest),
        }
    }
}
