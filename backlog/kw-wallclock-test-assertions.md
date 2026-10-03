<a id="kw-wallclock-test-assertions"></a>

## KW-WALLCLOCK-TEST-ASSERTIONS — Three test files still assert on elapsed time and sleep to synchronize [patch] — todo

priority: verification; needs: none; scope: `crates/kwavers/tests/production_benchmarks.rs`, `crates/kwavers/tests/quick_comparative_test.rs`, `crates/kwavers/tests/stream_visualization_test.rs`, `crates/kwavers/benches/`

- **Finding (lines at origin/main a23f2f86cb0).** Under nextest parallelism and host load, elapsed time is nondeterministic, so each assertion below passes or fails on load, not on a defect. `production_benchmarks.rs:18-21` asserts the whole run takes under 30 s and `:32` each result under 10 s. `quick_comparative_test.rs:267-275` asserts FDTD and PSTD each take under 5000 ms, and `:113-119` asserts `execution_time.as_millis() > 0`, which fails on a fast host. `stream_visualization_test.rs:495` asserts a frame drop rate under 5% over frames paced by `thread::sleep` (`:475`), and `:670` asserts under 10% likewise.
- **Sleeps used as synchronization** in `stream_visualization_test.rs` at `:93`, `:159`, `:475`, `:574`, `:578`, `:610`, `:647` and `:651`; each waits on elapsed time where the stream can expose an event, channel or counter to wait on.
- **Outcome:** timing moves to criterion benches under `crates/kwavers/benches/`, the method kwavers#926 used for `photoacoustic_pipeline`; the tests keep only value assertions (result structure, finiteness, energy, drop counts computed from injected frame timestamps) and synchronize on events, channels or an injected clock.
- **Acceptance:** no `Instant`, `elapsed`, `as_millis` threshold or `thread::sleep` remains in the three files (a `git grep -nE` for them over the three paths is empty); the benches build under the single-iteration smoke; the three tests pass 20 consecutive `cargo nextest run` iterations on a loaded host.
- **Next step:** read the stream module's public surface for an event or counter that replaces each sleep, then move the timing assertions.
