<a id="kw-gpu-host-tests-unrun"></a>

## KW-GPU-HOST-TESTS-UNRUN — CI runs no kwavers-gpu test, including the host-only ones [patch] — todo

priority: verification; needs: none; scope: .github/workflows/ci.yml, crates/kwavers-gpu

- **Finding.** The library-test shards run `cargo nextest run --workspace --exclude kwavers-python --exclude kwavers-gpu --lib`, and the only other kwavers-gpu job is `cargo check -p kwavers-gpu --features gpu --lib`. Tests that need no device never run in CI. Examples: the medium snapshot resolver tests, and `runner/absorption.rs`'s check that `PowerLawDispersion::Omitted` zeroes η. Only the local pre-push gate runs them.
- **Next step.** Add a job, or extend a shard, running `cargo nextest run -p kwavers-gpu --features gpu --lib` with a filterset that excludes device-acquiring tests. Those tests need a named test group or an `#[ignore]` with a reason. Budget the job's warm time against the five-minute target.
- **Acceptance.** A deliberately broken host-only kwavers-gpu assertion fails the CI gate on a pull request.
