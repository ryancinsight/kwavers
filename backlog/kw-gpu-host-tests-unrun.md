<a id="kw-gpu-host-tests-unrun"></a>

## KW-GPU-HOST-TESTS-UNRUN — kwavers-gpu device tests pass silently on an adapter-less runner [patch] — todo

priority: verification; needs: none; basis: 2824e0ecf45fd62c381bf935fd032326171b1fdd; scope: .github/workflows/ci.yml, crates/kwavers-gpu

- **Delivered.** [#965](https://github.com/ryancinsight/kwavers/pull/965) runs `cargo nextest run --locked -p kwavers-gpu --features gpu --lib` in the `validation` job: 175 tests, and a failure reaches the required `CI gate`.
- **Finding.** Device tests report PASS without asserting when no adapter exists: `backend/tests.rs:214`, `:243`, `:260`, `:280` return with no message. The full set of such sites is not enumerated, and test duration does not identify them. `neural_network/tests.rs:53-59` asserts the host-side `validate_activation_type` behind that early return, so a break there is uncaught in CI.
- **Next step.** Enumerate the early-return sites; split host-side assertions out of device tests; mark device-acquiring tests with a named test group or `#[ignore]` with a reason and exclude them by filterset; correct the stale timing comment on the `validation` job; measure the step's warm time against the five-minute target (3m43s cold).
- **Acceptance.** A deliberately broken `validate_activation_type` fails the CI gate on a pull request, and no test reports PASS without executing an assertion.
