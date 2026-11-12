<a id="kw-provider-convergence"></a>

## KW-PROVIDER-CONVERGENCE — Converge kwavers onto the fleet provider lines (eunomia 0.12, hermes 0.10, leto 0.48) [patch] — todo

priority: correctness; needs: sibling eunomia-<0.13.0 releases, aequitas-python pyo3-0.30, leto-0.48/hermes-0.10 tags; scope: `Cargo.toml`, `Cargo.lock`

- **Finding.** kwavers pins eunomia 0.11.0, hermes-simd 0.9.2, leto/leto-ops 0.46.5 while sibling mains carry eunomia 0.12.0, hermes 0.10.x and leto 0.48.x (both untagged). The Atlas overlay therefore rewrites these sources locally and `cargo` cannot resolve under it.
- **Rehearsal (2026-11-09, outside-atlas probe).** 15 release units cap eunomia below 0.12; widening all to `<0.13.0` selects eunomia 0.12.0, which needs pyo3 0.30 / numpy 0.30. Full-workspace check green with manifest-only changes; leto/hermes stay at 0.46.5/0.9.2 (no releasable 0.48/0.10 content upstream).
- **Acceptance:** `cargo metadata --locked` resolves the committed lock on a clean checkout with the overlay check clean; full-workspace check green; one lock identity for all three providers.
- **Event (2026-11-10, gate).** 18 push attempts prove the 600 s hook budget cannot hold this scope (root manifest = whole workspace: base+tip clippy, full nextest, rustdoc; repo derives a 25-minute serial bound for heavy sims). Attempts bank per-checkout progress but test execution never caches. Owner authorized one-time SKIP_LOCAL_GATE per SHA with CI as verifier.
- **Event (2026-11-10, #1024).** Peer took the CBR branch; judge failed the derivation prose (CBR(1) equality ignores the `max(0.0)` eigenvalue floor). Takeover after 2h silence: prose-only fix posted, peer returned and landed exact-rank assertions (`42a489f`, merged). Full sweep at lane tip showed that CBR test as the sole red; all else green.
- **Event (2026-11-10, transducer).** Pre-existing `unused_must_use` discard fixed (`let _ =`), verified in the converged export (package clippy zero warnings, focused test PASS), committed on `fix/transducer-test-must-use`; pushes after convergence.
- **Next step:** push this lane with the authorized skip → CI green → merge; then transducer push, ritk follow-ups, leto QR scoping.
