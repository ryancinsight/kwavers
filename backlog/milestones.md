# Milestone ladder ([ADR 138](../docs/adr/138-release-scope-4-0.md))
<a id="M-solvers-core"></a> **M-solvers-core** (3.1.0) needs: (none) — solver and GPU PSTD core on admitted provider floors.
- exit: pass solver tests (`cargo test -p kwavers-solver --locked --all-features` passes).
- exit: pass GPU host tests (`cargo test -p kwavers-gpu --locked --all-features` passes).
- standing: no open correctness item.
<a id="M-imaging-capability"></a> **M-imaging-capability** (3.2.0) needs: M-solvers-core — aperture, imaging, and medium capability.
- exit: pass imaging tests (`cargo test -p kwavers-imaging --locked --all-features` passes).
- exit: pass medium and transducer tests (`cargo test -p kwavers-medium --locked --all-features` and `cargo test -p kwavers-transducer --locked --all-features` pass).
- standing: no open correctness item.
<a id="M-4.0"></a> **M-4.0** (4.0.0) needs: M-imaging-capability — stable simulation release (authority-gated).
- standing: `cargo fmt --all -- --check`, `cargo clippy --workspace --locked --all-targets --all-features -- -D warnings`, `cargo test --workspace --locked --all-features`, `cargo test --workspace --locked --doc --all-features`, and `cargo doc --workspace --locked --no-deps --all-features` pass; SemVer and publish dry-run gates required for declared releases pass; README and rustdoc are current; no open correctness item.
