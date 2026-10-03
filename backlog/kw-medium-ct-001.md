<a id="kw-medium-ct-001"></a>

## KW-MEDIUM-CT-001 — Own complete CT medium assembly [arch] — review

priority: architecture; needs: none; scope: `kwavers-medium::heterogeneous` CT builder; the former `kwavers-physics` skull-owned builder

- **Driver:** a private downstream consumer requires the provider-owned standard-HU medium contract.
- **Acceptance:** `CtMediumBuilder` is exported only by `kwavers-medium`, maps all five acoustic fields through `HuAcousticModel`, rejects shape mismatch, and focused package gates pass.
- **Evidence:** warning-denied all-target/all-feature `kwavers-medium` Clippy and Nextest 187/187 on the aligned provider graph. At origin/main `CtMediumBuilder` lives in `crates/kwavers-medium/src/heterogeneous/ct.rs` and is used from `kwavers-solver`.
- **Next step:** confirm no `kwavers-physics` builder remains; close if so.
