<a id="kw-diag-037"></a>

## KW-DIAG-037 — Promote multimodal fusion to Diagnostics [major] — todo

priority: architecture; needs: KW-ARCH-036; scope: `kwavers-physics::acoustics::imaging::fusion`, `kwavers-diagnostics`

- **Outcome:** the complete `kwavers-physics::acoustics::imaging::fusion` ownership moves into `kwavers-diagnostics` with every call site rewritten directly; the old physics path is deleted and no re-export remains.
- **Acceptance:** Physics has no registration dependency and Diagnostics owns all fusion and registration contracts.
