# KW-PROVIDER-CUTOVERS — Candidate evidence and remaining gates

- The cutover started at Kwavers `45db5f1d0` and now includes upstream
  `7681bb35b` with Consus 0.5 storage contracts.
- Locked candidate sources are Horae `fd117ce2`, Iris `f4f6c951`, RITK
  `0602cfe8`, and Hephaestus `42fdff43`. Each revision is reachable through
  a named GitHub branch.
- FDTD preserves explicit sensor order while collecting whole-grid statistics.
  Where the replaced second recorder allocated S pressure rows with N samples,
  the redundant pressure allocation occupied 8 × S × N bytes.
- Iris provides the plot data contract and Canvas renderer. Chrome renders
  line, heatmap, and point-cloud exports and exercises drag, zoom, and reset.
  Kwavers tests assert pressure values, field coordinate order, and real
  recorded times. The plotted data and browser engine are in one file.
- Horae owns the Gregorian calendar and UTC serialization. Direct Chrono
  dependencies leave Kwavers and CFDrs. Hermes uses the same provider.
  Kwavers preserves the main branch's `SystemTime` diagnostic field and uses
  Horae for civil-time reports. Native RITK removes the final Chrono route
  through the retired DICOM dataset.
- RITK's native DIMSE transport and borrowed catalog selection pass 524
  selected DICOM tests. The catalog selection probe observes zero allocated
  bytes on a 32-frame input already in memory.
- The numerical module split retains the tests for the analytical imaging and
  FDTD algorithm families. The viscoacoustic odd-grid regression fails on
  the original frequency rule and passes with the shared signed spectrum.
- The PSTD runner rejects both oversized FFT requests before it stages
  boundary and absorption volumes. Without PML, six identity fields share
  one host allocation. See [KW-GPU-FFT-PLAN-CAPACITY](kw-gpu-fft-plan-capacity.md)
  for the numerical limits and device coverage.
- The earlier `1967dc251` candidate passes 1,277 selected Kwavers tests,
  including all 38 Python library tests. Six real-GPU capacity tests run
  separately. The new RITK pin passes 635 tests, including parser-budget
  and native dataset-writer integration cases.
- With RITK `0602cfe8`, 1,057 locked tests pass across
  `kwavers-imaging`, `kwavers-analysis`, and `kwavers-diagnostics` with
  all features. The split numerical module families peak at 398 lines.
- The locked all-feature, all-target Cargo tree and lockfile both exclude
  Chrono. `cargo-deny check advisories` passes without a Chrono exception.
- The plotting-disabled build, clinical example, and Python validation
  harness compile. The hosted wheel checks pass the k-Wave comparison,
  angular-spectrum and thermal-acoustic cases. The merged cutover and
  published cohort resolve from fresh workspaces with zero Chrono.
  [The registry release record](kw-release-crates-01.md) links the evidence.
- 103 other tracked Rust files still exceed the 500-line limit. The largest
  production module is the lithotripsy cavitation cloud. Split remaining
  modules by numerical operation and measure compile-time specialization,
  instruction size, and throughput on the same pinned graph.
- Separate work remains for full-stack runtime and binary-size profiling
  and the in-flight inverse block changes.
