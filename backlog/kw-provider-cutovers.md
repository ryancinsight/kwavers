# KW-PROVIDER-CUTOVERS — Candidate evidence and remaining gates

- Initial Kwavers baseline `45db5f1d0`. The candidate merges main `08a1a5054`.
- Locked Git candidates: Horae `fd117ce2`, Iris `f4f6c951`, RITK `9163710e`,
  and Hephaestus `42fdff43`. Reviewed source availability and fresh-consumer
  resolution remain release gates.
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
  Horae for civil-time reports. RITK reaches Chrono through the external
  DICOM object and codec graph.
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
- The final locked Nextest run passes 1,277 selected Kwavers tests,
  including all 38 Python library tests, and six real-GPU capacity tests
  run separately. The RITK pin passes 564 selected tests including the
  parser-budget integration cases. Each Nextest run uses a 60-second limit.
- The all-feature, all-target Cargo tree shows only the RITK route to
  Chrono. Formatting passes for 116 changed Kwavers Rust files. The split
  numerical module families peak at 398 lines.
- The plotting-disabled build, clinical example, and Python validation
  harness compile on the reconciled tree. Live external k-Wave parity,
  hosted CI, and fresh-consumer resolution remain open.
- The RITK comparison uses the verified `73f66b716` base. Its later
  upstream geometry changes require an integration replay.
- 103 other tracked Rust files still exceed the 500-line limit. The largest
  production module is the lithotripsy cavitation cloud. Split remaining
  modules by numerical operation and measure compile-time specialization,
  instruction size, and throughput on the same pinned graph.
- Separate work remains for the complete RITK dataset and writer, Helios
  migration, full-stack runtime and binary-size profiling, and the in-flight
  inverse block and coefficient cache changes.
