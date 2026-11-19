# KW-PROVIDER-CUTOVERS — Candidate evidence and remaining gates

- Baseline: Kwavers `45db5f1d009031758fb147f32514832995cfd374`.
- Locked Git candidates: Horae `fd117ce2`, Iris `f4f6c951`, RITK `e33229a4`,
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
  RITK still reaches Chrono through the external DICOM object and codec graph.
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
- Selected verification includes 1,194 integrated Kwavers tests, real-GPU
  capacity tests, and six Python array-boundary tests with value, pointer
  identity, strided order, and ownership assertions.
- Separate work remains for the complete RITK dataset and writer, Helios
  migration, full-stack runtime and binary-size profiling, and the in-flight
  inverse block and coefficient cache changes.
