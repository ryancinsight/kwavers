## KW-SOL-002 — apollo-fft kernel panics on transform lengths with unsupported radix (e.g. 380 = 2²·5·19) [major] — todo

priority: solver; needs: none; scope: kwavers-solver PSTD / apollo FFT backend; milestone: M-solver

- **Finding (Build 17 Lever 1, `B17_SCAN.md`):** a PSTD run at grid NY=380 panics deterministically in the apollo-fft kernel: `radix_composite/arity.rs:436: unsupported radix 19`. 380 = 2²·5·19 requires a radix-19 butterfly the kernel does not implement, and there is no fallback (Bluestein / mixed-radix decomposition / naive DFT) for lengths with large prime factors. Reproduced in isolation on an idle system (gen RSS ~146 MiB — not a memory event) and with RAYON_NUM_THREADS=1 (panics in <2 s; the FFT plan is length-dependent). Neighbouring lengths run fine (NY=384, 480, 736 verified in Build 17).
- **Impact:** any grid dimension whose factorisation contains an unsupported prime silently becomes unusable — the failure is a panic at plan time, not a wrong result, but it blocks legitimate geometries (Build 17's L=15 scan point needed a coordinator-declared domain-depth override to NY=384).
- **Ask:** (a) add a Bluestein (or equivalent) fallback for lengths with prime factors above the implemented radix set, OR (b) fail at plan construction with a typed error naming the unsupported factor and the nearest supported lengths; (c) add a plan-time test sweeping lengths 2..1024 asserting no panic.
- **Acceptance:** PSTD smoke runs at NY ∈ {380, 19·16=304, 17·24=408} complete without panic and match a reference DFT-based propagation on a linear homogeneous case.
