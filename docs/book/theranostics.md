# Ultrasound Theranostics

![Theranostic feedback loop](figures/theranostics_feedback_loop.svg)

## Scope

Theranostics chapters cover closed-loop workflows where diagnostic imaging estimates state and therapy updates exposure in response. Code ownership spans `clinical::imaging`, `clinical::therapy`, `simulation`, and shared acoustic/thermal/cavitation physics.

## Theorem: Closed-Loop Exposure Monotonicity

Let `D_k` be cumulative dose and `u_k >= 0` be delivered exposure at step `k`. If the dose update is

```text
D_{k+1} = D_k + phi(u_k, x_k) dt,
```

with `phi(u, x) >= 0`, then dose is monotone non-decreasing.

### Proof Sketch

`D_{k+1} - D_k = phi(u_k, x_k) dt`. Since `dt > 0` and `phi >= 0`, the increment is non-negative for every step. Induction over steps proves monotonicity.

## Algorithm: Image-Guided Therapy Loop

1. Acquire diagnostic state: anatomy, flow, bubble activity, temperature, or displacement.
2. Register state to the therapy coordinate frame using RITK-backed registration where applicable.
3. Predict exposure with the production acoustic solver.
4. Update therapy control under safety constraints.
5. Re-image and validate monotonic dose, bounded risk metrics, and target coverage.

## Implementation Targets

- Encode closed-loop state, registration, acoustic prediction, and control as separate vertical modules.
- Store therapy and diagnostic uncertainty explicitly instead of hiding it in scalar tolerances.
- Generate validation figures from executable examples and keep them linked from the chapter text.

## Recent Research Anchors

- Microbubble-enhanced focused ultrasound is being reviewed for precision glioma workflows: https://doi.org/10.3390/biomedicines12061230
- Ultrasound-stimulated microbubbles for radiotherapeutic enhancement are under active review: https://pubmed.ncbi.nlm.nih.gov/40397648/
- Ultrasound microbubble drug and gene delivery remains an active theranostic technology area: https://doi.org/10.1016/j.jddst.2023.105312

## 2026 Implementation Synchronization

- Theranostic examples must carry state-estimator uncertainty into controller decisions instead of encoding uncertainty as scalar tolerances hidden inside acceptance tests.
- Closed-loop focused-ultrasound workflows should validate monotone dose accumulation, bounded MI/TI/cavitation metrics, and registration consistency in one executable scenario when diagnostic imaging drives therapy updates.
- Microbubble-mediated theranostics require paired delivery and monitoring evidence: acoustic exposure, bubble activity, image-derived state, and controller action must remain separately inspectable in generated examples.
