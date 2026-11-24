## Physics 5.2.1 checklist

- [x] [patch] Use absolute circular-piston response for pressure magnitudes
  and the secondary-lobe ratio at the commanded angle.
- [x] [patch] Center the uniform-array phase around the nearest spatial
  replica and preserve the signed array factor for odd and even element counts.
- [x] [patch] Verify both counterexamples on the unchanged base and run
  the Physics library and steering integration tests on the candidate.

## Numerical contract

The magnitude of a coherent signed piston response uses the absolute value
of its element factor. This retains lobes whose phase changes by π. The
one-element regression checks both which lobe is counted and which pressure
normalizes the secondary response.

For N centered elements, write the uniform factor as
`sin(Nh)/(N sin(h))`, with `h = ψ/2`. Substituting `h = mπ+r` factors
out `(-1)^(m(N-1))` and leaves a ratio in the residual `r`. At `r=0`
the remaining ratio has limit one. This keeps the cost per angle constant.
The integration test compares the calculation with the finite cosine sum
at positive and negative replicas, their adjacent representable angles,
and off-replica directions.

The baseline is `2e2ef3028`. Both regression tests fail on that base and
the candidate passes all 1,605 Physics library and steering tests. Package
publication and registry acceptance belong to release. Rectangular-strip
directivity remains a distinct model from the circular-piston steering helper.
