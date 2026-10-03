<a id="kw-fft-050"></a>

## KW-FFT-050 — Direct Apollo axis FFT storage [patch] — review

priority: tightening; needs: none; scope: `crates/kwavers-math/src/fft/` axis-transform facade; the viscoacoustic solver caller

- **Driver:** each viscoacoustic derivative copied a full `Array3<Complex64>` into and out of Apollo although both sides use Leto storage and `eunomia::Complex64`, so the three velocity gradients and three divergence derivatives made twelve temporary full fields and twenty-four avoidable full-buffer copies per solver step.
- **Acceptance:** the facade delegates directly to Apollo's axis plan methods and `decay_matches_dispersion_3d_diagonal` passes under the unchanged Nextest timeout and workload (below the 60 s cap).
- **Next step:** re-validate that `fft_3d_axis_complex_inplace` and `ifft_3d_axis_complex_inplace` copy nothing, and that the regression meets the 30 s slow bound; close if so.
