# Test figure goldens

These PNG files are visual regression references for the plotting integration
tests. Ordinary test runs render into isolated temporary directories and compare
the generated pixels with these files; they do not rewrite this directory.

Regenerate all figure goldens only after an intentional rendering change:

```console
python scripts/regenerate_test_figures.py
```

Review every resulting image and binary diff before committing it. The normal
comparison requires identical PNG dimensions, color representation, and bit
depth. Every generated channel may differ from its golden by at most one 8-bit
code value, the quantization uncertainty of the raster channel. Domain values
and plot semantics remain covered by the integration tests' analytical checks.
Plotters uses the embedded Ubuntu font through its pure-Rust `ab_glyph` backend,
so system fonts cannot change chart layout across operating systems.
