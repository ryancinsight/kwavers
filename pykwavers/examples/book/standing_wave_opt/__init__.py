"""Standing-wave suppression figures package for Chapter 31.

Physics and optimization are implemented in Rust (kwavers) and exposed via
the PyO3 binding ``kw.run_standing_wave_suppression()``.  This package
contains only figure-rendering code that operates on the returned dict.

Public API:

    from standing_wave_opt.figures import (
        plot_geometry, plot_field_evolution, plot_convergence, plot_before_after,
    )
"""
