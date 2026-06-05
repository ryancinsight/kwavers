"""Segmented-lesion transducer planning — figure rendering for Chapter 32.

Physics is performed by pykwavers (Rust/kwavers library).
This package contains only matplotlib rendering helpers.
"""

from .figures import (
    plot_3d_placement,
    plot_exposure_slice,
    plot_fwi_convergence,
    plot_reconstructions,
    write_metrics,
)

__all__ = [
    "plot_3d_placement",
    "plot_exposure_slice",
    "plot_fwi_convergence",
    "plot_reconstructions",
    "write_metrics",
]
