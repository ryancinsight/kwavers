"""Segmented-lesion transducer planning example support."""

from .liver_dataset import load_lits_liver_planning_grid
from .phantom import build_segmented_therapy_phantom
from .solver import optimize_transducer_layout
from .types import HybridPlanConfig, Tissue, TransducerAperture

__all__ = [
    "HybridPlanConfig",
    "Tissue",
    "TransducerAperture",
    "build_segmented_therapy_phantom",
    "load_lits_liver_planning_grid",
    "optimize_transducer_layout",
]
