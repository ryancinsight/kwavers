#!/usr/bin/env python3
"""
Comparison Report Generator for pykwavers vs k-wave-python

Generates unified reports comparing simulation results between pykwavers
and k-wave-python across all test categories.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class MetricResult:
    """Single metric comparison result."""
    name: str
    pykwavers_value: float
    kwave_value: float
    l2_error: Optional[float] = None
    correlation: Optional[float] = None
    linf_error: Optional[float] = None
    rms_ratio: Optional[float] = None
    best_lag: Optional[int] = None
    passed: bool = True
    notes: str = ""


@dataclass
class CategoryReport:
    """Report for a single test category."""
    category: str
    description: str
    metrics: List[MetricResult] = field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    
    def add_metric(self, metric: MetricResult):
        self.metrics.append(metric)
        self.total_tests += 1
        if metric.passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1


@dataclass
class ComparisonReport:
    """Full comparison report."""
    timestamp: str
    pykwavers_version: str
    kwave_version: str
    categories: List[CategoryReport] = field(default_factory=list)
    
    @property
    def total_tests(self) -> int:
        return sum(c.total_tests for c in self.categories)
    
    @property
    def passed_tests(self) -> int:
        return sum(c.passed_tests for c in self.categories)
    
    @property
    def failed_tests(self) -> int:
        return sum(c.failed_tests for c in self.categories)
    
    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests * 100


class ComparisonReportGenerator:
    """Generate comparison reports between pykwavers and k-wave-python."""
    
    # Tolerance thresholds for different solver types
    SOLVER_TOLERANCES = {
        "fdtd": {"l2_max": 1.50, "linf_max": 2.00, "corr_min": 0.40},
        "pstd": {"l2_max": 0.90, "linf_max": 1.20, "corr_min": 0.65},
    }
    
    def __init__(self, pykwavers_version: str = "unknown", kwave_version: str = "unknown"):
        self.pykwavers_version = pykwavers_version
        self.kwave_version = kwave_version
        self.report = ComparisonReport(
            timestamp=datetime.utcnow().isoformat() + "Z",
            pykwavers_version=pykwavers_version,
            kwave_version=kwave_version,
        )
    
    def add_category(self, category: str, description: str) -> CategoryReport:
        """Add a new category to the report."""
        cat_report = CategoryReport(category=category, description=description)
        self.report.categories.append(cat_report)
        return cat_report
    
    def compute_metrics(
        self,
        kwave_data: np.ndarray,
        pykwavers_data: np.ndarray,
        name: str,
        solver_type: str = "fdtd",
        notes: str = "",
    ) -> MetricResult:
        """Compute comparison metrics between k-wave and pykwavers data."""
        # Ensure same length
        min_len = min(len(kwave_data), len(pykwavers_data))
        kw = kwave_data[:min_len].flatten()
        pk = pykwavers_data[:min_len].flatten()
        
        # Basic checks
        kw_finite = np.all(np.isfinite(kw))
        pk_finite = np.all(np.isfinite(pk))
        kw_nonzero = np.max(np.abs(kw)) > 0
        pk_nonzero = np.max(np.abs(pk)) > 0
        
        if not (kw_finite and pk_finite and kw_nonzero and pk_nonzero):
            return MetricResult(
                name=name,
                pykwavers_value=np.max(np.abs(pk)) if pk_finite else float('nan'),
                kwave_value=np.max(np.abs(kw)) if kw_finite else float('nan'),
                passed=False,
                notes="Data validation failed (non-finite or zero)",
            )
        
        # L2 error
        l2_error = np.linalg.norm(pk - kw) / np.linalg.norm(kw)
        
        # Linf error
        linf_error = np.max(np.abs(pk - kw)) / np.max(np.abs(kw))
        
        # Correlation
        kw_norm = kw - np.mean(kw)
        pk_norm = pk - np.mean(pk)
        correlation = np.sum(kw_norm * pk_norm) / (
            np.sqrt(np.sum(kw_norm ** 2)) * np.sqrt(np.sum(pk_norm ** 2)) + 1e-10
        )
        
        # RMS ratio
        rms_kw = np.sqrt(np.mean(kw ** 2))
        rms_pk = np.sqrt(np.mean(pk ** 2))
        rms_ratio = rms_pk / rms_kw if rms_kw > 0 else float('inf')
        
        # Best lag (cross-correlation)
        xcorr = np.correlate(kw_norm, pk_norm, mode='full')
        lags = np.arange(-len(kw) + 1, len(kw))
        best_lag = int(lags[np.argmax(xcorr)])
        
        # Check against tolerances
        tolerance = self.SOLVER_TOLERANCES.get(solver_type, self.SOLVER_TOLERANCES["fdtd"])
        passed = (
            l2_error < tolerance["l2_max"]
            and linf_error < tolerance["linf_max"]
            and correlation > tolerance["corr_min"]
        )
        
        return MetricResult(
            name=name,
            pykwavers_value=rms_pk,
            kwave_value=rms_kw,
            l2_error=l2_error,
            linf_error=linf_error,
            correlation=correlation,
            rms_ratio=rms_ratio,
            best_lag=best_lag,
            passed=passed,
            notes=notes,
        )
    
    def generate_markdown_report(self) -> str:
        """Generate a markdown formatted report."""
        lines = [
            "# pykwavers vs k-wave-python Comparison Report",
            "",
            f"**Generated:** {self.report.timestamp}",
            f"**pykwavers version:** {self.report.pykwavers_version}",
            f"**k-wave-python version:** {self.report.kwave_version}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Tests | {self.report.total_tests} |",
            f"| Passed | {self.report.passed_tests} |",
            f"| Failed | {self.report.failed_tests} |",
            f"| Pass Rate | {self.report.pass_rate:.1f}% |",
            "",
        ]
        
        for category in self.report.categories:
            lines.extend([
                f"## {category.category}",
                "",
                f"{category.description}",
                "",
                f"**Tests:** {category.total_tests} total, {category.passed_tests} passed, {category.failed_tests} failed",
                "",
            ])
            
            if category.metrics:
                lines.extend([
                    "| Test | L2 Error | Correlation | RMS Ratio | Best Lag | Status |",
                    "|------|----------|-------------|-----------|----------|--------|",
                ])
                
                for m in category.metrics:
                    status = "✅" if m.passed else "❌"
                    l2 = f"{m.l2_error:.3f}" if m.l2_error is not None else "N/A"
                    corr = f"{m.correlation:.3f}" if m.correlation is not None else "N/A"
                    rms = f"{m.rms_ratio:.3f}" if m.rms_ratio is not None else "N/A"
                    lag = f"{m.best_lag}" if m.best_lag is not None else "N/A"
                    
                    lines.append(f"| {m.name} | {l2} | {corr} | {rms} | {lag} | {status} |")
                
                lines.append("")
        
        return "\n".join(lines)
    
    def generate_json_report(self) -> str:
        """Generate a JSON formatted report."""
        def convert_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: convert_to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        return json.dumps(convert_to_dict(self.report), indent=2)
    
    def save_report(self, filepath: str, format: str = "markdown"):
        """Save report to file."""
        if format == "markdown":
            content = self.generate_markdown_report()
            ext = ".md"
        elif format == "json":
            content = self.generate_json_report()
            ext = ".json"
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Ensure extension
        if not filepath.endswith(ext):
            filepath += ext
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return filepath


def create_sample_report() -> ComparisonReport:
    """Create a sample report for demonstration."""
    generator = ComparisonReportGenerator(
        pykwavers_version="0.1.0",
        kwave_version="0.3.2",
    )
    
    # Grid tests
    grid_cat = generator.add_category("Grid", "Grid creation and k-space grid tests")
    
    # Sample data for grid test
    t = np.linspace(0, 1, 100)
    kwave_signal = np.sin(2 * np.pi * 5 * t)
    pykwavers_signal = np.sin(2 * np.pi * 5 * t + 0.01)  # Slight phase shift
    
    metric = generator.compute_metrics(kwave_signal, pykwavers_signal, "Grid creation", "fdtd")
    grid_cat.add_metric(metric)
    
    # Solver tests
    solver_cat = generator.add_category("Solver", "FDTD and PSTD solver parity tests")
    
    # FDTD test
    fdtd_kw = np.sin(2 * np.pi * 3 * t) * np.exp(-t)
    fdtd_pk = np.sin(2 * np.pi * 3 * t + 0.02) * np.exp(-t) * 0.95
    fdtd_metric = generator.compute_metrics(fdtd_kw, fdtd_pk, "FDTD plane wave", "fdtd")
    solver_cat.add_metric(fdtd_metric)
    
    # PSTD test
    pstd_kw = np.sin(2 * np.pi * 3 * t) * np.exp(-t)
    pstd_pk = np.sin(2 * np.pi * 3 * t + 0.1) * np.exp(-t) * 0.5  # Larger difference
    pstd_metric = generator.compute_metrics(pstd_kw, pstd_pk, "PSTD plane wave", "pstd")
    solver_cat.add_metric(pstd_metric)
    
    return generator.report


if __name__ == "__main__":
    # Generate sample report
    generator = ComparisonReportGenerator(
        pykwavers_version="0.1.0",
        kwave_version="0.3.2",
    )
    
    # Add sample categories and metrics
    grid_cat = generator.add_category("Grid", "Grid creation and k-space grid tests")
    
    t = np.linspace(0, 1, 100)
    kwave_signal = np.sin(2 * np.pi * 5 * t)
    pykwavers_signal = np.sin(2 * np.pi * 5 * t + 0.01)
    
    metric = generator.compute_metrics(kwave_signal, pykwavers_signal, "Grid creation", "fdtd")
    grid_cat.add_metric(metric)
    
    # Print markdown report
    print(generator.generate_markdown_report())
    
    # Save to file
    output_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(output_dir, exist_ok=True)
    
    generator.save_report(os.path.join(output_dir, "comparison_report"), format="markdown")
    generator.save_report(os.path.join(output_dir, "comparison_report"), format="json")
    
    print(f"\nReports saved to {output_dir}")
