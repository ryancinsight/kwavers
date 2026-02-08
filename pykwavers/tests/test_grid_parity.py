#!/usr/bin/env python3
"""
Grid Parity Tests: pykwavers.Grid vs kwave.kWaveGrid

Validates that pykwavers Grid creation, dimensions, spacing, and derived
properties match k-wave-python's kWaveGrid for identical configurations.
"""

import numpy as np
import pytest

import pykwavers as kw
from conftest import requires_kwave, HAS_KWAVE

if HAS_KWAVE:
    from kwave.kgrid import kWaveGrid
    from kwave.data import Vector


# ============================================================================
# pykwavers Grid standalone tests
# ============================================================================


class TestGridCreation:
    """Test pykwavers Grid construction and properties."""

    @pytest.mark.parametrize("nx,ny,nz", [
        (8, 8, 8),
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
        (32, 64, 16),
        (128, 128, 1),  # quasi-2D
    ])
    def test_grid_dimensions(self, nx, ny, nz):
        """Grid stores correct dimensions."""
        dx = 0.1e-3
        g = kw.Grid(nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, dz=dx)
        assert g.nx == nx
        assert g.ny == ny
        assert g.nz == nz

    @pytest.mark.parametrize("dx,dy,dz", [
        (0.1e-3, 0.1e-3, 0.1e-3),
        (0.05e-3, 0.05e-3, 0.05e-3),
        (0.2e-3, 0.2e-3, 0.2e-3),
        (0.1e-3, 0.2e-3, 0.15e-3),  # non-uniform
    ])
    def test_grid_spacing(self, dx, dy, dz):
        """Grid stores correct spacing."""
        g = kw.Grid(nx=16, ny=16, nz=16, dx=dx, dy=dy, dz=dz)
        assert abs(g.dx - dx) < 1e-15
        assert abs(g.dy - dy) < 1e-15
        assert abs(g.dz - dz) < 1e-15

    def test_total_points(self):
        """total_points() returns nx*ny*nz."""
        g = kw.Grid(nx=16, ny=32, nz=8, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        assert g.total_points() == 16 * 32 * 8

    def test_grid_repr(self):
        """Grid has meaningful string representation."""
        g = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        r = repr(g)
        assert "32" in r
        assert "Grid" in r

    def test_grid_domain_size(self):
        """Domain size = N * dx for each axis."""
        nx, ny, nz = 64, 32, 16
        dx, dy, dz = 0.1e-3, 0.2e-3, 0.3e-3
        g = kw.Grid(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
        # The pykwavers repr includes domain dimensions; verify from properties
        assert abs(g.nx * g.dx - nx * dx) < 1e-15
        assert abs(g.ny * g.dy - ny * dy) < 1e-15
        assert abs(g.nz * g.dz - nz * dz) < 1e-15


# ============================================================================
# Cross-validation: pykwavers Grid vs kWaveGrid
# ============================================================================


@requires_kwave
class TestGridParityWithKWave:
    """Compare pykwavers Grid properties against kWaveGrid."""

    @pytest.mark.parametrize("N,dx", [
        (32, 0.1e-3),
        (64, 0.05e-3),
        (128, 0.2e-3),
    ])
    def test_grid_shape_matches(self, N, dx):
        """pykwavers Grid dimensions match kWaveGrid."""
        g_pk = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
        g_kw = kWaveGrid(Vector([N, N, N]), Vector([dx, dx, dx]))

        assert g_pk.nx == g_kw.Nx
        assert g_pk.ny == g_kw.Ny
        assert g_pk.nz == g_kw.Nz

    @pytest.mark.parametrize("N,dx", [
        (32, 0.1e-3),
        (64, 0.05e-3),
    ])
    def test_grid_spacing_matches(self, N, dx):
        """pykwavers spacing matches kWaveGrid spacing."""
        g_pk = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
        g_kw = kWaveGrid(Vector([N, N, N]), Vector([dx, dx, dx]))

        assert abs(g_pk.dx - g_kw.dx) < 1e-15
        assert abs(g_pk.dy - g_kw.dy) < 1e-15
        assert abs(g_pk.dz - g_kw.dz) < 1e-15

    def test_total_grid_points_match(self):
        """Total grid points match between implementations."""
        N, dx = 48, 0.1e-3
        g_pk = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
        g_kw = kWaveGrid(Vector([N, N, N]), Vector([dx, dx, dx]))

        pk_total = g_pk.total_points()
        kw_total = g_kw.Nx * g_kw.Ny * g_kw.Nz
        assert pk_total == kw_total

    def test_nonuniform_spacing_match(self):
        """Non-uniform spacing matches between both implementations."""
        nx, ny, nz = 32, 48, 16
        dx, dy, dz = 0.1e-3, 0.15e-3, 0.2e-3

        g_pk = kw.Grid(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
        g_kw = kWaveGrid(Vector([nx, ny, nz]), Vector([dx, dy, dz]))

        assert g_pk.nx == g_kw.Nx
        assert g_pk.ny == g_kw.Ny
        assert g_pk.nz == g_kw.Nz
        assert abs(g_pk.dx - g_kw.dx) < 1e-15
        assert abs(g_pk.dy - g_kw.dy) < 1e-15
        assert abs(g_pk.dz - g_kw.dz) < 1e-15

    def test_quasi_2d_grid(self):
        """Quasi-2D grid (Nz=1) matches."""
        N = 64
        dx = 0.1e-3
        g_pk = kw.Grid(nx=N, ny=N, nz=1, dx=dx, dy=dx, dz=dx)
        g_kw = kWaveGrid(Vector([N, N, 1]), Vector([dx, dx, dx]))

        assert g_pk.nx == g_kw.Nx
        assert g_pk.ny == g_kw.Ny
        assert g_pk.nz == g_kw.Nz


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
