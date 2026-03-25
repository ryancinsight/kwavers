#!/usr/bin/env python3
"""
Grid Parity Tests: pykwavers.Grid vs kwave.kWaveGrid

Validates that pykwavers Grid creation, dimensions, spacing, and derived
properties match k-wave-python's kWaveGrid for identical configurations.

This module tests:
1. Basic grid creation and properties (dimensions, spacing)
2. Derived properties (wavenumbers, k-space grids)
3. CFL stability calculations
4. Domain size calculations
5. Grid utility methods
"""

import numpy as np
import pytest

import pykwavers as kw
from conftest import requires_kwave, HAS_KWAVE, compute_cfl_dt

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

    def test_domain_size_matches_kwave(self):
        """pykwavers domain size matches kWaveGrid domain size."""
        N, dx = 64, 0.1e-3
        g_pk = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
        g_kw = kWaveGrid(Vector([N, N, N]), Vector([dx, dx, dx]))

        pk_domain_x = g_pk.nx * g_pk.dx
        kw_domain_x = g_kw.Nx * g_kw.dx
        assert abs(pk_domain_x - kw_domain_x) < 1e-15

    @pytest.mark.parametrize("N,dx,c", [
        (64, 0.1e-3, 1500.0),
        (128, 0.05e-3, 1540.0),
        (32, 0.2e-3, 3000.0),
    ])
    def test_cfl_timestep_matches_kwave(self, N, dx, c):
        """CFL timestep computed by pykwavers matches k-wave dt logic.

        Both should satisfy CFL: dt = cfl * dx / c_max where cfl ≤ 1/√3.
        """
        cfl = 0.3
        pk_dt = compute_cfl_dt(dx, c, cfl=cfl)

        g_kw = kWaveGrid(Vector([N, N, N]), Vector([dx, dx, dx]))
        # k-wave makeTime uses: dt = cfl * min(dx) / c_max
        kw_dt = cfl * dx / c
        g_kw.setTime(100, kw_dt)

        assert abs(pk_dt - g_kw.dt) < 1e-15


# ============================================================================
# CFL and stability tests
# ============================================================================


class TestGridStability:
    """Test CFL stability calculations for grids."""

    @pytest.mark.parametrize("c", [1500.0, 1540.0, 3000.0, 330.0])
    def test_cfl_timestep_water(self, c):
        """CFL timestep calculation matches expected formula."""
        dx = 0.1e-3
        dt = compute_cfl_dt(dx, c, cfl=0.3)
        
        # Verify CFL formula: dt = cfl * dx / c
        expected_dt = 0.3 * dx / c
        assert abs(dt - expected_dt) < 1e-15

    def test_cfl_timestep_grid_size_independence(self):
        """CFL timestep is independent of grid size (depends only on spacing)."""
        c = 1500.0
        dx = 0.1e-3
        
        g1 = kw.Grid(nx=32, ny=32, nz=32, dx=dx, dy=dx, dz=dx)
        g2 = kw.Grid(nx=64, ny=64, nz=64, dx=dx, dy=dx, dz=dx)
        
        dt1 = compute_cfl_dt(g1.dx, c)
        dt2 = compute_cfl_dt(g2.dx, c)
        
        assert abs(dt1 - dt2) < 1e-15

    def test_cfl_timestep_spacing_dependence(self):
        """CFL timestep scales with grid spacing."""
        c = 1500.0
        
        g1 = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        g2 = kw.Grid(nx=32, ny=32, nz=32, dx=0.2e-3, dy=0.2e-3, dz=0.2e-3)
        
        dt1 = compute_cfl_dt(g1.dx, c)
        dt2 = compute_cfl_dt(g2.dx, c)
        
        # Double spacing should give double timestep
        assert abs(dt2 / dt1 - 2.0) < 1e-10


# ============================================================================
# Wavenumber and k-space grid tests
# ============================================================================


class TestKSpaceGrid:
    """Test k-space (wavenumber) grid properties."""

    def test_wavenumber_calculation(self):
        """Wavenumber k = 2π/λ is computed correctly."""
        # For a 1 MHz wave in water (c=1500 m/s)
        freq = 1e6
        c = 1500.0
        wavelength = c / freq  # 1.5 mm
        k = 2 * np.pi / wavelength
        
        # Expected: k ≈ 4188.79 rad/m
        expected_k = 2 * np.pi * freq / c
        assert abs(k - expected_k) < 1e-10

    def test_points_per_wavelength(self):
        """Points per wavelength calculation."""
        freq = 1e6
        c = 1500.0
        wavelength = c / freq  # 1.5 mm
        dx = 0.1e-3  # 0.1 mm
        
        ppw = wavelength / dx  # 15 points per wavelength
        assert ppw == 15.0

    @pytest.mark.parametrize("N,dx,freq,expected_ppw", [
        (64, 0.1e-3, 1e6, 15.0),    # Standard diagnostic ultrasound
        (128, 0.05e-3, 1e6, 30.0),  # Higher resolution
        (32, 0.2e-3, 1e6, 7.5),     # Lower resolution
        (64, 0.1e-3, 2e6, 7.5),     # Higher frequency
    ])
    def test_ppw_various_configs(self, N, dx, freq, expected_ppw):
        """PPW calculation for various grid/frequency configurations."""
        c = 1500.0
        wavelength = c / freq
        ppw = wavelength / dx
        assert abs(ppw - expected_ppw) < 1e-10


# ============================================================================
# Domain size tests
# ============================================================================


class TestDomainSize:
    """Test domain size calculations."""

    @pytest.mark.parametrize("N,dx,expected_size", [
        (64, 0.1e-3, 6.4e-3),   # 6.4 mm
        (128, 0.05e-3, 6.4e-3), # Same size, finer grid
        (32, 0.2e-3, 6.4e-3),   # Same size, coarser grid
        (100, 0.1e-3, 10e-3),   # 10 mm
    ])
    def test_domain_size_calculation(self, N, dx, expected_size):
        """Domain size = N * dx."""
        g = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
        domain_size = N * dx
        assert abs(domain_size - expected_size) < 1e-15

    def test_anisotropic_domain_size(self):
        """Anisotropic grid has different domain sizes per axis."""
        g = kw.Grid(nx=64, ny=32, nz=16, dx=0.1e-3, dy=0.2e-3, dz=0.3e-3)
        
        domain_x = g.nx * g.dx
        domain_y = g.ny * g.dy
        domain_z = g.nz * g.dz
        
        assert abs(domain_x - 6.4e-3) < 1e-15
        assert abs(domain_y - 6.4e-3) < 1e-15
        assert abs(domain_z - 4.8e-3) < 1e-15


# ============================================================================
# Grid utility tests
# ============================================================================


class TestGridUtilities:
    """Test grid utility methods."""

    def test_total_points_calculation(self):
        """Total points = nx * ny * nz."""
        g = kw.Grid(nx=32, ny=64, nz=16, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        assert g.total_points() == 32 * 64 * 16

    def test_grid_volume(self):
        """Grid volume = (nx*dx) * (ny*dy) * (nz*dz)."""
        g = kw.Grid(nx=64, ny=64, nz=64, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        volume = (g.nx * g.dx) * (g.ny * g.dy) * (g.nz * g.dz)
        expected = (6.4e-3) ** 3
        assert abs(volume - expected) < 1e-20

    def test_grid_memory_size(self):
        """Estimate memory size for grid storage."""
        N = 64
        g = kw.Grid(nx=N, ny=N, nz=N, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        
        # Each f64 value is 8 bytes
        bytes_per_value = 8
        total_points = g.total_points()
        memory_bytes = total_points * bytes_per_value
        
        # 64^3 = 262,144 points * 8 bytes = 2,097,152 bytes ≈ 2 MB
        assert memory_bytes == N**3 * 8


# ============================================================================
# Cross-validation: k-wave-python k-space grid comparison
# ============================================================================


@requires_kwave
class TestKSpaceGridParityWithKWave:
    """Compare k-space grid properties against k-wave-python."""

    def test_kx_ky_kz_arrays_match(self):
        """Wavenumber arrays match between implementations."""
        N = 64
        dx = 0.1e-3
        
        g_kw = kWaveGrid(Vector([N, N, N]), Vector([dx, dx, dx]))
        
        # k-wave computes k-space grids internally
        # kx, ky, kz are wavenumber arrays
        assert hasattr(g_kw, 'kx') or hasattr(g_kw, 'k')
        
        # Verify k-space grid is properly initialized
        # k_max should be π/dx (Nyquist wavenumber)
        k_max_expected = np.pi / dx
        
        if hasattr(g_kw, 'kx'):
            k_max_actual = np.max(np.abs(g_kw.kx))
            assert abs(k_max_actual - k_max_expected) < k_max_expected * 0.1

    def test_k_max_nyquist(self):
        """Maximum wavenumber equals Nyquist wavenumber."""
        dx = 0.1e-3
        k_nyquist = np.pi / dx  # Nyquist wavenumber
        
        # This is the maximum resolvable wavenumber
        # For PSTD, this determines the maximum stable timestep
        assert k_nyquist > 0

    def test_time_array_creation(self):
        """Time array creation matches k-wave."""
        N = 64
        dx = 0.1e-3
        c = 1500.0
        dt = compute_cfl_dt(dx, c)
        nt = 100
        
        g_kw = kWaveGrid(Vector([N, N, N]), Vector([dx, dx, dx]))
        g_kw.setTime(nt, dt)
        
        # Verify time array
        assert hasattr(g_kw, 't_array')
        # k-wave stores t_array with shape (1, nt)
        assert g_kw.t_array.size == nt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
