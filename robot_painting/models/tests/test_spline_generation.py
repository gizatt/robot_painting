"""
Generated with help from ChatGPT 4o, 2024 Aug 20.
"""

from dataclasses import fields
import pytest
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from robot_painting.models.spline_generation import (
    make_random_spline,
    SplineGenerationParams,
    SplineAndOffset,
    spline_from_dict,
    spline_to_dict,
)
import cProfile
import io
import pstats


def test_spline_generation():
    """Basic tests of spline generation."""
    rng = np.random.default_rng(42)
    spline = make_random_spline(SplineGenerationParams(n_steps=4), rng=rng)
    assert isinstance(spline, CubicHermiteSpline)
    t = np.linspace(spline.x[0], spline.x[-1], 100)
    xs = spline(t)
    assert xs.shape[1] == 3  # Should be 3D


def test_spline_start_end():
    """Test to ensure the spline starts at the origin and ends at zero velocity."""
    rng = np.random.default_rng(42)
    params = SplineGenerationParams(n_steps=4)
    spline = make_random_spline(params, rng=rng)
    assert np.allclose(spline(0), [0.0, 0.0, spline(0)[2]], atol=1e-6)
    assert np.allclose(spline.derivative()(spline.x[-1]), [0.0, 0.0, 0.0], atol=1e-6)


def test_randomness_of_spline():
    """Test that the generated splines are not identical (testing randomness)."""
    spline1 = make_random_spline(rng=np.random.default_rng(42))
    spline2 = make_random_spline(rng=np.random.default_rng(43))
    assert not np.allclose(spline1.x, spline2.x)
    assert not np.allclose(spline1.c, spline2.c)


def test_spline_with_different_rng():
    """Test generating splines with a fixed RNG to ensure reproducibility."""
    spline1 = make_random_spline(rng=np.random.default_rng(42))
    spline2 = make_random_spline(rng=np.random.default_rng(42))
    assert np.allclose(spline1.x, spline2.x)
    assert np.allclose(spline1.c, spline2.c)


def profile_spline_generation():
    """Profile the spline generation function."""
    rng = np.random.default_rng(42)
    pr = cProfile.Profile()
    pr.enable()
    for k in range(1000):
        make_random_spline(SplineGenerationParams(n_steps=10, rng=rng))
    pr.disable()

    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())  # This will print the profiling results


def test_spline_generation_performance(benchmark):
    """Benchmark the performance of spline generation."""
    rng = np.random.default_rng(42)
    benchmark(make_random_spline, SplineGenerationParams(n_steps=10), rng=rng)


def test_spline_dict_conversions():
    """Test that serializing and then deserializing returns an equivalent object."""
    spline = make_random_spline(SplineGenerationParams(n_steps=4))
    spline_dict = spline_to_dict(spline)
    deserialized_spline = spline_from_dict(spline_dict)

    # Ensure that the deserialized spline is equivalent to the original
    assert np.array_equal(spline.c, deserialized_spline.c)
    assert np.array_equal(spline.x, deserialized_spline.x)
    t = np.linspace(spline.x[0] - 1, spline.x[-1] + 1, 10)
    assert np.array_equal(spline(t), deserialized_spline(t), equal_nan=True)

def test_spline_and_offset():
    spline = make_random_spline(SplineGenerationParams(n_steps=4))
    offset = np.array([1., 2., 0.])
    spline_and_offset = SplineAndOffset(spline=spline, offset=offset)
    xs = spline_and_offset.sample(N=100)
    assert xs.shape == (100, 3)

    data_dict = spline_and_offset.to_dict()
    deserialized_spline_and_offset = SplineAndOffset.from_dict(data_dict)
    xs_deserialized = deserialized_spline_and_offset.sample(N=100)
    assert np.allclose(xs, xs_deserialized)

if __name__ == "__main__":
    profile_spline_generation()
