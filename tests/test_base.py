# rbartpackages/tests/test_base.py
#
# Copyright (c) 2026, The rbartpackages Contributors
#
# This file is part of rbartpackages.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tests for the shared wrapper plumbing in `rbartpackages.base`."""

import ctypes
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from rbartpackages import base
from rbartpackages._src.base import fork_safe_native_threads, robjects_r
from tests.util import assert_allclose, assert_array_equal


def stats4_loaded() -> bool:
    """Whether the R namespace stats4 is loaded."""
    return bool(robjects_r('isNamespaceLoaded("stats4")')[0])


def test_rfunction() -> None:
    """`rfunction` loads the R package at decoration time and converts arguments."""
    # stats4 ships with R, but neither R nor the wrapped packages load it:
    # the decoration must do it
    assert not stats4_loaded()

    @partial(base.rfunction, library='stats4', rname='mle')
    def mle(minuslogl: object, **kw: object) -> object:
        """Maximum likelihood estimation; returns an S4 object."""
        ...

    @partial(base.rfunction, library='stats4', rname='coef')
    def coef(fit: object) -> object:
        """Estimated parameters of a fit."""
        ...

    assert stats4_loaded()

    fit = mle(robjects_r('function(m) (m - 3) ^ 2'), start={'m': 0.0})
    assert_allclose(coef(fit).item(), 3.0, rtol=1e-4)


def test_rfunction_invalid_names() -> None:
    """Invalid R names are rejected eagerly, at decoration time."""

    def stub() -> object: ...

    with pytest.raises(ValueError, match='Invalid R package name'):
        base.rfunction(stub, library='not-a-package')
    with pytest.raises(ValueError, match='Invalid R function name'):
        base.rfunction(stub, library='base', rname='not a function')


def test_doc_pulled_from_r_when_missing() -> None:
    """A subclass without a docstring gets the R help page as documentation."""

    class Lm(base.RObjectBase):
        _rfuncname = 'stats::lm'

    assert Lm.__doc__ is not None
    assert 'R documentation\n---------------' in Lm.__doc__
    # content from the help page of stats::lm, indented as a literal block
    assert '    Fitting Linear Models' in Lm.__doc__


def test_r_dataframe_converts_to_polars() -> None:
    """With polars installed, R data frames convert to polars rather than pandas.

    The polars converter is summed after the pandas one, so its R-to-Python
    registration for data frames takes precedence; bare vectors are untouched.
    """
    pl = pytest.importorskip('polars')
    rdf = robjects_r('data.frame(a = c(1.0, 2.0, 3.0), b = c(4L, 5L, 6L))')
    out = base.RObjectBase._r2py(rdf)
    assert isinstance(out, pl.DataFrame)
    assert out.columns == ['a', 'b']
    assert out['a'].to_list() == [1.0, 2.0, 3.0]
    # a bare vector still converts to numpy, not a one-column frame
    assert_array_equal(
        base.RObjectBase._r2py(robjects_r('c(1.0, 2.0)')), np.array([1.0, 2.0])
    )


def test_fork_safe_native_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pools are capped at one thread in the context and restored on exit."""
    calls: list[int] = []

    def omp_get_max_threads() -> int:
        return 4

    def omp_set_num_threads(nthreads: int) -> None:
        calls.append(nthreads)

    # expose only the OpenMP pair: the OpenBLAS lookup must be skipped
    handle = SimpleNamespace(
        omp_get_max_threads=omp_get_max_threads, omp_set_num_threads=omp_set_num_threads
    )
    monkeypatch.setattr(ctypes, 'CDLL', lambda name: handle)  # noqa: ARG005

    with fork_safe_native_threads():
        assert calls == [1]
    assert calls == [1, 4]
    assert omp_get_max_threads.restype is ctypes.c_int
    assert omp_set_num_threads.argtypes == (ctypes.c_int,)
