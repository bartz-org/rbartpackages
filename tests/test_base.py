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

"""Tests for the shared wrapper plumbing in `rbartpackages._base`."""

import ctypes
from types import SimpleNamespace

import pytest

from rbartpackages import _base


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

    with _base.fork_safe_native_threads():
        assert calls == [1]
    assert calls == [1, 4]
    assert omp_get_max_threads.restype is ctypes.c_int
    assert omp_set_num_threads.argtypes == (ctypes.c_int,)
