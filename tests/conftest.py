# rbartpackages/tests/conftest.py
#
# Copyright (c) 2024-2026, The rbartpackages Contributors
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

"""Pytest configuration."""

from re import fullmatch
from sys import modules

import numpy as np
import pytest

from tests.util import int_seed, nnone


@pytest.fixture
def rng(request: pytest.FixtureRequest) -> np.random.Generator:
    """Return a deterministic per-test-case numpy random generator."""
    nodeid = request.node.nodeid
    # exclude xdist_group suffixes because they are active only under xdist
    match = fullmatch(r'(.+?\.py::.+?(\[.+?\])?)(@.+)?', nodeid)
    nodeid = nnone(match).group(1)
    seed = np.array([nodeid], np.bytes_).view(np.uint8)
    return np.random.default_rng(seed)


@pytest.fixture(autouse=True)
def seed_r(rng: np.random.Generator) -> None:
    """Seed the global R rng deterministically per test case.

    Skipped if rpy2 is not loaded, to keep R-free tests R-free; tests that use
    R load rpy2 at import time, so they are always covered.
    """
    if 'rpy2.robjects' in modules:
        # deferred to keep R optional
        from rbartpackages._src.base import robjects_r  # noqa: PLC0415

        robjects_r['set.seed'](int_seed(rng))
