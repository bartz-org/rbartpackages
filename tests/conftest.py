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
from jaxtyping import __version__ as jaxtyping_version
from jaxtyping import install_import_hook
from packaging.version import Version

from tests.util import int_seed, nnone

# Check the jaxtyping annotations (dtypes and shapes) against the actual values
# at runtime. The hook rewrites the wrappers as they are imported, so it must
# run before any test module pulls one in; conftest is loaded first, and none of
# the imports above reach `rbartpackages`.
#
# WORKAROUND(jaxtyping<0.3.11): the hook honours `@no_type_check` only from
# 0.3.11 on. Below that it also decorates the methods returning `Self`, which
# beartype rejects because it resolves `Self` from the class, and importing the
# wrappers fails outright. Skipping the checks on the oldest supported
# environment beats raising the floor of a runtime dependency for a test-only
# feature; the checks still run everywhere else.
if Version(jaxtyping_version) >= Version('0.3.11'):
    install_import_hook('rbartpackages', 'beartype.beartype')


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
