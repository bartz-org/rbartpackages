# rbartpackages/tests/test_dbarts.py
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

"""Tests for the dbarts wrapper."""

import numpy as np

from tests.util import import_or_skip

dbarts = import_or_skip('rbartpackages.dbarts')

NDPOST = 20
NSKIP = 20
NTREE = 10


def make_data(rng: np.random.Generator, n: int = 30, p: int = 3) -> tuple:
    """Generate a small regression dataset."""
    x_train = rng.standard_normal((n, p))
    y_train = x_train[:, 0] + 0.1 * rng.standard_normal(n)
    return x_train, y_train


def test_docstring() -> None:
    """The R documentation is attached to the wrapper class."""
    assert 'R documentation' in dbarts.bart.__doc__


def test_bart_fit(rng: np.random.Generator) -> None:
    """Fit `bart` and check the training-fit output shape."""
    x_train, y_train = make_data(rng)
    n, _ = x_train.shape
    bart = dbarts.bart(
        x_train=x_train,
        y_train=y_train,
        ntree=NTREE,
        nskip=NSKIP,
        ndpost=NDPOST,
        verbose=False,
    )
    assert bart.yhat_train.shape == (NDPOST, n)
