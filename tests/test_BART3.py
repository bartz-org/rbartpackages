# rbartpackages/tests/test_BART3.py
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

"""Tests for the BART3 wrapper."""

import math

import numpy as np
import pytest

from tests.util import assert_close_matrices, import_or_skip

BART3 = import_or_skip('rbartpackages.BART3')

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
    assert 'R documentation' in BART3.gbart.__doc__


def test_gbart_fit(rng: np.random.Generator) -> None:
    """Fit `gbart` and check the output shapes and predictions."""
    x_train, y_train = make_data(rng)
    n, _ = x_train.shape
    bart = BART3.gbart(
        x_train=x_train, y_train=y_train, ntree=NTREE, nskip=NSKIP, ndpost=NDPOST
    )
    assert bart.ndpost == NDPOST
    assert bart.yhat_train.shape == (NDPOST, n)
    assert bart.yhat_train_mean.shape == (n,)

    yhat = bart.predict(x_train)
    assert yhat.shape == (NDPOST, n)
    assert_close_matrices(yhat.mean(axis=0), bart.yhat_train_mean, rtol=1e-5)


def test_gbart_test_data(rng: np.random.Generator) -> None:
    """Passing `x_test` populates the test-set output attributes."""
    x_train, y_train = make_data(rng)
    x_test = rng.standard_normal((7, x_train.shape[1]))
    m, _ = x_test.shape
    bart = BART3.gbart(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        ntree=NTREE,
        nskip=NSKIP,
        ndpost=NDPOST,
    )
    assert bart.x_test.shape == x_test.shape
    assert bart.yhat_test.shape == (NDPOST, m)
    assert bart.yhat_test_mean.shape == (m,)
    assert bart.yhat_test_lower.shape == (m,)
    assert bart.yhat_test_upper.shape == (m,)
    assert np.all(bart.yhat_test_lower <= bart.yhat_test_upper)
    assert isinstance(bart.LPML, float)


def test_gbart_binary(rng: np.random.Generator) -> None:
    """The probit (`type='pbart'`) path exposes the probability attributes."""
    x_train, y_train = make_data(rng)
    n, _ = x_train.shape
    y_bin = (y_train > np.median(y_train)).astype(float)
    x_test = rng.standard_normal((7, x_train.shape[1]))
    m, _ = x_test.shape
    bart = BART3.gbart(
        x_train=x_train,
        y_train=y_bin,
        x_test=x_test,
        type='pbart',
        ntree=NTREE,
        nskip=NSKIP,
        ndpost=NDPOST,
    )
    assert bart.prob_train.shape == (NDPOST, n)
    assert bart.prob_test.shape == (NDPOST, m)
    assert bart.prob_test_lower.shape == (m,)
    assert bart.prob_test_upper.shape == (m,)
    assert np.all(bart.prob_test_lower <= bart.prob_test_upper)
    assert bart.sigest is None  # not estimated for binary outcomes
    assert isinstance(bart.LPML, float)


def test_sigest_is_float(rng: np.random.Generator) -> None:
    """`sigest` is a finite float for the serial `gbart` (continuous outcome)."""
    x_train, y_train = make_data(rng)
    bart = BART3.gbart(
        x_train=x_train, y_train=y_train, ntree=NTREE, nskip=NSKIP, ndpost=NDPOST
    )
    assert isinstance(bart.sigest, float)
    assert math.isfinite(bart.sigest)


@pytest.mark.timeout(180)
def test_mc_gbart_multicore(rng: np.random.Generator) -> None:
    """`mc.gbart` with ``mc_cores > 1`` runs without deadlocking.

    `mc.gbart` forks via `parallel::mcparallel`; GNU libgomp (reached through the
    threaded OpenBLAS that R's LAPACK uses) is not fork-safe and hangs the forked
    children. The wrapper caps native thread pools at one thread across the fork
    to avoid it. Warming up a threaded OpenMP region first (a `predict` call) is
    what makes the deadlock reliable, so this exercises the regression path.
    """
    x_train, y_train = make_data(rng)
    n, _ = x_train.shape

    # Arm the parent's libgomp thread pool, which is what poisons the fork.
    warm = BART3.gbart(
        x_train=x_train, y_train=y_train, ntree=NTREE, nskip=NSKIP, ndpost=NDPOST
    )
    warm.predict(x_train)

    mc_cores = 2
    bart = BART3.mc_gbart(
        x_train=x_train,
        y_train=y_train,
        ntree=NTREE,
        nskip=NSKIP,
        ndpost=NDPOST,
        mc_cores=mc_cores,
    )
    assert bart.chains == mc_cores
    # ndpost is rounded up to a whole number of draws per chain.
    assert bart.ndpost == mc_cores * math.ceil(NDPOST / mc_cores)
    assert bart.yhat_train.shape == (bart.ndpost, n)
    assert bart.yhat_train_mean.shape == (n,)
    # mc.gbart overwrites sigest with its logical-NA default; __init__ -> nan.
    assert isinstance(bart.sigest, float)
    assert math.isnan(bart.sigest)
