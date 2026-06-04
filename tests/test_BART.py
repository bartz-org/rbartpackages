# rbartpackages/tests/test_BART.py
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

"""Tests for the BART wrapper."""

import math

import numpy as np
import pytest

from tests.util import assert_array_equal, assert_close_matrices, import_or_skip

BART = import_or_skip('rbartpackages.BART')

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
    assert 'R documentation' in BART.gbart.__doc__


def test_gbart_no_test_data(rng: np.random.Generator) -> None:
    """Without `x_test` the derived test-set attributes are left unset.

    The fit also has a constant column (dropped, reported through `rm_const`).
    """
    x_train, y_train = make_data(rng)
    n, p = x_train.shape
    # R reports the dropped constant column as a negative rm.const index; the
    # wrapper turns that into the kept 0-based column indices
    x_train = np.insert(x_train, 1, 1.0, axis=1)
    bart = BART.gbart(
        x_train=x_train, y_train=y_train, ntree=NTREE, nskip=0, ndpost=NDPOST
    )
    assert bart.ndpost == NDPOST
    assert bart.yhat_train.shape == (NDPOST, n)
    assert bart.yhat_train_mean.shape == (n,)
    assert_array_equal(bart.rm_const, np.array([0, 2, 3], np.int32))
    assert bart.varcount.shape == (NDPOST, p)

    yhat = bart.predict(x_train[:, bart.rm_const])
    assert yhat.shape == (NDPOST, n)
    assert_close_matrices(yhat.mean(axis=0), bart.yhat_train_mean, rtol=1e-5)

    assert bart.yhat_test_mean is None


@pytest.mark.parametrize('binary', [False, True], ids=['continuous', 'binary'])
def test_gbart_test_data(rng: np.random.Generator, binary: bool) -> None:
    """Passing `x_test` populates the test-set outputs.

    The continuous (`wbart`) and binary (`pbart`) paths expose different
    attributes and `predict` return types.
    """
    x_train, y_train = make_data(rng)
    n, p = x_train.shape
    x_test = rng.standard_normal((7, p))
    m, _ = x_test.shape
    if binary:
        y_train = (y_train > np.median(y_train)).astype(float)
    bart = BART.gbart(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        type='pbart' if binary else 'wbart',
        ntree=NTREE,
        nskip=NSKIP,
        ndpost=NDPOST,
    )
    assert bart.ndpost == NDPOST
    assert bart.yhat_train.shape == (NDPOST, n)
    assert bart.yhat_test.shape == (NDPOST, m)

    if binary:
        assert bart.prob_train.shape == (NDPOST, n)
        assert bart.prob_train_mean.shape == (n,)
        assert bart.prob_test.shape == (NDPOST, m)
        assert bart.prob_test_mean.shape == (m,)

        # R's predict for binary fits returns a list; the wrapper exposes it as
        # a dict of arrays (continuous fits return a bare matrix instead).
        # Unlike BART3, there are no prob.test.lower/upper quantiles.
        pred = bart.predict(x_test)
        assert isinstance(pred, dict)
        expected_keys = ['binaryOffset', 'prob_test', 'prob_test_mean', 'yhat_test']
        assert sorted(pred) == expected_keys
        assert pred['yhat_test'].shape == (NDPOST, m)
        assert pred['prob_test'].shape == (NDPOST, m)
        assert pred['prob_test_mean'].shape == (m,)
        assert isinstance(pred['binaryOffset'], float)
    else:
        assert bart.yhat_train_mean.shape == (n,)
        assert bart.yhat_test_mean.shape == (m,)
        assert bart.prob_train is None
        assert bart.prob_test is None

        pred = bart.predict(x_test)
        assert pred.shape == (NDPOST, m)
        assert_close_matrices(pred.mean(axis=0), bart.yhat_test_mean, rtol=1e-5)


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
    warm = BART.gbart(
        x_train=x_train, y_train=y_train, ntree=NTREE, nskip=NSKIP, ndpost=NDPOST
    )
    warm.predict(x_train)

    mc_cores = 2
    bart = BART.mc_gbart(
        x_train=x_train,
        y_train=y_train,
        ntree=NTREE,
        nskip=NSKIP,
        ndpost=NDPOST,
        mc_cores=mc_cores,
    )
    # ndpost is rounded up to a whole number of draws per chain.
    assert bart.ndpost == mc_cores * math.ceil(NDPOST / mc_cores)
    assert bart.yhat_train.shape == (bart.ndpost, n)

    # predict with mc_cores > 1 forks via mc.pwbart; unlike mc.gbart's fork the
    # children run single-threaded, so this stays deadlock-free without a guard.
    yhat = bart.predict(x_train, **{'mc.cores': mc_cores})
    assert yhat.shape == (bart.ndpost, n)
