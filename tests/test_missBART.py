# rbartpackages/tests/test_missBART.py
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

"""Tests for the missBART wrapper."""

from dataclasses import dataclass

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray
from rpy2 import robjects

from rbartpackages import missBART
from tests.util import assert_array_equal, int_seed

N_TREES = 2
BURN = 3
ITERS = 4
THIN = 2
TOTAL_ITERS = BURN + THIN * ITERS

TREE_FIELDS = (
    'parent',
    'lower',
    'upper',
    'split_variable',
    'split_value',
    'depth',
    'direction',
    'NA_direction',
)


@dataclass(frozen=True)
class Data:
    """A small regression dataset with missing outcome entries."""

    x: Float64[ndarray, 'n q']
    """Predictors."""

    y: Float64[ndarray, 'n p']
    """Outcomes, a noisy linear function of `x`, with `n_missing` NaN entries."""

    x_predict: Float64[ndarray, 'm q']
    """Out-of-sample predictors."""

    n_missing: int
    """Number of NaN entries in `y`."""


def make_data(rng: np.random.Generator, p: int) -> Data:
    """Generate a small regression dataset with missing outcome entries."""
    n, q, m, n_missing = 25, 2, 5, 3
    x = rng.standard_normal((n, q))
    y = x @ rng.standard_normal((q, p)) + 0.1 * rng.standard_normal((n, p))
    missing = rng.choice(y.size, n_missing, replace=False)
    y.ravel()[missing] = np.nan
    x_predict = rng.standard_normal((m, q))
    return Data(x, y, x_predict, n_missing)


def fit_missBART2(data: Data, rng: np.random.Generator, **kw: object) -> object:
    """Fit `missBART2` on `data` with small, fast MCMC settings."""
    robjects.r['set.seed'](int_seed(rng))
    return missBART.missBART2(
        data.x,
        data.y,
        n_reg_trees=N_TREES,
        n_class_trees=N_TREES,
        burn=BURN,
        iters=ITERS,
        thin=THIN,
        mice_impute=False,
        show_progress=False,
        **kw,
    )


def test_docstring() -> None:
    """The R documentation is attached to the wrapper class."""
    assert 'R documentation' in missBART.missBART2.__doc__


@pytest.mark.parametrize('p', [1, 2])
def test_fit(rng: np.random.Generator, p: int) -> None:
    """Fit with out-of-sample predictions and check the output attributes."""
    data = make_data(rng, p)
    n, _ = data.x.shape
    m, _ = data.x_predict.shape
    fit = fit_missBART2(data, rng, x_predict=data.x_predict)

    # MCMC settings echoed back as scalars
    assert fit.burn == BURN
    assert fit.iters == ITERS
    assert fit.thin == THIN
    assert fit.MH_sd == 0.5 / p

    # data passed through
    assert_array_equal(fit.x, data.x)
    assert_array_equal(fit.min_y, np.nanmin(data.y, axis=0))
    assert_array_equal(fit.max_y, np.nanmax(data.y, axis=0))

    # posterior draws
    assert fit.y_post.shape == (ITERS, n, p)
    assert np.isfinite(fit.y_post).all()
    assert fit.z_post.shape == (ITERS, n, p)
    if p == 1:
        assert fit.omega_post.shape == (ITERS, 1, 1)
    else:
        # with scale=True (the default) only the variances are kept
        assert fit.omega_post.shape == (ITERS, p)
    assert (fit.omega_post > 0).all()
    assert fit.y_impute.shape == (ITERS, data.n_missing)
    assert np.isfinite(fit.y_impute).all()
    assert fit.new_y_post.shape == (ITERS, m, p)
    assert fit.y_miss_accept.shape == (TOTAL_ITERS, data.n_missing)
    assert fit.y_miss_accept.dtype == bool

    # trees and leaf parameters
    assert len(fit.reg_trees) == ITERS
    assert len(fit.class_trees) == ITERS
    for trees in (*fit.reg_trees, *fit.class_trees):
        assert len(trees) == N_TREES
        for tree in trees:
            assert tree.dtype.names == TREE_FIELDS
    assert len(fit.reg_mu) == TOTAL_ITERS
    assert len(fit.class_mu) == TOTAL_ITERS
    for mu in (*fit.reg_mu, *fit.class_mu):
        assert len(mu) == N_TREES
        for leaves in mu:
            n_leaves, p_ = leaves.shape
            assert n_leaves >= 1
            assert p_ == p

    # per-retained-iteration variable importance
    assert len(fit.var_imp) == ITERS

    # disabled outputs
    assert fit.pdp_out is None
    assert fit.y_pred == []


def test_missing_x(rng: np.random.Generator) -> None:
    """Missing values in `x` augment it with missingness indicator columns.

    Without `x_predict` there are no out-of-sample predictions.
    """
    data = make_data(rng, p=1)
    _, q = data.x.shape
    data.x[3, 1] = np.nan
    fit = fit_missBART2(data, rng)

    assert_array_equal(fit.x[:, :q], data.x)
    assert_array_equal(fit.x[:, q:], (~np.isnan(data.x)).astype(float))
    assert fit.new_y_post is None
