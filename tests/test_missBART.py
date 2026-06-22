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
from inspect import Parameter
from typing import Any

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray

from rbartpackages import missBART
from rbartpackages._src.base import robjects_r
from tests.util import (
    assert_array_equal,
    evaluated_r_formals,
    has_var_keyword,
    int_seed,
    mapped_params,
    nnone,
)

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


def fit_missBART2(
    data: Data, rng: np.random.Generator, **kw: Any
) -> missBART.missBART2:
    """Fit `missBART2` on `data` with small, fast MCMC settings."""
    robjects_r['set.seed'](int_seed(rng))
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
    assert nnone(fit.new_y_post).shape == (ITERS, m, p)
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


def test_predict_without_x_predict(rng: np.random.Generator) -> None:
    """Explicit `predict=True` without `x_predict` raises an error."""
    data = make_data(rng, p=1)
    with pytest.raises(ValueError, match='x_predict'):
        fit_missBART2(data, rng, predict=True)


def test_forwards_hyperparameter_kwargs(rng: np.random.Generator) -> None:
    """Extra keyword arguments reach R's ``...`` (``tree_list`` / ``hypers_list``).

    ``df`` and ``prior_alpha`` are not `missBART2` formals, so they can only
    reach R through its ``...``; the fit completing proves they were forwarded.
    """
    data = make_data(rng, p=1)
    fit = fit_missBART2(data, rng, df=5, prior_alpha=0.9)
    assert fit.y_post.shape == (ITERS, *data.y.shape)


def test_explicit_tree_prior_params(rng: np.random.Generator) -> None:
    """`tree_prior_params` accepts a complete R list passed as a dict."""
    data = make_data(rng, p=1)
    fit = fit_missBART2(
        data,
        rng,
        tree_prior_params=dict(
            prior_alpha=0.95, prior_beta=2.0, min_node=1, max_attempt=1
        ),
    )
    assert fit.y_post.shape == (ITERS, *data.y.shape)


# vestigial R arguments accepted but never used by the missBART2 implementation,
# so deliberately not exposed (they stay reachable through **hyperparams)
UNEXPOSED = {
    'true_trees_data',
    'true_trees_missing',
    'true_change_points',
    'true_change_points_miss',
}


def test_signature_defaults_match_r() -> None:
    """The explicit `missBART2` signature stays in sync with the R function.

    Every literal default in the Python signature must match its R counterpart,
    every R argument must be exposed or deliberately unexposed, and R's ``...``
    must be forwarded by a ``**kwargs`` catch-all, so that an upstream update
    that changes a default or adds an argument fails here instead of silently
    diverging.
    """
    rfuncname = 'missBART::missBART2'
    params = mapped_params(missBART.missBART2)
    rnames = set(robjects_r(f'names(formals({rfuncname}))'))
    # R's `...` is forwarded by a **kwargs catch-all, not a named parameter
    assert ('...' in rnames) == has_var_keyword(missBART.missBART2)
    rnames -= {'...'}
    assert params.keys() <= rnames
    assert rnames - params.keys() == UNEXPOSED

    rdefaults = evaluated_r_formals(rfuncname)
    for name, param in params.items():
        if param.default is Parameter.empty or param.default is None:
            continue  # required, or deferred to R
        # a literal Python default needs a comparable R default
        assert name in rdefaults, name
        # strict=False: R types its literals loosely (TRUE vs 1, 2L vs 2), so
        # compare values only
        assert_array_equal(
            np.ravel(param.default),
            np.ravel(rdefaults[name]),
            strict=False,
            err_msg=name,
        )
