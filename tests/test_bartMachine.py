# rbartpackages/tests/test_bartMachine.py
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

"""Tests for the bartMachine wrapper (needs R bartMachine + Java)."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter
from typing import Any

import numpy as np
import pandas as pd
import pytest
from rpy2.rinterface_lib.embedded import RRuntimeError

from rbartpackages import bartMachine
from rbartpackages._src.base import robjects_r
from tests.util import (
    assert_allclose,
    assert_array_equal,
    assert_close_matrices,
    evaluated_r_formals,
    has_var_keyword,
    int_seed,
    kwdict,
    mapped_params,
    nnone,
)

NTREE = 10
NBURN = 20
NPOST = 20


@dataclass(frozen=True)
class Data:
    """A small dataset in the pandas form bartMachine wants."""

    x: pd.DataFrame
    """Predictors; bartMachine requires a data frame."""

    y: pd.Series
    """Numeric outcomes, the first predictor plus noise.

    A Series rather than a numpy array, though the wrapper now accepts either
    (see `test_numpy_response`); ``.to_numpy()`` recovers the array form.
    """

    x_test: pd.DataFrame
    """Test-set predictors."""

    @property
    def labels(self) -> pd.Series:
        """`y` binarized at its median, as a categorical for classification fits."""
        return pd.Series(pd.Categorical(np.where(self.y > self.y.median(), 'b', 'a')))


@pytest.fixture
def data(rng: np.random.Generator) -> Data:
    """Generate a small regression dataset."""
    n, m, p = 40, 7, 3
    columns = [f'x{i}' for i in range(p)]
    x = pd.DataFrame(rng.standard_normal((n, p)), columns=columns)
    y = x['x0'] + 0.1 * rng.standard_normal(n)
    x_test = pd.DataFrame(rng.standard_normal((m, p)), columns=columns)
    return Data(x, y, x_test)


def fit(
    data: Data, rng: np.random.Generator, *, classification: bool = False, **kw: Any
) -> bartMachine.bartMachine:
    """Fit a small bartMachine model on `data`."""
    bartMachine.set_bart_machine_num_cores(1)
    return bartMachine.bartMachine(
        X=data.x,
        y=data.labels if classification else data.y,
        num_trees=NTREE,
        num_burn_in=NBURN,
        num_iterations_after_burn_in=NPOST,
        seed=int_seed(rng),
        verbose=False,
        **kw,
    )


def test_docstring() -> None:
    """The R documentation is attached to the wrapper class."""
    assert 'R documentation' in bartMachine.bartMachine.__doc__


def test_num_cores() -> None:
    """The thread count round-trips through the package-global setting."""
    initial = bartMachine.bart_machine_num_cores()
    try:
        bartMachine.set_bart_machine_num_cores(2)
        assert bartMachine.bart_machine_num_cores() == 2
    finally:
        bartMachine.set_bart_machine_num_cores(initial)
    assert bartMachine.bart_machine_num_cores() == initial


def check_common_attributes(bm: bartMachine.bartMachine, data: Data) -> None:
    """Check the attributes shared by regression and classification fits."""
    n, p = data.x.shape

    # R scalars become Python scalars of the right type
    assert (bm.n, bm.p) == (n, p)
    assert bm.num_trees == NTREE
    assert bm.num_burn_in == NBURN
    assert bm.num_iterations_after_burn_in == NPOST
    assert bm.num_gibbs == NBURN + NPOST
    assert bm.num_cores == 1
    assert isinstance(bm.num_rand_samps_in_library, int)
    assert isinstance(bm.seed, int)
    for name in ('alpha', 'beta', 'k', 'nu', 'prob_rule_class', 'q'):
        assert isinstance(getattr(bm, name), float)
    for name in ('mem_cache_for_speed', 'run_in_sample', 'use_missing_data'):
        assert isinstance(getattr(bm, name), bool)
    assert bm.s_sq_y == 'mse'
    assert isinstance(bm.time_to_build, float)
    assert bm.time_to_build > 0

    # the data is echoed back, plus the preprocessed model matrix
    assert list(bm.X.columns) == list(data.x.columns)
    assert_close_matrices(bm.X.to_numpy(), data.x.to_numpy())
    assert bm.model_matrix_training_data.shape == (n, p + 1)
    assert_close_matrices(bm.model_matrix_training_data[:, :p], data.x.to_numpy())
    assert list(bm.training_data_features) == list(data.x.columns)
    assert_close_matrices(bm.mh_prob_steps, np.array([2.5, 2.5, 4]) / 9, rtol=1e-7)

    # optional inputs that were not given are None
    assert bm.cov_prior_vec is None
    assert bm.interaction_constraints is None


def test_regression(data: Data, rng: np.random.Generator) -> None:
    """Fit a regression and check the attributes, `predict` and the module functions."""
    n, p = data.x.shape
    m, _ = data.x_test.shape
    bm = fit(data, rng)

    check_common_attributes(bm, data)
    assert bm.pred_type == 'regression'
    assert isinstance(bm.sig_sq_est, float)
    assert_close_matrices(bm.model_matrix_training_data[:, p], data.y.to_numpy())
    assert_close_matrices(bm.y, data.y.to_numpy())

    # in-sample outputs
    assert bm.y_hat_train is not None
    assert bm.residuals is not None
    assert bm.L1_err_train is not None
    assert bm.L2_err_train is not None
    assert bm.rmse_train is not None
    assert bm.y_hat_train.shape == (n,)
    assert_close_matrices(bm.residuals, data.y.to_numpy() - bm.y_hat_train, rtol=1e-7)
    assert_allclose(bm.L1_err_train, np.abs(bm.residuals).sum(), rtol=1e-7)
    assert_allclose(bm.L2_err_train, np.square(bm.residuals).sum(), rtol=1e-7)
    assert_allclose(bm.rmse_train, math.sqrt(bm.L2_err_train / n), rtol=1e-7)
    assert isinstance(bm.PseudoRsq, float)

    # classification-only outputs are unset
    assert bm.y_levels is None
    assert bm.p_hat_train is None
    assert bm.confusion_matrix is None
    assert bm.misclassification_error is None

    # predict and bart_machine_get_posterior match the in-sample outputs and
    # each other
    yhat = bm.predict(data.x)
    assert yhat.shape == (n,)
    assert_close_matrices(yhat, bm.y_hat_train, rtol=1e-7)
    post = bartMachine.bart_machine_get_posterior(bm, data.x_test, verbose=False)
    assert sorted(post) == ['X', 'y_hat', 'y_hat_posterior_samples']
    assert post['X'].shape == (m, p)
    assert post['y_hat_posterior_samples'].shape == (m, NPOST)
    assert_close_matrices(
        post['y_hat'], post['y_hat_posterior_samples'].mean(axis=1), rtol=1e-7
    )
    assert_close_matrices(post['y_hat'], bm.predict(data.x_test), rtol=1e-7)

    # error-variance draws, with and without burn-in
    sigsqs = bartMachine.get_sigsqs(bm)
    assert sigsqs.shape == (NPOST,)
    assert np.all(sigsqs > 0)
    # without after_burn_in the pre-MCMC initial value is kept as first entry
    all_sigsqs = bartMachine.get_sigsqs(bm, after_burn_in=False)
    assert all_sigsqs.shape == (1 + NBURN + NPOST,)
    assert_array_equal(all_sigsqs[-NPOST:], sigsqs)


def test_classification(data: Data, rng: np.random.Generator) -> None:
    """Fit a classification and check the attributes and prediction functions."""
    n, p = data.x.shape
    m, _ = data.x_test.shape
    labels = data.labels
    bm = fit(data, rng, classification=True)

    check_common_attributes(bm, data)
    assert bm.pred_type == 'classification'
    assert list(nnone(bm.y_levels)) == ['a', 'b']  # alphabetical; first is target
    assert list(bm.y) == list(labels)
    # the response is encoded as 1 for the first (target) level
    assert_close_matrices(
        bm.model_matrix_training_data[:, p],
        (labels == nnone(bm.y_levels)[0]).to_numpy().astype(float),
    )

    # in-sample outputs
    p_hat_train = nnone(bm.p_hat_train)
    y_levels = nnone(bm.y_levels)
    assert p_hat_train.shape == (n,)
    assert np.all((p_hat_train >= 0) & (p_hat_train <= 1))
    expected_labels = np.where(
        p_hat_train > bm.prob_rule_class, y_levels[0], y_levels[1]
    )
    assert_array_equal(nnone(bm.y_hat_train), expected_labels, strict=False)
    assert nnone(bm.confusion_matrix).shape == (3, 3)
    assert isinstance(bm.misclassification_error, float)

    # regression-only outputs are unset
    assert bm.sig_sq_est is None
    assert bm.y_hat_train is not None
    for name in (
        'L1_err_train',
        'L2_err_train',
        'PseudoRsq',
        'residuals',
        'rmse_train',
    ):
        assert getattr(bm, name) is None

    # predict returns probabilities by default, labels with type='class'
    p_hat = bm.predict(data.x, verbose=False)
    assert p_hat.shape == (n,)
    assert_close_matrices(p_hat, p_hat_train, rtol=1e-7)
    label_pred = bm.predict(data.x_test, type='class', verbose=False)
    assert label_pred.shape == (m,)
    assert set(label_pred) <= set(y_levels)

    # bart_machine_get_posterior draws are probabilities
    post = bartMachine.bart_machine_get_posterior(bm, data.x_test, verbose=False)
    samples = post['y_hat_posterior_samples']
    assert samples.shape == (m, NPOST)
    assert np.all((samples >= 0) & (samples <= 1))
    assert_close_matrices(
        post['y_hat'], bm.predict(data.x_test, verbose=False), rtol=1e-7
    )

    # there is no error variance for classification
    with pytest.raises(RRuntimeError, match='no sigsq'):
        bartMachine.get_sigsqs(bm)


def test_no_in_sample(data: Data, rng: np.random.Generator) -> None:
    """``run_in_sample=False`` leaves the in-sample outputs unset."""
    n, _ = data.x.shape
    bm = fit(data, rng, run_in_sample=False)
    assert bm.run_in_sample is False
    for name in (
        'L1_err_train',
        'L2_err_train',
        'PseudoRsq',
        'residuals',
        'rmse_train',
        'y_hat_train',
    ):
        assert getattr(bm, name) is None
    # predict still works
    assert bm.predict(data.x).shape == (n,)


def test_numpy_response(data: Data, rng: np.random.Generator) -> None:
    """`y` can be a numpy array: numeric for regression, string for classification.

    The wrapper builds a bare R numeric vector or factor directly, sidestepping
    the ``dim`` that rpy2's numpy bridge attaches (and that bartMachine rejects).
    """
    common: kwdict = dict(
        num_trees=NTREE,
        num_burn_in=NBURN,
        num_iterations_after_burn_in=NPOST,
        seed=int_seed(rng),
        verbose=False,
    )
    bartMachine.set_bart_machine_num_cores(1, verbose=False)

    reg = bartMachine.bartMachine(X=data.x, y=data.y.to_numpy(), **common)
    assert reg.pred_type == 'regression'
    assert_close_matrices(reg.y, data.y.to_numpy())

    # a plain numpy string array becomes a factor (alphabetical levels)
    labels = data.labels.to_numpy().astype(str)
    clf = bartMachine.bartMachine(X=data.x, y=labels, **common)
    assert clf.pred_type == 'classification'
    assert list(nnone(clf.y_levels)) == ['a', 'b']


def test_xy_interface(data: Data, rng: np.random.Generator) -> None:
    """`Xy` bundles the predictors and the response in one data frame.

    The response goes in a column named ``'y'``; it is an alternative to
    passing `X` and `y` separately.
    """
    n, p = data.x.shape
    bartMachine.set_bart_machine_num_cores(1, verbose=False)
    bm = bartMachine.bartMachine(
        Xy=data.x.assign(y=data.y),
        num_trees=NTREE,
        num_burn_in=NBURN,
        num_iterations_after_burn_in=NPOST,
        seed=int_seed(rng),
        verbose=False,
    )
    assert (bm.n, bm.p) == (n, p)
    assert bm.pred_type == 'regression'
    assert list(bm.X.columns) == list(data.x.columns)
    assert_close_matrices(bm.y, data.y.to_numpy())


def test_optional_inputs(data: Data, rng: np.random.Generator) -> None:
    """User-given optional inputs come back converted to Python values."""
    _, p = data.x.shape
    cov_prior_vec = np.arange(1.0, 1.0 + p)
    bm = fit(
        data,
        rng,
        cov_prior_vec=cov_prior_vec,
        # 1-based column indices; named entries because rpy2 converts dicts,
        # not lists, to the R list bartMachine wants
        interaction_constraints={'a': np.array([1, 2]), 'b': np.array([3])},
    )
    assert_array_equal(nnone(bm.cov_prior_vec), cov_prior_vec)
    # the constraints come back as a tuple of 0-based column indices
    assert isinstance(bm.interaction_constraints, tuple)
    first, second = bm.interaction_constraints
    assert_array_equal(first, np.array([0.0, 1.0]))
    assert_array_equal(second, np.array([2.0]))


# the wrapper callables, their R name, and the R arguments deliberately left
# unexposed (only the private `covariates_to_permute` of the constructor)
SIGNATURE_CASES = [
    (bartMachine.bartMachine, 'bartMachine::bartMachine', {'covariates_to_permute'}),
    (bartMachine.get_sigsqs, 'bartMachine::get_sigsqs', set()),
    (
        bartMachine.bart_machine_get_posterior,
        'bartMachine::bart_machine_get_posterior',
        set(),
    ),
    (
        bartMachine.set_bart_machine_num_cores,
        'bartMachine::set_bart_machine_num_cores',
        set(),
    ),
]


@pytest.mark.parametrize(
    ('obj', 'rfuncname', 'unexposed'),
    SIGNATURE_CASES,
    ids=[name for _, name, _ in SIGNATURE_CASES],
)
def test_signature_defaults_match_r(
    obj: Callable, rfuncname: str, unexposed: set[str]
) -> None:
    """The explicit signatures stay in sync with the wrapped R functions.

    Every literal default in a Python signature must match its R counterpart,
    every R argument must be either exposed or deliberately unexposed, and
    none of these functions take R's ``...``, so that an upstream update that
    changes a default or adds an argument fails here instead of silently
    diverging.
    """
    params = mapped_params(obj)
    rnames = set(robjects_r(f'names(formals({rfuncname}))'))
    assert '...' not in rnames, rfuncname
    assert not has_var_keyword(obj), rfuncname
    assert params.keys() <= rnames, rfuncname
    assert rnames - params.keys() == unexposed, rfuncname

    rdefaults = evaluated_r_formals(rfuncname)
    for name, param in params.items():
        if param.default is Parameter.empty or param.default is None:
            continue  # required, or deferred to R
        # a literal Python default needs a comparable R default
        assert name in rdefaults, f'{rfuncname}, argument {name}'
        # strict=False: R types its literals loosely (TRUE vs 1, 100L vs 100),
        # so compare values only
        assert_array_equal(
            np.ravel(param.default),
            np.ravel(rdefaults[name]),
            strict=False,
            err_msg=f'{rfuncname}, argument {name}',
        )


def test_predict_signature_matches_r() -> None:
    """The explicit `predict` signature tracks the R ``predict.bartMachine`` method.

    Every Python argument must appear in the S3 method's formals (minus the
    `object`/`new_data` the wrapper fills itself), R's ignored ``...`` is left
    unexposed, and the defaults defer to R with ``None``.
    """
    method = 'getS3method("predict", "bartMachine", envir = asNamespace("bartMachine"))'
    rnames = set(robjects_r(f'names(formals({method}))')) - {'object', 'new_data'}
    params = mapped_params(bartMachine.bartMachine.predict, skip={'new_data'})
    assert params.keys() <= rnames
    assert rnames - params.keys() == {'...'}
    for name, param in params.items():
        assert param.default is None, name


def test_constructor_rejects_unknown_arguments() -> None:
    """Arguments outside the explicit constructor signature fail before reaching R.

    `bartMachine` has no R ``...``, so its explicit signature replaces it: a
    misspelled or package-foreign argument fails as a `TypeError` instead of
    being silently swallowed.
    """
    with pytest.raises(TypeError, match='unexpected keyword'):
        bartMachine.bartMachine(num_tree=10)  # misspelled num_trees
