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
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from rpy2.rinterface_lib.embedded import RRuntimeError

from rbartpackages import bartMachine
from tests.util import (
    assert_allclose,
    assert_array_equal,
    assert_close_matrices,
    int_seed,
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

    A Series (not a numpy array) so it converts to a plain R atomic vector;
    numpy2ri gives even 1-D arrays a ``dim``, which bartMachine rejects.
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
    data: Data, rng: np.random.Generator, *, classification: bool = False, **kw: object
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
    assert list(bm.y_levels) == ['a', 'b']  # alphabetical; the first is the target
    assert list(bm.y) == list(labels)
    # the response is encoded as 1 for the first (target) level
    assert_close_matrices(
        bm.model_matrix_training_data[:, p],
        (labels == bm.y_levels[0]).to_numpy().astype(float),
    )

    # in-sample outputs
    assert bm.p_hat_train.shape == (n,)
    assert np.all((bm.p_hat_train >= 0) & (bm.p_hat_train <= 1))
    expected_labels = np.where(
        bm.p_hat_train > bm.prob_rule_class, bm.y_levels[0], bm.y_levels[1]
    )
    assert_array_equal(bm.y_hat_train, expected_labels, strict=False)
    assert bm.confusion_matrix.shape == (3, 3)
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
    assert_close_matrices(p_hat, bm.p_hat_train, rtol=1e-7)
    label_pred = bm.predict(data.x_test, type='class', verbose=False)
    assert label_pred.shape == (m,)
    assert set(label_pred) <= set(bm.y_levels)

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
    assert_array_equal(bm.cov_prior_vec, cov_prior_vec)
    # the constraints come back as a tuple of 0-based column indices
    assert isinstance(bm.interaction_constraints, tuple)
    first, second = bm.interaction_constraints
    assert_array_equal(first, np.array([0.0, 1.0]))
    assert_array_equal(second, np.array([2.0]))
