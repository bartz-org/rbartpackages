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

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from jaxtyping import Float64
from numpy import ndarray

from rbartpackages import dbarts
from tests.util import assert_array_equal, assert_close_matrices

NDPOST = 20
NSKIP = 20
NTREE = 10


def phi(x: Float64[ndarray, '...']) -> Float64[ndarray, '...']:
    """Apply the standard normal cumulative distribution function."""
    return (1 + np.vectorize(math.erf)(x / math.sqrt(2))) / 2


@dataclass(frozen=True)
class Data:
    """A small regression dataset."""

    x: Float64[ndarray, 'n p']
    """Predictors."""

    y: Float64[ndarray, ' n']
    """Outcomes, the first predictor plus noise."""

    x_test: Float64[ndarray, 'm p']
    """Test-set predictors."""

    @property
    def biny(self) -> Float64[ndarray, ' n']:
        """Outcomes binarized at their median, for binary-outcome fits."""
        return (self.y > np.median(self.y)).astype(float)

    @property
    def frame(self) -> pd.DataFrame:
        """`x` (columns ``x1 .. xp``) and `y` as a data frame, for the formula interfaces."""
        columns = {f'x{i}': c for i, c in enumerate(self.x.T, 1)}
        return pd.DataFrame(dict(columns, y=self.y))

    @property
    def test_frame(self) -> pd.DataFrame:
        """`x_test` as a data frame with the same columns as `frame`."""
        return pd.DataFrame({f'x{i}': c for i, c in enumerate(self.x_test.T, 1)})


@pytest.fixture
def data(rng: np.random.Generator) -> Data:
    """Generate a small regression dataset."""
    n, p = 30, 3
    x = rng.standard_normal((n, p))
    y = x[:, 0] + 0.1 * rng.standard_normal(n)
    x_test = rng.standard_normal((7, p))
    return Data(x, y, x_test)


def test_docstring() -> None:
    """The R documentation is attached to the wrapper classes."""
    classes = (dbarts.bart, dbarts.bart2, dbarts.rbart_vi, dbarts.dbarts)
    for cls in (*classes, dbarts.dbartsControl):
        assert 'R documentation' in cls.__doc__


def check_generics(bart: dbarts.bart, data: Data, binary: bool) -> None:
    """Check `predict`, `extract`, and `fitted` against the fit's own draws.

    The generics return expected values: probabilities for binary fits (the
    `yhat_*` attributes stay on the latent probit scale), the function draws
    for continuous ones.
    """
    m, _ = data.x_test.shape
    pred = bart.predict(data.x_test)
    assert pred.shape == (NDPOST, m)
    # the kept trees evaluated at x_test reproduce the fit's test draws
    latent = bart.predict(data.x_test, type='bart')
    assert_close_matrices(latent, bart.yhat_test, rtol=1e-7)
    if binary:
        assert np.all((pred > 0) & (pred < 1))
        assert_close_matrices(pred, phi(latent), rtol=1e-7)
    else:
        assert_array_equal(pred, latent)  # 'ev' and 'bart' agree

    draws = bart.extract()  # training draws, expected-value scale
    if binary:
        assert_close_matrices(draws, phi(bart.yhat_train), rtol=1e-7)
    else:
        assert_array_equal(draws, bart.yhat_train)
    assert_close_matrices(bart.fitted(), draws.mean(axis=0), rtol=1e-7)

    trees = bart.extract(type='trees')
    assert isinstance(trees, pd.DataFrame)
    assert {'sample', 'tree', 'n', 'var', 'value'} <= set(trees.columns)


@pytest.mark.parametrize('keeptrees', [False, True], ids=['no-trees', 'keeptrees'])
@pytest.mark.parametrize('binary', [False, True], ids=['continuous', 'binary'])
def test_bart(data: Data, binary: bool, keeptrees: bool) -> None:
    """Fit `bart` with test data and check the fit's outputs and generics.

    Binary (probit) fits drop the error-SD and derived-mean outputs and
    report the latent-scale offset instead; R fills the inapplicable list
    components with NULL, which the wrapper exposes as None. `keeptrees`
    retains the sampler, enabling the generics and tree extraction checked
    in `check_generics`.
    """
    n, p = data.x.shape
    m, _ = data.x_test.shape
    bart = dbarts.bart(
        x_train=data.x,
        y_train=data.biny if binary else data.y,
        x_test=data.x_test,
        ntree=NTREE,
        nskip=NSKIP,
        ndpost=NDPOST,
        keeptrees=keeptrees,
        verbose=False,
    )

    assert bart.yhat_train.shape == (NDPOST, n)
    assert bart.yhat_test.shape == (NDPOST, m)
    assert bart.varcount.shape == (NDPOST, p)
    assert bart.varcount.dtype == np.int32
    assert bart.k is None  # k is fixed by default, so it has no draws
    if keeptrees:
        assert bart.fit is not None
        assert bart.n_chains is None  # reported only when the sampler is dropped
    else:
        assert bart.fit is None
        assert bart.n_chains == 1
    if binary:
        assert_array_equal(bart.binaryOffset, np.zeros(n))
        assert bart.sigma is None
        assert bart.first_sigma is None
        assert bart.sigest is None
        assert bart.y is None
        assert bart.yhat_train_mean is None
        assert bart.yhat_test_mean is None
    else:
        assert_array_equal(bart.y, data.y)
        assert bart.sigma.shape == (NDPOST,)  # burn-in draws are in first_sigma
        assert bart.first_sigma.shape == (NSKIP,)
        assert isinstance(bart.sigest, float)
        assert math.isfinite(bart.sigest)
        assert bart.binaryOffset is None
        assert_close_matrices(
            bart.yhat_train_mean, bart.yhat_train.mean(axis=0), rtol=1e-7
        )
        assert_close_matrices(
            bart.yhat_test_mean, bart.yhat_test.mean(axis=0), rtol=1e-7
        )

    if keeptrees:
        check_generics(bart, data, binary)


def test_bart_no_test_data(data: Data) -> None:
    """Without `x_test` the test outputs are NULL in R, exposed as None.

    The fit also thins, keeping ``ndpost / keepevery`` draws.
    """
    n, _ = data.x.shape
    keepevery = 2
    kept = NDPOST // keepevery
    bart = dbarts.bart(
        x_train=data.x,
        y_train=data.y,
        ntree=NTREE,
        nskip=NSKIP,
        ndpost=NDPOST,
        keepevery=keepevery,
        verbose=False,
    )
    assert bart.yhat_train.shape == (kept, n)
    assert bart.sigma.shape == (kept,)
    assert bart.yhat_test is None
    assert bart.yhat_test_mean is None


@pytest.mark.parametrize('combine', [False, True], ids=['split', 'combined'])
def test_bart_chains(data: Data, combine: bool) -> None:
    """Each chain contributes `ndpost` draws.

    The chains add a leading axis, or stack into the draws axis when
    combined.
    """
    n, p = data.x.shape
    nchain = 2
    bart = dbarts.bart(
        x_train=data.x,
        y_train=data.y,
        ntree=NTREE,
        nskip=NSKIP,
        ndpost=NDPOST,
        nchain=nchain,
        combinechains=combine,
        nthread=1,
        verbose=False,
    )
    draws = (nchain * NDPOST,) if combine else (nchain, NDPOST)
    burnin = (nchain * NSKIP,) if combine else (nchain, NSKIP)
    assert bart.yhat_train.shape == (*draws, n)
    assert bart.sigma.shape == draws
    assert bart.first_sigma.shape == burnin
    assert bart.varcount.shape == (*draws, p)
    assert bart.n_chains == nchain
    assert bart.yhat_train_mean.shape == (n,)


def test_bart_splitprobs(data: Data) -> None:
    """Dict arguments become named R vectors (named columns required).

    Putting all the split probability on `x1` forces every split there.
    """
    _, p = data.x.shape
    bart = dbarts.bart(
        x_train=data.frame.drop(columns='y'),  # named columns for splitprobs
        y_train=data.y,
        splitprobs={'x1': 1.0, '.default': 0.0},
        proposalprobs={'birth_death': 0.5, 'change': 0.1, 'swap': 0.4, 'birth': 0.5},
        ntree=NTREE,
        nskip=NSKIP,
        ndpost=NDPOST,
        verbose=False,
    )
    assert np.all(bart.varcount[:, 0] > 0)
    assert_array_equal(bart.varcount[:, 1:], np.zeros((NDPOST, p - 1), np.int32))


def test_bart2(data: Data) -> None:
    """`bart2` takes a formula and a data frame.

    By default the chains are not combined, adding a leading axis to the
    draws.
    """
    n, p = data.x.shape
    m, _ = data.x_test.shape
    n_chains = 2
    bart = dbarts.bart2(
        'y ~ x1 + x2 + x3',
        data=data.frame,
        test=data.test_frame,
        n_trees=NTREE,
        n_burn=NSKIP,
        n_samples=NDPOST,
        n_chains=n_chains,
        n_threads=1,
        verbose=False,
    )
    assert bart.yhat_train.shape == (n_chains, NDPOST, n)
    assert bart.yhat_test.shape == (n_chains, NDPOST, m)
    assert bart.varcount.shape == (n_chains, NDPOST, p)
    assert bart.sigma.shape == (n_chains, NDPOST)
    assert bart.first_sigma.shape == (n_chains, NSKIP)
    assert bart.n_chains == n_chains
    assert bart.yhat_train_mean.shape == (n,)
    assert bart.yhat_test_mean.shape == (m,)
    assert_array_equal(bart.y, data.y)


def test_rbart_vi(data: Data, rng: np.random.Generator) -> None:
    """`rbart_vi` adds the random-intercept outputs to the `bart` ones.

    By default it keeps the per-chain samplers, so `predict` works; new
    points need a group each.
    """
    n, _ = data.x.shape
    m, _ = data.x_test.shape
    group = rng.integers(0, 3, n)
    n_groups = np.unique(group).size
    fit = dbarts.rbart_vi(
        'y ~ x1 + x2 + x3',
        data=data.frame,
        group_by=group,
        n_trees=NTREE,
        n_burn=NSKIP,
        n_samples=NDPOST,
        n_chains=1,
        n_threads=1,
        n_thin=1,
        verbose=False,
    )
    assert fit.yhat_train.shape == (NDPOST, n)
    assert fit.ranef.shape == (NDPOST, n_groups)
    assert fit.ranef_mean.shape == (n_groups,)
    assert fit.tau.shape == (NDPOST,)
    assert fit.first_tau.shape == (NSKIP,)
    assert fit.sigma.shape == (NDPOST,)
    assert isinstance(fit.sigest, float)
    assert fit.fit is not None  # keepTrees defaults to True for rbart_vi
    assert fit.n_chains is None
    assert fit.seed.dtype == np.int32  # an R .Random.seed vector
    assert_array_equal(fit.group_by, group.astype(str), strict=False)
    assert_array_equal(fit.y, data.y)

    pred = fit.predict(data.test_frame, group_by=group[:m])
    assert pred.shape == (NDPOST, m)


def test_dbarts(data: Data) -> None:
    """The sampler takes a formula string and runs on demand.

    Draws come back as a dict of arrays with the observations on the first
    axis. A copy of the sampler (possible only without cached state) runs
    independently, and the sampler can be modified in place.
    """
    n, p = data.x.shape
    m, _ = data.x_test.shape
    control = dbarts.dbartsControl(
        n_trees=NTREE, n_chains=1, n_threads=1, updateState=False
    )
    sampler = dbarts.dbarts('y ~ x1 + x2 + x3', data=data.frame, control=control)

    out = sampler.run(NSKIP, NDPOST)
    assert sorted(out) == ['sigma', 'test', 'train', 'varcount']
    assert out['train'].shape == (n, NDPOST)
    assert out['sigma'].shape == (NDPOST,)
    assert out['varcount'].shape == (p, NDPOST)
    assert out['test'] is None  # no test data given

    # without keepTrees, the current trees give a single prediction per point
    pred = sampler.predict(data.x_test)
    assert pred.shape == (m,)

    # a copy is a new wrapped sampler that runs independently
    copy = sampler.copy()
    assert isinstance(copy, dbarts.dbarts)
    assert copy is not sampler
    out2 = copy.run(NSKIP, NDPOST)
    assert out2['train'].shape == (n, NDPOST)

    # the sampler state can be drawn from the prior in place
    sampler.sampleTreesFromPrior()
    sampler.sampleNodeParametersFromPrior()

    # replacing the response redirects the fit
    sampler.setResponse(-data.y)
    out3 = sampler.run(NSKIP, NDPOST)
    assert_close_matrices(out3['train'].mean(axis=1), -data.y, rtol=0.5)


def test_dbarts_test_data(data: Data) -> None:
    """The sampler also takes bare matrices and returns test-point draws."""
    m, _ = data.x_test.shape
    control = dbarts.dbartsControl(n_trees=NTREE, n_chains=1, n_threads=1)
    sampler = dbarts.dbarts(data.x, data.y, test=data.x_test, control=control)
    out = sampler.run(NSKIP, NDPOST)
    assert out['test'].shape == (m, NDPOST)
