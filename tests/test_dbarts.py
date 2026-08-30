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
from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter
from typing import Literal, get_args, get_origin

import numpy as np
import pandas as pd
import pytest
from jaxtyping import Float64
from numpy import ndarray
from rpy2 import robjects
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.language import LangVector

from rbartpackages import dbarts
from rbartpackages._src.base import DataFrame, RObjectBase, robjects_r
from tests.util import (
    RegressionData,
    assert_array_equal,
    assert_close_matrices,
    evaluated_r_formals,
    has_var_keyword,
    kwdict,
    mapped_params,
    nnone,
    regression_arrays,
)

NDPOST = 20
NSKIP = 20
NTREE = 10


def phi(x: Float64[ndarray, '...']) -> Float64[ndarray, '...']:
    """Apply the standard normal cumulative distribution function."""
    return (1 + np.vectorize(math.erf)(x / math.sqrt(2))) / 2


def column(frame: DataFrame, name: str) -> ndarray:
    """Extract a column of a dataframe as an array, be it polars or pandas."""
    return frame[name].to_numpy()


@dataclass(frozen=True)
class Data(RegressionData):
    """A small regression dataset, with data-frame views for the formula interfaces."""

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
    return Data(*regression_arrays(rng))


def test_docstring() -> None:
    """The R documentation is attached to the wrapper classes."""
    classes = (
        dbarts.bart,
        dbarts.bart2,
        dbarts.rbart_vi,
        dbarts.dbarts,
        dbarts.dbartsControl,
        dbarts.dbartsData,
    )
    for cls in classes:
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
    assert_close_matrices(latent, nnone(bart.yhat_test), rtol=1e-7)
    if binary:
        assert np.all((pred > 0) & (pred < 1))
        assert_close_matrices(pred, phi(latent), rtol=1e-7)
    else:
        assert_array_equal(pred, latent)  # 'ev' and 'bart' agree

    draws = bart.extract()  # training draws, expected-value scale
    assert isinstance(draws, np.ndarray)
    if binary:
        assert_close_matrices(draws, phi(nnone(bart.yhat_train)), rtol=1e-7)
    else:
        assert_array_equal(draws, nnone(bart.yhat_train))
    assert_close_matrices(bart.fitted(), np.mean(draws, axis=0), rtol=1e-7)

    trees = bart.extract(type='trees')
    # the tree structure comes back as a dataframe (polars if installed, else
    # pandas); both expose the column names through `.columns` and their
    # columns through `[]`
    assert not isinstance(trees, np.ndarray)
    assert {'sample', 'tree', 'n', 'var', 'value'} <= set(trees.columns)

    # the tree-selection arguments reach the sampler's getTrees through R's
    # `...`, so they are usable with type='trees' only
    one = bart.extract(type='trees', treeNums=2, chainNums=1, sampleNums=3)
    assert not isinstance(one, np.ndarray)
    assert set(column(one, 'tree')) == {2}
    assert set(column(one, 'sample')) == {3}

    # newdata routes fresh observations through the frozen trees, leaving the
    # structure alone and counting them in the `n` column instead
    routed = bart.extract(type='trees', newdata=data.x_test)
    assert not isinstance(routed, np.ndarray)
    assert routed.shape == trees.shape
    assert column(routed, 'n').max() == m


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

    assert nnone(bart.yhat_train).shape == (NDPOST, n)
    assert nnone(bart.yhat_test).shape == (NDPOST, m)
    assert bart.varcount.shape == (NDPOST, p)
    assert bart.varcount.dtype == np.int32
    assert bart.k is None  # k is fixed by default, so it has no draws
    if keeptrees:
        # the kept sampler is wrapped, so the dbarts interface works on it
        assert isinstance(bart.fit, dbarts.dbarts)
        assert bart.fit.predict(data.x_test).shape == (m, NDPOST)
        assert bart.n_chains is None  # reported only when the sampler is dropped
    else:
        assert bart.fit is None
        assert bart.n_chains == 1
    if binary:
        assert_array_equal(nnone(bart.binaryOffset), np.zeros(n))
        assert bart.sigma is None
        assert bart.first_sigma is None
        assert bart.sigest is None
        assert bart.y is None
        assert bart.yhat_train_mean is None
        assert bart.yhat_test_mean is None
    else:
        assert_array_equal(nnone(bart.y), data.y)
        assert nnone(bart.sigma).shape == (NDPOST,)  # burn-in draws are in first_sigma
        assert nnone(bart.first_sigma).shape == (NSKIP,)
        assert isinstance(bart.sigest, float)
        assert math.isfinite(bart.sigest)
        assert bart.binaryOffset is None
        assert_close_matrices(
            nnone(bart.yhat_train_mean), nnone(bart.yhat_train).mean(axis=0), rtol=1e-7
        )
        assert_close_matrices(
            nnone(bart.yhat_test_mean), nnone(bart.yhat_test).mean(axis=0), rtol=1e-7
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
    assert nnone(bart.yhat_train).shape == (kept, n)
    assert nnone(bart.sigma).shape == (kept,)
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
    assert nnone(bart.yhat_train).shape == (*draws, n)
    assert nnone(bart.sigma).shape == draws
    assert nnone(bart.first_sigma).shape == burnin
    assert bart.varcount.shape == (*draws, p)
    assert bart.n_chains == nchain
    assert nnone(bart.yhat_train_mean).shape == (n,)


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


def test_bart_keepcall(data: Data) -> None:
    """The call component is an R language object.

    With ``keepcall=False`` R stores a dummy ``NULL()`` call rather than
    NULL, so the attribute is never exposed as None.
    """
    kw: kwdict = dict(
        x_train=data.x,
        y_train=data.y,
        ntree=NTREE,
        nskip=NSKIP,
        ndpost=NDPOST,
        verbose=False,
    )
    assert isinstance(dbarts.bart(**kw).call, LangVector)
    assert isinstance(dbarts.bart(**kw, keepcall=False).call, LangVector)


def test_bart2(data: Data) -> None:
    """`bart2` takes a formula and a data frame; dict args become named vectors.

    By default the chains are not combined, adding a leading axis to the
    draws. Putting all the split probability on `x1` forces every split
    there.
    """
    n, p = data.x.shape
    m, _ = data.x_test.shape
    n_chains = 2
    bart = dbarts.bart2(
        'y ~ x1 + x2 + x3',
        data=data.frame,
        test=data.test_frame,
        split_probs={'x1': 1.0, '.default': 0.0},
        proposal_probs={'birth_death': 0.5, 'change': 0.1, 'swap': 0.4, 'birth': 0.5},
        n_trees=NTREE,
        n_burn=NSKIP,
        n_samples=NDPOST,
        n_chains=n_chains,
        n_threads=1,
        verbose=False,
    )
    assert nnone(bart.yhat_train).shape == (n_chains, NDPOST, n)
    assert nnone(bart.yhat_test).shape == (n_chains, NDPOST, m)
    assert bart.varcount.shape == (n_chains, NDPOST, p)
    assert np.all(bart.varcount[..., 0] > 0)
    assert_array_equal(
        bart.varcount[..., 1:], np.zeros((n_chains, NDPOST, p - 1), np.int32)
    )
    assert nnone(bart.sigma).shape == (n_chains, NDPOST)
    assert nnone(bart.first_sigma).shape == (n_chains, NSKIP)
    assert bart.n_chains == n_chains
    assert nnone(bart.yhat_train_mean).shape == (n,)
    assert nnone(bart.yhat_test_mean).shape == (m,)
    assert_array_equal(nnone(bart.y), data.y)


def test_rbart_vi(data: Data, rng: np.random.Generator) -> None:
    """`rbart_vi` adds the random-intercept outputs to the `bart` ones.

    By default it keeps the per-chain samplers, so `predict` works; new
    points need a group each.
    """
    # draw group membership for random effects
    n, _ = data.x.shape
    # case-mixed names to trigger R/numpy differences in sorting due to locale
    group = np.array(['B', 'a', 'b'])[rng.integers(0, 3, n)]

    # run rbart_vi
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

    # check shapes and types
    assert nnone(fit.yhat_train).shape == (NDPOST, n)
    n_groups = np.unique(group).size
    assert fit.ranef.shape == (NDPOST, n_groups)
    assert fit.ranef_mean.shape == (n_groups,)
    assert_array_equal(np.sort(fit.ranef_levels), np.unique(group))
    assert fit.tau.shape == (NDPOST,)
    assert fit.first_tau.shape == (NSKIP,)
    assert nnone(fit.sigma).shape == (NDPOST,)
    assert isinstance(fit.sigest, float)
    # keepTrees defaults to True for rbart_vi; one wrapped sampler per chain
    (sampler,) = nnone(fit.fit)
    assert isinstance(sampler, dbarts.dbarts)
    assert fit.n_chains is None
    assert fit.seed.dtype == np.int32  # an R .Random.seed vector
    assert_array_equal(fit.group_by, group)
    assert_array_equal(fit.y, data.y)

    # check shape of predictions
    m, _ = data.x_test.shape
    pred = fit.predict(data.test_frame, group_by=group[:m])
    assert pred.shape == (NDPOST, m)

    # check shape of predicted random effects requesting 2 out of 3 groups
    test_group = np.resize(np.array(['b', 'B']), m)
    test_levels = np.unique(test_group)
    ranef_pred, pred_levels = fit.predict(
        data.test_frame, group_by=test_group, type='ranef'
    )
    assert ranef_pred.shape == (NDPOST, test_levels.size)

    # check the returned groups are those we requested, and that the names
    # coming with them place the columns within the training ones
    columns = np.isin(fit.ranef_levels, pred_levels)
    assert_array_equal(pred_levels, fit.ranef_levels[columns])
    assert_array_equal(ranef_pred, fit.ranef[:, columns])

    # check shape of predicted random effects requesting 1 out of 3 groups
    single, single_levels = fit.predict(
        data.test_frame, group_by=np.full(m, test_group[0]), type='ranef'
    )
    assert single.shape == (NDPOST, 1)

    # check the returned group is the one we requested; together with the
    # analogous check above on the size-2 subset, this fully checks that the
    # ordering of groups is the one given by the returned names
    assert_array_equal(single_levels, test_group[:1])
    (column,) = np.flatnonzero(fit.ranef_levels == test_group[0])
    assert_array_equal(single.squeeze(-1), fit.ranef[:, column])

    # check a group not seen in training gets a new independent effect drawn
    # from the prior, placed by level order rather than appended ('a' < 'aa' <
    # 'b' in any collation, so R and numpy agree on the order here)
    new_group = np.resize(np.array(['a', 'aa', 'b']), m)
    new_ranef, new_levels = fit.predict(
        data.test_frame, group_by=new_group, type='ranef'
    )
    assert_array_equal(new_levels, np.array(['a', 'aa', 'b']))
    assert new_ranef.shape == (NDPOST, new_levels.size)

    # the trained groups keep their effects, in place around the new one
    for index, level in enumerate(new_levels):
        trained = np.flatnonzero(fit.ranef_levels == level)
        if trained.size:
            assert_array_equal(new_ranef[:, index], fit.ranef[:, trained.item()])
        else:
            assert not any(
                np.array_equal(new_ranef[:, index], fit.ranef[:, other])
                for other in range(n_groups)
            )


def test_rbart_vi_group_level_order(data: Data) -> None:
    """The group names follow R's level order, not numpy's sorting of the labels.

    R orders the levels of a numeric grouping numerically, but the grouping
    comes back to Python as its string labels, which sort differently; the
    names must be read from R rather than recomputed from `group_by`.
    """
    # define groups and run `rbart_vi`
    n, _ = data.x.shape
    m, _ = data.x_test.shape
    # '10' sorts before '2' as a string, after it as a number
    groups = np.array([2, 3, 10])
    test_groups = groups[[0, 2]]
    fit = dbarts.rbart_vi(
        'y ~ x1 + x2 + x3',
        data=data.frame,
        group_by=np.resize(groups, n),
        test=data.test_frame,
        group_by_test=np.resize(test_groups, m),
        n_trees=NTREE,
        n_burn=NSKIP,
        n_samples=NDPOST,
        n_chains=1,
        n_threads=1,
        n_thin=1,
        verbose=False,
    )

    # the training names keep R's numeric ordering
    expected = np.array(['2', '3', '10'])
    assert_array_equal(fit.ranef_levels, expected)
    assert_array_equal(np.sort(expected), np.array(['10', '2', '3']))

    # the training methods return the attributes as they are, with those names
    train_ranef, train_levels = fit.extract(type='ranef')
    assert_array_equal(train_ranef, fit.ranef)
    assert_array_equal(train_levels, expected)
    train_mean, train_mean_levels = fit.fitted(type='ranef')
    assert_close_matrices(train_mean, fit.ranef_mean, rtol=1e-15)
    assert_array_equal(train_mean_levels, expected)

    # the test sample has its own level set, likewise in R's order, and the
    # columns it selects follow those names
    ranef, levels = fit.extract(type='ranef', sample='test')
    assert_array_equal(levels, np.array(['2', '10']))
    assert_array_equal(ranef, fit.ranef[:, [0, 2]])
    ranef_mean, mean_levels = fit.fitted(type='ranef', sample='test')
    assert_array_equal(mean_levels, levels)
    assert_close_matrices(ranef_mean, fit.ranef_mean[[0, 2]], rtol=1e-15)

    # `predict` labels its own result, whatever the grouping it is given
    _, pred_levels = fit.predict(
        data.test_frame, group_by=np.resize(groups, m), type='ranef'
    )
    assert_array_equal(pred_levels, expected)


def test_dbarts(data: Data) -> None:
    """The sampler takes a formula string and runs on demand.

    Draws come back as a dict of arrays with the observations on the first
    axis. A copy of the sampler runs independently, and the sampler can be
    modified in place, with the field properties tracking the updates.
    """
    n, p = data.x.shape
    m, _ = data.x_test.shape
    control = dbarts.dbartsControl(n_trees=NTREE, n_chains=1, n_threads=1)
    sampler = dbarts.dbarts(
        'y ~ x1 + x2 + x3',
        data=data.frame,
        control=control,
        # exercise the dict-to-named-vector conversion of proposal_probs
        proposal_probs={'birth_death': 0.5, 'change': 0.1, 'swap': 0.4, 'birth': 0.5},
    )

    out = nnone(sampler.run(NSKIP, NDPOST))
    assert sorted(out) == ['sigma', 'test', 'train', 'varcount']
    assert out['train'].shape == (n, NDPOST)
    assert out['sigma'].shape == (NDPOST,)
    assert out['varcount'].shape == (p, NDPOST)
    assert out['test'] is None  # no test data given

    # a burn-in-only run keeps zero samples: invisible NULL, exposed as None
    assert sampler.run(NSKIP, 0) is None

    # without keepTrees, the current trees give a single prediction per point
    pred = sampler.predict(data.x_test)
    assert pred.shape == (m,)

    # a copy is a new wrapped sampler that runs independently, cached state
    # and all
    copy = sampler.copy()
    assert isinstance(copy, dbarts.dbarts)
    assert copy is not sampler
    out2 = nnone(copy.run(NSKIP, NDPOST))
    assert out2['train'].shape == (n, NDPOST)

    # the sampler state can be drawn from the prior in place
    sampler.sampleTreesFromPrior()
    sampler.sampleNodeParametersFromPrior()

    # the field properties read off the live R object: setResponse shows
    # through data
    assert sampler.model.rclass[0] == 'dbartsModel'
    assert isinstance(sampler.control, dbarts.dbartsControl)
    assert isinstance(sampler.data, dbarts.dbartsData)
    assert sampler.state is not None

    # replacing the response redirects the fit
    sampler.setResponse(-data.y)
    y = robjects_r('function(d) d@y')(sampler.data._robject)
    assert_array_equal(np.asarray(y), -data.y)
    out3 = nnone(sampler.run(NSKIP, NDPOST))
    assert_close_matrices(out3['train'].mean(axis=1), -data.y, rtol=0.5)


def test_dbarts_test_data(data: Data) -> None:
    """The sampler also takes bare matrices and returns test-point draws.

    With the default ``updateState``, running the sampler caches the state,
    readable through the `state` property.
    """
    m, _ = data.x_test.shape
    control = dbarts.dbartsControl(n_trees=NTREE, n_chains=1, n_threads=1)
    sampler = dbarts.dbarts(data.x, data.y, test=data.x_test, control=control)
    out = nnone(sampler.run(NSKIP, NDPOST))
    assert nnone(out['test']).shape == (m, NDPOST)

    (item,) = sampler.state.items()  # one state per chain
    assert item.value.rclass[0] == 'dbartsState'


def test_dbarts_binary(data: Data) -> None:
    """With a binary response, `run` pins `sigma` at 1 and draws `k`.

    The sampler's default end-node prior for binary outcomes puts a
    hyperprior on `k`, so its draws appear in the output.
    """
    n, _ = data.x.shape
    control = dbarts.dbartsControl(n_trees=NTREE, n_chains=1, n_threads=1)
    sampler = dbarts.dbarts(data.x, data.biny, control=control)
    out = nnone(sampler.run(NSKIP, NDPOST))
    assert sorted(out) == ['k', 'sigma', 'test', 'train', 'varcount']
    assert out['train'].shape == (n, NDPOST)
    assert_array_equal(out['sigma'], np.ones(NDPOST))
    assert out['k'].shape == (NDPOST,)
    assert np.all(out['k'] > 0)


def test_dbarts_setters(data: Data) -> None:
    """The set* methods replace the sampler's components in place.

    Unforced predictor updates report success, the test offset enters the
    test fits, the train offset lands in the data object, a `dbartsData`
    swaps the data wholesale, and a ``keepTrees`` control makes `predict`
    return the kept draws.
    """
    n, _ = data.x.shape
    m, _ = data.x_test.shape
    control = dbarts.dbartsControl(
        n_trees=NTREE, n_chains=1, n_threads=1, n_samples=NDPOST
    )
    sampler = dbarts.dbarts(data.x, data.y, test=data.x_test, control=control)

    # unforced updates report success (the trees are stumps, so no leaf can
    # end up empty); whole-matrix updates are forced by default
    assert nnone(sampler.setPredictor(2 * data.x, forceUpdate=False)).item()
    assert nnone(sampler.setPredictor(data.x[:, 0], 1)).item()  # column 1, 1-based

    # a forced update reports nothing
    assert sampler.setPredictor(2 * data.x) is None
    assert sampler.setPredictor(data.x[:, 0], 1, forceUpdate=True) is None

    # a partial update reports one flag per observation instead
    installed = nnone(sampler.setPredictor(data.x[:, 1], 1, forceUpdate='partial'))
    assert installed.shape == (n,)
    assert installed.all()

    sampler.setSigma(1.0)

    # replacing the test predictors changes the test draws
    sampler.setTestPredictor(data.x[:10])
    out = nnone(sampler.run(NSKIP, NDPOST))
    assert nnone(out['test']).shape == (10, NDPOST)

    # the test offset enters the test draws only
    sampler.setTestPredictorAndOffset(data.x_test, 1e6)
    out = nnone(sampler.run(0, NDPOST))
    assert nnone(out['test']).shape == (m, NDPOST)
    assert np.all(nnone(out['test']) > 1e5)
    assert np.all(np.abs(out['train']) < 1e5)
    sampler.setTestOffset(0.0)
    out = nnone(sampler.run(0, NDPOST))
    assert np.all(np.abs(nnone(out['test'])) < 1e5)

    # the train offset lands in the data object; its effect on the draws is
    # not asserted because a large post-hoc offset makes the sampler bimodal
    # (absorbed by either the trees or sigma), so where the short-run draws
    # sit depends on the seed
    sampler.setOffset(np.full(n, 1e3))
    offset = robjects_r('function(d) d@offset')(sampler.data._robject)
    assert_array_equal(np.asarray(offset), np.full(n, 1e3))
    sampler.setOffset(0.0)  # scalars are expanded to the n observations
    offset = robjects_r('function(d) d@offset')(sampler.data._robject)
    assert_array_equal(np.asarray(offset), np.zeros(n))

    # the model (priors) can be grafted from another sampler, as the
    # dbartsModel constructor is not exported
    other = dbarts.dbarts(data.x, data.y, control=control)
    sampler.setModel(other.model)

    # a dbartsData replaces the training data (and drops the test data)
    sampler.setData(dbarts.dbartsData('y ~ x1 + x2 + x3', data.frame.iloc[: n // 2]))
    out = nnone(sampler.run(NSKIP, NDPOST))
    assert out['train'].shape == (n // 2, NDPOST)
    assert out['test'] is None

    # the wrapped data property feeds back into setData: grafting another
    # sampler's data restores the full training set
    sampler.setData(other.data)
    out = nnone(sampler.run(NSKIP, NDPOST))
    assert out['train'].shape == (n, NDPOST)

    # a keepTrees control makes predict return the kept draws
    keeping = dbarts.dbartsControl(
        n_trees=NTREE, n_chains=1, n_threads=1, n_samples=NDPOST, keepTrees=True
    )
    sampler.setControl(keeping)
    sampler.setControl(sampler.control)  # the control property round-trips
    sampler.run(NSKIP, NDPOST)
    assert sampler.predict(data.x_test).shape == (m, NDPOST)


def test_dbarts_setters_column(data: Data) -> None:
    """The predictor setters select the columns to replace by name or by index.

    A single column is selected by 1-based index or by name, several at once by
    an array of indices; names require the sampler to have column names.
    """
    control = dbarts.dbartsControl(
        n_trees=NTREE, n_chains=1, n_threads=1, n_samples=NDPOST
    )
    frame = data.frame.drop(columns='y')  # named columns, so 'x1' resolves
    sampler = dbarts.dbarts(frame, data.y, test=data.test_frame, control=control)
    sampler.run(NSKIP, NDPOST)

    def slot(name: str) -> ndarray:
        return np.asarray(robjects_r(f'function(d) d@{name}')(sampler.data._robject))

    new = data.x[:, 2]
    assert sampler.setPredictor(new, 'x1', forceUpdate=True) is None
    assert_array_equal(slot('x')[:, 0], new)

    pair = 2 * data.x[:, :2]
    assert sampler.setPredictor(pair, np.array([1, 2]), forceUpdate=True) is None
    assert_array_equal(slot('x')[:, :2], pair)

    # the test matrix takes the same selectors
    new_test = data.x_test[:, 2]
    sampler.setTestPredictor(new_test, 'x1')
    assert_array_equal(slot('x.test')[:, 0], new_test)
    pair_test = 2 * data.x_test[:, :2]
    sampler.setTestPredictor(pair_test, np.array([1, 2]))
    assert_array_equal(slot('x.test')[:, :2], pair_test)


def test_dbarts_setters_clear(data: Data) -> None:
    """``None`` clears the offsets and the test predictors."""
    n, _ = data.x.shape
    control = dbarts.dbartsControl(
        n_trees=NTREE, n_chains=1, n_threads=1, n_samples=NDPOST
    )
    sampler = dbarts.dbarts(data.x, data.y, test=data.x_test, control=control)

    def is_null(name: str) -> bool:
        out = robjects_r(f'function(d) is.null(d@{name})')(sampler.data._robject)
        return bool(np.asarray(out).item())

    sampler.setOffset(np.full(n, 1e3))
    assert not is_null('offset')
    sampler.setOffset(None)
    assert is_null('offset')

    sampler.setTestOffset(0.0)
    assert not is_null('offset.test')
    sampler.setTestOffset(None)
    assert is_null('offset.test')

    sampler.setTestPredictorAndOffset(data.x_test, 0.0)
    assert not is_null('offset.test')
    sampler.setTestPredictorAndOffset(data.x_test, None)
    assert is_null('offset.test')

    sampler.setTestPredictor(data.x_test)
    assert not is_null('x.test')
    sampler.setTestPredictor(None)
    assert is_null('x.test')

    # clearing the test matrix and its offset together is the supported way
    sampler.setTestPredictorAndOffset(data.x_test, 0.0)
    assert not is_null('x.test')
    sampler.setTestPredictorAndOffset(None, None)
    assert is_null('x.test')
    assert is_null('offset.test')

    # R rejects the leftovers: a column has no NULL meaning, and a NULL test
    # matrix cannot keep an offset
    sampler.setTestPredictor(data.x_test)
    with pytest.raises(RRuntimeError, match='length of new x does not match'):
        sampler.setTestPredictor(None, 1)
    with pytest.raises(RRuntimeError, match='test offset must be'):
        sampler.setTestPredictorAndOffset(None, 0.0)


def test_dbarts_get_trees(data: Data) -> None:
    """`getTrees` returns the structure of the sampler's trees as a data frame.

    With a ``keepTrees`` control the saved samples are returned, and the tree,
    chain, and sample indices select a subset of them; `current` asks for the
    live working trees instead, which have no sample dimension. `newdata`
    routes new observations through the frozen trees, so the `n` column counts
    those instead of the training ones.
    """
    n, _ = data.x.shape
    m, _ = data.x_test.shape
    n_chains = 2
    control = dbarts.dbartsControl(
        n_trees=NTREE, n_chains=n_chains, n_threads=1, n_samples=NDPOST, keepTrees=True
    )
    sampler = dbarts.dbarts(data.x, data.y, control=control)
    sampler.run(NSKIP, NDPOST)

    trees = sampler.getTrees()
    assert set(trees.columns) == {'chain', 'sample', 'tree', 'n', 'var', 'value'}
    assert set(column(trees, 'chain')) == set(range(1, n_chains + 1))
    assert column(trees, 'n').max() == n  # the root holds every observation

    # each index selects along its own axis, so they are all told apart
    subset = sampler.getTrees(np.array([3, 5]), 2, np.array([7, 8, 9]))
    assert set(column(subset, 'tree')) == {3, 5}
    assert set(column(subset, 'chain')) == {2}
    assert set(column(subset, 'sample')) == {7, 8, 9}

    # the live working trees are a single set per chain, so no sample column
    current = sampler.getTrees(current=True)
    assert set(current.columns) == {'chain', 'tree', 'n', 'var', 'value'}

    # new observations keep the tree structure but change the node counts
    routed = sampler.getTrees(newdata=data.x_test)
    assert routed.shape == trees.shape
    assert column(routed, 'n').max() == m


def test_update_predictor_jointly(data: Data) -> None:
    """The joint update replaces a shared predictor column across samplers.

    The column is matched by name, so it may sit at a different position in
    each sampler; the returned flags say which observations were installed,
    and those take the new value in every sampler.
    """
    n, _ = data.x.shape
    control = dbarts.dbartsControl(n_trees=NTREE, n_chains=1, n_threads=1)
    frame = data.frame.drop(columns='y')  # named columns to match x1 across
    first = dbarts.dbarts(frame, data.y, control=control)
    second = dbarts.dbarts(frame[['x2', 'x1', 'x3']], -data.y, control=control)
    first.run(NSKIP, NDPOST)
    second.run(NSKIP, NDPOST)

    new = data.x[:, 2]
    installed = dbarts.updatePredictorPerObservationJointly([first, second], new, 'x1')
    assert installed.shape == (n,)
    assert installed.dtype == np.bool_
    assert installed.any()
    for sampler, index in [(first, 0), (second, 1)]:
        x = np.asarray(robjects_r('function(d) d@x')(sampler.data._robject))
        assert_array_equal(x[installed, index], new[installed])

    # a lone sampler is accepted too, and a 1-based index names the column
    again = dbarts.updatePredictorPerObservationJointly(
        first, data.x[:, 1], 1, updateState=True
    )
    assert again.shape == (n,)


def test_dbarts_show_trees(data: Data, capfd: pytest.CaptureFixture) -> None:
    """`show` and `printTrees` write to the R console, `plotTree` to a device."""
    control = dbarts.dbartsControl(n_trees=NTREE, n_chains=1, n_threads=1)
    sampler = dbarts.dbarts(data.x, data.y, control=control)
    sampler.run(NSKIP, NDPOST)

    sampler.show()
    assert 'dbarts sampler' in capfd.readouterr().out

    sampler.printTrees(1)  # the current first tree
    assert capfd.readouterr().out.strip()

    # plot to a null device to keep the test headless
    robjects_r('pdf(NULL)')
    try:
        sampler.plotTree(1)
    finally:
        robjects_r('invisible(dev.off())')


# the wrapper constructors and the R arguments deliberately left unexposed (none:
# the constructors expose every named R argument, forwarding `...` where present)
CONSTRUCTOR_CASES = [
    (dbarts.bart, set()),
    (dbarts.bart2, set()),
    (dbarts.rbart_vi, set()),
    (dbarts.dbarts, set()),
    (dbarts.dbartsControl, set()),
    (dbarts.dbartsData, set()),
]


@pytest.mark.parametrize(
    ('cls', 'unexposed'),
    CONSTRUCTOR_CASES,
    ids=[c.__name__ for c, _ in CONSTRUCTOR_CASES],
)
def test_signature_defaults_match_r(
    cls: type[RObjectBase], unexposed: set[str]
) -> None:
    """The explicit constructor signatures stay in sync with the R functions.

    Every literal default in a Python signature must match its R counterpart,
    every R argument must be either exposed or deliberately unexposed, and R's
    ``...`` must be forwarded by a ``**kwargs`` catch-all, so that an upstream
    update that changes a default or adds an argument fails here instead of
    silently diverging.
    """
    rfuncname = cls._rfuncname
    params = mapped_params(cls, dots=True)
    rnames = set(robjects_r(f'names(formals({rfuncname}))'))
    # R's `...` is forwarded by a **kwargs catch-all, not a named parameter
    assert ('...' in rnames) == has_var_keyword(cls), rfuncname
    rnames -= {'...'}
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


# the tree-selection arguments `extract` documents but forwards through R's
# `...` to the sampler's getTrees rather than taking as named formals
TREE_ARGS = {'treeNums', 'chainNums', 'sampleNums', 'newdata'}

# the bart/rbart fit generics, their R class, the dispatch arguments the
# wrapper binds itself, the R arguments left unexposed, and the Python
# arguments that reach R through `...`
GENERIC_CASES = [
    (dbarts.bart.predict, 'predict', 'bart', {'object', 'newdata'}, {'...'}, set()),
    (dbarts.bart.extract, 'extract', 'bart', {'object'}, {'...'}, TREE_ARGS),
    (dbarts.bart.fitted, 'fitted', 'bart', {'object'}, {'...'}, set()),
    (
        dbarts.rbart_vi.predict,
        'predict',
        'rbart',
        {'object', 'newdata'},
        {'...'},
        set(),
    ),
    (dbarts.rbart_vi.extract, 'extract', 'rbart', {'object'}, {'...'}, TREE_ARGS),
    (dbarts.rbart_vi.fitted, 'fitted', 'rbart', {'object'}, {'...'}, set()),
]


@pytest.mark.parametrize(
    ('meth', 'generic', 'rclass', 'bound', 'unexposed', 'dotted'),
    GENERIC_CASES,
    ids=[f'{g}.{c}' for _, g, c, _, _, _ in GENERIC_CASES],
)
def test_generic_signatures_match_r(
    meth: Callable,
    generic: str,
    rclass: str,
    bound: set[str],
    unexposed: set[str],
    dotted: set[str],
) -> None:
    """The explicit `predict`/`extract`/`fitted` signatures track the R methods.

    Every Python argument must appear in the dispatched R method's formals
    (minus the dispatch arguments the wrapper fills itself) or be one of the
    arguments R takes through ``...``, every R argument must be exposed or
    deliberately unexposed, and the defaults vary with the fit, so the
    signature defers each to R with ``None``. The quantities offered by
    ``type`` differ per method, so its `Literal` must list R's own choices.
    """
    method = f'getS3method("{generic}", "{rclass}", envir = asNamespace("dbarts"))'
    rnames = set(robjects_r(f'names(formals({method}))')) - bound
    params = mapped_params(meth, skip=bound, dots=True)
    assert params.keys() - rnames == dotted
    assert rnames - params.keys() == unexposed
    for name, param in params.items():
        assert param.default is None, name

    # `type` is annotated as `Literal[...] | None`, R's choices as a `c(...)`
    # default that `evaluated_r_formals` evaluates to a character vector
    literal = next(
        arg for arg in get_args(params['type'].annotation) if get_origin(arg) is Literal
    )
    assert_array_equal(get_args(literal), evaluated_r_formals(method)['type'])


# the sampler reference-class methods and the R arguments left unexposed
SAMPLER_METHODS = [
    ('run', set()),
    ('copy', set()),
    ('predict', set()),
    ('sampleTreesFromPrior', set()),
    ('sampleNodeParametersFromPrior', set()),
    ('show', set()),
    ('setControl', set()),
    ('setModel', set()),
    ('setData', set()),
    ('setResponse', set()),
    ('setOffset', set()),
    ('setSigma', set()),
    ('setPredictor', set()),
    ('setTestPredictor', set()),
    ('setTestPredictorAndOffset', set()),
    ('setTestOffset', set()),
    ('printTrees', set()),
    ('getTrees', set()),
    ('plotTree', {'...'}),
]


@pytest.mark.parametrize(
    ('method', 'unexposed'), SAMPLER_METHODS, ids=[m for m, _ in SAMPLER_METHODS]
)
def test_sampler_method_signatures_match_r(method: str, unexposed: set[str]) -> None:
    """The explicit `dbarts` sampler methods track their R reference-class methods.

    Every Python argument must appear in the reference method's formals, every
    R argument must be exposed or deliberately unexposed, and the optional
    arguments defer their R defaults (``NA``, the control object) with ``None``.
    """
    refmethods = 'dbarts:::dbartsSampler$def@refMethods'
    rformals = robjects_r(f'names(formals({refmethods}${method}))')
    rnames = set() if rformals is robjects.NULL else set(rformals)
    params = mapped_params(getattr(dbarts.dbarts, method), dots=True)
    assert params.keys() <= rnames, method
    assert rnames - params.keys() == unexposed, method
    for name, param in params.items():
        if param.default is not Parameter.empty:
            assert param.default is None, f'{method}, argument {name}'


def test_update_predictor_jointly_signature_matches_r() -> None:
    """The `updatePredictorPerObservationJointly` signature tracks the R function.

    It exposes every R argument, and defers R's ``NA`` `updateState` default
    with ``None``.
    """
    rfuncname = 'dbarts::updatePredictorPerObservationJointly'
    rnames = set(robjects_r(f'names(formals({rfuncname}))'))
    params = mapped_params(dbarts.updatePredictorPerObservationJointly, dots=True)
    assert params.keys() == rnames
    assert params['updateState'].default is None


def test_constructors_reject_unknown_arguments(data: Data) -> None:
    """Arguments outside the explicit signatures of the dots-free constructors fail.

    `bart`, `dbarts`, and `dbartsControl` have no R ``...``, so their explicit
    signatures replace it: a misspelled or package-foreign argument fails as a
    `TypeError` instead of reaching R.
    """
    with pytest.raises(TypeError, match='unexpected keyword'):
        dbarts.bart(data.x, data.y, n_trees=NTREE)  # ty: ignore[unknown-argument] # the bart2 spelling of ntree
    with pytest.raises(TypeError, match='unexpected keyword'):
        dbarts.dbarts(data.x, data.y, ntree=NTREE)  # ty: ignore[unknown-argument] # the bart spelling
    with pytest.raises(TypeError, match='unexpected keyword'):
        dbarts.dbartsControl(bogus=1)  # ty: ignore[unknown-argument]


def test_bart2_forwards_control_kwargs(data: Data) -> None:
    """`bart2` forwards unrecognized keyword arguments to `dbartsControl`.

    R's ``...`` reaches `dbartsControl`, so a valid control argument (here a
    deterministic `rngSeed`) is accepted, while a bogus one is rejected by R.
    """
    common: kwdict = dict(n_trees=NTREE, n_burn=NSKIP, n_samples=NDPOST, verbose=False)
    fit = dbarts.bart2('y ~ x1 + x2 + x3', data=data.frame, rngSeed=1, **common)
    again = dbarts.bart2('y ~ x1 + x2 + x3', data=data.frame, rngSeed=1, **common)
    # the forwarded seed makes the single-threaded fit reproducible
    assert_array_equal(nnone(fit.yhat_train), nnone(again.yhat_train))

    with pytest.raises(RRuntimeError, match='unknown arguments'):
        dbarts.bart2('y ~ x1', data=data.frame, totallybogus=1, **common)


def test_bart_explicit_signature(data: Data) -> None:
    """The explicit `bart` signature forwards its scalar arguments to R faithfully.

    `sigest` overrides the calibrated error-SD estimate of a continuous fit,
    and `binaryOffset` shifts the latent scale of a binary fit (R fills the
    `binaryOffset` component with the per-observation value used).
    """
    n, _ = data.x.shape
    common: kwdict = dict(ntree=NTREE, nskip=NSKIP, ndpost=NDPOST, verbose=False)

    sigest = 2.5
    bart = dbarts.bart(data.x, data.y, sigest=sigest, **common)
    assert bart.sigest == sigest

    offset = 0.3
    binary = dbarts.bart(data.x, data.biny, binaryOffset=offset, **common)
    assert_array_equal(nnone(binary.binaryOffset), np.full(n, offset))
