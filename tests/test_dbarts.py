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
from inspect import Parameter, signature

import numpy as np
import pandas as pd
import pytest
from jaxtyping import Float64
from numpy import ndarray
from rpy2 import robjects
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.language import LangVector

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
        # the kept sampler is wrapped, so the dbarts interface works on it
        assert isinstance(bart.fit, dbarts.dbarts)
        assert bart.fit.predict(data.x_test).shape == (m, NDPOST)
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


def test_bart_keepcall(data: Data) -> None:
    """The call component is an R language object.

    With ``keepcall=False`` R stores a dummy ``NULL()`` call rather than
    NULL, so the attribute is never exposed as None.
    """
    kw = dict(
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
    assert bart.yhat_train.shape == (n_chains, NDPOST, n)
    assert bart.yhat_test.shape == (n_chains, NDPOST, m)
    assert bart.varcount.shape == (n_chains, NDPOST, p)
    assert np.all(bart.varcount[..., 0] > 0)
    assert_array_equal(
        bart.varcount[..., 1:], np.zeros((n_chains, NDPOST, p - 1), np.int32)
    )
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
    # keepTrees defaults to True for rbart_vi; one wrapped sampler per chain
    (sampler,) = fit.fit
    assert isinstance(sampler, dbarts.dbarts)
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
    independently, and the sampler can be modified in place, with the field
    properties tracking the updates.
    """
    n, p = data.x.shape
    m, _ = data.x_test.shape
    control = dbarts.dbartsControl(
        n_trees=NTREE, n_chains=1, n_threads=1, updateState=False
    )
    sampler = dbarts.dbarts(
        'y ~ x1 + x2 + x3',
        data=data.frame,
        control=control,
        # exercise the dict-to-named-vector conversion of proposal_probs
        proposal_probs={'birth_death': 0.5, 'change': 0.1, 'swap': 0.4, 'birth': 0.5},
    )

    out = sampler.run(NSKIP, NDPOST)
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

    # a copy is a new wrapped sampler that runs independently
    copy = sampler.copy()
    assert isinstance(copy, dbarts.dbarts)
    assert copy is not sampler
    out2 = copy.run(NSKIP, NDPOST)
    assert out2['train'].shape == (n, NDPOST)

    # the sampler state can be drawn from the prior in place
    sampler.sampleTreesFromPrior()
    sampler.sampleNodeParametersFromPrior()

    # the field properties read off the live R object: setResponse shows
    # through data, and the state is never cached with updateState=False
    assert sampler.model.rclass[0] == 'dbartsModel'
    assert isinstance(sampler.control, dbarts.dbartsControl)
    assert isinstance(sampler.data, dbarts.dbartsData)
    assert sampler.state is None

    # replacing the response redirects the fit
    sampler.setResponse(-data.y)
    y = robjects.r('function(d) d@y')(sampler.data._robject)
    assert_array_equal(np.asarray(y), -data.y)
    out3 = sampler.run(NSKIP, NDPOST)
    assert_close_matrices(out3['train'].mean(axis=1), -data.y, rtol=0.5)


def test_dbarts_test_data(data: Data) -> None:
    """The sampler also takes bare matrices and returns test-point draws.

    With the default ``updateState``, running the sampler caches the state,
    readable through the `state` property.
    """
    m, _ = data.x_test.shape
    control = dbarts.dbartsControl(n_trees=NTREE, n_chains=1, n_threads=1)
    sampler = dbarts.dbarts(data.x, data.y, test=data.x_test, control=control)
    out = sampler.run(NSKIP, NDPOST)
    assert out['test'].shape == (m, NDPOST)

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
    out = sampler.run(NSKIP, NDPOST)
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
    assert sampler.setPredictor(2 * data.x, forceUpdate=False).item()
    assert sampler.setPredictor(data.x[:, 0], 1).item()  # column 1, 1-based
    sampler.setSigma(1.0)

    # replacing the test predictors changes the test draws
    sampler.setTestPredictor(data.x[:10])
    out = sampler.run(NSKIP, NDPOST)
    assert out['test'].shape == (10, NDPOST)

    # the test offset enters the test draws only
    sampler.setTestPredictorAndOffset(data.x_test, 1e6)
    out = sampler.run(0, NDPOST)
    assert out['test'].shape == (m, NDPOST)
    assert np.all(out['test'] > 1e5)
    assert np.all(np.abs(out['train']) < 1e5)
    sampler.setTestOffset(0.0)
    out = sampler.run(0, NDPOST)
    assert np.all(np.abs(out['test']) < 1e5)

    # the train offset lands in the data object; its effect on the draws is
    # not asserted because a large post-hoc offset makes the sampler bimodal
    # (absorbed by either the trees or sigma), so where the short-run draws
    # sit depends on the seed
    sampler.setOffset(np.full(n, 1e3))
    offset = robjects.r('function(d) d@offset')(sampler.data._robject)
    assert_array_equal(np.asarray(offset), np.full(n, 1e3))
    sampler.setOffset(0.0)  # scalars are expanded to the n observations
    offset = robjects.r('function(d) d@offset')(sampler.data._robject)
    assert_array_equal(np.asarray(offset), np.zeros(n))

    # the model (priors) can be grafted from another sampler, as the
    # dbartsModel constructor is not exported
    other = dbarts.dbarts(data.x, data.y, control=control)
    sampler.setModel(other.model)

    # a dbartsData replaces the training data (and drops the test data)
    sampler.setData(dbarts.dbartsData('y ~ x1 + x2 + x3', data.frame.iloc[: n // 2]))
    out = sampler.run(NSKIP, NDPOST)
    assert out['train'].shape == (n // 2, NDPOST)
    assert out['test'] is None

    # the wrapped data property feeds back into setData: grafting another
    # sampler's data restores the full training set
    sampler.setData(other.data)
    out = sampler.run(NSKIP, NDPOST)
    assert out['train'].shape == (n, NDPOST)

    # a keepTrees control makes predict return the kept draws
    keeping = dbarts.dbartsControl(
        n_trees=NTREE, n_chains=1, n_threads=1, n_samples=NDPOST, keepTrees=True
    )
    sampler.setControl(keeping)
    sampler.setControl(sampler.control)  # the control property round-trips
    sampler.run(NSKIP, NDPOST)
    assert sampler.predict(data.x_test).shape == (m, NDPOST)


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
    robjects.r('pdf(NULL)')
    try:
        sampler.plotTree(1)
    finally:
        robjects.r('invisible(dev.off())')


def evaluated_r_formals(rfuncname: str) -> dict[str, ndarray]:
    """Evaluate the argument defaults of an R function in isolation.

    Defaults that are NULL, missing, or that cannot be evaluated standalone
    (e.g. because they reference other arguments) are omitted.
    """
    rdefaults = robjects.r(f"""
        Filter(
            Negate(is.null),
            lapply(
                formals({rfuncname}),
                function(d) tryCatch(eval(d, baseenv()), error = function(e) NULL)
            )
        )
    """)
    return {
        name: np.asarray(value)
        for name, value in zip(rdefaults.names, rdefaults, strict=True)
    }


def mapped_params(
    obj: Callable, *, skip: set[str] = frozenset()
) -> dict[str, Parameter]:
    """Return the named parameters of `obj`, keyed by R name (``_`` to ``.``).

    Skips ``self`` and the ``*args``/``**kwargs`` catch-alls, which have no R
    formal counterpart.
    """
    variadic = {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}
    return {
        name.removesuffix('_').replace('_', '.'): param
        for name, param in signature(obj).parameters.items()
        if name not in {'self', *skip} and param.kind not in variadic
    }


def has_var_keyword(obj: Callable) -> bool:
    """Whether `obj` has a ``**kwargs`` catch-all (forwarding R's ``...``)."""
    return any(
        param.kind is Parameter.VAR_KEYWORD
        for param in signature(obj).parameters.values()
    )


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
def test_signature_defaults_match_r(cls: type, unexposed: set[str]) -> None:
    """The explicit constructor signatures stay in sync with the R functions.

    Every literal default in a Python signature must match its R counterpart,
    every R argument must be either exposed or deliberately unexposed, and R's
    ``...`` must be forwarded by a ``**kwargs`` catch-all, so that an upstream
    update that changes a default or adds an argument fails here instead of
    silently diverging.
    """
    rfuncname = cls._rfuncname
    params = mapped_params(cls)
    rnames = set(robjects.r(f'names(formals({rfuncname}))'))
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


# the bart/rbart fit generics, their R class, the dispatch arguments the
# wrapper binds itself, and the R arguments left unexposed
GENERIC_CASES = [
    (dbarts.bart.predict, 'predict', 'bart', {'object', 'newdata'}, {'...'}),
    (dbarts.bart.extract, 'extract', 'bart', {'object'}, {'...'}),
    (dbarts.bart.fitted, 'fitted', 'bart', {'object'}, {'...'}),
    (dbarts.rbart_vi.predict, 'predict', 'rbart', {'object', 'newdata'}, {'...'}),
]


@pytest.mark.parametrize(
    ('meth', 'generic', 'rclass', 'bound', 'unexposed'),
    GENERIC_CASES,
    ids=[f'{g}.{c}' for _, g, c, _, _ in GENERIC_CASES],
)
def test_generic_signatures_match_r(
    meth: Callable, generic: str, rclass: str, bound: set[str], unexposed: set[str]
) -> None:
    """The explicit `predict`/`extract`/`fitted` signatures track the R methods.

    Every Python argument must appear in the dispatched R method's formals
    (minus the dispatch arguments the wrapper fills itself), every R argument
    must be exposed or deliberately unexposed, and the defaults vary with the
    fit, so the signature defers each to R with ``None``.
    """
    method = f'getS3method("{generic}", "{rclass}", envir = asNamespace("dbarts"))'
    rnames = set(robjects.r(f'names(formals({method}))')) - bound
    params = mapped_params(meth, skip={'newdata'})
    assert params.keys() <= rnames
    assert rnames - params.keys() == unexposed
    for name, param in params.items():
        assert param.default is None, name


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
    rformals = robjects.r(f'names(formals({refmethods}${method}))')
    rnames = set() if rformals is robjects.NULL else set(rformals)
    params = mapped_params(getattr(dbarts.dbarts, method))
    assert params.keys() <= rnames, method
    assert rnames - params.keys() == unexposed, method
    for name, param in params.items():
        if param.default is not Parameter.empty:
            assert param.default is None, f'{method}, argument {name}'


def test_constructors_reject_unknown_arguments(data: Data) -> None:
    """Arguments outside the explicit signatures of the dots-free constructors fail.

    `bart`, `dbarts`, and `dbartsControl` have no R ``...``, so their explicit
    signatures replace it: a misspelled or package-foreign argument fails as a
    `TypeError` instead of reaching R.
    """
    with pytest.raises(TypeError, match='unexpected keyword'):
        dbarts.bart(data.x, data.y, n_trees=NTREE)  # the bart2 spelling of ntree
    with pytest.raises(TypeError, match='unexpected keyword'):
        dbarts.dbarts(data.x, data.y, ntree=NTREE)  # the bart spelling
    with pytest.raises(TypeError, match='unexpected keyword'):
        dbarts.dbartsControl(bogus=1)


def test_bart2_forwards_control_kwargs(data: Data) -> None:
    """`bart2` forwards unrecognized keyword arguments to `dbartsControl`.

    R's ``...`` reaches `dbartsControl`, so a valid control argument (here a
    deterministic `rngSeed`) is accepted, while a bogus one is rejected by R.
    """
    common = dict(n_trees=NTREE, n_burn=NSKIP, n_samples=NDPOST, verbose=False)
    fit = dbarts.bart2('y ~ x1 + x2 + x3', data=data.frame, rngSeed=1, **common)
    again = dbarts.bart2('y ~ x1 + x2 + x3', data=data.frame, rngSeed=1, **common)
    # the forwarded seed makes the single-threaded fit reproducible
    assert_array_equal(fit.yhat_train, again.yhat_train)

    with pytest.raises(RRuntimeError, match='unknown arguments'):
        dbarts.bart2('y ~ x1', data=data.frame, totallybogus=1, **common)


def test_bart_explicit_signature(data: Data) -> None:
    """The explicit `bart` signature forwards its scalar arguments to R faithfully.

    `sigest` overrides the calibrated error-SD estimate of a continuous fit,
    and `binaryOffset` shifts the latent scale of a binary fit (R fills the
    `binaryOffset` component with the per-observation value used).
    """
    n, _ = data.x.shape
    common = dict(ntree=NTREE, nskip=NSKIP, ndpost=NDPOST, verbose=False)

    sigest = 2.5
    bart = dbarts.bart(data.x, data.y, sigest=sigest, **common)
    assert bart.sigest == sigest

    offset = 0.3
    binary = dbarts.bart(data.x, data.biny, binaryOffset=offset, **common)
    assert_array_equal(binary.binaryOffset, np.full(n, offset))
