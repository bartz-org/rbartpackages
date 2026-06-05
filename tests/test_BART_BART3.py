# rbartpackages/tests/test_BART_BART3.py
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

"""Tests for the BART and BART3 wrappers.

BART3 is a fork of BART with mostly the same interface, so the two wrappers
share their tests: the `pkg` fixture runs each test once per wrapper module,
with conditionals where the packages diverge (bugs specific to BART, features
added in BART3). Tests of BART-only behavior are not parametrized and import
the module directly.
"""

import math
from types import ModuleType

import numpy as np
import pytest

from tests.util import assert_array_equal, assert_close_matrices, import_or_skip

pd = import_or_skip('pandas')

NDPOST = 20
NSKIP = 20
NTREE = 10


@pytest.fixture(scope='module', params=['BART', 'BART3'])
def pkg(request: pytest.FixtureRequest) -> ModuleType:
    """Return the wrapper module under test."""
    return import_or_skip(f'rbartpackages.{request.param}')


def is_BART3(pkg: ModuleType) -> bool:
    """Tell the two wrapper modules apart in conditional test logic."""
    return pkg.__name__ == 'rbartpackages.BART3'


def make_data(rng: np.random.Generator, n: int = 30, p: int = 3) -> tuple:
    """Generate a small regression dataset."""
    x_train = rng.standard_normal((n, p))
    y_train = x_train[:, 0] + 0.1 * rng.standard_normal(n)
    return x_train, y_train


def test_docstring(pkg: ModuleType) -> None:
    """The R documentation is attached to the wrapper class."""
    assert 'R documentation' in pkg.gbart.__doc__


def test_gbart_no_test_data(rng: np.random.Generator, pkg: ModuleType) -> None:
    """Without `x_test` the derived test-set attributes are left unset.

    The fit also has a constant column (dropped, reported through `rm_const`),
    no burn-in (BART computes `LPML` anyway, BART3 does not), and thinning
    (BART's `sigma` keeps the thinned-away draws too, BART3's drops them).
    """
    x_train, y_train = make_data(rng)
    n, p = x_train.shape
    # R reports the dropped constant column as a negative rm.const index; the
    # wrapper turns that into the kept 0-based column indices
    x_train = np.insert(x_train, 1, 1.0, axis=1)
    keepevery = 2
    bart = pkg.gbart(
        x_train=x_train,
        y_train=y_train,
        ntree=NTREE,
        nskip=0,
        ndpost=NDPOST,
        keepevery=keepevery,
    )
    assert bart.ndpost == NDPOST
    assert bart.yhat_train.shape == (NDPOST, n)
    assert bart.yhat_train_mean.shape == (n,)
    assert_array_equal(bart.rm_const, np.array([0, 2, 3], np.int32))
    assert bart.varcount.shape == (NDPOST, p)
    assert isinstance(bart.sigma_mean, float)
    if is_BART3(pkg):
        assert bart.LPML is None  # only computed with burn-in
        assert bart.sigma.shape == (NDPOST,)  # thinned-away draws dropped
        assert bart.x_test is None
    else:
        assert isinstance(bart.LPML, float)  # computed even without burn-in
        assert bart.sigma.shape == (NDPOST * keepevery,)  # one draw per iteration
        assert_array_equal(bart.hostname, np.array([False]))

    yhat = bart.predict(x_train[:, bart.rm_const])
    assert yhat.shape == (NDPOST, n)
    assert_close_matrices(yhat.mean(axis=0), bart.yhat_train_mean, rtol=1e-5)

    # R's cgbart still returns yhat.test (empty), while the derived test
    # attributes are left unset; the wrapper mirrors both.
    assert bart.yhat_test.shape == (NDPOST, 0)
    assert bart.yhat_test_mean is None


def fit_gbart_test_data(
    rng: np.random.Generator, pkg: ModuleType, binary: bool
) -> tuple:
    """Fit `gbart` with test data and check the outputs common to both paths."""
    x_train, y_train = make_data(rng)
    n, p = x_train.shape
    x_test = rng.standard_normal((7, p))
    m, _ = x_test.shape
    if binary:
        y_train = (y_train > np.median(y_train)).astype(float)
    kw = dict() if is_BART3(pkg) else dict(hostname=True)  # BART3 dropped hostname
    bart = pkg.gbart(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        type='pbart' if binary else 'wbart',
        ntree=NTREE,
        nskip=NSKIP,
        ndpost=NDPOST,
        **kw,
    )
    assert bart.ndpost == NDPOST
    assert bart.yhat_train.shape == (NDPOST, n)
    assert bart.yhat_test.shape == (NDPOST, m)
    assert isinstance(bart.LPML, float)
    if is_BART3(pkg):
        assert bart.x_test.shape == x_test.shape
    else:
        assert bart.hostname.shape == (1,)
        assert bart.hostname.dtype.kind == 'U'  # the fitting machine's hostname
    return bart, x_test, n, m


def test_gbart_test_data_continuous(rng: np.random.Generator, pkg: ModuleType) -> None:
    """Passing `x_test` populates the test-set outputs of a `wbart` fit.

    `predict` returns the bare draws matrix; BART3 additionally exposes
    posterior quantiles and the prior's `sigest`.
    """
    bart, x_test, n, m = fit_gbart_test_data(rng, pkg, binary=False)
    assert bart.yhat_train_mean.shape == (n,)
    assert bart.yhat_test_mean.shape == (m,)
    assert bart.sigma.shape == (NSKIP + NDPOST,)
    assert isinstance(bart.sigma_mean, float)
    assert bart.prob_train is None
    assert bart.prob_test is None
    if is_BART3(pkg):
        assert bart.yhat_test_lower.shape == (m,)
        assert bart.yhat_test_upper.shape == (m,)
        assert np.all(bart.yhat_test_lower <= bart.yhat_test_upper)
        assert isinstance(bart.sigest, float)
        assert math.isfinite(bart.sigest)

    pred = bart.predict(x_test)
    assert pred.shape == (NDPOST, m)
    assert_close_matrices(pred.mean(axis=0), bart.yhat_test_mean, rtol=1e-5)


def test_gbart_test_data_binary(rng: np.random.Generator, pkg: ModuleType) -> None:
    """Passing `x_test` populates the test-set outputs of a `pbart` fit.

    The probability-scale `prob_*` outputs replace the derived `yhat_*_mean`
    ones; BART3 additionally exposes posterior quantiles.
    """
    bart, x_test, n, m = fit_gbart_test_data(rng, pkg, binary=True)
    assert bart.prob_train.shape == (NDPOST, n)
    assert bart.prob_train_mean.shape == (n,)
    assert bart.prob_test.shape == (NDPOST, m)
    assert bart.prob_test_mean.shape == (m,)
    assert bart.sigma is None
    assert bart.sigma_mean is None
    assert bart.yhat_train_mean is None  # R sets prob.train.mean instead
    assert bart.yhat_test_mean is None
    if is_BART3(pkg):
        assert bart.prob_test_lower.shape == (m,)
        assert bart.prob_test_upper.shape == (m,)
        assert np.all(bart.prob_test_lower <= bart.prob_test_upper)
        assert bart.sigest is None  # not estimated for binary outcomes

    # R's predict for binary fits returns a list; the wrapper exposes it as
    # a dict of arrays (continuous fits return a bare matrix instead).
    pred = bart.predict(x_test)
    assert isinstance(pred, dict)
    expected_keys = ['binaryOffset', 'prob_test', 'prob_test_mean', 'yhat_test']
    if is_BART3(pkg):
        # BART3 adds the prob.test.lower/upper quantiles to the output
        expected_keys += ['prob_test_lower', 'prob_test_upper']
        assert np.all(pred['prob_test_lower'] <= pred['prob_test_upper'])
    assert sorted(pred) == sorted(expected_keys)
    assert pred['yhat_test'].shape == (NDPOST, m)
    assert pred['prob_test'].shape == (NDPOST, m)
    assert pred['prob_test_mean'].shape == (m,)
    assert isinstance(pred['binaryOffset'], float)


@pytest.mark.timeout(180)
def test_mc_gbart_multicore(rng: np.random.Generator, pkg: ModuleType) -> None:
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
    warm = pkg.gbart(
        x_train=x_train, y_train=y_train, ntree=NTREE, nskip=NSKIP, ndpost=NDPOST
    )
    warm.predict(x_train)

    mc_cores = 2
    bart = pkg.mc_gbart(
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
    assert bart.yhat_train_mean.shape == (n,)
    assert isinstance(bart.LPML, float)
    assert bart.sigma.shape == (NSKIP + bart.ndpost // mc_cores, mc_cores)
    # without test data R does not even combine the chains' empty yhat.test
    assert bart.yhat_test.shape == (bart.ndpost // mc_cores, 0)
    if is_BART3(pkg):
        assert bart.chains == mc_cores
        # mc.gbart overwrites sigest with its logical-NA default; __init__ -> nan.
        assert isinstance(bart.sigest, float)
        assert math.isnan(bart.sigest)
    else:
        assert_array_equal(bart.hostname, np.array([False, False]))

    # predict with mc_cores > 1 forks via mc.pwbart; unlike mc.gbart's fork the
    # children run single-threaded, so this stays deadlock-free without a guard.
    yhat = bart.predict(x_train, **{'mc.cores': mc_cores})
    assert yhat.shape == (bart.ndpost, n)


@pytest.mark.timeout(180)
def test_mc_gbart_binary(rng: np.random.Generator) -> None:
    """`mc.gbart` with binary outcomes leaves some outputs uncombined.

    R combines `yhat_*` across the chains but forgets `prob_*`, which keep the
    first chain's draws only. The fit also has a constant column: R drops it
    (negative `rm_const`, fixed up by the wrapper) but then miscounts the kept
    columns and fails to update the serialized-ensemble header, so `predict`
    returns the first chain's draws only. These bugs are BART-specific, so the
    test is not shared with BART3.
    """
    BART = import_or_skip('rbartpackages.BART')
    x_train, y_train = make_data(rng)
    n, p = x_train.shape
    y_train = (y_train > np.median(y_train)).astype(float)
    x_test = rng.standard_normal((7, p))
    m, _ = x_test.shape
    x_train = np.insert(x_train, 1, 1.0, axis=1)
    x_test = np.insert(x_test, 1, 1.0, axis=1)

    mc_cores = 2
    bart = BART.mc_gbart(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        type='pbart',
        ntree=NTREE,
        nskip=NSKIP,
        ndpost=NDPOST,
        mc_cores=mc_cores,
    )
    chain_ndpost = bart.ndpost // mc_cores
    assert bart.ndpost == mc_cores * math.ceil(NDPOST / mc_cores)
    assert_array_equal(bart.rm_const, np.array([0, 2, 3], np.int32))
    assert bart.yhat_train.shape == (bart.ndpost, n)
    assert bart.yhat_test.shape == (bart.ndpost, m)
    assert bart.prob_train.shape == (chain_ndpost, n)
    assert bart.prob_test.shape == (chain_ndpost, m)
    assert bart.prob_train_mean.shape == (n,)
    assert bart.prob_test_mean.shape == (m,)
    assert bart.sigma is None
    assert bart.yhat_test_mean is None  # R sets prob.test.mean instead

    pred = bart.predict(x_test[:, bart.rm_const])
    assert pred['yhat_test'].shape == (chain_ndpost, m)  # broken trees header


@pytest.mark.parametrize('factor', [False, True], ids=['matrix', 'dataframe'])
@pytest.mark.parametrize('numcut', [0, 3])
def test_bartModelMatrix(pkg: ModuleType, numcut: int, factor: bool) -> None:
    """``numcut=0`` returns a bare matrix; ``numcut>0`` adds cutpoint metadata.

    A data-frame input gets its factor column expanded into one indicator
    column per level; the expansion is reported in `grp` with different
    conventions (BART maps each output column to its input column, BART3
    stores the group sizes).
    """
    x = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [3.0, 8.0]])
    if factor:
        arg = pd.DataFrame(
            {'a': x[:, 0], 'b': x[:, 1], 'c': pd.Categorical(['u', 'v', 'u', 'v'])}
        )
        x = np.c_[x, [1, 0, 1, 0], [0, 1, 0, 1]]
    else:
        arg = x
    _, p = x.shape
    out = pkg.bartModelMatrix(arg, numcut=numcut)
    if numcut == 0:
        assert isinstance(out, np.ndarray)
        assert not isinstance(out, pkg.bartModelMatrix)
        assert_close_matrices(out, x)
    else:
        assert isinstance(out, pkg.bartModelMatrix)
        assert_close_matrices(out.X, x)
        # binary indicators have a single midpoint cut, the rest get numcut
        expected_numcut = [numcut, numcut, 1, 1] if factor else numcut
        assert_array_equal(out.numcut, expected_numcut, strict=False)
        assert_array_equal(out.rm_const, np.arange(p, dtype=np.int32))
        assert out.xinfo.shape == (p, numcut)
        if not factor:
            # no grp: matrix input for BART, no factor columns for BART3
            assert out.grp is None
        elif is_BART3(pkg):
            # number of indicator columns each input column expands to
            assert_array_equal(out.grp, [1, 1, 2, 2], strict=False)
        else:
            # 1-based index of the input column each output column comes from
            assert_array_equal(out.grp, [1, 2, 3, 3], strict=False)


@pytest.mark.parametrize('removed', [False, True], ids=['detect', 'remove'])
def test_bartModelMatrix_constant(pkg: ModuleType, removed: bool) -> None:
    """A detected-constant column becomes a gap in the 0-based `rm_const`.

    R reports it as a negative 1-based index whether or not it is removed from
    `X` (the ``rm.const`` flag); the wrapper resolves both cases to the indices
    of the non-constant pre-removal columns.
    """
    x = np.array([[1.0, 1.0, 5.0], [2.0, 1.0, 6.0], [3.0, 1.0, 7.0]])
    n, p = x.shape
    out = pkg.bartModelMatrix(x, numcut=3, rm_const=removed)
    assert_array_equal(out.rm_const, np.array([0, 2], np.int32))
    assert out.X.shape == (n, p - 1 if removed else p)
