# rbartpackages/src/rbartpackages/dbarts.py
#
# Copyright (c) 2025-2026, The rbartpackages Contributors
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

"""Wrapper for the R package dbarts."""

# ruff: noqa: ANN002, ANN003

from functools import partial
from typing import Any

from jaxtyping import AbstractDtype, Float64, Int32
from numpy import ndarray
from rpy2 import robjects
from rpy2.rlike.container import NamedList
from rpy2.robjects.language import LangVector
from rpy2.robjects.methods import RS4

# WORKAROUND(python<3.11): import NotRequired, Self, TypedDict from typing
from typing_extensions import NotRequired, Self, TypedDict

from rbartpackages._base import RObjectBase, rmethod, rproperty


class String(AbstractDtype):
    """Represent a `numpy.str_` data dtype."""

    dtypes = r'<U\d+'


def formula_arg(
    args: tuple[Any, ...], kw: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Convert a string `formula` argument (positional or named) to an R formula."""
    if isinstance(kw.get('formula'), str):
        kw = dict(kw, formula=robjects.Formula(kw['formula']))
    elif args and isinstance(args[0], str):
        args = (robjects.Formula(args[0]), *args[1:])
    return args, kw


def named_vector_args(kw: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    """Convert the dict-valued arguments listed in `names` to R named numeric vectors."""
    for name in names:
        value = kw.get(name)
        if isinstance(value, dict):
            vector = robjects.FloatVector(list(value.values()))
            vector = robjects.r('setNames')(vector, list(value.keys()))
            kw = dict(kw, **{name: vector})
    return kw


class dbartsControl(RObjectBase):
    """
    Python interface to dbarts::dbartsControl.

    Wraps an R S4 object with no components exposed; pass it as the `control`
    argument of `dbarts`, which also hands it back through the
    `dbarts.control` property.
    """

    _rfuncname = 'dbarts::dbartsControl'


class dbartsData(RObjectBase):
    """
    Python interface to dbarts::dbartsData.

    A string `formula` argument is converted to an R formula; the
    backwards-compatible matrix form passes through unchanged. Wraps an R S4
    object with no components exposed; pass it to `dbarts.setData` or in
    place of the `formula` argument of the fitting interfaces.
    """

    _rfuncname = 'dbarts::dbartsData'

    def __init__(self, *args, **kw) -> None:
        args, kw = formula_arg(args, kw)
        super().__init__(*args, **kw)


class RunSamples(TypedDict):
    """Type of the return value of `dbarts.run`.

    Unlike the `bart` fit attributes, the observations are on the first axis,
    and multiple chains add a trailing `nchain` axis.
    """

    k: NotRequired[Float64[ndarray, ' ndpost'] | Float64[ndarray, 'ndpost nchain']]
    """End-node-prior `k` draws (only with a `k` hyperprior, the binary default)."""

    sigma: Float64[ndarray, ' ndpost'] | Float64[ndarray, 'ndpost nchain']
    """Error-SD draws; fixed at 1 for binary responses."""

    train: Float64[ndarray, 'n ndpost'] | Float64[ndarray, 'n ndpost nchain']
    """Training-point posterior function draws."""

    test: Float64[ndarray, 'm ndpost'] | Float64[ndarray, 'm ndpost nchain'] | None
    """Test-point posterior function draws; ``None`` without test data."""

    varcount: Int32[ndarray, 'p ndpost'] | Int32[ndarray, 'p ndpost nchain']
    """Per-draw count of splits on each variable, summed over trees."""


class dbarts(RObjectBase):
    """
    Python interface to dbarts::dbarts.

    A string `formula` argument is converted to an R formula; the
    backwards-compatible matrix form (`x_train` and `y_train` as the first
    two arguments) passes through unchanged. The named numeric vector form of
    the `proposal_probs` parameter must be specified as a dictionary in
    Python.

    The sampler is a mutable R object: the methods below modify it in place
    or return results. Its fields are exposed as read-only properties that
    read off the R object at each access, so they track the in-place updates.
    """

    _rfuncname = 'dbarts::dbarts'
    _named_vectors = ('proposal_probs',)

    def __init__(self, *args, **kw) -> None:
        args, kw = formula_arg(args, kw)
        super().__init__(*args, **named_vector_args(kw, self._named_vectors))

    @rproperty(wrap=dbartsControl)
    def control(self) -> dbartsControl:
        """The control object of the sampler, as a `dbartsControl` wrapper."""
        ...

    @rproperty(wrap=dbartsData)
    def data(self) -> dbartsData:
        """The data object of the sampler, as a `dbartsData` wrapper."""
        ...

    @rproperty
    def model(self) -> RS4:
        """The model (priors) object of the sampler (an R `dbartsModel`)."""
        ...

    @rproperty
    def state(self) -> NamedList | None:
        """The per-chain sampler states; ``None`` unless cached with ``updateState``."""
        ...

    @partial(rmethod, rname='run')
    def _run(self, *args, **kw) -> object:
        """Call the R run method; returns an R named list of draws."""
        ...

    def run(self, *args, **kw) -> RunSamples | None:
        """Run the sampler for (burn-in plus) the given number of iterations.

        The draws are returned as a dict of arrays, ``None`` if the sampler
        is run for zero samples.
        """
        out = self._run(*args, **kw)
        if out is robjects.NULL:
            return None  # R returns invisible NULL for zero samples
        return {
            str(it.name): None if it.value is robjects.NULL else it.value
            for it in out.items()
        }

    @rmethod
    def sampleTreesFromPrior(self, *args, **kw) -> object:
        """Draw the tree structures from the prior, keeping the node parameters.

        This leaves the sampler in an invalid state until the node parameters
        are drawn too.
        """
        ...

    @rmethod
    def sampleNodeParametersFromPrior(self, *args, **kw) -> object:
        """Draw the end-node parameters from the prior, keeping the trees."""
        ...

    @partial(rmethod, rname='copy')
    def _copy(self, *args, **kw) -> object:
        """Call the R copy method; returns the new sampler as an R object."""
        ...

    def copy(self, *args, **kw) -> Self:
        """Create a deep (default) or shallow copy of the sampler.

        The R method is broken once the sampler state has been cached; create
        the sampler with ``dbartsControl(updateState=False)`` to use it.
        """
        return self._wrap(self._copy(*args, **kw))

    @rmethod
    def show(self, *args, **kw) -> object:
        """Print a description of the sampler to the R console."""
        ...

    @rmethod
    def predict(self, *args, **kw) -> object:
        """Predict at new points without re-running the sampler.

        Uses the current trees, giving a single prediction per point, or each
        kept set of trees with ``keepTrees=True``.
        """
        ...

    @rmethod
    def setControl(self, *args, **kw) -> object:
        """Replace the control object of the sampler; needs `n_samples` set."""
        ...

    @rmethod
    def setModel(self, *args, **kw) -> object:
        """Replace the model (priors) object of the sampler."""
        ...

    @rmethod
    def setData(self, *args, **kw) -> object:
        """Replace the data object of the sampler (a `dbartsData`)."""
        ...

    @rmethod
    def setResponse(self, *args, **kw) -> object:
        """Replace the response vector."""
        ...

    @rmethod
    def setOffset(self, *args, **kw) -> object:
        """Replace the offset vector."""
        ...

    @rmethod
    def setSigma(self, *args, **kw) -> object:
        """Replace the per-chain residual standard deviations."""
        ...

    @rmethod
    def setPredictor(self, *args, **kw) -> object:
        """Replace the predictor matrix (or a single column).

        Unforced updates (``forceUpdate=False``, the single-column default)
        return whether the update succeeded: it fails if a tree ends up with
        an empty leaf, rolling back the change. Whole-matrix updates are
        forced by default.
        """
        ...

    @rmethod
    def setTestPredictor(self, *args, **kw) -> object:
        """Replace the test predictor matrix (or a single column)."""
        ...

    @rmethod
    def setTestPredictorAndOffset(self, *args, **kw) -> object:
        """Replace the test predictor matrix and the test offset."""
        ...

    @rmethod
    def setTestOffset(self, *args, **kw) -> object:
        """Replace the test offset vector."""
        ...

    @rmethod
    def printTrees(self, *args, **kw) -> object:
        """Print the given trees to the R console."""
        ...

    @rmethod
    def plotTree(self, *args, **kw) -> object:
        """Plot the given tree with R graphics."""
        ...


class bart(RObjectBase):
    """
    Python interface to dbarts::bart.

    The named numeric vector forms of the `splitprobs` and `proposalprobs`
    parameters must be specified as dictionaries in Python; a named
    `splitprobs` requires `x_train` to have column names (e.g., a data frame).

    In the attribute shapes below, `ndpost` counts the kept draws
    (``ndpost / keepevery``). Multiple chains are stacked into the `ndpost`
    axis when combined (``combinechains=True``, the `bart` default), and add
    a leading `nchain` axis otherwise.
    """

    _rfuncname = 'dbarts::bart'
    _named_vectors = ('splitprobs', 'proposalprobs')

    binaryOffset: Float64[ndarray, ' n'] | None = None
    """Per-observation offset on the latent probit scale (binary outcomes only)."""

    call: LangVector
    """The R call that created the fit.

    With ``keepcall=False`` this is a dummy ``NULL()`` call, not ``None``.
    """

    first_k: Float64[ndarray, ' nskip'] | Float64[ndarray, 'nchain nskip'] | None = None
    """Burn-in draws of `k` (only when `k` is given a hyperprior)."""

    first_sigma: (
        Float64[ndarray, ' nskip'] | Float64[ndarray, 'nchain nskip'] | None
    ) = None
    """Burn-in error-SD draws (continuous outcomes only)."""

    fit: dbarts | None = None
    """The sampler as a `dbarts` object, kept only with ``keeptrees`` or ``keepsampler``."""

    k: Float64[ndarray, ' ndpost'] | Float64[ndarray, 'nchain ndpost'] | None = None
    """End-node-prior `k` draws (only when `k` is given a hyperprior)."""

    n_chains: int | None = None
    """Number of MCMC chains; ``None`` when the sampler is kept in `fit`."""

    sigest: float | None = None
    """Rough residual SD used to set the sigma prior (continuous outcomes only)."""

    sigma: Float64[ndarray, ' ndpost'] | Float64[ndarray, 'nchain ndpost'] | None = None
    """Kept error-SD draws, continuous outcomes only (burn-in is in `first_sigma`)."""

    varcount: Int32[ndarray, 'ndpost p'] | Int32[ndarray, 'nchain ndpost p']
    """Per-draw count of splits on each variable, summed over trees."""

    y: Float64[ndarray, ' n'] | None = None
    """The training responses (continuous outcomes only)."""

    yhat_test: (
        Float64[ndarray, 'ndpost m'] | Float64[ndarray, 'nchain ndpost m'] | None
    ) = None
    """Test-point posterior function draws; ``None`` without test data."""

    yhat_test_mean: Float64[ndarray, ' m'] | None = None
    """Posterior mean of `yhat_test` (continuous outcomes with test data only)."""

    yhat_train: (
        Float64[ndarray, 'ndpost n'] | Float64[ndarray, 'nchain ndpost n'] | None
    ) = None
    """Training-point posterior function draws (latent probit scale for binary).

    ``None`` with ``keeptrainfits=False``.
    """

    yhat_train_mean: Float64[ndarray, ' n'] | None = None
    """Posterior mean of `yhat_train` (continuous outcomes only)."""

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **named_vector_args(kw, self._named_vectors))

        # fix up attributes
        # R fills optional components with NULL (e.g. yhat.test without test
        # data); expose them as None like the components that are dropped
        for name, value in list(vars(self).items()):
            if value is robjects.NULL:
                setattr(self, name, None)
        if self.n_chains is not None:
            self.n_chains = self.n_chains.item()
        if self.sigest is not None:
            self.sigest = self.sigest.item()
        if self.fit is not None:
            self.fit = self._wrap_fit(self.fit)

    @staticmethod
    def _wrap_fit(fit: RS4) -> dbarts:
        """Wrap the kept R sampler in the `dbarts` interface."""
        return dbarts._wrap(fit)  # noqa: SLF001, base-class access

    @rmethod
    def predict(self, *args, **kw) -> object:
        """Compute predictions at new points; requires a ``keeptrees=True`` fit.

        Returns expected values, i.e., probabilities for binary fits, unless
        the latent scale is requested with ``type='bart'``.
        """
        ...

    @rmethod
    def extract(self, *args, **kw) -> object:
        """Return the kept draws for the training (default) or test points.

        Like `predict`, the draws are on the expected-value scale by default.
        With ``type='trees'`` (requires ``keeptrees=True``), return the tree
        structures as a data frame instead.
        """
        ...

    @rmethod
    def fitted(self, *args, **kw) -> object:
        """Return the posterior mean for the training (default) or test points."""
        ...


class bart2(bart):
    """
    Python interface to dbarts::bart2.

    A string `formula` argument is converted to an R formula. The named
    numeric vector forms of the `split_probs` and `proposal_probs` parameters
    must be specified as dictionaries in Python.

    Unlike `bart`, by default this runs multiple chains without combining
    them, which adds a leading `nchain` axis to the draws attributes.
    """

    _rfuncname = 'dbarts::bart2'
    _named_vectors = ('split_probs', 'proposal_probs')

    def __init__(self, *args, **kw) -> None:
        args, kw = formula_arg(args, kw)
        super().__init__(*args, **kw)


class rbart_vi(bart2):
    """
    Python interface to dbarts::rbart_vi.

    A string `formula` argument is converted to an R formula. In addition to
    the `bart` components, the fit exposes the random-intercept outputs below.
    """

    _rfuncname = 'dbarts::rbart_vi'
    _named_vectors = ()

    callback: (
        Float64[ndarray, 'ndpost values']
        | Float64[ndarray, 'nchain ndpost values']
        | None
    ) = None
    """Stacked per-draw results of the `callback` function, if given."""

    first_tau: Float64[ndarray, ' nskip'] | Float64[ndarray, 'nchain nskip']
    """Burn-in draws of the random-effects SD."""

    fit: tuple[dbarts, ...] | None = None
    """The per-chain samplers as `dbarts` objects.

    Kept only with ``keepTrees`` or ``keepSampler`` (both on by default).
    """

    group_by: String[ndarray, ' n']
    """The training grouping factor, as the level name of each observation."""

    group_by_test: String[ndarray, ' m'] | None = None
    """The test grouping factor, if given."""

    ranef: Float64[ndarray, 'ndpost g'] | Float64[ndarray, 'nchain ndpost g']
    """Random-intercept draws for each of the `g` groups."""

    ranef_mean: Float64[ndarray, ' g']
    """Posterior mean of `ranef` per group."""

    seed: Int32[ndarray, ' state']
    """R RNG state used by `predict` to draw the effects of unseen groups."""

    tau: Float64[ndarray, ' ndpost'] | Float64[ndarray, 'nchain ndpost']
    """Random-effects SD draws."""

    y: Float64[ndarray, ' n']
    """The training responses."""

    @staticmethod
    def _wrap_fit(fit: NamedList) -> tuple[dbarts, ...]:
        """Wrap the R list of per-chain samplers in the `dbarts` interface."""
        return tuple(map(dbarts._wrap, fit))  # noqa: SLF001, base-class access
