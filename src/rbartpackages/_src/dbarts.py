# rbartpackages/src/rbartpackages/_src/dbarts.py
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

"""Implementation of `rbartpackages.dbarts`."""

from functools import partial
from typing import Literal, cast, no_type_check

from jaxtyping import Float64, Int32, Integer
from numpy import ndarray
from rpy2 import robjects
from rpy2.rlike.container import NamedList
from rpy2.robjects.language import LangVector
from rpy2.robjects.methods import RS4

# WORKAROUND(python<3.11): import NotRequired, Self, TypedDict from typing
from typing_extensions import NotRequired, Self, TypedDict

from rbartpackages._src.base import (
    DataFrame,
    RObjectBase,
    String,
    drop_none,
    namedlist_to_dict,
    robjects_r,
    rproperty,
)


def to_formula(formula: object) -> object:
    """
    Convert a string `formula` argument to an R formula; pass anything else through.

    The matrix interface (a `formula` that is really `x_train`) is left
    untouched.
    """
    return robjects.Formula(formula) if isinstance(formula, str) else formula


def to_named_vector(value: object) -> object:
    """
    Convert a dict argument to an R named numeric vector; pass anything else through.

    R takes the named-vector forms of `splitprobs`/`proposalprobs` (and their
    dotted `bart2` spellings) as a way to set per-variable or per-proposal
    values; in Python they are given as dictionaries.
    """
    if isinstance(value, dict):
        vector = robjects.FloatVector(list(value.values()))
        return robjects_r('setNames')(vector, list(value.keys()))
    return value


class dbartsControl(RObjectBase):
    """
    Configure a `dbarts` sampler.

    Python interface to R's ``dbarts::dbartsControl``, which bundles the
    sampler settings into an R S4 object with no components exposed; pass it
    as the `control` argument of `dbarts`, which also hands it back through
    the `dbarts.control` property. Arguments left to ``None`` are omitted from
    the R call, so R computes its own defaults, described below.

    Parameters
    ----------
    verbose
        Whether the sampler prints to the R console as it runs.
    keepTrainingFits
        Whether the training-point fits are returned when the sampler runs;
        they are always computed, so disabling only drops them from the
        output.
    useQuantiles
        Whether the tree decision rules use empirical quantiles of each
        predictor rather than values spaced uniformly over its range.
    keepTrees
        Whether the sampled trees are cached as drawn (``n_trees * n_samples``
        of them), which `dbarts.predict` and `bart` tree extraction require;
        memory-intensive.
    n_samples
        Default number of samples returned per run; usually set through
        `dbarts` and overridable per `dbarts.run`.
    n_cuts
        Number of decision rules per predictor (a scalar recycled over the
        predictors, or one value each); fewer may be used for a predictor with
        few unique values.
    n_burn
        Number of samples discarded at the start of a run.
    n_trees
        Number of trees in the sum-of-trees.
    n_chains
        Number of independent chains.
    n_threads
        Number of threads for internal calculations and chains; default the
        detected core count. Single-threaded is often faster below ~10k
        observations.
    n_thin
        Number of tree-only iterations between recorded samples (thinning).
    printEvery
        Interval, in post-thinning samples, of the progress messages (with
        `verbose`).
    printCutoffs
        Number of a variable's decision rules printed in verbose mode.
    rngKind
        Random-number-generator kind, as in R's ``set.seed``.
    rngNormalKind
        Random-number-generator normal kind, as in R's ``set.seed``.
    rngSeed
        Random-number-generator seed; ``None`` (R's ``NA``) seeds from the
        clock when applicable.
    updateState
        Default for whether the methods refresh the object's cached state,
        which is only needed to save/load a sampler.
    """

    _rfuncname = 'dbarts::dbartsControl'

    def __init__(
        self,
        *,
        verbose: bool = False,
        keepTrainingFits: bool = True,
        useQuantiles: bool = False,
        keepTrees: bool = False,
        n_samples: int | None = None,
        n_cuts: int | Integer[ndarray, ' p'] = 100,
        n_burn: int = 200,
        n_trees: int = 75,
        n_chains: int = 4,
        n_threads: int | None = None,
        n_thin: int = 1,
        printEvery: int = 100,
        printCutoffs: int = 0,
        rngKind: str = 'default',
        rngNormalKind: str = 'default',
        rngSeed: int | None = None,
        updateState: bool = True,
    ) -> None:
        kw = {
            'verbose': verbose,
            'keepTrainingFits': keepTrainingFits,
            'useQuantiles': useQuantiles,
            'keepTrees': keepTrees,
            'n.samples': n_samples,
            'n.cuts': n_cuts,
            'n.burn': n_burn,
            'n.trees': n_trees,
            'n.chains': n_chains,
            'n.threads': n_threads,
            'n.thin': n_thin,
            'printEvery': printEvery,
            'printCutoffs': printCutoffs,
            'rngKind': rngKind,
            'rngNormalKind': rngNormalKind,
            'rngSeed': rngSeed,
            'updateState': updateState,
        }
        super().__init__(**drop_none(kw))


class dbartsData(RObjectBase):
    """
    Bundle the data of a `dbarts` sampler.

    Python interface to R's ``dbarts::dbartsData``. A string `formula`
    argument is converted to an R formula; the backwards-compatible matrix
    form (`formula`/`data` as the `x_train`/`y_train` pair) passes through
    unchanged. Wraps an R S4 object with no components exposed; pass it to
    `dbarts.setData` or in place of the `formula` argument of the fitting
    interfaces. Arguments left to ``None`` are omitted from the R call.

    Parameters
    ----------
    formula
        A model formula (as a string), or, in matrix mode, the `x_train`
        predictor matrix.
    data
        The data frame the `formula` refers to, or, in matrix mode, the
        `y_train` response vector.
    test
        Test predictors, with the same columns as the training data.
    subset
        Subset of observations to keep.
    weights
        Per-observation weights; the model becomes ``y | x ~ N(f(x),
        sigma^2 / w)``.
    offset
        Offset added to ``f(x)``; useful for binary responses, where ``P(Y =
        1 | x) = Phi(f(x) + offset)``.
    offset_test
        The `offset` for the test data; defaults to `offset` when applicable.
    """

    _rfuncname = 'dbarts::dbartsData'

    def __init__(
        self,
        formula: str | Float64[ndarray, 'n p'] | DataFrame,
        data: Float64[ndarray, ' n'] | DataFrame | None = None,
        *,
        test: Float64[ndarray, 'm p'] | DataFrame | None = None,
        subset: Integer[ndarray, ' k'] | None = None,
        weights: Float64[ndarray, ' n'] | None = None,
        offset: Float64[ndarray, ' n'] | float | None = None,
        offset_test: Float64[ndarray, ' m'] | float | None = None,
    ) -> None:
        kw = {
            'formula': to_formula(formula),
            'data': data,
            'test': test,
            'subset': subset,
            'weights': weights,
            'offset': offset,
            'offset.test': offset_test,
        }
        super().__init__(**drop_none(kw))


class RunSamples(TypedDict):
    """
    Type of the return value of `dbarts.run`.

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
    Create a low-level `dbarts` sampler.

    Python interface to R's ``dbarts::dbarts``, a mutable reference-class
    sampler that can be run, stopped, restarted, and modified in place. A
    string `formula` argument is converted to an R formula; the
    backwards-compatible matrix form (`formula`/`data` as the
    `x_train`/`y_train` pair) passes through unchanged. The named numeric
    vector form of `proposal_probs` is given as a dictionary in Python.
    Arguments left to ``None`` are omitted from the R call, so R computes its
    own defaults, described below.

    The methods below modify the sampler in place or return results. Its
    fields are exposed as read-only properties read off the R object at each
    access, so they track the in-place updates.

    Parameters
    ----------
    formula
        A model formula (as a string), or, in matrix mode, the `x_train`
        predictor matrix.
    data
        The data frame the `formula` refers to, or, in matrix mode, the
        `y_train` response vector.
    test
        Test predictors, with the same columns as the training data.
    subset
        Subset of observations to keep.
    weights
        Per-observation weights; the model becomes ``y | x ~ N(f(x),
        sigma^2 / w)``.
    offset
        Offset added to ``f(x)``; useful for binary responses, where ``P(Y =
        1 | x) = Phi(f(x) + offset)``.
    offset_test
        The `offset` for the test data; defaults to `offset` when applicable.
    verbose
        Whether additional output is printed to the R console.
    n_samples
        Default number of posterior samples per run, overridable in
        `dbarts.run`; ``None`` keeps the `control` value (R's default 800
        otherwise). Passing it overrides `control`.
    tree_prior
        Tree-structure prior, as an R expression of the form ``cgm`` or
        ``cgm(power, base, split.probs)``.
    node_prior
        End-node prior, as an R expression of the form ``normal`` or
        ``normal(k)``.
    resid_prior
        Residual-variance prior, as an R expression of the form ``chisq`` or
        ``chisq(df, quant)``.
    proposal_probs
        Tree-proposal probabilities, as a dict with keys ``'birth_death'``,
        ``'change'``, ``'swap'`` (the proposal frequencies) and ``'birth'``
        (the birth/death split).
    control
        A `dbartsControl` configuring the sampler.
    sigma
        Estimate of the residual SD; ``None`` derives it from a linear fit.

    Notes
    -----
    The `tree_prior`, `node_prior`, and `resid_prior` arguments are evaluated
    by R with non-standard scoping; to depart from the defaults pass an R
    language object (e.g. from `rpy2.robjects.r`).
    """

    _rfuncname = 'dbarts::dbarts'

    def __init__(
        self,
        formula: str | Float64[ndarray, 'n p'] | DataFrame,
        data: Float64[ndarray, ' n'] | DataFrame | None = None,
        *,
        test: Float64[ndarray, 'm p'] | DataFrame | None = None,
        subset: Integer[ndarray, ' k'] | None = None,
        weights: Float64[ndarray, ' n'] | None = None,
        offset: Float64[ndarray, ' n'] | float | None = None,
        offset_test: Float64[ndarray, ' m'] | float | None = None,
        verbose: bool = False,
        n_samples: int | None = None,
        tree_prior: object | None = None,
        node_prior: object | None = None,
        resid_prior: object | None = None,
        proposal_probs: dict[str, float] | Float64[ndarray, ' 4'] | None = None,
        control: dbartsControl | None = None,
        sigma: float | None = None,
    ) -> None:
        kw = {
            'formula': to_formula(formula),
            'data': data,
            'test': test,
            'subset': subset,
            'weights': weights,
            'offset': offset,
            'offset.test': offset_test,
            'verbose': verbose,
            'n.samples': n_samples,
            'tree.prior': tree_prior,
            'node.prior': node_prior,
            'resid.prior': resid_prior,
            'proposal.probs': to_named_vector(proposal_probs),
            'control': control,
            'sigma': sigma,
        }
        super().__init__(**drop_none(kw))

    @partial(rproperty, wrap=dbartsControl)
    @no_type_check
    def control(self) -> dbartsControl:
        """Return the control object of the sampler, as a `dbartsControl` wrapper."""
        ...

    @partial(rproperty, wrap=dbartsData)
    @no_type_check
    def data(self) -> dbartsData:
        """Return the data object of the sampler, as a `dbartsData` wrapper."""
        ...

    @rproperty
    @no_type_check
    def model(self) -> RS4:
        """The model (priors) object of the sampler (an R ``dbartsModel``)."""
        ...

    @rproperty
    def state(self) -> NamedList | None:
        """The per-chain sampler states; ``None`` unless cached with ``updateState``."""
        ...

    def run(
        self,
        numBurnIn: int | None = None,
        numSamples: int | None = None,
        *,
        updateState: bool | None = None,
        numThreads: int | None = None,
    ) -> RunSamples | None:
        """
        Run the sampler for `numBurnIn` burn-in plus `numSamples` kept iterations.

        Either count left to ``None`` is filled from the `control` object. The
        draws are returned as a dict of arrays, ``None`` if the sampler is run
        for zero samples.

        Parameters
        ----------
        numBurnIn
            Number of burn-in iterations to discard.
        numSamples
            Number of posterior samples to keep.
        updateState
            Whether to refresh the cached state after the run.
        numThreads
            Number of threads to use for the run.

        Returns
        -------
        The draws as a dict of arrays, or ``None`` for a zero-sample run.
        """
        kw = {
            'numBurnIn': numBurnIn,
            'numSamples': numSamples,
            'updateState': updateState,
            'numThreads': numThreads,
        }
        out = self._call_rmethod('run', **drop_none(kw))
        if out is robjects.NULL:
            return None  # R returns invisible NULL for zero samples
        return cast(RunSamples, namedlist_to_dict(out))

    def sampleTreesFromPrior(self, *, updateState: bool | None = None) -> None:
        """
        Draw the tree structures from the prior, keeping the node parameters.

        This leaves the sampler in an invalid state until the node parameters
        are drawn too.

        Parameters
        ----------
        updateState
            Whether to refresh the sampler's cached state afterwards.
        """
        self._call_rmethod(
            'sampleTreesFromPrior', **drop_none({'updateState': updateState})
        )

    def sampleNodeParametersFromPrior(self, *, updateState: bool | None = None) -> None:
        """
        Draw the end-node parameters from the prior, keeping the trees.

        Parameters
        ----------
        updateState
            Whether to refresh the sampler's cached state afterwards.
        """
        self._call_rmethod(
            'sampleNodeParametersFromPrior', **drop_none({'updateState': updateState})
        )

    def copy(self, *, shallow: bool | None = None) -> Self:
        """
        Create a deep (default) or shallow copy of the sampler.

        The R method is broken once the sampler state has been cached; create
        the sampler with ``dbartsControl(updateState=False)`` to use it.

        Parameters
        ----------
        shallow
            Whether the copy shares the sampler's underlying data rather than
            holding its own.

        Returns
        -------
        A new wrapped sampler that runs independently.
        """
        out = self._call_rmethod('copy', **drop_none({'shallow': shallow}))
        return self._wrap(out)

    def show(self) -> None:
        """Print a description of the sampler to the R console."""
        self._call_rmethod('show')

    def predict(
        self,
        x_test: Float64[ndarray, 'm p'] | DataFrame,
        offset_test: Float64[ndarray, ' m'] | float | None = None,
        *,
        n_threads: int | None = None,
    ) -> Float64[ndarray, ' m'] | Float64[ndarray, 'm ndpost']:
        """
        Predict at new points without re-running the sampler.

        Uses the current trees, giving a single prediction per point, or each
        kept set of trees with a ``keepTrees`` control.

        Parameters
        ----------
        x_test
            New test predictors, with the same columns as the model.
        offset_test
            Offset for the new points.
        n_threads
            Number of threads to use; chains are predicted in parallel if more
            than one.

        Returns
        -------
        One prediction per point, or the kept-tree draws with a ``keepTrees`` control.
        """
        kw = {'offset.test': offset_test, 'n.threads': n_threads}
        return self._call_rmethod('predict', x_test, **drop_none(kw))

    def setControl(self, newControl: dbartsControl) -> None:
        """
        Replace the control object of the sampler; needs `n_samples` set.

        Parameters
        ----------
        newControl
            The replacement `dbartsControl`.
        """
        self._call_rmethod('setControl', newControl)

    def setModel(self, newModel: RS4) -> None:
        """
        Replace the model (priors) object of the sampler.

        Parameters
        ----------
        newModel
            The replacement R ``dbartsModel`` (e.g. another sampler's `model`).
        """
        self._call_rmethod('setModel', newModel)

    def setData(self, newData: dbartsData, *, updateState: bool | None = None) -> None:
        """
        Replace the data object of the sampler (a `dbartsData`).

        Parameters
        ----------
        newData
            The replacement `dbartsData`.
        updateState
            Whether to refresh the sampler's cached state afterwards.
        """
        self._call_rmethod(
            'setData', newData, **drop_none({'updateState': updateState})
        )

    def setResponse(
        self, y: Float64[ndarray, ' n'], *, updateState: bool | None = None
    ) -> None:
        """
        Replace the response vector.

        Parameters
        ----------
        y
            The replacement response, of the sampler's number of observations.
        updateState
            Whether to refresh the sampler's cached state afterwards.
        """
        self._call_rmethod('setResponse', y, **drop_none({'updateState': updateState}))

    def setOffset(
        self,
        offset: Float64[ndarray, ' n'] | float | None,
        *,
        updateScale: bool | None = None,
        updateState: bool | None = None,
    ) -> None:
        """
        Replace the offset vector.

        Parameters
        ----------
        offset
            The replacement offset (a scalar is expanded to all observations),
            or ``None`` to clear it.
        updateScale
            Whether BART's internal scale updates with the new offset; only
            valid during burn-in.
        updateState
            Whether to refresh the sampler's cached state afterwards.
        """
        kw = {'updateScale': updateScale, 'updateState': updateState}
        self._call_rmethod('setOffset', offset, **drop_none(kw))

    def setSigma(
        self,
        sigma: Float64[ndarray, ' nchain'] | float,
        *,
        updateState: bool | None = None,
    ) -> None:
        """
        Replace the per-chain residual standard deviations.

        Parameters
        ----------
        sigma
            The replacement residual SD, one per chain.
        updateState
            Whether to refresh the sampler's cached state afterwards.
        """
        self._call_rmethod('setSigma', sigma, **drop_none({'updateState': updateState}))

    def setPredictor(
        self,
        x: Float64[ndarray, 'n cols'] | Float64[ndarray, ' n'],
        column: int | String[ndarray, ' cols'] | None = None,
        forceUpdate: bool | None = None,
        *,
        updateCutPoints: bool | None = None,
        updateState: bool | None = None,
    ) -> Int32[ndarray, ' 1'] | None:
        """
        Replace the predictor matrix (or the 1-based `column`).

        Unforced updates (``forceUpdate=False``, the single-column default)
        return whether the update succeeded: it fails if a tree ends up with
        an empty leaf, rolling back the change. Whole-matrix updates are
        forced by default.

        Parameters
        ----------
        x
            The replacement predictors: a whole matrix, or a single column's
            values when `column` is given.
        column
            The 1-based index or name of the single column to replace; the
            whole matrix is replaced if omitted.
        forceUpdate
            Whether to keep the update even if it leaves a tree with an empty
            leaf; default ``True`` for a whole matrix, ``False`` for a column.
        updateCutPoints
            Whether to recompute the decision-rule cutpoints from the new
            predictors.
        updateState
            Whether to refresh the sampler's cached state afterwards.

        Returns
        -------
        Whether the update succeeded for an unforced update, else ``None``.
        """
        kw = {
            'forceUpdate': forceUpdate,
            'updateCutPoints': updateCutPoints,
            'updateState': updateState,
        }
        args = () if column is None else (column,)
        return self._call_rmethod('setPredictor', x, *args, **drop_none(kw))

    def setTestPredictor(
        self,
        x_test: Float64[ndarray, 'm cols'] | Float64[ndarray, ' m'],
        column: int | String[ndarray, ' cols'] | None = None,
    ) -> None:
        """
        Replace the test predictor matrix (or the 1-based `column`).

        Parameters
        ----------
        x_test
            The replacement test predictors: a whole matrix, or a single
            column's values when `column` is given.
        column
            The 1-based index or name of the single column to replace; the
            whole matrix is replaced if omitted.
        """
        args = () if column is None else (column,)
        self._call_rmethod('setTestPredictor', x_test, *args)

    def setTestPredictorAndOffset(
        self,
        x_test: Float64[ndarray, 'm p'],
        offset_test: Float64[ndarray, ' m'] | float | None,
    ) -> None:
        """
        Replace the test predictor matrix and the test offset.

        Parameters
        ----------
        x_test
            The replacement test predictor matrix.
        offset_test
            The replacement test offset (a scalar is expanded to all test
            points), or ``None`` to clear it.
        """
        self._call_rmethod('setTestPredictorAndOffset', x_test, offset_test)

    def setTestOffset(self, offset_test: Float64[ndarray, ' m'] | float | None) -> None:
        """
        Replace the test offset vector.

        Parameters
        ----------
        offset_test
            The replacement test offset (a scalar is expanded to all test
            points), or ``None`` to clear it.
        """
        self._call_rmethod('setTestOffset', offset_test)

    def printTrees(
        self,
        treeNums: int | Integer[ndarray, ' t'],
        chainNums: int | Integer[ndarray, ' c'] | None = None,
        sampleNums: int | Integer[ndarray, ' s'] | None = None,
    ) -> None:
        """
        Print the given trees to the R console.

        Parameters
        ----------
        treeNums
            1-based indices of the trees to print.
        chainNums
            1-based indices of the chains to print; all chains if omitted.
        sampleNums
            1-based indices of the samples to print; the current trees if
            omitted.
        """
        kw = {'chainNums': chainNums, 'sampleNums': sampleNums}
        self._call_rmethod('printTrees', treeNums, **drop_none(kw))

    def plotTree(
        self,
        treeNum: int,
        chainNum: int | None = None,
        sampleNum: int | None = None,
        *,
        treePlotPars: dict[str, float] | None = None,
    ) -> None:
        """
        Plot the given tree with R graphics.

        Parameters
        ----------
        treeNum
            1-based index of the tree to plot.
        chainNum
            1-based index of the chain to plot from.
        sampleNum
            1-based index of the sample to plot from.
        treePlotPars
            Plot geometry, as a dict with keys ``'nodeHeight'``,
            ``'nodeWidth'``, and ``'nodeGap'``.
        """
        kw = {
            'chainNum': chainNum,
            'sampleNum': sampleNum,
            'treePlotPars': to_named_vector(treePlotPars),
        }
        self._call_rmethod('plotTree', treeNum, **drop_none(kw))


class _BartBase(RObjectBase):
    """
    Common base of the `dbarts` BART fitters.

    Holds the output attributes and the post-processing and
    `extract`/`fitted` accessors shared by `bart`, `bart2`, and `rbart_vi`.
    The fitting interface and the `fit`/`_wrap_fit`/`predict` members, which
    differ by fit type, live on the concrete subclasses.
    """

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

    def _postprocess(self) -> None:
        """
        Normalize the fit's R components into Python values.

        R fills inapplicable list components with NULL (e.g. ``yhat.test``
        without test data); expose them as None like the dropped ones, unwrap
        the scalar attributes, and wrap the kept sampler.
        """
        for name, value in list(vars(self).items()):
            if value is robjects.NULL:
                setattr(self, name, None)
        if self.n_chains is not None:
            self.n_chains = cast(ndarray, self.n_chains).item()
        if self.sigest is not None:
            self.sigest = cast(ndarray, self.sigest).item()
        self._wrap_fit()

    def _wrap_fit(self) -> None:
        """Wrap the kept R sampler(s) in the `dbarts` interface, in place (per fit type)."""
        raise NotImplementedError  # pragma: no cover - abstract; subclasses override

    def extract(
        self,
        *,
        type: Literal['ev', 'ppd', 'bart', 'trees'] | None = None,  # noqa: A002 mirrors the R argument name
        sample: Literal['train', 'test'] | None = None,
        combineChains: bool | None = None,
    ) -> Float64[ndarray, 'ndpost n'] | Float64[ndarray, 'nchain ndpost n'] | DataFrame:
        """
        Return the kept draws for the training (default) or test points.

        Like `predict`, the draws are on the expected-value scale by default.
        With ``type='trees'`` (requires ``keeptrees=True``) the tree
        structures are returned as a data frame instead. Arguments left to
        ``None`` are omitted from the R call.

        Parameters
        ----------
        type
            Quantity returned: ``'ev'``, ``'ppd'``, ``'bart'`` (see `predict`),
            or ``'trees'`` for the tree structures.
        sample
            Which points to extract: ``'train'`` or ``'test'``.
        combineChains
            Whether the chains are stacked into the draws axis rather than
            kept on a leading `nchain` axis.

        Returns
        -------
        The draws at the requested points, or the tree-structure data frame with ``type='trees'``.
        """
        kw = {'type': type, 'sample': sample, 'combineChains': combineChains}
        return self._call_rmethod('extract', **drop_none(kw))

    def fitted(
        self,
        *,
        type: Literal['ev', 'ppd', 'bart'] | None = None,  # noqa: A002 mirrors the R argument name
        sample: Literal['train', 'test'] | None = None,
    ) -> Float64[ndarray, ' n']:
        """
        Return the posterior mean for the training (default) or test points.

        Parameters
        ----------
        type
            Quantity averaged: ``'ev'``, ``'ppd'``, or ``'bart'`` (see
            `predict`).
        sample
            Which points to use: ``'train'`` or ``'test'``.

        Returns
        -------
        The posterior mean at the requested points.
        """
        kw = {'type': type, 'sample': sample}
        return self._call_rmethod('fitted', **drop_none(kw))


class bart(_BartBase):
    """
    Fit BART to continuous or binary outcomes (matrix interface).

    Python interface to R's ``dbarts::bart``. The named numeric vector forms
    of `splitprobs` and `proposalprobs` are given as dictionaries in Python; a
    named `splitprobs` requires `x_train` to have column names (e.g. a data
    frame). Arguments left to ``None`` are omitted from the R call, so R
    computes its own defaults, described below.

    In the attribute shapes below, `ndpost` counts the kept draws
    (``ndpost / keepevery``). Multiple chains are stacked into the `ndpost`
    axis when combined (``combinechains=True``, the `bart` default), and add a
    leading `nchain` axis otherwise.

    Parameters
    ----------
    x_train
        Training predictors; rows are observations. A data frame's factor
        columns are expanded into indicator columns.
    y_train
        Training response: continuous, or binary coded as 0/1 (which switches
        to a probit fit).
    x_test
        Test predictors, with the same column structure as `x_train`.
    sigest
        Rough estimate of the error SD anchoring the sigma prior; default the
        least-squares estimate. Continuous only.
    sigdf
        Degrees of freedom of the (inverse-chi-squared) sigma prior.
        Continuous only.
    sigquant
        Quantile of the sigma prior placed at `sigest`; closer to 1 puts more
        prior weight below `sigest`. Continuous only.
    k
        Number of prior SDs between ``f`` and the data extremes (+/-0.5 of the
        rescaled y for continuous, +/-3 on the latent scale for binary);
        bigger is more conservative. Can also be a ``chi`` hyperprior.
    power
        Exponent of the tree depth prior.
    base
        Scale of the tree depth prior.
    splitprobs
        Prior split probabilities of the variables; a dict mapping a subset of
        the column names to values plus a ``'.default'`` entry, or one value
        each. Uniform by default.
    binaryOffset
        Latent-scale offset for binary outcomes; ``P(Y = 1 | x) = Phi(f(x) +
        binaryOffset)``.
    weights
        Per-observation weights; the model becomes ``y | x ~ N(f(x),
        sigma^2 / w)``.
    ntree
        Number of trees in the sum.
    ndpost
        Number of posterior draws; ``ndpost / keepevery`` are returned.
    nskip
        Number of burn-in iterations.
    printevery
        Interval, in draws, of the progress messages.
    keepevery
        Thinning: keep one draw out of `keepevery`.
    keeptrainfits
        Whether to return the training-point function draws.
    usequants
        Whether the decision rules use empirical quantiles of each predictor
        rather than a uniform grid over its range.
    numcut
        Maximum number of decision rules per predictor (a scalar recycled, or
        one each).
    printcutoffs
        Number of a variable's decision rules printed before the run.
    verbose
        Whether to print progress to the R console.
    nchain
        Number of independent chains.
    nthread
        Number of threads to use.
    combinechains
        Whether the chains are stacked into the draws axis rather than kept on
        a leading `nchain` axis.
    keeptrees
        Whether the trees are kept, which `predict`, `extract`, and tree
        extraction require; memory-intensive.
    keepcall
        Whether the originating R call is stored in `call`.
    sampleronly
        Whether to build and return the underlying `dbarts` sampler without
        running it (changing the return type, so unsupported here).
    seed
        Seed of the chains' RNG; ``None`` (R's ``NA``) seeds from the clock
        when multi-threaded. Single-threaded, seed R with ``set.seed``.
    proposalprobs
        Tree-proposal probabilities, as a dict with keys ``'birth_death'``,
        ``'change'``, ``'swap'``, and ``'birth'``.
    keepsampler
        Whether to keep the underlying sampler even without `keeptrees`;
        default `keeptrees`.
    """

    _rfuncname = 'dbarts::bart'

    fit: dbarts | None = None
    """The sampler as a `dbarts` object, kept only with ``keeptrees`` or ``keepsampler``."""

    def __init__(
        self,
        x_train: Float64[ndarray, 'n p'] | DataFrame,
        y_train: Float64[ndarray, ' n'],
        x_test: Float64[ndarray, 'm p'] | DataFrame | None = None,
        *,
        sigest: float | None = None,
        sigdf: float = 3.0,
        sigquant: float = 0.9,
        k: float = 2.0,
        power: float = 2.0,
        base: float = 0.95,
        splitprobs: dict[str, float] | Float64[ndarray, ' p'] | None = None,
        binaryOffset: float = 0.0,
        weights: Float64[ndarray, ' n'] | None = None,
        ntree: int = 200,
        ndpost: int = 1000,
        nskip: int = 100,
        printevery: int = 100,
        keepevery: int = 1,
        keeptrainfits: bool = True,
        usequants: bool = False,
        numcut: int | Integer[ndarray, ' p'] = 100,
        printcutoffs: int = 0,
        verbose: bool = True,
        nchain: int = 1,
        nthread: int = 1,
        combinechains: bool = True,
        keeptrees: bool = False,
        keepcall: bool = True,
        sampleronly: bool = False,
        seed: int | None = None,
        proposalprobs: dict[str, float] | Float64[ndarray, ' 4'] | None = None,
        keepsampler: bool | None = None,
    ) -> None:
        kw = {
            'x.train': x_train,
            'y.train': y_train,
            'x.test': x_test,
            'sigest': sigest,
            'sigdf': sigdf,
            'sigquant': sigquant,
            'k': k,
            'power': power,
            'base': base,
            'splitprobs': to_named_vector(splitprobs),
            'binaryOffset': binaryOffset,
            'weights': weights,
            'ntree': ntree,
            'ndpost': ndpost,
            'nskip': nskip,
            'printevery': printevery,
            'keepevery': keepevery,
            'keeptrainfits': keeptrainfits,
            'usequants': usequants,
            'numcut': numcut,
            'printcutoffs': printcutoffs,
            'verbose': verbose,
            'nchain': nchain,
            'nthread': nthread,
            'combinechains': combinechains,
            'keeptrees': keeptrees,
            'keepcall': keepcall,
            'sampleronly': sampleronly,
            'seed': seed,
            'proposalprobs': to_named_vector(proposalprobs),
            'keepsampler': keepsampler,
        }
        RObjectBase.__init__(self, **drop_none(kw))
        self._postprocess()

    def _wrap_fit(self) -> None:
        """Wrap the kept R sampler in the `dbarts` interface."""
        if self.fit is not None:
            self.fit = dbarts._wrap(self.fit)  # noqa: SLF001, base-class access

    def predict(
        self,
        newdata: Float64[ndarray, 'm p'] | DataFrame,
        *,
        offset: Float64[ndarray, ' m'] | float | None = None,
        weights: Float64[ndarray, ' m'] | None = None,
        type: Literal['ev', 'ppd', 'bart'] | None = None,  # noqa: A002 mirrors the R argument name
        combineChains: bool | None = None,
        n_threads: int | None = None,
    ) -> Float64[ndarray, 'ndpost m'] | Float64[ndarray, 'nchain ndpost m']:
        """
        Compute predictions at new points; requires a ``keeptrees=True`` fit.

        Arguments left to ``None`` are omitted from the R call, so R computes
        its own defaults, described below.

        Parameters
        ----------
        newdata
            New predictors, with the same column structure as `x_train`.
        offset
            Offset added to the predictions.
        weights
            Per-observation weights of the predictive distribution.
        type
            Quantity returned: ``'ev'`` (expected value, i.e. probabilities
            for binary fits), ``'ppd'`` (posterior predictive draws), or
            ``'bart'`` (the latent sum-of-trees).
        combineChains
            Whether the chains are stacked into the draws axis rather than
            kept on a leading `nchain` axis.
        n_threads
            Number of threads to use.

        Returns
        -------
        The predictions at `newdata`, on the expected-value scale unless ``type='bart'``.
        """
        kw = {
            'offset': offset,
            'weights': weights,
            'type': type,
            'combineChains': combineChains,
            'n.threads': n_threads,
        }
        return self._call_rmethod('predict', newdata, **drop_none(kw))


class bart2(bart):
    """
    Fit BART to continuous or binary outcomes (formula interface).

    Python interface to R's ``dbarts::bart2``. A string `formula` argument is
    converted to an R formula; the backwards-compatible matrix form
    (`formula`/`data` as the `x_train`/`y_train` pair) passes through
    unchanged. The named numeric vector forms of `split_probs` and
    `proposal_probs` are given as dictionaries in Python. Arguments left to
    ``None`` are omitted from the R call, so R computes its own defaults,
    described below.

    Unlike `bart`, by default this runs multiple chains without combining
    them, which adds a leading `nchain` axis to the draws attributes (see the
    `bart` attributes).

    Parameters
    ----------
    formula
        A model formula (as a string), or, in matrix mode, the `x_train`
        predictor matrix.
    data
        The data frame the `formula` refers to, or, in matrix mode, the
        `y_train` response vector.
    test
        Test predictors, with the same columns as the training data.
    subset
        Subset of observations to keep.
    weights
        Per-observation weights; the model becomes ``y | x ~ N(f(x),
        sigma^2 / w)``.
    offset
        Latent-scale offset; ``P(Y = 1 | x) = Phi(f(x) + offset)`` for binary
        outcomes.
    offset_test
        The `offset` for the test data; defaults to `offset` when applicable.
    sigest
        Rough estimate of the error SD anchoring the sigma prior; default the
        least-squares estimate. Continuous only.
    sigdf
        Degrees of freedom of the (inverse-chi-squared) sigma prior.
        Continuous only.
    sigquant
        Quantile of the sigma prior placed at `sigest`. Continuous only.
    k
        Number of prior SDs between ``f`` and the data extremes; ``None`` uses
        2 for continuous responses and a ``chi`` hyperprior for binary ones.
    power
        Exponent of the tree depth prior.
    base
        Scale of the tree depth prior.
    split_probs
        Prior split probabilities of the variables; a dict mapping a subset of
        the column names to values plus a ``'.default'`` entry, or one value
        each. Uniform by default.
    n_trees
        Number of trees in the sum.
    n_samples
        Number of posterior samples kept per chain.
    n_burn
        Number of burn-in iterations.
    n_chains
        Number of independent chains.
    n_threads
        Number of threads to use; default ``min(cores, n_chains)``.
    combineChains
        Whether the chains are stacked into the draws axis rather than kept on
        a leading `nchain` axis.
    n_cuts
        Maximum number of decision rules per predictor.
    useQuantiles
        Whether the decision rules use empirical quantiles of each predictor
        rather than a uniform grid over its range.
    n_thin
        Thinning: keep one sample out of `n_thin`.
    keepTrainingFits
        Whether to return the training-point function draws.
    printEvery
        Interval, in samples, of the progress messages.
    printCutoffs
        Number of a variable's decision rules printed before the run.
    verbose
        Whether to print progress to the R console.
    keepTrees
        Whether the trees are kept, which `predict`, `extract`, and tree
        extraction require; memory-intensive.
    keepCall
        Whether the originating R call is stored in `call`.
    samplerOnly
        Whether to build and return the underlying `dbarts` sampler without
        running it (changing the return type, so unsupported here).
    seed
        Seed of the chains' RNG; ``None`` (R's ``NA``) seeds from the clock
        when multi-threaded.
    proposal_probs
        Tree-proposal probabilities, as a dict with keys ``'birth_death'``,
        ``'change'``, ``'swap'``, and ``'birth'``.
    keepSampler
        Whether to keep the underlying sampler even without `keepTrees`;
        default `keepTrees`.
    **control
        Extra keyword arguments forwarded verbatim (R's ``...``) to
        `dbartsControl`, by their R names, e.g. ``rngSeed`` or ``updateState``.
    """

    _rfuncname = 'dbarts::bart2'

    def __init__(
        self,
        formula: str | Float64[ndarray, 'n p'] | DataFrame,
        data: Float64[ndarray, ' n'] | DataFrame | None = None,
        *,
        test: Float64[ndarray, 'm p'] | DataFrame | None = None,
        subset: Integer[ndarray, ' k'] | None = None,
        weights: Float64[ndarray, ' n'] | None = None,
        offset: Float64[ndarray, ' n'] | float | None = None,
        offset_test: Float64[ndarray, ' m'] | float | None = None,
        sigest: float | None = None,
        sigdf: float = 3.0,
        sigquant: float = 0.9,
        k: float | None = None,
        power: float = 2.0,
        base: float = 0.95,
        split_probs: dict[str, float] | Float64[ndarray, ' p'] | None = None,
        n_trees: int = 75,
        n_samples: int = 500,
        n_burn: int = 500,
        n_chains: int = 4,
        n_threads: int | None = None,
        combineChains: bool = False,
        n_cuts: int | Integer[ndarray, ' p'] = 100,
        useQuantiles: bool = False,
        n_thin: int = 1,
        keepTrainingFits: bool = True,
        printEvery: int = 100,
        printCutoffs: int = 0,
        verbose: bool = True,
        keepTrees: bool = False,
        keepCall: bool = True,
        samplerOnly: bool = False,
        seed: int | None = None,
        proposal_probs: dict[str, float] | Float64[ndarray, ' 4'] | None = None,
        keepSampler: bool | None = None,
        **control: object,
    ) -> None:
        kw = {
            'formula': to_formula(formula),
            'data': data,
            'test': test,
            'subset': subset,
            'weights': weights,
            'offset': offset,
            'offset.test': offset_test,
            'sigest': sigest,
            'sigdf': sigdf,
            'sigquant': sigquant,
            'k': k,
            'power': power,
            'base': base,
            'split.probs': to_named_vector(split_probs),
            'n.trees': n_trees,
            'n.samples': n_samples,
            'n.burn': n_burn,
            'n.chains': n_chains,
            'n.threads': n_threads,
            'combineChains': combineChains,
            'n.cuts': n_cuts,
            'useQuantiles': useQuantiles,
            'n.thin': n_thin,
            'keepTrainingFits': keepTrainingFits,
            'printEvery': printEvery,
            'printCutoffs': printCutoffs,
            'verbose': verbose,
            'keepTrees': keepTrees,
            'keepCall': keepCall,
            'samplerOnly': samplerOnly,
            'seed': seed,
            'proposal.probs': to_named_vector(proposal_probs),
            'keepSampler': keepSampler,
            **control,
        }
        RObjectBase.__init__(self, **drop_none(kw))
        self._postprocess()


class rbart_vi(_BartBase):
    """
    Fit BART with additive group random intercepts.

    Python interface to R's ``dbarts::rbart_vi``, which adds an i.i.d. random
    intercept per `group_by` level to a `bart2` fit. A string `formula`
    argument is converted to an R formula. In addition to the `bart`
    components, the fit exposes the random-intercept outputs below. Arguments
    left to ``None`` are omitted from the R call, so R computes its own
    defaults.

    Parameters
    ----------
    formula
        A model formula (as a string), or, in matrix mode, the `x_train`
        predictor matrix.
    data
        The data frame the `formula` refers to, or, in matrix mode, the
        `y_train` response vector.
    group_by
        Grouping factor (one level per random intercept), as a vector or a
        reference to a column of `data`.
    test
        Test predictors, with the same columns as the training data.
    subset
        Subset of observations to keep.
    weights
        Per-observation weights.
    offset
        Latent-scale offset for binary outcomes.
    offset_test
        The `offset` for the test data; defaults to `offset` when applicable.
    group_by_test
        Grouping factor for the test data.
    prior
        Prior over the random-effects SD, as an R function or built-in
        reference (``cauchy`` or ``gamma``).
    sigest
        Rough estimate of the error SD, as in `bart2`. Continuous only.
    sigdf
        Degrees of freedom of the sigma prior, as in `bart2`. Continuous only.
    sigquant
        Quantile of the sigma prior placed at `sigest`, as in `bart2`.
    k
        Number of prior SDs between ``f`` and the data extremes, as in
        `bart2` (but defaulting to 2).
    power
        Exponent of the tree depth prior, as in `bart2`.
    base
        Scale of the tree depth prior, as in `bart2`.
    n_trees
        Number of trees in the sum, as in `bart2`.
    n_samples
        Number of posterior samples kept per chain, as in `bart2`.
    n_burn
        Number of burn-in iterations, as in `bart2`.
    n_chains
        Number of independent chains, as in `bart2`.
    n_threads
        Number of threads to use, as in `bart2`.
    combineChains
        Whether the chains are stacked into the draws axis, as in `bart2`.
    n_cuts
        Maximum number of decision rules per predictor, as in `bart2`.
    useQuantiles
        Whether the decision rules use empirical quantiles, as in `bart2`.
    n_thin
        Thinning: keep one sample out of `n_thin` (defaulting to 5).
    keepTrainingFits
        Whether to return the training-point function draws, as in `bart2`.
    printEvery
        Interval, in samples, of the progress messages, as in `bart2`.
    printCutoffs
        Number of a variable's decision rules printed before the run, as in
        `bart2`.
    verbose
        Whether to print progress to the R console, as in `bart2`.
    keepTrees
        Whether the trees are kept (defaulting to True), as in `bart2`.
    keepCall
        Whether the originating R call is stored in `call`, as in `bart2`.
    seed
        Seed of the chains' RNG, as in `bart2`.
    keepSampler
        Whether to keep the underlying sampler, as in `bart2`.
    keepTestFits
        Whether the test fits are returned (useful to disable with `callback`).
    callback
        An R function of ``trainFits``, ``testFits``, ``ranef``, ``sigma``, and
        ``tau`` called after each kept iteration, its results collected in
        `callback`.
    **control
        Extra keyword arguments forwarded verbatim (R's ``...``) to
        `dbartsControl`, by their R names, e.g. ``rngSeed`` or ``updateState``.

    Notes
    -----
    `split_probs`, `proposal_probs`, and `samplerOnly` (in `bart2`) are not
    part of ``rbart_vi``'s interface and are unavailable here.
    """

    _rfuncname = 'dbarts::rbart_vi'

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

    def __init__(
        self,
        formula: str | Float64[ndarray, 'n p'] | DataFrame,
        data: Float64[ndarray, ' n'] | DataFrame | None = None,
        *,
        group_by: Integer[ndarray, ' n'] | String[ndarray, ' n'],
        test: Float64[ndarray, 'm p'] | DataFrame | None = None,
        subset: Integer[ndarray, ' k'] | None = None,
        weights: Float64[ndarray, ' n'] | None = None,
        offset: Float64[ndarray, ' n'] | float | None = None,
        offset_test: Float64[ndarray, ' m'] | float | None = None,
        group_by_test: Integer[ndarray, ' m'] | String[ndarray, ' m'] | None = None,
        prior: object | None = None,
        sigest: float | None = None,
        sigdf: float = 3.0,
        sigquant: float = 0.9,
        k: float = 2.0,
        power: float = 2.0,
        base: float = 0.95,
        n_trees: int = 75,
        n_samples: int = 1500,
        n_burn: int = 1500,
        n_chains: int = 4,
        n_threads: int | None = None,
        combineChains: bool = False,
        n_cuts: int | Integer[ndarray, ' p'] = 100,
        useQuantiles: bool = False,
        n_thin: int = 5,
        keepTrainingFits: bool = True,
        printEvery: int = 100,
        printCutoffs: int = 0,
        verbose: bool = True,
        keepTrees: bool = True,
        keepCall: bool = True,
        seed: int | None = None,
        keepSampler: bool | None = None,
        keepTestFits: bool = True,
        callback: object | None = None,
        **control: object,
    ) -> None:
        kw = {
            'formula': to_formula(formula),
            'data': data,
            'group.by': group_by,
            'test': test,
            'subset': subset,
            'weights': weights,
            'offset': offset,
            'offset.test': offset_test,
            'group.by.test': group_by_test,
            'prior': prior,
            'sigest': sigest,
            'sigdf': sigdf,
            'sigquant': sigquant,
            'k': k,
            'power': power,
            'base': base,
            'n.trees': n_trees,
            'n.samples': n_samples,
            'n.burn': n_burn,
            'n.chains': n_chains,
            'n.threads': n_threads,
            'combineChains': combineChains,
            'n.cuts': n_cuts,
            'useQuantiles': useQuantiles,
            'n.thin': n_thin,
            'keepTrainingFits': keepTrainingFits,
            'printEvery': printEvery,
            'printCutoffs': printCutoffs,
            'verbose': verbose,
            'keepTrees': keepTrees,
            'keepCall': keepCall,
            'seed': seed,
            'keepSampler': keepSampler,
            'keepTestFits': keepTestFits,
            'callback': callback,
            **control,
        }
        RObjectBase.__init__(self, **drop_none(kw))
        self._postprocess()

    def _wrap_fit(self) -> None:
        """Wrap the R list of per-chain samplers in the `dbarts` interface."""
        if self.fit is not None:
            self.fit = tuple(map(dbarts._wrap, cast(NamedList, self.fit)))  # noqa: SLF001, base-class access

    def predict(
        self,
        newdata: Float64[ndarray, 'm p'] | DataFrame,
        *,
        group_by: Integer[ndarray, ' m'] | String[ndarray, ' m'] | None = None,
        offset: Float64[ndarray, ' m'] | float | None = None,
        type: Literal['ev', 'ppd', 'bart', 'ranef'] | None = None,  # noqa: A002 mirrors the R argument name
        combineChains: bool | None = None,
    ) -> Float64[ndarray, 'ndpost m'] | Float64[ndarray, 'nchain ndpost m']:
        """
        Compute predictions at new points; requires a ``keepTrees=True`` fit.

        Each new point needs a `group_by` level. Arguments left to ``None``
        are omitted from the R call, so R computes its own defaults.

        Parameters
        ----------
        newdata
            New predictors, with the same column structure as `x_train`.
        group_by
            Grouping factor of the new points; out-of-sample groups draw fresh
            random effects.
        offset
            Offset added to the predictions.
        type
            Quantity returned: ``'ev'`` (expected value), ``'ppd'`` (posterior
            predictive), ``'bart'`` (the latent sum-of-trees), or ``'ranef'``
            (the random effects).
        combineChains
            Whether the chains are stacked into the draws axis rather than
            kept on a leading `nchain` axis.

        Returns
        -------
        The predictions at `newdata`, on the expected-value scale unless ``type`` says otherwise.
        """
        kw = {
            'group.by': group_by,
            'offset': offset,
            'type': type,
            'combineChains': combineChains,
        }
        return self._call_rmethod('predict', newdata, **drop_none(kw))
