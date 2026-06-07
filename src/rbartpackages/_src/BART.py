# rbartpackages/src/rbartpackages/_src/BART.py
#
# Copyright (c) 2024-2026, The rbartpackages Contributors
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

"""Implementation of `rbartpackages.BART`."""

from functools import partial
from typing import Any, Literal, NamedTuple, TypedDict, cast

import numpy as np
from jaxtyping import AbstractDtype, Bool, Float64, Int32, Integer
from numpy import ndarray
from rpy2 import robjects
from rpy2.rlike.container import NamedList

from rbartpackages._src.base import (
    DataFrame,
    RObjectBase,
    drop_none,
    fork_safe_native_threads,
    rmethod,
)


class TreeDraws(TypedDict):
    """Type of the `mc_gbart.treedraws` attribute."""

    cutpoints: dict[int | str, Float64[ndarray, ' numcut[i]']]
    """Per-variable grid of candidate split points, keyed by column index or name."""

    trees: str
    """Posterior tree ensemble serialized in BART's text format (read by `mc_gbart.predict`)."""


class PredictBinary(TypedDict):
    """Type of `mc_gbart.predict`'s return value for binary ('pbart'/'lbart') fits."""

    yhat_test: Float64[ndarray, 'ndpost m']
    """Posterior latent-function draws at the test points."""

    prob_test: Float64[ndarray, 'ndpost m']
    """Success-probability draws (inverse probit/logit transform of `yhat_test`)."""

    prob_test_mean: Float64[ndarray, ' m']
    """Posterior mean of `prob_test`."""

    binaryOffset: float
    """Data centering value on the latent scale."""


class String(AbstractDtype):
    """Represent a `numpy.str_` data dtype."""

    dtypes = r'<U\d+'


class ProcTime(NamedTuple):
    """Python representation of the output of R's ``proc.time``."""

    user_self: float
    """CPU seconds charged to the R process in user mode."""

    sys_self: float
    """CPU seconds charged to the R process in system (kernel) mode."""

    elapsed: float
    """Wall-clock seconds elapsed."""

    user_child: float
    """User-mode CPU seconds of forked child processes (``mc.gbart`` workers)."""

    sys_child: float
    """System-mode CPU seconds of forked child processes."""


class mc_gbart(RObjectBase):
    """
    Fit BART to continuous or binary outcomes with multiple MCMC chains.

    Python interface to R's ``BART::mc.gbart``, which runs `mc_cores` MCMC
    chains in forked R processes and pools their draws. Arguments left to
    ``None`` are omitted from the R call, so R computes its own defaults,
    described below; 'continuous' refers to ``type='wbart'`` fits and
    'binary' to ``type='pbart'/'lbart'`` fits.

    Parameters
    ----------
    x_train
        Covariates for training; rows are observations. A dataframe's factor
        columns are expanded into indicator columns; missing values are
        imputed by hot decking.
    y_train
        Dependent variable for training: continuous, or binary coded as 0/1
        (requires setting `type`).
    x_test
        Covariates for test data, with the same structure as `x_train`.
    type
        The type of fit: 'wbart' (continuous), 'pbart' (probit binary) or
        'lbart' (logit binary).
    sparse
        Whether to replace the uniform splitting-variable choice with the
        sparse Dirichlet (DART) variable-selection prior.
    theta
        `theta` parameter of the DART prior; 0 means random.
    omega
        `omega` parameter of the DART prior; 0 means random.
    a
        Shape parameter of the ``Beta(a, b)`` prior on the DART sparsity,
        between 0.5 and 1; lower values induce more sparsity.
    b
        Shape parameter of the ``Beta(a, b)`` prior on the DART sparsity.
    augment
        Whether to perform data augmentation in the sparse variable
        selection.
    rho
        Concentration of the DART prior; default the number of variables
        (after factor expansion), set it lower for more sparsity.
    xinfo
        Cutpoints to use, one row per variable; by default they are computed
        from `x_train` (see `numcut` and `usequants`).
    usequants
        Whether the computed cutpoints are quantiles of the data rather than
        uniformly spaced over its range.
    rm_const
        Whether to drop constant covariates.
    sigest
        Rough estimate of the error SD that anchors the sigma prior; default
        the residual SD of a linear fit (the SD of `y_train` if ``p >= n``).
        Continuous only.
    sigdf
        Degrees of freedom of the (scaled inverse chi-squared) sigma prior.
        Continuous only.
    sigquant
        Quantile of the sigma prior placed at `sigest`; closer to 1 puts more
        prior weight below `sigest`. Continuous only.
    k
        Number of prior SDs between f's mean and the data extremes (+/-0.5 of
        the rescaled y for continuous, +/-3 on the latent scale for binary);
        bigger is more conservative.
    power
        Exponent of the tree depth prior
        ``P(split node at depth d) = base / (1 + d)**power``.
    base
        Scale of the tree depth prior (see `power`).
    lambda_
        Scale of the sigma prior (R's ``lambda``); the default derives it
        from `sigest` and `sigquant`. 0 would fix the error SD at `sigest`,
        but R then crashes summarizing the sigma draws it no longer makes.
        Continuous only.
    tau_num
        Numerator of the leaf-value prior SD ``tau_num / (k * sqrt(ntree))``;
        default ``(max(y_train) - min(y_train)) / 2`` for continuous, 3 for
        'pbart' and 6 for 'lbart'.
    offset
        Centering subtracted from `y_train`; default its mean, mapped through
        the inverse link for binary outcomes.
    w
        Per-observation weights multiplying the error SD. Continuous only.
    ntree
        Number of trees in the sum; default 200 for continuous and 50 for
        binary outcomes.
    numcut
        Number of candidate cutpoints, for all variables or per column (the
        per-column form requires `transposed`: R's preprocessing mishandles
        it otherwise).
    ndpost
        Number of posterior draws to keep, after burn-in and thinning
        (rounded up to a whole number of draws per chain).
    nskip
        Number of burn-in MCMC iterations discarded, per chain.
    keepevery
        Thinning: keep one draw out of `keepevery`; default 1 for continuous
        and 10 for binary outcomes.
    printevery
        Interval, in MCMC iterations, of the progress messages.
    transposed
        Whether `x_train` and `x_test` are already preprocessed (see
        `bartModelMatrix`) and transposed to ``(p, n)``, as `mc_gbart` does
        when handing the data to its workers.
    hostname
        Whether to record the hostname the fit runs on (per chain for
        ``mc.gbart``), to track the nodes of a cluster.
    mc_cores
        Number of MCMC chains, run in forked R processes, capped at the
        detected core count. `gbart` runs a single chain in-process and
        ignores it.
    nice
        Unix niceness of the chain processes, from 0 (highest priority) to 19
        (lowest). `gbart` ignores it.
    seed
        Seed of the chains' L'Ecuyer-CMRG RNG streams; ``None`` seeds from
        the clock and process ID. `gbart` ignores it: seed R directly with
        ``set.seed``.

    Raises
    ------
    ValueError
        If the `rm_const` output of R cannot be parsed.

    Notes
    -----
    The R argument ``ntype`` (an internal device to compute the
    type-dependent defaults) is not exposed.
    """

    _rfuncname = 'BART::mc.gbart'

    LPML: float
    """Log pseudo-marginal likelihood; unstable for BART.

    Always computed, even without burn-in. Miscomputed by R for binary
    ``mc.gbart`` fits with ``mc_cores > 1`` (the chains' probabilities are not
    combined before the computation).
    """

    hostname: Bool[ndarray, ' mc_cores'] | String[ndarray, ' mc_cores']
    """Per-chain hostname if fitted with ``hostname=True``, else per-chain ``False``."""

    ndpost: int
    """Number of posterior draws kept, after burn-in and thinning."""

    offset: float
    """Data centering value for the response (link scale for binary)."""

    prob_test: None | Float64[ndarray, 'ndpost/mc_cores m'] = None
    """Test-point success-probability draws (binary outcomes only).

    ``mc.gbart`` with ``mc_cores > 1`` forgets to combine the chains, leaving
    only the first chain's draws.
    """

    prob_test_mean: None | Float64[ndarray, ' m'] = None
    """Posterior mean of `prob_test`."""

    prob_train: None | Float64[ndarray, 'ndpost/mc_cores n'] = None
    """Training-point success-probability draws (binary outcomes only).

    ``mc.gbart`` with ``mc_cores > 1`` forgets to combine the chains, leaving
    only the first chain's draws.
    """

    prob_train_mean: None | Float64[ndarray, ' n'] = None
    """Posterior mean of `prob_train`."""

    proc_time: ProcTime
    """Timing of the fit, from R's ``proc.time``."""

    rm_const: Int32[ndarray, '<=p']
    """0-based indices of the `x_train` columns kept (constant columns dropped).

    ``mc.gbart`` with ``mc_cores=1`` relabels the kept columns to ``0 .. kept-1``,
    losing which original columns were dropped.
    """

    sigma: (
        Float64[ndarray, ' nskip+ndpost*keepevery']
        | Float64[ndarray, 'nskip+ndpost*keepevery/mc_cores mc_cores']
        | None
    ) = None
    """Error-SD draws, continuous outcomes only (per chain for ``mc.gbart``).

    One draw per MCMC iteration: burn-in and the thinned-away iterations are
    included.
    """

    sigma_mean: float | None = None
    """Mean of the first `ndpost` post-burn-in `sigma` draws (continuous only)."""

    treedraws: TreeDraws
    """Sampled trees: per-variable cutpoint grid and the serialized ensemble."""

    varcount: Int32[ndarray, 'ndpost p']
    """Per-draw count of splits on each variable, summed over trees."""

    varcount_mean: Float64[ndarray, ' p']
    """Posterior mean of `varcount` per variable."""

    varprob: Float64[ndarray, 'ndpost p']
    """Per-draw probability assigned to each variable for splitting."""

    varprob_mean: Float64[ndarray, ' p']
    """Posterior mean of `varprob` per variable."""

    yhat_test: Float64[ndarray, 'ndpost m']
    """Test-point posterior function draws (latent scale for binary).

    Always present: R's ``cgbart`` allocates it unconditionally, so without
    test data it is an empty array rather than ``None`` (with the rows of the
    first chain only for ``mc.gbart``, which combines the chains just when
    there is test data).
    """

    yhat_test_mean: Float64[ndarray, ' m'] | None = None
    """Posterior mean of `yhat_test` (continuous with test data only)."""

    yhat_train: Float64[ndarray, 'ndpost n']
    """Training-point posterior function draws (latent scale for binary)."""

    yhat_train_mean: Float64[ndarray, ' n'] | None = None
    """Posterior mean of `yhat_train` (continuous only)."""

    def __init__(
        self,
        x_train: Float64[ndarray, 'n p'] | DataFrame,
        y_train: Float64[ndarray, ' n'],
        x_test: Float64[ndarray, 'm p'] | DataFrame | None = None,
        *,
        type: Literal['wbart', 'pbart', 'lbart'] = 'wbart',  # noqa: A002 because it mirrors the R argument name
        sparse: bool = False,
        theta: float = 0.0,
        omega: float = 1.0,
        a: float = 0.5,
        b: float = 1.0,
        augment: bool = False,
        rho: float | None = None,
        xinfo: Float64[ndarray, 'p numcut'] | None = None,
        usequants: bool = False,
        rm_const: bool = True,
        sigest: float | None = None,
        sigdf: float = 3.0,
        sigquant: float = 0.9,
        k: float = 2.0,
        power: float = 2.0,
        base: float = 0.95,
        lambda_: float | None = None,
        tau_num: float | None = None,
        offset: float | None = None,
        w: Float64[ndarray, ' n'] | None = None,
        ntree: int | None = None,
        numcut: int | Integer[ndarray, ' p'] = 100,
        ndpost: int = 1000,
        nskip: int = 100,
        keepevery: int | None = None,
        printevery: int = 100,
        transposed: bool = False,
        hostname: bool = False,
        mc_cores: int = 2,
        nice: int = 19,
        seed: int | None = 99,
    ) -> None:
        kw = {
            'x.train': x_train,
            'y.train': y_train,
            'x.test': x_test,
            'type': type,
            'sparse': sparse,
            'theta': theta,
            'omega': omega,
            'a': a,
            'b': b,
            'augment': augment,
            'rho': rho,
            'xinfo': xinfo,
            'usequants': usequants,
            'rm.const': rm_const,
            'sigest': sigest,
            'sigdf': sigdf,
            'sigquant': sigquant,
            'k': k,
            'power': power,
            'base': base,
            'lambda': lambda_,
            'tau.num': tau_num,
            'offset': offset,
            'w': w,
            'ntree': ntree,
            'numcut': numcut,
            'ndpost': ndpost,
            'nskip': nskip,
            'keepevery': keepevery,
            'printevery': printevery,
            'transposed': transposed,
            'hostname': hostname,
            'mc.cores': mc_cores,
            'nice': nice,
            # NULL is meaningful to R here (seed from the clock and process
            # ID), so it cannot stand for the R default like for the other
            # arguments
            'seed': robjects.NULL if seed is None else seed,
        }
        # mc.gbart forks via parallel::mcparallel; cap native thread pools at one
        # thread across the fork to avoid a libgomp deadlock in the children.
        with fork_safe_native_threads():
            super().__init__(**drop_none(kw))

        # fix up attributes
        self.LPML = self.LPML.item()
        self.ndpost = self.ndpost.astype(int).item()
        self.offset = self.offset.item()
        self.proc_time = ProcTime(*map(float, self.proc_time))

        if np.all(self.rm_const < 0):
            # R reports the dropped constant columns as negative indices into
            # the original design matrix, while varcount has the kept ones
            _, kept = self.varcount.shape
            p = kept + self.rm_const.size
            rm_const = np.ones(p, bool)
            rm_const[-self.rm_const - 1] = False
            self.rm_const = np.arange(p, dtype=np.int32)[rm_const]
        elif np.all(self.rm_const > 0):
            self.rm_const -= 1
        else:  # pragma: no cover - R gives all-positive or all-negative indices
            msg = 'failed to parse rm.const because indices change sign'
            raise ValueError(msg)

        if self.sigma_mean is not None:
            self.sigma_mean = self.sigma_mean.item()

        r_treedraws = cast(NamedList, self.treedraws)
        cutpoints: NamedList = r_treedraws.getbyname('cutpoints')
        self.treedraws = {
            'cutpoints': {
                i if it.name is None else it.name.item(): it.value
                for i, it in enumerate(cutpoints.items())
            },
            'trees': r_treedraws.getbyname('trees').item(),
        }

    @partial(rmethod, rname='predict')
    def _predict(self, newdata: Float64[ndarray, 'm p'], **kwargs: Any) -> object:
        """Call R's `predict`; returns a matrix (continuous) or a list (binary)."""
        ...

    def predict(
        self,
        newdata: Float64[ndarray, 'm p'] | DataFrame,
        *,
        mc_cores: int | None = None,
        openmp: bool | None = None,
        dodraws: bool | None = None,
        nice: int | None = None,
    ) -> Float64[ndarray, 'ndpost m'] | Float64[ndarray, ' m'] | PredictBinary:
        """
        Compute predictions at new covariate points.

        Python interface to R's ``predict`` method for the fit, dispatched
        on the fit `type`. For continuous ('wbart') fits the result is the
        matrix of posterior latent-function draws (their mean with
        ``dodraws=False``); for binary ('pbart'/'lbart') fits R returns a
        list, exposed here as a `PredictBinary` dict. Arguments left to
        ``None`` are omitted from the R call, so R computes its own
        defaults, described below.

        Parameters
        ----------
        newdata
            Covariates to predict at; rows are observations, with one
            column per kept `x_train` column (see `rm_const`). A
            dataframe's factor columns are expanded into indicator
            columns.
        mc_cores
            Number of OpenMP threads or forked R processes (see `openmp`)
            computing the predictions; default 1.
        openmp
            Whether `mc_cores` counts OpenMP threads rather than forked R
            processes; default whether BART was compiled with OpenMP.
        dodraws
            Whether to return the posterior draws (the default) rather
            than only their mean. 'wbart' fits only (the binary methods
            accept it but then crash summarizing the mean-only result).
        nice
            Unix niceness of the forked processes, from 0 (highest
            priority) to 19 (lowest, the default); ignored unless forking.

        Returns
        -------
        pred : Float64[ndarray, 'ndpost m'] | Float64[ndarray, ' m'] | PredictBinary
            The function draws at `newdata` for continuous fits (their mean with ``dodraws=False``), or a `PredictBinary` dict for binary fits.

        Notes
        -----
        For ``mc.gbart`` fits with ``mc_cores > 1`` that dropped constant
        columns, R miscounts the kept columns and fails to update the
        header of the serialized ensemble, so only the first chain's draws
        are returned.

        The R arguments ``mu`` (the method already fills it with the fit's
        offset, and a second value would be a duplicate-argument error) and
        ``transposed`` (a pre-transposed `newdata` cannot pass the method's
        own column-count check) are not exposed.
        """
        kw = {'mc.cores': mc_cores, 'openmp': openmp, 'dodraws': dodraws, 'nice': nice}
        out = self._predict(newdata, **drop_none(kw))
        if not hasattr(out, 'items'):
            return out  # continuous: a draws matrix or its column means

        # binary: convert R's list (a NamedList) to a dict of arrays
        out = cast(NamedList, out)
        result = {str(it.name).replace('.', '_'): it.value for it in out.items()}
        result['binaryOffset'] = result['binaryOffset'].item()
        return result


class bartModelMatrix(RObjectBase):
    """
    Convert covariates to a matrix and compute the BART cutpoints.

    Python interface to R's ``BART::bartModelMatrix``. With the default
    ``numcut=0`` the constructor returns the bare design matrix instead of a
    class instance; otherwise the instance carries the matrix together with
    the cutpoints metadata.

    Parameters
    ----------
    X
        The covariates to convert; rows are observations. A dataframe's
        factor columns are expanded into indicator columns.
    numcut
        Maximum number of cutpoints per variable; 0 means return the bare
        matrix without computing cutpoints.
    usequants
        Whether the cutpoints are quantiles of the data rather than uniformly
        spaced over its range.
    type
        The quantile algorithm used with `usequants` (see R's ``quantile``).
    rm_const
        Whether to remove the constant columns from `X` (they are flagged in
        `rm_const` either way).
    cont
        Whether to treat all variables as continuous, spacing `numcut`
        cutpoints over the range even when fewer unique values would do.
    xinfo
        Cutpoints to use, one row per variable; overrides the computed ones.
    """

    _rfuncname = 'BART::bartModelMatrix'

    X: Float64[ndarray, 'N p']
    """Design matrix: vectors and data frames coerced to numeric, factors expanded to indicators."""

    numcut: Int32[ndarray, ' p']
    """Number of cutpoints chosen per column."""

    rm_const: Int32[ndarray, '<=p']
    """0-based indices of the non-constant columns of the expanded design.

    The indices refer to the columns of `X` before removal: ``rm.const=True``
    removes the constant columns from `X`, `numcut` and `xinfo`, while the
    default only detects them.
    """

    xinfo: Float64[ndarray, 'p numcut']
    """Per-column cutpoint grid, NaN-padded to the maximum cut count."""

    grp: Int32[ndarray, ' p'] | Float64[ndarray, ' 1'] | None
    """1-based input-column index each output column comes from (factors expand
    to one indicator column per level); ``None`` for matrix input."""

    def __new__(
        cls,
        X: Float64[ndarray, 'N p'] | DataFrame,
        numcut: int = 0,
        *,
        usequants: bool = False,
        type: int = 7,  # noqa: A002 because it mirrors the R argument name
        rm_const: bool = False,
        cont: bool = False,
        xinfo: Float64[ndarray, 'p numcut'] | None = None,
    ) -> Float64[ndarray, 'N p'] | RObjectBase:
        """Match R: return the bare matrix for ``numcut=0``, else a populated instance."""
        # __init__ cannot change the return type, so everything happens here;
        # returning a non-instance (the matrix) skips __init__.
        kw = {
            'X': X,
            'numcut': numcut,
            'usequants': usequants,
            'type': type,
            'rm.const': rm_const,
            'cont': cont,
            'xinfo': xinfo,
        }
        self = super().__new__(cls)
        self._robject = self._invoke_rfunc((), drop_none(kw))
        if not self._has_named_components(self._robject):
            return self._r2py(self._robject)
        self._set_attrs_from_robject()

        # grp is R NULL for matrix input; expose it as None.
        if self.grp is robjects.NULL:
            self.grp = None

        if np.all(self.rm_const < 0):
            # R flags detected-constant columns as negative indices into the
            # pre-removal design matrix; whether they were also removed from X
            # depends on the rm_const argument
            _, n_cols = self.X.shape
            p = n_cols + self.rm_const.size if rm_const else n_cols
            keep = np.ones(p, bool)
            keep[-self.rm_const - 1] = False
            self.rm_const = np.arange(p, dtype=np.int32)[keep]
        elif np.all(self.rm_const > 0):
            self.rm_const -= 1
        else:  # pragma: no cover - R gives all-positive or all-negative indices
            msg = 'failed to parse rm.const because indices change sign'
            raise ValueError(msg)

        return self

    def __init__(
        self,
        X: Float64[ndarray, 'N p'] | DataFrame,
        numcut: int = 0,
        *,
        usequants: bool = False,
        type: int = 7,  # noqa: A002 because it mirrors the R argument name
        rm_const: bool = False,
        cont: bool = False,
        xinfo: Float64[ndarray, 'p numcut'] | None = None,
    ) -> None:
        # Everything happens in __new__ because the numcut=0 case changes the
        # return type; this stub (whose signature mirrors __new__'s for
        # introspection) only stops the inherited RObjectBase.__init__ from
        # invoking R a second time.
        ...


class gbart(mc_gbart):
    """
    Fit BART to continuous or binary outcomes with a single MCMC chain.

    Python interface to R's ``BART::gbart``. Same parameters as `mc_gbart`,
    but the fit runs in the current R process, ignoring `mc_cores`, `nice`
    and `seed`; seed the fit through R's ``set.seed`` instead.
    """

    _rfuncname = 'BART::gbart'

    hostname: Bool[ndarray, ' 1'] | String[ndarray, ' 1']
    """Hostname the fit ran on if fitted with ``hostname=True``, else ``False``."""

    prob_test: None | Float64[ndarray, 'ndpost m'] = None
    """Test-point success-probability draws (binary outcomes only)."""

    prob_train: None | Float64[ndarray, 'ndpost n'] = None
    """Training-point success-probability draws (binary outcomes only)."""

    sigma: Float64[ndarray, ' nskip+ndpost*keepevery'] | None = None
    """Error-SD draws for every MCMC iteration, burn-in included (continuous only)."""
