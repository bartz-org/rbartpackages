# rbartpackages/src/rbartpackages/_src/BART3.py
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

"""Implementation of `rbartpackages.BART3`."""

from functools import partial
from typing import Any, Literal, NamedTuple, TypedDict, cast

import numpy as np
from jaxtyping import AbstractDtype, Float64, Int32, Integer, Real
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

    prob_test_lower: Float64[ndarray, ' m']
    """Lower `probs` quantile of `prob_test` (default 2.5%)."""

    prob_test_upper: Float64[ndarray, ' m']
    """Upper `probs` quantile of `prob_test` (default 97.5%)."""

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

    Python interface to R's ``BART3::mc.gbart``, which runs `mc_cores` MCMC
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
        Concentration of the DART prior; the default 0 means ``sum(1 / grp)``
        (the number of variables when there are no factors), set it lower for
        more sparsity.
    grp
        Inverse weight of each variable in the DART prior: the number of
        indicator columns it expanded to; derived from `x_train` by default.
    varprob
        Initial splitting probability of each variable; uniform by default.
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
    impute_mult
        1-based `x_train` column indices forming one multinomial indicator
        set whose missing values need imputation (at least two columns).
    impute_prob
        Per-observation category probabilities for the multinomial
        imputation; default the observed category frequencies.
    impute_miss
        Per-observation 0/1 indicator of the rows to impute; derived from the
        missing values in `x_train` by default.
    lambda_
        Scale of the sigma prior (R's ``lambda``); 0 fixes the error SD at
        `sigest`, the default derives it from `sigest` and `sigquant`.
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
        Number of candidate cutpoints, for all variables or per column.
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
    probs
        Lower and upper quantiles of the ``*_lower``/``*_upper`` summaries.
    mc_cores
        Number of MCMC chains, run in forked R processes; default R's
        ``mc.cores`` option, or 2. `gbart` runs a single chain in-process and
        only records this value (default 1) in `chains`.
    nice
        Unix niceness of the chain processes, from 0 (highest priority) to 19
        (lowest). `gbart` ignores it.
    seed
        Seed of the chains' L'Ecuyer-CMRG RNG streams; ``None`` leaves R's
        RNG state alone. `gbart` ignores it: seed R directly with
        ``set.seed``.
    meta
        Whether to produce meta-analysis-like estimates of a sharded analysis
        rather than Modified LISA (forces ``shards=1``). `gbart` ignores it.
    verbose
        Set to 0 to compute silently.
    shards
        Number of shards of the Modified LISA method.
    weight
        Per-shard combination weights of the Modified LISA method (currently
        accepted but unused by BART3).

    Raises
    ------
    ValueError
        If the `rm_const` output of R cannot be parsed.

    Notes
    -----
    The R arguments ``ntype`` (an internal device to compute the
    type-dependent defaults) and ``TSVS`` (strips the output down to what the
    ``BART3::tsvs`` variable-selection routine needs, which the wrapper could
    not digest) are not exposed.
    """

    _rfuncname = 'BART3::mc.gbart'

    LPML: float | None = None
    """Log pseudo-marginal likelihood; ``None`` without burn-in. Unstable for BART."""

    accept: (
        Float64[ndarray, ' nskip+ndpost*keepevery']
        | Float64[ndarray, 'nskip+ndpost*keepevery/mc_cores mc_cores']
    )
    """Per-iteration Metropolis-Hastings acceptance rate (per chain for ``mc.gbart``).

    Recorded for every MCMC iteration, including the thinned-away ones (unlike
    `sigma`, which keeps only burn-in plus retained draws).
    """

    chains: int
    """Number of MCMC chains, i.e. the `mc_cores` actually used."""

    grp: Float64[ndarray, ' p']
    """Group index of each column for the sparse (DART) variable-selection prior."""

    impute_miss: Int32[ndarray, ' n'] | None = None
    """Missingness indicator of each training row (multinomial imputation only)."""

    ndpost: int
    """Number of posterior draws kept, after burn-in and thinning."""

    offset: float
    """Data centering value for the response (link scale for binary)."""

    prob_test: None | Float64[ndarray, 'ndpost m'] = None
    """Test-point success-probability draws (binary outcomes only)."""

    prob_test_lower: Float64[ndarray, ' m'] | None = None
    """Lower `probs` quantile of `prob_test` (default 2.5%)."""

    prob_test_mean: None | Float64[ndarray, ' m'] = None
    """Posterior mean of `prob_test`."""

    prob_test_upper: Float64[ndarray, ' m'] | None = None
    """Upper `probs` quantile of `prob_test` (default 97.5%)."""

    prob_train: None | Float64[ndarray, 'ndpost n'] = None
    """Training-point success-probability draws (binary outcomes only)."""

    prob_train_mean: None | Float64[ndarray, ' n'] = None
    """Posterior mean of `prob_train`."""

    proc_time: ProcTime
    """Timing of the fit, from R's ``proc.time``."""

    rho: float
    """Concentration of the sparse (DART) prior; defaults to ``sum(1/grp)``."""

    rm_const: Int32[ndarray, '<=p']
    """0-based indices of the `x_train` columns kept (constant columns dropped)."""

    sigest: float | None = None
    """Rough residual SD used to set the sigma prior (continuous only).

    ``None`` for binary outcomes; ``nan`` when the ``mc.gbart`` ``mc_cores > 1``
    bug overwrites it with a logical missing value.
    """

    sigma: (
        Float64[ndarray, ' nskip+ndpost']
        | Float64[ndarray, 'nskip+ndpost/mc_cores mc_cores']
        | None
    ) = None
    """Error-SD draws including burn-in, continuous only (per chain for ``mc.gbart``)."""

    sigma_: Float64[ndarray, ' ndpost'] | None = None
    """Kept `sigma` draws with burn-in dropped; ``None`` without burn-in."""

    sigma_mean: float | None = None
    """Mean of `sigma_`; falls back to `sigest` when no draws are kept."""

    treedraws: TreeDraws
    """Sampled trees, as a per-variable cutpoint grid and the serialized ensemble."""

    varcount: Int32[ndarray, 'ndpost p']
    """Per-draw count of splits on each variable, summed over trees."""

    varcount_mean: Float64[ndarray, ' p']
    """Posterior mean of `varcount` per variable."""

    varprob: Float64[ndarray, 'ndpost p']
    """Per-draw probability assigned to each variable for splitting."""

    varprob_mean: Float64[ndarray, ' p']
    """Posterior mean of `varprob` per variable."""

    x_test: Float64[ndarray, ' m <=p'] | None = None
    """Test design matrix as used (imputed, factors expanded, constant columns dropped)."""

    x_train: Float64[ndarray, ' n <=p']
    """Training design matrix as used (original scale, not binned; constant columns dropped)."""

    yhat_test: Float64[ndarray, 'ndpost m']
    """Test-point posterior function draws (latent scale for binary).

    Always present: R's ``cgbart`` allocates it unconditionally, so without
    test data it is an empty ``(ndpost, 0)`` array rather than ``None``
    (unlike the derived `yhat_test_mean`/`yhat_test_lower`/`yhat_test_upper`,
    which R only fills when test data is given).
    """

    yhat_test_lower: Float64[ndarray, ' m'] | None = None
    """Lower `probs` quantile of `yhat_test` (default 2.5%, continuous only)."""

    yhat_test_mean: Float64[ndarray, ' m'] | None = None
    """Posterior mean of `yhat_test`."""

    yhat_test_upper: Float64[ndarray, ' m'] | None = None
    """Upper `probs` quantile of `yhat_test` (default 97.5%, continuous only)."""

    yhat_train: Float64[ndarray, 'ndpost n']
    """Training-point posterior function draws (latent scale for binary)."""

    yhat_train_lower: Float64[ndarray, ' n'] | None = None
    """Lower `probs` quantile of `yhat_train` (default 2.5%, continuous only)."""

    yhat_train_mean: Float64[ndarray, ' n'] | None = None
    """Posterior mean of `yhat_train`."""

    yhat_train_upper: Float64[ndarray, ' n'] | None = None
    """Upper `probs` quantile of `yhat_train` (default 97.5%, continuous only)."""

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
        rho: float = 0.0,
        grp: Real[ndarray, ' p'] | None = None,
        varprob: Float64[ndarray, ' p'] | None = None,
        xinfo: Float64[ndarray, 'p numcut'] | None = None,
        usequants: bool = False,
        rm_const: bool = True,
        sigest: float | None = None,
        sigdf: float = 3.0,
        sigquant: float = 0.9,
        k: float = 2.0,
        power: float = 2.0,
        base: float = 0.95,
        impute_mult: Integer[ndarray, ' mult'] | None = None,
        impute_prob: Float64[ndarray, 'n mult'] | None = None,
        impute_miss: Integer[ndarray, ' n'] | None = None,
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
        probs: tuple[float, float] = (0.025, 0.975),
        mc_cores: int | None = None,
        nice: int = 19,
        seed: int | None = 99,
        meta: bool = False,
        verbose: int = 1,
        shards: int = 1,
        weight: Float64[ndarray, ' shards'] | None = None,
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
            'grp': grp,
            'varprob': varprob,
            'xinfo': xinfo,
            'usequants': usequants,
            'rm.const': rm_const,
            'sigest': sigest,
            'sigdf': sigdf,
            'sigquant': sigquant,
            'k': k,
            'power': power,
            'base': base,
            'impute.mult': impute_mult,
            'impute.prob': impute_prob,
            'impute.miss': impute_miss,
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
            'probs': np.asarray(probs),
            'mc.cores': mc_cores,
            'nice': nice,
            # NULL is meaningful to R here (skip the seeding), so it cannot
            # stand for the R default like for the other arguments
            'seed': robjects.NULL if seed is None else seed,
            'meta': meta,
            'verbose': verbose,
            'shards': shards,
            'weight': weight,
        }
        # mc.gbart forks via parallel::mcparallel; cap native thread pools at one
        # thread across the fork to avoid a libgomp deadlock in the children.
        with fork_safe_native_threads():
            super().__init__(**drop_none(kw))

        # fix up attributes
        self.chains = self.chains.item()
        self.ndpost = self.ndpost.astype(int).item()
        self.offset = self.offset.item()
        self.proc_time = ProcTime(*map(float, self.proc_time))
        self.rho = self.rho.item()

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

        if self.LPML is not None:
            self.LPML = self.LPML.item()
        if self.sigest is not None:
            if self.sigest.dtype == bool:
                # BART3 bug: mc.gbart with mc_cores > 1 overwrites sigest with
                # its logical-NA default instead of the estimate.
                self.sigest = float('nan')
            else:
                self.sigest = self.sigest.item()
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
        mult_impute: int | None = None,
        seed: int | None = None,
        mu: float | None = None,
        probs: tuple[float, float] | None = None,
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
        defaults, described below; R rejects the arguments marked for
        specific fit types when used with the others.

        Parameters
        ----------
        newdata
            Covariates to predict at; rows are observations, with one
            column per kept `x_train` column (see `rm_const`). A
            dataframe's factor columns are expanded into indicator
            columns.
        mc_cores
            Number of OpenMP threads or forked R processes (see `openmp`)
            computing the predictions; default R's ``mc.cores`` option,
            or 1.
        openmp
            Whether `mc_cores` counts OpenMP threads rather than forked R
            processes; default whether BART3 was compiled with OpenMP.
        mult_impute
            Number of hot-deck imputations averaged over when `newdata`
            has missing values; default 4. Not accepted by 'lbart' fits.
        seed
            Seed set in R before imputing missing values (default 99).
            'wbart' fits only ('pbart' accepts but ignores it).
        mu
            Value added to the function draws in place of the fit's
            `offset`. 'wbart' fits only.
        probs
            Lower and upper quantiles of the ``prob_test_lower``/``_upper``
            summaries; default ``(0.025, 0.975)``. Binary fits only.
        dodraws
            Whether to return the posterior draws (the default) rather
            than only their mean. 'wbart' fits only.
        nice
            Unix niceness of the forked processes, from 0 (highest
            priority) to 19 (lowest, the default); ignored unless forking.

        Returns
        -------
        pred : Float64[ndarray, 'ndpost m'] | Float64[ndarray, ' m'] | PredictBinary
            The function draws at `newdata` for continuous fits (their mean with ``dodraws=False``), or a `PredictBinary` dict for binary fits.

        Notes
        -----
        The R arguments ``cutpoints`` and ``trees`` (fallbacks for fits
        missing `treedraws`, which `gbart` fits always carry) and
        ``transposed`` (a pre-transposed `newdata` cannot pass the
        method's own column-count check) are not exposed.
        """
        kw = {
            'mc.cores': mc_cores,
            'openmp': openmp,
            'mult.impute': mult_impute,
            'seed': seed,
            'mu': mu,
            'probs': None if probs is None else np.asarray(probs),
            'dodraws': dodraws,
            'nice': nice,
        }
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

    Python interface to R's ``BART3::bartModelMatrix``. With the default
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

    _rfuncname = 'BART3::bartModelMatrix'

    X: Float64[ndarray, 'N p']
    """Design matrix, with vectors and data frames coerced to numeric and factors expanded to indicators."""

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

    grp: Float64[ndarray, ' p'] | None
    """Group size of each expanded factor's indicator columns, or None if no factors."""

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

        # grp is R NULL unless the input had factor columns; expose it as None.
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

    Python interface to R's ``BART3::gbart``. Same parameters as `mc_gbart`,
    but the fit runs in the current R process: `mc_cores` (defaulting to 1
    here) is only recorded in the `chains` attribute, and `nice`, `seed` and
    `meta` are ignored; seed the fit through R's ``set.seed`` instead.
    """

    _rfuncname = 'BART3::gbart'

    accept: Float64[ndarray, ' nskip+ndpost*keepevery']
    """Per-iteration Metropolis-Hastings acceptance rate (every MCMC iteration)."""

    sigma: Float64[ndarray, ' nskip+ndpost'] | None = None
    """Error-SD draws including burn-in (continuous only)."""
