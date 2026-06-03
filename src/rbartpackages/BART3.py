# rbartpackages/src/rbartpackages/BART3.py
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

"""Wrapper for the R package BART3."""

# ruff: noqa: ANN002, ANN003

from functools import partial
from typing import NamedTuple, TypedDict

import numpy as np
from jaxtyping import AbstractDtype, Float64, Int32
from numpy import ndarray
from rpy2 import robjects

from rbartpackages._base import RObjectBase, fork_safe_native_threads, rmethod


class TreeDraws(TypedDict):
    """Type of the `treedraws` attribute of `mc_gbart`."""

    cutpoints: dict[int | str, Float64[ndarray, ' numcut[i]']]
    """Per-variable grid of candidate split points, keyed by column index or name."""

    trees: str
    """Posterior tree ensemble serialized in BART's text format (read by `predict`)."""


class PredictBinary(TypedDict):
    """Type of `predict`'s return value for binary (`pbart`/`lbart`) fits."""

    yhat_test: Float64[ndarray, 'ndpost m']
    """Posterior latent-function draws at the test points."""

    prob_test: Float64[ndarray, 'ndpost m']
    """Success-probability draws (probit/logit transform of `yhat_test`)."""

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
    """Python representation of the output of R's `proc.time`."""

    user_self: float
    """CPU seconds charged to the R process in user mode."""

    sys_self: float
    """CPU seconds charged to the R process in system (kernel) mode."""

    elapsed: float
    """Wall-clock seconds elapsed."""

    user_child: float
    """User-mode CPU seconds of forked child processes (`mc.gbart` workers)."""

    sys_child: float
    """System-mode CPU seconds of forked child processes."""


class mc_gbart(RObjectBase):  # noqa: D101 because the R doc is added automatically
    _rfuncname = 'BART3::mc.gbart'

    LPML: float | None = None
    """Log pseudo-marginal likelihood; ``None`` without burn-in. Unstable for BART."""

    accept: (
        Float64[ndarray, ' nskip+ndpost*keepevery']
        | Float64[ndarray, 'nskip+ndpost*keepevery/mc_cores mc_cores']
    )
    """Per-iteration Metropolis-Hastings acceptance rate (per chain for `mc.gbart`).

    Recorded for every MCMC iteration, including the thinned-away ones (unlike
    `sigma`, which keeps only burn-in plus retained draws).
    """

    chains: int
    """Number of MCMC chains, i.e. the `mc_cores` actually used."""

    grp: Float64[ndarray, ' p']
    """Group index of each column for the sparse (DART) variable-selection prior."""

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
    """Timing of the fit, from R's `proc.time`."""

    rho: float
    """Concentration of the sparse (DART) prior; defaults to ``sum(1/grp)``."""

    rm_const: Int32[ndarray, '<=p']
    """0-based indices of the `x_train` columns kept (constant columns dropped)."""

    sigest: float | None = None
    """Rough residual SD used to set the sigma prior (continuous only).

    ``None`` for binary outcomes; ``nan`` when the `mc.gbart` ``mc_cores > 1``
    bug overwrites it with a logical missing value.
    """

    sigma: (
        Float64[ndarray, ' nskip+ndpost']
        | Float64[ndarray, 'nskip+ndpost/mc_cores mc_cores']
        | None
    ) = None
    """Error-SD draws including burn-in, continuous only (per chain for `mc.gbart`)."""

    sigma_: Float64[ndarray, ' ndpost'] | None = None
    """Kept `sigma` draws with burn-in dropped; ``None`` without burn-in."""

    sigma_mean: float | None = None
    """Mean of `sigma_`; falls back to `sigest` when no draws are kept."""

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

    x_test: Float64[ndarray, ' m p'] | None = None
    """Test design matrix as used (after imputation and factor expansion)."""

    x_train: Float64[ndarray, ' n p']
    """Training design matrix as used (original scale, not binned)."""

    yhat_test: Float64[ndarray, 'ndpost m'] | None = None
    """Test-point posterior function draws (latent scale for binary)."""

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

    def __init__(self, *args, **kw) -> None:
        # mc.gbart forks via parallel::mcparallel; cap native thread pools at one
        # thread across the fork to avoid a libgomp deadlock in the children.
        with fork_safe_native_threads():
            super().__init__(*args, **kw)

        # fix up attributes
        self.chains = self.chains.item()
        self.ndpost = self.ndpost.astype(int).item()
        self.offset = self.offset.item()
        self.proc_time = ProcTime(*map(float, self.proc_time))
        self.rho = self.rho.item()

        if np.all(self.rm_const < 0):
            _, p = self.varcount.shape
            rm_const = np.ones(p, bool)
            rm_const[-self.rm_const - 1] = False
            self.rm_const = np.arange(p, dtype=np.int32)[rm_const]
        elif np.all(self.rm_const > 0):
            self.rm_const -= 1
        else:
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

        if hasattr(self.treedraws, 'getbyname'):
            # it's a NamedList
            self.treedraws = {
                'cutpoints': {
                    i if it.name is None else it.name.item(): it.value
                    for i, it in enumerate(
                        self.treedraws.getbyname('cutpoints').items()
                    )
                },
                'trees': self.treedraws.getbyname('trees').item(),
            }
        else:
            # it's an OrdDict
            self.treedraws = {
                'cutpoints': {
                    i if k is None else k.item(): v
                    for i, (k, v) in enumerate(self.treedraws['cutpoints'].items())
                },
                'trees': self.treedraws['trees'].item(),
            }

    @partial(rmethod, rname='predict')
    def _predict(self, newdata: Float64[ndarray, 'm p'], *args, **kwargs) -> object:
        """Call R's `predict`; returns a matrix (continuous) or a list (binary)."""
        ...

    def predict(
        self, newdata: Float64[ndarray, 'm p'], *args, **kwargs
    ) -> Float64[ndarray, 'ndpost m'] | PredictBinary:
        """Compute predictions.

        For continuous (`wbart`) fits this is the matrix of posterior
        latent-function draws. For binary (`pbart`/`lbart`) fits R returns a
        list, exposed here as a `PredictBinary` dict.
        """
        out = self._predict(newdata, *args, **kwargs)
        if not hasattr(out, 'items'):
            return out  # continuous: already a matrix

        # binary: convert R's list (NamedList or OrdDict) to a dict of arrays
        if hasattr(out, 'getbyname'):
            items = [(it.name, it.value) for it in out.items()]
        else:
            items = list(out.items())
        result = {str(k).replace('.', '_'): v for k, v in items}
        result['binaryOffset'] = result['binaryOffset'].item()
        return result


class bartModelMatrix(RObjectBase):  # noqa: D101 because the R doc is added automatically
    _rfuncname = 'BART3::bartModelMatrix'

    X: Float64[ndarray, 'N p']
    """Design matrix: vectors and data frames coerced to numeric, factors expanded to indicators."""

    numcut: Int32[ndarray, ' p']
    """Number of cutpoints chosen per column."""

    rm_const: Int32[ndarray, ' p']
    """1-based indices of the columns kept after removing constant ones."""

    xinfo: Float64[ndarray, 'p numcut']
    """Per-column cutpoint grid, NaN-padded to the maximum cut count."""

    grp: Float64[ndarray, ' p'] | None
    """Group size of each expanded factor's indicator columns, or None if no factors."""

    def __new__(cls, *args, **kw) -> Float64[ndarray, 'N p'] | RObjectBase:
        """Match R: return the bare matrix for ``numcut=0``, else a populated instance."""
        # __init__ cannot change the return type, so the matrix-or-list choice
        # is made here; returning a non-instance (the matrix) skips __init__.
        self = super().__new__(cls)
        self._robject = self._invoke_rfunc(args, kw)
        if self._has_named_components(self._robject):
            return self
        return self._r2py(self._robject)

    def __init__(self, *args, **kw) -> None:  # noqa: ARG002
        # Only reached for the named-list case (numcut > 0); __new__ already
        # invoked R and stored `_robject`, so just expose its components rather
        # than calling super().__init__ (which would invoke R a second time).
        self._set_attrs_from_robject()

        # grp is R NULL unless the input had factor columns; expose it as None.
        if self.grp is robjects.NULL:
            self.grp = None


class gbart(mc_gbart):  # noqa: D101 because the R doc is added automatically
    _rfuncname = 'BART3::gbart'

    accept: Float64[ndarray, ' nskip+ndpost*keepevery']
    """Per-iteration Metropolis-Hastings acceptance rate (every MCMC iteration)."""

    sigma: Float64[ndarray, ' nskip+ndpost'] | None = None
    """Error-SD draws including burn-in (continuous only)."""
