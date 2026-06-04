# rbartpackages/src/rbartpackages/BART.py
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

"""Wrapper for the R package BART."""

# ruff: noqa: ANN002, ANN003

from functools import partial
from typing import NamedTuple, TypedDict, cast

import numpy as np
from jaxtyping import AbstractDtype, Bool, Float64, Int32
from numpy import ndarray
from rpy2.rlike.container import NamedList

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
    _rfuncname = 'BART::mc.gbart'

    LPML: float
    """Log pseudo-marginal likelihood; unstable for BART.

    Always computed, even without burn-in. Miscomputed by R for binary
    `mc.gbart` fits with ``mc_cores > 1`` (the chains' probabilities are not
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

    `mc.gbart` with ``mc_cores > 1`` forgets to combine the chains, leaving
    only the first chain's draws.
    """

    prob_test_mean: None | Float64[ndarray, ' m'] = None
    """Posterior mean of `prob_test`."""

    prob_train: None | Float64[ndarray, 'ndpost/mc_cores n'] = None
    """Training-point success-probability draws (binary outcomes only).

    `mc.gbart` with ``mc_cores > 1`` forgets to combine the chains, leaving
    only the first chain's draws.
    """

    prob_train_mean: None | Float64[ndarray, ' n'] = None
    """Posterior mean of `prob_train`."""

    proc_time: ProcTime
    """Timing of the fit, from R's `proc.time`."""

    rm_const: Int32[ndarray, '<=p']
    """0-based indices of the `x_train` columns kept (constant columns dropped).

    `mc.gbart` with ``mc_cores=1`` relabels the kept columns to ``0 .. kept-1``,
    losing which original columns were dropped.
    """

    sigma: (
        Float64[ndarray, ' nskip+ndpost*keepevery']
        | Float64[ndarray, 'nskip+ndpost*keepevery/mc_cores mc_cores']
        | None
    ) = None
    """Error-SD draws, continuous outcomes only (per chain for `mc.gbart`).

    One draw per MCMC iteration: burn-in and the thinned-away iterations are
    included.
    """

    sigma_mean: float | None = None
    """Mean of the post-burn-in `sigma` draws (continuous only)."""

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

    Always present: R's `cgbart` allocates it unconditionally, so without test
    data it is an empty array rather than ``None`` (with the rows of the first
    chain only for `mc.gbart`, which combines the chains just when there is
    test data).
    """

    yhat_test_mean: Float64[ndarray, ' m'] | None = None
    """Posterior mean of `yhat_test` (continuous with test data only)."""

    yhat_train: Float64[ndarray, 'ndpost n']
    """Training-point posterior function draws (latent scale for binary)."""

    yhat_train_mean: Float64[ndarray, ' n'] | None = None
    """Posterior mean of `yhat_train` (continuous only)."""

    def __init__(self, *args, **kw) -> None:
        # mc.gbart forks via parallel::mcparallel; cap native thread pools at one
        # thread across the fork to avoid a libgomp deadlock in the children.
        with fork_safe_native_threads():
            super().__init__(*args, **kw)

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

        For `mc.gbart` fits with ``mc_cores > 1`` that dropped constant
        columns, R miscounts the kept columns and fails to update the header
        of the serialized ensemble, so only the first chain's draws are
        returned.
        """
        out = self._predict(newdata, *args, **kwargs)
        if not hasattr(out, 'items'):
            return out  # continuous: already a matrix

        # binary: convert R's list (a NamedList) to a dict of arrays
        out = cast(NamedList, out)
        result = {str(it.name).replace('.', '_'): it.value for it in out.items()}
        result['binaryOffset'] = result['binaryOffset'].item()
        return result


class bartModelMatrix(RObjectBase):  # noqa: D101 because the R doc is added automatically
    _rfuncname = 'BART::bartModelMatrix'


class gbart(mc_gbart):  # noqa: D101 because the R doc is added automatically
    _rfuncname = 'BART::gbart'

    hostname: Bool[ndarray, ' 1'] | String[ndarray, ' 1']
    """Hostname the fit ran on if fitted with ``hostname=True``, else ``False``."""

    prob_test: None | Float64[ndarray, 'ndpost m'] = None
    """Test-point success-probability draws (binary outcomes only)."""

    prob_train: None | Float64[ndarray, 'ndpost n'] = None
    """Training-point success-probability draws (binary outcomes only)."""

    sigma: Float64[ndarray, ' nskip+ndpost*keepevery'] | None = None
    """Error-SD draws for every MCMC iteration, burn-in included (continuous only)."""
