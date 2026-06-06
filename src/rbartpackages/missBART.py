# rbartpackages/src/rbartpackages/missBART.py
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

"""Wrapper for the R package missBART."""

# ruff: noqa: ANN002, ANN003

from typing import Any

import numpy as np
from jaxtyping import Bool, Float64, Shaped
from numpy import ndarray
from rpy2.rlike.container import NamedList

from rbartpackages._base import RObjectBase


def _values(nl: NamedList) -> list[Any]:
    return [it.value for it in nl.items()]


class missBART2(RObjectBase):
    """
    Python interface to missBART::missBART2.

    If `x_predict` is not specified, the wrapper passes ``predict=False`` and a
    placeholder `x_predict`, because the R code crashes on its own default
    ``x_predict = c()`` (`as.matrix(NULL)` is an error). Explicitly passing
    ``predict=True`` without `x_predict` raises `ValueError`.
    """

    _rfuncname = 'missBART::missBART2'

    MH_sd: float
    """Standard deviation of the Metropolis-Hastings proposal used to update
    the missing entries of `y`. If not supplied at construction, the R code
    sets it to ``0.5 / p``."""

    burn: int
    """Number of burn-in MCMC iterations (discarded)."""

    iters: int
    """Number of post-burn-in MCMC iterations retained after thinning."""

    thin: int
    """Thinning interval applied to the post-burn-in chain. The total number
    of MCMC iterations is ``burn + thin * iters``."""

    max_y: Float64[ndarray, ' p']
    """Per-output-column maxima of `y` computed before scaling. Used to
    invert the [-0.5, 0.5] scaling when reporting predictions."""

    min_y: Float64[ndarray, ' p']
    """Per-output-column minima of `y` computed before scaling."""

    x: Float64[ndarray, 'n q'] | Float64[ndarray, 'n 2*q']
    """Covariate matrix actually used by the sampler. If the input `x`
    contained missing values, this is the input augmented column-wise with
    binary missingness indicators (one per original column). The missingness
    indicator columns come all together after the value columns."""

    y_miss_accept: Bool[ndarray, 'total_iters n_missing']
    """Acceptance flags of the Metropolis-Hastings proposals for the
    missing `y` entries. One row per MCMC iteration (including burn-in),
    one column per missing entry, listed in column-major order of `y`."""

    y_post: Float64[ndarray, 'iters n p']
    """Posterior draws of the BART regression mean for the training rows,
    on the original (un-scaled) scale of `y`."""

    z_post: Float64[ndarray, 'iters n p']
    """Posterior draws of the latent probit variables of the missingness
    model."""

    omega_post: (
        Float64[ndarray, 'iters 1 1']
        | Float64[ndarray, 'iters p']
        | Float64[ndarray, 'iters p p']
    )
    """Posterior draws of the residual variance of the BART regression, on
    the original scale of `y`. Shape ``(iters, 1, 1)`` for univariate `y`;
    for multivariate `y` the full covariance matrix ``(iters, p, p)`` with
    `scale=False`, but only its diagonal ``(iters, p)`` with `scale=True`."""

    y_impute: Float64[ndarray, 'iters n_missing']
    """Posterior draws of the imputed values for the missing entries of
    `y`, on the original scale. Columns are ordered as in
    `y_miss_accept`."""

    var_imp: list[Float64[ndarray, '...']]
    """Per-retained-iteration variable importance scores derived from the
    classification (probit) BART trees that model missingness. The
    upstream code stores one entry per variable that was actually used as
    a split during that iteration, so the per-iteration vector length
    varies and the attribute is left as a list of arrays."""

    new_y_post: Float64[ndarray, 'iters n_predict p'] | None = None
    """Posterior predictive draws (incl. error term) at the out-of-sample
    covariates `x_predict`, on the original scale. ``None`` if `predict=False`
    or `x_predict` was not supplied. With `scale=False` the values are garbled
    because the upstream code applies the un-scaling anyway."""

    pdp_out: Any | None = None
    """Partial dependence plot output. ``None`` unless `make_pdp=True` and
    `y` is univariate."""

    y_pred: list
    """In-sample posterior predictive draws. Currently always empty in the
    upstream R implementation."""

    reg_trees: list[list[Shaped[ndarray, ' num_nodes']]]
    """Accepted regression-BART tree structures, indexed as
    ``reg_trees[i][j]`` for retained iteration ``i`` and tree ``j``. Each
    tree is a numpy structured array whose records carry the fields
    ``parent``, ``lower``, ``upper``, ``split_variable``, ``split_value``,
    ``depth``, ``direction``, ``NA_direction``."""

    class_trees: list[list[Shaped[ndarray, ' num_nodes']]]
    """Accepted probit-BART tree structures for the missingness model,
    same layout as `reg_trees`."""

    reg_mu: list[list[Float64[ndarray, 'n_leaves p']]]
    """Leaf-node parameters of the regression-BART trees, indexed as
    ``reg_mu[i][j]`` for iteration ``i`` and tree ``j``. The outer list has
    length ``burn + thin * iters`` (i.e. it includes burn-in iterations,
    unlike `reg_trees`); each leaf array has shape ``(n_leaves, p)``."""

    class_mu: list[list[Float64[ndarray, 'n_leaves p']]]
    """Leaf-node parameters of the probit-BART trees, same layout as
    `reg_mu`. The trailing dimension is ``p`` (one mean per response
    column, since the probit trees model the per-column missingness
    indicators of `y`)."""

    def __init__(self, *args, **kw) -> None:
        # x is the 1st parameter of R's missBART2, x_predict the 3rd; reuse x
        # as a placeholder that predict=False leaves untouched (see class doc)
        if len(args) < 3 and 'x_predict' not in kw:
            if kw.get('predict'):
                msg = 'predict=True requires x_predict'
                raise ValueError(msg)
            x = args[0] if args else kw.get('x')
            if x is not None:
                kw = dict(kw, x_predict=x, predict=False)

        super().__init__(*args, **kw)

        self.MH_sd = self.MH_sd.item()
        self.burn = int(self.burn.item())
        self.iters = int(self.iters.item())
        self.thin = int(self.thin.item())

        # NA-when-disabled fields come back as plain (NA-filled) ndarrays
        # instead of NamedLists; normalize to None.
        if isinstance(self.new_y_post, NamedList):
            self.new_y_post = np.stack(_values(self.new_y_post))
        else:
            self.new_y_post = None

        if isinstance(self.pdp_out, np.ndarray):
            self.pdp_out = None

        self.y_post = np.stack(_values(self.y_post))
        self.z_post = np.stack(_values(self.z_post))
        self.omega_post = np.stack(_values(self.omega_post))
        self.y_impute = np.stack(_values(self.y_impute))
        self.var_imp = _values(self.var_imp)
        self.y_pred = _values(self.y_pred)

        self.reg_trees = [_values(it) for it in _values(self.reg_trees)]
        self.class_trees = [_values(it) for it in _values(self.class_trees)]
        self.reg_mu = [_values(it) for it in _values(self.reg_mu)]
        self.class_mu = [_values(it) for it in _values(self.class_mu)]
