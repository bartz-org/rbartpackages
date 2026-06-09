# rbartpackages/src/rbartpackages/_src/missBART.py
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

"""Implementation of `rbartpackages.missBART`."""

from typing import Any

import numpy as np
from jaxtyping import Bool, Float64, Shaped
from numpy import ndarray
from rpy2.rlike.container import NamedList

# WORKAROUND(python<3.11): import TypedDict, Unpack from typing
from typing_extensions import TypedDict, Unpack

from rbartpackages._src.base import RObjectBase, drop_none


def _values(nl: NamedList) -> list[Any]:
    return [it.value for it in nl.items()]


class TreePriorParams(TypedDict, total=False):
    """Tree-prior parameters, the keys of R's ``tree_list``.

    The prior probability of splitting a node at depth ``d`` is
    ``prior_alpha * (1 + d) ** -prior_beta``. Entries left out fall back to
    ``tree_list``'s own defaults, noted below.
    """

    prior_alpha: float
    """Base of the node-splitting probability, in ``(0, 1)``. Default 0.95."""

    prior_beta: float
    """Power of the node-splitting probability, nonnegative. Default 2."""

    min_node: int
    """Minimum number of observations per leaf, raised to ``p + 1`` internally. Default 1."""

    max_attempt: int
    """Number of attempts to propose a valid split before giving up. Default 1."""


class Hypers(TypedDict, total=False):
    """Prior hyperparameters, the keys of R's ``hypers_list``.

    Entries left out fall back to ``hypers_list``'s own defaults, noted below;
    `kappa`, `alpha` and `V` also accept ``None`` to request their data-based
    default explicitly.
    """

    mu0: float
    """Prior mean of the tree-leaf parameters. Default 0."""

    kappa: float | None
    """Prior precision of the regression-tree leaf parameters. Default ``4 * qnorm(0.975) ** 2 * n_reg_trees``."""

    alpha: float | None
    """Wishart degrees of freedom of the residual precision, multivariate `y` only. Default `df`."""

    V: Float64[ndarray, 'p p'] | None
    """Wishart scale matrix of the residual precision, multivariate `y` only. Default data-based."""

    df: float
    """Degrees of freedom of the error-variance prior. Default 10."""

    q: float
    """Prior quantile of the error variance at the data-based estimate. Default 0.75."""

    tau_b: float
    """Unused by `missBART2`. Default 100."""


class Hyperparams(TreePriorParams, Hypers, total=False):
    """Union of `TreePriorParams` and `Hypers`, forwarded through ``**hyperparams``."""


class missBART2(RObjectBase):
    """
    Fit BART to outcomes with missing entries, imputing them.

    Python interface to R's ``missBART::missBART2``. It jointly fits a
    regression BART to the (possibly multivariate) outcome `y` and a probit
    BART to its missingness pattern, imputing the missing entries of `y` along
    the MCMC. Missing entries of the predictors `x` are handled by augmenting
    `x` with binary missingness-indicator columns. Arguments left to ``None``
    are omitted from the R call, so R computes its own defaults, described
    below.

    Parameters
    ----------
    x
        Predictor matrix; rows are observations. Missing entries, marked with
        ``NaN``, augment it with one binary missingness-indicator column per
        predictor (they are not imputed, unlike those of `y`).
    y
        Outcome matrix (one column per response) or vector; ``NaN`` marks the
        entries to impute.
    x_predict
        Out-of-sample predictors at which to draw the posterior predictive; if
        omitted, no out-of-sample predictions are made (see Notes).
    n_reg_trees
        Number of trees of the regression BART modeling `y`.
    n_class_trees
        Number of trees of the probit BART modeling the missingness of `y`.
    burn
        Number of burn-in MCMC iterations discarded.
    iters
        Number of post-burn-in iterations retained after thinning.
    thin
        Thinning interval; the chain runs ``burn + thin * iters`` iterations.
    predict
        Whether to draw the posterior predictive at `x_predict`; see Notes for
        the interaction with `x_predict`.
    MH_sd
        Standard deviation of the Metropolis-Hastings proposal updating the
        missing entries of `y`; default ``0.5 / p``.
    tree_prior_params
        Tree-prior parameters as a `TreePriorParams` dict; it must be complete,
        as passing it directly skips ``tree_list``'s own defaults. Setting the
        individual parameters as keyword arguments instead (see
        ``**hyperparams``) avoids this.
    hypers
        Prior hyperparameters as a `Hypers` dict; it must be complete, as
        passing it directly skips ``hypers_list``'s own defaults. Setting the
        individual hyperparameters as keyword arguments instead (see
        ``**hyperparams``) avoids this.
    scale
        Whether to scale `y` to ``[-0.5, 0.5]`` before fitting.
    include_x
        Whether the missingness probit model uses `x` as predictors.
    include_y
        Whether the missingness probit model uses `y` as predictors.
    show_progress
        Whether to display a progress bar in the R console.
    progress_every
        Update the progress bar every this many iterations.
    pdp_range
        Range over which the partial dependence plot is evaluated (with
        `make_pdp`).
    make_pdp
        Whether to compute partial dependence output; univariate `y` only.
    mice_impute
        Whether the missing entries of `y` are initialized with ``mice::mice``;
        otherwise they start at zero.
    **hyperparams
        Extra keyword arguments, of the `Hyperparams` keys, forwarded verbatim
        (R's ``...``), which populate the unset entries of `tree_prior_params`
        (through ``tree_list``) and `hypers` (through ``hypers_list``); the
        intended way to set individual tree-prior parameters and
        hyperparameters.

    Raises
    ------
    ValueError
        If ``predict=True`` is passed without `x_predict`.

    Notes
    -----
    If `x_predict` is not specified, the wrapper passes ``predict=False`` and a
    placeholder `x_predict`, because the R code crashes on its own default
    ``x_predict = c()`` (``as.matrix(NULL)`` is an error). Explicitly passing
    ``predict=True`` without `x_predict` raises `ValueError`.

    The R arguments ``true_trees_data``, ``true_trees_missing``,
    ``true_change_points`` and ``true_change_points_miss`` are accepted but
    never used by the upstream implementation, so they are not exposed (they
    remain reachable through ``**hyperparams`` if ever needed).
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
    ``scale=False``, but only its diagonal ``(iters, p)`` with
    ``scale=True``."""

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
    covariates `x_predict`, on the original scale. ``None`` if
    ``predict=False`` or `x_predict` was not supplied. With ``scale=False``
    the values are garbled because the upstream code applies the un-scaling
    anyway."""

    pdp_out: Any | None = None
    """Partial dependence plot output. ``None`` unless ``make_pdp=True`` and
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

    def __init__(
        self,
        x: Float64[ndarray, 'n q'],
        y: Float64[ndarray, 'n p'],
        x_predict: Float64[ndarray, 'm q'] | None = None,
        *,
        n_reg_trees: int = 100,
        n_class_trees: int = 100,
        burn: int = 1000,
        iters: int = 1000,
        thin: int = 2,
        predict: bool | None = None,
        MH_sd: float | None = None,
        tree_prior_params: TreePriorParams | None = None,
        hypers: Hypers | None = None,
        scale: bool = True,
        include_x: bool = True,
        include_y: bool = True,
        show_progress: bool = True,
        progress_every: int = 10,
        pdp_range: Float64[ndarray, ' 2'] | tuple[float, float] = (-0.5, 0.5),
        make_pdp: bool = False,
        mice_impute: bool = True,
        **hyperparams: Unpack[Hyperparams],
    ) -> None:
        if x_predict is None:
            if predict:
                msg = 'predict=True requires x_predict'
                raise ValueError(msg)
            # reuse x as a placeholder that predict=False leaves untouched: R's
            # own default x_predict = c() makes as.matrix(NULL) error
            x_predict = x
            predict = False

        kw = {
            'x': x,
            'y': y,
            'x_predict': x_predict,
            'n_reg_trees': n_reg_trees,
            'n_class_trees': n_class_trees,
            'burn': burn,
            'iters': iters,
            'thin': thin,
            'predict': predict,
            'MH_sd': MH_sd,
            'tree_prior_params': tree_prior_params,
            'hypers': hypers,
            'scale': scale,
            'include_x': include_x,
            'include_y': include_y,
            'show_progress': show_progress,
            'progress_every': progress_every,
            'pdp_range': np.asarray(pdp_range, float),
            'make_pdp': make_pdp,
            'mice_impute': mice_impute,
            **hyperparams,
        }
        super().__init__(**drop_none(kw))

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
