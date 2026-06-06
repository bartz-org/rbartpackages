# rbartpackages/src/rbartpackages/bartMachine.py
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

"""Wrapper for the R package bartMachine.

bartMachine wants data frames, so this module requires the ``pandas`` extra.

Importing this module loads bartMachine's R namespace, which starts the JVM.
JVM options can only be set before that: the import defaults the heap size
limit to 5 GB and enables the Vector API module bartMachine uses on JDK 16+;
to customize, set R's ``java.parameters`` option before the first import.
"""

# ruff: noqa: ANN002, ANN003

from functools import partial
from typing import TypedDict, cast

from jaxtyping import AbstractDtype, Float64
from numpy import ndarray
from pandas import DataFrame
from rpy2 import robjects
from rpy2.rlike.container import NamedList
from rpy2.robjects.methods import RS4

from rbartpackages._base import RObjectBase, rfunction, rmethod

# The JVM reads its options only at startup; rJava starts it when the
# bartMachine namespace is first loaded, which the rfunction decorations below
# do at import time. So set the options now: default the heap size limit
# unless the user already set java.parameters, and always enable the
# (incubating) Vector API, which bartMachine uses on JDK 16+ (fitting raises
# NoClassDefFoundError without it).
robjects.r("""local({
    params <- getOption("java.parameters")
    if (is.null(params)) params <- "-Xmx5000m"
    options(java.parameters = union(params, "--add-modules=jdk.incubator.vector"))
})""")


class Posterior(TypedDict):
    """Type of `bart_machine_get_posterior`'s return value."""

    y_hat: Float64[ndarray, ' m']
    """Posterior mean of f(x) at each point (probability of the first level for classification)."""

    X: Float64[ndarray, 'm p']
    """The prediction points after preprocessing (factors expanded to indicators)."""

    y_hat_posterior_samples: Float64[ndarray, 'm num_iterations_after_burn_in']
    """Posterior draws of f(x), one column per kept MCMC iteration."""


class String(AbstractDtype):
    """Represent a `numpy.str_` data dtype."""

    dtypes = r'<U\d+'


class bartMachine(RObjectBase):
    """
    Python interface to bartMachine::bartMachine.

    The number of fitting threads is a package-global setting, see
    `set_bart_machine_num_cores`.
    """

    _rfuncname = 'bartMachine::bartMachine'

    L1_err_train: float | None = None
    """In-sample L1 error (regression with `run_in_sample` only)."""

    L2_err_train: float | None = None
    """In-sample L2 error (regression with `run_in_sample` only)."""

    PseudoRsq: float | None = None
    """In-sample ``1 - L2_err_train / L2 of the mean`` (regression with `run_in_sample` only)."""

    X: DataFrame
    """Training predictors as supplied (factors not expanded)."""

    alpha: float
    """Base of the nonterminal-node probability in the tree prior."""

    beta: float
    """Power of the nonterminal-node probability in the tree prior."""

    confusion_matrix: DataFrame | None = None
    """In-sample confusion matrix with error rates (classification with `run_in_sample` only)."""

    cov_prior_vec: Float64[ndarray, ' p'] | None = None
    """Relative split-proposal weight of each predictor.

    ``None`` unless given, except that expanded factors get a default
    down-weighting their indicator columns by the number of levels.
    """

    debug_log: bool
    """Whether the Java backend logged to a file."""

    flush_indices_to_save_RAM: bool
    """Whether the Java backend flushed internal indices to save memory."""

    impute_missingness_with_rf_impute: bool
    """Whether missing training entries got `randomForest::rfImpute` imputations added."""

    impute_missingness_with_x_j_bar_for_lm: bool
    """Whether the linear model behind `sig_sq_est` imputed missing entries with column averages."""

    interaction_constraints: tuple[Float64[ndarray, ' group[i]'], ...] | None = None
    """Groups of predictors allowed to interact, as 0-based column indices; ``None`` if not given."""

    java_bart_machine: RS4
    """RJava reference to the Java model; opaque to Python."""

    k: float
    """Number of prior SDs of E[y|x] in half the response range; larger shrinks more."""

    mem_cache_for_speed: bool
    """Whether the Java backend cached predictor-index sets at the nodes."""

    mh_prob_steps: Float64[ndarray, ' 3']
    """Probabilities of the grow/prune/change Metropolis-Hastings proposals (normalized)."""

    misclassification_error: float | None = None
    """In-sample misclassification rate (classification with `run_in_sample` only)."""

    model_matrix_training_data: Float64[ndarray, 'n p+1']
    """Preprocessed training matrix; the response is the last column (1 = first level)."""

    n: int
    """Number of training observations."""

    nu: float
    """Degrees of freedom of the inverse-chi-squared error-variance prior (regression)."""

    num_burn_in: int
    """Number of burn-in MCMC iterations discarded."""

    num_cores: int
    """Number of threads used to fit the model."""

    num_gibbs: int
    """Total number of MCMC iterations, burn-in included."""

    num_iterations_after_burn_in: int
    """Number of posterior draws kept."""

    num_rand_samps_in_library: int
    """Size of the pre-drawn normal/chi-squared sample library passed to Java."""

    num_trees: int
    """Number of trees in the sum-of-trees model."""

    p: int
    """Number of predictors after preprocessing (factors expanded, missingness dummies included)."""

    p_hat_train: Float64[ndarray, ' n'] | None = None
    """In-sample probability of the first level (classification with `run_in_sample` only)."""

    pred_type: str
    """``'regression'`` or ``'classification'``."""

    prob_rule_class: float
    """Probability threshold above which class predictions get the first level."""

    q: float
    """Quantile of the error-variance prior at which `sig_sq_est` is placed (regression)."""

    replace_missing_data_with_x_j_bar: bool
    """Whether missing entries were imputed with column averages/modes."""

    residuals: Float64[ndarray, ' n'] | None = None
    """In-sample ``y - y_hat_train`` (regression with `run_in_sample` only)."""

    rmse_train: float | None = None
    """In-sample root-mean-square error (regression with `run_in_sample` only)."""

    run_in_sample: bool
    """Whether the in-sample (`*_train`) outputs were computed."""

    s_sq_y: str
    """How `sig_sq_est` is estimated, ``'mse'`` (linear model) or ``'var'`` (sample variance)."""

    seed: int | None = None
    """Seed of the Java RNG; ``None`` if not given."""

    serialize: bool
    """Whether the Java model was serialized into the R object (to survive saving)."""

    sig_sq_est: float | None = None
    """Data-based error-variance estimate anchoring the prior (regression only)."""

    time_to_build: float
    """Wall-clock seconds taken to fit the model."""

    training_data_features: String[ndarray, ' <=p']
    """Names of the design-matrix columns, excluding the missingness dummies."""

    training_data_features_with_missing_features: String[ndarray, ' p']
    """Names of all design-matrix columns, missingness dummies included (if used)."""

    use_missing_data: bool
    """Whether missing entries were handled natively by the splits."""

    use_missing_data_dummies_as_covars: bool
    """Whether per-predictor missingness dummies were added to the design matrix."""

    use_xoshiro: bool
    """Whether the Java backend used the xoshiro RNG."""

    verbose: bool
    """Whether fitting messages were printed."""

    y: Float64[ndarray, ' n'] | String[ndarray, ' n']
    """Training response, numeric for regression and the labels for classification."""

    y_hat_train: Float64[ndarray, ' n'] | String[ndarray, ' n'] | None = None
    """In-sample posterior means (regression) or thresholded labels (classification).

    Computed with `run_in_sample` only.
    """

    y_levels: String[ndarray, ' 2'] | None = None
    """The response levels, the first being the target one (classification only)."""

    # components that are R NULL when absent, exposed as None
    _null_components = (
        'cov_prior_vec',
        'interaction_constraints',
        'seed',
        'sig_sq_est',
        'y_levels',
    )

    # scalar components, as (name, type): R returns them as length-1 vectors,
    # sometimes of the wrong type (e.g. doubles for integer parameters)
    _scalar_components = (
        ('L1_err_train', float),
        ('L2_err_train', float),
        ('PseudoRsq', float),
        ('alpha', float),
        ('beta', float),
        ('debug_log', bool),
        ('flush_indices_to_save_RAM', bool),
        ('impute_missingness_with_rf_impute', bool),
        ('impute_missingness_with_x_j_bar_for_lm', bool),
        ('k', float),
        ('mem_cache_for_speed', bool),
        ('misclassification_error', float),
        ('n', int),
        ('nu', float),
        ('num_burn_in', int),
        ('num_cores', int),
        ('num_gibbs', int),
        ('num_iterations_after_burn_in', int),
        ('num_rand_samps_in_library', int),
        ('num_trees', int),
        ('p', int),
        ('pred_type', str),
        ('prob_rule_class', float),
        ('q', float),
        ('replace_missing_data_with_x_j_bar', bool),
        ('rmse_train', float),
        ('run_in_sample', bool),
        ('s_sq_y', str),
        ('seed', int),
        ('serialize', bool),
        ('sig_sq_est', float),
        ('use_missing_data', bool),
        ('use_missing_data_dummies_as_covars', bool),
        ('use_xoshiro', bool),
        ('verbose', bool),
    )

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)

        # fix up attributes
        for name in self._null_components:
            if getattr(self, name) is robjects.NULL:
                setattr(self, name, None)
        for name, pytype in self._scalar_components:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, pytype(value.item()))

        # R's difftime auto-selects its unit; have R convert to seconds
        as_secs = robjects.r('function(t) as.numeric(t, units = "secs")')
        time_to_build = as_secs(self._robject.rx2('time_to_build'))
        self.time_to_build = self._r2py(time_to_build).item()

        if self.interaction_constraints is not None:
            constraints = cast(NamedList, self.interaction_constraints)
            self.interaction_constraints = tuple(it.value for it in constraints.items())

    @rmethod
    def predict(
        self, new_data: DataFrame, *args, **kw
    ) -> Float64[ndarray, ' m'] | String[ndarray, ' m']:
        """Posterior-mean predictions at the rows of `new_data`.

        For regression fits, the posterior mean of f(x). For classification
        fits, the probability of the first level (``type='prob'``, the
        default) or the corresponding labels (``type='class'``).
        """
        ...


@partial(rfunction, library='bartMachine', rname='bart_machine_get_posterior')
def _bart_machine_get_posterior(
    bart_machine: bartMachine, new_data: DataFrame, *args, **kw
) -> object:
    """Call R's `bart_machine_get_posterior`; returns an R list."""
    ...


def bart_machine_get_posterior(
    bart_machine: bartMachine, new_data: DataFrame, *args, **kw
) -> Posterior:
    """Posterior draws of f(x) at the rows of `new_data`.

    The draws are probabilities of the first level for classification fits.
    R returns a list, exposed here as a `Posterior` dict.
    """
    out = cast(
        NamedList, _bart_machine_get_posterior(bart_machine, new_data, *args, **kw)
    )
    return cast(Posterior, {str(it.name): it.value for it in out.items()})


@partial(rfunction, library='bartMachine', rname='bart_machine_num_cores')
def _bart_machine_num_cores() -> object:
    """Call R's `bart_machine_num_cores`; returns a length-1 vector."""
    ...


def bart_machine_num_cores() -> int:
    """Return the number of threads `bartMachine` uses to fit the model."""
    return int(cast(ndarray, _bart_machine_num_cores()).item())


@partial(rfunction, library='bartMachine')
def get_sigsqs(
    bart_machine: bartMachine, *args, **kw
) -> (
    Float64[ndarray, ' num_iterations_after_burn_in'] | Float64[ndarray, ' num_gibbs+1']
):
    """Posterior draws of the error variance (regression fits only).

    Burn-in draws are dropped unless ``after_burn_in=False``, which also
    keeps the pre-MCMC initial value as first entry.
    """
    ...


@partial(rfunction, library='bartMachine', rname='set_bart_machine_num_cores')
def _set_bart_machine_num_cores(num_cores: int) -> object:
    """Call R's `set_bart_machine_num_cores`; returns NULL."""
    ...


def set_bart_machine_num_cores(num_cores: int) -> None:
    """Set the number of threads `bartMachine` uses to fit the model.

    A package-global setting that persists across fits.
    """
    _set_bart_machine_num_cores(num_cores)
