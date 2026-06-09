# rbartpackages/src/rbartpackages/_src/bartMachine.py
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

"""Implementation of `rbartpackages.bartMachine`."""

from functools import partial
from typing import Literal, TypedDict, cast

from jaxtyping import AbstractDtype, Float64, Integer
from numpy import ndarray
from rpy2 import robjects
from rpy2.rlike.container import NamedList
from rpy2.robjects.methods import RS4

from rbartpackages._src.base import DataFrame, RObjectBase, drop_none, rfunction

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


def to_response(y: object) -> object:
    """
    Convert a numpy array `y` to a bare R vector; pass anything else through.

    rpy2's numpy bridge tags even a 1-D array with a ``dim``, which bartMachine
    rejects (it wants a bare atomic vector), so the R vector is built directly:
    a numeric array becomes a numeric vector (a regression response), a string
    or object array becomes a factor with alphabetically ordered levels (a
    classification response). pandas/polars series, which already convert to
    the right thing, pass through unchanged.
    """
    if isinstance(y, ndarray):
        if y.dtype.kind in 'OSU':
            return robjects.r['as.factor'](robjects.StrVector(y.astype(str).tolist()))
        return robjects.FloatVector(y.astype(float).tolist())
    return y


class bartMachine(RObjectBase):
    """
    Fit BART to continuous or binary outcomes.

    Python interface to R's ``bartMachine::bartMachine``. The predictors `X`
    must be a data frame (factor columns are expanded internally); `y` is a
    numeric vector for regression or a two-level factor for classification.
    Pass `X` and `y` separately, or combined as `Xy`. The number of fitting
    threads is a package-global setting, see `set_bart_machine_num_cores`.
    Arguments left to ``None`` are omitted from the R call, so R computes its
    own defaults, described below.

    Parameters
    ----------
    X
        Data frame of predictors; rows are observations. Factors are expanded
        into indicator columns internally.
    y
        Response: numeric for regression, two-level categorical for
        classification. A numeric numpy array or pandas/polars ``Series`` works
        for regression; classification needs a factor, so pass a string/object
        numpy array (whose levels are then ordered alphabetically, the first
        being the positive one) or a categorical ``Series`` (which controls the
        level order).
    Xy
        Predictors and response combined in one data frame, the response in a
        column named ``'y'``; an alternative to passing `X` and `y`.
    num_trees
        Number of trees in the sum-of-trees model.
    num_burn_in
        Number of burn-in MCMC iterations discarded.
    num_iterations_after_burn_in
        Number of posterior draws kept after burn-in.
    alpha
        Base of the nonterminal-node probability in the tree prior.
    beta
        Power of the nonterminal-node probability in the tree prior.
    k
        Number of prior SDs of E[y|x] in half the response range; larger
        shrinks more.
    q
        Quantile of the error-variance prior at which the data-based estimate
        is placed (regression only).
    nu
        Degrees of freedom of the inverse-chi-squared error-variance prior
        (regression only).
    prob_rule_class
        Probability threshold above which a class prediction gets the first
        (positive) level (classification only).
    mh_prob_steps
        Prior probabilities of the grow/prune/change Metropolis-Hastings tree
        proposals; default ``(2.5, 2.5, 4) / 9``.
    debug_log
        Whether the Java backend logs to a file in the working directory.
    run_in_sample
        Whether the in-sample (``*_train``) statistics are computed.
    s_sq_y
        How the error-variance estimate is computed, ``'mse'`` (least-squares
        residuals) or ``'var'`` (response variance); regression only.
    sig_sq_est
        Data-based error-variance estimate anchoring the prior; default a
        linear-model estimate (regression only).
    print_tree_illustrations
        Whether every Gibbs iteration prints a side-by-side tree illustration;
        extremely slow.
    cov_prior_vec
        Relative split-proposal weight of each predictor (after dummification
        and missingness augmentation); internally normalized.
    interaction_constraints
        Groups of predictors allowed to interact, as a dict of vectors of
        1-based column indices or column names (e.g.
        ``{'a': [1, 2], 'b': ['nox']}``); rpy2 converts a dict, not a list, to
        the R list bartMachine wants.
    use_missing_data
        Whether missing entries are handled natively by the splits, without
        imputation.
    num_rand_samps_in_library
        Size of the pre-drawn normal/chi-squared sample library passed to Java.
    use_missing_data_dummies_as_covars
        Whether per-predictor missingness indicators are added to the design
        matrix.
    replace_missing_data_with_x_j_bar
        Whether missing entries are imputed with column averages/modes.
    impute_missingness_with_rf_impute
        Whether missing entries are filled with ``randomForest::rfImpute``.
    impute_missingness_with_x_j_bar_for_lm
        Whether the linear model behind `sig_sq_est` imputes missing entries
        with column averages/modes.
    mem_cache_for_speed
        Whether the Java backend caches the candidate split values at each
        node; faster but memory-hungry.
    flush_indices_to_save_RAM
        Whether the Java backend flushes internal indices to save memory
        (disables ``node_prediction_training_data_indices`` and
        ``get_projection_weights``).
    serialize
        Whether the Java model is serialized into the R object so it survives
        saving and reloading; memory-hungry.
    seed
        Seed of the R and Java RNGs; ``None`` does not seed. Deterministic only
        when fitting single-threaded.
    use_xoshiro
        Whether the Java backend uses the Xoshiro256PlusPlus RNG rather than
        the legacy MersenneTwister.
    verbose
        Whether fitting progress is printed to the screen.

    Notes
    -----
    The private R argument ``covariates_to_permute`` (used internally by
    ``cov_importance_test``) is not exposed.
    """

    _rfuncname = 'bartMachine::bartMachine'

    L1_err_train: float | None = None
    """In-sample L1 error (regression with `run_in_sample` only)."""

    L2_err_train: float | None = None
    """In-sample L2 error (regression with `run_in_sample` only)."""

    PseudoRsq: float | None = None
    """In-sample ``1 - L2_err_train / L2 of the mean`` (regression with `run_in_sample` only)."""

    X: DataFrame
    """Training predictors as supplied (factors not expanded).

    A polars frame if polars is installed, else a pandas one.
    """

    alpha: float
    """Base of the nonterminal-node probability in the tree prior."""

    beta: float
    """Power of the nonterminal-node probability in the tree prior."""

    confusion_matrix: DataFrame | None = None
    """In-sample confusion matrix with error rates (classification with `run_in_sample` only).

    A polars frame (so without the row labels) if polars is installed, else a pandas one.
    """

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
    """Whether missing training entries got ``randomForest::rfImpute`` imputations added."""

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
    """Whether the in-sample (``*_train``) outputs were computed."""

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

    def __init__(
        self,
        X: DataFrame | None = None,
        y: Float64[ndarray, ' n'] | String[ndarray, ' n'] | None = None,
        *,
        Xy: DataFrame | None = None,
        num_trees: int = 50,
        num_burn_in: int = 250,
        num_iterations_after_burn_in: int = 1000,
        alpha: float = 0.95,
        beta: float = 2.0,
        k: float = 2.0,
        q: float = 0.9,
        nu: float = 3.0,
        prob_rule_class: float = 0.5,
        mh_prob_steps: Float64[ndarray, ' 3'] | None = None,
        debug_log: bool = False,
        run_in_sample: bool = True,
        s_sq_y: Literal['mse', 'var'] = 'mse',
        sig_sq_est: float | None = None,
        print_tree_illustrations: bool = False,
        cov_prior_vec: Float64[ndarray, ' p'] | None = None,
        interaction_constraints: dict[
            str, Integer[ndarray, ' k'] | String[ndarray, ' k']
        ]
        | None = None,
        use_missing_data: bool = False,
        num_rand_samps_in_library: int = 10000,
        use_missing_data_dummies_as_covars: bool = False,
        replace_missing_data_with_x_j_bar: bool = False,
        impute_missingness_with_rf_impute: bool = False,
        impute_missingness_with_x_j_bar_for_lm: bool = True,
        mem_cache_for_speed: bool = True,
        flush_indices_to_save_RAM: bool = True,
        serialize: bool = False,
        seed: int | None = None,
        use_xoshiro: bool = False,
        verbose: bool = True,
    ) -> None:
        kw = {
            'X': X,
            'y': to_response(y),
            'Xy': Xy,
            'num_trees': num_trees,
            'num_burn_in': num_burn_in,
            'num_iterations_after_burn_in': num_iterations_after_burn_in,
            'alpha': alpha,
            'beta': beta,
            'k': k,
            'q': q,
            'nu': nu,
            'prob_rule_class': prob_rule_class,
            'mh_prob_steps': mh_prob_steps,
            'debug_log': debug_log,
            'run_in_sample': run_in_sample,
            's_sq_y': s_sq_y,
            'sig_sq_est': sig_sq_est,
            'print_tree_illustrations': print_tree_illustrations,
            'cov_prior_vec': cov_prior_vec,
            'interaction_constraints': interaction_constraints,
            'use_missing_data': use_missing_data,
            'num_rand_samps_in_library': num_rand_samps_in_library,
            'use_missing_data_dummies_as_covars': use_missing_data_dummies_as_covars,
            'replace_missing_data_with_x_j_bar': replace_missing_data_with_x_j_bar,
            'impute_missingness_with_rf_impute': impute_missingness_with_rf_impute,
            'impute_missingness_with_x_j_bar_for_lm': impute_missingness_with_x_j_bar_for_lm,
            'mem_cache_for_speed': mem_cache_for_speed,
            'flush_indices_to_save_RAM': flush_indices_to_save_RAM,
            'serialize': serialize,
            'seed': seed,
            'use_xoshiro': use_xoshiro,
            'verbose': verbose,
        }
        super().__init__(**drop_none(kw))

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

    def predict(
        self,
        new_data: DataFrame,
        *,
        type: Literal['prob', 'class'] | None = None,  # noqa: A002 mirrors the R argument name
        prob_rule_class: float | None = None,
        verbose: bool | None = None,
    ) -> Float64[ndarray, ' m'] | String[ndarray, ' m']:
        """
        Posterior-mean predictions at the rows of `new_data`.

        For regression fits, the posterior mean of f(x). For classification
        fits, the probability of the first level (``type='prob'``, the
        default) or the corresponding labels (``type='class'``). Arguments
        left to ``None`` are omitted from the R call, so R computes its own
        defaults.

        Parameters
        ----------
        new_data
            Predictors to predict at, with the same columns as the training
            data.
        type
            For classification fits, whether to return the first-level
            probability (``'prob'``) or the predicted labels (``'class'``);
            ignored for regression.
        prob_rule_class
            Probability threshold for a ``'class'`` prediction; default the
            fit's `prob_rule_class`.
        verbose
            Whether to print prediction messages to the R console.

        Returns
        -------
        The posterior means (or labels with ``type='class'``) at `new_data`.
        """
        kw = {'type': type, 'prob_rule_class': prob_rule_class, 'verbose': verbose}
        return self._call_rmethod('predict', new_data, **drop_none(kw))


@partial(rfunction, library='bartMachine', rname='bart_machine_get_posterior')
def _bart_machine_get_posterior(
    bart_machine: bartMachine, new_data: DataFrame, verbose: bool
) -> object:
    """Call R's `bart_machine_get_posterior`; returns an R list."""
    ...


def bart_machine_get_posterior(
    bart_machine: bartMachine, new_data: DataFrame, *, verbose: bool = True
) -> Posterior:
    """
    Posterior draws of f(x) at the rows of `new_data`.

    The draws are probabilities of the first level for classification fits.
    R returns a list, exposed here as a `Posterior` dict.

    Parameters
    ----------
    bart_machine
        The fitted model to predict from.
    new_data
        Predictors to predict at, with the same columns as the training data.
    verbose
        Whether to print prediction messages to the R console.

    Returns
    -------
    The posterior draws at `new_data`, as a `Posterior` dict.
    """
    out = cast(
        NamedList, _bart_machine_get_posterior(bart_machine, new_data, verbose=verbose)
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
    bart_machine: bartMachine,
    after_burn_in: bool = True,
    plot_hist: bool = False,
    plot_CI: float = 0.95,
    plot_sigma: bool = False,
    verbose: bool = True,
) -> (
    Float64[ndarray, ' num_iterations_after_burn_in'] | Float64[ndarray, ' num_gibbs+1']
):
    """
    Posterior draws of the error variance (regression fits only).

    Burn-in draws are dropped unless ``after_burn_in=False``, which also
    keeps the pre-MCMC initial value as first entry.

    Parameters
    ----------
    bart_machine
        The fitted model to read the draws from.
    after_burn_in
        Whether to drop the burn-in draws (and the pre-MCMC initial value).
    plot_hist
        Whether to plot a histogram of the post-burn-in draws.
    plot_CI
        Credible-interval level marked on the histogram (with `plot_hist`).
    plot_sigma
        Whether the histogram is of the SD rather than the variance (with
        `plot_hist`).
    verbose
        Whether to print messages to the R console.

    Returns
    -------
    The error-variance draws (the pre-MCMC initial value first when ``after_burn_in=False``).
    """
    ...


@partial(rfunction, library='bartMachine', rname='set_bart_machine_num_cores')
def _set_bart_machine_num_cores(num_cores: int, verbose: bool) -> object:
    """Call R's `set_bart_machine_num_cores`; returns NULL."""
    ...


def set_bart_machine_num_cores(num_cores: int, *, verbose: bool = True) -> None:
    """
    Set the number of threads `bartMachine` uses to fit the model.

    A package-global setting that persists across fits.

    Parameters
    ----------
    num_cores
        Number of threads to use.
    verbose
        Whether to print a confirmation message to the R console.
    """
    _set_bart_machine_num_cores(num_cores, verbose=verbose)
