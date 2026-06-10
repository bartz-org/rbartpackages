# rbartpackages/src/rbartpackages/_src/base.py
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

"""Implementation of `rbartpackages.base`."""

import ctypes
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from functools import partial, wraps
from inspect import cleandoc
from re import fullmatch, match
from textwrap import indent
from typing import Any, NamedTuple, Protocol, TypedDict, cast

import numpy as np
from jaxtyping import AbstractDtype, Float64, Int32
from numpy import ndarray
from rpy2 import robjects
from rpy2.rlike.container import NamedList
from rpy2.robjects import BoolVector, conversion, numpy2ri
from rpy2.robjects.help import Package
from rpy2.robjects.methods import RS4

# WORKAROUND(python<3.11): import Self from typing
from typing_extensions import Self

# converter for pandas
PANDAS_CONVERTER = conversion.Converter('pandas')
try:
    from rpy2.robjects import pandas2ri
except ImportError:  # pragma: no cover - optional dep always present in CI
    pass
else:
    PANDAS_CONVERTER = pandas2ri.converter

# converter for polars
POLARS_CONVERTER = conversion.Converter('polars')
try:
    import polars as pl
    from rpy2.robjects import pandas2ri
except ImportError:  # pragma: no cover - optional dep always present in CI
    pass
else:

    def polars_to_r(df: pl.DataFrame) -> object:
        """Convert a polars dataframe or series to R through pandas."""
        df = df.to_pandas()
        return pandas2ri.py2rpy(df)

    def r_to_polars(df: object) -> pl.DataFrame:
        """
        Convert an R data frame to polars through pandas.

        Registered so that, when polars is installed, it wins over the pandas
        converter for R data frames (the converters are summed with polars
        after pandas, so its later registration takes precedence). polars has
        no row index, so an R data frame's row names are dropped.
        """
        return pl.from_pandas(pandas2ri.rpy2py(df))

    POLARS_CONVERTER.py2rpy.register(pl.DataFrame, polars_to_r)
    POLARS_CONVERTER.py2rpy.register(pl.Series, polars_to_r)
    POLARS_CONVERTER.rpy2py.register(robjects.vectors.DataFrame, r_to_polars)

# converter for jax
JAX_CONVERTER = conversion.Converter('jax')
try:
    import jax
except ImportError:  # pragma: no cover - optional dep always present in CI
    pass
else:

    def jax_to_r(x: jax.Array) -> object:
        """Convert a jax array to R, unwrapping 0-dim arrays to scalars."""
        x = np.asarray(x)
        if x.ndim == 0:
            x = x[()]
        return numpy2ri.py2rpy(x)

    JAX_CONVERTER.py2rpy.register(jax.Array, jax_to_r)

# converter for numpy
NUMPY_CONVERTER = numpy2ri.converter


# converter for BoolVector (why isn't it in the numpy converter?)
def bool_vector_to_python(x: BoolVector) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Convert an R logical vector to a numpy boolean array."""
    return np.array(x, bool)


BOOL_VECTOR_CONVERTER = conversion.Converter('bool_vector')
BOOL_VECTOR_CONVERTER.rpy2py.register(BoolVector, bool_vector_to_python)


# converter for python dictionaries
DICT_CONVERTER = conversion.Converter('dict')


def dict_to_r(x: dict[str, Any]) -> robjects.ListVector:
    """Convert a dict to an R named list."""
    return robjects.ListVector(x)


DICT_CONVERTER.py2rpy.register(dict, dict_to_r)


class DataFrame(Protocol):
    """
    Duck type of the dataframe arguments accepted by the wrappers.

    Both `pandas.DataFrame` and :doc:`polars.DataFrame
    <polars:reference/dataframe/index>` match; they are converted to R data
    frames, with categorical columns becoming factors.
    """

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        """Export as an Arrow PyCapsule stream."""


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


def drop_none(kw: dict[str, Any]) -> dict[str, Any]:
    """
    Drop the arguments left to ``None`` to let R compute its defaults.

    Parameters
    ----------
    kw
        Arguments of an R function, by name.

    Returns
    -------
    The arguments whose value is not ``None``.
    """
    return {name: value for name, value in kw.items() if value is not None}


def namedlist_to_dict(namedlist: NamedList) -> dict[str, Any]:
    """
    Convert an R named list to a dict.

    Parameters
    ----------
    namedlist
        The converted R list.

    Returns
    -------
    The list values by name, with ``.`` in names replaced by ``_`` and NULL values by ``None``.
    """
    return {
        str(it.name).replace('.', '_'): None if it.value is robjects.NULL else it.value
        for it in namedlist.items()
    }


def parse_rm_const(
    rm_const: Int32[ndarray, ' k'], ncols: int, *, removed: bool
) -> Int32[ndarray, ' <=p']:
    """
    Convert the ``rm.const`` component of a BART/BART3 object to 0-based indices.

    Parameters
    ----------
    rm_const
        The ``rm.const`` component: R reports the kept design-matrix columns
        as positive 1-based indices, or the dropped constant ones as negative
        1-based indices into the pre-removal design matrix.
    ncols
        Number of columns of the design matrix at hand, after the removal if
        any.
    removed
        Whether the constant columns were removed from that matrix, making
        the pre-removal column count ``ncols + rm_const.size`` when `rm_const`
        lists the dropped columns.

    Returns
    -------
    0-based indices of the kept columns of the pre-removal design matrix.

    Raises
    ------
    ValueError
        If the indices cannot be parsed because they change sign.
    """
    if np.all(rm_const < 0):
        p = ncols + rm_const.size if removed else ncols
        keep = np.ones(p, bool)
        keep[-rm_const - 1] = False
        return np.arange(p, dtype=np.int32)[keep]
    elif np.all(rm_const > 0):
        return rm_const - 1
    else:  # pragma: no cover - R gives all-positive or all-negative indices
        msg = 'failed to parse rm.const because indices change sign'
        raise ValueError(msg)


def convert_treedraws(treedraws: NamedList) -> TreeDraws:
    """
    Convert the ``treedraws`` component of a ``gbart``-like fit to a `TreeDraws` dict.

    Parameters
    ----------
    treedraws
        The converted R ``treedraws`` list.

    Returns
    -------
    The per-variable cutpoints (keyed by column index or name) and the serialized trees.
    """
    cutpoints: NamedList = treedraws.getbyname('cutpoints')
    return {
        'cutpoints': {
            i if it.name is None else it.name.item(): it.value
            for i, it in enumerate(cutpoints.items())
        },
        'trees': treedraws.getbyname('trees').item(),
    }


def convert_gbart_predict(
    out: object,
) -> Float64[ndarray, 'ndpost m'] | Float64[ndarray, ' m'] | PredictBinary:
    """
    Normalize the output of R's ``predict`` method for ``gbart``-like fits.

    Parameters
    ----------
    out
        The converted R output: an array of function draws (or their means)
        for continuous fits, an R named list for binary ones.

    Returns
    -------
    The array unchanged, or the list as a dict of arrays with the scalar ``binaryOffset`` unwrapped.
    """
    if not hasattr(out, 'items'):
        return out  # continuous: a draws matrix or its column means
    else:
        # binary: convert R's list (a NamedList) to a dict of arrays
        result = namedlist_to_dict(cast(NamedList, out))
        result['binaryOffset'] = result['binaryOffset'].item()
        return result


R_IDENTIFIER = r'(?:[a-zA-Z]|\.(?![0-9]))[a-zA-Z0-9._]*'

# In-process native thread pools to cap before R forks. R's
# `parallel::mcparallel` (used by the `mc.*` BART functions) forks, but GNU
# libgomp is not fork-safe: a forked child that enters an OpenMP parallel region
# hangs forever on a barrier because the worker threads do not survive the fork.
# The threaded OpenBLAS that R's LAPACK calls (e.g. `summary(lm(...))` for the
# `sigest` default) dispatches through libgomp, so a child deadlocks there.
# Running these pools single-threaded across the fork stops the thread team from
# being started at all, sidestepping the deadlock. Each entry is a (getter,
# setter) pair of C symbols; missing ones (e.g. a single-threaded reference BLAS)
# are skipped.
NATIVE_THREAD_POOLS = (
    ('omp_get_max_threads', 'omp_set_num_threads'),
    ('openblas_get_num_threads', 'openblas_set_num_threads'),
)


@contextmanager
def fork_safe_native_threads() -> Iterator[None]:
    """
    Cap OpenMP/OpenBLAS thread pools at one thread for the duration.

    Workaround for the deadlock that hangs the children forked by R's
    ``parallel::mcparallel`` when GNU libgomp has a live thread pool (see
    `NATIVE_THREAD_POOLS`). The previous thread counts are restored on exit.
    """
    handle = ctypes.CDLL(None)
    saved = []
    for getter_name, setter_name in NATIVE_THREAD_POOLS:
        try:
            getter = getattr(handle, getter_name)
            setter = getattr(handle, setter_name)
        except AttributeError:
            continue
        getter.restype = ctypes.c_int
        setter.argtypes = (ctypes.c_int,)
        saved.append((setter, getter()))
        setter(1)
    try:
        yield
    finally:
        for setter, nthreads in saved:
            setter(nthreads)


class RObjectBase:
    """
    Base class for Python wrappers of R objects creators.

    Subclasses should define the class attribute `_rfuncname`, and declare
    stub methods decorated with `rmethod`.
    """

    _converter = (
        robjects.default_converter
        + PANDAS_CONVERTER
        + POLARS_CONVERTER
        + NUMPY_CONVERTER
        + BOOL_VECTOR_CONVERTER
        + JAX_CONVERTER
        + DICT_CONVERTER
    )
    _convctx = conversion.localconverter(_converter)

    @classmethod
    def _py2r(cls, x: object) -> object:
        if isinstance(x, __class__):
            return x._robject  # noqa: SLF001, same-class access
        with cls._convctx:
            return cls._converter.py2rpy(x)

    @classmethod
    def _r2py(cls, x: object) -> object:
        with cls._convctx:
            return cls._converter.rpy2py(x)

    @classmethod
    def _args2r(cls, args: Iterable[Any]) -> tuple[Any, ...]:
        return tuple(map(cls._py2r, args))

    @classmethod
    def _kw2r(cls, kw: Mapping[str, Any]) -> dict[str, Any]:
        return {key: cls._py2r(value) for key, value in kw.items()}

    _rfuncname: str = NotImplemented
    """R function to call, as ``'<package>::<function>'``.

    Called with the initialization arguments converted to R objects; the R
    object it returns is stored in `_robject`, and that object's named
    components are converted to Python values and set as attributes.
    """

    _robject: object
    """The R object returned by `_rfuncname`, whose components become attributes."""

    @property
    def _library(self) -> str:
        """Parse `_rfuncname` to get the library. Also checks `_rfuncname` is valid."""
        pattern = rf'^({R_IDENTIFIER})::({R_IDENTIFIER})$'
        m = match(pattern, self._rfuncname)
        if m is None:
            msg = f'Invalid _rfuncname: {self._rfuncname}.'
            raise ValueError(msg)
        return m.group(1)

    @staticmethod
    def _has_named_components(obj: object) -> bool:
        """
        Whether `obj` exposes named components to set as attributes.

        Only an R named list qualifies. A bare matrix (as `bartModelMatrix`
        gives with ``numcut=0``) is excluded by the `ListVector` check: rpy2
        reports a matrix's dimnames as ``names``, so the names check alone
        would not cut it out.
        """
        names = getattr(obj, 'names', None)
        return (
            isinstance(obj, robjects.vectors.ListVector)
            and names is not None
            and names is not robjects.NULL
        )

    def _invoke_rfunc(self, args: Iterable[Any], kw: Mapping[str, Any]) -> object:
        """Load the namespace and call `_rfuncname` on the converted arguments."""
        robjects.r(f'loadNamespace("{self._library}")')
        func = robjects.r(self._rfuncname)
        return func(*self._args2r(args), **self._kw2r(kw))

    def _set_attrs_from_robject(self) -> None:
        """Set the named components of `_robject` as Python attributes."""
        if self._has_named_components(self._robject):
            for s, v in self._robject.items():
                setattr(self, s.replace('.', '_'), self._r2py(v))

    def __init__(self, *args: Any, **kw: Any) -> None:
        self._robject = self._invoke_rfunc(args, kw)
        self._set_attrs_from_robject()

    @classmethod
    def _wrap(cls, robject: object) -> Self:
        """
        Wrap an existing R object, skipping the call to the R function.

        The named components of `robject` are exposed as attributes like in
        `__init__`, but any subclass post-processing of the attributes is
        skipped.
        """
        self = object.__new__(cls)
        self._robject = robject
        self._set_attrs_from_robject()
        return self

    def _call_rmethod(self, rname: str, *args: Any, **kw: Any) -> object:
        """
        Call the R method `rname` of `_robject` on the converted arguments.

        Dispatches on the kind of R object: an S4 object (e.g. a reference
        class such as the `dbarts` sampler) carries its methods as fields, so
        the method is fetched with R's ``$`` operator; for anything else the
        S3 method matching `_robject`'s class is looked up in `_library`. The
        result is converted back to Python.
        """
        if isinstance(self._robject, RS4):
            func = robjects.r['$'](self._robject, rname)
            out = func(*self._args2r(args), **self._kw2r(kw))
        else:
            if not fullmatch(R_IDENTIFIER, rname):
                msg = f'Invalid R method name: {rname}'
                raise ValueError(msg)
            rclass = self._robject.rclass[0]
            func = robjects.r(
                f'getS3method("{rname}", "{rclass}", envir = asNamespace("{self._library}"))'
            )
            out = func(self._robject, *self._args2r(args), **self._kw2r(kw))
        return self._r2py(out)

    def __init_subclass__(cls, **kw: Any) -> None:
        """Automatically add R documentation to subclasses."""
        if cls._rfuncname is NotImplemented:
            # an intermediate class shared between wrapper modules; only its
            # concrete subclasses have an R function to document
            return
        library, name = cls._rfuncname.split('::')
        page = Package(library).fetch(name)
        # the leading empty line keeps the docstring processors' dedent of
        # everything-after-the-first-line a no-op
        parts = ['']
        if cls.__doc__:
            # dedent the hand-written docstring so that the appended text sits
            # at the same indentation level (sections would not parse otherwise)
            parts.append(cleandoc(cls.__doc__))
        # a numpy-style section header (registered with napoleon in the docs
        # config) so that a Parameters section in the docstring cannot absorb
        # the appendix; the R help text is plain text, not valid RST, hence
        # the literal block, which docutils renders verbatim
        parts.append(
            'R documentation\n---------------\n::\n\n'
            + indent(page.to_docstring(), '    ')
        )
        cls.__doc__ = '\n\n'.join(parts)


def rmethod(meth: Callable, *, rname: str | None = None) -> Callable:
    """
    Automatically implement a method using the correspoding R method.

    Parameters
    ----------
    meth
        A method in a subclass of `RObjectBase`.
    rname
        The name of the method in R. If not specified, use the name of `meth`.

    Returns
    -------
    An implementation of the method that calls the R method, discarding the original implementation of `meth`.

    Examples
    --------
    >>> class MyRObject(RObjectBase):
    ...     _rfuncname = 'mypackage::myfunction'
    ...     @partial(rmethod, rname='my.method')
    ...     def my_method(self, arg1: int, arg2: str):
    ...         ...
    """
    if rname is None:
        rname = meth.__name__

    # I can't automatically add a docstring to the method because the R class
    # can be determined at runtime

    @wraps(meth)
    def impl(self: RObjectBase, *args: Any, **kw: Any) -> object:
        return self._call_rmethod(rname, *args, **kw)

    return impl


def rproperty(
    meth: Callable | None = None,
    *,
    rname: str | None = None,
    wrap: type[RObjectBase] | None = None,
) -> property | Callable[[Callable], property]:
    """
    Automatically implement a read-only property using the corresponding R field.

    Unlike the attributes `RObjectBase` snapshots at initialization, the field
    is extracted from the R object at each access, so it tracks mutable
    objects such as reference-class instances.

    Parameters
    ----------
    meth
        A method in a subclass of `RObjectBase`. The original implementation
        is completely discarded. If not given, return a decorator instead, to
        allow using the keyword arguments.
    rname
        The name of the field in R. If not specified, use the name of `meth`.
    wrap
        A `RObjectBase` subclass to wrap the field with, instead of
        converting it to a Python value.

    Returns
    -------
    A read-only property that extracts the field with R's ``$`` operator. NULL fields are exposed as ``None``.

    Examples
    --------
    >>> class MyRObject(RObjectBase):
    ...     _rfuncname = 'mypackage::myfunction'
    ...     @rproperty(rname='my.field')
    ...     def my_field(self):
    ...         ...
    """
    if meth is None:
        return partial(rproperty, rname=rname, wrap=wrap)
    if rname is None:
        rname = meth.__name__

    @wraps(meth)
    def impl(self: RObjectBase) -> object:
        out = robjects.r['$'](self._robject, rname)
        if out is robjects.NULL:
            return None
        elif wrap is None:
            return self._r2py(out)
        else:
            return wrap._wrap(out)  # noqa: SLF001, base-class access

    return property(impl)


def rfunction(func: Callable, *, library: str, rname: str | None = None) -> Callable:
    """
    Automatically implement a function using the corresponding R function.

    Parameters
    ----------
    func
        A function. Its original implementation is completely discarded.
    library
        The R package the function is fetched from. The fetch is eager: it
        happens at decoration time and loads the package namespace.
    rname
        The name of the function in R. If not specified, use the name of
        `func`.

    Returns
    -------
    An implementation that calls the R function on the converted arguments; `RObjectBase` instances are passed as their wrapped R objects.

    Raises
    ------
    ValueError
        If `library` or `rname` is not a valid R identifier.

    Examples
    --------
    >>> @partial(rfunction, library='mypackage', rname='my.function')
    ... def my_function(obj: MyRObject, arg1: int, arg2: str):
    ...     ...
    """
    if rname is None:
        rname = func.__name__
    if not fullmatch(R_IDENTIFIER, library):
        msg = f'Invalid R package name: {library}'
        raise ValueError(msg)
    if not fullmatch(R_IDENTIFIER, rname):
        msg = f'Invalid R function name: {rname}'
        raise ValueError(msg)
    robjects.r(f'loadNamespace("{library}")')
    rfunc = robjects.r(f'{library}::{rname}')

    @wraps(func)
    def impl(*args: Any, **kw: Any) -> object:
        out = rfunc(*RObjectBase._args2r(args), **RObjectBase._kw2r(kw))  # noqa: SLF001, base-class access
        return RObjectBase._r2py(out)  # noqa: SLF001, base-class access

    return impl


class bartModelMatrixBase(RObjectBase):
    """
    Shared implementation of the BART/BART3 ``bartModelMatrix`` wrappers.

    Subclasses define `_rfuncname` and the `grp` attribute, whose meaning
    differs between the two packages.
    """

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

        # grp is R NULL when absent (see the subclasses); expose it as None.
        if self.grp is robjects.NULL:
            self.grp = None

        # whether R also removed the flagged constant columns from X depends
        # on the rm_const argument
        _, n_cols = self.X.shape
        self.rm_const = parse_rm_const(self.rm_const, n_cols, removed=rm_const)

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
