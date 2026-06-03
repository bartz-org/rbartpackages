# rbartpackages/src/rbartpackages/_base.py
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

import ctypes
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from functools import wraps
from re import fullmatch, match
from typing import Any

import numpy as np
from rpy2 import robjects
from rpy2.robjects import BoolVector, conversion, numpy2ri
from rpy2.robjects.help import Package
from rpy2.robjects.methods import RS4

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
        df = df.to_pandas()
        return pandas2ri.py2rpy(df)

    POLARS_CONVERTER.py2rpy.register(pl.DataFrame, polars_to_r)
    POLARS_CONVERTER.py2rpy.register(pl.Series, polars_to_r)

# converter for jax
JAX_CONVERTER = conversion.Converter('jax')
try:
    import jax
except ImportError:  # pragma: no cover - optional dep always present in CI
    pass
else:

    def jax_to_r(x: jax.Array) -> object:
        x = np.asarray(x)
        if x.ndim == 0:
            x = x[()]
        return numpy2ri.py2rpy(x)

    JAX_CONVERTER.py2rpy.register(jax.Array, jax_to_r)

# converter for numpy
NUMPY_CONVERTER = numpy2ri.converter


# converter for BoolVector (why isn't it in the numpy converter?)
def bool_vector_to_python(x: BoolVector) -> np.ndarray[Any, np.dtype[np.bool_]]:
    return np.array(x, bool)


BOOL_VECTOR_CONVERTER = conversion.Converter('bool_vector')
BOOL_VECTOR_CONVERTER.rpy2py.register(BoolVector, bool_vector_to_python)


# converter for python dictionaries
DICT_CONVERTER = conversion.Converter('dict')


def dict_to_r(x: dict[str, Any]) -> robjects.ListVector:
    return robjects.ListVector(x)


DICT_CONVERTER.py2rpy.register(dict, dict_to_r)

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
    """Cap OpenMP/OpenBLAS thread pools at one thread for the duration.

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

    _rfuncname : str
        An R function in the format ``'<package>::<function>``. The function is
        called with the initialization arguments, converted to R objects, and is
        expected to return an R object. The attributes of the R object are
        converted to equivalent Python values and set as attributes of the
        Python object. The R object itself is assigned to the member `_robject`.
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

    def _py2r(self, x: object) -> object:
        if isinstance(x, __class__):
            return x._robject  # noqa: SLF001, same-class access
        with self._convctx:
            return self._converter.py2rpy(x)

    def _r2py(self, x: object) -> object:
        with self._convctx:
            return self._converter.rpy2py(x)

    def _args2r(self, args: Iterable[Any]) -> tuple[Any, ...]:
        return tuple(map(self._py2r, args))

    def _kw2r(self, kw: Mapping[str, Any]) -> dict[str, Any]:
        return {key: self._py2r(value) for key, value in kw.items()}

    _rfuncname: str = NotImplemented

    @property
    def _library(self) -> str:
        """Parse `_rfuncname` to get the library. Also checks `_rfuncname` is valid."""
        pattern = rf'^({R_IDENTIFIER})::({R_IDENTIFIER})$'
        m = match(pattern, self._rfuncname)
        if m is None:
            msg = f'Invalid _rfuncname: {self._rfuncname}.'
            raise ValueError(msg)
        return m.group(1)

    def __init__(self, *args: Any, **kw: Any) -> None:
        robjects.r(f'loadNamespace("{self._library}")')
        func = robjects.r(self._rfuncname)
        obj = func(*self._args2r(args), **self._kw2r(kw))
        self._robject = obj
        if hasattr(obj, 'items'):
            for s, v in obj.items():
                setattr(self, s.replace('.', '_'), self._r2py(v))

    def __init_subclass__(cls, **kw: Any) -> None:
        """Automatically add R documentation to subclasses."""
        library, name = cls._rfuncname.split('::')
        page = Package(library).fetch(name)
        if cls.__doc__ is None:
            cls.__doc__ = ''
        cls.__doc__ += 'R documentation:\n' + page.to_docstring()


def rmethod(meth: Callable, *, rname: str | None = None) -> Callable:
    """Automatically implement a method using the correspoding R method.

    Parameters
    ----------
    meth
        A method in a subclass of `RObjectBase`.
    rname
        The name of the method in R. If not specified, use the name of `meth`.

    Returns
    -------
    methimpl
        An implementation of the method that calls the R method. The original
        implementation of meth is completely discarded.

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

    return impl
