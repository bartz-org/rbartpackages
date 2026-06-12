# rbartpackages/tests/util.py
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

"""Functions intended to be shared across the test suite."""

from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter, signature
from operator import ge, le
from typing import Any

import numpy as np
from jaxtyping import Float64
from numpy import ndarray
from numpy.linalg import norm
from numpy.testing import assert_allclose as _np_assert_allclose  # noqa: TID251
from numpy.testing import assert_array_equal as _np_assert_array_equal  # noqa: TID251
from numpy.typing import ArrayLike


def assert_close_matrices(
    actual: ArrayLike,
    desired: ArrayLike,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    tozero: bool = False,
    negate: bool = False,
    ord: int | float | str | None = 2,  # noqa: A002
    err_msg: str = '',
    reduce_rank: bool = False,
) -> None:
    """
    Check if two matrices are similar.

    Parameters
    ----------
    actual
    desired
        The two matrices to be compared. Must be scalars, vectors, or 2d arrays.
        Scalars and vectors are intepreted as 1x1 and Nx1 matrices, but the two
        arrays must have the same shape and dtype beforehand.
    rtol
    atol
        Relative and absolute tolerances for the comparison. The closeness
        condition is:

            ||actual - desired|| <= atol + rtol * ||desired||,

        where the norm is the matrix 2-norm, i.e., the maximum (in absolute
        value) singular value.
    tozero
        If True, use the following codition instead:

            ||actual|| <= atol + rtol * ||desired||

        So `actual` is compared to zero, and `desired` is only used as a
        reference to set the threshold.
    negate
        If True, invert the inequality, replacing <= with >=. This makes the
        function check the two matrices are different instead of similar.
    ord
        Passed to `numpy.linalg.norm` to specify the matrix norm to use, the
        default is 2 which differs from numpy.
    err_msg
        Prefix prepended to the error message (without adding newlines).
    reduce_rank
        If True, reduce the input arrays to 2d by collapsing leading dimensions.

    Notes
    -----
    Boolean values are converted to uint8.
    """
    actual = np.asarray(actual)
    desired = np.asarray(desired)

    assert actual.shape == desired.shape
    assert actual.dtype == desired.dtype

    if actual.dtype == bool:
        actual = actual.astype(np.uint8)
        desired = desired.astype(np.uint8)

    if actual.size > 0:
        actual = np.atleast_1d(actual)
        desired = np.atleast_1d(desired)

        if actual.ndim > 2 and reduce_rank:
            n = actual.shape[-1]
            actual = actual.reshape(-1, n)
            desired = desired.reshape(-1, n)

        if tozero:
            expr = 'actual'
            ref = 'zero'
        else:
            expr = 'actual - desired'
            ref = 'desired'

        if negate:
            cond = 'different'
            op = ge
        else:
            cond = 'close'
            op = le

        dnorm = norm(desired, ord)
        adnorm = norm(eval(expr), ord)  # noqa: S307, expr is a literal
        ratio = adnorm / dnorm if dnorm else np.nan

        msg = f"""{err_msg}\
matrices actual and {ref} are not {cond} enough in {ord}-norm
matrix shape: {desired.shape}
norm(desired) = {dnorm:.2g}
norm({expr}) = {adnorm:.2g}  (atol = {atol:.2g})
ratio = {ratio:.2g}  (rtol = {rtol:.2g})"""

        assert op(adnorm, atol + rtol * dnorm), msg


def assert_different_matrices(*args: ArrayLike, **kwargs: Any) -> None:
    """Invoke `assert_close_matrices` with negate=True and default inf tolerance."""
    default_kwargs: dict = dict(rtol=np.inf, atol=np.inf)
    default_kwargs.update(kwargs)
    assert_close_matrices(*args, negate=True, **default_kwargs)


def assert_allclose(
    actual: ArrayLike,
    desired: ArrayLike,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    allow_non_scalar: bool = False,
    **kwargs: Any,
) -> None:
    """Wrap `numpy.testing.assert_allclose` with zero default tolerances.

    By default, both `actual` and `desired` must be scalars or 0-d arrays;
    use `assert_close_matrices` for vectors/matrices/tensors. Pass
    ``allow_non_scalar=True`` to bypass this restriction.
    """
    if not allow_non_scalar:
        actual_arr = np.asarray(actual)
        desired_arr = np.asarray(desired)
        if actual_arr.size != 1 or desired_arr.size != 1:
            msg = (
                'assert_allclose requires scalar inputs; got shapes '
                f'{actual_arr.shape} and {desired_arr.shape}. Use '
                'assert_close_matrices for vectors/matrices/tensors, or '
                'pass allow_non_scalar=True to bypass.'
            )
            raise AssertionError(msg)
    _np_assert_allclose(actual, desired, rtol=rtol, atol=atol, **kwargs)


def assert_array_equal(
    actual: ArrayLike, desired: ArrayLike, *, strict: bool = True, **kwargs: Any
) -> None:
    """Wrap `numpy.testing.assert_array_equal` with `strict=True` default."""
    _np_assert_array_equal(actual, desired, strict=strict, **kwargs)


def int_seed(rng: np.random.Generator) -> int:
    """Draw an integer random seed from a numpy random generator."""
    return int(rng.integers(0, 2**31 - 1))


@dataclass(frozen=True)
class RegressionData:
    """A small regression dataset shared by the BART-family wrapper tests."""

    x: Float64[ndarray, 'n p']
    """Predictors."""

    y: Float64[ndarray, ' n']
    """Outcomes, the first predictor plus noise."""

    x_test: Float64[ndarray, 'm p']
    """Test-set predictors."""

    @property
    def biny(self) -> Float64[ndarray, ' n']:
        """Outcomes binarized at their median, for binary-outcome fits."""
        return (self.y > np.median(self.y)).astype(float)


def regression_arrays(
    rng: np.random.Generator,
) -> tuple[Float64[ndarray, 'n p'], Float64[ndarray, ' n'], Float64[ndarray, 'm p']]:
    """Generate the ``(x, y, x_test)`` of a small regression dataset (n=30, p=3, m=7)."""
    n, p = 30, 3
    x = rng.standard_normal((n, p))
    y = x[:, 0] + 0.1 * rng.standard_normal(n)
    x_test = rng.standard_normal((7, p))
    return x, y, x_test


def evaluated_r_formals(rfuncname: str) -> dict[str, ndarray]:
    """Evaluate the argument defaults of an R function in isolation.

    Defaults that are NULL, missing, or that cannot be evaluated standalone
    (e.g. because they reference other arguments) are omitted.
    """
    # deferred to keep R optional
    from rbartpackages._src.base import robjects_r  # noqa: PLC0415

    rdefaults = robjects_r(f"""
        Filter(
            Negate(is.null),
            lapply(
                formals({rfuncname}),
                function(d) tryCatch(eval(d, baseenv()), error = function(e) NULL)
            )
        )
    """)
    return {
        name: np.asarray(value)
        for name, value in zip(rdefaults.names, rdefaults, strict=True)
    }


def mapped_params(
    obj: Callable, *, skip: set[str] = frozenset(), dots: bool = False
) -> dict[str, Parameter]:
    """Return the named parameters of `obj`, keyed by R name.

    Strips a trailing underscore (used to dodge Python keywords) from each
    name; with ``dots=True`` also maps ``_`` to ``.`` for R packages whose
    argument names use dots (the others already use Python-style underscores).
    Skips ``self``, anything in `skip`, and the ``*args``/``**kwargs``
    catch-alls, which have no R formal counterpart.
    """
    variadic = {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}

    def rname(name: str) -> str:
        name = name.removesuffix('_')
        return name.replace('_', '.') if dots else name

    return {
        rname(name): param
        for name, param in signature(obj).parameters.items()
        if name not in {'self', *skip} and param.kind not in variadic
    }


def has_var_keyword(obj: Callable) -> bool:
    """Whether `obj` has a ``**kwargs`` catch-all (forwarding R's ``...``)."""
    return any(
        param.kind is Parameter.VAR_KEYWORD
        for param in signature(obj).parameters.values()
    )
