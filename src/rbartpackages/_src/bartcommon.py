# rbartpackages/src/rbartpackages/_src/bartcommon.py
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

"""
Private helpers shared by the `rbartpackages.BART` and `rbartpackages.BART3` wrappers.

The two packages are forks with a near-identical interface, so their wrappers
duplicate the same post-processing of R's output. The pieces that reduce to a
helper not in the public API live here; the public classes and types stay
separate in each wrapper module (sharing them would either force importing one
package's R namespace to use the other, or have sphinx document them twice).
"""

from typing import Any, cast

import numpy as np
from jaxtyping import Int32
from numpy import ndarray
from rpy2 import robjects
from rpy2.rlike.container import NamedList

from rbartpackages._src.base import RObjectBase, drop_none, namedlist_to_dict


def parse_rm_const(
    rm_const: Int32[ndarray, ' k'], ncols: int, *, removed: bool
) -> Int32[ndarray, ' <=p']:
    """
    Convert the ``rm.const`` component of a BART/BART3 object to 0-based indices.

    Parameters
    ----------
    rm_const
        The ``rm.const`` component: R reports the kept design-matrix columns as
        positive 1-based indices, or the dropped constant ones as negative
        1-based indices into the pre-removal design matrix.
    ncols
        Number of columns of the design matrix at hand, after the removal if
        any.
    removed
        Whether the constant columns were removed from that matrix, making the
        pre-removal column count ``ncols + rm_const.size`` when `rm_const` lists
        the dropped columns.

    Returns
    -------
    0-based indices of the kept columns of the pre-removal design matrix.

    Raises
    ------
    ValueError
        If the indices cannot be parsed because they change sign.
    """
    if np.all(rm_const < 0):
        # R flags the dropped constant columns as negative indices into the
        # pre-removal design matrix
        p = ncols + rm_const.size if removed else ncols
        keep = np.ones(p, bool)
        keep[-rm_const - 1] = False
        return np.arange(p, dtype=np.int32)[keep]
    elif np.all(rm_const > 0):
        return rm_const - 1
    else:  # pragma: no cover - R gives all-positive or all-negative indices
        msg = 'failed to parse rm.const because indices change sign'
        raise ValueError(msg)


def convert_treedraws(treedraws: NamedList) -> dict[str, Any]:
    """
    Convert the ``treedraws`` component of a ``gbart``-like fit to a dict.

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


def convert_gbart_predict(out: object) -> ndarray | dict[str, Any]:
    """
    Normalize the output of R's ``predict`` method for ``gbart``-like fits.

    Parameters
    ----------
    out
        The converted R output: an array of function draws (or their means) for
        continuous fits, an R named list for binary ones.

    Returns
    -------
    The array unchanged, or the list as a dict of arrays with the scalar ``binaryOffset`` unwrapped.
    """
    if not hasattr(out, 'items'):
        return cast(ndarray, out)  # continuous: a draws matrix or its column means
    else:
        # binary: convert R's list (a NamedList) to a dict of arrays
        result = namedlist_to_dict(cast(NamedList, out))
        result['binaryOffset'] = result['binaryOffset'].item()
        return result


def populate_model_matrix(
    self: RObjectBase, kw: dict[str, Any], *, removed: bool
) -> ndarray | RObjectBase:
    """
    Run ``bartModelMatrix`` in R and populate a wrapper instance from its output.

    Shared body of the BART/BART3 ``bartModelMatrix.__new__``: everything
    happens there because the ``numcut=0`` case returns a bare matrix instead of
    an instance, which ``__init__`` could not do.

    Parameters
    ----------
    self
        A freshly allocated ``bartModelMatrix`` instance to populate in place.
    kw
        The R call arguments by name, ``None`` values still to be dropped.
    removed
        Whether the ``rm.const`` argument removed the constant columns from the
        design matrix (see `parse_rm_const`).

    Returns
    -------
    The bare design matrix for ``numcut=0``, else the populated `self`.
    """
    self._robject = self._invoke_rfunc((), drop_none(kw))
    if not self._has_named_components(self._robject):
        return self._r2py(self._robject)
    self._set_attrs_from_robject()

    # grp is R NULL when absent (see the subclasses); expose it as None
    if self.grp is robjects.NULL:
        self.grp = None

    self.rm_const = parse_rm_const(self.rm_const, self.X.shape[1], removed=removed)
    return self
