# rbartpackages/src/rbartpackages/BART3.py
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

"""Wrapper for the R package BART3 (`rsparapa/bnptools <https://github.com/rsparapa/bnptools>`_, ``BART3`` subdirectory).

Model fitting
-------------

.. autosummary::
    :toctree:

    gbart
    mc_gbart

Data preprocessing
------------------

.. autosummary::
    :toctree:

    bartModelMatrix

Supporting types
----------------

.. autosummary::
    :toctree:

    TreeDraws
    PredictBinary
    ProcTime
    String
"""

# this facade only re-exports the public symbols of its `_src` counterpart
# ruff: noqa: F401

from rbartpackages._src.BART3 import (
    PredictBinary,
    ProcTime,
    String,
    TreeDraws,
    bartModelMatrix,
    gbart,
    mc_gbart,
)
