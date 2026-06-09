# rbartpackages/src/rbartpackages/dbarts.py
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
Wrapper for the R package dbarts (on CRAN).

Model fitting
-------------

.. autosummary::
    :toctree:

    bart
    bart2
    rbart_vi

Low-level sampler
-----------------

.. autosummary::
    :toctree:

    dbarts
    dbartsControl
    dbartsData

Supporting types
----------------

.. autosummary::
    :toctree:

    RunSamples
    String
"""

# this facade only re-exports the public symbols of its `_src` counterpart
# ruff: noqa: F401

from rbartpackages._src.dbarts import (
    RunSamples,
    String,
    bart,
    bart2,
    dbarts,
    dbartsControl,
    dbartsData,
    rbart_vi,
)
