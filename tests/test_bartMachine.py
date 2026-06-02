# rbartpackages/tests/test_bartMachine.py
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

"""Tests for the bartMachine wrapper (needs R bartMachine + Java)."""

import numpy as np

from tests.util import import_or_skip

bartmachine = import_or_skip('rbartpackages.bartMachine')
pd = import_or_skip('pandas')

NTREE = 10
NBURN = 20
NPOST = 20


def make_data(rng: np.random.Generator, n: int = 40, p: int = 3) -> tuple:
    """Generate a small regression dataset as a pandas frame + pandas target.

    `y` is a pandas Series (not a numpy array) so it converts to a plain R
    atomic vector; numpy2ri gives even 1-D arrays a ``dim``, which bartMachine
    rejects.
    """
    x = pd.DataFrame(rng.standard_normal((n, p)), columns=[f'x{i}' for i in range(p)])
    y = x['x0'] + 0.1 * rng.standard_normal(n)
    return x, y


def test_docstring() -> None:
    """The R documentation is attached to the wrapper class."""
    assert 'R documentation' in bartmachine.bartMachine.__doc__


def test_bartmachine_fit(rng: np.random.Generator) -> None:
    """Fit `bartMachine` and check predictions have the right length."""
    x, y = make_data(rng)
    n, _ = x.shape
    bm = bartmachine.bartMachine(
        X=x,
        y=y,
        num_trees=NTREE,
        num_burn_in=NBURN,
        num_iterations_after_burn_in=NPOST,
        verbose=False,
        num_cores=1,
    )
    yhat = np.asarray(bm.predict(x))
    assert yhat.shape == (n,)
