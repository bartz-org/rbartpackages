.. rbartpackages/docs/guide/quickstart.rst
..
.. Copyright (c) 2024-2026, The rbartpackages Contributors
..
.. This file is part of rbartpackages.
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.

Quickstart
==========

Each wrapped R package has its own submodule. Import the one you need and call
the wrapper class like the corresponding R function; arguments are converted to
R, and the fitted R object's components become Python attributes (with ``.``
replaced by ``_``).

.. code-block:: python

    import numpy as np
    from rbartpackages import BART3

    x_train = np.random.randn(100, 5)
    y_train = x_train[:, 0] + 0.1 * np.random.randn(100)

    bart = BART3.gbart(x_train=x_train, y_train=y_train, ndpost=200)
    y_pred = bart.predict(x_train)  # shape (ndpost, n)

Argument names use Python underscores in place of R dots: pass ``x_train`` for
the R argument ``x.train``. The same pattern works for the other wrappers, e.g.
`rbartpackages.BART`, `rbartpackages.dbarts`, and `rbartpackages.bartMachine`.

Data frames and other array types
---------------------------------

With the matching extra installed, you can pass `pandas` / `polars` data frames
or `jax` arrays directly; they are converted to the appropriate R object. This
is required for `rbartpackages.bartMachine`, whose ``X`` argument must be an R
data frame.

R documentation
---------------

The original R documentation of each function is appended to the corresponding
wrapper class docstring, so :code:`help(BART3.gbart)` shows the upstream
reference.
