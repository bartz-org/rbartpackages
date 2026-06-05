.. rbartpackages/docs/guide/custom-wrapper.rst
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

Writing your own wrapper
========================

The built-in wrappers are thin subclasses of
`rbartpackages._base.RObjectBase`; you can wrap any R function the same way.

.. note::

    `~rbartpackages._base.RObjectBase` currently lives in the private module
    `rbartpackages._base`; it will move to a public location in a future
    release, so expect the import path to change.

First install the R package to wrap, e.g. ``BART`` from CRAN:

.. code-block:: r

    install.packages("BART")

Then subclass `~rbartpackages._base.RObjectBase`, setting ``_rfuncname`` to the
R function to call. R methods of the result are bound with
`~rbartpackages._base.rmethod`; the decorated body is discarded, so a stub
suffices:

.. code-block:: python

    from functools import partial
    from rbartpackages._base import RObjectBase, rmethod

    class gbart(RObjectBase):
        _rfuncname = 'BART::gbart'

        @partial(rmethod, rname='predict')
        def predict(self, newdata, *args, **kw): ...

That's all. Calling the class runs ``BART::gbart`` with the arguments converted
to R, and the components of the returned R object become Python attributes:

.. code-block:: python

    import numpy as np

    x = np.random.randn(100, 3)
    y = x[:, 0] + 0.1 * np.random.randn(100)

    fit = gbart(x_train=x, y_train=y, ndpost=200)
    fit.yhat_train.shape  # (200, 100)
    fit.predict(x).shape  # (200, 100)

The upstream R documentation is appended to the class docstring, so
``help(gbart)`` shows the reference of ``BART::gbart``.

The conversion is mechanical, so R idiosyncrasies show through. For example, R
has no scalars, only length-1 vectors, which convert to length-1 arrays:

.. code-block:: python

    fit.offset  # array([-0.1063...]) — not a float

The built-in wrappers smooth out these rough edges: `rbartpackages.BART.gbart`
unwraps the scalars, converts R's 1-based indices, and documents every
attribute. Prefer them when one exists.
