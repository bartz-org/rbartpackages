.. rbartpackages/docs/guide/installation.rst
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

Installation
============

``rbartpackages`` drives R packages through `rpy2 <https://rpy2.github.io>`_, so
you need both a Python and an R installation.

Python package
--------------

.. code-block:: sh

    pip install rbartpackages

(Or the equivalent for your package manager, e.g. :code:`uv add rbartpackages`.)
To install the latest development version:

.. code-block:: sh

    pip install git+https://github.com/bartz-org/rbartpackages.git

Optional extras enable the corresponding input-conversion paths: ``pandas`` and
``polars`` let you pass data frames, ``jax`` lets you pass jax arrays.

.. code-block:: sh

    pip install rbartpackages[pandas,polars,jax]

R packages
----------

Install `R <https://cran.r-project.org>`_, then install the wrapped packages you
intend to use. ``BART``, ``dbarts`` and ``bartMachine`` are on CRAN; ``BART3``
lives on GitHub:

.. code-block:: r

    install.packages(c("BART", "dbarts", "bartMachine"))
    # BART3:
    install.packages("remotes")
    remotes::install_github("rsparapa/bnptools/BART3")

``bartMachine`` is a Java package and additionally requires a working Java
toolchain and ``rJava`` (run ``R CMD javareconf`` after installing a JDK).

Importing a wrapper (e.g. ``from rbartpackages import BART3``) requires the
matching R package to be installed, because the class docstrings are pulled from
the R documentation at import time.
