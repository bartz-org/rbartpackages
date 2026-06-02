.. rbartpackages/docs/development.rst
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

Development
===========

Initial setup
-------------

`Fork <https://github.com/bartz-org/rbartpackages/fork>`_ the repository on
Github, then clone the fork:

.. code-block:: shell

    git clone git@github.com:YourGithubUserName/rbartpackages.git
    cd rbartpackages

Install `R <https://cran.r-project.org>`_ and `uv
<https://docs.astral.sh/uv/getting-started/installation/>`_ (for example, with
`Homebrew <https://brew.sh>`_ do :literal:`brew install r uv`). The ``bartMachine``
R package additionally needs a Java toolchain (e.g. :literal:`brew install
temurin`, then :literal:`R CMD javareconf`). Then run

.. code-block:: shell

    make setup

to set up the Python and R environments. (Note: at the time of writing, the `R
installation instructions for ubuntu
<https://cran.r-project.org/bin/linux/ubuntu>`_ miss a :code:`sudo apt install
r-base-dev` at the end.)

The Python environment is managed by uv. To run commands that involve the Python
installation, do :literal:`uv run <command>`. The R environment (managed by
`renv <https://rstudio.github.io/renv/>`_) is automatically active when you use
:literal:`R` in the project directory; the wrapped R packages are pinned in
``renv.lock`` and listed in ``DESCRIPTION``.

Contributing
------------

To contribute code changes to the main repository, create a `pull request
<https://github.com/bartz-org/rbartpackages/pulls>`_ from your fork to the main
repo.

Pre-defined commands
--------------------

Development commands are defined in a makefile. Run :literal:`make` without
arguments to list the targets. All commands that simply invoke a tool with the
right command line arguments use the :literal:`ARGS` variable to add extra
arguments, for example:

.. code-block:: shell

    make tests ARGS='-k test_gbart_fit'

Documentation
-------------

To build the documentation for the current working copy, do

.. code-block:: shell

    make docs

Building the API reference imports the wrappers, so it needs the R packages
installed. To debug the build, do :literal:`make docs SPHINXOPTS='--fresh-env
--pdb'`.

Unit tests
----------

The typical workflow to debug new changes is to first run all tests with

.. code-block:: shell

    make tests

Then, if some tests fail, use :literal:`pytest` directly to run and debug only
the relevant tests, e.g. with :literal:`uv run pytest --lf --sw --pdb`. Tests
that need an R package which is not installed are skipped rather than failing.

Debugging dependencies
----------------------

To debug tests that fail with old versions of dependencies, piggyback on the
predefined make target using :code:`ARGS`:

.. code-block:: shell

    make tests-old ARGS='-n0 -k test_gbart_fit'

Where :code:`-n0` disables test parallelization.

Benchmarks
----------

The benchmarks are managed with `asv <https://asv.readthedocs.io/en/latest>`_.
The basic asv workflow is:

.. code-block:: shell

    uv run asv run      # run and save benchmarks on main branch
    uv run asv publish  # create html report
    uv run asv preview  # start a local server to view the report

The most useful command during development is

.. code-block:: shell

    make asv-quick ARGS='--bench <pattern>'

This runs only benchmarks whose name matches <pattern>, only once, within the
working copy and current Python environment.
