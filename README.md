# rbartpackages

Python wrappers of R BART (Bayesian Additive Regression Trees) packages.

`rbartpackages` lets you use a few R BART implementations (including the 3 most popular ones) from Python with a natural Python experience. The Python wrappers closely follow the R packages they wrap, so they should be obvious to use coming from R if one wants to switch to Python.

The packages included are:

- [`BART`](https://cran.r-project.org/package=BART)
- [`BART3`](https://github.com/rsparapa/bnptools) (the development superset of `BART`)
- [`bartMachine`](https://cran.r-project.org/package=bartMachine)
- [`dbarts`](https://cran.r-project.org/package=dbarts)
- [`missBART`](https://github.com/yongchengoh/missBART) (multivariate BART with non-ignorable missing responses)

The wrappers are incomplete and only wrap the main functions and objects of the packages. The most complete wrapper is `dbarts`. If you need more functionality wrapped, feel free to [open an issue on github](https://github.com/bartz-org/rbartpackages/issues). Alternatively, this library provides simplified tools and utilities to let the user wrap arbitrary R classes and functions, see [the guide](https://bartz-org.github.io/rbartpackages/docs/guide/custom-wrapper.html) and [the reference](https://bartz-org.github.io/rbartpackages/docs/reference/_autogen/mod/rbartpackages.base.html#module-rbartpackages.base).

## Installation

```sh
pip install rbartpackages
```

You also need R with the latest version of the package(s) you want to use installed (`BART`, `dbarts`, `bartMachine` from CRAN; `BART3` from `rsparapa/bnptools` and `missBART` from `yongchengoh/missBART` on GitHub). `bartMachine` additionally requires Java. If you install `polars[pyarrow]`, dataframes are returned as `polars` dataframes instead of `pandas` dataframes. For convenience, you can install both with `pip install 'rbartpackages[polars]'`. If you install `jax` (separately or with `pip install 'rbartpackages[jax]'`), jax arrays are accepted as input in place of numpy arrays, but output arrays remain numpy.

## Usage

```python
import numpy as np
from rbartpackages import BART3

x_train = np.random.randn(100, 5)
y_train = x_train[:, 0] + 0.1 * np.random.randn(100)

bart = BART3.gbart(x_train=x_train, y_train=y_train, ndpost=200)
y_pred = bart.predict(x_train)  # shape (ndpost, n)
```

R argument names with dots are passed with underscores (`x.train` → `x_train`).

## Links

- [Documentation (latest release)](https://bartz-org.github.io/rbartpackages/docs)
- [Documentation (development version)](https://bartz-org.github.io/rbartpackages/docs-dev)
- [Repository](https://github.com/bartz-org/rbartpackages)
- [List of BART packages](https://bartz-org.github.io/bartz/docs-dev/pkglist.html) (maintained in the bartz docs)
- Check out [bartz](https://github.com/bartz-org/bartz), a fast Python implementation of BART
