# rbartpackages

Python wrappers of R BART (Bayesian Additive Regression Trees) packages, built on [rpy2](https://rpy2.github.io).

`rbartpackages` lets you call several R BART implementations from Python with a uniform, lightly-typed interface: arguments are converted to R, the fitted R object's components become Python attributes, and the original R documentation is attached to each wrapper class. It currently wraps:

- [`BART`](https://cran.r-project.org/package=BART)
- [`BART3`](https://github.com/rsparapa/bnptools) (the development superset of `BART`)
- [`bartMachine`](https://cran.r-project.org/package=bartMachine)
- [`dbarts`](https://cran.r-project.org/package=dbarts)
- [`missBART`](https://github.com/yongchengoh/missBART) (multivariate BART with non-ignorable missing responses)

## Installation

```sh
pip install rbartpackages
```

You also need R with the package(s) you want to use installed (`BART`, `dbarts`, `bartMachine` from CRAN; `BART3` from `rsparapa/bnptools` and `missBART` from `yongchengoh/missBART` on GitHub). `bartMachine` additionally requires Java. Optional extras `pandas`, `polars`, and `jax` enable passing those array/frame types directly. See the documentation for details.

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

These wrappers originated in the [bartz](https://github.com/bartz-org/bartz) project, where they are used to validate against reference R implementations.
