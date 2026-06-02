# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project

rbartpackages — Python wrappers of R BART (Bayesian Additive Regression Trees) packages, built on `rpy2`. It wraps `BART`, `BART3`, `bartMachine`, and `dbarts` behind a uniform interface: arguments are converted to R, the fitted R object's components become Python attributes, and each wrapper class's docstring is augmented with the upstream R documentation (fetched at import time).

The wrappers originated inside the `bartz` project (where they validate against reference R implementations) and were extracted into this standalone package.

## Commands

All development commands are make targets. All make targets use `uv run` under the hood.

## Directory layout

We often use a worktree-first layout where the top directory is not a worktree, while each subdirectory is a worktree. When started from a subdirectory, stay there, and don't try to read/access stuff outside of the worktree.

## Workflow

To check the code you write:
- `make lint`
    - cheap to run, unleashes all linters on everything
    - don't show your work without this first!
- when changing/writing documentation for public stuff:
    - run `make docs` and check the html documentation is fine (type hints, especially return ones, break most often)
    - building the reference imports the wrappers, so it needs the R packages installed
- `make setup` if needed
    - restores the R environment (renv) and syncs the Python environment (uv)
    - `bartMachine` needs a working Java toolchain (rJava); run `R CMD javareconf` after installing a JDK
    - cheap to run (cached), idempotent; use liberally if R looks broken
- run the unit tests relevant to your changes with `uv run pytest ...`
- at the end, run the full suite with `make tests`

## Architecture

**Source layout:** `src/rbartpackages/`

| Module | Role |
|---|---|
| `_base.py` | `RObjectBase` (base class that calls an R function, converts args, and exposes the result's components as attributes) and the `rmethod` decorator; rpy2 converters for numpy/pandas/polars/jax/dict/bool |
| `BART.py` | wrappers for the R package `BART` (`gbart`, `mc_gbart`, ...) |
| `BART3.py` | wrappers for the R package `BART3` |
| `bartMachine.py` | wrapper for the R package `bartMachine` (needs Java) |
| `dbarts.py` | wrappers for the R package `dbarts` (`bart`, `bart2`, `rbart_vi`, ...) |

Importing a wrapper submodule requires the corresponding R package to be installed (the R documentation is pulled at class-definition time). The top-level `import rbartpackages` does not import the submodules, so it works without R.

The R dependencies are pinned in `renv.lock` (regenerate via renv, do not hand-edit) and declared in `DESCRIPTION`.

## Code style

- **Formatter/linter:** ruff with single quotes
- **Imports:** generally use `from foo import bar` (absolute; relative imports are banned by ruff) instead of `import foo; foo.bar`
- **Headers:** all source files carry an MIT copyright header
- **docstrings:**
    - numpy convention
    - class attributes documented individually with a string just below (not in the class docstring)
    - keep docstrings short; no redundant comments; timeless, not a narration of the development work
    - keep return value descriptions on one line (the html render garbles multi-line ones)
- **type annotations:**
    - do not stringify type annotations
    - jaxtyping for array shapes (`Float64[ndarray, 'n p']`); these annotate numpy arrays and are documentation only (not runtime-checked)
    - space before a single-axis annotation `Float64[ndarray, ' n']` because of a linter bug
    - type hints go in signatures, not docstrings; when returning multiple values, copy the hints verbatim in the return list (the html doc render needs it)
- **python conventions:**
    - use dicts as if frozen: `d = dict(d, a=1)` rather than `d['a'] = 1`
    - prefer tuples to lists; make dataclasses frozen unless mutability is needed
    - prefer `if ...: return; else: return` to early returns for readability
- _src-like layout: don't prepend redundant underscores to private functions (modules already gate the public surface)
- **WORKAROUND markers:** comments like `# WORKAROUND(jax<99): remove this patch when we bump jax to v99`, enforced by `make lint` against the oldest supported version of the package (also works with python versions and rbartpackages itself)

## Testing

- pytest, with parametrization and subtests
- global `rng` fixture provides a deterministic per-test `numpy.random.Generator` (use it directly)
    - to seed R, do `from tests.util import int_seed; int_seed(rng)`
- tests that need an R package which may be missing should use `tests.util.import_or_skip` so they skip (rather than error) when the package, R, or Java is unavailable
- to compare vectors/matrices/tensors, use `tests.util.assert_close_matrices` (and `assert_different_matrices`) instead of numpy's `assert_allclose`
    - use `rtol`, add `atol` only when comparing values near zero
- prefer the `assert_*` helpers from `tests.util` and `numpy.testing` to plain `assert` where appropriate

## Benchmarks

- in `benchmarks/`, managed by `asv`; currently a placeholder suite. Real benchmarks compare the speed of the wrapped R packages.
- test them with `make asv-quick ARGS='--bench <pattern>'`
