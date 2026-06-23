# rbartpackages/docs/conf.py
#
# Copyright (c) 2024-2026, The rbartpackages Contributors
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

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import pathlib
import re
import sys
from contextlib import chdir  # ty: ignore[unresolved-import] (runs in py 3.14)
from enum import Enum
from functools import cached_property
from inspect import getsourcefile, getsourcelines, isclass, unwrap
from os import getenv
from typing import Any

import git
from docutils import nodes
from sphinx.ext.intersphinx import missing_reference

# -- Version info ------------------------------------------------------------

REPO = git.Repo(search_parent_directories=True)

COMMIT = REPO.head.commit.hexsha
UNCOMMITTED_STUFF = REPO.is_dirty()

# Check if current commit has a version tag (vX.Y.Z)
version = None
for tag in REPO.tags:
    if tag.commit == REPO.head.commit:
        MATCH = re.match(r'^v(\d+\.\d+\.\d+)$', tag.name)
        if MATCH:
            version = MATCH.group(1)
            break

if version is None:
    version = f'{COMMIT[:7]}{"+" if UNCOMMITTED_STUFF else ""}'

import rbartpackages

# rpy2 boots the embedded R on first import, and R sources `.Rprofile` (the
# renv activation) only from its startup cwd, so trigger the boot with the cwd
# pinned to the repository root: autodoc imports the wrapper modules later,
# from whatever directory sphinx-build was invoked in (`make docs` uses
# `docs/`), and without renv the R packages are missing or unpinned.
with chdir(REPO.working_tree_dir):
    import rbartpackages.base

# -- Project information -----------------------------------------------------

project = f'rbartpackages {version}'
author = 'The rbartpackages Contributors'

NOW = datetime.datetime.now(tz=datetime.timezone.utc)
YEAR = '2024'
if NOW.year > int(YEAR):
    YEAR += '-' + str(NOW.year)
copyright = YEAR + ', ' + author  # noqa: A001, because sphinx uses this variable

release = version

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',  # generate per-object pages and index tables
    'sphinx_autodoc_typehints',  # (!) keep after napoleon
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',  # link to other documentations automatically
    'myst_nb',  # markdown + jupyter notebook support
]

# WORKAROUND(sphinx-autodoc-typehints<3.10.0): on Python 3.14 Union type aliases
# have __module__='typing' and __qualname__='Union', so build_type_mapping()
# creates a bogus mapping that renders every Union as that alias.
# See https://github.com/tox-dev/sphinx-autodoc-typehints/issues/677.
if sys.version_info >= (3, 14):
    import importlib as _importlib
    import types as _types

    def _resolves_to_union_instance(dotted_path) -> bool:  # noqa: ANN001
        """Check whether *dotted_path* points at a Union instance."""
        mod_path, _, attr = dotted_path.rpartition('.')
        if not mod_path:
            return False
        try:
            obj = getattr(_importlib.import_module(mod_path), attr)
            return isinstance(obj, _types.UnionType) and not isinstance(obj, type)
        except Exception:  # noqa: BLE001
            return False

    def _remove_union_aliases_from_mapping(app, _env, _docnames) -> None:  # noqa: ANN001
        mapping = getattr(app.config, '_intersphinx_type_mapping', None)
        if mapping:
            app.config._intersphinx_type_mapping = {  # noqa: SLF001
                k: v for k, v in mapping.items() if not _resolves_to_union_instance(v)
            }


def _retry_typing_special_forms(app, env, node, contnode) -> nodes.reference | None:  # noqa: ANN001
    """Retry failed class references to typing special forms as py:obj.

    sphinx-autodoc-typehints renders e.g. ``typing.Self`` with a :py:class:
    role, but the python inventory lists the typing special forms as py:data,
    so the reference does not resolve.
    """
    if (
        node.get('refdomain') == 'py'
        and node.get('reftype') == 'class'
        and node.get('reftarget', '').startswith('typing.')
    ):
        node = node.deepcopy()
        node['reftype'] = 'obj'
        return missing_reference(app, env, node, contnode)
    else:
        return None


# manual targets for modules whose documentation does not put the top-level
# module object in its intersphinx inventory
MODULE_DOC_URLS = {
    'jax': 'https://docs.jax.dev/en/latest/jax.html',
    'polars': 'https://docs.pola.rs/api/python/stable/reference/index.html',
}


def _link_uninventoried_modules(_app, _env, node, contnode) -> nodes.reference | None:  # noqa: ANN001
    """Resolve references to the top-level modules listed in `MODULE_DOC_URLS`."""
    url = MODULE_DOC_URLS.get(node.get('reftarget'))
    if node.get('refdomain') == 'py' and url is not None:
        return nodes.reference('', '', contnode, internal=False, refuri=url)
    else:
        return None


def setup(app) -> None:  # noqa: ANN001
    if sys.version_info >= (3, 14):
        # priority 501 runs after validate_config (default 500) which populates
        # the mapping
        app.connect(
            'env-before-read-docs', _remove_union_aliases_from_mapping, priority=501
        )
    # priority 501 runs after intersphinx's own handler has given up
    app.connect('missing-reference', _retry_typing_special_forms, priority=501)
    app.connect('missing-reference', _link_uninventoried_modules, priority=501)


# decide whether to use viewcode or linkcode extension
EXT = 'viewcode'  # copy source code in static website
if getenv('RBARTPACKAGES_FORCE_LINKCODE'):
    EXT = 'linkcode'  # links to code on github
elif not UNCOMMITTED_STUFF:
    BRANCHES = REPO.git.branch('--remotes', '--contains', COMMIT)
    COMMIT_ON_GITHUB = bool(BRANCHES.strip())
    if COMMIT_ON_GITHUB:
        EXT = 'linkcode'  # links to code on github
extensions.append(f'sphinx.ext.{EXT}')

myst_enable_extensions = ['dollarmath']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'

html_title = f'{project} documentation'

html_theme_options = dict(
    description='Python wrappers of R BART packages via rpy2',
    fixed_sidebar=True,
    github_button=True,
    github_type='star',
    github_repo='rbartpackages',
    github_user='bartz-org',
    show_relbars=True,
)

# Add any paths that contain custom static files (such as style sheets) here.
html_static_path = ['_static']

master_doc = 'index'

# -- Other options -------------------------------------------------

default_role = 'py:obj'

# autodoc
autoclass_content = 'class'
# default arguments are printed as in source instead of being evaluated
autodoc_preserve_defaults = True
autodoc_default_options = {'member-order': 'bysource'}

# autosummary
# generate the per-object stub pages at build time
autosummary_generate = True
# the documented modules are the facades, which re-export their `_src`
# counterpart's public symbols and contain nothing else, so every imported
# member is a public symbol to document.
autosummary_imported_members = True

# autodoc-typehints
typehints_use_rtype = False
typehints_document_rtype = True
always_use_bars_union = True
typehints_defaults = 'comma'

# napoleon
napoleon_google_docstring = False
napoleon_use_ivar = True
napoleon_use_rtype = False
# the appendix section that RObjectBase.__init_subclass__ adds to the wrappers
napoleon_custom_sections = ['R documentation']

# intersphinx
intersphinx_mapping = dict(
    python=('https://docs.python.org/3', None),
    numpy=('https://numpy.org/doc/stable', None),
    jax=('https://docs.jax.dev/en/latest', None),
    pandas=('https://pandas.pydata.org/docs', None),
    polars=('https://docs.pola.rs/api/python/stable', None),
    rpy2=('https://rpy2.github.io/doc/latest/html', None),
)

# myst_nb
nb_execution_mode = 'off'

# viewcode
viewcode_line_numbers = True


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """
    Determine the URL corresponding to Python object, for extension linkcode.

    Adapted from scipy/doc/release/conf.py.
    """
    assert domain == 'py'

    modname = info['module']
    assert modname.startswith('rbartpackages')
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    assert submod

    obj: Any = submod
    for part in fullname.split('.'):
        if isclass(obj) and any(
            part in getattr(klass, '__annotations__', {}) for klass in obj.__mro__
        ):
            # a class data attribute (an annotation); no source line to link to
            return None
        else:
            obj = getattr(obj, part)

    if isinstance(obj, cached_property):
        obj = obj.func
    elif isinstance(obj, property):
        obj = obj.fget
    elif isinstance(obj, Enum):
        obj = type(obj)
    obj = unwrap(obj)

    try:
        fn = getsourcefile(obj)
    except TypeError:
        # C-implemented object, e.g. tuple methods inherited by a NamedTuple
        return None
    assert fn

    source, lineno = getsourcelines(obj)
    assert lineno
    linespec = f'#L{lineno}-L{lineno + len(source) - 1}'

    prefix = 'https://github.com/bartz-org/rbartpackages/blob'
    root = pathlib.Path(rbartpackages.__file__).parent
    fn_path = pathlib.Path(fn)
    if not fn_path.is_relative_to(root):
        # re-exported foreign symbol; no in-repo source to link to
        return None
    path = fn_path.relative_to(root).as_posix()
    return f'{prefix}/{COMMIT}/src/rbartpackages/{path}{linespec}'
