# rbartpackages/Makefile
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

# Makefile for running tests, prepare and upload a release.

# define command to run python
UV_RUN = uv run --dev

# define command to run python with oldest supported dependencies
# OLD_DATE / OLD_DELAY_DAYS / BUMP_PYTHON_VERSION_DATE / NUM_SUPPORTED_PYTHON_RELEASES
# drive the `update-oldest-deps` policy.
OLD_DATE = 2025-05-27
OLD_DELAY_DAYS = 365
BUMP_PYTHON_VERSION_DATE = 10-31
NUM_SUPPORTED_PYTHON_RELEASES = 5
OLD_PYTHON = $(shell grep 'requires-python' pyproject.toml | sed 's/.*>=\([0-9.]*\).*/\1/')
UV_RUN_OLD = $(UV_RUN) --python=$(OLD_PYTHON) --resolution=lowest-direct --exclude-newer=$(OLD_DATE) --isolated

.PHONY: help
help:
	@echo "Available targets:"
	@echo "- setup: create R and Python environments for development"
	@echo "- tests: run unit tests, saving coverage information"
	@echo "- tests-old: run unit tests with oldest supported python and dependencies"
	@echo "- docs: build html documentation"
	@echo "- docs-latest: build html documentation for latest release"
	@echo "- covreport: build html coverage report"
	@echo "- covcheck: check coverage is above some thresholds"
	@echo "- diffcov: check changed-lines coverage vs DIFF_BASE (default origin/main)"
	@echo "- update-deps: remove .venv, upgrade uv.lock, update pre-commit hooks"
	@echo "- update-oldest-deps: advance OLD_DATE and refresh oldest-supported pins in pyproject.toml"
	@echo "- copy-version: sync version from pyproject.toml to _version.py"
	@echo "- check-committed: verify there are no uncommitted changes"
	@echo "- check-changelog: verify the topmost changelog section matches the current version and today's date"
	@echo "- build: build the python wheel and sdist"
	@echo "- release: run tests, build, and upload to PyPI (run on main)"
	@echo "- version-tag: create and push git tag for current version"
	@echo "- upload: upload release to PyPI"
	@echo "- upload-test: upload release to TestPyPI"
	@echo "- gh-release: create draft GitHub release from docs/changelog.md"
	@echo "- asv-machine: initialize ~/.asv-machine.json with a human-readable id"
	@echo "- asv-run: run benchmarks on all unbenchmarked tagged releases and main"
	@echo "- asv-discover: write benchmarks.json (discovery only) so asv-publish has a target"
	@echo "- asv-publish: create html benchmark report"
	@echo "- asv-preview: create html report and start server"
	@echo "- asv-main: run benchmarks on main branch"
	@echo "- asv-quick: run quick benchmarks on current code, no saving"
	@echo "- ipython: start an ipython shell with stuff pre-imported"
	@echo "- ipython-old: start an ipython shell with oldest supported python and dependencies"
	@echo "- lint: run pre-commit hooks on all files"
	@echo
	@echo "Release workflow:"
	@echo "- do a PR that re-runs benchmarks"
	@echo "- $$ uv version --bump major|minor|patch"
	@echo "- describe release in docs/changelog.md"
	@echo "- $$ make release, will not release but runs all tests, iterate and debug"
	@echo "- merge a PR with the changes"
	@echo "- on main: $$ make release"
	@echo "- merge fix PR and try again until make release passes"
	@echo "- publish the draft github release created by make release (updates zenodo automatically)"
	@echo "- if the online docs are not up-to-date, merge another PR to trigger a new merge CI"


################# SETUP #################

# bartMachine needs a working Java toolchain (rJava) at R-package install time.
.PHONY: setup
setup:
	Rscript -e "renv::restore()"
	$(UV_RUN) pre-commit install --install-hooks
	$(UV_RUN) python -c 'import rbartpackages; print(rbartpackages.__version__)'

.PHONY: lint
lint:
	$(UV_RUN) pre-commit run $(if $(ARGS),$(ARGS),--all-files)

.PHONY: clean
clean:
	rm -fr .venv
	rm -fr dist
	rm -fr docs/_build
	rm -fr .coverage* coverage.xml diffcov.md
	# `renv::clean()` only removes locks/tempdirs/unused packages, not the
	# whole library, so wipe the gitignored renv subdirs by hand to mirror
	# `rm -fr .venv`.
	rm -fr renv/library renv/staging renv/local renv/cellar renv/lock renv/python renv/sandbox

################# TESTS #################

# Test groups: each is a chunk of pytest args (paths/nodeids + -k expression)
# that selects a slice of the suite. CI runs one group per matrix cell so the
# wall time is the slowest cell; the groups keep the (slow) R fits balanced. To
# run a single group locally (composes with any tests target):
#   make tests      GROUP=bart
#   make tests-old  GROUP=others
# Leaving GROUP unset runs the whole suite. The matrix in
# `.github/workflows/tests.yml` lists these same names; keep them in sync.
GROUP_bart    := tests/test_BART.py tests/test_BART3.py tests/test_base.py tests/test_version.py
GROUP_others  := tests/test_dbarts.py tests/test_bartMachine.py

GROUPS := bart others

SELECT = $(if $(GROUP),$(GROUP_$(GROUP)))

# Number of xdist workers. Default to 2 for local speed; CI overrides to 0
# (xdist off) because the small runners OOM under parallel test execution.
NPROC ?= 2

TESTS_VARS = COVERAGE_FILE=.coverage.$@$(if $(GROUP),-$(GROUP))
TESTS_COMMAND = python -m pytest --cov --cov-context=test --dist=worksteal --durations=1000 --numprocesses=$(NPROC) $(SELECT)

.PHONY: tests
tests:
	$(TESTS_VARS) $(UV_RUN) $(TESTS_COMMAND) $(ARGS)

.PHONY: tests-old
tests-old:
	$(TESTS_VARS) $(UV_RUN_OLD) $(TESTS_COMMAND) $(ARGS)


################# DOCS #################

.PHONY: docs
docs:
	$(UV_RUN) make -C docs html
	test ! -d _site/docs-dev || rm -r _site/docs-dev
	mv docs/_build/html _site/docs-dev
	@echo
	@echo "Now open _site/index.html"

.PHONY: docs-latest
docs-latest:
	@LATEST_TAG=$$(git tag --list 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$$' | sort -V | tail -1) && \
	if [ -z "$$LATEST_TAG" ]; then echo "No release tags found, skipping docs-latest"; exit 0; fi && \
	echo "Building docs for $$LATEST_TAG" && \
	WORKTREE_DIR=$$(mktemp -d) && \
	trap "git worktree remove --force '$$WORKTREE_DIR' 2>/dev/null || rm -rf '$$WORKTREE_DIR'" EXIT && \
	git worktree add --detach "$$WORKTREE_DIR" "$$LATEST_TAG" && \
	$(MAKE) -C "$$WORKTREE_DIR" docs && \
	test ! -d _site/docs || rm -r _site/docs && \
	mv "$$WORKTREE_DIR/_site/docs-dev" _site/docs
	@echo
	@echo "Now open _site/index.html"


################# COVERAGE #################

.PHONY: covreport
covreport:
	$(UV_RUN) coverage html --include='src/*'

.PHONY: covcheck
covcheck:
	$(UV_RUN) coverage report --include='tests/**/test_*.py'
	$(UV_RUN) coverage report --include='src/*'
	$(UV_RUN) coverage report --include='tests/**/test_*.py' --fail-under=99 --format=total
	$(UV_RUN) coverage report --include='src/*' --fail-under=85 --format=total

# Branch (changed-lines) coverage: fail if new/modified lines in src and tests
# are not covered above the threshold. DIFF_BASE is the ref to diff against;
# locally a feature branch is compared to origin/main. Writes a markdown report
# (used by CI to populate the job summary) and prints the text report.
DIFF_BASE ?= origin/main
DIFFCOV_FAIL_UNDER ?= 99
DIFFCOV_REPORT ?= diffcov.md

.PHONY: diffcov
diffcov:
	# -i: the xml is only an input to diff-cover, which assesses just the
	# changed files (always present in the checkout); never fail xml generation
	# over an unrelated path missing in the combined data.
	$(UV_RUN) coverage xml -i -o coverage.xml
	$(UV_RUN) diff-cover coverage.xml --compare-branch=$(DIFF_BASE) --fail-under=$(DIFFCOV_FAIL_UNDER) --format report:- --format markdown:$(DIFFCOV_REPORT)


################# DEPENDENCIES #################

.PHONY: update-deps
update-deps:
	uv lock --upgrade
	$(UV_RUN) pre-commit autoupdate

.PHONY: update-oldest-deps
update-oldest-deps:
	$(UV_RUN) python config/update_python_version.py --bump-date=$(BUMP_PYTHON_VERSION_DATE) --num-supported=$(NUM_SUPPORTED_PYTHON_RELEASES)
	$(UV_RUN) python config/update_oldest_deps.py --min-old-date=$(OLD_DATE) --delay-days=$(OLD_DELAY_DAYS)
	uv lock


################# RELEASE #################

.PHONY: copy-version
copy-version: src/rbartpackages/_version.py
src/rbartpackages/_version.py: pyproject.toml
	$(UV_RUN) python config/util.py update_version

.PHONY: check-committed
check-committed:
	git diff --quiet
	git diff --quiet --staged

.PHONY: check-changelog
check-changelog:
	$(UV_RUN) python config/util.py check_changelog

.PHONY: build
build:
	uv build

.PHONY: release
release: check-changelog clean setup update-oldest-deps update-deps copy-version check-committed tests tests-old docs build upload gh-release
	@echo "Done!"

.PHONY: version-tag
version-tag: copy-version check-committed
	test $(shell git rev-parse --abbrev-ref HEAD) = main
	git fetch --tags
	$(eval VERSION_TAG := v$(shell uv run python -c 'import rbartpackages; print(rbartpackages.__version__)'))
	@if git rev-parse -q --verify refs/tags/$(VERSION_TAG) >/dev/null; then \
		test "$$(git rev-list -n 1 $(VERSION_TAG))" = "$$(git rev-parse HEAD)" \
			|| { echo "Tag $(VERSION_TAG) exists but points to a different commit"; exit 1; }; \
		echo "Tag $(VERSION_TAG) already exists on current commit"; \
	else \
		git tag --message=$(VERSION_TAG) $(VERSION_TAG); \
	fi
	git push origin $(VERSION_TAG)

.PHONY: smoke-test
smoke-test:
	uv run --isolated --no-project --with dist/*.whl python -c 'import rbartpackages'
	uv run --isolated --no-project --with dist/*.tar.gz python -c 'import rbartpackages'

.PHONY: upload
upload: smoke-test version-tag
	@echo "Enter PyPI token:"
	@read -s UV_PUBLISH_TOKEN && \
	export UV_PUBLISH_TOKEN && \
	uv publish
	@VERSION=$$(uv run python -c 'import rbartpackages; print(rbartpackages.__version__)') && \
	echo "Try to install rbartpackages $$VERSION from PyPI" && \
	uv tool run --exclude-newer-package="rbartpackages=0 days" --with="rbartpackages==$$VERSION" python -c 'import rbartpackages; print(rbartpackages.__version__)'

.PHONY: upload-test
upload-test: smoke-test check-committed
	@echo "Enter TestPyPI token:"
	@read -s UV_PUBLISH_TOKEN && \
	export UV_PUBLISH_TOKEN && \
	uv publish --check-url=https://test.pypi.org/simple/ --publish-url=https://test.pypi.org/legacy/
	@VERSION=$$($(UV_RUN) python config/util.py get_version) && \
	echo "Try to install rbartpackages $$VERSION from TestPyPI" && \
	uv tool run --exclude-newer-package="rbartpackages=0 days" --index=https://test.pypi.org/simple/ --index-strategy=unsafe-best-match --with="rbartpackages==$$VERSION" python -c 'import rbartpackages; print(rbartpackages.__version__)'

.PHONY: gh-release
gh-release: version-tag
	$(UV_RUN) python config/util.py gh_release


################# BENCHMARKS #################

ASV = $(UV_RUN) python -m asv

.PHONY: asv-machine
asv-machine:
	$(UV_RUN) python config/asv_machine.py

.PHONY: asv-run
asv-run: ASV_REFS = $(shell $(UV_RUN) python config/refs_for_asv.py)
asv-run: asv-machine
	$(ASV) run --durations=all --skip-existing-successful --show-stderr "$(ASV_REFS)" $(ARGS)

.PHONY: asv-discover
asv-discover: asv-machine
	# Write .asv/results/benchmarks.json (discovery only, no timing) using the
	# current env, so `asv publish` has a target even with no committed results.
	$(ASV) run --python=same --bench just-discover $(ARGS)

.PHONY: asv-publish
asv-publish:
	$(ASV) publish $(ARGS)

.PHONY: asv-preview
asv-preview: asv-publish
	$(ASV) preview $(ARGS)

.PHONY: asv-main
asv-main: asv-machine
	$(ASV) run --show-stderr main^! $(ARGS)

.PHONY: asv-quick
asv-quick: asv-machine
	$(ASV) run --durations=all --python=same --quick --dry-run --show-stderr $(ARGS)


################# IPYTHON SHELL #################

.PHONY: ipython
ipython:
	IPYTHONDIR=config/ipython $(UV_RUN) python -m IPython $(ARGS)

.PHONY: ipython-old
ipython-old:
	IPYTHONDIR=config/ipython $(UV_RUN_OLD) python -m IPython $(ARGS)
