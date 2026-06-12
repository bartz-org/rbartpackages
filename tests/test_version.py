# rbartpackages/tests/test_version.py
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

"""Tests that do not need R: package metadata."""

import rbartpackages


def test_version() -> None:
    """`__version__` and `__version_info__` are consistent."""
    assert isinstance(rbartpackages.__version__, str)
    assert isinstance(rbartpackages.__version_info__, tuple)
    # hatch-vcs derives both from git, so off a release tag the version carries
    # a `.devN+g<commit>` suffix that does not survive a plain `.`-join. Only the
    # leading integer release components (before any dev/local part) are required
    # to match the start of the version string.
    release = []
    for part in rbartpackages.__version_info__:
        if not isinstance(part, int):
            break
        release.append(part)
    assert rbartpackages.__version__.startswith('.'.join(map(str, release)))
