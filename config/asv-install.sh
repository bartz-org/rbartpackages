#!/bin/bash
# rbartpackages/config/asv-install.sh
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

# Helper script for ASV to install the rbartpackages wheel into an asv env.
# Usage: asv-install.sh <rbartpackages_wheel> <venv_dir>

set -e

WHEEL_FILE="$1"
ENV_DIR="$2"

if [ -z "$WHEEL_FILE" ]; then
    echo "Error: No wheel file specified" >&2
    exit 1
fi

if [ -z "$ENV_DIR" ]; then
    echo "Error: No environment directory specified" >&2
    exit 1
fi

# Install the wheel
uv pip install --python="$ENV_DIR" "${WHEEL_FILE}"
