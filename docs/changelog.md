<!--
rbartpackages/docs/changelog.md

Copyright (c) 2026, The rbartpackages Contributors

This file is part of rbartpackages.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

<!-- This changelog is written in Markdown and without hard breaks to make it
  copy-pastable to github releases. -->


# Changelog

## 0.2.0 Baruch's Algorithm for Recursive Trainspotting (2026-06-25)

- new wrapper module for `missBART`, wrapping `missBART2`
- additional wrapped objects in existing modules:
    - `bartMachine`: `bart_machine_get_posterior`, `get_sigsqs`, `bart_machine_num_cores`, `set_bart_machine_num_cores`, `Posterior`
    - `dbarts`: `dbartsData`, `RunSamples`
- explicit Python signatures and docstrings in all wrappers
- make `rbartpackages.base` public to help users write their own wrappers quickly
- the package is now fully typed

## 0.1.0 Butterjit Advanced Retrieval Trainchoke (2026-06-05)

Initial standalone release. `rbartpackages` provides Python wrappers, built on `rpy2`, of the R BART packages `BART`, `BART3`, `bartMachine`, and `dbarts`. The code was previously developed inside the bartz repository.
