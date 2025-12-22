# Contributing to torcheval
We want to make contributing to this project as easy and transparent as
possible.

## Development Installation
To get the development installation with all the necessary dependencies for
linting, testing, and building the documentation, run the following:
```bash
git clone https://github.com/pytorch/torcheval
cd torcheval
pip install -r requirements.txt
pip install -r audio-requirements.txt
pip install -r dev-requirements.txt
pip install -r image-requirements.txt
pip install -r docs/requirements.txt
pip install --no-build-isolation -e ".[dev]"
```

## Pull Requests
We actively welcome your pull requests.

1. Create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
    - To build docs
    ```bash
    cd docs; make html
    ```
    - To view docs
    ```bash
    cd build/html; python -m http.server
    ```
4. Ensure the test suite passes.
    - To run all tests
    ```bash
    python -m pytest tests/
    ```
    - To run a single test
    ```bash
    python -m pytest -v tests/metrics/test_metric.py::MetricBaseClassTest::test_add_state_invalid
    ```

5. Make sure your code lints.
    ```bash
    pre-commit run --all-files
    ```
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to torcheval, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
