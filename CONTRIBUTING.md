# Contributing to FLSim

We want to make contributing to FLSim as easy and transparent as possible.

## Development installation

To get the development installation with all the necessary dependencies for
linting, testing, and building the documentation, run the following:
```bash
git clone https://github.com/facebookresearch/FLSim.git
cd FLSim
pip install -e .
```

## Our Development Process

#### Code Style

FLSim soon will conform to [black](https://github.com/ambv/black) and [flake8](https://github.com/PyCQA/flake8)
code formatter to enforce a common code style across the code base. black is installed easily via
pip using `pip install black`, and run locally by calling
```bash
black .
flake8 --config ./.circleci/flake8_config.ini
```
from the repository root. No additional configuration should be needed (see the
[black documentation](https://black.readthedocs.io/en/stable/installation_and_usage.html#usage)
for advanced usage).

FLSim also soon will use [isort](https://github.com/timothycrosley/isort) to sort imports
alphabetically and separate them into sections. isort is installed easily via
pip using `pip install isort`, and run locally by calling
```bash
isort -v -l 88 -o FLSim --lines-after-imports 2 -m 3 --trailing-comma  .
```
from the repository root. Configuration for isort is located in .isort.cfg.

We feel strongly that having a consistent code style is extremely important, so
CircleCI will fail on your PR if it does not adhere to the black or flake8 formatting style or isort import ordering.

#### Type Hints

FLSim is fully typed using Python 3.6+
[type hints](https://www.python.org/dev/peps/pep-0484/).
We expect any contributions to also use proper type annotations.
While we currently do not enforce full consistency of these in our continuous integration
test, you should strive to type check your code locally. For this we recommend
using [mypy](http://mypy-lang.org/).

#### Unit Tests

To run the unit tests, you can either use `pytest` (if installed):
```bash
pytest -ra
```
or Python's `unittest`:
```bash
python -m unittest
```

To get coverage reports we recommend using the `pytest-cov` plugin:
```bash
pytest -ra --cov=. --cov-report term-missing
```

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you have added code that should be tested, add unit tests.
   In other words, always add unit tests :)
3. If you have changed APIs, document the API change in the PR.
4. Ensure the test suite passes.
5. Make sure your code passes the above [styling requirement](#code-style).



## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.


## License

By contributing to FLSim, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
