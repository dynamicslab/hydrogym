# Contributing Guide

Thank you for your interest in contributing to Hydrogym! There are many ways to contribute, and appreciate any and all contributions. In case you have questions please do not hesitate to open an [issue](https://github.com/dynamicslab/hydrogym/issues) to open up a discussion.

## Getting started

1. Fork the library on GitHub.
2. Clone and install the library in development mode

    ```bash
    git clone https://github.com/your-username-to-go-here/hydrogym.git
    cd hydrogym
    pip install -e .
    ```

3. Install the pre-commit hooks

    ```bash
    pip install pre-commit
    pre-commit install
    ```

which will use [Black](https://black.readthedocs.io/en/stable/) and [isort](https://github.com/PyCQA/isort) to format the code before linting it with [flake8](https://flake8.pycqa.org/en/latest/).

## Making changes to the code

If possible, please try to add isolated tests for your changes where possible, and add comments regarding the verification of results against the respective results in literature. You can then subseqently add a pull-request against the main repository to add your changes.

## Code of Conduct

We expect all participants to abide by the [Python Community Code of Conduct](https://www.python.org/psf/conduct/).
