# Contributing

Thanks for helping improve CPD-PyTorch. Bug reports, questions and pull requests are all
welcome on [GitHub](https://github.com/YuliangXiaoYLX/CPD-Pytorch/issues).

## Development setup

The project uses a `src/` layout and [uv](https://docs.astral.sh/uv/) (plain `pip` works
too).

```bash
git clone https://github.com/YuliangXiaoYLX/CPD-Pytorch.git
cd CPD-Pytorch
uv venv
uv pip install -e . --group dev      # or: pip install -e . --group dev  (pip >= 25.1)
uv run pre-commit install            # lint and format on every commit
```

## Checks

```bash
uv run pytest                  # tests, doctests and example scripts
uv run pytest -m "not slow"    # skip the example scripts
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run mkdocs serve            # preview the documentation at http://127.0.0.1:8000
```

The test suite compares every method with [pycpd](https://github.com/siavashk/pycpd) in
float64 (`tests/test_parity.py`). A change to the mathematics must keep these tests
passing, or explain in the pull request why the result should differ.

Tests run on every GPU available on the machine (CUDA or Apple MPS) in addition to the
CPU. Please run them on a GPU if your change touches device-specific code.

## Pull requests

- Keep each pull request focused on one change and add tests for it.
- Add an entry to `CHANGELOG.md` under "Unreleased".
- Public functions and classes need NumPy-style docstrings; they are rendered in the API
  reference.

## Documentation

The documentation is built by [Read the Docs](https://cpd-pytorch.readthedocs.io/) from
`.readthedocs.yaml`: every push to `main` updates the `latest` version and every release
tag adds a versioned copy (and updates `stable`). The Docs workflow checks the build with
`mkdocs build --strict` on every pull request.

One-time setup: sign in to <https://readthedocs.org> with GitHub, choose *Import a
Project*, select `CPD-Pytorch` and keep the project name `cpd-pytorch`. Optionally enable
*Build pull requests* in the project's settings to get a preview of the docs for each pull
request.

## Releasing

Releases are published to PyPI by GitHub Actions with
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/), so no API token is stored
anywhere.

One-time setup:

1. On PyPI, add a *pending publisher* for the project `cpd-pytorch`
   (<https://pypi.org/manage/account/publishing/>): owner `YuliangXiaoYLX`, repository
   `CPD-Pytorch`, workflow `release.yml`, environment `pypi`.
2. Optionally do the same on TestPyPI with environment `testpypi`.
3. In the GitHub repository settings, create the environments `pypi` (and `testpypi`);
   requiring a reviewer for `pypi` adds a manual approval step.

For each release:

1. Update `__version__` in `src/cpd_pytorch/_version.py`, the version in `CITATION.cff`,
   and move the "Unreleased" changelog entries under the new version.
2. Merge to `main` and wait for CI to pass.
3. Optional dry run: run the *Release* workflow manually (it uploads to TestPyPI).
4. Create a GitHub release with the tag `vX.Y.Z`. Publishing it uploads the package to
   PyPI; the workflow refuses to publish if the tag and `__version__` differ.
