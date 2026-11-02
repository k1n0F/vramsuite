# Releasing

Releases are built by GitHub Actions ([`.github/workflows/release.yml`](../.github/workflows/release.yml)) when a `v*` tag is pushed:

```text
tests + tag/version check  ->  build  ->  PyPI (Trusted Publishing)
                           ->  vramsuite.exe  ->  GitHub Release (exe, SHA256, notes)
```

No PyPI token is stored anywhere. PyPI trusts this exact workflow in this repository.

## One-time setup

1. **PyPI.** On pypi.org, go to *Account settings → Publishing → Add a new pending publisher* and fill in:

   ```text
   PyPI Project Name   vramsuite
   Owner               k1n0F
   Repository name     vramsuite
   Workflow name       release.yml
   Environment name    pypi
   ```

   A pending publisher does not reserve the name. The first successful upload creates the project.

2. **GitHub.** In the repository, go to *Settings → Environments → New environment* and name it `pypi`. Optionally, add yourself as a required reviewer: every PyPI upload then waits for your approval.

## Every release

1. Set the version in `pyproject.toml` (`version = "0.5.0"`) and run `uv lock`.
2. Add a `## [0.5.0] - date` section on top of `CHANGELOG.md`. It becomes the first part of the release notes, followed by [`.github/release-notes.md`](../.github/release-notes.md).
3. Commit, then tag and push:

   ```bash
   git tag v0.5.0
   git push origin v0.5.0
   ```

The workflow stops before publishing anything if the tag does not match the version in `pyproject.toml`. A version uploaded to PyPI cannot be replaced, only yanked.

## Build locally

```bash
uv build                                   # dist/*.whl and dist/*.tar.gz
uv run --with pyinstaller pyinstaller --onefile --console --name vramsuite --collect-submodules vramsuite --collect-submodules rich --copy-metadata vramsuite --noconfirm vramsuite/__main__.py
```

The sdist contains only an explicit list of files (`[tool.hatch.build.targets.sdist]` in `pyproject.toml`), so local folders and scripts never ship.
