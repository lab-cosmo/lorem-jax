Create a new release of lorem-jax on PyPI. Optional argument: version bump level (`patch`, `minor`, or `major`).

## Steps

### 1. Pre-flight checks

- Work from a clean checkout of `main` at `origin/main`. If the current checkout is on another branch or has changes, do not switch it; create a worktree instead: `git fetch origin main && git worktree add <scratch>/release origin/main`, and run everything below in there. Remove the worktree at the end.
- Check that CI is passing on the latest commit of `main`: `gh run list --repo lab-cosmo/lorem-jax --branch main --limit 3`
- If CI is not green, do NOT proceed — investigate and fix first
- Check that `pyproject.toml` has no direct-URL dependencies (`@ git+...`). PyPI rejects them, and the publish job would fail after the tag is already pushed. In particular `jax-pme` must be a versioned PyPI dependency.

### 2. Run full local verification

- Run `uvx tox -e lint` — must pass
- Run `uvx tox -e tests` — must pass
- Run `uvx tox -e examples` — must pass (calculator, train-mlp, train-finetune, train-bec)
- Do NOT proceed if any of these fail

### 3. Determine version

- Find the latest git tag with `git describe --tags --abbrev=0` (or note if there are no tags yet; the first release is then `0.1.0` unless the user says otherwise)
- If a bump level was given ($ARGUMENTS), compute the new version following semver (e.g., `0.1.0` → `0.1.1` for patch, `0.2.0` for minor, `1.0.0` for major)
- If NO bump level was given, review the changes (step 4) first, then discuss with the user what the appropriate level should be based on the nature of the changes (breaking → major, new features → minor, fixes/maintenance → patch)
- Confirm the new version with the user before proceeding

### 4. Review changes and write changelog

- Run `git log <last-tag>..HEAD --oneline` to see all commits since the last release (or all commits if no prior tag)
- Write a brief changelog summarising the changes, grouped by category where appropriate (features, fixes, breaking changes, maintenance, etc.)
- Present the changelog to the user for review and approval

### 5. Tag and push

- Create an annotated tag: `git tag -a v<version> -m "Release v<version>"`
- Push the tag: `git push origin v<version>`
- This triggers `.github/workflows/release.yml`, which builds the package and publishes it to PyPI via trusted publishing from the `release` environment
- Watch it: `gh run watch` (or `gh run list --repo lab-cosmo/lorem-jax --workflow release.yml --limit 1`), and confirm the publish job succeeded before continuing

### 6. Create GitHub release

- Use `gh release create v<version> --repo lab-cosmo/lorem-jax --title "v<version>" --notes "<changelog>"` to create a GitHub release with the changelog from step 4
- Confirm the new version is visible on PyPI: `curl -s https://pypi.org/pypi/lorem-jax/json | python -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"`

## Notes

- The version is derived from git tags by `setuptools_scm` (written to `src/lorem/_version.py`, which is gitignored). No files need to be modified for a release.
- The release workflow uses PyPI trusted publishing (no tokens). The one-time setup, already done, is a `release` environment on the GitHub repo and a trusted publisher for `lorem-jax` on pypi.org pointing at `lab-cosmo/lorem-jax`, `release.yml`, environment `release`.
- Sandbox: `git fetch`, `git push`, and `gh` must run as bare commands in their own Bash call, with the shell already inside the repo (use a separate `cd` call first). Wrapping them in `cd … &&`, `$(…)`, `;` or a pipe keeps them inside the sandbox, where ssh keys and the gh token are unreachable.
