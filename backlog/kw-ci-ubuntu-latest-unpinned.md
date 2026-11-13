<a id="kw-ci-ubuntu-latest-unpinned"></a>

## KW-CI-UBUNTU-LATEST-UNPINNED — 23 workflow jobs and one wheel-smoke matrix cell run on the moving `ubuntu-latest` image [patch] [ci] — todo

priority: verification; needs: none; basis: 38eaf2f90eccd74d11ce3d0e836cbad7d4606ae0; scope: `.github/workflows/ci.yml`, `compile-timing.yml`, `python-release.yml`, `python-wheel-smoke.yml`

- **Finding.** `runs-on: ubuntu-latest` appears 21 times in `ci.yml`, once in `compile-timing.yml:21` and once in `python-release.yml:38`; the `Base wheel` matrix of `python-wheel-smoke.yml:69` also lists it. Pinned `ubuntu-24.04` today: `ci.yml:645`, `benchmark-regression.yml:61,237`, `python-wheel-smoke.yml:118`.
- **Cost already paid.** The move to 26.04 failed the k-Wave comparison in scheduled run 43368710771: 26.04 ships no package for `libhdf5_serial.so.103`. 0bb515a69 pinned that one job. In CI/CD Pipeline run 43470841463 (4a5c8db88) 20 `ubuntu-latest` jobs started on `Image: ubuntu-26.04` and the run concluded `success`; the jobs that did not start in it (SemVer, coverage, the heavy validations, the toolchain roll) and every wheel-smoke cell are unshown.
- **Acceptance:** `git grep -n 'ubuntu-latest' -- .github/workflows` returns no `runs-on` or matrix hit, or each remaining hit carries a comment naming a run that passed on the current image and why the job reads nothing the image supplies.
- **Next step:** pin the jobs that install system packages or link system libraries first, then decide the rest.
