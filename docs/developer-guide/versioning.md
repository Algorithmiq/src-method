# Versioning scheme

Aurora follows [semantic versioning] and [PEP440] for its version numbers.
We use [setuptools-scm] to compute unambiguous version numbers for any revision
on any branch, either pushed or in local development.
We adopt the default
[convention](https://setuptools-scm.readthedocs.io/en/latest/usage/#default-versioning-scheme) in [setuptools-scm].

The version number can be obtained as `aurora.__version__` and it will look
something like `0.2.1.dev1151+g51faf286.d20231003`:

* The first part `X.Y.Z` is the _current latest tag_ with the rightmost version
  number augmented by 1. Thus, `0.2.1` is the _next_ tag that we are working
  towards, starting from the currently released tag `0.2.0`.
* The second part `.devN` is the _number of commits since the latest tag_, in
  this case 1151 commits.
* The third part `+gXXXXXXXX` is the _current commit short hash_.
* The fourth part `.dYYYYMMDD` is only added in local development, when there
  are uncommitted changes in the tree (dirty state).

[semantic versioning]: https://semver.org/
[PEP440]: https://peps.python.org/pep-0440/
[setuptools-scm]: https://setuptools-scm.readthedocs.io
