# Writing Tests

Unit tests are an important part of writing software. Their usefulness comes in many ways:

* it allows you to make quick verification of the changes applied
* it makes sure that the code you are using is correct
* it speeds up refactoring and optimization of the code

Therefore it is important to have many tests, which helps identifying potential issues.
​
Below we present some technical aspects of testing. The list is by no means
complete and if you're interested, please check the [pytest documentation].

## General rules

There are several simple rules that should be followed in order for unit testing to serve its purpose:

1. Unit tests should test single functionality in a meaningful way,
2. All public methods should have their own tests,
3. All lines should be tested by at least one test ([code coverage]), and
4. Tests should be as fast as possible (yet meaningful)

The last point is particularly important as all tests are performed on the
continuous integration (CI) service, on which we have large, yet limited number
of computational resources available.

Furthermore, having too many slow tests prohibits effective refactoring and
optimization.
On the other hand, speed is not the main factor: if you see that increasing time
is essential for making good tests, feel free to write such a test (but mark it
as slow as explained below).
In any case, a single test should not take more than a few seconds. Ideally one
test case should run in a fraction of a second.

## Technical aspects

### Naming convention
​
All tests should be placed in the `tests` folder. `pytest` test discovery analyzes the
whole repository looking for Python scripts whose names are of
the form `test_<whatever>.py`.

???+ warning

    You should always run the test scripts as:

    ```python
    pytest test_<whatever>.py
    ```
    and _never_ as:

    ```python
    python test_<whatever>.py
    ```
    Running in "isolation" will, in many cases, _not work_ as you'd expect.
​
All test functions within the files should have names following the pattern
`test_<whatever>`. Optionally, tests can be grouped inside classes whose name
follows the pattern `Test<whatever>`.
​
### Simple tests
​
Within a test function, checks are performed using the `assert` statement:
​
```python
def test_addition():
    assert 2*2 == 4
```

The test passes if no error occurred when it was run _and_ all assertions are
satisfied.
​
If error was the expected behavior, you can use `pytest.raises`:
​
```python
import pytest
​
def test_value():
    with pytest.raises(ValueError, match=r'must be \d+$'):
        raise ValueError("value must be 42")
```

The first argument to `pytest.raises` is the exception type. While it can be
omitted, it is bad practice to do so,[^1] as the test will pass for _any_ other
exception that might be raised by the code you are trying to test.

The second argument, `match`, is useful to further _narrow down_ the exception
we are trying to test.  You should pass is a string or a regular expression
(like in the example) to match against the error message reported by the
exception. When trying to test raised standard library exceptions (like
`ValueError` or `TypeError`) it is good practice to also use the `match`
argument to `pytest.raises`.

### The `pytest` command line
​
You can alter the default behavior of the `pytest` invocation with command line
arguments. Some useful options:

1. `-x` run `pytest` until first error encountered
2. `-m "not slow"` only run tests not marked as slow.
3. `-m "slow"` only run tests marked as slow.
3. `--reruns=[m]` rerun failed tests at most `m` times (useful if there are flaky tests)
4. `--durations=[m]` show `m` slowest test durations
5. `--durations-min=[m]` minimal duration in seconds for inclusion in slowest list, defaults to 0.005
6. `-v` run in verbose mode. You can add multiple `v`s to increase the verbosity further
7. `-s` to not capture stdout. Useful if you are using `print` for debugging

### Marking tests
​
We may organize tests in our test suite more effectively
by adding _marks_ to them.
Below we present a list of marks which may be particularly useful. All the marks
should be placed right before the method.

1. `@pytest.mark.slow`: indicates that the test function will take a long time
to execute.
1. `@pytest.mark.skip("[Message]")`: skip the test. In place of `[Message]` you
should write the reason why the test is skipped. Please use this mark **only if
absolutely necessary**.
1. `@pytest.mark.parametrize("[varname]", [val1, val2, ...])`: parametrize the
test over different values of the arguments to the test function. See a simple
example below.
​
```python
import pytest
​
@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("3+5", 8),
        ("2+4", 6),
        ("6*9", 42),
    ],
)
def test_eval(test_input, expected):
    assert eval(test_input) == expected
```

### Collecting test cases with `pytest-cases`

Test parametrization is a powerful technique to avoid code repetition when
writing tests.  However, it can be hard to understand _exactly_ what and how
it's being parameterized. `pytest-cases` (see [documentation]) allows to collect the different
parametrizations into case functions/classes, which can the subsequently be
(re-)used in a parameterization.  This leads to tidier test scripts.

See this example from the `pytest-cases` documentation:

```python
from pytest_cases import parametrize_with_cases, case

class Foo:
    @case(tags=["one"], id="1")
    def case_a_positive_int(self):
        return 1

    @case(tags=["two"], id="2")
    def case_another_positive_int(self):
        return 2

@parametrize_with_cases("a", cases=Foo)
def test_foo(a):
    assert a > 0
```

In the parameterization, test cases can be filtered using tags or by globbing
function names.

???+ info

    By default, `pytest-cases` only considers function names following the pattern
    `case_<whatever>` to be valid test cases.

### Reusable testing functionality

#### Reusing data with test fixtures

Often times you find yourself writing tests that use _the same_ input data to
perform their checks. In order to reduce code duplication, the standard approach
is to use test [fixtures].
It is very easy to create fixtures with `pytest`:

```python
import pytest


@pytest.fixture
def first_entry():
    return "a"
```

and using them is a matter of adding an input argument to your test function:

```python
def test_string(first_entry):
    # Assert
    assert first_entry == "a"
```

Fixtures can be defined in the same file as the test or placed in `conftest.py`
files in the _same_ or _any parent_ folder of the test file.
`conftest.py` files are special files for `pytest`: they are "evaluated" during
test collection, such that any globally defined data is available to the test
functions when they are executed.[^2]

Fixtures can be used as arguments to other fixtures. Also, they can be _scoped_,
to decide _when_ the fixture should be executed.
By default, each fixture is function-scoped and is executed _before_ every test
function that needs it.
However, it's better to execute expensive fixtures fewer times and cache their
result.
This is achieved using the `scope` keyword argument to the the `pytest.fixture`
decorator.  The available scopes are [documented here].

#### Reusing functions with `pytest-helpers-namespace`

Other times what you want to reuse is not some _data_ produced by a function, but the function itself.
A typical use case is as follows:

* You're creating a parameterized test.
* The value of the parameter is computed through other parameters.

`pytest` offers functionality for this in terms of [parameterized fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html#parametrizing-fixtures)
and [indirect parameterization](https://docs.pytest.org/en/stable/example/parametrize.html#indirect-parametrization), however:

* The former requires one central place (the fixture definition) where to list
all possible combinations of valid parameters. When running the test using the
parameterized fixture, _all parameter combinations_ will be executed. Excluding
cases to run is cumbersome.
* The latter results in a rather unintuitive way of preparing data for a test.
Especially for those use cases where _multiple arguments_ are needed to
prepare the data for the test.

While parameterized fixture and indirect parameterization are very useful,
`pytest-helpers-namespace` offers a [lightweight alternative].
One can define _functions_ in any of the `conftest.py` files
and decorate them appropriately:

```python
import pytest


@pytest.helpers.register
def foo(bar):
    return bar
```

such that they can be reused in the test files, without having to import
`conftest.py` first:

```python
def test_helper_namespace():
    assert pytest.helpers.foo(True) is True
```

### How to use logging with tests

Within `aurora`, we adopt a `pytest` configuration that allows to see the output
from the logger when executing tests:

```toml
[tool.pytest.ini_options]
log_cli = true
log_cli_format = "%(asctime)s %(levelname)s %(message)s"
log_date_format = "%Y-%m-%d %H:%M:%S"
```

The default logging level is `logging.WARNING`.

When writing your tests, you might find useful to obtain output to screen to
check what is going on with the code you are trying to test. Traditionally, one
would reach for the `print` function, but we explicitly discourage such uses and
favour the use of logging instead.
Also, while we don't require it, you might want to test the log messages emitted
by your code, which would require _capturing_ the output from the logger within
the test function.

The `caplog` [fixture] from `pytest` comes to the rescue in both cases.
The following example shows how to set the log level for the
`aurora.chemistry.eos` submodule to `logging.INFO`:

```python
def test_foo(caplog):
    caplog.set_level(logging.INFO, logger="aurora.chemistry.eos")
```

Running the test on the command line will show logging output from the submodule.
The fixture has attributes `records` and `record_tuples`, which store the data
sent to the logger. These can be used if you want to check that specific log
messages have been emitted by your code.


[pytest documentation]: https://docs.pytest.org
[code coverage]: https://en.wikipedia.org/wiki/Code_coverage
[documentation]: https://smarie.github.io/python-pytest-cases/
[fixtures]: https://docs.pytest.org/en/stable/explanation/fixtures.html#
[^1]: luckily, Ruff will flag this bad practice.
[^2]: this is why you should _never_ think of the test scripts as usual Python scripts! Details of how `conftest.py` works can be found [here](https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files). As a corollary to this, never add the boilerplate:
```python
if __name__ == "__main__":
```
to the bottom of the test file. You can execute any test through `pytest` by simply _selecting_ it with the `-k` flag.
[documented here]: https://docs.pytest.org/en/stable/how-to/fixtures.html#fixture-scopes
[lightweight alternative]: https://pytest-helpers-namespace.readthedocs.io/en/latest/
[fixture]: https://docs.pytest.org/en/stable/how-to/logging.html
