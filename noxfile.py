import nox

# Default versions
POLARS_DEFAULT = None  # None means latest
PANDAS_DEFAULT = None  # None means latest


def _install_with_version(session, package, version):
    """Install a package with an optional specific version."""
    if version and version.lower() != "latest":
        session.install(f"{package}=={version}")
    else:
        session.install(package)


@nox.session
@nox.parametrize("polars_version", [POLARS_DEFAULT])
def tests_polars(session, polars_version):
    """Run tests with polars and all plotting libraries installed.

    Use --polars-version to specify a polars version:
        nox -s tests_polars -- --polars-version 1.2.0
    """
    # Check for --polars-version in posargs
    posargs = list(session.posargs)
    if "--polars-version" in posargs:
        idx = posargs.index("--polars-version")
        polars_version = posargs[idx + 1]
        posargs = posargs[:idx] + posargs[idx + 2 :]

    session.install(".[polars,plotly,plotnine,matplotlib]")
    session.install("pytest")
    _install_with_version(session, "polars", polars_version)
    session.run("pytest", "tests/plot", *posargs)


@nox.session
@nox.parametrize("pandas_version", [PANDAS_DEFAULT])
def tests_pandas(session, pandas_version):
    """Run tests with pandas and all plotting libraries installed.

    Use --pandas-version to specify a pandas version:
        nox -s tests_pandas -- --pandas-version 2.2.0
    """
    # Check for --pandas-version in posargs
    posargs = list(session.posargs)
    if "--pandas-version" in posargs:
        idx = posargs.index("--pandas-version")
        pandas_version = posargs[idx + 1]
        posargs = posargs[:idx] + posargs[idx + 2 :]

    session.install(".[pandas,plotly,plotnine,matplotlib]")
    session.install("pytest")
    _install_with_version(session, "pandas", pandas_version)
    session.run("pytest", "tests/plot", *posargs)


@nox.session
def tests_plotly(session):
    """Run tests with pandas and plotly installed."""
    session.install(".[pandas,polars,plotly]")
    session.install("pytest")
    session.run("pytest", "tests/plot", *session.posargs)


@nox.session
def tests_plotnine(session):
    """Run tests with pandas, polars, and plotnine installed."""
    session.install(".[pandas,polars,plotnine]")
    session.install("pytest")
    session.run("pytest", "tests/plot", *session.posargs)


@nox.session
def tests_matplotlib(session):
    """Run tests with pandas, polars, and matplotlib installed."""
    session.install(".[pandas,polars,matplotlib]")
    session.install("pytest")
    session.run("pytest", "tests/plot", *session.posargs)


@nox.session
def tests_all(session):
    """Run tests with all dependencies installed."""
    session.install(".[all]")
    session.install("pytest")
    session.run("pytest", "tests/", *session.posargs)
