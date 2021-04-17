from setuptools import setup, find_packages

def readme():
    with open("README.md") as f:
        README = f.read()
    return README


with open("requirements.txt") as f:
    required = f.read().splitlines()


setup(
    name="timetk",
    version="0.0.0.9000",
    description="TimeTK - The time series toolkit for Python.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/business-science/pytimetk",
    author="Matt Dancho",
    author_email="mdancho@business-science.io",
    copyright="Business Science",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    install_requires=required,
)