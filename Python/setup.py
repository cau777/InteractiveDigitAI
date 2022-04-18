import setuptools

setuptools.setup(
    name="libs",
    version="0.0.1",
    package_dir={"": "."},
    packages=setuptools.find_packages(exclude=["tests", "client_scripts"]),
)
