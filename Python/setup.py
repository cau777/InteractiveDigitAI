import setuptools

setuptools.setup(
    name="codebase",
    version="0.0.1",
    package_dir={"": "."},
    packages=setuptools.find_packages(exclude=["tests", "client_scripts", "admin_scripts"]),
    install_requires=[
        "numpy>=1.22.3"
    ]
)
