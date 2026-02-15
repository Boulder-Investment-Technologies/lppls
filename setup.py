import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lppls",
    version="0.6.23",
    description="A Python module for fitting the LPPLS model to data.",
    package_dir={"": "src"},
    packages=["lppls"],
    author="Josh Nielsen",
    author_email="josh@boulderinvestment.tech",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Boulder-Investment-Technologies/lppls",
    python_requires=">=3.10",
    install_requires=[
        "cma>=3.3.0",
        "matplotlib>=3.5.0",
        "numba>=0.56.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scipy>=1.9.0",
        "tqdm>=4.64.0",
        "xarray>=2024.1.0",
        "scikit-learn>=1.2.0",
    ],
    zip_safe=False,
    include_package_data=True,
    package_data={"": ["data/*.csv"]},
)
