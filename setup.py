import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="robust_deconfounding",
    version="0.0.1",
    author="Felix Schur",
    author_email="felix.m.schur@gmail.com",
    description="DecorR (Deconfounding with Robust Regression)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={'robust_deconfounding': 'robust_deconfounding'},
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'statsmodels',
        'matplotlib',
    ],
)