from setuptools import setup, find_packages

setup(
    name="marketsim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "tabulate"
    ],
) 