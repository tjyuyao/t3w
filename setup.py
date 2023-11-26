import os
from setuptools import setup


__version__ = '0.2.0'

with open("README.md", 'r') as readme_file:
    long_description = readme_file.read()

setup(
    name="t3w",
    version=__version__,
    author="Yuyao Huang",
    author_email="huangyuyao@outlook.com",
    url="https://github.com/tjyuyao/t3w",
    description="Typed Thin PyTorch Wrapper",
    long_description=long_description,
    long_description_content_type='text/markdown',
    py_modules=["t3w"],
    install_requires=[
        "jaxtyping",
        "pandas",
        "aim",
        "tqdm"
        # "typer[all]",
    ],
    extras_require={
        # "all": [
        #     "aim",
        #     "tqdm",
        # ]
    },
)
