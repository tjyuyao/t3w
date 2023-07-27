from setuptools import setup

with open("t3w.py", 'r') as script_file:
    for line in script_file:
        if not line.startswith("__version__ = "): continue
        __version__ = eval(line[14:].rstrip()); break
    else:
        __version__ = '0.0.0'

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
        "typer[all]",
    ],
    extras_require={
        "common": [
            "aim",
            "tqdm",
        ],
        "all": [
            "aim",
            "tqdm",
        ]
    },
)