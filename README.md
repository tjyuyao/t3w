# `t3w.py` - Typed Thin (Py)Torch Wrapper

T3W is a lightweight framework for training PyTorch models written by Yuyao Huang during his PhD at Tongji University.
- T3W is "typed". It leverages stronger and static typing compared to normal python code for clearer architecture and less bugs. The programming model is object-oriented. Users (you) are required to implement interfaces as subclasses and inject them as dependencies.
- T3W is "thin". With the philosophy "less is more" in mind, it leverages a minimal self-contained codebase that basically only depends on PyTorch to run. The plugin system under interface `ISideEffect` makes T3W not only thin, but also highly extensible.
- T3W stands with "PyTorch".

See the concise example [mnist_example.py](https://github.com/tjyuyao/t3w/blob/main/mnist_example.py).

If you feel like using `t3w.py`, you can install it with `pip install t3w`.

API documentation is currently available at https://tjyuyao.github.io/t3w/api/. We are going to add more detailed user guide in the future.
