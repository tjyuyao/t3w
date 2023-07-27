# `t3w.py` - Typed Thin (Py)Torch Wrapper

T3W is a lightweight framework for training PyTorch models written by Yuyao Huang during his PhD at Tongji University.
- T3W is "typed". It leverages stronger and static typing compared to normal python code for clearer architecture and less bugs. The programming model is object-oriented. Users (you) are required to implement interfaces as subclasses and inject them as dependencies.
- T3W is "thin". With the philosophy "less is more" in mind, it leverages a minimal codebase in a self-contained single python script that basically only depends on PyTorch to run. The plugin system under interface `ISideEffect` makes T3W not only thin, but also highly extensible.
- T3W stands with "PyTorch".

See the concise example [mnist_example.py](https://github.com/tjyuyao/t3w/blob/main/mnist_example.py).

If you feel like using `t3w.py`, you can either

- install it with `pip install t3w`, `pip install t3w[common]`, or `pip install t3w[all]`, where the `[common]` tag will install dependencies for commonly used side effects (like `tqdm` etc), and `[all]` tag will install dependencies for all supported side effects. Note that the mnist example requires installing `t3w[common]`.
- just download and integrate the [t3w.py](https://github.com/tjyuyao/t3w/blob/main/t3w.py) source file into your own project if you feel like freezing the version and/or ready to make some of your favorite hacks at low-level.

Detailed documentation will come in the future.