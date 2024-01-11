from typing import Callable
import sys

_runtime_subcommand_registry = dict()


def cmd(subcommand: Callable):

    name = subcommand.__name__

    if name in _runtime_subcommand_registry:
        raise RuntimeError(f"subcommand {name} duplicated.")

    _runtime_subcommand_registry[name] = subcommand

    return subcommand


def main():

    if len(sys.argv) > 1 and sys.argv[1] in _runtime_subcommand_registry:
        _runtime_subcommand_registry[sys.argv[1]](sys.argv[2:])
    else:
        print(f"Usage: python {sys.argv[0]} {'|'.join(list(_runtime_subcommand_registry.keys()))}")
