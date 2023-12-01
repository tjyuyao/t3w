"""t3w.utils.argparse submodule is inspired and modified from the project https://pypi.org/project/typed-argument-parser/, credit to the original author. """
from __future__ import annotations

import argparse
import sys
from argparse import ArgumentTypeError
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property, partial
from inspect import Parameter, signature
from typing import (
    Any,
    Callable,
    Dict,
    get_type_hints,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from docstring_parser import Docstring, parse
from typing_inspect import (
    get_args as typing_inspect_get_args,
    get_origin as typing_inspect_get_origin,
    is_literal_type,
)

PRIMITIVES = (str, int, float, bool)


def get_argument_name(*name_or_flags) -> str:
    """Gets the name of the argument.

    :param name_or_flags: Either a name or a list of option strings, e.g. foo or -f, --foo.
    :return: The name of the argument (extracted from name_or_flags).
    """
    if "-h" in name_or_flags or "--help" in name_or_flags:
        return "help"

    if len(name_or_flags) > 1:
        name_or_flags = [n_or_f for n_or_f in name_or_flags if n_or_f.startswith("--")]

    if len(name_or_flags) != 1:
        raise ValueError(
            f"There should only be a single canonical name for argument {name_or_flags}!"
        )

    return name_or_flags[0].lstrip("-")


def get_literals(
    literal: Literal, variable: str
) -> Tuple[Callable[[str], Any], List[str]]:
    """Extracts the values from a Literal type and ensures that the values are all primitive types."""
    literals = list(get_args(literal))

    if not all(isinstance(literal, PRIMITIVES) for literal in literals):
        raise ArgumentTypeError(
            f'The type for variable "{variable}" contains a literal'
            f"of a non-primitive type e.g. (str, int, float, bool).\n"
            f"Currently only primitive-typed literals are supported."
        )

    str_to_literal = {str(literal): literal for literal in literals}

    if len(literals) != len(str_to_literal):
        raise ArgumentTypeError("All literals must have unique string representations")

    def var_type(arg: str) -> Any:
        if arg not in str_to_literal:
            raise ArgumentTypeError(
                f'Value for variable "{variable}" must be one of {literals}.'
            )

        return str_to_literal[arg]

    return var_type, literals


# TODO: remove this once typing_inspect.get_origin is fixed for Python 3.8, 3.9, and 3.10
# https://github.com/ilevkivskyi/typing_inspect/issues/64
# https://github.com/ilevkivskyi/typing_inspect/issues/65
def get_origin(tp: Any) -> Any:
    """Same as typing_inspect.get_origin but fixes unparameterized generic types like Set."""
    origin = typing_inspect_get_origin(tp)

    if origin is None:
        origin = tp

    if sys.version_info >= (3, 10) and isinstance(origin, UnionType):
        origin = UnionType

    return origin


# TODO: remove this once typing_insepct.get_args is fixed for Python 3.10 union types
def get_args(tp: Any) -> Tuple[type, ...]:
    """Same as typing_inspect.get_args but fixes Python 3.10 union types."""
    if sys.version_info >= (3, 10) and isinstance(tp, UnionType):
        return tp.__args__

    return typing_inspect_get_args(tp)


def boolean_type(flag_value: str) -> bool:
    """Convert a string to a boolean if it is a prefix of 'True' or 'False' (case insensitive) or is '1' or '0'."""
    if "true".startswith(flag_value.lower()) or flag_value == "1":
        return True
    if "false".startswith(flag_value.lower()) or flag_value == "0":
        return False
    raise ArgumentTypeError(
        'Value has to be a prefix of "True" or "False" (case insensitive) or "1" or "0".'
    )


class TupleTypeEnforcer:
    """The type argument to argparse for checking and applying types to Tuples."""

    def __init__(self, types: List[type], loop: bool = False):
        self.types = [boolean_type if t == bool else t for t in types]
        self.loop = loop
        self.index = 0

    def __call__(self, arg: str) -> Any:
        arg = self.types[self.index](arg)
        self.index += 1

        if self.loop:
            self.index %= len(self.types)

        return arg


def type_to_str(type_annotation: Union[type, Any]) -> str:
    """Gets a string representation of the provided type.

    :param type_annotation: A type annotation, which is either a built-in type or a typing type.
    :return: A string representation of the type annotation.
    """
    # Built-in type
    if type(type_annotation) == type:
        return type_annotation.__name__

    # Typing type
    return str(type_annotation).replace("typing.", "")


if sys.version_info >= (3, 10):
    from types import UnionType


# Constants
EMPTY_TYPE = get_args(List)[0] if len(get_args(List)) > 0 else tuple()
BOXED_COLLECTION_TYPES = {List, list, Set, set, Tuple, tuple}
UNION_TYPES = {Union} | ({UnionType} if sys.version_info >= (3, 10) else set())
OPTIONAL_TYPES = {Optional} | UNION_TYPES
BOXED_TYPES = BOXED_COLLECTION_TYPES | OPTIONAL_TYPES

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
FunctionType = Callable[[InputType], OutputType]
ClassType = Type[OutputType]


class ArgumentParser(argparse.ArgumentParser):
    @cached_property
    def _typed_group_args(self) -> Dict[str, OutputBuilder | OutputType]:
        return dict()

    def add_options_for(
        self,
        class_or_function: Union[FunctionType, ClassType],
        dest: str = None,
        partial: bool = False,
        include_args: Optional[List[str]] = None,
        ignore_args: Optional[List[str]] = [],
        help: str = None,
        prefix: bool = False,
        **defaults,
    ):
        """adds a group of arguments based on the parsing of the signature of ``class_or_function``.

        The group of arguments can be composed into an instance of the class_or_function type / output type

        Args:
            class_or_function: The class or function to run with the provided arguments.
            dest: The attribute name used in the result namespace if not None. Defaults to None.
            partial: The result attribute is a functools.partial function to be updated. This is forced True when mandatory arguments are listed in the ``ignore_args``. Defaults to False.
            include_args: A list of argument names to be parsed from the command line. Defaults to "All".
            ignore_args: A list of argument names not to be parsed from the command line. Defaults to [].
            **defaults: Overwrites the defaults in function signature.
        """

        # Infer prefix
        if dest and (prefix is True):
            prefix = dest + "_"
        elif isinstance(prefix, str):
            prefix = prefix + "_"
        else:
            prefix = ""

        # Preprocess defaults
        if 1 == len(defaults) and "defaults" in defaults:
            defaults = defaults["defaults"]
        defaults = {(prefix + name): value for name, value in defaults.items()}

        # Create an OutputBuilder to store parsed arguments
        if dest:
            builder = OutputBuilder(class_or_function, partial)
            self._typed_group_args[dest] = builder

        # Get the group interface to add arguments
        group = self.add_argument_group(dest, help)

        # Get signature from class or function
        sig = signature(class_or_function)

        # Parse class or function docstring in one line
        if (
            isinstance(class_or_function, type)
            and class_or_function.__init__.__doc__ is not None
        ):
            doc = class_or_function.__init__.__doc__
        else:
            doc = class_or_function.__doc__

        # Parse docstring
        docstring = parse(doc)

        # Get the description of each argument in the class init or function
        param_to_description = {
            param.arg_name: param.description for param in docstring.params
        }

        is_ignored = lambda param_name: (param_name in ignore_args) or (
            isinstance(include_args, Sequence) and param_name not in include_args
        )

        # Add arguments for command line processing
        for param_name, param in sig.parameters.items():
            # Skip **kwargs
            if dest and param.kind == Parameter.VAR_KEYWORD:
                builder.pass_extra_kwargs = True
                continue

            ignored_flag = is_ignored(param_name)

            # Parse argument define
            argument_kwargs = kwargs_from_param(
                param,
                doc=param_to_description,
                class_or_function_name=class_or_function.__name__,
                ignored_flag=ignored_flag,
                defaults=defaults,
                prefix=prefix,
            )

            # Skip ignored args
            if ignored_flag:
                if argument_kwargs.get("required", False):
                    builder.partial = True
                continue

            # Add argument
            group.add_argument(f"--{prefix}{param_name}", **argument_kwargs)

            # Fill builder param names
            if dest:
                builder.keyword_arguments[param_name] = prefix + param_name

        if dest and builder.pass_extra_kwargs:
            trim_left = len(prefix)
            for prefixed_name, value in defaults.items():
                param_name = prefixed_name[trim_left:]
                if is_ignored(param_name): continue
                group.add_argument(f"--{prefixed_name}", **kwargs_from_value(value))
                builder.keyword_arguments[param_name] = prefixed_name


    def add_options(self, **kwargs):
        for argname, value in kwargs.items():
            self.add_argument(f"--{argname}", **kwargs_from_value(value))

    def parse_known_args(self, args=None, namespace=None):
        args, argv = super().parse_known_args(args=args, namespace=namespace)
        for dest, builder in self._typed_group_args.items():
            for param_name, prefixed_name in builder.keyword_arguments.items():
                builder.keyword_arguments[param_name] = getattr(args, prefixed_name)
            if not builder.partial:
                builder = builder()
            setattr(args, dest, builder)
        return args, argv


@dataclass
class OutputBuilder:
    class_or_function: Union[FunctionType, ClassType]
    partial: bool = False
    keyword_arguments: Dict = field(default_factory=dict)
    pass_extra_kwargs: bool = False

    def __call__(self, *args: Any, **kwds: Any) -> OutputType:
        return partial(self.class_or_function, **self.keyword_arguments)(*args, **kwds)

    def __repr__(self) -> str:
        params = [f"{k}={v}" for k, v in self.keyword_arguments.items()]
        if self.pass_extra_kwargs:
            params.append("**kwargs")
        params = ', '.join(params)
        return f"<partial {self.class_or_function.__name__}({params})>"


def kwargs_from_param(
    param: Parameter,
    class_or_function_name: str,
    doc: Dict[str, str] = dict(),
    defaults=dict(),
    explicit_bool=True,
    ignored_flag=False,
    prefix="",
):
    kwargs = dict()

    prefixed_name = prefix + param.name

    # Get the default or required of the argument
    if prefixed_name in defaults:
        kwargs["default"] = defaults[prefixed_name]
        del defaults[prefixed_name]
    elif param.default != Parameter.empty:
        kwargs["default"] = param.default
    else:
        kwargs["required"] = True

    # Parse type of the argument
    if not ignored_flag:
        # Get type of the argument
        if param.annotation != Parameter.empty:
            # Any type defaults to str (needed for dataclasses where all non-default attributes must have a type)
            if param.annotation is Any:
                var_type = str
            # Otherwise, get the type of the argument
            else:
                var_type = param.annotation
        # Infer type by the default value
        elif not kwargs.get("required", False):
            var_type = type(kwargs["default"])
        else:
            raise ArgumentTypeError(
                f"The parameter named '{param.name}' of '{class_or_function_name}' has not defined type annotation or default value to infer its type. Please add either to it, or add the param name to the 'ignore_args'. "
            )

        # Unbox Union[type] (Optional[type]) and set var_type = type
        if get_origin(var_type) in OPTIONAL_TYPES:
            var_args = get_args(var_type)

            # If type is Union or Optional without inner types, set type to equivalent of Optional[str]
            if len(var_args) == 0:
                var_args = (str, type(None))

            if len(var_args) > 0:
                var_type = var_args[0]

                # If var_type is tuple as in Python 3.6, change to a typing type
                # (e.g., (typing.List, <class 'bool'>) ==> typing.List[bool])
                if isinstance(var_type, tuple):
                    var_type = var_type[0][var_type[1:]]

                explicit_bool = True

        # First check whether it is a literal type or a boxed literal type
        if is_literal_type(var_type):
            var_type, kwargs["choices"] = get_literals(var_type, param.name)

        elif (
            get_origin(var_type) in (List, list, Set, set)
            and len(get_args(var_type)) > 0
            and is_literal_type(get_args(var_type)[0])
        ):
            var_type, kwargs["choices"] = get_literals(
                get_args(var_type)[0], param.name
            )
            if kwargs.get("action") not in {"append", "append_const"}:
                kwargs["nargs"] = kwargs.get("nargs", "*")

        # Handle Tuple type (with type args) by extracting types of Tuple elements and enforcing them
        elif get_origin(var_type) in (Tuple, tuple) and len(get_args(var_type)) > 0:
            loop = False
            types = get_args(var_type)

            # Handle Tuple[type, ...]
            if len(types) == 2 and types[1] == Ellipsis:
                types = types[0:1]
                loop = True
                kwargs["nargs"] = "*"
            # Handle Tuple[()]
            elif len(types) == 1 and types[0] == tuple():
                types = [str]
                loop = True
                kwargs["nargs"] = "*"
            else:
                kwargs["nargs"] = len(types)

            # Handle Literal types
            types = [
                get_literals(tp, param.name)[0] if is_literal_type(tp) else tp
                for tp in types
            ]

            var_type = TupleTypeEnforcer(types=types, loop=loop)

        if get_origin(var_type) in BOXED_TYPES:
            # If List or Set or Tuple type, set nargs
            if get_origin(var_type) in BOXED_COLLECTION_TYPES and kwargs.get(
                "action"
            ) not in {
                "append",
                "append_const",
            }:
                kwargs["nargs"] = kwargs.get("nargs", "*")

            # Extract boxed type for Optional, List, Set
            arg_types = get_args(var_type)

            # Set defaults type to str for Type and Type[()]
            if len(arg_types) == 0 or arg_types[0] == EMPTY_TYPE:
                var_type = str
            else:
                var_type = arg_types[0]

            # Handle the cases of List[bool], Set[bool], Tuple[bool]
            if var_type == bool:
                var_type = boolean_type

        # If bool then set action, otherwise set type
        if var_type == bool:
            if explicit_bool:
                kwargs["type"] = boolean_type
                kwargs["choices"] = [
                    True,
                    False,
                ]  # this makes the help message more helpful
            else:
                action_cond = (
                    "true"
                    if kwargs.get("required", False) or not kwargs["default"]
                    else "false"
                )
                kwargs["action"] = kwargs.get("action", f"store_{action_cond}")
        elif kwargs.get("action") not in {"count", "append_const"}:
            kwargs["type"] = var_type

    # Set help
    if not ignored_flag:
        kwargs["help"] = "("

        # Type
        kwargs["help"] += type_to_str(var_type) + ", "

        # Required/default
        if kwargs.get("required", False):
            kwargs["help"] += "required"
        else:
            kwargs["help"] += f'default={repr(kwargs.get("default", None))}'

        kwargs["help"] += ")"

        # Description
        docstring_help = " " + doc.get(param.name, "")
        docstring_help = docstring_help.replace("Default", "docstring default")
        docstring_help = docstring_help.replace("DEFAULT", "docstring default")
        docstring_help = docstring_help.replace("default", "docstring default")
        kwargs["help"] += docstring_help

    return kwargs


def kwargs_from_value(
    value: Any
):
    kwargs = dict()
    kwargs["type"] = type(value)
    kwargs["default"] = value

    # Set help
    if True:
        kwargs["help"] = "("

        # Type
        kwargs["help"] += type_to_str(kwargs["type"]) + ", "

        # Required/default
        if kwargs.get("required", False):
            kwargs["help"] += "required"
        else:
            kwargs["help"] += f'default={repr(value)}'

        kwargs["help"] += ")"

    return kwargs
