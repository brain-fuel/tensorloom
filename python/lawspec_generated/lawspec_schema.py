"""Typed data validation and native bridges for Python, with no test-framework
dependency.

Generated data types are described once, as Core schemas, and every value
crossing between a law and native code is validated against them, so an adapter
cannot hand a law a value outside its declared type.
ref:DEC-typed-core-boundary ref:DEC-native-bindings-typed-identity
"""

from collections import deque
from dataclasses import dataclass, replace

import lawspec_runtime as ls


@dataclass(frozen=True)
class Parameter:
    index: int


@dataclass(frozen=True)
class Named:
    name: str
    arguments: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "arguments", tuple(self.arguments))


@dataclass(frozen=True)
class Field:
    name: str
    type: object


def parse_index_term(tokens, at):
    """Parse a prefix index term: c<n>, f<field>[.<index>] or an operator."""
    token = tokens[at]
    if token[0] == "c":
        return ("c", int(token[1:])), at + 1
    if token[0] in ("f", "p"):
        positions, _, index = token[1:].partition(".")
        path = tuple(int(position) for position in positions.split("/"))
        index = int(index or 0)
        if (token[0] == "f" and len(path) != 1) or min(path) < 0 or index < 0:
            raise ValueError("malformed index field path")
        return ("f", path, index), at + 1
    if token in ("+", "-", "*", "div", "mod", "^"):
        left, at = parse_index_term(tokens, at + 1)
        right, at = parse_index_term(tokens, at)
        return (token, left, right), at
    raise ValueError("malformed index term")


def eval_index_term(term, field):
    """Natural index arithmetic; None when an operation has no value."""
    if term[0] == "c":
        return term[1]
    if term[0] == "f":
        return field(term[1], term[2])
    x = eval_index_term(term[1], field)
    y = eval_index_term(term[2], field)
    if x is None or y is None:
        return None
    if term[0] == "+":
        return x + y
    if term[0] == "-":
        return x - y if x >= y else None
    if term[0] == "*":
        return x * y
    if term[0] == "div":
        return x // y if y > 0 else None
    if term[0] == "mod":
        return x % y if y > 0 else None
    return x**y if 0 <= y <= 64 else None


@dataclass(frozen=True)
class Constructor:
    tag: str
    fields: tuple
    native: type
    predicates: tuple = ()
    native_fields: tuple | None = None
    indices: tuple = ()
    # A GADT constructor fixes parameters to patterns; its existentials are
    # the parameters numbered after the definition's own.
    refinements: tuple = ()
    existentials: int = 0
    # Existentials only a value determines (parameter numbers); their types
    # travel as the trailing Text witness fields.
    witnesses: tuple = ()
    # Texts a refined field draws besides its native ones (field index,
    # texts): samples of the regex its refinement matches it against.
    hints: tuple = ()
    native_paths: tuple | None = None

    def __post_init__(self):
        if self.native_paths is not None:
            object.__setattr__(
                self, "native_paths", tuple(tuple(path) for path in self.native_paths)
            )
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(
            self, "hints", tuple((index, tuple(texts)) for index, texts in self.hints)
        )
        object.__setattr__(self, "predicates", tuple(self.predicates))
        object.__setattr__(self, "indices", tuple(self.indices))
        object.__setattr__(
            self,
            "refinements",
            tuple((index, pattern) for index, pattern in self.refinements),
        )
        if self.native_fields is not None:
            object.__setattr__(self, "native_fields", tuple(self.native_fields))


@dataclass(frozen=True)
class Definition:
    name: str
    parameters: int
    constructors: tuple
    # A handle has no constructors: its values are adapters' own objects,
    # passed along unopened and equal only to themselves.
    handle: bool = False

    def __post_init__(self):
        object.__setattr__(self, "constructors", tuple(self.constructors))


class Maybe[T]:
    """Distinct Absent and Present states, including nested absence.

    Python's None cannot tell Absent from Present(None), so Maybe is its own
    type with nested absence kept distinct. ref:DEC-algebraic-maybe-either"""

    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        if cls is Maybe:
            raise TypeError("construct Absent or Present")
        return object.__new__(cls)


@dataclass(frozen=True, slots=True, eq=False)
class Absent[T](Maybe[T]):
    pass


@dataclass(frozen=True, slots=True, eq=False)
class Present[T](Maybe[T]):
    value: T


class Either[L, R]:
    """Base for distinct Left and Right alternatives.

    Either is a real sum rather than an exception, so a law can state what
    happens on each side. ref:DEC-algebraic-maybe-either"""

    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        if cls is Either:
            raise TypeError("construct Left or Right")
        return object.__new__(cls)


@dataclass(frozen=True, slots=True, eq=False)
class Left[L, R](Either[L, R]):
    value: L


@dataclass(frozen=True, slots=True, eq=False)
class Right[L, R](Either[L, R]):
    value: R


# Built-in collections and their single constructors. Natively a Set is a
# frozenset and a KeyVal a dict when Python compares their elements or keys by
# value, and otherwise a tuple of items or of (key, value) pairs in canonical
# order; Queue, Stack and Deque are deques (a Stack's top last, so append
# and pop use it).
_COLLECTIONS_UNIT = "lawspec.collections::type::"
COLLECTIONS = {
    _COLLECTIONS_UNIT + name: _COLLECTIONS_UNIT + name + "::" + tag
    for name, tag in (
        ("Set", "SetItems"),
        ("KeyVal", "KeyValEntries"),
        ("Queue", "QueueItems"),
        ("Stack", "StackItems"),
        ("Deque", "DequeItems"),
    )
}
_ENTRY = _COLLECTIONS_UNIT + "Entry::Entry"
_HASHABLE = frozenset(
    [
        "Bool",
        "Char",
        "Text",
        "Bytes",
        "Decimal",
        "Rational",
        "BigInt",
        "BigUInt",
        "Integer",
        "Natural",
        "CodePoint",
        "CodeUnit16",
        "Unit",
    ]
    + [
        f"{sign}Int{width}"
        for sign in ("", "U")
        for width in (8, 16, 32, 64, 128, "Size")
    ]
    + ["UIntPtr"]
)


# A Duration, whole microseconds from 0 to about 146 years, is natively a
# timedelta, which holds that range exactly.
_DURATION = "lawspec.time::type::Duration"
_DURATION_TAG = _DURATION + "::Duration"
_DURATION_LIMIT = 4611686018426999


def hashable(reference):
    """Whether Python compares this type's natives by value."""
    return (
        isinstance(reference, Named)
        and not reference.arguments
        and (reference.name in _HASHABLE)
    )


def canonical_items(items, key):
    """Sorted by key, keeping the last of equal keys."""
    from functools import cmp_to_key

    ordered = sorted(
        items, key=cmp_to_key(lambda a, b: ls.compare_values(key(a), key(b)))
    )
    result = []
    for item in ordered:
        if result and ls.compare_values(key(result[-1]), key(item)) == 0:
            result[-1] = item
        else:
            result.append(item)
    return result


def witness_key(reference):
    """Spell a type as its name, or a parenthesized application."""
    if not reference.arguments:
        return reference.name
    return (
        "("
        + " ".join([reference.name] + [witness_key(a) for a in reference.arguments])
        + ")"
    )


def parse_witness(text):
    """Read a witness key back into a type reference."""
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()

    def read(at):
        if tokens[at] == "(":
            name, at = tokens[at + 1], at + 2
            arguments = []
            while tokens[at] != ")":
                child, at = read(at)
                arguments.append(child)
            return Named(name, tuple(arguments)), at + 1
        return Named(tokens[at]), at + 1

    if not tokens:
        raise ValueError("empty type witness")
    reference, at = read(0)
    if at != len(tokens):
        raise ValueError("malformed type witness")
    return reference


def witnessed(constructor, fields):
    """Field types with witnessed existentials read from a value's fields."""
    if not constructor.witnesses:
        return [field.type for field in constructor.fields]
    known = {}
    for index, text in zip(
        constructor.witnesses, fields[-len(constructor.witnesses) :]
    ):
        if not isinstance(text, str):
            raise TypeError("type witness must be text")
        known[index] = parse_witness(text)
    arguments = [known.get(i) for i in range(max(known) + 1)]
    return [substitute(field.type, arguments) for field in constructor.fields]


# Types a generator may choose for an existential that only a value fixes.
WITNESS_POOL = (Named("Bool"), Named("Int32"))


def witness_instances(constructor):
    """Each choice of witnesses: (declared fields at it, witness texts)."""
    count = len(constructor.witnesses)
    declared = constructor.fields[: len(constructor.fields) - count]
    if not count:
        return [(declared, ())]
    from itertools import product

    found = []
    for choice in product(WITNESS_POOL, repeat=count):
        known = dict(zip(constructor.witnesses, choice))
        arguments = [known.get(i) for i in range(max(known) + 1)]
        found.append(
            (
                tuple(
                    Field(field.name, substitute(field.type, arguments))
                    for field in declared
                ),
                tuple(witness_key(item) for item in choice),
            )
        )
    return found


def refine(constructor, arguments, parameters):
    """Arguments extended with the existentials a constructor's refinements
    bind; None when the refinements do not match."""
    bound = {}

    def match(pattern, actual):
        if isinstance(pattern, Parameter):
            if pattern.index < parameters:
                return arguments[pattern.index] == actual
            previous = bound.setdefault(pattern.index, actual)
            return previous == actual
        return (
            isinstance(actual, Named)
            and actual.name == pattern.name
            and len(actual.arguments) == len(pattern.arguments)
            and all(match(p, a) for p, a in zip(pattern.arguments, actual.arguments))
        )

    for index, pattern in constructor.refinements:
        if not match(pattern, arguments[index]):
            return None
    return tuple(arguments) + tuple(
        bound.get(parameters + k) for k in range(constructor.existentials)
    )


def substitute(reference, arguments):
    if isinstance(reference, Parameter):
        # A witnessed existential stays open until a value supplies it.
        if reference.index >= len(arguments) or arguments[reference.index] is None:
            return reference
        return arguments[reference.index]
    return Named(
        reference.name,
        tuple(substitute(child, arguments) for child in reference.arguments),
    )


def type_key(reference):
    """Produce a scalar-runtime key with concrete type arguments."""
    if not isinstance(reference, Named):
        raise TypeError("type key requires an instantiated reference")
    arguments = tuple(map(type_key, reference.arguments))
    if not arguments:
        return reference.name
    if reference.name == "Either" and len(arguments) == 2:
        return f"Either ({arguments[0]}) ({arguments[1]})"
    if len(arguments) == 1:
        return reference.name + " " + arguments[0]

    def pretty(value):
        return value.name + "".join(
            " (" + pretty(child) + ")" for child in value.arguments
        )

    return pretty(reference)


class RefinementViolation(ValueError):
    """A well-formed candidate failed a Boolean field predicate."""


class Schema:
    """Validated Core metadata and native constructor classes.

    One schema serves validation, construction, matching and conversion to
    and from native classes, so these cannot disagree about a type.
    ref:DEC-native-bindings-typed-identity"""

    def __init__(self, definitions, primitives):
        self._native_codecs = {}
        self._canonical = None
        self._definitions = {}
        self._arity = dict.fromkeys(primitives, 0)
        self._arity.update(List=1, Maybe=1, Either=2, Nullable=1, Optional=1)
        for definition in definitions:
            if definition.name in self._arity:
                raise ValueError("duplicate type: " + definition.name)
            if type(definition.parameters) is not int or definition.parameters < 0:
                raise ValueError("invalid parameter count: " + definition.name)
            self._definitions[definition.name] = definition
            self._arity[definition.name] = definition.parameters
        tags, native_classes = set(), set()
        for definition in self._definitions.values():
            for constructor in definition.constructors:
                if constructor.tag in tags:
                    raise ValueError("duplicate constructor: " + constructor.tag)
                tags.add(constructor.tag)
                if definition.name in COLLECTIONS or definition.name == _DURATION:
                    continue  # Native collections and durations convert separately.
                if not isinstance(constructor.native, type):
                    raise TypeError("native constructor must be a class")
                if constructor.native in native_classes:
                    raise ValueError("duplicate native constructor class")
                native_classes.add(constructor.native)
                names = [field.name for field in constructor.fields]
                if len(names) != len(set(names)):
                    raise ValueError("duplicate constructor field")
                if constructor.native_fields is not None:
                    native_names = constructor.native_fields
                    if (
                        len(native_names)
                        != (
                            len(names)
                            if constructor.native_paths is None
                            else len(constructor.native_paths)
                        )
                        or len(set(native_names)) != len(native_names)
                        or not all(
                            isinstance(name, str) and name.isidentifier()
                            for name in native_names
                        )
                    ):
                        raise ValueError("invalid native field mapping")
                scope = definition.parameters + constructor.existentials
                for field in constructor.fields:
                    self._check(field.type, scope)
                for index, pattern in constructor.refinements:
                    if not 0 <= index < definition.parameters:
                        raise ValueError("invalid refined parameter")
                    self._check(pattern, scope)
                if not all(callable(test) for test in constructor.predicates):
                    raise TypeError("constructor predicate must be callable")

        # Paths are checked against the entire product, including generic
        # child types, before an application value can cross this schema.
        for definition in self._definitions.values():
            for constructor in definition.constructors:
                if constructor.native_paths is not None:
                    if constructor.native_fields is None:
                        raise ValueError("native paths require native fields")
                    self._native_layout(constructor.fields, constructor.native_paths)

    def _native_layout(self, fields, paths):
        """A complete, nonoverlapping packing tree; leaves keep native order."""

        def build(fields, entries):
            if any(not path for _, path in entries) or set(
                path[0] for _, path in entries
            ) != set(field.name for field in fields):
                raise ValueError("native paths must cover exactly every field")
            nodes = []
            for field in fields:
                children = [
                    (index, path[1:])
                    for index, path in entries
                    if path[0] == field.name
                ]
                direct = [index for index, path in children if not path]
                if direct:
                    if len(children) != 1:
                        raise ValueError("overlapping native paths")
                    nodes.append((field, direct[0]))
                    continue
                definition = self._definitions.get(getattr(field.type, "name", None))
                if definition is None or len(definition.constructors) != 1:
                    raise ValueError("native path must enter a record")
                constructor = definition.constructors[0]
                if constructor.existentials or constructor.refinements:
                    raise ValueError("indexed native record path is unsupported")
                instantiated = tuple(
                    Field(child.name, substitute(child.type, field.type.arguments))
                    for child in constructor.fields
                )
                nodes.append((field, (constructor.tag, build(instantiated, children))))
            return tuple(nodes)

        return build(fields, tuple(enumerate(paths)))

    def _layout_walk(self, constructor, fields, bits, mode, symbols):
        tree = self._native_layout(constructor.fields, constructor.native_paths)
        leaves = [None] * len(constructor.native_paths)

        def gather(nodes, values):
            for index, (field, child) in enumerate(nodes):
                if isinstance(child, int):
                    leaves[child] = (field.type, values[index])
                else:
                    tag, nested = child
                    value = values[index]
                    if (
                        not isinstance(value, ls.DataValue)
                        or value.tag != tag
                        or len(value.fields) != len(nested)
                    ):
                        raise ValueError("invalid native record payload")
                    gather(nested, value.fields)

        def types(nodes):
            for field, child in nodes:
                if isinstance(child, int):
                    leaves[child] = field.type
                else:
                    types(child[1])

        def pack(nodes, values):
            return tuple(
                values[child]
                if isinstance(child, int)
                else ls.DataValue(child[0], pack(child[1], values))
                for _, child in nodes
            )

        if mode == "native":
            gather(tree, fields)
            converted = [
                self._walk(ty, value, bits, mode, symbols) for ty, value in leaves
            ]
            return constructor.native(**dict(zip(constructor.native_fields, converted)))
        types(tree)
        converted = [
            self._walk(ty, value, bits, mode, symbols)
            for ty, value in zip(leaves, fields)
        ]
        # from_native performs the ordinary recursive validation afterwards,
        # checking both the packed record and its enclosing constructor.
        return ls.DataValue(constructor.tag, pack(tree, converted))

    def _check(self, reference, parameters=0):
        if isinstance(reference, Parameter):
            if (
                type(reference.index) is not int
                or not 0 <= reference.index < parameters
            ):
                raise ValueError("unbound schema parameter")
        elif isinstance(reference, Named):
            if self._arity.get(reference.name) != len(reference.arguments):
                raise ValueError("unknown type or wrong arity: " + reference.name)
            for child in reference.arguments:
                self._check(child, parameters)
        else:
            raise TypeError("invalid schema type reference")

    def constructors(self, reference):
        """Return instantiated fields, or None for a built-in type."""
        self._check(reference)
        definition = self._definitions.get(reference.name)
        if definition is None:
            return None
        found = []
        for constructor in definition.constructors:
            # A GADT constructor whose refinements do not match these
            # arguments builds no value of this type.
            arguments = refine(constructor, reference.arguments, definition.parameters)
            if arguments is None:
                continue
            found.append(
                Constructor(
                    constructor.tag,
                    tuple(
                        Field(field.name, substitute(field.type, arguments))
                        for field in constructor.fields
                    ),
                    constructor.native,
                    constructor.predicates,
                    constructor.native_fields,
                    constructor.indices,
                    witnesses=constructor.witnesses,
                    native_paths=constructor.native_paths,
                )
            )
        return tuple(found)

    def is_handle(self, reference):
        """Whether this checked type has opaque native identity.

        Embedding codecs must distinguish a handle from a scalar or collection
        even when its application representation uses the same Python class.
        ref:DEC-prolog-janus
        """
        self._check(reference)
        definition = self._definitions.get(reference.name)
        return definition is not None and definition.handle

    def recorded_text(self, reference, value):
        """Read field types without rerunning predicates or native codecs.

        ref:REQ-law-primitives ref:DEC-typed-core-boundary
        """

        def fields(t, item):
            constructor = next(
                (
                    candidate
                    for candidate in self.constructors(t) or ()
                    if candidate.tag == item.tag
                ),
                None,
            )
            if constructor is None or len(constructor.fields) != len(item.fields):
                raise TypeError("invalid recorded constructor")
            return witnessed(constructor, item.fields)

        return ls.recorded_text(value, reference, fields)

    def recorded(self, reference, key, value, bits=64, symbols=None):
        return ls.recorded(key, value, lambda item: self.recorded_text(reference, item))

    def with_native_bindings(self, bindings, codecs=None):
        """Copy this schema with application classes and field names.

        Keys are resolved constructor tags. Values pair a class with its
        field names in LawSpec declaration order. Predicates retain their
        logical fields and run against the same checked values. Optional codec
        entries map a type identity to (native class, to_native, from_native).
        Hooks use canonical values and receive directional child converters.
        """
        bindings = dict(bindings)
        hooks = dict(self._native_codecs)
        hooks.update({} if codecs is None else codecs)
        for name, hook in hooks.items():
            definition = self._definitions.get(name)
            if definition is None or not definition.constructors:
                raise ValueError("unknown or empty native codec type: " + name)
            if (
                not isinstance(hook, tuple)
                or len(hook) != 3
                or not isinstance(hook[0], type)
                or not all(callable(part) for part in hook[1:])
            ):
                raise TypeError("invalid native codec: " + name)
            if any(item.tag in bindings for item in definition.constructors):
                raise ValueError("codec conflicts with native mapping: " + name)
        definitions = []
        for definition in self._definitions.values():
            constructors = []
            for constructor in definition.constructors:
                mapped = bindings.pop(constructor.tag, None)
                if mapped is not None:
                    if len(mapped) not in (2, 3):
                        raise ValueError("invalid native constructor mapping")
                    native, fields = mapped[:2]
                    paths = mapped[2] if len(mapped) == 3 else None
                    constructor = replace(
                        constructor,
                        native=native,
                        native_fields=fields,
                        native_paths=paths,
                    )
                constructors.append(constructor)
            definitions.append(replace(definition, constructors=tuple(constructors)))
        if bindings:
            raise ValueError("unknown native constructor binding")
        primitives = set(self._arity) - set(self._definitions)
        primitives -= {"List", "Maybe", "Either", "Nullable", "Optional"}
        result = Schema(definitions, primitives)
        result._canonical = self._canonical or self
        result._native_codecs = hooks
        return result

    @staticmethod
    def _bits(bits):
        if type(bits) is not int or bits not in (32, 64):
            raise ValueError("machineBits must be 32 or 64")

    def _check_indices(self, constructor, fields):
        """Check an indexed family's guards against its fields' indices."""
        for text in constructor.indices:
            tokens = text.split()
            if tokens[0] not in ("==", ">="):
                continue
            left, at = parse_index_term(tokens, 1)
            right, _ = parse_index_term(tokens, at)

            def field(path, index):
                return self._index_at_path(constructor.fields, fields, path, index)

            x = eval_index_term(left, field)
            y = eval_index_term(right, field)
            if x is None or y is None or (x != y if tokens[0] == "==" else x < y):
                raise RefinementViolation(
                    f"{constructor.tag}: index guard {text} failed"
                )

    def _index_of(self, reference, value, index):
        constructors = self.constructors(reference)
        constructor = next(
            (item for item in constructors or () if item.tag == value.tag), None
        )
        terms = [
            text
            for text in (constructor.indices if constructor else ())
            if text.split()[0] not in ("==", ">=")
        ]
        if index >= len(terms):
            raise ValueError("no index for " + reference.name)
        term, _ = parse_index_term(terms[index].split(), 0)
        return eval_index_term(
            term,
            lambda path, child: self._index_at_path(
                constructor.fields, value.fields, path, child
            ),
        )

    def _index_at_path(self, types, values, path, index):
        if not path or path[0] < 0 or path[0] >= len(types) or path[0] >= len(values):
            raise ValueError("invalid index field path")
        position, *rest = path
        reference, value = types[position].type, values[position]
        if not rest:
            return self._index_of(reference, value, index)
        if not getattr(reference, "name", "").startswith(
            "lawspec::structural::record::"
        ):
            raise ValueError("index field path must traverse a structural record")
        constructors = self.constructors(reference) or ()
        if len(constructors) != 1 or constructors[0].tag != value.tag:
            raise ValueError("invalid structural record in index field path")
        return self._index_at_path(constructors[0].fields, value.fields, rest, index)

    def validate(self, reference, value, bits=64, symbols=None):
        """Rejects a value outside the declared type, refinements included,
        because native code may return anything its own type allows.
        ref:DEC-portable-exact-arithmetic"""
        self._check(reference)
        self._bits(bits)
        return self._walk(
            reference, value, bits, "validate", {} if symbols is None else symbols
        )

    def all_payloads(self, reference, value, predicates, bits=64, symbols=None):
        """Check stored arguments, preserving parameter provenance."""
        self._check(reference)
        structural = ("List", "Maybe", "Either", "Nullable", "Optional")
        if reference.name not in self._definitions and reference.name not in structural:
            raise ValueError("payload predicates require a data type")
        predicates = tuple(predicates)
        if len(predicates) != len(reference.arguments):
            raise ValueError("payload predicate arity mismatch")
        if not all(callable(predicate) for predicate in predicates):
            raise TypeError("payload predicate must be callable")
        checked = self.validate(reference, value, bits, symbols)

        def recipe(field, arguments):
            if isinstance(field, Parameter):
                return arguments[field.index]
            children = tuple(recipe(child, arguments) for child in field.arguments)
            return (
                Named(field.name, children)
                if any(child is not None for child in children)
                else None
            )

        def contextual(plan, child, context):
            try:
                return walk(plan, child)
            except (ValueError, TypeError, ArithmeticError) as error:
                raise ValueError(f"{context}: {error}") from error

        def walk(plan, child):
            if plan is None:
                return True
            if isinstance(plan, Parameter):
                result = predicates[plan.index](child)
                if type(result) is not bool:
                    raise TypeError("payload predicate did not produce Bool")
                return result
            name, arguments = plan.name, plan.arguments
            if name == "List":
                return all(
                    contextual(arguments[0], item, f"List[{index}]")
                    for index, item in enumerate(child)
                )
            if name in ("Nullable", "Optional"):
                return not child.present or contextual(
                    arguments[0], child.value, name + ".value"
                )
            if name == "Maybe":
                return child.tag == "Maybe::Absent" or contextual(
                    arguments[0], child.fields[0], "Maybe::Present.value"
                )
            if name == "Either":
                index = 0 if child.tag == "Either::Left" else 1
                return contextual(
                    arguments[index], child.fields[0], child.tag + ".value"
                )
            definition = self._definitions[name]
            constructor = next(
                item for item in definition.constructors if item.tag == child.tag
            )
            return all(
                contextual(
                    recipe(field.type, arguments),
                    item,
                    constructor.tag + "." + field.name,
                )
                for field, item in zip(constructor.fields, child.fields)
            )

        plan = Named(
            reference.name, tuple(Parameter(index) for index in range(len(predicates)))
        )
        return walk(plan, checked)

    def to_native(self, reference, value, bits=64, symbols=None):
        """Converts a checked value into the class the adapter declared, so
        adapters work with their own types and never with LawSpec's.
        ref:DEC-native-bindings-typed-identity"""
        symbols = {} if symbols is None else symbols
        checked = self.validate(reference, value, bits, symbols)
        return self._walk(reference, checked, bits, "native", symbols)

    def from_native(self, reference, value, bits=64, symbols=None):
        """Converts and checks a value an adapter returned, so a law never sees
        a native value that is outside its domain.
        ref:DEC-native-bindings-typed-identity"""
        self._check(reference)
        self._bits(bits)
        symbols = {} if symbols is None else symbols
        logical = self._walk(reference, value, bits, "logical", symbols)
        return self.validate(reference, logical, bits, symbols)

    def _codec_walk(self, reference, value, bits, mode, symbols):
        native, to_native, from_native = self._native_codecs[reference.name]
        canonical = self._canonical
        direction = "toNative" if mode == "native" else "fromNative"

        def converter(argument):
            if mode == "native":
                return lambda child: self.to_native(
                    argument,
                    canonical.from_native(argument, child, bits, symbols),
                    bits,
                    symbols,
                )
            return lambda child: canonical.to_native(
                argument,
                self.from_native(argument, child, bits, symbols),
                bits,
                symbols,
            )

        try:
            children = tuple(map(converter, reference.arguments))
            if mode == "native":
                logical = canonical.to_native(reference, value, bits, symbols)
                result = to_native(logical, *children)
                if not isinstance(result, native):
                    raise TypeError("hook returned the wrong native type")
                return result
            if not isinstance(value, native):
                raise TypeError("expected the bound native type")
            result = from_native(value, *children)
            return canonical.from_native(reference, result, bits, symbols)
        except Exception as error:
            raise ValueError(
                f"native codec {reference.name} {direction}: {error}"
            ) from error

    def _collection_walk(self, reference, value, bits, mode, symbols):
        short = reference.name[len(_COLLECTIONS_UNIT) :]
        arguments = reference.arguments
        tag = COLLECTIONS[reference.name]

        def walk(ty, child):
            return self._walk(ty, child, bits, mode, symbols)

        if mode == "native":
            if not isinstance(value, ls.DataValue) or value.tag != tag:
                raise TypeError("expected " + short)
            items = value.fields[0]
            if short == "KeyVal":
                pairs = [
                    (
                        walk(arguments[0], entry.fields[0]),
                        walk(arguments[1], entry.fields[1]),
                    )
                    for entry in items
                ]
                return dict(pairs) if hashable(arguments[0]) else tuple(pairs)
            natives = [walk(arguments[0], item) for item in items]
            if short == "Set":
                return frozenset(natives) if hashable(arguments[0]) else tuple(natives)
            return deque(reversed(natives) if short == "Stack" else natives)
        if short == "KeyVal":
            if isinstance(value, dict):
                pairs = list(value.items())
            elif isinstance(value, (tuple, list)):
                pairs = list(value)
            else:
                raise TypeError("expected a dict or (key, value) pairs")
            entries = []
            for pair in pairs:
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise TypeError("expected (key, value) pairs")
                entries.append(
                    ls.DataValue(
                        _ENTRY,
                        (walk(arguments[0], pair[0]), walk(arguments[1], pair[1])),
                    )
                )
            items = canonical_items(entries, lambda entry: entry.fields[0])
        else:
            if not isinstance(value, (frozenset, set, tuple, list, deque)):
                raise TypeError("expected a collection for " + short)
            items = [walk(arguments[0], item) for item in value]
            if short == "Set":
                items = canonical_items(items, lambda item: item)
            elif short == "Stack":
                items.reverse()
        return ls.DataValue(tag, (items,))

    def _walk(self, reference, value, bits, mode, symbols):
        definition = self._definitions.get(reference.name)
        if definition is not None and definition.handle:
            return ls.handle(value, reference.name)
        if mode in ("native", "logical") and reference.name in self._native_codecs:
            return self._codec_walk(reference, value, bits, mode, symbols)
        if mode in ("native", "logical") and reference.name in COLLECTIONS:
            return self._collection_walk(reference, value, bits, mode, symbols)
        if mode in ("native", "logical") and reference.name == _DURATION:
            return _duration_walk(value, mode)
        constructors = self.constructors(reference)
        if constructors is not None:
            if mode == "logical":
                constructor = next(
                    (item for item in constructors if type(value) is item.native), None
                )
                if constructor is None:
                    raise TypeError("invalid native " + reference.name)
                names = (
                    constructor.native_fields
                    if constructor.native_fields is not None
                    else tuple(field.name for field in constructor.fields)
                )
                fields = tuple(getattr(value, name) for name in names)
            else:
                if not isinstance(value, ls.DataValue):
                    raise TypeError("expected " + reference.name)
                constructor = next(
                    (item for item in constructors if value.tag == item.tag), None
                )
                if constructor is None:
                    raise ValueError("foreign constructor for " + reference.name)
                fields = value.fields
            if constructor.native_paths is not None and mode in ("native", "logical"):
                return self._layout_walk(constructor, fields, bits, mode, symbols)
            if len(fields) != len(constructor.fields):
                raise ValueError("wrong field count: " + constructor.tag)
            converted = []
            types = witnessed(constructor, fields)
            # A shallow check trusts fields, which were checked when built.
            for field, field_type, child in zip(constructor.fields, types, fields):
                if mode == "shallow":
                    converted.append(child)
                    continue
                try:
                    if field_type != field.type:
                        self._check(field_type)
                    converted.append(self._walk(field_type, child, bits, mode, symbols))
                except (ValueError, TypeError, OverflowError) as error:
                    error_type = (
                        RefinementViolation
                        if isinstance(error, RefinementViolation)
                        else ValueError
                    )
                    raise error_type(
                        constructor.tag + "." + field.name + ": " + str(error)
                    ) from error
            if mode == "native":
                if constructor.native_fields is not None:
                    return constructor.native(
                        **dict(zip(constructor.native_fields, converted))
                    )
                return constructor.native(*converted)
            if mode in ("validate", "shallow"):
                # Check shapes first. Failed conditions stop before
                # predicates whose definedness depends on them.
                for index, predicate in enumerate(constructor.predicates):
                    context = f"{constructor.tag}: field refinement {index + 1}"
                    try:
                        accepted = predicate(
                            self, reference.arguments, tuple(converted), bits, symbols
                        )
                    except (ValueError, TypeError, ArithmeticError) as error:
                        raise ValueError(f"{context}: {error}") from error
                    if accepted is False:
                        raise RefinementViolation(f"{context} failed")
                    if accepted is not True:
                        raise ValueError(f"{context} did not produce Bool")
                self._check_indices(constructor, converted)
                if (
                    reference.name == "lawspec.collections::type::EqSet"
                    and not ls.distinct_members(
                        converted[0],
                        lambda a, b: self.equal(
                            reference.arguments[0], a, b, bits, symbols
                        ),
                    )
                ):
                    raise RefinementViolation("Eq-only set: duplicate equal elements")
            return ls.DataValue(constructor.tag, converted)
        name, arguments = reference.name, reference.arguments
        if not arguments:
            return ls.validate(value, name, bits)
        if name == "List":
            if not isinstance(value, list):
                raise TypeError("expected List")
            result = []
            for index, child in enumerate(value):
                try:
                    result.append(self._walk(arguments[0], child, bits, mode, symbols))
                except (ValueError, TypeError, OverflowError) as error:
                    error_type = (
                        RefinementViolation
                        if isinstance(error, RefinementViolation)
                        else ValueError
                    )
                    raise error_type(f"List[{index}]: {error}") from error
            return result
        if name in ("Nullable", "Optional"):
            if (
                not isinstance(value, ls.Presence)
                or value.kind != name
                or type(value.present) is not bool
            ):
                raise TypeError("invalid tagged presence: " + name)
            payload = None
            if value.present:
                payload = self._walk(arguments[0], value.value, bits, mode, symbols)
            return ls.Presence(name, value.present, payload)
        if name in ("Maybe", "Either"):
            classes = (Absent, Present) if name == "Maybe" else (Left, Right)
            tags = (
                ("Maybe::Absent", "Maybe::Present")
                if name == "Maybe"
                else ("Either::Left", "Either::Right")
            )
            if mode == "logical":
                if type(value) not in classes:
                    raise TypeError("invalid native " + name)
                index = classes.index(type(value))
                fields = () if name == "Maybe" and index == 0 else (value.value,)
            else:
                if not isinstance(value, ls.DataValue) or value.tag not in tags:
                    raise ValueError("invalid constructor for " + name)
                index, fields = tags.index(value.tag), value.fields
            expected = 0 if name == "Maybe" and index == 0 else 1
            if len(fields) != expected:
                raise ValueError("wrong constructor arity for " + name)
            converted = (
                ()
                if not expected
                else (
                    self._walk(
                        arguments[0 if name == "Maybe" else index],
                        fields[0],
                        bits,
                        mode,
                        symbols,
                    ),
                )
            )
            return (
                classes[index](*converted)
                if mode == "native"
                else ls.DataValue(tags[index], converted)
            )
        raise ValueError("unsupported type: " + name)

    # Values in generated code are checked where they are built, decoded or
    # drawn, so constructing checks only the new node and matching only
    # dispatches; a deep check of each would make recursion quadratic.
    def construct(self, reference, tag, fields, bits=64, symbols=None):
        if reference.name == "List":
            return ls.construct(tag, fields)
        self._check(reference)
        self._bits(bits)
        return self._walk(
            reference,
            ls.DataValue(tag, fields),
            bits,
            "shallow",
            {} if symbols is None else symbols,
        )

    def match(self, reference, value, branches, bits=64, symbols=None):
        checked = value
        if reference.name == "List":
            return ls.match_list(checked, branches)
        for tag, branch in branches:
            if checked.tag == tag:
                return branch(*checked.fields)
        raise ValueError("missing match branch: " + checked.tag)

    def equal(self, reference, left, right, bits=64, symbols=None):
        symbols = {} if symbols is None else symbols
        left = self.validate(reference, left, bits, symbols)
        right = self.validate(reference, right, bits, symbols)
        if reference.name == "lawspec.collections::type::EqSet":
            return ls.same_members(
                left.fields[0],
                right.fields[0],
                lambda a, b: self.equal(reference.arguments[0], a, b, bits, symbols),
            )
        definition = self._definitions.get(reference.name)
        if definition is not None and definition.handle:
            return left is right
        constructors = self.constructors(reference)
        if constructors is not None:
            if left.tag != right.tag:
                return False
            constructor = next(item for item in constructors if item.tag == left.tag)
            return all(
                self.equal(field.type, a, b, bits, symbols)
                for field, a, b in zip(constructor.fields, left.fields, right.fields)
            )
        name, arguments = reference.name, reference.arguments
        if not arguments:
            return ls.equal(left, right, name, name)
        if name == "List":
            return len(left) == len(right) and all(
                self.equal(arguments[0], a, b, bits, symbols)
                for a, b in zip(left, right)
            )
        if name in ("Nullable", "Optional"):
            return left.present == right.present and (
                not left.present
                or self.equal(arguments[0], left.value, right.value, bits, symbols)
            )
        if left.tag != right.tag:
            return False
        if not left.fields:
            return True
        index = 1 if left.tag == "Either::Right" else 0
        return self.equal(
            arguments[index], left.fields[0], right.fields[0], bits, symbols
        )


def _duration_walk(value, mode):
    if mode == "native":
        if not isinstance(value, ls.DataValue) or value.tag != _DURATION_TAG:
            raise TypeError("expected Duration")
        return ls.timedelta(microseconds=value.fields[0])
    if type(value) is not ls.timedelta:
        raise TypeError("expected a timedelta for Duration")
    micros = (value.days * 86400 + value.seconds) * 1000000 + value.microseconds
    if not 0 <= micros <= _DURATION_LIMIT:
        raise ValueError(
            "Duration outside 0 to " + str(_DURATION_LIMIT) + " microseconds"
        )
    return ls.DataValue(_DURATION_TAG, (micros,))
