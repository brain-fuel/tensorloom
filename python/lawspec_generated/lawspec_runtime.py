"""LawSpec's runtime for Python: the portable scalar domain, seeded generation,
models, actors, sessions and nodes, with no test-framework dependency.

A law must mean the same thing on every target, so this module implements
LawSpec's own arithmetic, equality and conversions instead of Python's:
integers are exact and reach a bounded type only through a checked conversion,
exact division gives a Fraction, decimals ignore the ambient context, and
floats follow IEEE 754 at their declared precision.
ref:DEC-portable-exact-arithmetic ref:ieee-754 ref:decimal-arithmetic

It is emitted unchanged into every generated Python project, so the generated
tests and the adapters share one definition of the domain.
ref:DEC-typed-core-boundary
"""

from collections import deque  # noqa: F401  (native Queue, Stack, Deque)
from datetime import timedelta  # noqa: F401  (native Duration)
from dataclasses import dataclass
from fractions import Fraction
from decimal import Decimal
import math
import struct
import threading


def _finite_cover_rows(cover, prefix):
    """Walk planner-owned ordered domains without expanding a product table."""
    if isinstance(cover, int):
        for _ in range(cover):
            yield prefix
        return
    for domain, suffix in cover:
        if domain[0] == 0:
            values = domain[1]
        else:
            values = (
                lambda _symbols, value=value: value
                for value in range(domain[1], domain[2] + 1)
            )
        for factory in values:
            yield from _finite_cover_rows(suffix, [*prefix, factory])


def finite_cases(cover):
    """Case indices and factories; values are created in each test's context."""
    yield from enumerate(_finite_cover_rows(cover, []))


def search_elements(universal, values, evaluate):
    """Fold three outcomes, preserving unknown and ordered single evaluation."""
    decisive = -1 if universal else 1
    result = -decisive
    for value in values:
        code = evaluate(value)
        if type(code) is not int:
            raise ValueError("invalid finite search outcome")
        if code == decisive:
            return code
        if code == 0:
            result = 0
        elif code != -decisive:
            raise ValueError("invalid finite search outcome")
    return result


def register_finite_cases(namespace, prefix, cover, check, context, wrap=None):
    """Register individually selectable pytest functions with a shared check."""

    def make_case(factories, name, number):
        def run():
            check(factories, f"{context} boundary {number}")

        if wrap is None:
            case = run
        else:

            def case():
                wrap(name, run)

        case.__name__ = name
        return case

    registered = []
    for index, factories in finite_cases(cover):
        name = f"{prefix}_boundary_{index + 1}"
        if name in namespace:
            raise ValueError(f"Duplicate finite case test: {name}")
        case = make_case(factories, name, index + 1)
        namespace[name] = case
        registered.append(case)
    return registered


@dataclass(frozen=True)
class Presence:
    kind: str
    present: bool
    value: object = None


@dataclass(eq=False, frozen=True)
class DataValue:
    tag: str
    fields: tuple

    def __post_init__(self):
        object.__setattr__(self, "fields", tuple(self.fields))


@dataclass(frozen=True)
class Raw:
    kind: str
    units: tuple

    def __post_init__(self):
        object.__setattr__(self, "units", tuple(self.units))


@dataclass(eq=False, frozen=True)
class Symbol:
    description: str


@dataclass(frozen=True)
class Absence:
    kind: str


(UNIT, NULL, UNDEFINED) = (Absence(t) for t in ("Unit", "Null", "Undefined"))


def integer_type(t):
    return t.startswith(("Int", "UInt")) or t in ("BigInt", "BigUInt")


def exact_type(t):
    return integer_type(t) or t in ("Decimal", "Rational")


def bounds(t, bits=64):
    if t in ("Integer", "BigInt", "BigUInt"):
        return (0 if t == "BigUInt" else None, None)
    width = (
        bits
        if t in ("IntSize", "UIntSize", "UIntPtr")
        else int(t[4:] if t.startswith("UInt") else t[3:])
    )
    return (
        (0, 2**width - 1)
        if t.startswith("U")
        else (-(2 ** (width - 1)), 2 ** (width - 1) - 1)
    )


def f32(x):
    try:
        return struct.unpack(">f", struct.pack(">f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def ratio(x):
    if isinstance(x, bool):
        raise ValueError("Bool is not an integer")
    return Fraction(x)


def finite_decimal(r):
    r = ratio(r)
    (d, a, b) = (r.denominator, 0, 0)
    while d % 2 == 0:
        d //= 2
        a += 1
    while d % 5 == 0:
        d //= 5
        b += 1
    if d != 1:
        raise ValueError("Decimal conversion is not finite; use round")
    scale = max(a, b)
    c = r.numerator * 2 ** (scale - a) * 5 ** (scale - b)
    return Decimal((int(c < 0), tuple(map(int, integer_text(abs(c)))), -scale))


def convert(x, t, bits=64):
    """Moves a value into the declared type, failing instead of wrapping or
    rounding silently, because a bounded type is only ever reached through a
    checked conversion. ref:DEC-portable-exact-arithmetic"""
    if integer_type(t):
        r = ratio(x)
        if r.denominator != 1:
            raise ValueError("fractional conversion to " + t)
        x = r.numerator
        (lo, hi) = bounds(t, bits)
        if lo is not None and x < lo or (hi is not None and x > hi):
            raise ValueError("integer outside " + t + " range")
        return x
    if t == "Rational":
        return ratio(x)
    if t == "Decimal":
        return finite_decimal(x)
    if t in ("Float32", "Float64"):
        if not isinstance(x, float):
            return exact_float(ratio(x), t)
        try:
            v = float(x)
        except OverflowError:
            v = math.copysign(math.inf, -1 if x < 0 else 1)
        return f32(v) if t == "Float32" else v
    if t in ("Complex64", "Complex128"):
        z = (
            x
            if isinstance(x, complex)
            else complex(convert(x, "Float32" if t == "Complex64" else "Float64"), 0)
        )
        return complex(f32(z.real), f32(z.imag)) if t == "Complex64" else z
    return validate(x, t, bits)


def valid_unit(t, c):
    maximum = (
        255 if t == "Bytes" else 65535 if t in ("Utf16Text", "CodeUnit16") else 1114111
    )
    return (
        type(c) is int
        and 0 <= c <= maximum
        and (t not in ("Text", "Char") or not 55296 <= c <= 57343)
    )


def validate(x, t, bits=64):
    """Checks a value an adapter produced against its declared domain: native
    code may return anything its own type allows, and a law quantifies only
    over the declared domain. ref:DEC-portable-exact-arithmetic"""
    if t.startswith("Either "):
        types = either_arguments(t)
        if (
            not isinstance(x, DataValue)
            or len(x.fields) != 1
            or x.tag not in ("Either::Left", "Either::Right")
        ):
            raise ValueError("invalid Either constructor or arity")
        inner = types[0 if x.tag == "Either::Left" else 1]
        return DataValue(x.tag, (validate(x.fields[0], inner, bits),))
    if t.startswith("Maybe "):
        if not isinstance(x, DataValue):
            raise TypeError("expected Maybe")
        if x.tag == "Maybe::Absent" and (not x.fields):
            return x
        if x.tag == "Maybe::Present" and len(x.fields) == 1:
            return DataValue(x.tag, (validate(x.fields[0], t[6:], bits),))
        raise ValueError("invalid Maybe constructor or arity")
    if t.startswith("List "):
        if not isinstance(x, list):
            raise TypeError("expected List")
        return [validate(value, t[5:], bits) for value in x]
    if t.startswith(("Nullable ", "Optional ")):
        (k, inner) = t.split(" ", 1)
        if not isinstance(x, Presence) or x.kind != k or type(x.present) is not bool:
            raise ValueError("tagged presence required for " + t)
        if x.present:
            return Presence(k, True, validate(x.value, inner, bits))
    elif integer_type(t):
        if type(x) is not int:
            raise ValueError("integer required for " + t)
        convert(x, t, bits)
    elif t == "Bool":
        if type(x) is not bool:
            raise ValueError("Bool required")
    elif t in ("Text", "Char"):
        if (
            not isinstance(x, str)
            or (t == "Char" and len(x) != 1)
            or not all(valid_unit(t, ord(c)) for c in x)
        ):
            raise ValueError("invalid " + t)
    elif t in ("CodePoint", "CodeUnit16"):
        if not valid_unit(t, x):
            raise ValueError("invalid " + t)
    elif t == "Bytes":
        if type(x) is not bytes:
            raise ValueError("Bytes required")
    elif t in ("CodePointText", "Utf16Text"):
        if (
            not isinstance(x, Raw)
            or x.kind != t
            or not all(valid_unit(t, c) for c in x.units)
        ):
            raise ValueError("invalid " + t)
    elif t in ("Unit", "Null", "Undefined"):
        if x != Absence(t):
            raise ValueError("invalid " + t)
    elif t == "Symbol":
        if not isinstance(x, Symbol):
            raise ValueError("Symbol required")
    elif t == "Decimal":
        if not isinstance(x, Decimal) or not x.is_finite():
            raise ValueError("finite Decimal required")
    elif t == "Rational":
        if not isinstance(x, Fraction):
            raise ValueError("Rational required")
    elif t in ("Float32", "Float64"):
        if type(x) is not float or (
            t == "Float32" and not math.isnan(x) and f32(x) != x
        ):
            raise ValueError("invalid " + t)
    elif t in ("Complex64", "Complex128"):
        if type(x) is not complex:
            raise ValueError("complex required")
        if t == "Complex64":
            validate(x.real, "Float32")
            validate(x.imag, "Float32")
    else:
        raise ValueError("unknown scalar type " + t)
    return x


def literal(v, symbols=None):
    t = v["type"]
    if integer_type(t):
        return parse_integer(v["value"])
    if t == "Bool":
        return v["value"]
    if t == "Decimal":
        c = parse_integer(v["coefficient"])
        return Decimal(
            (int(c < 0), tuple(map(int, integer_text(abs(c)))), int(v["exponent"]))
        )
    if t == "Rational":
        return Fraction(parse_integer(v["numerator"]), parse_integer(v["denominator"]))
    if t.startswith("Float"):
        return struct.unpack(
            ">f" if t == "Float32" else ">d", bytes.fromhex(v["bits"])
        )[0]
    if t.startswith("Complex"):
        return complex(literal(v["real"]), literal(v["imaginary"]))
    if t == "Text":
        return "".join(map(chr, v["units"]))
    if t == "Char":
        return chr(v["value"])
    if t in ("CodePoint", "CodeUnit16"):
        return v["value"]
    if t == "Bytes":
        return bytes(v["units"])
    if t in ("CodePointText", "Utf16Text"):
        return Raw(t, tuple(v["units"]))
    if t == "Symbol":
        if symbols is None:
            symbols = {}
        return symbols.setdefault(v["id"], Symbol(v["description"]))
    if t in ("Nullable", "Optional"):
        return Presence(
            t,
            v["value"] is not None,
            literal(v["value"], symbols) if v["value"] is not None else None,
        )
    return Absence(t)


def promote(a, b, op):
    if exact_type(a) != exact_type(b):
        raise ValueError("exact/inexact mixing requires explicit conversion")
    if exact_type(a):
        return (
            "Rational"
            if op == "/" or "Rational" in (a, b)
            else "Decimal"
            if "Decimal" in (a, b)
            else "Integer"
        )
    if a.startswith("Complex") or b.startswith("Complex"):
        return (
            "Complex128"
            if a in ("Float64", "Complex128") or b in ("Float64", "Complex128")
            else "Complex64"
        )
    return "Float64" if "Float64" in (a, b) else "Float32"


def ieee_div(a, b):
    if b != 0:
        return a / b
    if a == 0 or math.isnan(a):
        return math.nan
    return math.copysign(math.inf, math.copysign(1, a) * math.copysign(1, b))


def binary(op, a, b, ta, tb):
    """Applies a LawSpec operator with LawSpec's semantics rather than
    Python's, so a law computes the same result on every target.
    ref:DEC-portable-exact-arithmetic ref:ieee-754"""
    if (
        op in ("==", "!=")
        and not exact_type(ta)
        and ta not in ("Float32", "Float64", "Complex64", "Complex128")
    ):
        result = equal(a, b, ta, tb)
        return result if op == "==" else not result
    t = promote(ta, tb, op)
    if exact_type(ta):
        (a, b) = (ratio(a), ratio(b))
    elif t.startswith("Complex"):
        (a, b) = (complex(a), complex(b))
        rnd = f32 if t == "Complex64" else float
        if op == "+":
            return complex(rnd(a.real + b.real), rnd(a.imag + b.imag))
        if op == "-":
            return complex(rnd(a.real - b.real), rnd(a.imag - b.imag))
        if op == "*":
            return complex(
                rnd(rnd(a.real * b.real) - rnd(a.imag * b.imag)),
                rnd(rnd(a.real * b.imag) + rnd(a.imag * b.real)),
            )
        if op == "/":
            d = rnd(rnd(b.real * b.real) + rnd(b.imag * b.imag))
            return complex(
                rnd(ieee_div(rnd(rnd(a.real * b.real) + rnd(a.imag * b.imag)), d)),
                rnd(ieee_div(rnd(rnd(a.imag * b.real) - rnd(a.real * b.imag)), d)),
            )
    if op == "pow":
        if a.denominator != 1 or b.denominator != 1:
            raise TypeError("pow requires integer operands")
        if b < 0:
            raise ValueError("negative exponent")
        return a.numerator**b.numerator
    if op in ("quot", "rem"):
        q = abs(a.numerator) // abs(b.numerator)
        if (a < 0) != (b < 0):
            q = -q
        return q if op == "quot" else a.numerator - q * b.numerator
    if op in ("<", "<=", ">", ">=", "==", "!="):
        return {
            "<": lambda: a < b,
            "<=": lambda: a <= b,
            ">": lambda: a > b,
            ">=": lambda: a >= b,
            "==": lambda: a == b,
            "!=": lambda: a != b,
        }[op]()
    if op == "+":
        v = a + b
    elif op == "-":
        v = a - b
    elif op == "*":
        v = a * b
    elif op == "/":
        v = a / b if exact_type(ta) else ieee_div(a, b)
    else:
        raise ValueError("unknown operation " + op)
    return convert(v, t)


# Handles are values only adapters create, passed along unopened. The schema
# registers each one as it crosses into LawSpec, so the runtime knows it: it
# is equal only to itself, has no portable order, and renders as a stable
# label numbered by first appearance in the process (Jobs#1).
_handles = {}
_handle_counts = {}
_handle_lock = threading.Lock()


def handle(value, name):
    """Registers a handle of the named type; returns it unchanged."""
    if id(value) not in _handles:
        with _handle_lock:
            if id(value) not in _handles:
                short = name.rsplit("::", 1)[-1]
                count = _handle_counts.get(short, 0) + 1
                _handle_counts[short] = count
                # The entry keeps the value alive, so its id stays its own.
                _handles[id(value)] = (value, short + "#" + str(count))
    return value


def is_handle(value):
    entry = _handles.get(id(value))
    return entry is not None and entry[0] is value


def handle_label(value):
    return _handles[id(value)][1]


def same_members(left, right, equal):
    return all(
        any(equal(value, candidate) for candidate in right) for value in left
    ) and all(any(equal(value, candidate) for candidate in left) for value in right)


def distinct_members(items, equal):
    return all(
        not any(equal(value, candidate) for candidate in items[:index])
        for index, value in enumerate(items)
    )


def equal(a, b, ta, tb):
    """Equality defined once for all targets instead of borrowed from each
    language's: NaN differs from itself, signed zeros are equal, handles and
    symbols compare by identity, and data compares field by field.
    ref:DEC-portable-exact-arithmetic"""
    if is_handle(a) or is_handle(b):
        return a is b
    if ta.startswith("Either ") and tb.startswith("Either "):
        if a.tag != b.tag:
            return False
        index = 0 if a.tag == "Either::Left" else 1
        return equal(
            a.fields[0],
            b.fields[0],
            either_arguments(ta)[index],
            either_arguments(tb)[index],
        )
    if ta.startswith("Maybe ") and tb.startswith("Maybe "):
        return a.tag == b.tag and (
            a.tag == "Maybe::Absent" or equal(a.fields[0], b.fields[0], ta[6:], tb[6:])
        )
    if ta.startswith("List ") and tb.startswith("List "):
        return len(a) == len(b) and all(
            equal(x, y, ta[5:], tb[5:]) for x, y in zip(a, b)
        )
    if exact_type(ta) and exact_type(tb):
        return ratio(a) == ratio(b)
    if isinstance(a, Presence) and isinstance(b, Presence):
        return (
            a.kind == b.kind
            and a.present == b.present
            and (
                not a.present
                or equal(a.value, b.value, ta.split(" ", 1)[1], tb.split(" ", 1)[1])
            )
        )
    if isinstance(a, complex) and isinstance(b, complex):
        return a.real == b.real and a.imag == b.imag
    return a == b


def _run_blocking(main):
    """Runs a coroutine to completion and returns its result. Called where an
    event loop is already running (a synchronous definition called from an
    asynchronous adapter), it runs on a thread of its own."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(main)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, main).result()


def await_task(task):
    """An async adapter's result: run its coroutine (or awaitable) to completion."""

    async def result():
        return await task

    return _run_blocking(result())


ORDERING = "lawspec.collections::type::Ordering::"


def compare_values(a, b):
    """The portable total order: -1, 0 or 1.

    Exact numbers by value, text and raw sequences by unit, False before True,
    absence before presence, lists element by element, Absent before Present,
    and other data by constructor identity, then fields left to right.
    Handles have no order: one equals only itself.
    """
    if is_handle(a) and is_handle(b):
        if a is b:
            return 0
        raise TypeError("handles have no portable order")
    if isinstance(a, bool) and isinstance(b, bool):
        return (a > b) - (a < b)
    if isinstance(a, (int, Fraction, Decimal)) and not isinstance(a, bool):
        x, y = ratio(a), ratio(b)
        return (x > y) - (x < y)
    if isinstance(a, str) and isinstance(b, str):
        return (a > b) - (a < b)
    if isinstance(a, Raw) and isinstance(b, Raw):
        return (a.units > b.units) - (a.units < b.units)
    if isinstance(a, Absence) and isinstance(b, Absence):
        return 0
    if isinstance(a, Presence) and isinstance(b, Presence):
        if a.present != b.present:
            return 1 if a.present else -1
        return compare_values(a.value, b.value) if a.present else 0
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        for x, y in zip(a, b):
            order = compare_values(x, y)
            if order:
                return order
        return (len(a) > len(b)) - (len(a) < len(b))
    if isinstance(a, DataValue) and isinstance(b, DataValue):
        if a.tag != b.tag:
            if {a.tag, b.tag} == {"Maybe::Absent", "Maybe::Present"}:
                return -1 if a.tag == "Maybe::Absent" else 1
            return (a.tag > b.tag) - (a.tag < b.tag)
        return compare_values(list(a.fields), list(b.fields))
    raise TypeError("values have no portable order")


def helper(n, args, types, bits=64):
    if n == "incompleteSearch":
        raise ValueError(args[0])
    x = args[0] if args else None
    if n in _SEQUENCE_HELPERS:
        return _SEQUENCE_HELPERS[n](*args)
    if n == "startsWith":
        return args[0].startswith(args[1])
    if n == "endsWith":
        return args[0].endswith(args[1])
    if n == "textContains":
        return args[1] in args[0]
    if n == "regexMatches":
        return regex_matches(args[0], args[1])
    if n == "recorded":
        return recorded(
            args[0],
            args[1],
            (lambda value: recorded_text(value, types[1])) if len(types) > 1 else None,
        )
    if n == "acquireResource":
        return acquire_resource(args[0])
    if n == "releaseResource":
        return release_resource(args[0], args[1])
    if n == "freePort":
        return free_port()
    if n == "checked":
        return True
    if n == "select":
        return args[1] if args[0] else args[2]
    if n == "compare":
        order = compare_values(args[0], args[1])
        return DataValue(ORDERING + ("Less", "Equal", "Greater")[order + 1], ())
    if n == "length":
        return len(x.units) if isinstance(x, Raw) else len(x)
    if n == "isPresent":
        return x.present
    if n == "presentValue":
        if not x.present:
            raise ValueError("absent presence value")
        return x.value
    if n == "real":
        return x.real
    if n == "imag":
        return x.imag
    if n == "negate":
        if exact_type(types[0]):
            return binary("-", 0, x, "BigInt", types[0])
        return convert(-x, types[0])
    if n in ("quot", "rem", "pow"):
        return binary(n, *args, *types)
    if n == "isNaN":
        return math.isnan(x)
    if n == "isInfinite":
        return math.isinf(x)
    if n == "isFinite":
        return math.isfinite(x)
    if n == "isNegativeZero":
        return x == 0 and math.copysign(1, x) < 0
    if n == "round":
        scale = convert(args[1], "Int32", bits)
        factor = Fraction(10**scale) if scale >= 0 else Fraction(1, 10 ** (-scale))
        return finite_decimal(Fraction(round(ratio(x) * factor), 1) / factor)
    return convert(x, n, bits)


# Sequences: Text (str), Bytes (bytes) and List (list). Indices count code
# points, octets or items, and a slice is clamped to the sequence.
def _append(a, b):
    if isinstance(a, Raw):
        return Raw(a.kind, a.units + b.units)
    return a + b


def _slice(x, start, end):
    count = len(x.units) if isinstance(x, Raw) else len(x)
    i = max(0, min(count, start))
    j = max(i, min(count, end))
    return Raw(x.kind, x.units[i:j]) if isinstance(x, Raw) else x[i:j]


def _at(x, i):
    if 0 <= i < len(x):
        return construct("Maybe::Present", [x[i]])
    return construct("Maybe::Absent", [])


_SEQUENCE_HELPERS = {
    "append": _append,
    "slice": _slice,
    "at": _at,
    "range": lambda lo, hi: list(range(lo, hi)),
    "octets": lambda b: list(b),
    "bytesOf": lambda xs: bytes(xs),
    "utf8": lambda t: t.encode("utf-8"),
    "textOf": lambda n: str(n),
    "join": lambda separator, parts: separator.join(parts),
}


# Portable regular expressions (see LawSpec.Regex): the subset of RE2 and
# ECMAScript that means the same in both, matched against a whole text, code
# point by code point. The compiler has checked every pattern; a pattern
# that is not portable raises ValueError here too.
_REGEX_DIGITS = [(48, 57)]
_REGEX_WORD = [(48, 57), (65, 90), (95, 95), (97, 122)]
_REGEX_SPACE = [(9, 13), (32, 32)]
_REGEX_CACHE = {}


def _regex_parse(pattern):
    cs = [ord(c) for c in pattern]
    n = len(cs)
    pos = [0]

    def peek():
        return cs[pos[0]] if pos[0] < n else None

    def fail(message):
        raise ValueError(f"regex {pattern!r} is not portable: {message}")

    def escape(in_class):
        pos[0] += 1
        c = peek()
        if c is None:
            fail("the regex ends with a lone \\")
        pos[0] += 1
        ch = chr(c)
        classes = {
            "d": (False, _REGEX_DIGITS),
            "D": (True, _REGEX_DIGITS),
            "w": (False, _REGEX_WORD),
            "W": (True, _REGEX_WORD),
            "s": (False, _REGEX_SPACE),
            "S": (True, _REGEX_SPACE),
        }
        if ch in classes:
            return classes[ch]
        controls = {"n": 10, "t": 9, "r": 13, "f": 12, "v": 11}
        if ch in controls:
            return (False, [(controls[ch], controls[ch])])
        if ch in "\\.^$|?*+()[]{}-/":
            return (False, [(c, c)])
        fail(f"\\{ch} is not a portable escape")

    def char_class():
        pos[0] += 1
        negated = peek() == ord("^")
        if negated:
            pos[0] += 1
        items = []
        first = True
        while True:
            c = peek()
            if c is None:
                fail("a [ is never closed")
            if c == ord("]"):
                if first:
                    fail("an empty class is not portable")
                pos[0] += 1
                return ("set", negated, items)
            first = False
            item = escape(True) if c == ord("\\") else _literal()
            if (
                not item[0]
                and len(item[1]) == 1
                and item[1][0][0] == item[1][0][1]
                and peek() == ord("-")
                and pos[0] + 1 < n
                and cs[pos[0] + 1] != ord("]")
            ):
                pos[0] += 1
                high = escape(True) if peek() == ord("\\") else _literal()
                if high[0] or len(high[1]) != 1 or high[1][0][0] != high[1][0][1]:
                    fail("a range ends with one character")
                if high[1][0][0] < item[1][0][0]:
                    fail("a range must run from low to high")
                items.append((False, [(item[1][0][0], high[1][0][0])]))
            else:
                items.append(item)

    def _literal():
        c = peek()
        if c == ord("["):
            fail("write \\[ for the character inside a class")
        pos[0] += 1
        return (False, [(c, c)])

    def atom():
        c = peek()
        if c == ord("("):
            pos[0] += 1
            if peek() == ord("?"):
                if pos[0] + 1 < n and cs[pos[0] + 1] == ord(":"):
                    pos[0] += 2
                else:
                    fail("only (?: ...) groups are portable")
            node = alternatives()
            if peek() != ord(")"):
                fail("a ( is never closed")
            pos[0] += 1
            return node
        if c == ord("["):
            return char_class()
        if c == ord("."):
            pos[0] += 1
            return ("set", True, [(False, [(10, 10)])])
        if c == ord("\\"):
            return ("set", False, [escape(False)])
        if chr(c) in "*+?{^$]}":
            fail(f"unexpected {chr(c)}")
        pos[0] += 1
        return ("set", False, [(False, [(c, c)])])

    def quantified(node):
        c = peek()
        if c is None or chr(c) not in "*+?{":
            return node
        pos[0] += 1
        if c == ord("*"):
            node = ("repeat", node, 0, None)
        elif c == ord("+"):
            node = ("repeat", node, 1, None)
        elif c == ord("?"):
            node = ("repeat", node, 0, 1)
        else:
            start = pos[0]
            while peek() is not None and chr(peek()).isdigit():
                pos[0] += 1
            low_text = pattern[start : pos[0]]
            high = None
            if peek() == ord("}"):
                high = low_text
            elif peek() == ord(","):
                pos[0] += 1
                start = pos[0]
                while peek() is not None and chr(peek()).isdigit():
                    pos[0] += 1
                high = pattern[start : pos[0]] or None
                if peek() != ord("}"):
                    fail("a repetition is {n}, {n,} or {n,m}")
            else:
                fail("a repetition is {n}, {n,} or {n,m}")
            pos[0] += 1
            if not low_text:
                fail("a repetition is {n}, {n,} or {n,m}")
            low = int(low_text)
            high = None if high is None else int(high)
            if low > 1000 or (high is not None and (high > 1000 or high < low)):
                fail("a repetition count is at most 1000, and n must not exceed m")
            node = ("repeat", node, low, high)
        if peek() is not None and chr(peek()) in "*+?{":
            fail("a repetition cannot itself be repeated")
        return node

    def sequence():
        items = []
        while peek() is not None and peek() not in (ord("|"), ord(")")):
            items.append(quantified(atom()))
        return ("seq", items)

    def alternatives():
        branches = [sequence()]
        while peek() == ord("|"):
            pos[0] += 1
            branches.append(sequence())
        return branches[0] if len(branches) == 1 else ("alt", branches)

    node = alternatives()
    if pos[0] != n:
        fail("a ) has no ( before it")
    return node


def _regex_reach(node, text, positions):
    kind = node[0]
    if kind == "set":
        _, negated, items = node
        result = set()
        for p in positions:
            if p < len(text):
                c = text[p]
                inside = any(
                    item_negated != any(lo <= c <= hi for lo, hi in ranges)
                    for item_negated, ranges in items
                )
                if inside != negated:
                    result.add(p + 1)
        return result
    if kind == "seq":
        for item in node[1]:
            positions = _regex_reach(item, text, positions)
        return positions
    if kind == "alt":
        result = set()
        for branch in node[1]:
            result |= _regex_reach(branch, text, positions)
        return result
    _, body, low, high = node
    for _ in range(low):
        positions = _regex_reach(body, text, positions)
    seen = set(positions)
    frontier = set(positions)
    limit = None if high is None else high - low
    while frontier and limit != 0:
        frontier = _regex_reach(body, text, frontier) - seen
        seen |= frontier
        if limit is not None:
            limit -= 1
    return seen


def regex_matches(pattern, text):
    """Whether the portable regex matches the whole text."""
    node = _REGEX_CACHE.get(pattern)
    if node is None:
        node = _REGEX_CACHE[pattern] = _regex_parse(pattern)
    points = [ord(c) for c in text]
    return len(points) in _regex_reach(node, points, {0})


# Recorded values: a law compares a value's portable rendering with the
# text stored under recorded/<unit>/<name> in the project. LAWSPEC_RECORDED
# names the folder; otherwise it is recorded/ in the nearest folder, from
# the working one up, that holds lawspec.edn or recorded/. With
# LAWSPEC_UPDATE_RECORDED=1 (lawspec test --update-recorded) a law records
# the value instead.
def recorded_root():
    import os

    given = os.environ.get("LAWSPEC_RECORDED")
    if given:
        return given
    folder = os.path.abspath(os.getcwd())
    while True:
        if os.path.isfile(os.path.join(folder, "lawspec.edn")) or os.path.isdir(
            os.path.join(folder, "recorded")
        ):
            return os.path.join(folder, "recorded")
        parent = os.path.dirname(folder)
        if parent == folder:
            return os.path.join(os.path.abspath(os.getcwd()), "recorded")
        folder = parent


def recorded(key, value, render_value=None):
    import os

    text = (render if render_value is None else render_value)(value)
    path = os.path.join(recorded_root(), *key.split("/"))
    if os.environ.get("LAWSPEC_UPDATE_RECORDED") == "1":
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
        return True
    if not os.path.isfile(path):
        raise AssertionError(
            f"no recording recorded/{key}; run lawspec test --update-recorded "
            f"to record {text}"
        )
    with open(path, encoding="utf-8", newline="") as f:
        stored = f.read()
    if stored.endswith("\n"):
        stored = stored[:-1]
    if stored != text:
        raise AssertionError(
            f"recorded/{key} differs: expected {stored}, actual {text}"
            " (lawspec test --update-recorded records the new value)"
        )
    return True


def recorded_text(value, type_ref, field_types=None):
    """Typed recording text, without rerunning native conversions or predicates.

    ref:REQ-law-primitives ref:DEC-portable-exact-arithmetic
    """
    identities = {}

    def quote(text):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'

    def identity(kind, item):
        known = identities.setdefault(kind, {})
        key = id(item)
        if key not in known:
            known[key] = len(known) + 1
        return known[key]

    def reference(t):
        if not isinstance(t, str):
            return t
        for name in ("List", "Maybe", "Nullable", "Optional"):
            if t.startswith(name + " "):
                inner = t[len(name) + 1 :]
                if inner.startswith("(") and inner.endswith(")"):
                    inner = inner[1:-1]
                return name, (reference(inner),)
        if t.startswith("Either "):
            return "Either", tuple(reference(child) for child in either_arguments(t))
        return t, ()

    def float_text(v, width):
        if math.isnan(v):
            return f"float{width}NaN"
        return (
            f'float{width}Bits("{struct.pack(">f" if width == 32 else ">d", v).hex()}")'
        )

    def units(name, values):
        return name + "([" + ", ".join(integer_text(v) for v in values) + "])"

    def visit(v, input_type):
        t = reference(input_type)
        name, arguments = (t.name, t.arguments) if hasattr(t, "name") else t
        if name == "List":
            return "[" + ", ".join(visit(item, arguments[0]) for item in v) + "]"
        if name in ("Nullable", "Optional"):
            if not v.present:
                return "null" if name == "Nullable" else "undefined"
            return name.lower() + "(" + visit(v.value, arguments[0]) + ")"
        if isinstance(v, DataValue):
            if name == "Maybe":
                types = arguments if v.fields else ()
            elif name == "Either":
                types = (arguments[0 if v.tag == "Either::Left" else 1],)
            else:
                if field_types is None:
                    raise TypeError("recording data requires its schema")
                types = field_types(t, v)
            tag = v.tag.split("::")[-1]
            return (
                tag
                if not v.fields
                else tag
                + "("
                + ", ".join(visit(item, child) for item, child in zip(v.fields, types))
                + ")"
            )
        if name == "Symbol":
            return f"symbol({identity('symbol', v)}, {quote(v.description)})"
        if name in ("Unit", "Null", "Undefined"):
            return {"Unit": "()", "Null": "null", "Undefined": "undefined"}[name]
        if name in ("Text", "Char"):
            return quote(v)
        if name == "Bytes":
            return units("bytes", v)
        if name in ("CodePointText", "Utf16Text"):
            return units("codePoints" if name == "CodePointText" else "utf16", v.units)
        if name == "Decimal":
            sign, digits, exponent = v.as_tuple()
            coefficient = parse_integer("".join(map(str, digits))) * (-1 if sign else 1)
            if not coefficient:
                return "0e0"
            while coefficient % 10 == 0:
                coefficient //= 10
                exponent += 1
            return integer_text(coefficient) + "e" + integer_text(exponent)
        if name == "Rational":
            return (
                f"rational({integer_text(v.numerator)}, {integer_text(v.denominator)})"
            )
        if name in ("Float32", "Float64"):
            return float_text(v, 32 if name == "Float32" else 64)
        if name in ("Complex64", "Complex128"):
            width = 32 if name == "Complex64" else 64
            return (
                name
                + "("
                + float_text(v.real, width)
                + ", "
                + float_text(v.imag, width)
                + ")"
            )
        if name == "Bool":
            return "true" if v else "false"
        if integer_type(name) or name in ("CodePoint", "CodeUnit16"):
            return integer_text(v)
        if is_handle(v):
            return name.split("::")[-1] + "#" + str(identity("handle:" + name, v))
        raise TypeError("unsupported recorded type: " + name)

    return visit(value, type_ref)


# Built-in resources (see LawSpec.Resources): a law acquires them before
# each case and releases them after it.
def acquire_resource(kind):
    import json
    import os
    import tempfile

    if kind == "temporaryDirectory":
        return tempfile.mkdtemp(prefix="lawspec-")
    if kind == "temporaryFile":
        handle, path = tempfile.mkstemp(prefix="lawspec-")
        os.close(handle)
        return path
    if kind == "environment":
        return json.dumps(dict(os.environ), sort_keys=True)
    raise ValueError(f"unknown resource kind {kind}")


def release_resource(kind, value):
    import json
    import os
    import shutil

    if kind == "temporaryDirectory":
        shutil.rmtree(value, ignore_errors=True)
    elif kind == "temporaryFile":
        if os.path.exists(value):
            os.remove(value)
    elif kind == "environment":
        saved = json.loads(value)
        for name in list(os.environ):
            if name not in saved:
                del os.environ[name]
        for name, text in saved.items():
            if os.environ.get(name) != text:
                os.environ[name] = text
    else:
        raise ValueError(f"unknown resource kind {kind}")
    return True


# Shared resources (share R per group | unit | run, in a harness): one value
# per scope key. The first use acquires it; every later use resets it first,
# so no case sees what another left. One case holds it at a time, and it is
# released when the test process ends.
_shared = {}
_shared_guard = threading.Lock()


def share(key, acquire, reset, release, concurrent=False):
    """A concurrent resource (resource T is concurrent) is held by any
    number of cases at once, and reset only when none holds it."""
    import atexit

    with _shared_guard:
        entry = _shared.setdefault(
            key,
            {
                "lock": threading.Lock(),
                "held": False,
                "users": 0,
                "concurrent": concurrent,
            },
        )
    if concurrent:
        with entry["lock"]:
            if entry["held"] and entry["users"] == 0:
                reset(entry["value"])
            elif not entry["held"]:
                entry["value"] = acquire()
                entry["held"] = True
                atexit.register(lambda: release(entry["value"]))
            entry["users"] += 1
        return entry["value"]
    entry["lock"].acquire()
    try:
        if entry["held"]:
            reset(entry["value"])
        else:
            entry["value"] = acquire()
            entry["held"] = True
            atexit.register(lambda: release(entry["value"]))
    except BaseException:
        entry["lock"].release()
        raise
    return entry["value"]


def unshare(key):
    entry = _shared[key]
    if entry["concurrent"]:
        with entry["lock"]:
            entry["users"] -= 1
    else:
        entry["lock"].release()


def free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_decimal(c, e):
    return Decimal((int(c < 0), tuple(map(int, integer_text(abs(c)))), e))


def exact_float(r, t):
    if not r:
        return 0.0
    negative = r < 0
    (n, d) = (abs(r.numerator), r.denominator)
    single = t == "Float32"
    (p, bias) = (24, 127) if single else (53, 1023)
    (emin, emax) = (1 - bias, bias)
    e = n.bit_length() - d.bit_length()
    if n < d << e if e >= 0 else n << -e < d:
        e -= 1
    if e > emax:
        return -math.inf if negative else math.inf
    scale = max(e, emin) - (p - 1)
    (num, den) = (n << -scale if scale < 0 else n, d << scale if scale > 0 else d)
    (q, rem) = divmod(num, den)
    if 2 * rem > den or (2 * rem == den and q % 2):
        q += 1
    e = max(e, emin)
    if q == 1 << p:
        q >>= 1
        e += 1
    if e > emax:
        return -math.inf if negative else math.inf
    hidden = 1 << p - 1
    exponent = 0 if q < hidden else e + bias
    mantissa = q if q < hidden else q - hidden
    bits = int(negative) << (31 if single else 63) | exponent << p - 1 | mantissa
    return struct.unpack(
        ">f" if single else ">d", bits.to_bytes(4 if single else 8, "big")
    )[0]


def parse_integer(text):
    negative = text.startswith("-")
    text = text.lstrip("+-")
    if not text or not text.isascii() or (not text.isdigit()):
        raise ValueError("invalid integer representation")
    result = 0
    for start in range(0, len(text), 9):
        chunk = text[start : start + 9]
        result = result * 10 ** len(chunk) + int(chunk)
    return -result if negative else result


def integer_text(value):
    if value == 0:
        return "0"
    (negative, value) = (value < 0, abs(value))
    chunks = []
    while value:
        (value, rest) = divmod(value, 1000000000)
        chunks.append(rest)
    return (
        ("-" if negative else "")
        + str(chunks[-1])
        + "".join(f"{n:09d}" for n in reversed(chunks[:-1]))
    )


def unit_result(value):
    return UNIT if value is None else validate(value, "Unit")


# Abilities (docs/explanation/abilities.md). Handlers travel in the symbols
# context generated code passes to every definition: that context is the
# evidence of evidence-passing compilation. A law installs one handler per
# ability; an operation finds the handler of its ability there. The Fail
# ability's handlers abort, so raise is an exception and attempt catches it.
_HANDLERS = "_lawspec_handlers"


class Failure(Exception):
    """A failure raised with the Fail ability, aborting to the nearest
    attempt of the same ability."""

    def __init__(self, ability, value):
        super().__init__(f"failed with {value!r} ({ability})")
        self.ability = ability
        self.value = value


def install_handlers(symbols, handlers):
    table = dict(symbols.get(_HANDLERS, {}))
    table.update(handlers)
    symbols[_HANDLERS] = table
    return symbols


def handler(symbols, ability):
    table = symbols.get(_HANDLERS) if isinstance(symbols, dict) else None
    if not table or ability not in table:
        raise LookupError(
            f"no handler for the ability {ability}: a law "
            "names one with `using`, or runs under each "
            "lawful handler"
        )
    return table[ability]


def raise_failure(ability, value):
    raise Failure(ability, value)


def attempt(ability, body, right, left):
    try:
        value = body()
    except Failure as failure:
        if failure.ability != ability:
            raise
        return left(failure.value)
    return right(value)


def native_handler(module, name, *made_with):
    """A production handler bound in lawspec.edn: its class (or function of
    no arguments), called; with the case's resource, for an ability made
    with one."""
    import importlib

    return getattr(importlib.import_module(module), name)(*made_with)


_HANDLER_SCHEMA = None


def set_handler_schema(schema):
    """The schema a handler's values cross with: lawspec_native sets it to
    the bound native types when lawspec.edn binds them."""
    global _HANDLER_SCHEMA
    _HANDLER_SCHEMA = schema


def handler_schema(default):
    return default if _HANDLER_SCHEMA is None else _HANDLER_SCHEMA


def with_handlers(symbols, handlers, body):
    """handle e with h end: runs body with these handlers installed, then
    puts back the ones they replaced."""
    previous = symbols.get(_HANDLERS)
    table = dict(previous or {})
    table.update(handlers)
    symbols[_HANDLERS] = table
    try:
        return body()
    finally:
        if previous is None:
            symbols.pop(_HANDLERS, None)
        else:
            symbols[_HANDLERS] = previous


def native_class(module, name):
    """A class of a generated or user-owned module, by name."""
    import importlib

    return getattr(importlib.import_module(module), name)


def unknown_handler(name):
    raise LookupError(f"no spec handler called {name}")


class Fail(Exception):
    """Raised by native code (an adapter, or a production handler) to fail
    with a value of the failure type its signature names: `fails with E`.
    The generated code turns it into the Fail ability's failure."""

    def __init__(self, value):
        super().__init__(f"failed with {value!r}")
        self.value = value


def native_failures(ability, convert, body, mapped=()):
    """Calls native code that may fail: ls.Fail(value), or an exception
    lawspec.edn maps to a failure (mapped holds (class, make) pairs),
    becomes a failure of the ability."""
    try:
        return body()
    except Fail as failure:
        raise Failure(ability, convert(failure.value)) from failure
    except Exception as error:
        for kind, make in mapped:
            if isinstance(error, kind):
                raise Failure(ability, make(error)) from error
        raise


def count_calls(recording, operation, matches=None):
    calls = getattr(recording, "calls", None)
    if calls is None:
        raise TypeError("calls of needs a recording handler: `using recording`")
    return sum(
        1
        for (name, arguments) in calls
        if name == operation and (matches is None or matches(arguments))
    )


# Dependent-domain operations are independent of test frameworks.
def sample(t, seed, bits=64):
    """Draws a value of a scalar type from a seed, the same value every target
    draws, so a failing seed reproduces anywhere.
    ref:DEC-portable-seeded-generation"""
    import random

    r = random.Random(seed)
    if t.startswith(("Nullable ", "Optional ")):
        (kind, inner) = t.split(" ", 1)
        return Presence(
            kind, bool(seed % 2), sample(inner, seed // 2, bits) if seed % 2 else None
        )
    if integer_type(t):
        (lo, hi) = bounds(t, bits)
        return r.randint(
            lo if lo is not None else -(2**256), hi if hi is not None else 2**256
        )
    if t == "Bool":
        return bool(seed % 2)
    if t == "Decimal":
        return make_decimal(r.randint(-(2**128), 2**128), r.randint(-20, 20))
    if t == "Rational":
        return Fraction(r.randint(-(2**128), 2**128), r.randint(1, 2**128))
    if t.startswith("Float"):
        return literal(
            {
                "type": t,
                "bits": format(
                    r.getrandbits(32 if t == "Float32" else 64),
                    "08x" if t == "Float32" else "016x",
                ),
            }
        )
    if t.startswith("Complex"):
        c = "Float32" if t == "Complex64" else "Float64"
        return complex(sample(c, seed, bits), sample(c, seed + 1, bits))
    if t in ("Unit", "Null", "Undefined"):
        return {"Unit": UNIT, "Null": NULL, "Undefined": UNDEFINED}[t]
    if t == "Symbol":
        return Symbol("same")
    maximum = (
        255 if t == "Bytes" else 65535 if t in ("Utf16Text", "CodeUnit16") else 1114111
    )

    def unit():
        while True:
            c = r.randint(0, maximum)
            if t not in ("Text", "Char") or not 55296 <= c <= 57343:
                return c

    if t == "Char":
        return chr(unit())
    if t in ("CodePoint", "CodeUnit16"):
        return unit()
    units = tuple((unit() for _ in range(r.randint(0, 39))))
    if t == "Text":
        return "".join(map(chr, units))
    if t == "Bytes":
        return bytes(units)
    return Raw(t, units)


def domain_candidates(t, seed, bits, restrictions, hints):
    candidates = []
    for hint in hints:
        try:
            candidates.append(convert(hint, t, bits))
        except (ValueError, TypeError, OverflowError):
            pass
    if integer_type(t):
        (lo, hi) = bounds(t, bits)
        for op, value in restrictions:
            r = ratio(value)
            if op in (">", ">=", "=="):
                bound = math.floor(r) + 1 if op == ">" else math.ceil(r)
                lo = bound if lo is None else max(lo, bound)
            if op in ("<", "<=", "=="):
                bound = math.ceil(r) - 1 if op == "<" else math.floor(r)
                hi = bound if hi is None else min(hi, bound)
        if lo is not None and hi is not None and (lo > hi):
            return []
        lower = lo if lo is not None else min(-(2**256), (hi or 0) - 2**256)
        upper = hi if hi is not None else max(2**256, (lo or 0) + 2**256)
        candidates += [lower, upper, 0, 1, -1, lower + 1, upper - 1]
        import random

        r = random.Random(seed)
        candidates += [r.randint(lower, upper) for _ in range(8)]
        candidates = [
            v
            for v in candidates
            if isinstance(v, int) and not isinstance(v, bool) and lower <= v <= upper
        ]
    else:
        candidates += [sample(t, seed + j * 7919, bits) for j in range(8)]
    if candidates:
        offset = seed % len(candidates)
        candidates = candidates[offset:] + candidates[:offset]
    return candidates


def generate_tuple(domains, seed, attempts, prefix=()):
    used = 0
    last_prefix = list(prefix)

    def search(values):
        nonlocal used, last_prefix
        last_prefix = values
        if len(values) == len(domains):
            return values
        if used >= attempts:
            return None
        used += 1
        (candidates, accept) = domains[len(values)]
        for value in candidates(values, seed + used * 7919):
            if used >= attempts:
                break
            used += 1
            next_values = values + [value]
            if accept(next_values):
                result = search(next_values)
                if result is not None:
                    return result
        return None

    while used < attempts:
        result = search(list(prefix))
        if result is not None:
            return result
    raise ValueError(
        f"refinement-generation-exhausted after {used} attempts; "
        f"prefix={last_prefix!r}; seed={seed}"
    )


def require_contract(condition, context):
    if condition is not True:
        raise ValueError(context)


def refined_case(domains, seed, attempts, shrinks, check, context="refinement"):
    try:
        values = generate_tuple(domains, seed, attempts)
    except Exception as error:
        raise ValueError(f"{context}: {error}") from error
    try:
        check(values)
    except Exception as original:
        best = values
        budget = shrinks
        for index in range(len(best)):
            value = best[index]
            candidates = domains[index][0](best[:index], 0)
            if isinstance(value, int) and (not isinstance(value, bool)):
                candidates = [0, 1 if value > 0 else -1] + candidates
                reduced = value
                while abs(reduced) > 1:
                    reduced = abs(reduced) // 2 * (1 if reduced > 0 else -1)
                    candidates.insert(2, reduced)
            for candidate in candidates:
                if budget <= 0:
                    break
                budget -= 1
                if complexity(candidate) >= complexity(best[index]):
                    continue
                prefix = best[:index] + [candidate]
                if not domains[index][1](prefix):
                    continue
                try:
                    trial = generate_tuple(domains, seed, min(attempts, 100), prefix)
                except ValueError as error:
                    if str(error).startswith("refinement-generation-exhausted"):
                        continue
                    raise
                try:
                    check(trial)
                except Exception:
                    best = trial
        raise AssertionError(
            f"{context}: {original}; refined counterexample={best!r}; seed={seed}"
        ) from original


def complexity(value):
    if isinstance(value, Presence):
        return 1 + complexity(value.value) if value.present else 0
    if isinstance(value, Raw):
        return len(value.units)
    if isinstance(value, (str, bytes)):
        return len(value)
    if isinstance(value, Absence):
        return 0
    if isinstance(value, Symbol):
        return 1
    if isinstance(value, (Fraction, Decimal)):
        r = ratio(value)
        return abs(r.numerator) + r.denominator - 1
    if isinstance(value, complex):
        return complexity(value.real) + complexity(value.imag)
    if isinstance(value, float):
        return struct.unpack(">Q", struct.pack(">d", abs(value)))[0]
    return abs(value)


def construct(tag, fields):
    if (
        tag == "Maybe::Absent"
        and not fields
        or (
            tag in ("Maybe::Present", "Either::Left", "Either::Right")
            and len(fields) == 1
        )
    ):
        return DataValue(tag, fields)
    if tag == "List::Nil" and (not fields):
        return []
    if tag == "List::Cons" and len(fields) == 2 and isinstance(fields[1], list):
        return [fields[0], *fields[1]]
    raise ValueError("invalid constructor or arity: " + tag)


def all_elements(value, predicate):
    if not isinstance(value, list):
        raise TypeError("expected List in element predicate")
    for index, item in enumerate(value):
        try:
            accepted = predicate(item)
            if type(accepted) is not bool:
                raise TypeError("element predicate must return Bool")
            if not accepted:
                return False
        except Exception as error:
            raise ValueError(f"List element {index}: {error}") from error
    return True


def match_list(value, branches):
    if not isinstance(value, list):
        raise TypeError("expected List in match")
    tag = "List::Cons" if value else "List::Nil"
    for candidate, branch in branches:
        if candidate == tag:
            return branch(value[0], value[1:]) if value else branch()
    raise ValueError("missing match branch: " + tag)


def match_value(value, branches):
    if isinstance(value, list):
        return match_list(value, branches)
    if not isinstance(value, DataValue):
        raise TypeError("expected data in match")
    arity = {
        "Maybe::Absent": 0,
        "Maybe::Present": 1,
        "Either::Left": 1,
        "Either::Right": 1,
    }.get(value.tag, -1)
    if len(value.fields) != arity:
        raise ValueError("invalid match constructor or arity")
    for tag, branch in branches:
        if tag == value.tag:
            return branch(*value.fields)
    raise ValueError("missing match branch: " + value.tag)


def either_arguments(type_key):
    """Decode a parenthesized, two-argument runtime type key."""
    args = []
    depth = 0
    start = 0
    for index in range(7, len(type_key)):
        char = type_key[index]
        if char == "(":
            if depth == 0:
                start = index + 1
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("invalid Either type key")
            if depth == 0:
                args.append(type_key[start:index])
        elif depth == 0 and char != " ":
            raise ValueError("invalid Either type key")
    if depth != 0 or len(args) != 2 or any((not arg.strip() for arg in args)):
        raise ValueError("invalid Either type key")
    return args


def integer_literal(value):
    """Decode an arbitrary integer without a host-sized intermediate."""
    return int(value)


def bytes_literal(values):
    """Construct octets without text encoding or replacement."""
    return bytes(values)


# Workflow runtime. A workflow runs under a runtime: a clock, a seeded
# random source, a trace of what happened, and the state of stateful stages.
# The runtime travels in the symbols context every generated function takes;
# without one, the default runtime applies (real time, unless the tests
# installed a virtual clock). Durations are whole microseconds.

import time as _time

_WORKFLOW = "\x00lawspec.workflow"
_MASK64 = (1 << 64) - 1


class RealClock:
    virtual = False

    def now(self):
        return _time.monotonic_ns() // 1000

    def sleep(self, micros):
        _time.sleep(micros / 1_000_000)


class VirtualClock:
    """Sleeping advances the clock and returns at once."""

    virtual = True

    def __init__(self, start=0):
        self.time = start

    def now(self):
        return self.time

    def sleep(self, micros):
        self.time += micros


class SplitMix64:
    """The same sequence on every target for the same seed.

    SplitMix64 is small, fast and specified exactly, so every runtime
    implements the same generator and a seed names the same case everywhere.
    ref:splitmix ref:DEC-portable-seeded-generation"""

    def __init__(self, seed=0):
        self.state = seed & _MASK64

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return z ^ (z >> 31)

    def below(self, bound):
        """Uniform in [0, bound); 0 when bound is 0."""
        return self.next() % bound if bound > 0 else 0


class WorkflowRuntime:
    """gates: whether rate limits, breakers, bulkheads and caches apply. The
    runtime generated tests install has them off: a workflow law calls the
    workflow and its composition, which would see each other's state."""

    def __init__(self, clock=None, seed=0, gates=True):
        self.clock = RealClock() if clock is None else clock
        self.random = SplitMix64(seed)
        self.trace = []
        self.state = {}
        self.gates = gates
        # A frame per running workflow: the undos of its completed stages.
        self.frames = []
        # When the running attempt of a stage with a timeout must end
        # (time.monotonic seconds), or None.
        self.deadline = None
        # The running attempt's stage and hedge (delay, most), or None.
        self.hedge = None

    def context(self, symbols=None):
        """A symbols context that runs workflows under this runtime."""
        symbols = {} if symbols is None else symbols
        symbols[_WORKFLOW] = self
        return symbols


_default_runtime = [None]
_CLOCK_KEY = "lawspec.time::ability::Clock"
_CLOCK_VIEW = "\x00lawspec.workflow.clock"


_CLOCK_READER = [None]


def register_clock_ability(now, sleep, real_time):
    """How the runtime reads a Clock handler, registered by lawspec.time's
    register_clock (the generated tests call it): now(handler) gives
    microseconds, sleep(handler, micros) waits, and real_time(handler) says
    whether the handler is the default real clock."""
    _CLOCK_READER[0] = (now, sleep, real_time)


def _ability_clock(handler):
    reader = _CLOCK_READER[0]
    return None if reader is None or handler is None else AbilityClock(handler, reader)


class AbilityClock:
    """A workflow runtime's clock read through the Clock ability: the handler
    a law installs (the virtual clock, or the default real one). Every handler
    but the default real clock is virtual: waits pass at once, and timeouts
    and hedges count only the time it reports."""

    def __init__(self, handler, reader):
        self.handler = handler
        self._now, self._sleep, real_time = reader
        self.virtual = not real_time(handler)

    def now(self):
        return self._now(self.handler)

    def sleep(self, micros):
        self._sleep(self.handler, micros)


def use_virtual_clock(seed=0):
    """Make the default runtime virtual, as generated tests do. Timeouts and
    hedges stay on: they count virtual time (see _scoped)."""
    _default_runtime[0] = WorkflowRuntime(VirtualClock(), seed, gates=False)


def workflow_runtime(symbols):
    runtime = symbols.get(_WORKFLOW) if isinstance(symbols, dict) else None
    if runtime is not None:
        return runtime
    if _default_runtime[0] is None:
        _default_runtime[0] = WorkflowRuntime()
    runtime = _default_runtime[0]
    # Workflow time is the Clock ability's: where a law has installed a
    # Clock handler, the default runtime waits and times out on it.
    table = symbols.get(_HANDLERS) if isinstance(symbols, dict) else None
    clock = _ability_clock(table.get(_CLOCK_KEY) if table else None)
    if clock is None:
        return runtime
    view = symbols.get(_CLOCK_VIEW)
    if view is None or view[0] is not clock.handler or view[1] is not runtime:
        under = WorkflowRuntime.__new__(WorkflowRuntime)
        under.__dict__.update(runtime.__dict__)
        under.clock = clock
        view = (clock.handler, runtime, under)
        symbols[_CLOCK_VIEW] = view
    return view[2]


@dataclass(frozen=True)
class Retry:
    # ('immediate',), ('fixed', d), ('linear', d, step),
    # ('exponential', d, factor, cap or None), ('fibonacci', d) or
    # ('custom', decide), where decide(attempt, error, previous) returns a
    # delay in microseconds or None to stop.
    strategy: tuple
    attempts: int
    jitter: str
    when: object = None


@dataclass(frozen=True)
class Gate:
    """A stateful policy: start(now) gives its state, admit(state, now) a
    Step of the next state and a Gate (Admit, WaitFor or Reject), and
    finish(state, now, succeeded) the state after the call. wait is None to
    fail at once when not admitted, or the most it waits (-1 for no bound)."""

    kind: str
    start: object
    admit: object
    finish: object
    wait: object = None


@dataclass(frozen=True)
class StagePolicy:
    stage: str
    retry: object = None
    timeout: object = None
    # The stage's state key, its gates (breaker, limit, bulkhead), how long
    # a success is cached (or None), and whether failures are StageFailures.
    key: str = ""
    gates: tuple = ()
    cache: object = None
    wraps: bool = False
    # Undoes the stage's success value when its workflow fails.
    compensate: object = None
    # (delay, most): when an attempt has not succeeded after delay, another
    # starts beside it, up to most in all; the first success wins.
    hedge: object = None


def _fibonacci(n):
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a


def retry_delay(strategy, attempt):
    """The delay before attempt (2 or more), before jitter."""
    kind, n = strategy[0], attempt - 1
    if kind == "immediate":
        return 0
    if kind == "fixed":
        return strategy[1]
    if kind == "linear":
        return strategy[1] + strategy[2] * (n - 1)
    if kind == "exponential":
        delay = strategy[1] * strategy[2] ** (n - 1)
        return delay if strategy[3] is None else min(delay, strategy[3])
    if kind == "fibonacci":
        return strategy[1] * _fibonacci(n)
    raise ValueError("unknown retry strategy: " + kind)


def jittered(jitter, delay, previous, base, random):
    """Full: [0, delay]; equal: delay/2 + [0, delay/2]; decorrelated:
    [base, previous * 3], capped at delay."""
    if jitter == "full":
        return random.below(delay + 1)
    if jitter == "equal":
        half = delay // 2
        return half + random.below(delay - half + 1)
    if jitter == "decorrelated":
        high = max(base, previous * 3)
        return min(delay, base + random.below(high - base + 1))
    return delay


_STAGE_FAILURE = "lawspec.resilience::type::StageFailure::"
_GATE = "lawspec.resilience::type::Gate::"


def stage_failure(kind):
    return DataValue("Either::Left", (DataValue(_STAGE_FAILURE + kind, ()),))


def _pass_gate(runtime, policy, gate):
    """Admits the call or returns the failure to give instead."""
    key = policy.key + "/" + gate.kind
    waited = 0
    while True:
        now = runtime.clock.now()
        state = runtime.state.get(key)
        if state is None:
            state = gate.start(now)
        step = gate.admit(state, now)
        runtime.state[key] = step.fields[0]
        decision = step.fields[1]
        if decision.tag == _GATE + "Admit":
            return None
        failure = {
            "breaker": "CircuitOpen",
            "limit": "RateLimited",
            "bulkhead": "Saturated",
        }[gate.kind]
        if decision.tag == _GATE + "Reject" or gate.wait is None:
            return failure
        delay = decision.fields[0]
        if gate.wait >= 0 and waited + delay > gate.wait:
            return failure
        runtime.trace.append(("wait", policy.stage, delay))
        runtime.clock.sleep(delay)
        waited += delay


def run_stage(symbols, policy, attempt, key=None):
    """Runs a stage's attempts under its policy; a Left is a failure.
    attempt() returns the stage's Either; key is its input, for the cache."""
    runtime = workflow_runtime(symbols)
    gates = policy.gates if runtime.gates else ()
    cached = (
        runtime.state.setdefault(policy.key + "/cache", [])
        if policy.cache is not None and runtime.gates
        else None
    )
    if cached is not None:
        now = runtime.clock.now()
        for entry_key, value, expires in cached:
            if now < expires and equal_values(entry_key, key):
                runtime.trace.append(("cached", policy.stage, 0))
                return value
    for gate in gates:
        failure = _pass_gate(runtime, policy, gate)
        if failure is not None:
            for passed in gates[: gates.index(gate)]:
                _finish(runtime, policy, passed, False)
            return stage_failure(failure)
    result = _attempts(runtime, policy, attempt)
    succeeded = not (isinstance(result, DataValue) and result.tag == "Either::Left")
    for gate in gates:
        _finish(runtime, policy, gate, succeeded)
    if succeeded and policy.compensate is not None and runtime.frames:
        value = result.fields[0]
        runtime.frames[-1].append((policy.stage, lambda: policy.compensate(value)))
    if cached is not None and succeeded:
        cached[:] = [entry for entry in cached if not equal_values(entry[0], key)]
        cached.append((key, result, runtime.clock.now() + policy.cache))
    return result


def run_workflow(symbols, attempt):
    """Runs a workflow whose stages compensate: when it fails, the undos of
    its completed stages run, last first."""
    runtime = workflow_runtime(symbols)
    frame = []
    runtime.frames.append(frame)
    try:
        result = attempt()
    finally:
        runtime.frames.pop()
    if isinstance(result, DataValue) and result.tag == "Either::Left":
        for stage, undo in reversed(frame):
            runtime.trace.append(("compensate", stage, 0))
            undo()
    return result


def _finish(runtime, policy, gate, succeeded):
    if gate.finish is not None:
        key = policy.key + "/" + gate.kind
        runtime.state[key] = gate.finish(
            runtime.state[key], runtime.clock.now(), succeeded
        )


def equal_values(a, b):
    return a == b


class StageTimedOut(Exception):
    """An attempt outlived its stage's timeout."""


def await_step(symbols, start, convert):
    """An asynchronous step's logical result: start() begins the step and
    convert turns its native result into a logical value. The step runs
    within its stage's timeout and hedge, if any."""
    runtime = workflow_runtime(symbols)
    deadline, hedge = runtime.deadline, runtime.hedge
    if deadline is None and hedge is None:
        return convert(await_task(start()))
    if isinstance(hedge, _VirtualHedge):
        return _virtual_hedge(runtime, hedge, start, convert)
    import asyncio

    def left():
        return None if deadline is None else max(0.0, deadline - _time.monotonic())

    async def within():
        return convert(await asyncio.wait_for(start(), left()))

    async def hedged():
        (stage, (delay, most)) = hedge
        pending, started, last = set(), 0, None

        def launch():
            nonlocal started
            started += 1
            if started > 1:
                runtime.trace.append(("hedge", stage, started))
            pending.add(asyncio.ensure_future(start()))

        launch()
        try:
            while True:
                wait = delay / 1_000_000 if started < most else None
                if deadline is not None:
                    wait = left() if wait is None else min(wait, left())
                done, _ = await asyncio.wait(
                    pending, timeout=wait, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    pending.discard(task)
                    value = convert(task.result())
                    if not (
                        isinstance(value, DataValue) and value.tag == "Either::Left"
                    ):
                        return value
                    last = value
                if deadline is not None and _time.monotonic() >= deadline:
                    raise asyncio.TimeoutError()
                if pending and done:
                    continue
                if not pending and started >= most:
                    return last
                launch()
        finally:
            for task in pending:
                task.cancel()

    try:
        return _run_blocking(within() if hedge is None else hedged())
    except asyncio.TimeoutError:
        raise StageTimedOut() from None


class NativeAsync:
    """The Async ability's default handler: the interpreter's own threads and
    asyncio. LawSpec code performs pause; workflows reach the rest natively:
    spawn starts a function as a task, wait gives a task's result (a
    coroutine is run to completion), and all runs functions side by side and
    gives their results in order. An `async` adapter is an adapter that uses
    Async, so its coroutine is awaited with wait."""

    def pause(self):
        _time.sleep(0)

    def spawn(self, fn):
        import concurrent.futures

        future = concurrent.futures.Future()

        def run():
            try:
                future.set_result(fn())
            except BaseException as error:  # given back by wait
                future.set_exception(error)

        threading.Thread(target=run, daemon=True).start()
        return future

    def wait(self, task):
        if (
            hasattr(task, "result")
            and hasattr(task, "done")
            and not hasattr(task, "__await__")
        ):
            return task.result()
        return await_task(task)

    def all(self, fns):
        """Every function's result, in order; all finish before the first
        exception (in order) is raised."""
        tasks = [self.spawn(fn) for fn in fns]
        import concurrent.futures

        concurrent.futures.wait(tasks)
        return [task.result() for task in tasks]


ASYNC = NativeAsync()


def concurrently(steps):
    """An all group's step results, in declaration order. The steps run side
    by side as tasks of the Async ability's default handler, so an
    asynchronous step waits only for itself. Every step finishes before a
    step's exception (the first, in declaration order) is raised."""
    return ASYNC.all(steps)


@dataclass(frozen=True)
class _VirtualHedge:
    stage: str
    delay: int
    most: int


def _is_left(value):
    return isinstance(value, DataValue) and value.tag == "Either::Left"


def _virtual_hedge(runtime, hedge, start, convert):
    """A hedge on a virtual clock: attempts run one after another, and the
    next starts when one fails, so the first success wins as it would in
    real time when no attempt outlives the delay."""
    started = 1
    value = convert(await_task(start()))
    while _is_left(value) and started < hedge.most:
        started += 1
        runtime.trace.append(("hedge", hedge.stage, started))
        value = convert(await_task(start()))
    return value


def _scoped(runtime, policy, attempt):
    """An attempt under its stage's timeout (failing with TimedOut when it
    outlives it) and hedge: the Timeout and Hedge transformers of the Async
    ability, measured on the runtime's Clock. On a virtual clock (generated
    tests, or a law using virtual clock) an attempt takes the virtual time
    that passes while it runs, so both are deterministic."""
    timeout = policy.timeout is not None and policy.timeout > 0
    if not timeout and policy.hedge is None:
        return attempt()
    if getattr(runtime.clock, "virtual", False):
        return _virtually_scoped(runtime, policy, attempt, timeout)
    if not runtime.gates:
        return attempt()
    outer = (runtime.deadline, runtime.hedge)
    if timeout:
        runtime.deadline = _time.monotonic() + policy.timeout / 1_000_000
    runtime.hedge = None if policy.hedge is None else (policy.stage, policy.hedge)
    try:
        return attempt()
    except StageTimedOut:
        return stage_failure("TimedOut")
    finally:
        (runtime.deadline, runtime.hedge) = outer


def _virtually_scoped(runtime, policy, attempt, timeout):
    outer = (runtime.deadline, runtime.hedge)
    began = runtime.clock.now()
    runtime.deadline = None
    runtime.hedge = (
        None if policy.hedge is None else _VirtualHedge(policy.stage, *policy.hedge)
    )
    try:
        result = attempt()
    except StageTimedOut:
        return stage_failure("TimedOut")
    finally:
        (runtime.deadline, runtime.hedge) = outer
    if timeout and runtime.clock.now() - began > policy.timeout:
        return stage_failure("TimedOut")
    return result


def _attempts(runtime, policy, attempt):
    retry = policy.retry
    number, previous = 1, 0
    while True:
        runtime.trace.append(("start", policy.stage, number))
        result = _scoped(runtime, policy, attempt)
        failed = isinstance(result, DataValue) and result.tag == "Either::Left"
        runtime.trace.append(("finish", policy.stage, number, not failed))
        if not failed or retry is None:
            return result
        if retry.attempts > 0 and number >= retry.attempts:
            return result
        error = result.fields[0]
        if policy.wraps:
            # Only the step's own failures and timeouts are retried.
            if error.tag == _STAGE_FAILURE + "StepFailed":
                error = error.fields[0]
            elif error.tag != _STAGE_FAILURE + "TimedOut":
                return result
        if retry.when is not None and not retry.when(error):
            return result
        number += 1
        if retry.strategy[0] == "custom":
            delay = retry.strategy[1](number, error, previous)
            if delay is None:
                return result
        else:
            base = (
                retry_delay(retry.strategy, 2)
                if retry.strategy[0] != "immediate"
                else 0
            )
            delay = jittered(
                retry.jitter,
                retry_delay(retry.strategy, number),
                previous,
                base,
                runtime.random,
            )
        runtime.trace.append(("sleep", policy.stage, delay))
        runtime.clock.sleep(delay)
        previous = delay


def retry_decision(decision):
    """A RetryDecision's delay in microseconds, or None to stop."""
    if decision.tag != "lawspec.time::type::RetryDecision::RetryAfter":
        return None
    return decision.fields[0].fields[0]


# Portable generation for stateful models. A type descriptor is an
# s-expression: (int T lo hi) with _ for no bound, (bool), (text), (unit),
# (list D), (maybe D), (either L R), (data NAME (ctor TAG D...) ...) and
# (ref NAME) for a data type declared in the model's table. Every target
# generates, shrinks and renders the same values for the same seed.


class Quoted(str):
    """A string atom of a descriptor."""


def read_descriptor(text):
    """Parses s-expressions: lists, integers, strings, symbols and _ (None)."""
    position = [0]

    def skip():
        while position[0] < len(text) and text[position[0]] in " \t\r\n":
            position[0] += 1

    def item():
        skip()
        c = text[position[0]]
        if c == "(":
            position[0] += 1
            items = []
            skip()
            while text[position[0]] != ")":
                items.append(item())
                skip()
            position[0] += 1
            return items
        if c == '"':
            position[0] += 1
            out = []
            while text[position[0]] != '"':
                if text[position[0]] == "\\":
                    position[0] += 1
                out.append(text[position[0]])
                position[0] += 1
            position[0] += 1
            return Quoted("".join(out))
        start = position[0]
        while position[0] < len(text) and text[position[0]] not in " \t\r\n()":
            position[0] += 1
        atom = text[start : position[0]]
        if atom == "_":
            return None
        if atom.lstrip("-").isdigit():
            return int(atom)
        return atom

    items = []
    skip()
    while position[0] < len(text):
        items.append(item())
        skip()
    return items


_UNBOUNDED = 1_000_000


class Values:
    """Generation, shrinking and rendering over a table of data types.

    Shrinking stays inside the declared domain, so a reported counterexample
    is always a value the law quantifies over. ref:DEC-shrink-within-domain
    ref:DEC-structural-size-budget"""

    def __init__(self, table):
        self.table = table

    def resolve(self, d):
        return self.table[d[1]] if d[0] == "ref" else d

    def bounds(self, d):
        """An integer's range: a missing bound is 1,000,000 from zero, or
        2,000,000 from the other bound when that is beyond it."""
        lo, hi = d[2], d[3]
        if lo is None and hi is None:
            return -_UNBOUNDED, _UNBOUNDED
        if lo is None:
            return min(-_UNBOUNDED, hi - 2 * _UNBOUNDED), hi
        if hi is None:
            return lo, max(_UNBOUNDED, lo + 2 * _UNBOUNDED)
        return lo, hi

    def base(self, d):
        """The constructors whose fields mention no data type."""
        found = [c for c in d[2:] if not any(_mentions_data(f) for f in c[2:])]
        return found or d[2:]

    def generate(self, d, random, size):
        d = self.resolve(d)
        kind = d[0]
        if kind == "int":
            lo, hi = self.bounds(d)
            if random.below(10) < 2:
                specials = [lo, hi, min(max(0, lo), hi), min(max(1, lo), hi)]
                return specials[random.below(4)]
            return lo + random.below(hi - lo + 1)
        if kind == "bool":
            return random.below(2) == 1
        if kind == "text":
            return "".join(
                chr(32 + random.below(95)) for _ in range(random.below(size + 1))
            )
        if kind == "unit":
            return UNIT
        if kind == "list":
            return [
                self.generate(d[1], random, size) for _ in range(random.below(size + 1))
            ]
        if kind == "maybe":
            if random.below(4) == 0:
                return DataValue("Maybe::Absent", ())
            return DataValue("Maybe::Present", (self.generate(d[1], random, size),))
        if kind == "either":
            if random.below(2) == 0:
                return DataValue("Either::Left", (self.generate(d[1], random, size),))
            return DataValue("Either::Right", (self.generate(d[2], random, size),))
        if kind == "data":
            choices = self.base(d) if size <= 0 else d[2:]
            ctor = choices[random.below(len(choices))]
            return DataValue(
                str(ctor[1]),
                tuple(self.generate(f, random, max(size - 1, 0)) for f in ctor[2:]),
            )
        raise ValueError("unknown descriptor " + str(d))

    def minimal(self, d):
        d = self.resolve(d)
        kind = d[0]
        if kind == "int":
            lo, hi = self.bounds(d)
            return min(max(0, lo), hi)
        if kind == "bool":
            return False
        if kind == "text":
            return ""
        if kind == "unit":
            return UNIT
        if kind == "list":
            return []
        if kind == "maybe":
            return DataValue("Maybe::Absent", ())
        if kind == "either":
            return DataValue("Either::Left", (self.minimal(d[1]),))
        ctor = self.base(d)[0]
        return DataValue(str(ctor[1]), tuple(self.minimal(f) for f in ctor[2:]))

    def shrink(self, d, v):
        """Smaller candidates for v, most aggressive first."""
        d = self.resolve(d)
        kind = d[0]
        out = []
        if kind == "int":
            target = self.minimal(d)
            if v != target:
                out = [
                    target,
                    v - _toward_zero(v - target, 2),
                    v - (1 if v > target else -1),
                ]
        elif kind == "bool":
            out = [False] if v else []
        elif kind == "text":
            if v:
                out = (
                    [""]
                    + [v[: len(v) // 2]]
                    + [v[:i] + v[i + 1 :] for i in range(len(v))]
                )
        elif kind == "list":
            if v:
                out = (
                    [[]]
                    + [v[: len(v) // 2]]
                    + [v[:i] + v[i + 1 :] for i in range(len(v))]
                )
                for i, item in enumerate(v):
                    out += [v[:i] + [c] + v[i + 1 :] for c in self.shrink(d[1], item)]
        elif kind == "maybe":
            if v.tag == "Maybe::Present":
                out = [DataValue("Maybe::Absent", ())] + [
                    DataValue("Maybe::Present", (c,))
                    for c in self.shrink(d[1], v.fields[0])
                ]
        elif kind == "either":
            inner = d[1] if v.tag == "Either::Left" else d[2]
            out = [DataValue(v.tag, (c,)) for c in self.shrink(inner, v.fields[0])]
        elif kind == "data":
            ctor = next(c for c in d[2:] if str(c[1]) == v.tag)
            out = [self.minimal(d)]
            # A field of the same type is a smaller value of it.
            out += [f for f, fd in zip(v.fields, ctor[2:]) if fd == ["ref", d[1]]]
            for i, (field, fd) in enumerate(zip(v.fields, ctor[2:])):
                out += [
                    DataValue(v.tag, v.fields[:i] + (c,) + v.fields[i + 1 :])
                    for c in self.shrink(fd, field)
                ]
        seen, unique = [], []
        for c in out:
            if not _same(c, v) and not any(_same(c, s) for s in seen):
                seen.append(c)
                unique.append(c)
        return unique


def _toward_zero(n, d):
    q = abs(n) // d
    return q if n >= 0 else -q


def _mentions_data(d):
    return isinstance(d, list) and (
        d[0] in ("ref", "data") or any(_mentions_data(x) for x in d[1:])
    )


def _same(a, b):
    return render(a) == render(b)


def render(v):
    """A value's canonical text, the same on every target."""
    if is_handle(v):
        return handle_label(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if v is UNIT:
        return "()"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(render(x) for x in v) + "]"
    if isinstance(v, DataValue):
        name = v.tag.split("::")[-1]
        return (
            name
            if not v.fields
            else name + "(" + ", ".join(render(x) for x in v.fields) + ")"
        )
    return str(v)


def difference(actual, expected):
    """Where a structured actual value first differs from the expected one,
    by the portable rendering: '' when they agree or neither has parts."""
    path = []
    a, b = actual, expected
    while True:
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            for i, (x, y) in enumerate(zip(a, b)):
                if render(x) != render(y):
                    path.append(f"item {i + 1}")
                    a, b = x, y
                    break
            else:
                if len(a) == len(b):
                    return ""
                path.append("length")
                a, b = len(a), len(b)
                break
        elif (
            isinstance(a, DataValue)
            and isinstance(b, DataValue)
            and a.tag == b.tag
            and len(a.fields) == len(b.fields)
        ):
            name = a.tag.split("::")[-1]
            for i, (x, y) in enumerate(zip(a.fields, b.fields)):
                if render(x) != render(y):
                    path.append(f"field {i + 1} of {name}")
                    a, b = x, y
                    break
            else:
                return ""
        else:
            break
    if not path:
        return ""
    shown = (lambda v: str(v)) if path[-1] == "length" else render
    return (
        f" | first difference at {', '.join(path)}: expected "
        f"{shown(b)}, actual {shown(a)}"
    )


def values_from(text):
    """A descriptor text's data types and its last form, the one generated."""
    forms = read_descriptor(text)
    table = {str(f[1]): f for f in forms if isinstance(f, list) and f[0] == "data"}
    return Values(table), forms[-1]


def generated(text, seed, size, count):
    """count values generated from one SplitMix64 seed, rendered."""
    values, d = values_from(text)
    random = SplitMix64(seed)
    return [render(values.generate(d, random, size)) for _ in range(count)]


def shrunk(text, seed, size):
    """The shrink candidates of the first value generated, rendered."""
    values, d = values_from(text)
    return [
        render(c) for c in values.shrink(d, values.generate(d, SplitMix64(seed), size))
    ]


# Actors. An actor owns a state and handles one message at a time, in the
# order they arrive. It is not a thread: a message sent to an idle actor
# starts a short-lived worker that drains its mailbox and then stops, so an
# idle actor costs only its state and queue.


class ActorStopped(Exception):
    """A message sent to an actor that has stopped."""


class ActorCrashed(Exception):
    """A handler failed, so the actor crashed; the cause is the handler's
    exception. A supervised actor restarts; any other stops."""

    def __init__(self, cause):
        super().__init__(f"the actor crashed: {cause!r}")
        self.cause = cause


class _Restart(Exception):
    """A message that crashes the actor on purpose (crash, links)."""

    def __init__(self, cause, origin):
        super().__init__(cause)
        self.cause, self.origin = cause, origin


_crash_ids = iter(range(1, 1 << 62))
_crash_lock = threading.Lock()


def _next_crash():
    with _crash_lock:
        return next(_crash_ids)


class Actor:
    """An actor: a state, a mailbox, and one message handled at a time.

    restart(last state) gives the state after a crash; without it, a crash
    stops the actor even under a supervisor. A supervised actor restarts in
    place: it keeps its address and the messages waiting for it."""

    def __init__(self, state, restart=None):
        self._state = state
        self._restart_state = restart
        self._mailbox = deque()
        self._lock = threading.Lock()
        self._draining = False
        self._stopped = False
        self._supervisor = None
        self._monitors = []
        self._links = []
        self._seen = set()

    def _post(self, message):
        with self._lock:
            if self._stopped:
                raise ActorStopped("the actor has stopped")
            self._mailbox.append(message)
            if self._draining:
                return
            self._draining = True
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        while True:
            with self._lock:
                if not self._mailbox:
                    self._draining = False
                    return
                handler, reply = self._mailbox.popleft()
            try:
                result, self._state = handler(self._state)
                outcome = (True, result)
            except _Restart as crash:
                outcome = (True, None)
                self._crashed(crash.cause, crash.origin)
            except BaseException as error:  # noqa: BLE001 - the actor crashes
                outcome = (False, ActorCrashed(error))
                self._crashed(error, _next_crash())
            if reply is not None:
                reply.append(outcome)
                reply.event.set()

    def _crashed(self, cause, origin):
        """Runs on the actor's turn: restart or stop, then tell monitors and
        links. origin names the first crash, so a crash crosses each link once."""
        self._seen.add(origin)
        restarted = (
            self._supervisor is not None
            and self._restart_state is not None
            and self._supervisor._child_crashed(self, cause)
        )
        if not restarted:
            self._halt()
        for monitor in list(self._monitors):
            monitor(("crashed", cause))
        for other in list(self._links):
            other._link_crash(cause, origin)

    def _restart_now(self):
        """On the actor's turn: the restarted state from the last one."""
        self._state = self._restart_state(self._state)

    def _restart_later(self):
        """A restart a supervisor asks of a sibling, in mailbox order."""

        def restart(s):
            return None, self._restart_state(s)

        try:
            self._post((restart, None))
        except ActorStopped:
            pass

    def _link_crash(self, cause, origin):
        if origin in self._seen:
            return

        def crash(s):
            # The same crash can arrive by two links before either is handled.
            if origin in self._seen:
                return None, s
            raise _Restart(cause, origin)

        try:
            self._post((crash, None))
        except ActorStopped:
            pass

    def _halt(self):
        """Stops the actor; messages still waiting fail with ActorStopped."""
        with self._lock:
            self._stopped = True
            waiting, self._mailbox = list(self._mailbox), deque()
        for _, reply in waiting:
            if reply is not None:
                reply.append((False, ActorStopped("the actor has stopped")))
                reply.event.set()

    def call(self, handler):
        """Runs handler(state) -> (result, next state) in turn and returns
        the result. A handler that raises crashes the actor, and call raises
        ActorCrashed."""
        reply = _Reply()
        self._post((handler, reply))
        reply.event.wait()
        ok, value = reply[0]
        if not ok:
            raise value
        return value

    def cast(self, handler):
        """Queues handler(state) -> (result, next state) without waiting."""
        self._post((handler, None))

    def crash(self, cause="crashed on purpose"):
        """Crashes the actor once the messages before this one are handled,
        as a failing handler would: for testing supervision."""
        origin = _next_crash()

        def crash(s):
            raise _Restart(cause, origin)

        self.call(crash)

    def restart(self, restart):
        """Replaces the state by restart(last state) between messages, as a
        supervised restart does (crash injection in model runs)."""
        self.call(lambda s: (None, restart(s)))

    def state(self):
        """The state after every message sent before this call."""
        return self.call(lambda s: (s, s))

    def monitor(self, notify):
        """notify(('crashed', cause)) after each crash, and
        notify(('stopped', None)) once it stops."""
        self._monitors.append(notify)

    def link(self, other):
        """Links two actors: when either crashes, the other crashes too."""
        self._links.append(other)
        other._links.append(self)

    def stop(self):
        """Refuses further messages; those already queued still run. A
        permanent child of a supervisor restarts instead."""
        if self._supervisor is not None and self._supervisor._child_stopped(self):
            return
        with self._lock:
            already = self._stopped
            self._stopped = True
        if not already:
            for monitor in list(self._monitors):
                monitor(("stopped", None))


class Supervisor:
    """Starts children (actors or supervisors) and restarts them after a
    crash. strategy 'one_for_one' restarts the child that crashed,
    'one_for_all' every child, 'rest_for_one' it and those added after it.
    A child's lifetime: 'permanent' restarts after a crash or a stop,
    'transient' only after a crash, 'temporary' never. More than
    max_restarts within period seconds is the supervisor's own crash: its
    supervisor restarts all of its children, or, at the top, every child
    stops.

    Supervision follows OTP's strategies and restart types, so a supervision
    tree means what an Erlang programmer expects on every target.
    ref:DEC-actors-otp-supervision ref:erlang-otp-supervisors"""

    def __init__(self, strategy="one_for_one", max_restarts=3, period=5.0):
        if strategy not in ("one_for_one", "one_for_all", "rest_for_one"):
            raise ValueError(f"unknown strategy {strategy!r}")
        self.strategy, self.max_restarts, self.period = strategy, max_restarts, period
        self._children = []
        self._restarts = deque()
        self._lock = threading.RLock()
        self._supervisor = None
        self._stopped = False
        self._monitors = []

    def supervise(self, child, lifetime="permanent"):
        """Adds a started child, and returns it."""
        if lifetime not in ("permanent", "transient", "temporary"):
            raise ValueError(f"unknown lifetime {lifetime!r}")
        with self._lock:
            child._supervisor = self
            self._children.append([child, lifetime])
        return child

    def children(self):
        return [c for c, _ in self._children]

    def _allow_restart(self):
        import time

        now = time.monotonic()
        while self._restarts and now - self._restarts[0] > self.period:
            self._restarts.popleft()
        if len(self._restarts) >= self.max_restarts:
            return False
        self._restarts.append(now)
        return True

    def _entry(self, child):
        return next((e for e in self._children if e[0] is child), None)

    def _restarting(self, entry, cause, crashed):
        """Under the lock: the children to restart for entry's crash, or
        None when the supervisor gives up."""
        if self._allow_restart():
            index = self._children.index(entry)
            return {
                "one_for_one": [entry],
                "one_for_all": list(self._children),
                "rest_for_one": self._children[index:],
            }[self.strategy]
        parent = self._supervisor
        if parent is not None and parent._child_failed(self, cause):
            self._restarts.clear()
            return list(self._children)
        self._fail(crashed, cause)
        return None

    def _child_crashed(self, child, cause):
        """On child's turn: True when it restarts now."""
        with self._lock:
            entry = self._entry(child)
            if self._stopped or entry is None:
                return False
            if entry[1] == "temporary":
                self._children.remove(entry)
                return False
            group = self._restarting(entry, cause, child)
            if group is None:
                return False
        for other, _ in group:
            if other is not child:
                other._restart_later()
        child._restart_now()
        return True

    def _child_failed(self, child, cause):
        """A child supervisor gave up: True when it may restart its children."""
        with self._lock:
            entry = self._entry(child)
            if self._stopped or entry is None:
                return False
            if entry[1] == "temporary":
                self._children.remove(entry)
                return False
            group = self._restarting(entry, cause, child)
            if group is None:
                return False
        for other, _ in group:
            if other is not child:
                other._restart_later()
        return True

    def _child_stopped(self, child):
        """True when a stopped child is permanent and restarts instead."""
        with self._lock:
            entry = self._entry(child)
            if entry is None or self._stopped:
                return False
            if entry[1] != "permanent":
                self._children.remove(entry)
                return False
            group = self._restarting(entry, "stopped", child)
            if group is None:
                return False
        for other, _ in group:
            other._restart_later()
        return True

    def _fail(self, crashed, cause):
        """Under the lock: every child but the one crashing (which stops
        itself) stops, and so does the supervisor."""
        children, self._children = self._children, []
        self._stopped = True
        for other, _ in reversed(children):
            other._supervisor = None
            if other is not crashed:
                other._halt()
        for monitor in list(self._monitors):
            monitor(("crashed", cause))

    def _restart_later(self):
        """Restarted by its own supervisor: every child restarts."""
        with self._lock:
            self._restarts.clear()
            children = list(self._children)
        for child, _ in children:
            child._restart_later()

    def _halt(self):
        self.stop()

    def monitor(self, notify):
        """notify(('crashed', cause)) when it gives up, and
        notify(('stopped', None)) once stopped."""
        self._monitors.append(notify)

    def stop(self):
        """Stops every child, last added first, without restarting them."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            children, self._children = list(self._children), []
        for child, _ in reversed(children):
            child._supervisor = None
            child.stop()
        for monitor in list(self._monitors):
            monitor(("stopped", None))


def check_supervision():
    """The runtime's own check of crashes, links, monitors and supervision:
    every strategy, lifetime, the restart limit and escalation. Raises
    AssertionError naming the first behaviour that differs."""
    import time

    def counter():
        return Actor(0, restart=lambda s: 0)

    def bump(a):
        return a.call(lambda s: (s + 1, s + 1))

    def fail(a):
        try:
            a.call(lambda s: 1 // 0)
        except ActorCrashed:
            return
        raise AssertionError("a failing handler did not raise ActorCrashed")

    def stopped(a, what):
        try:
            a.state()
        except ActorStopped:
            return
        raise AssertionError(what + " should have stopped")

    def expect(actual, wanted, what):
        if actual != wanted:
            raise AssertionError(f"{what}: got {actual!r}, expected {wanted!r}")

    a = counter()
    bump(a)
    fail(a)
    stopped(a, "an unsupervised actor that crashed")
    sup = Supervisor("one_for_one")
    x, y = sup.supervise(counter()), sup.supervise(counter())
    bump(x), bump(y), bump(y)
    fail(x)
    expect(
        (x.state(), y.state()), (0, 2), "one for one restarts only the crashed child"
    )
    sup = Supervisor("one_for_all")
    x, y = sup.supervise(counter()), sup.supervise(counter())
    bump(x), bump(y)
    fail(x)
    expect((x.state(), y.state()), (0, 0), "one for all restarts every child")
    sup = Supervisor("rest_for_one")
    x, y, z = [sup.supervise(counter()) for _ in range(3)]
    bump(x), bump(y), bump(z)
    fail(y)
    expect(
        (x.state(), y.state(), z.state()),
        (1, 0, 0),
        "rest for one restarts the child and later ones",
    )
    sup = Supervisor()
    t = sup.supervise(counter(), "temporary")
    fail(t)
    stopped(t, "a temporary child that crashed")
    sup = Supervisor()
    p, q = sup.supervise(counter(), "permanent"), sup.supervise(counter(), "transient")
    bump(p)
    p.stop()
    expect(p.state(), 0, "a permanent child restarts after a stop")
    q.stop()
    stopped(q, "a transient child that was stopped")
    events = []
    sup = Supervisor("one_for_one", max_restarts=2, period=10)
    sup.monitor(events.append)
    x, y = sup.supervise(counter()), sup.supervise(counter())
    fail(x), fail(x), fail(x)
    stopped(y, "a child of a supervisor past its restart limit")
    expect(
        [e[0] for e in events],
        ["crashed"],
        "a supervisor past its limit tells its monitors",
    )
    outer = Supervisor("one_for_one", max_restarts=5, period=10)
    inner = outer.supervise(Supervisor("one_for_one", max_restarts=1, period=10))
    x, y = inner.supervise(counter()), inner.supervise(counter())
    bump(y)
    fail(x), fail(x)
    expect(
        (x.state(), y.state()),
        (0, 0),
        "a supervisor past its limit is restarted by its own",
    )
    seen = []
    a, b = counter(), counter()
    a.link(b)
    b.monitor(seen.append)
    fail(a)
    for _ in range(100):
        if seen:
            break
        time.sleep(0.01)
    stopped(b, "an unsupervised actor linked to one that crashed")
    expect([e[0] for e in seen], ["crashed"], "a monitor hears of a crash")
    sup = Supervisor(max_restarts=10)
    a, b, c = [sup.supervise(counter()) for _ in range(3)]
    a.link(b), b.link(c), c.link(a)
    bump(a), bump(b), bump(c)
    fail(a)
    for _ in range(100):
        if len(sup._restarts) >= 3:
            break
        time.sleep(0.01)
    time.sleep(0.05)
    expect(
        (a.state(), b.state(), c.state(), len(sup._restarts)),
        (0, 0, 0, 3),
        "a crash crosses each link once",
    )


class Mailbox:
    """A queue with many senders and one receiver: the channel form of an
    actor. A process that loops over receive() and answers each message is
    an actor written by hand; send() never waits."""

    def __init__(self):
        self._items = deque()
        self._ready = threading.Condition()
        self._closed = False

    def send(self, value):
        with self._ready:
            if self._closed:
                raise ActorStopped("the mailbox is closed")
            self._items.append(value)
            self._ready.notify()

    def receive(self, timeout=None):
        """The next message; waits up to timeout seconds (forever when None)
        and raises TimeoutError, or ActorStopped once closed and empty."""
        with self._ready:
            if not self._ready.wait_for(lambda: self._items or self._closed, timeout):
                raise TimeoutError("no message arrived in time")
            if not self._items:
                raise ActorStopped("the mailbox is closed")
            return self._items.popleft()

    def receive_within(self, micros, clock=None):
        """The Mailbox ability's receive ... within d: the next message, or
        None when none arrives within micros microseconds. On a virtual
        clock (any Clock handler but the default real one, as lawspec.time's
        register_clock tells) it waits no real time:
        it takes a message already sent, or lets the time pass on that clock
        and gives None. Raises ActorStopped once closed and empty."""
        read = _ability_clock(clock)
        if read is not None and read.virtual:
            with self._ready:
                if self._items:
                    return self._items.popleft()
                if self._closed:
                    raise ActorStopped("the mailbox is closed")
            read.sleep(micros)
            return None
        try:
            return self.receive(max(0, micros) / 1_000_000)
        except TimeoutError:
            return None

    def close(self):
        """Refuses further messages; those already sent can still be received."""
        with self._ready:
            self._closed = True
            self._ready.notify_all()


class _Crash:
    """An injected crash of an actor model, as a step with no arguments."""

    name = "crash"
    arguments = []
    state = 0
    unit = True
    when = None
    key = None
    restart = False

    def __init__(self, start_run, start_model, restart):
        self._start_run, self._start_model, self._restart = (
            start_run,
            start_model,
            restart,
        )

    def admits(self, indices):
        return True

    def shifted(self, indices):
        return indices

    def run(self, symbols, actor):
        if self._restart is not None:
            actor.restart(lambda s: self._restart.run(symbols, s))
        else:
            actor.restart(
                lambda s: self._start_run(symbols, *symbols["_lawspec_start"])
            )
        return UNIT

    def reference(self, symbols, state):
        if self._restart is not None:
            return self._restart.reference(symbols, state)
        return self._start_model(symbols, *symbols["_lawspec_start"])


class _Reply(list):
    def __init__(self):
        super().__init__()
        self.event = threading.Event()


def _actor_command(run, unit):
    """A handler bridge (state first, returning Pair result state, or the
    state alone for a Unit result) as a command on an actor."""
    if unit:
        return lambda symbols, actor, *args: actor.call(
            lambda s: (UNIT, run(symbols, s, *args))
        )
    return lambda symbols, actor, *args: actor.call(
        lambda s: tuple(run(symbols, s, *args).fields)
    )


# Stateful models. A model's spec (see LawSpec.MachineSpec) lists its data
# types, start and commands; the callbacks beside it are the generated
# definitions that call the adapters, the references over the model state,
# preconditions, the abstraction and invariants, each taking symbols first.
# A run is generated by simulating the pure model, so every command in it is
# allowed by typestate, its precondition and its reference; it is then
# executed against the adapters and every result, abstracted state and
# invariant is checked. A failing run is shrunk by dropping commands and
# shrinking arguments, replaying the model to keep each candidate valid.


class ModelCommand:
    def __init__(self, form, callbacks):
        fields = {f[0]: f[1:] for f in form[2:]}
        self.name = str(form[1])
        self.arguments = fields["arguments"]
        self.state = fields["state"][0]
        self.unit = fields["unit"][0] == "true"
        self.needs = fields["needs"]
        self.shifts = fields["shifts"]
        # The argument naming the key the command touches, for per-key checks.
        key = fields.get("key", ["none"])[0]
        self.key = None if key == "none" else key
        self.restart = fields.get("restart", ["false"])[0] == "true"
        self.run, self.reference, self.when = callbacks

    def admits(self, indices):
        return all(
            (i >= n[1]) if n[0] == "atleast" else (i == n[1])
            for n, i in zip(self.needs, indices)
        )

    def shifted(self, indices):
        return [
            i + s[1] if s[0] == "by" else s[1] for s, i in zip(self.shifts, indices)
        ]


class Model:
    def __init__(self, spec, start, commands, abstract=None, invariants=()):
        forms = read_descriptor(spec)
        self.name = str(forms[0][1])
        self.shared = forms[0][2] == "shared"
        self.values = Values({str(f[1]): f for f in forms if f[0] == "data"})
        start_form = next(f for f in forms if f[0] == "start")
        start_fields = {f[0]: f[1:] for f in start_form[1:]}
        self.start_indices = start_fields.get("indices", [])
        self.start_arguments = start_fields.get("arguments", [])
        self.start_run, self.start_model = start
        self.commands = [
            ModelCommand(f, c)
            for f, c in zip((f for f in forms if f[0] == "command"), commands)
        ]
        self.abstract = abstract
        kinds = next(f for f in forms if f[0] == "invariants")[1:]
        self.invariants = list(zip(kinds, invariants))
        self.per_key = any(f[0] == "perkey" and f[1] == "true" for f in forms)
        self.consistency = next(
            (str(f[1]) for f in forms if f[0] == "consistency"), "linearizable"
        )
        # An actor model's start and handlers run inside an actor; the
        # abstraction and state invariants read its state between messages.
        # Sequential runs of an actor also inject crashes: the actor restarts
        # from its last state (restart from) or its start, and the model
        # follows the restart's reference (or the start's model state).
        restarts = [c for c in self.commands if c.restart]
        self.commands = [c for c in self.commands if not c.restart]
        self.steps = self.commands
        if any(f[0] == "actor" and f[1] == "true" for f in forms):
            run, begin = self.start_run, self.start_model

            def start_run(symbols, *args):
                symbols["_lawspec_start"] = args
                return Actor(run(symbols, *args))

            def start_model(symbols, *args):
                symbols["_lawspec_start"] = args
                return begin(symbols, *args)

            self.start_run, self.start_model = start_run, start_model
            for c in self.commands:
                c.run = _actor_command(c.run, c.unit)
            self.steps = self.commands + [
                _Crash(run, begin, restarts[0] if restarts else None)
            ]
            if abstract is not None:
                self.abstract = lambda symbols, actor: abstract(symbols, actor.state())
            self.invariants = [
                (
                    k,
                    (lambda p: lambda symbols, s: p(symbols, s.state()))(p)
                    if k == "state"
                    else p,
                )
                for k, p in self.invariants
            ]


class _Invalid(Exception):
    """A command the model does not allow here."""


def _model_argument(d, indices):
    if d[0] != "dependent":
        return d
    def value(e):
        if not isinstance(e, list):
            return int(e)
        if e[0] == "index":
            return indices[int(e[1])]
        a, b = value(e[1]), value(e[2])
        if e[0] == "+": return a + b
        if e[0] == "-": return a - b
        if e[0] == "*": return a * b
        raise RuntimeError("unsupported model argument bound")
    base = list(d[1])
    if base[0] != "int":
        raise RuntimeError("unsupported dependent model argument type")
    lo, hi = base[2:4]
    for op, expression in d[2:]:
        n = value(expression)
        if op in (">", ">=", "=="):
            low = n + (1 if op == ">" else 0)
            lo = low if lo is None else max(lo, low)
        if op in ("<", "<=", "=="):
            high = n - (1 if op == "<" else 0)
            hi = high if hi is None else min(hi, high)
        if op not in (">", ">=", "<", "<=", "=="):
            raise RuntimeError("unsupported dependent model argument comparison")
    return base[:2] + [lo, hi]


def _model_arguments(command, indices):
    return [_model_argument(d, indices) for d in command.arguments]


def _empty_arguments(ds):
    return any(d[0] == "int" and d[2] is not None and d[3] is not None and d[2] > d[3] for d in ds)


def _valid_arguments(command, indices, args):
    return all(d[0] != "int" or ((d[2] is None or v >= d[2]) and (d[3] is None or v <= d[3]))
               for d, v in zip(_model_arguments(command, indices), args))


def _simulate(model, symbols, run):
    """The model states along a run, or _Invalid."""
    start_args, steps = run
    try:
        state = model.start_model(symbols, *start_args)
    except Exception:
        raise _Invalid()
    indices = list(model.start_indices)
    states = [state]
    for index, args in steps:
        command = model.steps[index]
        if not command.admits(indices) or not _valid_arguments(command, indices, args):
            raise _Invalid()
        state, _ = _step_model(command, symbols, args, state)
        indices = command.shifted(indices)
        states.append(state)
    return states


def _step_model(command, symbols, args, state):
    try:
        if command.when is not None and not command.when(symbols, state):
            raise _Invalid()
        out = command.reference(symbols, *args, state)
    except _Invalid:
        raise
    except Exception:
        raise _Invalid()
    if command.unit:
        return out, UNIT
    return out.fields[1], out.fields[0]


def _generate_run(model, random, length, size, crashes=False):
    symbols = {}
    start_args = [model.values.generate(d, random, size) for d in model.start_arguments]
    try:
        state = model.start_model(symbols, *start_args)
    except Exception:
        return (start_args, [])
    indices = list(model.start_indices)
    steps = []
    for _ in range(length):
        allowed = [i for i, c in enumerate(model.commands) if c.admits(indices) and not _empty_arguments(_model_arguments(c, indices))]
        if not allowed:
            break
        index = allowed[random.below(len(allowed))]
        # One step in eight of an actor's run is a crash.
        if crashes and len(model.steps) > len(model.commands) and random.below(8) == 0:
            index = len(model.commands)
        command = model.steps[index]
        args = [model.values.generate(d, random, size) for d in _model_arguments(command, indices)]
        try:
            state, _ = _step_model(command, symbols, args, state)
        except _Invalid:
            if any(d[0] == "dependent" for d in command.arguments) and (command.when is None or command.when(symbols, state)):
                raise RuntimeError("incomplete model generation: dependent arguments rejected by the reference (1 attempt)")
            continue
        steps.append((index, args))
        indices = command.shifted(indices)
    return (start_args, steps)


def _execute(model, run):
    """None when the system agrees with the model along the run; otherwise
    the failing step's number and what went wrong."""
    start_args, steps = run
    symbols = {}
    step = 0
    try:
        state = model.start_run(symbols, *start_args)
        expected = model.start_model(symbols, *start_args)
        failure = _check_state(model, symbols, state, expected)
        if failure:
            return step, failure
        for index, args in steps:
            step += 1
            command = model.steps[index]
            full = list(args)
            full.insert(command.state, state)
            out = command.run(symbols, *full)
            if model.shared:
                result = out
            else:
                result = UNIT if command.unit else out.fields[0]
                state = out.fields[-1]
            expected, wanted = _step_model(command, symbols, args, expected)
            if not command.unit and compare_values(result, wanted) != 0:
                return (
                    step,
                    f"returned {render(result)}; the model returns {render(wanted)}",
                )
            failure = _check_state(model, symbols, state, expected)
            if failure:
                return step, failure
    except _Invalid:
        return step, "the model does not allow this step"
    except Exception as error:
        return step, f"raised {type(error).__name__}: {error}"
    return None


def _check_state(model, symbols, state, expected):
    if model.abstract is not None:
        actual = model.abstract(symbols, state)
        if compare_values(actual, expected) != 0:
            return f"the state is {render(actual)}; the model is {render(expected)}"
    for kind, invariant in model.invariants:
        if not invariant(symbols, expected if kind == "model" else state):
            return f"an invariant on the {kind} fails"
    return None


def _shrink_candidates(model, run):
    start_args, steps = run
    n = len(steps)
    size = n // 2
    while size >= 1:
        for begin in range(0, n, size):
            yield (start_args, steps[:begin] + steps[begin + size :])
        size //= 2
    for k, (index, args) in enumerate(steps):
        command = model.steps[index]
        for j, (d, arg) in enumerate(zip(command.arguments, args)):
            for c in model.values.shrink(d[1] if d[0] == "dependent" else d, arg):
                yield (
                    start_args,
                    steps[:k]
                    + [(index, args[:j] + [c] + args[j + 1 :])]
                    + steps[k + 1 :],
                )
    for j, (d, arg) in enumerate(zip(model.start_arguments, start_args)):
        for c in model.values.shrink(d, arg):
            yield (start_args[:j] + [c] + start_args[j + 1 :], steps)


def _shrink_run(model, run, failure, budget):
    while budget > 0:
        for candidate in _shrink_candidates(model, run):
            budget -= 1
            if budget <= 0:
                break
            try:
                _simulate(model, {}, candidate)
            except _Invalid:
                continue
            found = _execute(model, candidate)
            if found is not None:
                run, failure = candidate, found
                break
        else:
            break
    return run, failure


def _describe_run(model, run):
    start_args, steps = run
    parts = ["start(" + ", ".join(render(a) for a in start_args) + ")"]
    parts += [
        model.steps[i].name + "(" + ", ".join(render(a) for a in args) + ")"
        for i, args in steps
    ]
    return "; ".join(parts)


def check_model(model, cases=100, max_length=20, max_shrinks=2000, seed=None):
    """Checks the system against its model on generated runs; a failure
    raises AssertionError naming the shortest failing run found.

    A stateful model is checked by running generated command sequences
    against the system and the model side by side, then shrinking a failing
    run. ref:DEC-stateful-models-linearizability"""
    import os

    if seed is None:
        seed = int(os.environ.get("LAWSPEC_SEED", "0"))
    random = SplitMix64(seed)
    for case in range(cases):
        run = _generate_run(
            model, random, random.below(max_length + 1), 1 + case % 8, crashes=True
        )
        failure = _execute(model, run)
        if failure is not None:
            run, (step, message) = _shrink_run(model, run, failure, max_shrinks)
            raise AssertionError(
                f"model {model.name} fails at step {step} of "
                f"{_describe_run(model, run)}: {message}"
            )


# Parallel runs of a shared model. A case is a sequential prefix and one
# branch per thread, generated so that the model allows every interleaving
# of the branches (a search over each thread's position and the model state,
# memoized). The system runs the branches at the same time, each call's start
# and return recorded on one counter, with random yields and short sleeps
# around calls to shake out rare schedules. The history must be
# linearizable: some interleaving that keeps every call after those that
# returned before it started must give every result the model gives and
# leave the state it leaves (a Wing-Gong search, memoized on the same
# positions and model state). Each case runs several times.

_THREADS = 3
_BRANCH = 5


def _parallel_allowed(model, prefix, branches):
    """Whether the model allows the prefix then every interleaving."""
    symbols = {}
    try:
        state = _simulate(model, symbols, prefix)[-1]
    except _Invalid:
        return False
    seen = set()

    def visit(positions, state):
        key = (positions, render(state))
        if key in seen:
            return True
        seen.add(key)
        for i, branch in enumerate(branches):
            k = positions[i]
            if k < len(branch):
                index, args = branch[k]
                try:
                    after, _ = _step_model(model.commands[index], symbols, args, state)
                except _Invalid:
                    return False
                if not visit(positions[:i] + (k + 1,) + positions[i + 1 :], after):
                    return False
        return True

    return visit(tuple(0 for _ in branches), state)


def _generate_branch(model, random, state, length, size):
    symbols = {}
    steps = []
    for _ in range(length):
        index = random.below(len(model.commands))
        command = model.commands[index]
        args = [model.values.generate(d, random, size) for d in command.arguments]
        try:
            state, _ = _step_model(command, symbols, args, state)
        except _Invalid:
            continue
        steps.append((index, args))
    return steps


def _generate_parallel(model, random, size, threads, branch_length):
    prefix = _generate_run(model, random, random.below(4), size)
    try:
        state = _simulate(model, {}, prefix)[-1]
    except _Invalid:
        return prefix, [[] for _ in range(threads)]
    branches = [
        _generate_branch(model, random, state, 1 + random.below(branch_length), size)
        for _ in range(threads)
    ]
    # Drop the last step of the longest branch (the first, among equals)
    # until every interleaving is allowed.
    while not _parallel_allowed(model, prefix, branches):
        longest = max(range(threads), key=lambda i: (len(branches[i]), -i))
        branches[longest] = branches[longest][:-1]
    return prefix, branches


def _perturb(random):
    """Nothing, a yield, or a sleep of 10 or 100 microseconds."""
    import time

    choice = random.below(4)
    if choice == 1:
        time.sleep(0)
    elif choice >= 2:
        time.sleep(1e-5 if choice == 2 else 1e-4)


def _execute_parallel(model, case, shake):
    """None when the history is linearizable; otherwise what went wrong."""
    import threading

    prefix, branches = case
    start_args, steps = prefix
    symbols = {}
    try:
        state = model.start_run(symbols, *start_args)
        for index, args in steps:
            command = model.commands[index]
            full = list(args)
            full.insert(command.state, state)
            command.run(symbols, *full)
    except Exception as error:
        return f"the prefix raised {type(error).__name__}: {error}"
    clock = [0]
    lock = threading.Lock()
    history = [[None] * len(b) for b in branches]
    errors = []

    def tick():
        with lock:
            clock[0] += 1
            return clock[0]

    def branch(i):
        own = {}
        random = SplitMix64(shake ^ ((i + 1) * 0x9E3779B97F4A7C15))
        for k, (index, args) in enumerate(branches[i]):
            command = model.commands[index]
            full = list(args)
            full.insert(command.state, state)
            _perturb(random)
            called = tick()
            try:
                result = command.run(own, *full)
            except Exception as error:
                with lock:
                    errors.append(
                        f"{command.name} raised {type(error).__name__}: {error}"
                    )
                result = None
            history[i][k] = (called, tick(), result)
            _perturb(random)

    threads = [threading.Thread(target=branch, args=(i,)) for i in range(len(branches))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        return errors[0]
    expected = _simulate(model, symbols, prefix)[-1]
    final = model.abstract(symbols, state) if model.abstract is not None else None
    if _linearizable(model, symbols, branches, history, expected, final, state):
        return None
    observed = "; ".join(
        f"{_branch_name(i)}: {model.commands[branches[i][k][0]].name}() returned {render(history[i][k][2])}"
        for i in range(len(branches))
        for k in range(len(branches[i]))
    )
    return f"no order of the parallel calls agrees with the model ({observed})"


def _branch_name(i):
    return chr(ord("A") + i)


def _linearizable(model, symbols, branches, history, expected, final, state):
    """Whether the history linearizes, with the final state and invariants
    the model gives. For a set or map whose every call touches one key, each
    key's calls are linearized separately (the keys are independent), one
    group after another; otherwise all calls at once."""

    def finish(model_state):
        if final is not None and compare_values(final, model_state) != 0:
            return False
        for kind, invariant in model.invariants:
            if not invariant(symbols, model_state if kind == "model" else state):
                return False
        return True

    if not model.per_key:
        return _linearize(model, symbols, branches, history, expected, finish)
    groups = {}
    for i, branch in enumerate(branches):
        for k, (index, args) in enumerate(branch):
            key = render(args[model.commands[index].key])
            groups.setdefault(key, [[] for _ in branches])[i].append(
                (branch[k], history[i][k])
            )
    model_state = expected
    for key in sorted(groups):
        parts = groups[key]
        ends = []
        if not _linearize(
            model,
            symbols,
            [[s for s, _ in p] for p in parts],
            [[h for _, h in p] for p in parts],
            model_state,
            lambda end: ends.append(end) or True,
        ):
            return False
        model_state = ends[0]
    return finish(model_state)


_CONSISTENT = {
    "linearizable": "linearizable",
    "sequential": "sequentially consistent",
    "causal": "causally consistent",
    "eventual": "eventually consistent",
}


def _linearize(model, symbols, branches, history, expected, finish):
    """A Wing-Gong search: linearize, next, a call no pending call on another
    thread returned before; memoized on positions and the model state.
    finish judges each complete order's final model state.

    With weaker consistency: sequential drops real time (each thread's own
    order remains); causal checks each thread's results alone, since threads
    that never message each other see only their own calls; eventual checks
    no results, only the final state.

    ref:wing-gong-linearizability"""
    mode = model.consistency
    if mode == "causal":
        for i, branch in enumerate(branches):
            state = expected
            for k, (index, args) in enumerate(branch):
                command = model.commands[index]
                try:
                    state, wanted = _step_model(command, symbols, args, state)
                except _Invalid:
                    return False
                if not command.unit and compare_values(history[i][k][2], wanted) != 0:
                    return False
        return True
    seen = set()

    def visit(positions, model_state):
        key = (positions, render(model_state))
        if key in seen:
            return False
        seen.add(key)
        if all(k == len(b) for k, b in zip(positions, branches)):
            return finish(model_state)
        for i, branch in enumerate(branches):
            k = positions[i]
            if k == len(branch):
                continue
            called = history[i][k][0]
            if mode == "linearizable" and any(
                positions[j] < len(branches[j]) and history[j][positions[j]][1] < called
                for j in range(len(branches))
                if j != i
            ):
                continue
            index, args = branch[k]
            command = model.commands[index]
            try:
                after, wanted = _step_model(command, symbols, args, model_state)
            except _Invalid:
                continue
            if (
                mode != "eventual"
                and not command.unit
                and compare_values(history[i][k][2], wanted) != 0
            ):
                continue
            if visit(positions[:i] + (k + 1,) + positions[i + 1 :], after):
                return True
        return False

    return visit(tuple(0 for _ in branches), expected)


def _parallel_fails(model, case, repeats, shake):
    for attempt in range(repeats):
        failure = _execute_parallel(model, case, shake + attempt)
        if failure is not None:
            return failure
    return None


def _shrink_parallel(model, case, failure, repeats, budget, shake):
    while budget > 0:
        prefix, branches = case
        candidates = []
        start_args, steps = prefix
        for k in range(len(steps)):
            candidates.append(((start_args, steps[:k] + steps[k + 1 :]), branches))
        for i in range(len(branches)):
            for k in range(len(branches[i])):
                shorter = [
                    b if j != i else b[:k] + b[k + 1 :] for j, b in enumerate(branches)
                ]
                candidates.append((prefix, shorter))
        # Then smaller arguments, branch by branch, step by step.
        for i in range(len(branches)):
            for k, (index, args) in enumerate(branches[i]):
                command = model.commands[index]
                for a, (d, arg) in enumerate(zip(command.arguments, args)):
                    for c in model.values.shrink(d, arg):
                        step = (index, args[:a] + [c] + args[a + 1 :])
                        changed = [
                            b if j != i else b[:k] + [step] + b[k + 1 :]
                            for j, b in enumerate(branches)
                        ]
                        candidates.append((prefix, changed))
        for candidate in candidates:
            budget -= 1
            if budget <= 0:
                break
            if not _parallel_allowed(model, *candidate):
                continue
            found = _parallel_fails(model, candidate, repeats, shake)
            if found is not None:
                case, failure = candidate, found
                break
        else:
            break
    return case, failure


def _describe_parallel(model, case):
    prefix, branches = case
    describe = lambda steps: "; ".join(
        model.commands[i].name + "(" + ", ".join(render(a) for a in args) + ")"
        for i, args in steps
    )
    parts = [
        f"{_branch_name(i)}: {describe(b) or 'nothing'}" for i, b in enumerate(branches)
    ]
    return f"{_describe_run(model, prefix)}, then {', '.join(parts[:-1])} and {parts[-1]} at the same time"


def check_model_parallel(
    model,
    cases=50,
    repeats=10,
    max_shrinks=300,
    seed=None,
    threads=_THREADS,
    branch_length=_BRANCH,
):
    """Checks a shared model's histories under concurrency; a failure
    raises AssertionError naming the smallest failing case found.

    A shared model promises linearizability unless it names a weaker
    consistency, so concurrent histories are judged against some sequential
    order of the calls. ref:herlihy-wing-linearizability
    ref:DEC-stateful-models-linearizability"""
    import os

    if seed is None:
        seed = int(os.environ.get("LAWSPEC_SEED", "0"))
    random = SplitMix64(seed ^ 0x5BD1E995)
    for case_number in range(cases):
        case = _generate_parallel(
            model, random, 1 + case_number % 8, threads, branch_length
        )
        shake = random.next()
        failure = _parallel_fails(model, case, repeats, shake)
        if failure is not None:
            case, failure = _shrink_parallel(
                model, case, failure, max(2, repeats // 2), max_shrinks, shake
            )
            raise AssertionError(
                f"model {model.name} is not {_CONSISTENT[model.consistency]}: {_describe_parallel(model, case)}: {failure}"
            )


# Scenarios: processes that drive a shared model's commands at the same time
# and talk over channels (see LawSpec.Core.Program for the spec). Each
# channel has a queue per direction; a process holds an end of a channel as
# (channel, side), the first branch of a par to use a channel taking side 0.
# A channel end sent over a channel moves to the receiver. Every command's
# call and return are stamped on one counter; the history must linearize
# against the model, and every expect must hold, on each of many schedules.


class _Channel:
    def __init__(self):
        import queue

        self.queues = (queue.Queue(), queue.Queue())
        self.ended = [False, False]
        self.lock = threading.Lock()

    def receive(self, side):
        """The next value for side, _GONE once the other side has ended."""
        value = self.queues[1 - side].get(timeout=5)
        if value is _GONE:
            self.queues[1 - side].put(_GONE)
        return value

    def send(self, side, value):
        """A channel end sent to a process that has ended is given up."""
        with self.lock:
            if not (self.ended[1 - side] and isinstance(value, _End)):
                self.queues[side].put(value)
                return
        value.channel.gone(value.side)

    def gone(self, side):
        """side's process has ended: the other side's receives that find
        nothing more fail instead of waiting (again and again), and channel
        ends on their way to side are given up too."""
        import queue

        stranded = []
        with self.lock:
            if self.ended[side]:
                return
            self.ended[side] = True
            self.queues[side].put(_GONE)
            while True:
                try:
                    value = self.queues[1 - side].get_nowait()
                except queue.Empty:
                    break
                if isinstance(value, _End):
                    stranded.append(value)
                elif value is _GONE:
                    self.queues[1 - side].put(value)
                    break
        for end in stranded:
            end.channel.gone(end.side)


_GONE = object()


class _NetScenarioChannel:
    """A scenario channel whose two sides are endpoints on two nodes of a
    faulty in-memory network. A channel end sent over it travels as its
    name, and the receiver uses the end where it is (its owner)."""

    def __init__(self, network, name, steps, values, registry):
        self.name, self.registry = name, registry
        self.nodes = [
            Node(network.insecure_transport_for_tests(f"{name}-{side}"))
            for side in (0, 1)
        ]

        def wire(sends, d):
            return (sends, ["text"] if d == ["end"] else d)

        self.ends = [self.nodes[0].listen(name, [wire(s, d) for s, d in steps], values)]
        self.ends.append(
            self.nodes[1].dial(
                f"{self.nodes[0].address}/{name}",
                [wire(not s, d) for s, d in steps],
                values,
            )
        )
        self.done = [False, False]
        registry[name] = self

    def send(self, side, value):
        if isinstance(value, _End):
            value = f"{value.channel.name}#{value.side}"
        self.ends[side].send(side, value)

    def receive(self, side):
        try:
            value = self.ends[side].receive(side, timeout=5)
        except PeerFailed:
            return _GONE
        if (
            isinstance(value, str)
            and "#" in value
            and value.rpartition("#")[0] in self.registry
        ):
            owner, _, which = value.rpartition("#")
            return _End(self.registry[owner], int(which))
        return value

    def gone(self, side):
        if not self.done[side]:
            self.done[side] = True
            self.ends[side].abandon(side)

    def close(self):
        for node in self.nodes:
            node.close()


class _ScenarioMailbox:
    """A scenario's mailbox: any process sends, one receives. expected is
    how many sends the scenario makes; a process that ends gives up the
    sends it did not make, and a receive with nothing left to come fails
    (_GONE) instead of waiting. Over a network, messages go from a sender
    node to the receiver's node, each send waiting until it is delivered."""

    def __init__(
        self, name, expected, network=None, descriptor=None, values=None, registry=None
    ):
        self.name, self.expected = name, expected
        self.received, self.abandoned = 0, 0
        self.ready = threading.Condition()
        self.items = deque()
        self.registry = registry
        self.nodes = []
        if network is not None:
            owner, senders = (
                Node(network.insecure_transport_for_tests(f"{name}-owner")),
                Node(network.insecure_transport_for_tests(f"{name}-senders")),
            )
            self.nodes = [owner, senders]
            d = ["text"] if descriptor == ["end"] else descriptor
            self.inbox = owner.mailbox(name, d, values)
            self.remote = senders.remote_mailbox(
                f"{owner.address}/{name}", d, values, timeout=5.0
            )
        else:
            self.inbox = None

    def send(self, value, clock):
        if self.inbox is None:
            with self.ready:
                self.items.append((value, clock))
                self.ready.notify_all()
            return
        if isinstance(value, _End):
            value = f"{value.channel.name}#{value.side}"
        # The clock travels beside the network, in send order.
        with self.ready:
            self.items.append((None, clock))
        self.remote.send(value)
        with self.ready:
            self.ready.notify_all()

    def give_up(self, count):
        with self.ready:
            self.abandoned += count
            self.ready.notify_all()

    def receive(self):
        """(value, sender's clock), or (_GONE, {})."""
        import time

        give_up = time.monotonic() + 5
        while True:
            with self.ready:
                if self.inbox is None and self.items:
                    self.received += 1
                    return self.items.popleft()
                if self.received + self.abandoned >= self.expected and (
                    self.inbox is None or not self.items
                ):
                    return _GONE, {}
            if self.inbox is not None:
                try:
                    value = self.inbox.receive(timeout=0.02)
                except TimeoutError:
                    value = None
                    if time.monotonic() > give_up:
                        raise TimeoutError("a mailbox receive waited too long")
                else:
                    with self.ready:
                        self.received += 1
                        clock = self.items.popleft()[1] if self.items else {}
                    if (
                        isinstance(value, str)
                        and "#" in value
                        and value.rpartition("#")[0] in self.registry
                    ):
                        owner, _, which = value.rpartition("#")
                        value = _End(self.registry[owner], int(which))
                    return value, clock
            else:
                with self.ready:
                    if not self.ready.wait(
                        timeout=max(0.0, give_up - time.monotonic())
                    ):
                        raise TimeoutError("a mailbox receive waited too long")

    def close(self):
        for node in self.nodes:
            node.close()


def _scenario_sends(acts, name):
    """How many times these acts (not nested pars) send to name."""
    return sum(1 for act in acts if act[0] == "send" and str(act[1]) == name)


def _all_sends(acts, name):
    """How many sends to name the whole program makes."""
    total = 0
    for act in acts:
        if act[0] == "send" and str(act[1]) == name:
            total += 1
        elif act[0] == "par":
            total += sum(_all_sends(branch[1:], name) for branch in act[1:])
    return total


def _scenario_processes(acts, found):
    """Every process of a par, outermost and first first (not or else)."""
    for act in acts:
        if act[0] == "par":
            for branch in act[1:]:
                found.append(id(branch))
                _scenario_processes(branch[1:], found)
    return found


class _End:
    """A channel end in transit or held by a process."""

    def __init__(self, channel, side):
        self.channel, self.side = channel, side


def _acts_channels(acts):
    """The names an act list sends, receives or sends away, with nested pars."""
    names = []
    for act in acts:
        if act[0] == "send":
            names.append(str(act[1]))
            if act[2][0] == "var":
                names.append(str(act[2][1]))
        elif act[0] == "receive":
            names.append(str(act[1]))
        elif act[0] == "receiveor":
            names.append(str(act[1]))
            names += _acts_channels(act[3][1:])
        elif act[0] == "par":
            for branch in act[1:]:
                names += _acts_channels(branch[1:])
    return names


def _constant(form):
    kind = form[0]
    if kind == "int":
        return form[1]
    if kind == "text":
        return str(form[1])
    if kind == "bool":
        return form[1] == "true"
    return DataValue(str(form[1]), ())


def _run_scenario(model, spec, shake, crash=False, network=False):
    import threading

    forms = read_descriptor(spec)
    title = str(forms[0][1])
    names = [str(c) for c in next(f for f in forms if f[0] == "channels")[1:]]
    body = next(f for f in forms if f[0] == "process")[1:]
    wire = next((f for f in forms if f[0] == "wire"), None)
    boxes = [str(m) for f in forms if f[0] == "mailboxes" for m in f[1:]]
    if network and wire is not None:
        # Loss, duplication and delay (which reorders); the channels'
        # numbered, acknowledged frames must hide them all.
        net = MemoryNetwork(
            seed=shake ^ 0x7F4A7C159E3779B9, loss=0.1, duplicate=0.1, delay=0.002
        )
        types = Values({str(f[1]): f for f in wire[1:] if f[0] == "data"})
        steps = {
            str(f[1]): [(s[0] == "send", s[1]) for s in f[2:]]
            for f in wire[1:]
            if f[0] == "channel"
        }
        kinds = {str(f[1]): f[2] for f in wire[1:] if f[0] == "mailbox"}
        registry = {}
        channels = {
            name: _NetScenarioChannel(net, name, steps[name], types, registry)
            for name in names
        }
        mailboxes = {
            m: _ScenarioMailbox(m, _all_sends(body, m), net, kinds[m], types, registry)
            if m in kinds
            else _ScenarioMailbox(m, _all_sends(body, m))
            for m in boxes
        }
    else:
        channels = {name: _Channel() for name in names}
        mailboxes = {m: _ScenarioMailbox(m, _all_sends(body, m)) for m in boxes}
    commands = {c.name: c for c in model.commands}
    symbols = {}
    start_args = [model.values.minimal(d) for d in model.start_arguments]
    state = model.start_run(symbols, *start_args)
    expected = model.start_model(symbols, *start_args)
    lock = threading.Lock()
    clock = [0]
    history = []
    failures = []
    # The crashed process (a par's branch) and the act it crashes before.
    processes = _scenario_processes(body, [])
    victim = None
    if crash and processes:
        chooser = SplitMix64(shake ^ 0xC3A5C85C97CB3127)
        branch = processes[chooser.below(len(processes))]
        victim = (branch, chooser.below(_branch_length(body, branch) + 1))

    def tick():
        with lock:
            clock[0] += 1
            return clock[0]

    # Vector clocks: each value sent carries its sender's clock (kept here,
    # in order per channel direction), so calls can be ordered by what
    # happened before what.
    stamps = {}

    def stamp(channel, side, clock):
        with lock:
            stamps.setdefault((id(channel), side), deque()).append(dict(clock))

    def unstamp(channel, side, clock, me):
        with lock:
            queue_ = stamps.get((id(channel), 1 - side))
            sent = queue_.popleft() if queue_ else {}
        for p, n in sent.items():
            clock[p] = max(clock.get(p, 0), n)
        clock[me] = clock.get(me, 0) + 1

    def process(acts, env, ends, random, identity=None, clock=None):
        """'done' or 'failed'; either way, the ends still held are given up."""
        clock = {} if clock is None else clock
        sent = {m: 0 for m in mailboxes}
        try:
            return steps(
                acts,
                env,
                ends,
                random,
                identity,
                clock,
                "root" if identity is None else identity,
                sent,
            )
        finally:
            for channel, side in ends.values():
                channel.gone(side)
            # Sends this process will never make.
            for m, box in mailboxes.items():
                missing = _scenario_sends(acts, m) - sent[m]
                if missing > 0:
                    box.give_up(missing)

    def steps(acts, env, ends, random, identity, clock, me, sent):
        own = {}
        for index, act in enumerate(acts):
            if failures:
                return "failed"
            if victim is not None and victim == (identity, index):
                return "failed"
            kind = act[0]
            if kind == "call":
                command = commands[str(act[1])]
                args = [
                    env[str(o[1])] if o[0] == "var" else _constant(o) for o in act[3:]
                ]
                full = list(args)
                full.insert(command.state, state)
                _perturb(random)
                clock[me] = clock.get(me, 0) + 1
                at_call = dict(clock)
                called = tick()
                try:
                    result = command.run(own, *full)
                except Exception as error:
                    failures.append(
                        f"{command.name} raised {type(error).__name__}: {error}"
                    )
                    return "failed"
                returned = tick()
                clock[me] += 1
                with lock:
                    history.append(
                        (
                            command,
                            args,
                            result,
                            called,
                            returned,
                            me,
                            at_call,
                            dict(clock),
                        )
                    )
                if act[2] != "_":
                    env[str(act[2])] = result
            elif kind == "send" and str(act[1]) in mailboxes:
                operand = act[2]
                if operand[0] == "var" and str(operand[1]) in ends:
                    value = _End(*ends.pop(str(operand[1])))
                else:
                    value = (
                        env[str(operand[1])]
                        if operand[0] == "var"
                        else _constant(operand)
                    )
                _perturb(random)
                clock[me] = clock.get(me, 0) + 1
                try:
                    mailboxes[str(act[1])].send(value, dict(clock))
                except Unreachable as error:
                    failures.append(f"a send to mailbox {act[1]} failed: {error}")
                    return "failed"
                sent[str(act[1])] += 1
            elif kind in ("receive", "receiveor") and str(act[1]) in mailboxes:
                try:
                    value, carried = mailboxes[str(act[1])].receive()
                except TimeoutError:
                    failures.append(
                        f"a receive on mailbox {act[1]} waited too long: the processes are blocked"
                    )
                    return "failed"
                if value is _GONE:
                    if kind == "receive":
                        return "failed"
                    return steps(act[3][1:], env, ends, random, None, clock, me, sent)
                for p, n in carried.items():
                    clock[p] = max(clock.get(p, 0), n)
                clock[me] = clock.get(me, 0) + 1
                if isinstance(value, _End):
                    ends[str(act[2])] = (value.channel, value.side)
                else:
                    env[str(act[2])] = value
            elif kind == "send":
                channel, side = ends[str(act[1])]
                operand = act[2]
                if operand[0] == "var" and str(operand[1]) in ends:
                    value = _End(*ends.pop(str(operand[1])))
                else:
                    value = (
                        env[str(operand[1])]
                        if operand[0] == "var"
                        else _constant(operand)
                    )
                _perturb(random)
                clock[me] = clock.get(me, 0) + 1
                stamp(channel, side, clock)
                channel.send(side, value)
            elif kind in ("receive", "receiveor"):
                channel, side = ends[str(act[1])]
                try:
                    value = channel.receive(side)
                except Exception:
                    failures.append(
                        f"a receive on {act[1]} waited too long: the processes are blocked"
                    )
                    return "failed"
                if value is _GONE:
                    # The other process ended: or else runs instead of the
                    # rest; without it, this process fails too.
                    if kind == "receive":
                        return "failed"
                    del ends[str(act[1])]
                    return steps(act[3][1:], env, ends, random, None, clock, me, sent)
                unstamp(channel, side, clock, me)
                if isinstance(value, _End):
                    ends[str(act[2])] = (value.channel, value.side)
                else:
                    env[str(act[2])] = value
            elif kind == "par":
                branches = act[1:]
                owned = {}
                for i, branch in enumerate(branches):
                    for name in _acts_channels(branch[1:]):
                        if name not in owned:
                            owned[name] = []
                        if i not in owned[name]:
                            owned[name].append(i)
                threads = []
                outcomes = [None] * len(branches)
                clocks = [dict(clock) for _ in branches]
                for i, branch in enumerate(branches):
                    mine = {}
                    for name, users in owned.items():
                        if i in users:
                            if name in ends:
                                mine[name] = ends.pop(name)
                            elif name in channels:
                                mine[name] = (channels[name], users.index(i))

                    def run(
                        i=i,
                        branch=branch,
                        mine=mine,
                        random=SplitMix64(
                            shake ^ ((len(threads) + 1) * 0x9E3779B97F4A7C15)
                        ),
                    ):
                        outcomes[i] = process(
                            branch[1:], dict(env), mine, random, id(branch), clocks[i]
                        )

                    threads.append(threading.Thread(target=run))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                for child in clocks:
                    for p, n in child.items():
                        clock[p] = max(clock.get(p, 0), n)
                clock[me] = clock.get(me, 0) + 1
                # A failed branch fails the process that ran the par.
                if "failed" in outcomes:
                    return "failed"
            elif kind == "expect":
                actual, wanted = env.get(str(act[1])), _constant(act[2])
                if actual is None or compare_values(actual, wanted) != 0:
                    failures.append(
                        f"expect {act[1]} = {render(wanted)} failed: {act[1]} is {render(actual)}"
                    )
                    return "failed"
        if victim is not None and victim == (identity, len(acts)):
            return "failed"
        return "done"

    outcome = process(body, {}, {}, SplitMix64(shake))
    for channel in channels.values():
        if isinstance(channel, _NetScenarioChannel):
            channel.close()
    for box in mailboxes.values():
        box.close()
    if failures:
        return title, failures[0] + (
            " (with a process crashed)" if victim is not None else ""
        )
    if outcome == "failed" and victim is None:
        return title, "a process failed"
    final = model.abstract(symbols, state) if model.abstract is not None else None
    if not _linearizes_history(model, symbols, history, expected, final, state):
        observed = "; ".join(
            f"{c.name}({', '.join(render(a) for a in args)}) returned {render(r)}"
            for c, args, r, *_ in sorted(history, key=lambda h: h[3])
        )
        return (
            title,
            f"the calls are not {_CONSISTENT[model.consistency]} with the model ({observed})",
        )
    return title, None


def _branch_length(acts, identity):
    """How many acts the branch with this identity has."""
    for act in acts:
        if act[0] == "par":
            for branch in act[1:]:
                if id(branch) == identity:
                    return len(branch) - 1
                found = _branch_length(branch[1:], identity)
                if found is not None:
                    return found
    return None


def _happened_before(a, b):
    """Whether call a returned before call b began, as far as messages tell:
    a's return clock is at or below b's call clock everywhere."""
    return all(b[6].get(p, 0) >= n for p, n in a[7].items())


def _linearizes_history(model, symbols, history, expected, final, state):
    """A Wing-Gong search over the scenario's calls, memoized on the calls
    done and the state. Each call is (command, args, result, called,
    returned, process, call clock, return clock). Linearizable: next, a call
    no pending call returned before (real time). Sequential: next, a call
    every call that happened before it (its process's order, and messages)
    is done. Causal: each process's results from an order of what happened
    before them. Eventual: no results, only the final state."""
    mode = model.consistency
    count = len(history)

    def before(j, i):
        if mode == "linearizable":
            return history[j][4] < history[i][3]
        return _happened_before(history[j], history[i])

    def search(members, checked, judge_final):
        seen = set()
        full = 0
        for i in members:
            full |= 1 << i

        def visit(done, model_state):
            key = (done, render(model_state))
            if key in seen:
                return False
            seen.add(key)
            if done == full:
                if not judge_final:
                    return True
                if final is not None and compare_values(final, model_state) != 0:
                    return False
                return all(
                    invariant(symbols, model_state if kind == "model" else state)
                    for kind, invariant in model.invariants
                )
            for i in members:
                if done & (1 << i):
                    continue
                if any(not done & (1 << j) and before(j, i) for j in members if j != i):
                    continue
                command, args, result = history[i][0], history[i][1], history[i][2]
                try:
                    after, wanted = _step_model(command, symbols, args, model_state)
                except _Invalid:
                    continue
                if (
                    i in checked
                    and not command.unit
                    and compare_values(result, wanted) != 0
                ):
                    continue
                if visit(done | (1 << i), after):
                    return True
            return False

        return visit(0, expected)

    everything = list(range(count))
    if mode == "causal":
        for process in {h[5] for h in history}:
            own = [i for i in everything if history[i][5] == process]
            seen_by = sorted(
                set(own)
                | {
                    j
                    for j in everything
                    for i in own
                    if j != i and _happened_before(history[j], history[i])
                }
            )
            if not search(seen_by, set(own), False):
                return False
        return True
    return search(everything, set() if mode == "eventual" else set(everything), True)


def check_scenario(model, spec, runs=30, seed=None):
    """Runs a scenario on many schedules; a failure raises AssertionError."""
    import os

    if seed is None:
        seed = int(os.environ.get("LAWSPEC_SEED", "0"))
    random = SplitMix64(seed ^ 0x2545F4914F6CDD1D)
    for run in range(runs):
        # Every third run crashes one process of a par at a random point, and
        # every third other one sends each channel over a faulty network.
        title, failure = _run_scenario(
            model, spec, random.next(), crash=run % 3 == 2, network=run % 3 == 1
        )
        if failure is not None:
            raise AssertionError(f"scenario {title} fails: {failure}")


# Sessions: typed channel ends for implementation code. The generated
# lawspec_sessions module gives every protocol step its own class; these are
# the pieces those classes share.


class SessionError(Exception):
    """An end of a session channel was used wrongly."""


class Channel:
    """A two-way channel between side 0 and side 1, one queue per direction.

    Ends talk to it only through send(side, value) and receive(side), so a
    network transport can stand in for it by providing the same two methods.
    """

    def __init__(self):
        import queue

        self._queues = (queue.Queue(), queue.Queue())

    def send(self, side, value):
        """Sends value from side to the other side."""
        self._queues[side].put(value)

    def receive(self, side):
        """Waits for the next value the other side sent to side; raises
        PeerFailed once the other side has given up and nothing is left."""
        value = self._queues[1 - side].get()
        if value is _ABANDONED:
            self._queues[1 - side].put(value)
            raise PeerFailed(
                "the other end gave up the conversation (its process failed or abandoned it)"
            )
        return value

    def abandon(self, side):
        """side gives up: the other side's receives fail after the values
        already sent."""
        self._queues[side].put(_ABANDONED)


_ABANDONED = object()


class PeerFailed(Exception):
    """A receive whose other end gave up: its process failed, or it called
    abandon(). Catch it to handle the failure (or else); otherwise this
    process fails too."""


class SessionEnd:
    """One end of a channel, before one step of its protocol. An end can be
    used once: each send or receive returns the end for the next step.

    Using an end exactly once is what makes a session follow its protocol;
    with channels joined in a tree, scenarios are deadlock-free by
    construction. ref:DEC-sessions-by-construction
    ref:caires-pfenning-session-types ref:wadler-propositions-as-sessions"""

    def __init__(self, channel, side):
        self._channel = channel
        self._side = side
        self._used = False
        self._lock = threading.Lock()

    def _take(self):
        with self._lock:
            if self._used:
                raise SessionError(
                    f"{type(self).__qualname__}: this end was already used; "
                    "use the end its last step returned"
                )
            self._used = True
        return self._channel

    def _send(self, value, after):
        channel = self._take()
        channel.send(self._side, value)
        return after(channel, self._side)

    def _send_end(self, end, start, after):
        """Sends end, which must be an unused start end of class start; the
        receiver gets it, and this side must not use it any more."""
        if not isinstance(end, start):
            raise TypeError(
                f"{type(self).__qualname__}.send expects a "
                f"{start.__qualname__}, not {type(end).__qualname__}"
            )
        moved = start(end._take(), end._side)
        return self._send(moved, after)

    def _receive(self, after):
        channel = self._take()
        value = channel.receive(self._side)
        return value, after(channel, self._side)

    def abandon(self):
        """Gives up the conversation: the other end's receives fail with
        PeerFailed once it has received what was already sent."""
        self._take().abandon(self._side)


def check_send(value, t):
    """Checks that value is a native value of the scalar type t."""
    try:
        validate(value, t)
    except (TypeError, ValueError) as error:
        raise TypeError(f"cannot send {value!r} as {t}: {error}") from None
    return value


class Spawned:
    """A function running in its own thread; join() waits for it."""

    def __init__(self, fn, args):
        self._result = None
        self._error = None

        def run():
            try:
                self._result = fn(*args)
            except BaseException as error:  # re-raised by join()
                self._error = error
                # A failed process gives up the channel ends it was given.
                for arg in args:
                    if isinstance(arg, SessionEnd):
                        arg._channel.abandon(arg._side)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def join(self):
        """Waits for the function; returns its result or raises its error."""
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self._result


def spawn(fn, *args):
    """Runs fn(*args) in a new thread and returns a handle with join()."""
    return Spawned(fn, args)


def par(*fns):
    """Runs the functions at once, waits for all of them and returns their
    results in order; if any fails, raises the first failure."""
    handles = [spawn(fn) for fn in fns]
    results, failure = [], None
    for handle in handles:
        try:
            results.append(handle.join())
        except BaseException as error:
            if failure is None:
                failure = error
            results.append(None)
    if failure is not None:
        raise failure
    return results


# Distribution. Values cross the network in a canonical binary encoding
# driven by their type descriptor (the same descriptors as generation), so
# no tags are sent and every target writes the same bytes:
#   int: zigzag LEB128 of the integer (any size)      bool: 0 or 1
#   text, bytes: LEB128 length, then UTF-8 or raw     unit: nothing
#   list: LEB128 count, then items                     maybe: 0, or 1 then the value
#   either: 0 then left, or 1 then right               data: LEB128 constructor index, then fields
# A node sends frames over a Transport (in memory, TCP or HTTP): kind,
# entity name, the sender's address, an id and a payload.


class WireError(ValueError):
    """Bytes that are not an encoding of a value of the expected type."""


class Unreachable(Exception):
    """A node could not be reached, or did not answer in time."""


def _put_varint(out, n):
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return


def _get_varint(buf, pos):
    result, shift = 0, 0
    while True:
        if pos >= len(buf):
            raise WireError("the bytes end in the middle of a value")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7


def _wire_put(values, d, v, out):
    d = values.resolve(d)
    kind = d[0]
    if kind == "int":
        lo, hi = d[2], d[3]
        if (
            isinstance(v, bool)
            or not isinstance(v, int)
            or (lo is not None and v < lo)
            or (hi is not None and v > hi)
        ):
            raise WireError(f"{v!r} is not a {d[1]}")
        _put_varint(out, v * 2 if v >= 0 else -v * 2 - 1)
    elif kind == "bool":
        out.append(1 if v else 0)
    elif kind in ("text", "bytes"):
        raw = v.encode("utf-8") if kind == "text" else bytes(v)
        _put_varint(out, len(raw))
        out.extend(raw)
    elif kind == "unit":
        pass
    elif kind == "list":
        _put_varint(out, len(v))
        for item in v:
            _wire_put(values, d[1], item, out)
    elif kind == "maybe":
        if v.tag.endswith("Absent"):
            out.append(0)
        else:
            out.append(1)
            _wire_put(values, d[1], v.fields[0], out)
    elif kind == "either":
        left = v.tag.endswith("Left")
        out.append(0 if left else 1)
        _wire_put(values, d[1] if left else d[2], v.fields[0], out)
    elif kind == "data":
        for index, ctor in enumerate(d[2:]):
            if str(ctor[1]) == v.tag:
                _put_varint(out, index)
                for field, fd in zip(v.fields, ctor[2:]):
                    _wire_put(values, fd, field, out)
                return
        raise WireError(f"{v.tag} is not a constructor of {d[1]}")
    elif kind == "end":
        _wire_put(values, ["text"], v, out)
    else:
        raise WireError("unknown descriptor " + str(d))


def _wire_get(values, d, buf, pos):
    d = values.resolve(d)
    kind = d[0]
    if kind == "int":
        z, pos = _get_varint(buf, pos)
        v = z // 2 if z % 2 == 0 else -(z + 1) // 2
        lo, hi = d[2], d[3]
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            raise WireError(f"{v} is out of range for {d[1]}")
        return v, pos
    if kind == "bool":
        if pos >= len(buf) or buf[pos] > 1:
            raise WireError("not a Bool")
        return buf[pos] == 1, pos + 1
    if kind in ("text", "bytes", "end"):
        n, pos = _get_varint(buf, pos)
        if pos + n > len(buf):
            raise WireError("the bytes end in the middle of a value")
        raw = bytes(buf[pos : pos + n])
        if kind == "bytes":
            return raw, pos + n
        try:
            return raw.decode("utf-8"), pos + n
        except UnicodeDecodeError:
            raise WireError("text that is not UTF-8") from None
    if kind == "unit":
        return UNIT, pos
    if kind == "list":
        n, pos = _get_varint(buf, pos)
        items = []
        for _ in range(n):
            item, pos = _wire_get(values, d[1], buf, pos)
            items.append(item)
        return items, pos
    if kind in ("maybe", "either"):
        if pos >= len(buf) or buf[pos] > 1:
            raise WireError(f"not a {kind.capitalize()}")
        which = buf[pos]
        pos += 1
        if kind == "maybe":
            if which == 0:
                return DataValue("Maybe::Absent", ()), pos
            v, pos = _wire_get(values, d[1], buf, pos)
            return DataValue("Maybe::Present", (v,)), pos
        v, pos = _wire_get(values, d[1] if which == 0 else d[2], buf, pos)
        return DataValue("Either::Left" if which == 0 else "Either::Right", (v,)), pos
    if kind == "data":
        index, pos = _get_varint(buf, pos)
        ctors = d[2:]
        if index >= len(ctors):
            raise WireError(f"no constructor {index} in {d[1]}")
        fields = []
        for fd in ctors[index][2:]:
            v, pos = _wire_get(values, fd, buf, pos)
            fields.append(v)
        return DataValue(str(ctors[index][1]), tuple(fields)), pos
    raise WireError("unknown descriptor " + str(d))


def wire_encode(values, d, v):
    """The value's canonical bytes."""
    out = bytearray()
    _wire_put(values, d, v, out)
    return bytes(out)


def wire_decode(values, d, data):
    """The value encoded by exactly these bytes."""
    v, pos = _wire_get(values, d, data, 0)
    if pos != len(data):
        raise WireError("extra bytes after the value")
    return v


def wire_encoded(text, seed, size, count):
    """count values generated from one seed, encoded, in hexadecimal."""
    values, d = values_from(text)
    random = SplitMix64(seed)
    return [
        wire_encode(values, d, values.generate(d, random, size)).hex()
        for _ in range(count)
    ]


# The failure database's inputs (see LawSpec.Search): a failing case's
# inputs, each in the wire encoding of its descriptor, kept under
# LAWSPEC_FAILURES/inputs and replayed before the law's next generated cases.
def _search_file(law):
    import os

    directory = os.environ.get("LAWSPEC_FAILURES")
    if not directory:
        return None
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in law)
    return os.path.join(directory, "inputs", safe + ".json")


def search_remember(law, descriptors, values):
    """Keeps a failing case's inputs; the last one kept is the smallest the
    property library found."""
    import json
    import os

    path = _search_file(law)
    if path is None:
        return
    try:
        encoded = []
        for text, value in zip(descriptors, values):
            table, d = values_from(text)
            encoded.append(wire_encode(table, d, value).hex())
    except Exception:  # noqa: BLE001 - kept by seed only
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        json.dump({"law": law, "inputs": encoded}, out)


def search_replay(law, descriptors, case):
    """Runs the law on the inputs its last failure kept, if any; they leave
    the database once the law holds for them."""
    import json
    import os

    path = _search_file(law)
    if path is None or not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as source:
        entry = json.load(source)
    try:
        values = []
        for text, encoded in zip(descriptors, entry.get("inputs", [])):
            table, d = values_from(text)
            values.append(wire_decode(table, d, bytes.fromhex(encoded)))
    except (WireError, ValueError):
        values = []
    if len(values) == len(descriptors):
        print(f"{law}: replaying the failing inputs kept in .lawspec/failures")
        case(values)
    os.remove(path)


def wire_round_trips(text, seed, size, count):
    """Whether count generated values decode to themselves."""
    values, d = values_from(text)
    random = SplitMix64(seed)
    for _ in range(count):
        v = values.generate(d, random, size)
        if not _same(wire_decode(values, d, wire_encode(values, d, v)), v):
            return False
    return True


_NO_TYPES = Values({})
_FRAME = [["text"], ["text"], ["text"], ["int", "UInt64", 0, None], ["bytes"]]


def _frame_encode(kind, to, source, ident, payload):
    out = bytearray()
    for d, v in zip(_FRAME, (kind, to, source, ident, payload)):
        _wire_put(_NO_TYPES, d, v, out)
    return bytes(out)


def _frame_decode(data):
    pos, fields = 0, []
    for d in _FRAME:
        v, pos = _wire_get(_NO_TYPES, d, data, pos)
        fields.append(v)
    if pos != len(data):
        raise WireError("extra bytes after a frame")
    return fields


def _split_address(address):
    """'tcp://host:port/name' as ('tcp://host:port', 'name')."""
    node, _, name = address.rpartition("/")
    if not node or "://" not in node:
        raise ValueError(
            f"{address!r} is not an address such as tcp://127.0.0.1:7000/name"
        )
    return node, name


class Transport:
    """Moves frames between nodes. start(deliver) begins calling
    deliver(frame) for every frame that arrives; send(node, frame) sends one
    to the node at that address, best effort; close() stops."""

    address = None

    def start(self, deliver):
        raise NotImplementedError

    def send(self, node, frame):
        raise NotImplementedError

    def close(self):
        pass


class MemoryNetwork:
    """Nodes in one process, with faults for testing: each frame may be lost
    or duplicated, and is delayed by up to delay seconds (so frames can
    overtake each other); partition(...) cuts nodes off until heal()."""

    def __init__(self, seed=0, loss=0.0, duplicate=0.0, delay=0.0, record=False):
        self._random = SplitMix64(seed)
        self.loss, self.duplicate, self.delay = loss, duplicate, delay
        # With record, every record sent, as the network saw it.
        self.recorded = [] if record else None
        self._nodes = {}
        self._groups = None
        self._lock = threading.Lock()

    def transport(self, name):
        return _MemoryTransport(self, "mem://" + name)

    def insecure_transport_for_tests(self, name):
        """A transport whose node skips the handshake and sends frames in
        the clear: for tests of the frame layer only. Only an in-memory
        network makes one, and no configuration selects it."""
        return InsecureMemoryTransport(self, "mem://" + name)

    def partition(self, *groups):
        """Only nodes named in the same group reach each other."""
        with self._lock:
            self._groups = [set("mem://" + n for n in g) for g in groups]

    def heal(self):
        with self._lock:
            self._groups = None

    def _chance(self, p):
        return p > 0 and self._random.below(1 << 30) < p * (1 << 30)

    def _send(self, source, node, frame):
        with self._lock:
            if self.recorded is not None:
                self.recorded.append(bytes(frame))
            deliver = self._nodes.get(node)
            if deliver is None:
                raise Unreachable(f"no node at {node}")
            if self._groups is not None and not any(
                source in g and node in g for g in self._groups
            ):
                return
            if self._chance(self.loss):
                return
            copies = 2 if self._chance(self.duplicate) else 1
            delays = [
                self._random.below(1001) * self.delay / 1000 for _ in range(copies)
            ]
        for wait in delays:
            if wait <= 0:
                threading.Thread(target=deliver, args=(frame,), daemon=True).start()
            else:
                timer = threading.Timer(wait, deliver, args=(frame,))
                timer.daemon = True
                timer.start()


class _MemoryTransport(Transport):
    insecure_for_tests = False

    def __init__(self, network, address):
        self._network, self.address = network, address

    def start(self, deliver):
        with self._network._lock:
            self._network._nodes[self.address] = deliver

    def send(self, node, frame):
        self._network._send(self.address, node, frame)

    def close(self):
        with self._network._lock:
            self._network._nodes.pop(self.address, None)


class InsecureMemoryTransport(_MemoryTransport):
    """In memory, without the handshake: tests only."""

    insecure_for_tests = True


class TcpTransport(Transport):
    """Frames over TCP, each a 4-byte big-endian length then the frame.
    port 0 picks a free port; the address is tcp://host:port."""

    def __init__(self, host="127.0.0.1", port=0):
        import socket

        self._server = socket.create_server((host, port))
        self.address = f"tcp://{host}:{self._server.getsockname()[1]}"
        self._connections = {}
        self._lock = threading.Lock()
        self._closed = False

    def start(self, deliver):
        def accept():
            while not self._closed:
                try:
                    connection, _ = self._server.accept()
                except OSError:
                    return
                threading.Thread(
                    target=self._read, args=(connection, deliver), daemon=True
                ).start()

        threading.Thread(target=accept, daemon=True).start()

    @staticmethod
    def _read(connection, deliver):
        def exactly(n):
            data = bytearray()
            while len(data) < n:
                chunk = connection.recv(n - len(data))
                if not chunk:
                    return None
                data.extend(chunk)
            return bytes(data)

        with connection:
            while True:
                header = exactly(4)
                if header is None:
                    return
                frame = exactly(int.from_bytes(header, "big"))
                if frame is None:
                    return
                deliver(frame)

    def send(self, node, frame):
        import socket

        host, _, port = node[len("tcp://") :].rpartition(":")
        data = len(frame).to_bytes(4, "big") + frame
        with self._lock:
            for attempt in range(2):
                connection = self._connections.get(node)
                try:
                    if connection is None:
                        connection = socket.create_connection(
                            (host, int(port)), timeout=5
                        )
                        self._connections[node] = connection
                    connection.sendall(data)
                    return
                except OSError as error:
                    self._connections.pop(node, None)
                    if attempt == 1:
                        raise Unreachable(f"cannot reach {node}: {error}") from None

    def close(self):
        self._closed = True
        self._server.close()
        with self._lock:
            for connection in self._connections.values():
                connection.close()
            self._connections.clear()


class HttpTransport(Transport):
    """Frames as HTTP POST bodies to /lawspec; the address is http://host:port."""

    def __init__(self, host="127.0.0.1", port=0):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        transport = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(204 if self.path == "/lawspec" else 404)
                self.end_headers()
                if self.path == "/lawspec" and transport._deliver is not None:
                    transport._deliver(body)

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        self.address = f"http://{host}:{self._server.server_address[1]}"
        self._deliver = None

    def start(self, deliver):
        self._deliver = deliver
        threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        ).start()

    def send(self, node, frame):
        import urllib.request

        request = urllib.request.Request(
            node + "/lawspec",
            data=frame,
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        try:
            with urllib.request.urlopen(request, timeout=5):
                pass
        except OSError as error:
            raise Unreachable(f"cannot reach {node}: {error}") from None

    def close(self):
        self._server.shutdown()
        self._server.server_close()


# The secure network handler lives in lawspec_network, which the compiler
# writes beside this runtime when a program imports lawspec.network: it needs
# the crypto libraries, which programs without nodes do without. It
# registers itself here when imported (register_secure_network).
_SECURE_NETWORK = [None]


def register_secure_network(provider):
    """provider.layer(node, identity, trusted) makes a node's secure layer:
    send(peer, frame) and receive(record) -> frame or None."""
    _SECURE_NETWORK[0] = provider


def _secure_network():
    if _SECURE_NETWORK[0] is None:
        try:
            import lawspec_network  # noqa: F401  (registers itself)
        except ImportError as error:
            if getattr(error, "name", None) not in (None, "lawspec_network"):
                raise
    if _SECURE_NETWORK[0] is None:
        raise LookupError(
            "a node needs the secure network handler: add `import lawspec.network` "
            "to a unit of the program, so lawspec_network is generated"
        )
    return _SECURE_NETWORK[0]


def _secure_random(n):
    import os

    return os.urandom(n)


def secure_token():
    """A one-time token from the operating system's secure generator: 32
    bytes as 64 hexadecimal digits, as SecureRandom's secureToken gives."""
    return _secure_random(32).hex()


class Node:
    """A process's presence on a network: it names local mailboxes, actors,
    channel ends and definitions, so other nodes can reach them at
    <node address>/<name>, and it sends to theirs.

    Order is kept within one channel; a mailbox or an actor call is best
    effort: a lost call fails with Unreachable after its timeout.

    Every target's node speaks the same frames, so nodes written in
    different languages talk to each other.
    ref:DEC-distribution-canonical-wire"""

    def __init__(self, transport, identity=None, trusted=None):
        """identity: a NodeIdentity (by default the one lawspec.edn binds,
        or a fresh one); trusted: the fingerprints of the only peers to talk
        to (by default any peer, each address keeping the first identity it
        shows). A transport made for tests only (insecure_for_tests) skips
        the handshake; no other transport can."""
        self.transport = transport
        self.address = transport.address
        self._secure = (
            None
            if getattr(transport, "insecure_for_tests", False)
            else _secure_network().layer(self, identity, trusted)
        )
        self.identity = None if self._secure is None else self._secure.identity
        self._entities = {}
        self._pending = {}
        # Requests already seen, by sender and id, with their reply once
        # sent: a request sent again (lost reply, duplicated frame) is
        # answered again without running twice.
        self._seen = {}
        self._ids = iter(range(1, 1 << 62))
        self._lock = threading.Lock()
        self.closed = False
        transport.start(self._arrive)

    def _arrive(self, record):
        if self._secure is None:
            self._deliver(record)
            return
        frame = self._secure.receive(record)
        if frame is not None:
            self._deliver(frame)

    def _transmit(self, node, frame):
        if self._secure is None:
            self.transport.send(node, frame)
        else:
            self._secure.send(node, frame)

    def close(self):
        self.closed = True
        self.transport.close()

    def _next_id(self):
        with self._lock:
            return next(self._ids)

    def _send(self, address, kind, payload, ident=0):
        node, name = _split_address(address)
        self._transmit(node, _frame_encode(kind, name, self.address, ident, payload))

    def _forward(self, address, kind, source, ident, payload):
        """Passes a frame on to address unchanged, keeping its source."""
        node, name = _split_address(address)
        try:
            self._transmit(node, _frame_encode(kind, name, source, ident, payload))
        except Unreachable:
            pass

    def _register(self, name, entity):
        if "/" in name or not name:
            raise ValueError(f"{name!r} is not a name: use letters, digits and dashes")
        with self._lock:
            if name in self._entities:
                raise ValueError(f"{name} is already registered on {self.address}")
            self._entities[name] = entity
        return f"{self.address}/{name}"

    def _deliver(self, frame):
        try:
            kind, to, source, ident, payload = _frame_decode(frame)
        except WireError:
            return
        if kind == "reply":
            with self._lock:
                slot = self._pending.pop(ident, None)
            if slot is not None:
                slot.append(payload)
                slot.event.set()
            return
        entity = self._entities.get(to)
        if entity is None:
            if ident:
                self._reply(
                    source, ident, 3, f"nothing is registered as {to} on {self.address}"
                )
            return
        if ident:
            key = (source, ident)
            with self._lock:
                if key in self._seen:
                    answer = self._seen[key]
                    if answer is not None:
                        threading.Thread(
                            target=self._send_reply,
                            args=(source, ident, answer),
                            daemon=True,
                        ).start()
                    return
                self._seen[key] = None
                if len(self._seen) > 10000:
                    for old in list(self._seen)[:5000]:
                        del self._seen[old]
        # Handled off the transport's thread, so a slow handler does not
        # hold up other frames.
        threading.Thread(
            target=entity._receive,
            args=(self, kind, source, ident, payload),
            daemon=True,
        ).start()

    def _reply(self, source, ident, status, body):
        payload = bytes([status]) + (
            body if isinstance(body, bytes) else body.encode("utf-8")
        )
        with self._lock:
            if (source, ident) in self._seen:
                self._seen[(source, ident)] = payload
        self._send_reply(source, ident, payload)

    def _send_reply(self, source, ident, payload):
        # A retried request can reach us after its caller received the first
        # reply and closed. Original and cached replies have the same lifetime.
        try:
            self._send(source + "/", "reply", payload, ident)
        except Unreachable:
            pass

    def _request(self, address, kind, payload, timeout):
        """Sends a request and waits for its reply: (status, body)."""
        ident = self._next_id()
        slot = _Reply()
        with self._lock:
            self._pending[ident] = slot
        import time

        # Sent again until answered: the receiver runs it once.
        give_up = time.monotonic() + timeout
        while True:
            try:
                self._send(address, kind, payload, ident)
            except Unreachable:
                pass
            if slot.event.wait(min(0.1, max(0.0, give_up - time.monotonic()))):
                return slot[0][0], slot[0][1:]
            if time.monotonic() >= give_up:
                with self._lock:
                    self._pending.pop(ident, None)
                raise Unreachable(f"{address} did not answer within {timeout}s")

    # Mailboxes: values of one type sent by any node.
    def mailbox(self, name, descriptor, values=_NO_TYPES):
        """A local Mailbox that other nodes send to at <address>/name."""
        box = Mailbox()
        self._register(name, _MailEntity(box, values, descriptor))
        return box

    def remote_mailbox(self, address, descriptor, values=_NO_TYPES, timeout=5.0):
        return RemoteMailbox(self, address, descriptor, values, timeout)

    # Actors: calls by message name, with each message's types.
    def serve(self, name, actor, handlers, values=_NO_TYPES):
        """Lets other nodes call actor at <address>/name. handlers maps a
        message name to (handler(state, *args) -> (reply, state), argument
        descriptors, reply descriptor)."""
        return self._register(name, _ActorEntity(actor, handlers, values))

    def remote_actor(self, address, signatures, values=_NO_TYPES, timeout=5.0):
        """A proxy calling the actor at address; signatures maps a message
        name to (argument descriptors, reply descriptor)."""
        return RemoteActor(self, address, signatures, values, timeout)

    # Definitions, by content hash.
    def serve_definitions(self, table, values=_NO_TYPES, name="definitions"):
        """Lets other nodes evaluate definitions: table maps a content hash
        to (function, argument descriptors, result descriptor)."""
        return self._register(name, _DefinitionEntity(table, values))

    def evaluate(
        self,
        node,
        digest,
        args,
        arguments,
        result,
        values=_NO_TYPES,
        timeout=5.0,
        name="definitions",
    ):
        """Evaluates the definition with this content hash on another node."""
        payload = bytearray()
        _wire_put(_NO_TYPES, ["text"], digest, payload)
        for d, v in zip(arguments, args):
            _wire_put(values, d, v, payload)
        status, body = self._request(f"{node}/{name}", "eval", bytes(payload), timeout)
        return _reply_value(status, body, values, result)

    # Channels: one side here, the other on any node.
    def listen(self, name, steps, values=_NO_TYPES, deadline=5.0):
        """The first end of a channel named name here; its other end is
        dial(...)ed from any node. steps: (sends, descriptor) per step, from
        this end's side."""
        endpoint = _NetEndpoint(self, steps, values, 0, deadline)
        endpoint.address = self._register(name, endpoint)
        return endpoint

    def dial(self, address, steps, values=_NO_TYPES, deadline=5.0):
        """The second end of the channel listening at address. steps are from
        this end's side."""
        endpoint = _NetEndpoint(self, steps, values, 1, deadline)
        endpoint.address = self._register(f"end-{self._next_id()}", endpoint)
        endpoint._connect(address)
        return endpoint

    def take(self, address, steps, values=_NO_TYPES, deadline=5.0):
        """Takes over a channel end another node moves here: address is
        <old address>?take=<token>, as that node offered it. Returns once the
        end's state has arrived and its peer has been told (or after the
        deadline; the old node then forwards to the end)."""
        endpoint = _NetEndpoint(self, steps, values, 0, deadline)
        endpoint.address = self._register(f"end-{self._next_id()}", endpoint)
        endpoint._take_over(address)
        return endpoint


def _reply_value(status, body, values, d):
    if status == 0:
        return wire_decode(values, d, body)
    message = body.decode("utf-8", "replace")
    if status == 1:
        raise ActorCrashed(message)
    if status == 2:
        raise ActorStopped(message)
    raise Unreachable(message)


class _MailEntity:
    def __init__(self, box, values, descriptor):
        self.box, self.values, self.descriptor = box, values, descriptor

    def _receive(self, node, kind, source, ident, payload):
        if kind == "mail":
            try:
                self.box.send(wire_decode(self.values, self.descriptor, payload))
                status, body = 0, b""
            except WireError as error:
                status, body = 3, f"not a message of this mailbox: {error}"
            except ActorStopped as error:
                status, body = 2, str(error)
            if ident:
                node._reply(source, ident, status, body)


class RemoteMailbox:
    """Sends to a mailbox on another node. A send waits until the mailbox
    has the message (resending a lost one; the mailbox takes it once), and
    raises Unreachable after the timeout, or ActorStopped if it is closed."""

    def __init__(self, node, address, descriptor, values, timeout=5.0):
        self._node, self.address, self._descriptor, self._values = (
            node,
            address,
            descriptor,
            values,
        )
        self.timeout = timeout

    def send(self, value):
        status, body = self._node._request(
            self.address,
            "mail",
            wire_encode(self._values, self._descriptor, value),
            self.timeout,
        )
        if status != 0:
            _reply_value(status, body, self._values, ["unit"])


class _ActorEntity:
    def __init__(self, actor, handlers, values):
        self.actor, self.handlers, self.values = actor, handlers, values

    def _receive(self, node, kind, source, ident, payload):
        if kind != "call":
            return
        try:
            message, pos = _wire_get(_NO_TYPES, ["text"], payload, 0)
            handler, arguments, reply = self.handlers[message]
            args = []
            for d in arguments:
                v, pos = _wire_get(self.values, d, payload, pos)
                args.append(v)
            if pos != len(payload):
                raise WireError("extra bytes after the arguments")
        except (WireError, KeyError) as error:
            node._reply(source, ident, 3, f"not a message this actor handles: {error}")
            return
        try:
            result = self.actor.call(lambda s: handler(s, *args))
            node._reply(source, ident, 0, wire_encode(self.values, reply, result))
        except ActorCrashed as error:
            node._reply(source, ident, 1, str(error))
        except ActorStopped as error:
            node._reply(source, ident, 2, str(error))


class RemoteActor:
    """Calls an actor on another node: call(message, *args) sends the
    message and waits for the reply, raising Unreachable after the timeout,
    or what the actor's call raised (ActorCrashed, ActorStopped)."""

    def __init__(self, node, address, signatures, values, timeout):
        self._node, self.address = node, address
        self._signatures, self._values, self.timeout = signatures, values, timeout

    def call(self, message, *args):
        arguments, reply = self._signatures[message]
        payload = bytearray()
        _wire_put(_NO_TYPES, ["text"], message, payload)
        for d, v in zip(arguments, args):
            _wire_put(self._values, d, v, payload)
        status, body = self._node._request(
            self.address, "call", bytes(payload), self.timeout
        )
        return _reply_value(status, body, self._values, reply)


class _DefinitionEntity:
    def __init__(self, table, values):
        self.table, self.values = table, values

    def _receive(self, node, kind, source, ident, payload):
        if kind != "eval":
            return
        try:
            digest, pos = _wire_get(_NO_TYPES, ["text"], payload, 0)
            function, arguments, result = self.table[digest]
            args = []
            for d in arguments:
                v, pos = _wire_get(self.values, d, payload, pos)
                args.append(v)
        except (WireError, KeyError):
            node._reply(
                source, ident, 3, "this node has no definition with that content hash"
            )
            return
        try:
            node._reply(
                source, ident, 0, wire_encode(self.values, result, function(*args))
            )
        except Exception as error:  # noqa: BLE001 - reported to the caller
            node._reply(source, ident, 1, f"{type(error).__name__}: {error}")


_SEQ = ["int", "Int64", None, None]
_TEXT = ["text"]
_BYTES = ["bytes"]


def _put_texts(out, texts):
    _put_varint(out, len(texts))
    for t in texts:
        _wire_put(_NO_TYPES, _TEXT, t, out)


def _get_texts(buf, pos):
    count, pos = _get_varint(buf, pos)
    texts = []
    for _ in range(count):
        t, pos = _wire_get(_NO_TYPES, _TEXT, buf, pos)
        texts.append(t)
    return texts, pos


def _put_numbered(out, items):
    _put_varint(out, len(items))
    for seq, body in items:
        _wire_put(_NO_TYPES, _SEQ, seq, out)
        _wire_put(_NO_TYPES, _BYTES, body, out)


def _get_numbered(buf, pos):
    count, pos = _get_varint(buf, pos)
    items = []
    for _ in range(count):
        seq, pos = _wire_get(_NO_TYPES, _SEQ, buf, pos)
        body, pos = _wire_get(_NO_TYPES, _BYTES, buf, pos)
        items.append((seq, bytes(body)))
    return items, pos


class _NetEndpoint:
    """One end of a channel between nodes, with the Channel interface
    (send(side, value), receive(side)). Each value travels in a numbered
    frame that is sent again until acknowledged, so loss, duplication and
    reordering are repaired; a peer silent for deadline seconds is treated
    as failed (PeerFailed). Order is kept within the channel.

    An unused end can move to another node: offer() gives the address the
    new node takes it over from (<address>?take=<token>). On a take frame
    with that token, this end hands its state over (a state frame) and from
    then on forwards every frame it gets to the new end; the new end tells
    the peer (a moved frame) so the peer sends to it directly."""

    def __init__(self, node, steps, values, side, deadline):
        import queue

        self._node, self._steps, self._values, self.side = node, steps, values, side
        self._deadline = deadline
        self.address = None
        self._peer = None
        self._peer_known = threading.Event()
        self._out = 0
        self._unacked = {}
        self._expected = 0
        self._early = {}
        self._inbox = queue.Queue()
        self._lock = threading.Lock()
        self._step = 0
        self._gone = False
        self._failure = None
        # Moving: the addresses this end had before (oldest first), the
        # token a taker must show, where the end went and the state frame
        # it was given, and, on the new node, the takeover in progress.
        self._history = []
        self._token = None
        self._moved_to = None
        self._state = None
        self._taking = None
        self._taken = threading.Event()
        self._announcing = False
        self._announced_at = 0.0
        self._confirmed = threading.Event()
        threading.Thread(target=self._resend, daemon=True).start()

    def _connect(self, address):
        with self._lock:
            self._peer = address
        self._peer_known.set()
        self._transmit(-1, b"hello")

    def _frame(self, seq, body):
        payload = bytearray()
        _wire_put(_NO_TYPES, _SEQ, seq, payload)
        _wire_put(_NO_TYPES, _TEXT, self.address, payload)
        payload.extend(body)
        return bytes(payload)

    def _transmit(self, seq, body):
        """Sends a numbered frame (seq -1 is the hello) until it is acked."""
        import time

        payload = self._frame(seq, body)
        with self._lock:
            self._unacked[seq] = [payload, time.monotonic(), time.monotonic(), body]
            peer = self._peer
        if peer is not None:
            try:
                self._node._send(peer, "chan", payload)
            except Unreachable:
                pass

    def _resend(self):
        import time

        while not self._gone and not self._node.closed:
            time.sleep(0.02)
            now = time.monotonic()
            with self._lock:
                if self._moved_to is not None:
                    return
                if self._taking is not None and not self._taken.is_set():
                    continue
                peer = self._peer
                due = [
                    (seq, entry)
                    for seq, entry in self._unacked.items()
                    if now - entry[2] > 0.05
                ]
                stale = any(now - entry[1] > self._deadline for _, entry in due)
                announce = (
                    self._announcing
                    and peer is not None
                    and not self._confirmed.is_set()
                    and now - self._announced_at > 0.05
                )
                if announce:
                    self._announced_at = now
                    moved = bytearray()
                    _put_texts(moved, self._history)
                    _wire_put(_NO_TYPES, _TEXT, self.address, moved)
            if stale:
                self._fail("the other end did not answer in time (unreachable)")
                return
            if peer is None:
                continue
            if announce:
                try:
                    self._node._send(peer, "moved", bytes(moved))
                except Unreachable:
                    pass
            for seq, entry in due:
                entry[2] = now
                try:
                    self._node._send(peer, "chan", entry[0])
                except Unreachable:
                    pass

    def _fail(self, reason):
        with self._lock:
            if self._gone:
                return
            self._gone = True
            self._failure = reason
            self._unacked.clear()
            self._inbox.put((_ABANDONED, reason))

    def _receive(self, node, kind, source, ident, payload):
        if kind == "take":
            self._give(payload)
            return
        with self._lock:
            forward = self._moved_to
            waiting = self._taking is not None and not self._taken.is_set()
        if forward is not None:
            self._node._forward(forward, kind, source, ident, payload)
            return
        if waiting:
            # Until the state arrives, frames are dropped: their senders send
            # them again.
            if kind == "state":
                self._install(payload)
            return
        if kind == "ack":
            seq, _ = _wire_get(_NO_TYPES, _SEQ, payload, 0)
            with self._lock:
                self._unacked.pop(seq, None)
            return
        if kind == "moved":
            self._peer_moved(payload)
            return
        if kind == "moved-ack":
            to, _ = _wire_get(_NO_TYPES, _TEXT, payload, 0)
            if to == self.address:
                self._confirmed.set()
            return
        if kind != "chan":
            return
        seq, pos = _wire_get(_NO_TYPES, _SEQ, payload, 0)
        sender, pos = _wire_get(_NO_TYPES, _TEXT, payload, pos)
        body = payload[pos:]
        with self._lock:
            forward = self._moved_to
            if forward is None:
                if seq == -1:
                    if self._peer is None:
                        self._peer = sender
                    self._peer_known.set()
                elif seq >= self._expected and seq not in self._early:
                    self._early[seq] = body
                    while self._expected in self._early:
                        self._inbox.put((None, self._early.pop(self._expected)))
                        self._expected += 1
        if forward is not None:
            # Moved meanwhile: the new end acknowledges it.
            self._node._forward(forward, kind, source, ident, payload)
            return
        ack = bytearray()
        _wire_put(_NO_TYPES, _SEQ, seq, ack)
        try:
            self._node._send(sender, "ack", bytes(ack))
        except Unreachable:
            pass

    # Moving an end to another node.
    def _offer(self):
        """The address another node takes this unused end over from."""
        with self._lock:
            if self._token is None:
                self._token = secure_token()
            return f"{self.address}?take={self._token}"

    def _give(self, payload):
        """A take frame: hands the state over once, to the first taker with
        the token, and answers that taker's repeats with the same state."""
        try:
            token, pos = _wire_get(_NO_TYPES, _TEXT, payload, 0)
            taker, _ = _wire_get(_NO_TYPES, _TEXT, payload, pos)
        except WireError:
            return
        with self._lock:
            if self._token is None or token != self._token:
                return
            if self._moved_to is None:
                received = [
                    body for marker, body in list(self._inbox.queue) if marker is None
                ]
                state = bytearray()
                _wire_put(_NO_TYPES, _TEXT, token, state)
                _wire_put(_NO_TYPES, _TEXT, self._failure or "", state)
                _wire_put(_NO_TYPES, _TEXT, self._peer or "", state)
                _put_texts(state, self._history + [self.address])
                _wire_put(_NO_TYPES, _SEQ, self._out, state)
                _wire_put(_NO_TYPES, _SEQ, self._expected, state)
                _put_numbered(
                    state,
                    sorted((seq, entry[3]) for seq, entry in self._unacked.items()),
                )
                _put_numbered(state, sorted(self._early.items()))
                _put_varint(state, len(received))
                for body in received:
                    _wire_put(_NO_TYPES, _BYTES, body, state)
                self._moved_to, self._state = taker, bytes(state)
                self._unacked.clear()
                self._early.clear()
            elif self._moved_to != taker:
                return
            state = self._state
        try:
            self._node._send(taker, "state", state)
        except Unreachable:
            pass

    def _take_over(self, address):
        """Takes over the end offered at address (<old address>?take=<token>):
        asks for its state until it comes, then tells the peer where the end
        is now. Returns once the peer knows, or after the deadline (the old
        node then keeps forwarding to this end, as a relay would)."""
        import time

        old, _, query = address.partition("?")
        token = query[len("take=") :]
        with self._lock:
            self._taking = (old, token)
        request = bytearray()
        _wire_put(_NO_TYPES, _TEXT, token, request)
        _wire_put(_NO_TYPES, _TEXT, self.address, request)
        give_up = time.monotonic() + self._deadline
        while not self._taken.is_set():
            try:
                self._node._send(old, "take", bytes(request))
            except Unreachable:
                pass
            if self._taken.wait(0.05):
                break
            if time.monotonic() >= give_up:
                with self._lock:
                    self._taking = None
                self._fail(
                    "the node the end came from did not hand it over in time (unreachable)"
                )
                return
        self._confirmed.wait(max(0.0, give_up - time.monotonic()))

    def _install(self, payload):
        import time

        try:
            token, pos = _wire_get(_NO_TYPES, _TEXT, payload, 0)
            failure, pos = _wire_get(_NO_TYPES, _TEXT, payload, pos)
            peer, pos = _wire_get(_NO_TYPES, _TEXT, payload, pos)
            history, pos = _get_texts(payload, pos)
            out, pos = _wire_get(_NO_TYPES, _SEQ, payload, pos)
            expected, pos = _wire_get(_NO_TYPES, _SEQ, payload, pos)
            unacked, pos = _get_numbered(payload, pos)
            early, pos = _get_numbered(payload, pos)
            count, pos = _get_varint(payload, pos)
            received = []
            for _ in range(count):
                body, pos = _wire_get(_NO_TYPES, _BYTES, payload, pos)
                received.append(bytes(body))
        except WireError:
            return
        now = time.monotonic()
        with self._lock:
            if self._taking is None or token != self._taking[1] or self._taken.is_set():
                return
            self._peer = peer or None
            self._history = history
            self._out, self._expected = out, expected
            # Sent again from here at once, under this end's address.
            self._unacked = {
                seq: [self._frame(seq, body), now, 0.0, body] for seq, body in unacked
            }
            self._early = dict(early)
            for body in received:
                self._inbox.put((None, body))
            self._announcing = True
            self._taken.set()
        if peer:
            self._peer_known.set()
        if failure:
            self._fail(failure)

    def _peer_moved(self, payload):
        """The peer moved: from now on send to its new address."""
        try:
            history, pos = _get_texts(payload, 0)
            to, _ = _wire_get(_NO_TYPES, _TEXT, payload, pos)
        except WireError:
            return
        with self._lock:
            if self._peer is None or self._peer in history:
                self._peer = to
            known = self._peer == to
        if known:
            answer = bytearray()
            _wire_put(_NO_TYPES, _TEXT, to, answer)
            try:
                self._node._send(to, "moved-ack", bytes(answer))
            except Unreachable:
                pass

    def _step_descriptor(self, sends):
        if self._step >= len(self._steps):
            raise SessionError("this channel's protocol has ended")
        step_sends, d = self._steps[self._step]
        if step_sends != sends:
            raise SessionError(
                "this step "
                + ("receives" if step_sends == 0 or not step_sends else "sends")
            )
        self._step += 1
        return d

    def send(self, side, value):
        if self._gone:
            raise PeerFailed("the other end has failed")
        d = self._step_descriptor(True)
        body = bytearray([0])
        _wire_put(self._values, d, value, body)
        with self._lock:
            seq = self._out
            self._out += 1
        self._transmit(seq, bytes(body))

    def receive(self, side, timeout=None):
        import queue

        d = self._step_descriptor(False)
        try:
            marker, body = self._inbox.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("no message arrived in time") from None
        if marker is _ABANDONED:
            self._inbox.put((marker, body))
            raise PeerFailed(body)
        if body[0] == 1:
            self._fail("the other end gave up the conversation")
            raise PeerFailed(
                "the other end gave up the conversation (its process failed or abandoned it)"
            )
        return wire_decode(self._values, d, bytes(body[1:]))

    def abandon(self, side):
        """Gives up: the other end's receives fail after what was sent."""
        with self._lock:
            seq = self._out
            self._out += 1
        self._transmit(seq, b"\x01")


class EndPart:
    """A step that sends another protocol's first end: start() is that end's
    start class, and wire() that protocol's (steps, parts) from it."""

    def __init__(self, start, wire):
        self.start, self.wire = start, wire


class NativeChannel:
    """A network channel end seen through native values: each step's part
    converts its value with the schema (None needs no conversion), or, for
    an EndPart, sends a channel end and receives one. A network end moves
    to the receiving node (the value is <address>?take=<token>); a local
    end stays here behind a relay (the value is the relay's address)."""

    def __init__(self, endpoint, parts, schema=None):
        self._endpoint, self._parts, self._schema = endpoint, parts, schema
        self._step = 0
        self.address = endpoint.address

    def _part(self):
        part = self._parts[self._step] if self._step < len(self._parts) else None
        self._step += 1
        return part

    def send(self, side, value):
        part = self._part()
        if isinstance(part, EndPart):
            value = _offer_end(
                self._endpoint._node, self._endpoint._values, value, part, self._schema
            )
        elif part is not None:
            value = self._schema.from_native(part, value)
        self._endpoint.send(side, value)

    def receive(self, side):
        part = self._part()
        value = self._endpoint.receive(side)
        if isinstance(part, EndPart):
            steps, parts = part.wire()
            node = self._endpoint._node
            if "?take=" in value:
                endpoint = node.take(value, steps, self._endpoint._values)
            else:
                endpoint = node.dial(value, steps, self._endpoint._values)
            return part.start()(NativeChannel(endpoint, parts, self._schema), 0)
        return value if part is None else self._schema.to_native(part, value)

    def abandon(self, side):
        self._endpoint.abandon(side)


def _offer_end(node, values, end, part, schema):
    """The text that gives an unused channel end to another node. An end
    that is itself between nodes moves there; a local end stays here and a
    relay on node carries its conversation."""
    channel, side = end._take(), end._side
    if isinstance(channel, NativeChannel) and channel._step == 0:
        return channel._endpoint._offer()
    return _relay_end(node, values, channel, side, part, schema)


def _relay_end(node, values, channel, side, part, schema):
    """Offers a local channel end to another node: a relay on node listens
    for the receiver and passes each step between it and the end, which
    stays here. Returns the relay's address. A failure on either side gives
    up the other."""
    steps, parts = part.wire()
    relay = node.listen(
        f"relay-{node._next_id()}", [(not s, d) for s, d in steps], values
    )
    relayed = NativeChannel(relay, parts, schema)

    def pump():
        try:
            for sends, _ in steps:
                if sends:
                    channel.send(side, relayed.receive(0))
                else:
                    relayed.send(0, channel.receive(side))
        except (PeerFailed, SessionError, Unreachable, TimeoutError, ActorStopped):
            for give_up in (lambda: channel.abandon(side), lambda: relay.abandon(0)):
                try:
                    give_up()
                except Exception:  # noqa: BLE001 - already failed
                    pass

    threading.Thread(target=pump, daemon=True).start()
    return relay.address
