"""Wireshark-flavoured display filter over PacketRecord fields.

A hand-written tokeniser and recursive-descent parser producing a small tree of
predicate objects.  Explicitly not `eval`: the filter bar is typed by the
operator and could be pasted from anywhere, and a capture tool that executes
arbitrary Python from its own UI is a much worse problem than a missing feature.

Grammar:

    expr    := or_expr
    or_expr := and_expr ( ("||" | "or") and_expr )*
    and_expr:= not_expr ( ("&&" | "and") not_expr )*
    not_expr:= ("!" | "not") not_expr | primary
    primary := "(" expr ")" | comparison | field
    comparison := FIELD OP VALUE
    OP      := "==" | "!=" | ">" | ">=" | "<" | "<=" | "contains" | "~"
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


class FilterError(ValueError):
    """Raised for any syntax or field error; the bar turns red, view unchanged."""


# --------------------------------------------------------------------------
# field access
# --------------------------------------------------------------------------

def _f(name: str):
    return lambda r: r.feature(name)


FIELDS: dict[str, object] = {
    # link layer
    "adva": lambda r: r.adva,
    "adva_kind": lambda r: r.adva_kind,
    "pdu_type": lambda r: r.pdu_name,
    "pdu": lambda r: r.pdu_name,
    "type": lambda r: r.pdu_name,
    "length": lambda r: r.length,
    "len": lambda r: r.length,
    "channel": lambda r: r.channel,
    "chan": lambda r: r.channel,
    "crc": lambda r: "pass" if r.crc_ok else "fail",
    "crc_ok": lambda r: r.crc_ok,
    "info": lambda r: r.short_info(),
    "tx_add": lambda r: "random" if r.tx_add_random else "public",
    "number": lambda r: r.number,
    "frame": lambda r: r.number,
    "time": lambda r: r.timestamp_us / 1e6,
    "payload": lambda r: r.payload.hex(),
    # radio
    "rssi": lambda r: r.rssi_dbfs,
    "rssi_dbfs": lambda r: r.rssi_dbfs,
    "rssi_dbm": lambda r: r.rssi_dbm,
    "snr": lambda r: r.snr_db,
    "gain": lambda r: r.gain_db,
    "temperature": lambda r: r.temperature_c,
    "calibrated": lambda r: r.calibrated,
    # analysis
    "cluster": lambda r: r.cluster_id,
    "anomaly": lambda r: r.anomaly_score,
    "alerts": lambda r: ";".join(r.alerts),
    "event": lambda r: r.event_kind if r.is_event else "",
    # physical-layer features
    "cfo_ppm": _f("cfo_ppm"),
    "cfo_hz": _f("cfo_hz"),
    "drift_hz": _f("drift_hz"),
    "drift_rate": _f("drift_rate"),
    "modulation_index": _f("modulation_index"),
    "mod_index": _f("modulation_index"),
    "bt": _f("effective_bt"),
    "symbol_clock_ppm": _f("symbol_clock_ppm"),
    "jitter_ps": _f("symbol_jitter_ps"),
    "rise_time_us": _f("rise_time_us"),
    "overshoot": _f("overshoot"),
    "delay_spread_us": _f("delay_spread_us"),
    "splatter_db": _f("splatter_db"),
    "eye": _f("eye_opening"),
    "isi": _f("residual_isi"),
    "freq_error_rms": _f("freq_error_rms"),
    "transition_asymmetry": _f("transition_asymmetry"),
    "aoa": _f("aoa_deg"),
    "aoa_deg": _f("aoa_deg"),
    "antenna_phase_deg": _f("antenna_phase_deg"),
    "antennas": lambda r: r.n_antennas,
}

FIELD_NAMES = tuple(sorted(FIELDS))


# --------------------------------------------------------------------------
# tokenising
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    \s*(?:
      (?P<lparen>\()
    | (?P<rparen>\))
    | (?P<op>==|!=|>=|<=|>|<|~)
    | (?P<and>&&|\band\b)
    | (?P<or>\|\||\bor\b)
    | (?P<not>!|\bnot\b)
    | (?P<kw>\bcontains\b)
    | (?P<str>"[^"]*"|'[^']*')
    | (?P<mac>(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2})
    | (?P<num>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
    | (?P<word>[A-Za-z_][A-Za-z0-9_./]*)
    )
    """,
    re.VERBOSE,
)


@dataclass
class Token:
    kind: str
    text: str
    pos: int


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        m = _TOKEN_RE.match(text, i)
        if not m or m.end() == m.start():
            raise FilterError(f"unexpected character {text[i]!r} at position {i}")
        kind = m.lastgroup
        tokens.append(Token(kind, m.group(kind).strip(), m.start(kind)))
        i = m.end()
    return tokens


# --------------------------------------------------------------------------
# predicate tree
# --------------------------------------------------------------------------

class Node:
    def __call__(self, rec) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class And(Node):
    a: Node
    b: Node

    def __call__(self, rec):
        return self.a(rec) and self.b(rec)

    def describe(self):
        return f"({self.a.describe()} && {self.b.describe()})"


@dataclass
class Or(Node):
    a: Node
    b: Node

    def __call__(self, rec):
        return self.a(rec) or self.b(rec)

    def describe(self):
        return f"({self.a.describe()} || {self.b.describe()})"


@dataclass
class Not(Node):
    a: Node

    def __call__(self, rec):
        return not self.a(rec)

    def describe(self):
        return f"!{self.a.describe()}"


@dataclass
class Truthy(Node):
    """A bare field name: true when the field is set / non-zero / non-empty."""

    field: str

    def __call__(self, rec):
        v = FIELDS[self.field](rec)
        if isinstance(v, float) and math.isnan(v):
            return False
        return bool(v)

    def describe(self):
        return self.field


@dataclass
class Compare(Node):
    field: str
    op: str
    value: object
    raw: str

    def __call__(self, rec):
        left = FIELDS[self.field](rec)
        return _compare(left, self.op, self.value)

    def describe(self):
        return f"{self.field} {self.op} {self.raw}"


def _compare(left, op: str, right) -> bool:
    # A missing measurement matches nothing except an explicit "!=".
    if isinstance(left, float) and math.isnan(left):
        return op == "!="
    if op in ("contains", "~"):
        return str(right).lower() in str(left).lower()

    if isinstance(left, bool):
        rb = _as_bool(right)
        if rb is None:
            return False
        return (left == rb) if op == "==" else (left != rb) if op == "!=" else False

    if isinstance(left, (int, float)) and not isinstance(right, str):
        r = float(right)
        l = float(left)
    elif isinstance(left, (int, float)) and isinstance(right, str):
        try:
            r = float(right)
            l = float(left)
        except ValueError:
            return False
    else:
        # string comparison, case-insensitive, and MAC-normalised
        l = str(left).upper()
        r = str(right).upper()
        if op == "==":
            return l == r
        if op == "!=":
            return l != r
        return False

    if op == "==":
        return l == r
    if op == "!=":
        return l != r
    if op == ">":
        return l > r
    if op == ">=":
        return l >= r
    if op == "<":
        return l < r
    if op == "<=":
        return l <= r
    return False


def _as_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "yes", "1", "pass", "ok"):
        return True
    if s in ("false", "no", "0", "fail"):
        return False
    return None


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.t = tokens
        self.i = 0

    def peek(self) -> Token | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self) -> Token:
        if self.i >= len(self.t):
            raise FilterError("unexpected end of expression")
        tok = self.t[self.i]
        self.i += 1
        return tok

    def parse(self) -> Node:
        node = self.parse_or()
        if self.i != len(self.t):
            tok = self.t[self.i]
            raise FilterError(f"unexpected {tok.text!r} at position {tok.pos}")
        return node

    def parse_or(self) -> Node:
        node = self.parse_and()
        while (tok := self.peek()) and tok.kind == "or":
            self.take()
            node = Or(node, self.parse_and())
        return node

    def parse_and(self) -> Node:
        node = self.parse_not()
        while (tok := self.peek()) and tok.kind == "and":
            self.take()
            node = And(node, self.parse_not())
        return node

    def parse_not(self) -> Node:
        tok = self.peek()
        if tok and tok.kind == "not":
            self.take()
            return Not(self.parse_not())
        return self.parse_primary()

    def parse_primary(self) -> Node:
        tok = self.take()
        if tok.kind == "lparen":
            node = self.parse_or()
            closing = self.peek()
            if not closing or closing.kind != "rparen":
                raise FilterError("unbalanced parenthesis")
            self.take()
            return node
        if tok.kind != "word":
            raise FilterError(f"expected a field name, got {tok.text!r}")

        field = tok.text.lower()
        if field not in FIELDS:
            raise FilterError(
                f"unknown field {tok.text!r}; try one of: "
                + ", ".join(FIELD_NAMES[:12])
                + ", ..."
            )

        nxt = self.peek()
        if nxt is None or nxt.kind in ("and", "or", "rparen"):
            return Truthy(field)
        if nxt.kind == "kw":  # contains
            self.take()
            val = self.take()
            return Compare(field, "contains", _literal(val), val.text)
        if nxt.kind != "op":
            raise FilterError(f"expected an operator after {tok.text!r}")
        op_tok = self.take()
        val_tok = self.take()
        if val_tok.kind not in ("num", "word", "str", "mac"):
            raise FilterError(f"expected a value after {op_tok.text!r}")
        return Compare(field, op_tok.text, _literal(val_tok), val_tok.text)


def _literal(tok: Token):
    if tok.kind == "num":
        return float(tok.text)
    if tok.kind == "str":
        return tok.text[1:-1]
    return tok.text


def compile_filter(text: str):
    """Compile a filter string to a callable, or None for an empty filter.

    Raises FilterError with a human-readable message for anything invalid; the
    caller turns the bar red and leaves the view alone.
    """
    if text is None or not text.strip():
        return None
    node = Parser(tokenize(text)).parse()

    def predicate(rec) -> bool:
        try:
            return bool(node(rec))
        except Exception:
            # A filter must never take the capture down.  An expression that
            # cannot be evaluated for a particular record simply excludes it.
            return False

    predicate.describe = node.describe  # type: ignore[attr-defined]
    return predicate
