"""
JSON Schema -> token-level CFG compiler.

Produces a Grammar where terminals are JSON token types (T_STRING, T_INTEGER,
T_KEY_BASE+i, structural chars) rather than raw bytes. The scanner (scanner.py)
maps byte sequences to these terminal IDs; the LR automaton (cfg.py) operates
on token-level terminals.

This keeps the LR state count small (O(grammar structure), not O(256)) and
makes BoundedLRAutomaton tractable for realistic JSON Schema grammars.

Supported schema features:
  - type: object, array, string, number, integer, boolean, null
  - properties, required, additionalProperties: false/true
  - items (uniform array element type)
  - enum (string and number literals — matched at scanner level as T_STRING
    or T_INTEGER/T_NUMBER; we use type constraints, not exact-value matching)
  - anyOf, oneOf

Key ordering: flat O(n) ambiguous grammar. Accepts any key in any order,
over-accepts duplicates and missing required keys. Post-generation validation
checks required-key presence and uniqueness. This is the right tradeoff:
2^n powerset encoding made configs intractable (548K+ for 5 keys); the flat
encoding keeps LR configs in the hundreds regardless of key count.

Public API:
    key_strings, grammar = compile_schema(schema)
    # key_strings: list of property keys the scanner must distinguish
    # grammar: Grammar for BoundedLRAutomaton

    required_keys = get_required_keys(schema)
    # For post-generation validation of required-key presence
"""

from __future__ import annotations
from constrained.cfg import Grammar, Symbol, TERMINAL, NONTERMINAL
from constrained.scanner import T_STRING, T_NUMBER, T_INTEGER, T_TRUE, T_FALSE, T_NULL, T_KEY_BASE

# Structural byte terminals (used directly in grammar rules)
_LB  = ord('{')
_RB  = ord('}')
_LA  = ord('[')
_RA  = ord(']')
_COL = ord(':')
_COM = ord(',')


def _T(byte_or_token: int) -> Symbol:
    return Symbol(TERMINAL, byte_or_token)

def _NT(nt_id: int) -> Symbol:
    return Symbol(NONTERMINAL, nt_id)


# ---------------------------------------------------------------------------
# Schema compiler
# ---------------------------------------------------------------------------

class _Compiler:
    """
    Accumulates nonterminals and rules while compiling a JSON Schema.

    Two passes:
      1. collect_keys(schema)  — gather all property key strings
      2. compile(schema)       — build grammar rules
    """

    def __init__(self) -> None:
        self._nts: list[str] = []
        self._nt_idx: dict[str, int] = {}
        self._rules: list[tuple[int, tuple[Symbol, ...]]] = []
        self._cache: dict[str, int] = {}
        # Key registry: maps key string -> terminal ID
        self._key_strings: list[str] = []
        self._key_terminal: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Key collection (pass 1)

    def collect_keys(self, schema: dict) -> None:
        """Walk schema tree and register all property keys."""
        if not isinstance(schema, dict):
            return
        for key in schema.get("properties", {}):
            if key not in self._key_terminal:
                tid = T_KEY_BASE + len(self._key_strings)
                self._key_strings.append(key)
                self._key_terminal[key] = tid
        for sub in schema.get("properties", {}).values():
            self.collect_keys(sub)
        for sub in schema.get("anyOf", []) + schema.get("oneOf", []):
            self.collect_keys(sub)
        if "items" in schema:
            self.collect_keys(schema["items"])

    # ------------------------------------------------------------------
    # Nonterminal management

    def _nt(self, name: str) -> int:
        if name not in self._nt_idx:
            self._nt_idx[name] = len(self._nts)
            self._nts.append(name)
        return self._nt_idx[name]

    def _fresh(self, prefix: str) -> int:
        name = f"{prefix}_{len(self._nts)}"
        return self._nt(name)

    def _rule(self, lhs: int, rhs: list[Symbol]) -> None:
        self._rules.append((lhs, tuple(rhs)))

    # ------------------------------------------------------------------
    # Schema compilation (pass 2)

    def compile(self, schema: dict, hint: str = "Root") -> int:
        import json as _j
        key = _j.dumps(schema, sort_keys=True)
        if key in self._cache:
            return self._cache[key]
        nt = self._compile_node(schema, hint)
        self._cache[key] = nt
        return nt

    def _compile_node(self, schema: dict, hint: str) -> int:
        if "anyOf" in schema or "oneOf" in schema:
            alts = schema.get("anyOf", schema.get("oneOf", []))
            nt = self._fresh(hint + "_union")
            for i, alt in enumerate(alts):
                child = self.compile(alt, f"{hint}_u{i}")
                self._rule(nt, [_NT(child)])
            return nt

        if "enum" in schema:
            return self._compile_enum(schema["enum"], hint)

        typ = schema.get("type", "any")
        if isinstance(typ, list):
            nt = self._fresh(hint + "_multi")
            for t in typ:
                child = self._compile_node({"type": t}, f"{hint}_{t}")
                self._rule(nt, [_NT(child)])
            return nt

        if typ == "string":
            return self._ensure_string_nt()
        elif typ == "number":
            return self._ensure_number_nt()
        elif typ == "integer":
            return self._ensure_integer_nt()
        elif typ == "boolean":
            return self._ensure_boolean_nt()
        elif typ == "null":
            return self._ensure_null_nt()
        elif typ == "object":
            return self._compile_object(schema, hint)
        elif typ == "array":
            return self._compile_array(schema, hint)
        else:  # "any" or unknown
            return self._ensure_any_value_nt()

    # ------------------------------------------------------------------
    # Leaf type nonterminals (singletons)

    def _ensure_string_nt(self) -> int:
        if "Str" in self._nt_idx:
            return self._nt_idx["Str"]
        nt = self._nt("Str")
        self._rule(nt, [_T(T_STRING)])
        # Also accept key tokens as strings (a key token in value position is a string)
        for key in self._key_strings:
            self._rule(nt, [_T(self._key_terminal[key])])
        return nt

    def _ensure_number_nt(self) -> int:
        if "Num" in self._nt_idx:
            return self._nt_idx["Num"]
        nt = self._nt("Num")
        self._rule(nt, [_T(T_NUMBER)])
        self._rule(nt, [_T(T_INTEGER)])  # integers are numbers
        return nt

    def _ensure_integer_nt(self) -> int:
        if "Int" in self._nt_idx:
            return self._nt_idx["Int"]
        nt = self._nt("Int")
        self._rule(nt, [_T(T_INTEGER)])
        return nt

    def _ensure_boolean_nt(self) -> int:
        if "Bool" in self._nt_idx:
            return self._nt_idx["Bool"]
        nt = self._nt("Bool")
        self._rule(nt, [_T(T_TRUE)])
        self._rule(nt, [_T(T_FALSE)])
        return nt

    def _ensure_null_nt(self) -> int:
        if "Null" in self._nt_idx:
            return self._nt_idx["Null"]
        nt = self._nt("Null")
        self._rule(nt, [_T(T_NULL)])
        return nt

    def _ensure_any_value_nt(self) -> int:
        if "AnyVal" in self._nt_idx:
            return self._nt_idx["AnyVal"]
        nt = self._nt("AnyVal")
        # Forward declare array and object to handle recursion
        any_obj = self._nt("AnyObj")
        any_arr = self._nt("AnyArr")
        self._rule(nt, [_T(T_STRING)])
        self._rule(nt, [_T(T_NUMBER)])
        self._rule(nt, [_T(T_INTEGER)])
        self._rule(nt, [_T(T_TRUE)])
        self._rule(nt, [_T(T_FALSE)])
        self._rule(nt, [_T(T_NULL)])
        self._rule(nt, [_NT(any_obj)])
        self._rule(nt, [_NT(any_arr)])
        for key in self._key_strings:
            self._rule(nt, [_T(self._key_terminal[key])])

        # AnyObj: { AnyMembers }
        any_members = self._nt("AnyMem")
        any_kv      = self._nt("AnyKV")
        any_kv_tail = self._nt("AnyKVT")
        any_key     = self._nt("AnyKey")  # any string token as a key
        self._rule(any_obj, [_T(_LB), _NT(any_members), _T(_RB)])
        self._rule(any_members, [])
        self._rule(any_members, [_NT(any_kv), _NT(any_kv_tail)])
        self._rule(any_kv_tail, [])
        self._rule(any_kv_tail, [_T(_COM), _NT(any_kv), _NT(any_kv_tail)])
        self._rule(any_kv, [_NT(any_key), _T(_COL), _NT(nt)])
        # any key = T_STRING or any T_KEY_BASE+i
        self._rule(any_key, [_T(T_STRING)])
        for key in self._key_strings:
            self._rule(any_key, [_T(self._key_terminal[key])])

        # AnyArr: [ AnyElems ]
        any_elems = self._nt("AnyElm")
        any_etail = self._nt("AnyET")
        self._rule(any_arr, [_T(_LA), _NT(any_elems), _T(_RA)])
        self._rule(any_elems, [])
        self._rule(any_elems, [_NT(nt), _NT(any_etail)])
        self._rule(any_etail, [])
        self._rule(any_etail, [_T(_COM), _NT(nt), _NT(any_etail)])
        return nt

    # ------------------------------------------------------------------
    # Enum

    def _compile_enum(self, values: list, hint: str) -> int:
        nt = self._fresh(hint + "_enum")
        for v in values:
            if isinstance(v, bool):
                self._rule(nt, [_T(T_TRUE if v else T_FALSE)])
            elif v is None:
                self._rule(nt, [_T(T_NULL)])
            elif isinstance(v, int):
                self._rule(nt, [_T(T_INTEGER)])
            elif isinstance(v, float):
                self._rule(nt, [_T(T_NUMBER)])
            elif isinstance(v, str):
                # If the string happens to match a known key, use its terminal
                if v in self._key_terminal:
                    self._rule(nt, [_T(self._key_terminal[v])])
                else:
                    self._rule(nt, [_T(T_STRING)])
            # else: skip (unsupported enum type)
        return nt

    # ------------------------------------------------------------------
    # Object

    def _compile_object(self, schema: dict, hint: str) -> int:
        props      = schema.get("properties", {})
        required   = list(schema.get("required", []))
        additional = schema.get("additionalProperties", True)

        # Compile value nonterminals for each known property
        val_nts: dict[str, int] = {
            key: self.compile(val_schema, f"{hint}_{key}")
            for key, val_schema in props.items()
        }
        optional = [k for k in props if k not in required]

        members = self._compile_members(required, optional, val_nts, additional, hint)

        obj = self._fresh(hint + "_obj")
        self._rule(obj, [_T(_LB), _NT(members), _T(_RB)])
        return obj

    def _compile_members(self, required, optional, val_nts, allow_additional, hint) -> int:
        """
        Flat O(n) members grammar. Accepts any key in any order, any number of times.

        Grammar:
            Members    → KeyValue | KeyValue COMMA Members
            KeyValue   → KEY_k1 COLON V1 | KEY_k2 COLON V2 | ...

        This is ambiguous (multiple derivations for any permutation, and it
        over-accepts duplicate/missing keys), but for constrained decoding we
        only need acceptance of structurally valid JSON with correct value types.
        Required-key presence and uniqueness are checked post-generation.

        If additionalProperties is true, KeyValue also includes a generic
        STRING COLON AnyValue alternative.
        """
        all_keys = list(required) + [k for k in optional if k not in required]

        if not all_keys and not allow_additional:
            # Empty object only: { }
            empty = self._fresh(f"{hint}_empty_members")
            self._rule(empty, [])
            return empty

        # KeyValue: one alternative per known key
        kv = self._fresh(f"{hint}_kv")
        for key in all_keys:
            kid = self._key_terminal.get(key)
            if kid is None:
                continue
            val_nt = val_nts[key]
            self._rule(kv, [_T(kid), _T(_COL), _NT(val_nt)])

        if allow_additional is True:
            any_key = self._ensure_any_key_nt()
            any_val = self._ensure_any_value_nt()
            self._rule(kv, [_NT(any_key), _T(_COL), _NT(any_val)])

        # Members: right-recursive list of KeyValue separated by commas
        # Members → KeyValue | KeyValue COMMA Members
        members = self._fresh(f"{hint}_members")
        self._rule(members, [_NT(kv)])
        self._rule(members, [_NT(kv), _T(_COM), _NT(members)])

        return members

    def _compile_opt_suffix(self, optional, val_nts, allow_additional, hint, first: bool) -> int:
        """
        No longer used — flat encoding in _compile_members handles required +
        optional keys uniformly. Kept as a no-op stub for any call sites that
        may remain; returns an epsilon nonterminal.
        """
        nt = self._fresh(f"{hint}_opt_stub")
        self._rule(nt, [])
        return nt

    def _compile_members_fixed(self, required, optional, val_nts, allow_additional, hint) -> int:
        """Unused — flat encoding replaces this. Delegates to _compile_members."""
        return self._compile_members(required, optional, val_nts, allow_additional, hint)

    def _compile_opt_suffix_recursive(self, optional, val_nts, allow_additional, hint, first: bool) -> int:
        """Unused — flat encoding replaces this."""
        return self._compile_opt_suffix(optional, val_nts, allow_additional, hint, first)

    # ------------------------------------------------------------------
    # Array

    def _compile_array(self, schema: dict, hint: str) -> int:
        items_schema = schema.get("items", {"type": "any"})
        elem_nt = self.compile(items_schema, f"{hint}_elem")

        elems = self._fresh(f"{hint}_elems")
        etail = self._fresh(f"{hint}_etail")
        self._rule(elems, [])
        self._rule(elems, [_NT(elem_nt), _NT(etail)])
        self._rule(etail, [])
        self._rule(etail, [_T(_COM), _NT(elem_nt), _NT(etail)])

        arr = self._fresh(f"{hint}_arr")
        self._rule(arr, [_T(_LA), _NT(elems), _T(_RA)])
        return arr

    # ------------------------------------------------------------------
    # Final Grammar assembly

    def build(self, start_nt: int) -> Grammar:
        # Reindex so start_nt is at position 0
        old_to_new: dict[int, int] = {}
        new_nts: list[str] = [self._nts[start_nt]]
        old_to_new[start_nt] = 0
        for old, name in enumerate(self._nts):
            if old == start_nt:
                continue
            old_to_new[old] = len(new_nts)
            new_nts.append(name)

        def remap(sym: Symbol) -> Symbol:
            return Symbol(NONTERMINAL, old_to_new[sym.value]) if sym.kind == NONTERMINAL else sym

        new_rules = [
            (old_to_new[lhs], tuple(remap(s) for s in rhs))
            for lhs, rhs in self._rules
        ]
        return Grammar(nonterminals=new_nts, rules=new_rules, start=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _schema_stats(schema: dict) -> tuple[int, int]:
    """Return (max_nesting_depth, max_optional_count_at_any_level).
    
    With the flat encoding, optional count no longer affects depth,
    but we keep it for informational purposes.
    """
    if not isinstance(schema, dict):
        return 0, 0
    typ = schema.get("type")
    if typ == "object":
        props = schema.get("properties", {})
        req = set(schema.get("required", []))
        n_opt = len(props) - len(req)
        sub_nest = sub_opt = 0
        for v in props.values():
            sn, so = _schema_stats(v)
            sub_nest = max(sub_nest, sn)
            sub_opt = max(sub_opt, so)
        return 1 + sub_nest, max(n_opt, sub_opt)
    if typ == "array":
        return _schema_stats(schema.get("items", {}))
    for key in ("anyOf", "oneOf"):
        if key in schema:
            results = [_schema_stats(s) for s in schema[key]]
            if results:
                return max(r[0] for r in results), max(r[1] for r in results)
    return 0, 0


def recommended_depth(schema: dict) -> int:
    """
    Compute the recommended BoundedLRAutomaton depth for a schema.

    With the flat O(n) members grammar:
      - Depth scales with nesting level, not key count or optional count.
      - Formula: 4 * (2 * max_nesting + 2)
        = 12 for nesting=1, 20 for nesting=2, 28 for nesting=3
      - Empirically validated: covers schemas with up to 5 keys per level.

    The +2 base accounts for the outer Object → { Members } structure.
    The *2 multiplier on nesting accounts for the right-recursive Members
    rule needing stack slots at each nesting level for both the outer
    Object frame and the inner Members recursion.
    """
    nesting, _ = _schema_stats(schema)
    return max(12, 4 * (2 * nesting + 2))


def get_required_keys(schema: dict) -> dict[str, list[str]]:
    """
    Extract required keys at each nesting level for post-generation validation.

    Returns a dict mapping a path string to a list of required key names.
    The path "" is the top-level object.

    Example:
        {"type": "object", "properties": {"name": ..., "addr": {"type": "object",
         "properties": {"city": ...}, "required": ["city"]}},
         "required": ["name", "addr"]}
        → {"": ["name", "addr"], "addr": ["city"]}

    Used by the eval script to check required-key presence after generation,
    since the flat grammar over-accepts (allows missing required keys).
    """
    result: dict[str, list[str]] = {}

    def walk(s, path):
        if not isinstance(s, dict):
            return
        if s.get("type") == "object":
            req = s.get("required", [])
            if req:
                result[path] = list(req)
            for key, val_schema in s.get("properties", {}).items():
                child_path = f"{path}.{key}" if path else key
                walk(val_schema, child_path)
        elif s.get("type") == "array":
            walk(s.get("items", {}), path)
        for k in ("anyOf", "oneOf"):
            for alt in s.get(k, []):
                walk(alt, path)

    walk(schema, "")
    return result


def compile_schema(schema: dict) -> tuple[list[str], Grammar]:
    """
    Compile a JSON Schema to a token-level Grammar.

    Returns:
        (key_strings, grammar) where:
        - key_strings: ordered list of property keys the scanner must distinguish.
                       key_strings[i] maps to terminal T_KEY_BASE + i.
        - grammar: Grammar suitable for BoundedLRAutomaton.

    The grammar accepts any key in any order and over-accepts (allows missing
    required keys, duplicate keys). Use get_required_keys(schema) to validate
    required-key presence post-generation.

    Usage:
        key_strings, grammar = compile_schema(schema)
        depth = recommended_depth(schema)
        scanner = JsonScanner(key_strings=key_strings)
        automaton = BoundedLRAutomaton(grammar, depth=depth)
    """
    c = _Compiler()
    c.collect_keys(schema)
    start_nt = c.compile(schema)
    grammar = c.build(start_nt)
    return c._key_strings, grammar