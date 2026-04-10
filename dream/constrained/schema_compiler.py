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

Key ordering: powerset approach for required keys (2^n nonterminals).
Falls back to lexicographic fixed order for n > 10.

Public API:
    key_strings, grammar = compile_schema(schema)
    # key_strings: list of property keys the scanner must distinguish
    # grammar: Grammar for BoundedLRAutomaton
"""

from __future__ import annotations
from cfg import Grammar, Symbol, TERMINAL, NONTERMINAL
from scanner import T_STRING, T_NUMBER, T_INTEGER, T_TRUE, T_FALSE, T_NULL, T_KEY_BASE

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
        n = len(required)

        if n == 0:
            return self._compile_opt_suffix(optional, val_nts, allow_additional, hint, first=True)

        if n > 10:
            return self._compile_members_fixed(
                sorted(required), optional, val_nts, allow_additional, hint)

        # Powerset: nonterminal for each subset of remaining required keys
        subset_nts: dict[frozenset, int] = {}

        def get_sub_nt(rem: frozenset) -> int:
            if rem in subset_nts:
                return subset_nts[rem]
            tag = "_".join(sorted(rem)) if rem else "done"
            nt = self._nt(f"{hint}_req_{tag}")
            subset_nts[rem] = nt
            return nt

        full = frozenset(required)

        # Build rules bottom-up (empty set first).
        # Comma placement: comma is a LEADING separator on each non-first key.
        # req_{keys} nonterminals consume one key (with leading comma) and recur.
        # Optional keys may appear interspersed: each req_{keys} state also has
        # rules for optional keys (with leading comma), looping back to itself.
        for size in range(len(required) + 1):
            from itertools import combinations
            for combo in combinations(sorted(required), size):
                rem = frozenset(combo)
                nt = get_sub_nt(rem)
                if not rem:
                    # All required seen: optional suffix (may be empty)
                    opt = self._compile_opt_suffix(optional, val_nts, allow_additional, hint, first=False)
                    self._rule(nt, [_NT(opt)])
                else:
                    # Consume one required key (with leading comma)
                    for key in sorted(rem):
                        rest_nt = get_sub_nt(rem - {key})
                        kid = self._key_terminal.get(key)
                        if kid is None:
                            continue
                        val_nt = val_nts[key]
                        self._rule(nt, [_T(_COM), _T(kid), _T(_COL), _NT(val_nt), _NT(rest_nt)])
                    # Also allow optional keys to appear before remaining required keys
                    for key in optional:
                        kid = self._key_terminal.get(key)
                        if kid is None:
                            continue
                        val_nt = val_nts[key]
                        # Optional key with leading comma, then continue with same required set
                        self._rule(nt, [_T(_COM), _T(kid), _T(_COL), _NT(val_nt), _NT(nt)])

        # First-entry wrapper: no leading comma before the first key.
        # First key can be any required key OR any optional key.
        first_nt = self._fresh(f"{hint}_members")
        for key in sorted(full):
            rest_nt = get_sub_nt(full - {key})
            kid = self._key_terminal.get(key)
            if kid is None:
                continue
            val_nt = val_nts[key]
            self._rule(first_nt, [_T(kid), _T(_COL), _NT(val_nt), _NT(rest_nt)])
        # First key can also be an optional key
        for key in optional:
            kid = self._key_terminal.get(key)
            if kid is None:
                continue
            val_nt = val_nts[key]
            # After optional first key, still need all required keys (with leading commas)
            self._rule(first_nt, [_T(kid), _T(_COL), _NT(val_nt), _NT(get_sub_nt(full))])

        return first_nt

    def _compile_opt_suffix(self, optional, val_nts, allow_additional, hint, first: bool) -> int:
        """
        Nonterminal for the optional-key suffix after all required keys.

        Uses a powerset over optional keys (2^k nonterminals) to avoid
        right-recursive stack accumulation. Each nonterminal represents
        the set of optional keys still available to appear.

        With k optional keys, produces 2^k nonterminals each with at most
        k+1 rules (one per remaining optional key + epsilon). No recursion,
        so stack depth is bounded regardless of how many optional keys appear.

        Falls back to recursive encoding if k > 10 (to keep grammar size bounded).
        """
        k = len(optional)

        if k == 0:
            # No optional keys: just epsilon (possibly with additionalProperties)
            nt = self._fresh(f"{hint}_opt")
            self._rule(nt, [])
            if allow_additional is True:
                any_key = self._ensure_any_key_nt()
                any_val = self._ensure_any_value_nt()
                # additionalProperties: allow generic kv pairs with comma separator
                # Use a simple right-recursive rule here (additionalProperties is rare)
                add_nt = self._fresh(f"{hint}_add")
                self._rule(add_nt, [])
                if first:
                    self._rule(add_nt, [_NT(any_key), _T(_COL), _NT(any_val), _NT(add_nt)])
                else:
                    self._rule(add_nt, [_T(_COM), _NT(any_key), _T(_COL), _NT(any_val), _NT(add_nt)])
                self._rule(nt, [_NT(add_nt)])
            return nt

        if k > 10:
            # Fallback: right-recursive (may need deep stack, but rare)
            return self._compile_opt_suffix_recursive(optional, val_nts, allow_additional, hint, first)

        # Powerset: each state = frozenset of optional keys STILL AVAILABLE
        # (not yet emitted). Start state = all optional keys available.
        # At each state, can emit any available key (with comma prefix if non-first
        # or if any required/optional key was already emitted), then transition
        # to state with that key removed.
        # Epsilon is always valid (end of optional section).

        opt_nts: dict[frozenset, int] = {}

        def get_opt_nt(available: frozenset, is_first_entry: bool) -> int:
            cache_key = (available, is_first_entry)
            if cache_key in opt_nts:
                return opt_nts[cache_key]
            tag = "_".join(sorted(available)) if available else "empty"
            mode = "f" if is_first_entry else "nf"
            nt = self._nt(f"{hint}_avail_{tag}_{mode}")
            opt_nts[cache_key] = nt
            return nt

        # Build rules for all subsets, bottom-up
        from itertools import combinations
        all_subsets = []
        for size in range(k + 1):
            for combo in combinations(sorted(optional), size):
                all_subsets.append(frozenset(combo))

        for available in all_subsets:
            for is_first_entry in [True, False]:
                nt = get_opt_nt(available, is_first_entry)

                # Epsilon: no more optional keys
                self._rule(nt, [])

                # Emit each available optional key
                for key in sorted(available):
                    kid = self._key_terminal.get(key)
                    if kid is None:
                        continue
                    val_nt = val_nts[key]
                    rest_nt = get_opt_nt(available - {key}, False)  # after emitting, never first again

                    if is_first_entry:
                        # First entry in object: no leading comma
                        self._rule(nt, [_T(kid), _T(_COL), _NT(val_nt), _NT(rest_nt)])
                    else:
                        # Non-first: leading comma
                        self._rule(nt, [_T(_COM), _T(kid), _T(_COL), _NT(val_nt), _NT(rest_nt)])

                # additionalProperties (if allowed): keep simple for now, just epsilon
                # Full additionalProperties support with powerset would be 2^k * additional
                # We skip it here; it's not needed for additionalProperties: false schemas

        return get_opt_nt(frozenset(optional), first)

    def _compile_opt_suffix_recursive(self, optional, val_nts, allow_additional, hint, first: bool) -> int:
        """Right-recursive fallback for large optional sets (k > 10)."""
        nt = self._fresh(f"{hint}_opt_rec")
        self._rule(nt, [])
        for key in optional:
            val_nt = val_nts[key]
            kid = self._key_terminal.get(key)
            if kid is None:
                continue
            if first:
                self._rule(nt, [_T(kid), _T(_COL), _NT(val_nt), _NT(nt)])
            else:
                self._rule(nt, [_T(_COM), _T(kid), _T(_COL), _NT(val_nt), _NT(nt)])
        return nt

    def _ensure_any_key_nt(self) -> int:
        if "AnyKey" in self._nt_idx:
            return self._nt_idx["AnyKey"]
        nt = self._nt("AnyKey")
        self._rule(nt, [_T(T_STRING)])
        for key in self._key_strings:
            self._rule(nt, [_T(self._key_terminal[key])])
        return nt

    def _compile_members_fixed(self, required, optional, val_nts, allow_additional, hint) -> int:
        """Fixed lexicographic order fallback for large required sets (n > 10)."""
        nt = cur = self._fresh(f"{hint}_mfix")
        for i, key in enumerate(required):
            nxt = self._fresh(f"{hint}_af{i}")
            kid = self._key_terminal.get(key)
            val_nt = val_nts[key]
            if i == 0:
                # First key: no leading comma
                self._rule(cur, [_T(kid), _T(_COL), _NT(val_nt), _NT(nxt)])
            else:
                # Subsequent keys: leading comma (cur is "rest after previous key")
                self._rule(cur, [_T(_COM), _T(kid), _T(_COL), _NT(val_nt), _NT(nxt)])
            cur = nxt
        opt = self._compile_opt_suffix(optional, val_nts, allow_additional, hint, first=False)
        self._rule(cur, [_NT(opt)])
        return nt

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
    """Return (max_nesting_depth, max_optional_count_at_any_level)."""
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

    Formula: 4 * (max_nesting + max_optional + 2)
    - max_nesting: maximum object-in-object depth
    - max_optional: maximum number of optional properties at any object level
    - +2 base accounts for structural rules and single-key objects
    """
    nesting, n_opt = _schema_stats(schema)
    return 4 * (nesting + n_opt + 2)


def compile_schema(schema: dict) -> tuple[list[str], Grammar]:
    """
    Compile a JSON Schema to a token-level Grammar.

    Returns:
        (key_strings, grammar) where:
        - key_strings: ordered list of property keys the scanner must distinguish.
                       key_strings[i] maps to terminal T_KEY_BASE + i.
        - grammar: Grammar suitable for BoundedLRAutomaton.

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