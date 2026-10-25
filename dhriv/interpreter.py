"""
interpreter.py — Runtime ad albero (tree-walking) del linguaggio Dhriv.

Esegue l'AST prodotto da `DhrivParser` collegandolo a:
- il registro OOP Tri-Assiale (`TriAxialOOPRegistry`): risoluzione Self -> Verticale -> Orizzontale -> Diagonale;
- il motore di ricerca su larga scala (`TriAxialLargeScaleSearch`);
- il motore di onde / elettromagnetismo (`wave_electromagnetic`);
- il motore di auto-apprendimento opzionale (memoizzazione appresa di funzioni pure).

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

from __future__ import annotations

import copy
import math
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .ast_nodes import (
    ASTNode,
    AssignStmtNode,
    BinaryOpNode,
    CallNode,
    EntityDeclNode,
    ExprStmtNode,
    ForStmtNode,
    FunctionDeclNode,
    IdentifierNode,
    IfStmtNode,
    IndexAccessNode,
    LetStmtNode,
    ListLiteralNode,
    LiteralNode,
    MapLiteralNode,
    MemberAccessNode,
    NewInstanceNode,
    ProgramNode,
    ReturnStmtNode,
    TriAxialOpNode,
    UnaryOpNode,
    WhileStmtNode,
)
from .parser import DhrivParser
from .triaxial_oop import (
    TriAxialClass,
    TriAxialLargeScaleSearch,
    TriAxialObject,
    TriAxialOOPRegistry,
)
from .wave_electromagnetic import (
    SPEED_OF_LIGHT_MPS,
    ElectromagneticBinaryModem,
    WaveSignal,
    analyze_spectrum,
    classify_em_band,
    free_space_path_loss_db,
    superpose_waves,
    synthesize_wave,
)


class DhrivRuntimeError(Exception):
    """Errore a tempo di esecuzione con riga sorgente (se nota)."""

    def __init__(self, message: str, line: Optional[int] = None) -> None:
        prefix = f"[Dhriv RuntimeError L{line}] " if line else "[Dhriv RuntimeError] "
        super().__init__(prefix + message)
        self.message = message
        self.line = line


class DhrivCancelled(DhrivRuntimeError):
    """Esecuzione interrotta dal WatchDog (timeout o stallo)."""


class _ReturnSignal(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


class Environment:
    def __init__(self, parent: Optional["Environment"] = None) -> None:
        self.vars: Dict[str, Any] = {}
        self.parent = parent

    def define(self, name: str, value: Any) -> None:
        self.vars[name] = value

    def lookup(self, name: str, line: Optional[int] = None) -> Any:
        env: Optional[Environment] = self
        while env is not None:
            if name in env.vars:
                return env.vars[name]
            env = env.parent
        raise DhrivRuntimeError(f"Variabile non definita: '{name}'", line)

    def assign(self, name: str, value: Any, line: Optional[int] = None) -> None:
        env: Optional[Environment] = self
        while env is not None:
            if name in env.vars:
                env.vars[name] = value
                return
            env = env.parent
        raise DhrivRuntimeError(f"Assegnazione a variabile non dichiarata: '{name}' (usa 'let')", line)


@dataclass
class DhrivFunction:
    node: FunctionDeclNode
    closure: Environment

    @property
    def name(self) -> str:
        return self.node.name


@dataclass
class BoundMethod:
    instance: TriAxialObject
    node: FunctionDeclNode
    closure: Environment


WAVE_PUBLIC_ATTRS = (
    "frequency_hz",
    "amplitude",
    "phase_rad",
    "sample_rate_hz",
    "wavelength_meters",
    "photon_energy_joules",
    "band_name",
    "wave_medium",
)


def format_value(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_inner(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_format_inner(k)}: {_format_inner(v)}" for k, v in value.items()) + "}"
    if isinstance(value, TriAxialObject):
        inner = ", ".join(f"{k}={_format_inner(v)}" for k, v in value.fields.items())
        return f"<{value.dhriv_class.name} {inner}>".replace(" >", ">")
    if isinstance(value, WaveSignal):
        return repr(value)
    if isinstance(value, (DhrivFunction, BoundMethod)):
        return f"<fn {value.node.name}>"
    return str(value)


def _format_inner(value: Any) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    return format_value(value)


class DhrivInterpreter:
    MAX_CALL_DEPTH = 300

    def __init__(
        self,
        echo: bool = False,
        max_steps: int = 20_000_000,
        allow_legacy: bool = False,
        learner: Any = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self.registry = TriAxialOOPRegistry()
        self.search = TriAxialLargeScaleSearch()
        self.output_lines: List[str] = []
        self.echo = echo
        self.max_steps = max_steps
        self.allow_legacy = allow_legacy
        self.learner = learner
        self.cancel_event = cancel_event
        self.steps = 0
        self.heartbeat = time.monotonic()
        self._depth = 0
        self._closures: Dict[int, Environment] = {}
        self.globals = Environment()
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 12_000))
        self._install_builtins()
        if self.learner is not None:
            self.learner.bind(self)

    # ------------------------------------------------------------------ API

    def run(self, source: str) -> str:
        program = DhrivParser.parse_source(source)
        return self.run_program(program)

    def run_program(self, program: ProgramNode) -> str:
        self.heartbeat = time.monotonic()
        for stmt in program.statements:
            self._exec(stmt, self.globals)
        return "\n".join(self.output_lines)

    def evaluate_expression(self, source: str) -> Any:
        """Valuta una singola espressione nel contesto globale (usata dal REPL)."""
        program = DhrivParser.parse_source(source)
        if len(program.statements) == 1 and isinstance(program.statements[0], ExprStmtNode):
            return self._eval(program.statements[0].expr, self.globals)
        self.run_program(program)
        return None

    def emit(self, text: str) -> None:
        self.output_lines.append(text)
        if self.echo:
            print(text, flush=True)

    # ----------------------------------------------------------- esecuzione

    def _tick(self) -> None:
        self.steps += 1
        if self.steps > self.max_steps:
            raise DhrivRuntimeError("Budget di passi esaurito (possibile ciclo infinito).")
        if (self.steps & 1023) == 0:
            self.heartbeat = time.monotonic()
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise DhrivCancelled("Esecuzione interrotta dal WatchDog.")

    def _exec_stmts(self, stmts: List[ASTNode], env: Environment) -> None:
        for s in stmts:
            self._exec(s, env)

    def _exec_block(self, stmts: List[ASTNode], env: Environment) -> None:
        self._exec_stmts(stmts, Environment(env))

    def _exec(self, node: ASTNode, env: Environment) -> None:
        self._tick()
        t = type(node)
        if t is ExprStmtNode:
            self._eval(node.expr, env)
        elif t is LetStmtNode:
            value = self._eval(node.value, env)
            if node.is_wave and isinstance(value, (int, float)) and not isinstance(value, bool):
                value = synthesize_wave(float(value))
            env.define(node.name, value)
        elif t is AssignStmtNode:
            self._exec_assign(node, env)
        elif t is IfStmtNode:
            if self._eval(node.condition, env):
                self._exec_block(node.then_branch, env)
            elif node.else_branch is not None:
                self._exec_block(node.else_branch, env)
        elif t is WhileStmtNode:
            while self._eval(node.condition, env):
                self._exec_block(node.body, env)
                self._tick()
        elif t is ForStmtNode:
            self._exec_for(node, env)
        elif t is ReturnStmtNode:
            raise _ReturnSignal(self._eval(node.value, env) if node.value is not None else None)
        elif t is FunctionDeclNode:
            env.define(node.name, DhrivFunction(node, env))
        elif t is EntityDeclNode:
            self._exec_entity(node, env)
        else:
            raise DhrivRuntimeError(f"Istruzione non supportata: {t.__name__}", getattr(node, "line", None))

    def _exec_for(self, node: ForStmtNode, env: Environment) -> None:
        iterable = self._eval(node.iterable, env)
        if isinstance(iterable, dict):
            items = list(iterable.keys())
        elif isinstance(iterable, (list, str)):
            items = list(iterable)
        else:
            raise DhrivRuntimeError("Il ciclo 'for' richiede lista, testo o mappa.", node.line)
        for item in items:
            loop_env = Environment(env)
            loop_env.define(node.iterator_name, item)
            self._exec_stmts(node.body, loop_env)
            self._tick()

    def _exec_assign(self, node: AssignStmtNode, env: Environment) -> None:
        value = self._eval(node.value, env)
        target = node.target
        if isinstance(target, IdentifierNode):
            env.assign(target.name, value, node.line)
        elif isinstance(target, MemberAccessNode):
            obj = self._eval(target.obj, env)
            if isinstance(obj, TriAxialObject):
                obj.fields[target.member] = value
            elif isinstance(obj, dict):
                obj[target.member] = value
            else:
                raise DhrivRuntimeError("Impossibile assegnare un attributo a questo valore.", node.line)
        elif isinstance(target, IndexAccessNode):
            obj = self._eval(target.obj, env)
            idx = self._eval(target.index, env)
            try:
                obj[idx] = value
            except (IndexError, KeyError, TypeError) as exc:
                raise DhrivRuntimeError(f"Assegnazione per indice non valida: {exc}", node.line)
        else:
            raise DhrivRuntimeError("Bersaglio di assegnazione non valido.", node.line)

    def _exec_entity(self, node: EntityDeclNode, env: Environment) -> None:
        for axis_list in (node.vertical_parents, node.horizontal_peers, node.diagonal_bridges):
            for parent in axis_list:
                if parent == node.name or parent not in self.registry.classes:
                    raise DhrivRuntimeError(
                        f"Entità '{node.name}': l'entità collegata '{parent}' non è ancora dichiarata.", node.line
                    )
        default_fields = {f.name: self._eval(f.value, env) for f in node.fields}
        methods = {m.name: m for m in node.methods}
        for m in node.methods:
            self._closures[id(m)] = env
        self.registry.register_class(
            TriAxialClass(
                name=node.name,
                vertical_parents=list(node.vertical_parents),
                horizontal_peers=list(node.horizontal_peers),
                diagonal_bridges=list(node.diagonal_bridges),
                default_fields=default_fields,
                methods=methods,
            )
        )

    # ------------------------------------------------------------ espressioni

    def _eval(self, node: ASTNode, env: Environment) -> Any:
        t = type(node)
        if t is LiteralNode:
            return node.value
        if t is IdentifierNode:
            return env.lookup(node.name, node.line)
        if t is BinaryOpNode:
            return self._eval_binary(node, env)
        if t is CallNode:
            callee = self._eval(node.callee, env)
            args = [self._eval(a, env) for a in node.args]
            return self._invoke(callee, args, node.line)
        if t is MemberAccessNode:
            return self._eval_member(node, env)
        if t is UnaryOpNode:
            v = self._eval(node.operand, env)
            if node.op == "not":
                return not v
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return -v
            raise DhrivRuntimeError("Operatore '-' valido solo su numeri.", node.line)
        if t is ListLiteralNode:
            return [self._eval(e, env) for e in node.elements]
        if t is MapLiteralNode:
            return {self._eval(k, env): self._eval(v, env) for k, v in node.entries}
        if t is IndexAccessNode:
            obj = self._eval(node.obj, env)
            idx = self._eval(node.index, env)
            try:
                return obj[idx]
            except (IndexError, KeyError, TypeError) as exc:
                raise DhrivRuntimeError(f"Accesso per indice non valido: {exc}", node.line)
        if t is NewInstanceNode:
            return self._eval_new(node, env)
        if t is TriAxialOpNode:
            return self._eval_triaxial(node, env)
        raise DhrivRuntimeError(f"Espressione non supportata: {t.__name__}", getattr(node, "line", None))

    def _eval_binary(self, node: BinaryOpNode, env: Environment) -> Any:
        op = node.op
        if op == "and":
            left = self._eval(node.left, env)
            return self._eval(node.right, env) if left else left
        if op == "or":
            left = self._eval(node.left, env)
            return left if left else self._eval(node.right, env)

        l = self._eval(node.left, env)
        r = self._eval(node.right, env)
        try:
            if op == "+":
                if isinstance(l, str) or isinstance(r, str):
                    return format_value(l) + format_value(r)
                return l + r
            if op == "-":
                return l - r
            if op == "*":
                return l * r
            if op == "/":
                if r == 0:
                    raise DhrivRuntimeError("Divisione per zero.", node.line)
                if isinstance(l, int) and isinstance(r, int) and l % r == 0:
                    return l // r
                return l / r
            if op == "%":
                if r == 0:
                    raise DhrivRuntimeError("Modulo per zero.", node.line)
                return l % r
            if op == "**":
                if isinstance(r, (int, float)) and abs(r) > 10_000:
                    raise DhrivRuntimeError("Esponente troppo grande.", node.line)
                return l ** r
            if op == "==":
                return l == r
            if op == "!=":
                return l != r
            if op == "<":
                return l < r
            if op == "<=":
                return l <= r
            if op == ">":
                return l > r
            if op == ">=":
                return l >= r
        except DhrivRuntimeError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise DhrivRuntimeError(f"Operazione '{op}' non valida per questi operandi: {exc}", node.line)
        raise DhrivRuntimeError(f"Operatore sconosciuto '{op}'", node.line)

    def _eval_member(self, node: MemberAccessNode, env: Environment) -> Any:
        obj = self._eval(node.obj, env)
        member = node.member
        if isinstance(obj, TriAxialObject):
            res = self.registry.resolve_instance_member(obj, member)
            if not res.found:
                raise DhrivRuntimeError(
                    f"L'entità '{obj.dhriv_class.name}' non ha il membro '{member}' su nessun asse (V/H/D).",
                    node.line,
                )
            value = res.value
            if isinstance(value, FunctionDeclNode):
                return BoundMethod(obj, value, self._closures.get(id(value), self.globals))
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[0], TriAxialObject)
                and isinstance(value[1], FunctionDeclNode)
            ):
                peer, fnode = value
                return BoundMethod(peer, fnode, self._closures.get(id(fnode), self.globals))
            return value
        if isinstance(obj, WaveSignal):
            if member in WAVE_PUBLIC_ATTRS:
                return getattr(obj, member)
            raise DhrivRuntimeError(f"Le onde non hanno l'attributo '{member}'.", node.line)
        if isinstance(obj, dict):
            if member in obj:
                return obj[member]
            raise DhrivRuntimeError(f"Chiave '{member}' assente nella mappa.", node.line)
        raise DhrivRuntimeError(f"Accesso al membro '{member}' non supportato su questo valore.", node.line)

    def _eval_new(self, node: NewInstanceNode, env: Environment) -> TriAxialObject:
        if node.entity_name not in self.registry.classes:
            raise DhrivRuntimeError(f"Entità '{node.entity_name}' non dichiarata.", node.line)
        args = [self._eval(a, env) for a in node.args]
        instance = self.registry.instantiate(node.entity_name)
        instance.fields = copy.deepcopy(instance.fields)
        ctor = self.registry.resolve_method(node.entity_name, "init")
        if ctor.found and isinstance(ctor.value, FunctionDeclNode):
            closure = self._closures.get(id(ctor.value), self.globals)
            self._call_user(ctor.value, args, closure, instance, node.line)
        elif args:
            raise DhrivRuntimeError(
                f"L'entità '{node.entity_name}' non ha un metodo 'init' ma riceve {len(args)} argomenti.", node.line
            )
        return instance

    def _eval_triaxial(self, node: TriAxialOpNode, env: Environment) -> Any:
        if node.axis == "VERTICAL":
            left = self._eval(node.left, env)
            right = node.right
            if isinstance(right, CallNode):
                callee = self._eval(right.callee, env)
                args = [left] + [self._eval(a, env) for a in right.args]
                return self._invoke(callee, args, node.line)
            callee = self._eval(right, env)
            return self._invoke(callee, [left], node.line)

        l = self._eval(node.left, env)
        r = self._eval(node.right, env)
        if node.axis == "HORIZONTAL":
            if isinstance(l, TriAxialObject) and isinstance(r, TriAxialObject):
                return l.link_horizontal(r)
            if isinstance(l, WaveSignal) and isinstance(r, WaveSignal):
                return superpose_waves(l, r)
            raise DhrivRuntimeError("'<->' collega due entità oppure sovrappone due onde.", node.line)
        if node.axis == "DIAGONAL":
            if isinstance(l, TriAxialObject) and isinstance(r, TriAxialObject):
                return l.link_diagonal(r)
            raise DhrivRuntimeError("'</>' collega due entità con un ponte diagonale.", node.line)
        raise DhrivRuntimeError(f"Asse sconosciuto '{node.axis}'", node.line)

    # -------------------------------------------------------------- chiamate

    def _invoke(self, callee: Any, args: List[Any], line: Optional[int]) -> Any:
        if isinstance(callee, BoundMethod):
            return self._call_user(callee.node, args, callee.closure, callee.instance, line)
        if isinstance(callee, DhrivFunction):
            return self._call_user(callee.node, args, callee.closure, None, line)
        if callable(callee):
            try:
                return callee(*args)
            except DhrivRuntimeError:
                raise
            except (TypeError, ValueError, KeyError, IndexError, OSError, ZeroDivisionError, OverflowError) as exc:
                raise DhrivRuntimeError(f"Errore nella funzione nativa: {exc}", line)
        raise DhrivRuntimeError("Il valore non è invocabile.", line)

    def _call_user(
        self,
        node: FunctionDeclNode,
        args: List[Any],
        closure: Environment,
        self_obj: Optional[TriAxialObject],
        line: Optional[int],
    ) -> Any:
        if len(args) != len(node.params):
            raise DhrivRuntimeError(
                f"'{node.name}' richiede {len(node.params)} argomenti, ricevuti {len(args)}.", line
            )
        self._depth += 1
        try:
            if self._depth > self.MAX_CALL_DEPTH:
                raise DhrivRuntimeError("Profondità massima di chiamata superata.", line)
            use_learner = self.learner is not None and self_obj is None
            if use_learner:
                hit, cached = self.learner.lookup(node, args)
                if hit:
                    return cached
            fenv = Environment(closure)
            if self_obj is not None:
                fenv.define("self", self_obj)
            for pname, value in zip(node.params, args):
                fenv.define(pname, value)
            result: Any = None
            try:
                self._exec_stmts(node.body, fenv)
            except _ReturnSignal as ret:
                result = ret.value
            if use_learner:
                self.learner.store(node, args, result)
            return result
        finally:
            self._depth -= 1

    # -------------------------------------------------------------- builtins

    def _install_builtins(self) -> None:
        g = self.globals.define
        interp = self

        def _print(*args: Any) -> None:
            interp.emit(" ".join(format_value(a) for a in args))

        def _range(a: int, b: Optional[int] = None, step: int = 1) -> List[int]:
            lo, hi = (0, a) if b is None else (a, b)
            result = list(range(int(lo), int(hi), int(step)))
            if len(result) > 1_000_000:
                raise DhrivRuntimeError("range() troppo grande.")
            return result

        def _push(lst: List[Any], value: Any) -> List[Any]:
            lst.append(value)
            return lst

        def _pop(lst: List[Any]) -> Any:
            return lst.pop()

        def _num(x: Any) -> Any:
            if isinstance(x, (int, float)):
                return x
            f = float(x)
            return int(f) if f.is_integer() else f

        def _wave_new(freq: float, amp: float = 1.0, phase: float = 0.0, medium: str = "ELECTROMAGNETIC") -> WaveSignal:
            return synthesize_wave(float(freq), float(amp), float(phase), medium=str(medium).upper())

        def _wavelength(x: Any) -> float:
            f = x.frequency_hz if isinstance(x, WaveSignal) else float(x)
            return SPEED_OF_LIGHT_MPS / f

        def _band(x: Any) -> str:
            f = x.frequency_hz if isinstance(x, WaveSignal) else float(x)
            return classify_em_band(f)

        def _photon_ev(freq: float) -> float:
            return 6.62607015e-34 * float(freq) / 1.602176634e-19

        def _em_transmit(text: str, snr_db: float = 20.0, scheme: str = "BFSK") -> Dict[str, Any]:
            modem = ElectromagneticBinaryModem()
            payload = str(text).encode("utf-8")
            sig = modem.modulate_binary(payload, scheme=str(scheme))
            noisy = modem.inject_channel_noise(sig, snr_db=float(snr_db))
            try:
                data, info = modem.demodulate_binary(noisy, scheme=str(scheme))
            except ValueError as exc:
                return {"ok": False, "error": str(exc), "text": None}
            return {
                "ok": True,
                "text": data.decode("utf-8", errors="replace"),
                "corrected_bits": info["fec_corrected_bit_errors"],
                "scheme": info["scheme"],
            }

        def _scale_register(node_id: str, h: int, v: int, caps: List[str]) -> str:
            interp.search.register_node(str(node_id), int(h), int(v), [str(c) for c in caps])
            return str(node_id)

        def _search_h(v: int, cap: Optional[str] = None) -> List[Dict[str, Any]]:
            return interp.search.search_horizontal(int(v), cap)

        def _search_v(h: int, cap: Optional[str] = None) -> List[Dict[str, Any]]:
            return interp.search.search_vertical(int(h), cap)

        def _search_d(h: int, v: int, cap: Optional[str] = None) -> List[Dict[str, Any]]:
            return interp.search.search_diagonal(int(h), int(v), cap)

        def _legacy_run(command: str, timeout: float = 10.0) -> Dict[str, Any]:
            if not interp.allow_legacy:
                raise DhrivRuntimeError(
                    "legacy_run è disabilitato: avvia con --allow-legacy per eseguire software esistente."
                )
            from .compat import LegacyBridge

            return LegacyBridge().run(command, timeout=float(timeout))

        def _type_of(x: Any) -> str:
            if x is None:
                return "null"
            if isinstance(x, bool):
                return "bool"
            if isinstance(x, (int, float)):
                return "number"
            if isinstance(x, str):
                return "text"
            if isinstance(x, list):
                return "list"
            if isinstance(x, dict):
                return "map"
            if isinstance(x, TriAxialObject):
                return x.dhriv_class.name
            if isinstance(x, WaveSignal):
                return "wave"
            return "function"

        builtins: Dict[str, Callable[..., Any]] = {
            "print": _print,
            "stampa": _print,
            "len": len,
            "str": format_value,
            "num": _num,
            "int": lambda x: int(float(x)),
            "float": float,
            "range": _range,
            "push": _push,
            "pop": _pop,
            "keys": lambda m: list(m.keys()),
            "abs": abs,
            "min": min,
            "max": max,
            "floor": math.floor,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "round": lambda x, n=0: round(x, int(n)),
            "upper": lambda s: str(s).upper(),
            "lower": lambda s: str(s).lower(),
            "join": lambda lst, sep="": str(sep).join(format_value(x) for x in lst),
            "split": lambda s, sep=" ": str(s).split(str(sep)),
            "type_of": _type_of,
            "clock": time.perf_counter,
            "wave_new": _wave_new,
            "spectrum": analyze_spectrum,
            "wavelength": _wavelength,
            "band": _band,
            "photon_ev": _photon_ev,
            "fspl": free_space_path_loss_db,
            "em_transmit": _em_transmit,
            "scale_register": _scale_register,
            "search_h": _search_h,
            "search_v": _search_v,
            "search_d": _search_d,
            "legacy_run": _legacy_run,
        }
        for name, fn in builtins.items():
            g(name, fn)
        g("pi", math.pi)
        g("c_light", SPEED_OF_LIGHT_MPS)
