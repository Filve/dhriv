"""
interpreter.py — Runtime di esecuzione del linguaggio Dhriv.

Collega il linguaggio a:
- Fondamenta Binarie, Porte Logiche e ALU a N bit (`binary_core`);
- Registro OOP Tri-Assiale (`TriAxialOOPRegistry`): risoluzione Self -> Verticale -> Orizzontale -> Diagonale,
  dispatch assiale esplicito e broadcast a sciame (`axis_call`, `axis_broadcast`);
- Motore di ricerca su larga scala (`TriAxialLargeScaleSearch`);
- Motore di onde / elettromagnetismo (`wave_electromagnetic`);
- Motore nativo di Intelligenza Artificiale, Tensori, Autograd, Self-Attention e Reti Tri-Assiali (`ai_engine`);
- Motore di auto-apprendimento opzionale (`selflearn`).

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

from __future__ import annotations

import copy
import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .ai_engine import (
    AdamOptimizer,
    LinearLayer,
    MindMemory,
    NeuralModule,
    SGDOptimizer,
    SelfAttentionLayer,
    Tensor,
    TriAxialNetwork,
    bce_loss,
    cosine_similarity,
    cross_entropy_loss,
    mse_loss,
    pack_weights_binary,
    unpack_weights_binary,
)
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
    LambdaNode,
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
from .binary_core import (
    BinaryALU,
    bits_to_bitstring,
    bits_to_int,
    eval_gate,
    full_adder,
    half_adder,
    int_to_bits,
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
    if isinstance(value, (WaveSignal, Tensor, NeuralModule, MindMemory)):
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
        self._vm: Any = None
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
        if t is LambdaNode:
            fnode = FunctionDeclNode("<lambda>", list(node.params), list(node.body), is_method=False, line=node.line)
            return DhrivFunction(fnode, env)
        if t is BinaryOpNode:
            return self._eval_binary(node, env)
        if t is CallNode:
            callee = self._eval(node.callee, env)
            args = [self._eval(a, env) for a in node.args]
            return self._invoke(callee, args, node.line)
        if t is MemberAccessNode:
            obj = self._eval(node.obj, env)
            return self.resolve_member_value(obj, node.member, node.line)
        if t is UnaryOpNode:
            v = self._eval(node.operand, env)
            return self.eval_unary_value(node.op, v, node.line)
        if t is ListLiteralNode:
            return [self._eval(e, env) for e in node.elements]
        if t is MapLiteralNode:
            return {self._eval(k, env): self._eval(v, env) for k, v in node.entries}
        if t is IndexAccessNode:
            obj = self._eval(node.obj, env)
            idx = self._eval(node.index, env)
            return self.eval_index_value(obj, idx, node.line)
        if t is NewInstanceNode:
            args = [self._eval(a, env) for a in node.args]
            return self.instantiate_entity(node.entity_name, args, node.line)
        if t is TriAxialOpNode:
            return self._eval_triaxial(node, env)
        raise DhrivRuntimeError(f"Espressione non supportata: {t.__name__}", getattr(node, "line", None))

    def eval_unary_value(self, op: str, v: Any, line: Optional[int] = None) -> Any:
        if op == "not":
            return not v
        if op == "~":
            if isinstance(v, int) and not isinstance(v, bool):
                return ~v
            raise DhrivRuntimeError("Operatore bitwise '~' valido solo su interi.", line)
        if op == "-":
            if isinstance(v, Tensor):
                return v.neg()
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return -v
            raise DhrivRuntimeError("Operatore '-' valido solo su numeri o Tensor.", line)
        raise DhrivRuntimeError(f"Operatore unario sconosciuto '{op}'", line)

    def eval_binary_values(self, op: str, l: Any, r: Any, line: Optional[int] = None) -> Any:
        try:
            # Supporto algebrico differenziabile per Tensor
            if isinstance(l, Tensor) or isinstance(r, Tensor):
                if op == "+" and not (isinstance(l, str) or isinstance(r, str)):
                    return (l if isinstance(l, Tensor) else Tensor(l)).add(r)
                if op == "-":
                    return (l if isinstance(l, Tensor) else Tensor(l)).sub(r)
                if op == "*":
                    return (l if isinstance(l, Tensor) else Tensor(l)).mul(r)
                if op == "/":
                    return (l if isinstance(l, Tensor) else Tensor(l)).div(r)

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
                    raise DhrivRuntimeError("Divisione per zero.", line)
                if isinstance(l, int) and isinstance(r, int) and l % r == 0:
                    return l // r
                return l / r
            if op == "%":
                if r == 0:
                    raise DhrivRuntimeError("Modulo per zero.", line)
                return l % r
            if op == "**":
                if isinstance(r, (int, float)) and abs(r) > 10_000:
                    raise DhrivRuntimeError("Esponente troppo grande.", line)
                return l ** r
            if op in ("&", "|", "^", "<<", ">>"):
                if not (isinstance(l, int) and isinstance(r, int) and not isinstance(l, bool) and not isinstance(r, bool)):
                    raise DhrivRuntimeError(f"Operatore bitwise '{op}' richiede due interi.", line)
                if op == "&":
                    return l & r
                if op == "|":
                    return l | r
                if op == "^":
                    return l ^ r
                if r < 0 or r > 4096:
                    raise DhrivRuntimeError("Conteggio shift fuori intervallo.", line)
                return (l << r) if op == "<<" else (l >> r)
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
        except ZeroDivisionError:
            raise DhrivRuntimeError("Divisione per zero.", line)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DhrivRuntimeError(f"Operazione '{op}' non valida per questi operandi: {exc}", line)
        raise DhrivRuntimeError(f"Operatore sconosciuto '{op}'", line)

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
        return self.eval_binary_values(op, l, r, node.line)

    def eval_index_value(self, obj: Any, idx: Any, line: Optional[int] = None) -> Any:
        if isinstance(obj, Tensor):
            try:
                row = obj.data[int(idx)]
                return list(row)
            except (IndexError, ValueError, TypeError) as exc:
                raise DhrivRuntimeError(f"Indice Tensor non valido: {exc}", line)
        try:
            return obj[idx]
        except (IndexError, KeyError, TypeError) as exc:
            raise DhrivRuntimeError(f"Accesso per indice non valido: {exc}", line)

    def resolve_member_value(self, obj: Any, member: str, line: Optional[int] = None) -> Any:
        if isinstance(obj, TriAxialObject):
            res = self.registry.resolve_instance_member(obj, member)
            if not res.found:
                raise DhrivRuntimeError(
                    f"L'entità '{obj.dhriv_class.name}' non ha il membro '{member}' su nessun asse (V/H/D).",
                    line,
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
            raise DhrivRuntimeError(f"Le onde non hanno l'attributo '{member}'.", line)
        if isinstance(obj, Tensor):
            if member == "shape":
                return obj.shape
            if member == "rows":
                return obj.rows
            if member == "cols":
                return obj.cols
            if member == "values":
                return obj.values
            if member == "grad":
                return obj.grad_values
            if member == "T":
                return obj.T
            if member == "requires_grad":
                return obj.requires_grad
            raise DhrivRuntimeError(f"Tensor non ha l'attributo '{member}'.", line)
        if isinstance(obj, NeuralModule):
            if member == "parameters":
                return obj.parameters()
            if hasattr(obj, member) and isinstance(getattr(obj, member), (Tensor, int, float, str)):
                return getattr(obj, member)
            raise DhrivRuntimeError(f"Il modulo neurale non espone l'attributo '{member}'.", line)
        if isinstance(obj, dict):
            if member in obj:
                return obj[member]
            raise DhrivRuntimeError(f"Chiave '{member}' assente nella mappa.", line)
        raise DhrivRuntimeError(f"Accesso al membro '{member}' non supportato su questo valore.", line)

    def instantiate_entity(self, entity_name: str, args: List[Any], line: Optional[int] = None) -> TriAxialObject:
        if entity_name not in self.registry.classes:
            raise DhrivRuntimeError(f"Entità '{entity_name}' non dichiarata.", line)
        instance = self.registry.instantiate(entity_name)
        instance.fields = copy.deepcopy(instance.fields)
        ctor = self.registry.resolve_method(entity_name, "init")
        if ctor.found and isinstance(ctor.value, FunctionDeclNode):
            closure = self._closures.get(id(ctor.value), self.globals)
            self._call_user(ctor.value, args, closure, instance, line)
        elif args:
            raise DhrivRuntimeError(
                f"L'entità '{entity_name}' non ha un metodo 'init' ma riceve {len(args)} argomenti.", line
            )
        return instance

    def apply_triaxial_values(self, axis: str, l: Any, r: Any, line: Optional[int] = None) -> Any:
        if axis == "HORIZONTAL":
            if isinstance(l, TriAxialObject) and isinstance(r, TriAxialObject):
                return l.link_horizontal(r)
            if isinstance(l, WaveSignal) and isinstance(r, WaveSignal):
                return superpose_waves(l, r)
            if isinstance(l, Tensor) and isinstance(r, Tensor):
                return l.add(r).mul(0.5)
            raise DhrivRuntimeError(
                "'<->' collega due entità, sovrappone due onde o fonde due Tensor.", line
            )
        if axis == "DIAGONAL":
            if isinstance(l, TriAxialObject) and isinstance(r, TriAxialObject):
                return l.link_diagonal(r)
            if isinstance(l, Tensor) and isinstance(r, Tensor):
                try:
                    return l.matmul(r)
                except ValueError as exc:
                    raise DhrivRuntimeError(str(exc), line)
            raise DhrivRuntimeError(
                "'</>' crea un ponte diagonale tra due entità o proietta due Tensor (matmul).", line
            )
        raise DhrivRuntimeError(f"Asse sconosciuto '{axis}'", line)

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
        return self.apply_triaxial_values(node.axis, l, r, node.line)

    # -------------------------------------------------------------- chiamate

    def _call_method_value(
        self, owner: TriAxialObject, val: Any, args: List[Any], line: Optional[int] = None
    ) -> Any:
        if isinstance(val, FunctionDeclNode):
            return self._call_user(val, args, self._closures.get(id(val), self.globals), owner, line)
        if self._vm is not None and hasattr(val, "instructions"):
            return self._vm.call_chunk(val, args, self._closures.get(id(val), self.globals), owner, line)
        if isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], TriAxialObject):
            peer, mval = val
            return self._call_method_value(peer, mval, args, line)
        return val

    def _invoke(self, callee: Any, args: List[Any], line: Optional[int]) -> Any:
        if isinstance(callee, BoundMethod):
            return self._call_user(callee.node, args, callee.closure, callee.instance, line)
        if isinstance(callee, DhrivFunction):
            return self._call_user(callee.node, args, callee.closure, None, line)
        if isinstance(callee, TriAxialObject):
            # Entità OOP invocabili nelle pipeline verticali `|>` tramite `forward` / `call` / `esegui`
            for hook in ("forward", "call", "esegui"):
                res = self.registry.resolve_instance_member(callee, hook)
                if res.found:
                    return self._call_method_value(callee, res.value, args, line)
            raise DhrivRuntimeError(
                f"L'entità '{callee.dhriv_class.name}' non definisce 'forward', 'call' o 'esegui'.", line
            )
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
            use_learner = self.learner is not None and self_obj is None and node.name != "<lambda>"
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

        # --- Primitive Binarie & ALU ---
        def _bit_gate(gate: str, a: int, b: int = 0) -> int:
            return eval_gate(gate, int(a), int(b))

        def _half_adder(a: int, b: int) -> Dict[str, int]:
            s, c = half_adder(int(a), int(b))
            return {"sum": s, "carry": c}

        def _full_adder(a: int, b: int, cin: int = 0) -> Dict[str, int]:
            s, c = full_adder(int(a), int(b), int(cin))
            return {"sum": s, "carry": c}

        def _alu(op: str, a: int, b: int = 0, width: int = 8) -> Dict[str, Any]:
            return BinaryALU(int(width)).execute(op, int(a), int(b))

        def _bits_of(n: int, width: int = 8) -> List[int]:
            return int_to_bits(int(n), int(width))

        def _bitstr(n: int, width: int = 8, group: int = 0) -> str:
            return bits_to_bitstring(int_to_bits(int(n), int(width)), group=int(group))

        def _from_bits(bits_or_str: Any, signed: bool = False) -> int:
            if isinstance(bits_or_str, str):
                clean = [1 if c == "1" else 0 for c in bits_or_str if c in ("0", "1")]
                return bits_to_int(clean, signed=bool(signed))
            return bits_to_int([int(x) for x in bits_or_str], signed=bool(signed))

        # --- OOP Tri-Assiale Avanzato ---
        def _coords_of(obj: Any) -> Dict[str, Any]:
            if not isinstance(obj, TriAxialObject):
                raise DhrivRuntimeError("coords_of() richiede un'entità OOP Dhriv.")
            return interp.registry.inspect_coordinates(obj)

        def _axis_call(obj: Any, axis: str, method_name: str, call_args: Optional[List[Any]] = None) -> Any:
            if not isinstance(obj, TriAxialObject):
                raise DhrivRuntimeError("axis_call() richiede un'entità OOP Dhriv.")
            res = interp.registry.resolve_specific_axis(obj, axis, str(method_name))
            if not res.found:
                raise DhrivRuntimeError(
                    f"Nessun membro '{method_name}' trovato sull'asse {axis} per '{obj.dhriv_class.name}'."
                )
            return interp._call_method_value(obj, res.value, list(call_args or []), None)

        def _axis_broadcast(obj: Any, axis: str, method_name: str, call_args: Optional[List[Any]] = None) -> List[Any]:
            if not isinstance(obj, TriAxialObject):
                raise DhrivRuntimeError("axis_broadcast() richiede un'entità OOP Dhriv.")
            swarm = interp.registry.collect_swarm(obj, axis)
            arglist = list(call_args or [])
            results: List[Any] = []
            for target in swarm:
                res = interp.registry.resolve_instance_member(target, str(method_name))
                if not res.found:
                    continue
                results.append(interp._call_method_value(target, res.value, arglist, None))
            return results

        # --- Helper Funzionali ---
        def _map_list(lst: List[Any], fn_val: Any) -> List[Any]:
            return [interp._invoke(fn_val, [item], None) for item in lst]

        def _filter_list(lst: List[Any], fn_val: Any) -> List[Any]:
            return [item for item in lst if interp._invoke(fn_val, [item], None)]

        def _reduce_list(lst: List[Any], fn_val: Any, init: Any) -> Any:
            acc = init
            for item in lst:
                acc = interp._invoke(fn_val, [acc, item], None)
            return acc

        # --- Primitive AI, Tensori, Autograd & Memoria Neuro-Simbolica ---
        def _tensor(data: Any, requires_grad: bool = False) -> Tensor:
            return Tensor(data, requires_grad=bool(requires_grad))

        def _zero_grad(target: Any) -> None:
            if isinstance(target, Tensor):
                target.zero_grad()
            elif isinstance(target, (SGDOptimizer, AdamOptimizer)):
                target.zero_grad()
            elif isinstance(target, NeuralModule):
                for p in target.parameters():
                    p.zero_grad()
            elif isinstance(target, list):
                for p in target:
                    if isinstance(p, Tensor):
                        p.zero_grad()
            else:
                raise DhrivRuntimeError("zero_grad() accetta Tensor, lista di Tensor, modulo neurale o ottimizzatore.")

        def _opt_step(opt: Any) -> None:
            if not isinstance(opt, (SGDOptimizer, AdamOptimizer)):
                raise DhrivRuntimeError("opt_step() richiede un ottimizzatore (nn_sgd o nn_adam).")
            opt.step()

        def _params_of(mod_or_list: Any) -> List[Tensor]:
            if isinstance(mod_or_list, NeuralModule):
                return mod_or_list.parameters()
            if isinstance(mod_or_list, list):
                out: List[Tensor] = []
                for x in mod_or_list:
                    if isinstance(x, NeuralModule):
                        out.extend(x.parameters())
                    elif isinstance(x, Tensor):
                        out.append(x)
                return out
            raise DhrivRuntimeError("Atteso modulo neurale o lista di parametri Tensor.")

        def _ai_pack_hex(mod_or_params: Any) -> str:
            return pack_weights_binary(_params_of(mod_or_params)).hex()

        def _ai_unpack_hex(hex_str: str, target: Any = None) -> List[Tensor]:
            blob = bytes.fromhex(str(hex_str))
            t_params = _params_of(target) if target is not None else None
            return unpack_weights_binary(blob, t_params)

        def _em_transmit_ai(mod_or_params: Any, snr_db: float = 16.0, scheme: str = "BFSK") -> Dict[str, Any]:
            """Serializza i pesi di un modello neurale in binario IEEE-754 e li trasmette via onda EM con FEC."""
            modem = ElectromagneticBinaryModem()
            raw_weights = pack_weights_binary(_params_of(mod_or_params))
            sig = modem.modulate_binary(raw_weights, scheme=str(scheme))
            noisy = modem.inject_channel_noise(sig, snr_db=float(snr_db))
            try:
                rx_bytes, info = modem.demodulate_binary(noisy, scheme=str(scheme))
                tensors = unpack_weights_binary(rx_bytes)
            except ValueError as exc:
                return {"ok": False, "error": str(exc), "tensors": []}
            return {
                "ok": True,
                "bytes": len(rx_bytes),
                "corrected_bits": info["fec_corrected_bit_errors"],
                "tensors": tensors,
            }

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
            if isinstance(x, Tensor):
                return "tensor"
            if isinstance(x, NeuralModule):
                return "neural_module"
            if isinstance(x, MindMemory):
                return "mind_memory"
            if isinstance(x, (SGDOptimizer, AdamOptimizer)):
                return "optimizer"
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
            "exp": math.exp,
            "log": math.log,
            "round": lambda x, n=0: round(x, int(n)),
            "upper": lambda s: str(s).upper(),
            "lower": lambda s: str(s).lower(),
            "join": lambda lst, sep="": str(sep).join(format_value(x) for x in lst),
            "split": lambda s, sep=" ": str(s).split(str(sep)),
            "type_of": _type_of,
            "clock": time.perf_counter,
            # Onde & Elettromagnetismo
            "wave_new": _wave_new,
            "spectrum": analyze_spectrum,
            "wavelength": _wavelength,
            "band": _band,
            "photon_ev": _photon_ev,
            "fspl": free_space_path_loss_db,
            "em_transmit": _em_transmit,
            # Ricerca Tri-Assiale Larga Scala
            "scale_register": _scale_register,
            "search_h": _search_h,
            "search_v": _search_v,
            "search_d": _search_d,
            # Retrocompatibilità
            "legacy_run": _legacy_run,
            # Livello Binario, Porte Logiche & ALU
            "bit_gate": _bit_gate,
            "half_adder": _half_adder,
            "full_adder": _full_adder,
            "alu": _alu,
            "bits_of": _bits_of,
            "bitstr": _bitstr,
            "from_bits": _from_bits,
            "hex_of": lambda n: hex(int(n)),
            # OOP Tri-Assiale Avanzato & Funzionale
            "coords_of": _coords_of,
            "axis_call": _axis_call,
            "axis_broadcast": _axis_broadcast,
            "map_list": _map_list,
            "filter_list": _filter_list,
            "reduce_list": _reduce_list,
            # AI, Tensori, Autograd, Strati Neurali & Memoria Neuro-Simbolica
            "tensor": _tensor,
            "tensore": _tensor,
            "matmul": lambda a, b: Tensor._ensure(a).matmul(Tensor._ensure(b)),
            "backward": lambda t: t.backward(),
            "zero_grad": _zero_grad,
            "item": lambda t: t.item() if isinstance(t, Tensor) else float(t),
            "relu": lambda t: Tensor._ensure(t).relu(),
            "sigmoid": lambda t: Tensor._ensure(t).sigmoid(),
            "tanh": lambda t: Tensor._ensure(t).tanh(),
            "gelu": lambda t: Tensor._ensure(t).gelu(),
            "softmax": lambda t: Tensor._ensure(t).softmax(),
            "t_sum": lambda t: Tensor._ensure(t).sum(),
            "t_mean": lambda t: Tensor._ensure(t).mean(),
            "mse_loss": mse_loss,
            "bce_loss": bce_loss,
            "cross_entropy": cross_entropy_loss,
            "nn_linear": lambda i, o, seed=42: LinearLayer(int(i), int(o), int(seed)),
            "nn_attention": lambda d, seed=42: SelfAttentionLayer(int(d), int(seed)),
            "nn_triaxial": lambda i, h, o, depth=2, act="sigmoid", seed=42: TriAxialNetwork(
                int(i), int(h), int(o), int(depth), str(act), int(seed)
            ),
            "nn_sgd": lambda p, lr=0.05, momentum=0.0: SGDOptimizer(_params_of(p), float(lr), float(momentum)),
            "nn_adam": lambda p, lr=0.02: AdamOptimizer(_params_of(p), float(lr)),
            "opt_step": _opt_step,
            "opt_zero_grad": _zero_grad,
            "ai_memory": MindMemory,
            "mem_store": lambda mem, k, emb, payload=None, h=0, v=0: mem.store(k, emb, payload, int(h), int(v)),
            "mem_recall": lambda mem, q, top_k=3: mem.recall(q, int(top_k)),
            "cosine_sim": lambda a, b: cosine_similarity(
                MindMemory._flat_vec(a), MindMemory._flat_vec(b)
            ),
            "ai_pack_hex": _ai_pack_hex,
            "ai_unpack_hex": _ai_unpack_hex,
            "em_transmit_ai": _em_transmit_ai,
        }
        for name, fn in builtins.items():
            g(name, fn)
        g("pi", math.pi)
        g("c_light", SPEED_OF_LIGHT_MPS)
