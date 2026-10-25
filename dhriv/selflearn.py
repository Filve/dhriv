"""
selflearn.py — Auto-apprendimento di Dhriv: memoizzazione appresa di funzioni pure.

Il motore osserva a runtime quante volte ogni funzione utente viene chiamata. Quando una
funzione supera la soglia E l'analisi statica dell'AST dimostra che è PURA (nessun I/O,
nessuna mutazione di stato esterno, nessuna lettura di globali), le chiamate con
argomenti scalari vengono memoizzate. La conoscenza acquisita (quali funzioni sono
"calde e pure") è persistita su disco e riutilizzata alle esecuzioni successive: dalla
seconda esecuzione la memoizzazione parte subito, senza fase di riscaldamento.

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .ast_nodes import (
    AssignStmtNode,
    BinaryOpNode,
    CallNode,
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
    ReturnStmtNode,
    UnaryOpNode,
    WhileStmtNode,
    walk,
)

PURE_BUILTINS = {
    "len", "str", "num", "int", "float", "abs", "min", "max", "floor", "sqrt",
    "sin", "cos", "round", "upper", "lower", "join", "split", "type_of", "range",
}

_SCALARS = (int, float, str, bool, type(None))


class SelfLearningEngine:
    def __init__(
        self,
        knowledge_path: Optional[Path] = None,
        memo_threshold: int = 8,
        max_memo_entries: int = 200_000,
    ) -> None:
        self.knowledge_path = Path(knowledge_path) if knowledge_path else None
        self.memo_threshold = memo_threshold
        self.max_memo_entries = max_memo_entries
        self.knowledge: Dict[str, Any] = {"version": 1, "runs": 0, "hot_pure": {}, "languages": {}}
        self.profile: Dict[str, Dict[str, int]] = {}
        self.memo: Dict[Tuple[Any, ...], Any] = {}
        self._pure_cache: Dict[int, bool] = {}
        self._interp: Any = None
        self.load()

    # ----------------------------------------------------------- persistenza

    def load(self) -> None:
        if self.knowledge_path and self.knowledge_path.exists():
            try:
                data = json.loads(self.knowledge_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("version") == 1:
                    self.knowledge.update(data)
            except (OSError, ValueError):
                pass

    def save(self) -> None:
        if not self.knowledge_path:
            return
        self.knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        self.knowledge_path.write_text(
            json.dumps(self.knowledge, indent=2, sort_keys=True), encoding="utf-8"
        )

    def bind(self, interpreter: Any) -> None:
        self._interp = interpreter

    def observe_language(self, language: str) -> None:
        langs = self.knowledge.setdefault("languages", {})
        langs[language] = langs.get(language, 0) + 1

    def preferred_language(self, default: str = "en") -> str:
        langs = self.knowledge.get("languages", {})
        return max(langs, key=langs.get) if langs else default

    def finish_run(self) -> None:
        """Consolida ciò che è stato appreso in questa esecuzione e lo salva."""
        self.knowledge["runs"] = int(self.knowledge.get("runs", 0)) + 1
        hot = self.knowledge.setdefault("hot_pure", {})
        for name, prof in self.profile.items():
            if prof.get("pure") and prof["calls"] >= self.memo_threshold:
                entry = hot.setdefault(name, {"calls_total": 0, "memo_hits_total": 0})
                entry["calls_total"] += prof["calls"]
                entry["memo_hits_total"] += prof.get("memo_hits", 0)
        self.save()

    # ----------------------------------------------------------- hook runtime

    def lookup(self, node: FunctionDeclNode, args: List[Any]) -> Tuple[bool, Any]:
        prof = self.profile.setdefault(node.name, {"calls": 0, "memo_hits": 0, "pure": 0})
        prof["calls"] += 1
        if not self._eligible(node, prof):
            return False, None
        key = self._key(node, args)
        if key is None:
            return False, None
        if key in self.memo:
            prof["memo_hits"] += 1
            return True, self.memo[key]
        return False, None

    def store(self, node: FunctionDeclNode, args: List[Any], result: Any) -> None:
        prof = self.profile.get(node.name)
        if prof is None or not self._eligible(node, prof) or not isinstance(result, _SCALARS):
            return
        key = self._key(node, args)
        if key is not None and len(self.memo) < self.max_memo_entries:
            self.memo[key] = result

    def _eligible(self, node: FunctionDeclNode, prof: Dict[str, int]) -> bool:
        known_hot = node.name in self.knowledge.get("hot_pure", {})
        if prof["calls"] < self.memo_threshold and not known_hot:
            return False
        pure = self.is_pure(node)
        prof["pure"] = 1 if pure else 0
        return pure

    @staticmethod
    def _key(node: FunctionDeclNode, args: List[Any]) -> Optional[Tuple[Any, ...]]:
        if not all(isinstance(a, _SCALARS) for a in args):
            return None
        return (id(node),) + tuple((type(a).__name__, a) for a in args)

    # ------------------------------------------------------- analisi di purezza

    def is_pure(self, node: FunctionDeclNode) -> bool:
        cached = self._pure_cache.get(id(node))
        if cached is not None:
            return cached
        result = self._pure_function(node, set())
        self._pure_cache[id(node)] = result
        return result

    def _user_function(self, name: str) -> Optional[FunctionDeclNode]:
        if self._interp is None:
            return None
        value = self._interp.globals.vars.get(name)
        return getattr(value, "node", None) if hasattr(value, "closure") and not hasattr(value, "instance") else None

    def _pure_function(self, node: FunctionDeclNode, visiting: Set[int]) -> bool:
        if id(node) in visiting:
            return True  # ricorsione: assumiamo pura, la verdetto dipende dal resto del corpo
        visiting = visiting | {id(node)}
        local_names: Set[str] = set(node.params)
        for n in walk(node.body):
            if isinstance(n, (LetStmtNode,)):
                local_names.add(n.name)
            elif isinstance(n, ForStmtNode):
                local_names.add(n.iterator_name)
        return all(self._pure_node(s, local_names, visiting) for s in node.body)

    def _pure_node(self, n: Any, local_names: Set[str], visiting: Set[int]) -> bool:
        t = type(n)
        if t is LiteralNode:
            return True
        if t is IdentifierNode:
            return n.name in local_names
        if t is LetStmtNode:
            return self._pure_node(n.value, local_names, visiting)
        if t is AssignStmtNode:
            return (
                isinstance(n.target, IdentifierNode)
                and n.target.name in local_names
                and self._pure_node(n.value, local_names, visiting)
            )
        if t is IfStmtNode:
            ok = self._pure_node(n.condition, local_names, visiting)
            ok = ok and all(self._pure_node(s, local_names, visiting) for s in n.then_branch)
            if n.else_branch:
                ok = ok and all(self._pure_node(s, local_names, visiting) for s in n.else_branch)
            return ok
        if t is WhileStmtNode:
            return self._pure_node(n.condition, local_names, visiting) and all(
                self._pure_node(s, local_names, visiting) for s in n.body
            )
        if t is ForStmtNode:
            return self._pure_node(n.iterable, local_names, visiting) and all(
                self._pure_node(s, local_names, visiting) for s in n.body
            )
        if t is ReturnStmtNode:
            return n.value is None or self._pure_node(n.value, local_names, visiting)
        if t is ExprStmtNode:
            return self._pure_node(n.expr, local_names, visiting)
        if t is BinaryOpNode:
            return self._pure_node(n.left, local_names, visiting) and self._pure_node(n.right, local_names, visiting)
        if t is UnaryOpNode:
            return self._pure_node(n.operand, local_names, visiting)
        if t is ListLiteralNode:
            return all(self._pure_node(e, local_names, visiting) for e in n.elements)
        if t is MapLiteralNode:
            return all(
                self._pure_node(k, local_names, visiting) and self._pure_node(v, local_names, visiting)
                for k, v in n.entries
            )
        if t is IndexAccessNode:
            return self._pure_node(n.obj, local_names, visiting) and self._pure_node(n.index, local_names, visiting)
        if t is CallNode:
            if not isinstance(n.callee, IdentifierNode) or n.callee.name in local_names:
                return False
            if not all(self._pure_node(a, local_names, visiting) for a in n.args):
                return False
            name = n.callee.name
            if name in PURE_BUILTINS:
                return True
            target = self._user_function(name)
            return target is not None and self._pure_function(target, visiting)
        return False

    # ----------------------------------------------------------------- report

    def report(self) -> Dict[str, Any]:
        functions = {}
        for name, prof in sorted(self.profile.items()):
            functions[name] = {
                "calls": prof["calls"],
                "memo_hits": prof.get("memo_hits", 0),
                "pure": bool(prof.get("pure")),
            }
        hot = [n for n, p in self.profile.items() if p.get("pure") and p["calls"] >= self.memo_threshold]
        return {
            "runs_learned": self.knowledge.get("runs", 0),
            "functions": functions,
            "memoized_functions": sorted(hot),
            "memo_entries": len(self.memo),
            "known_hot_pure": sorted(self.knowledge.get("hot_pure", {}).keys()),
            "preferred_language": self.preferred_language(),
        }
