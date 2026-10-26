"""
ast_nodes.py — Albero Sintattico Astratto (AST) formale del linguaggio Dhriv.

Rappresenta:
- Entità OOP Tri-Assiali (Verticale, Orizzontale, Diagonale);
- Funzioni, Lambda, Metodi, Campi e Costruttori;
- Operatori Tri-Assiali (`|>`, `<->`, `</>`) e Operatori Bitwise (`&`, `|`, `^`, `~`, `<<`, `>>`);
- Dichiarazioni di Onde di Frequenza / Elettromagnetiche (`wave`);
- Controllo di flusso, Collezioni ed Espressioni.

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


class ASTNode:
    """Nodo radice di tutte le strutture AST di Dhriv."""


@dataclass
class ProgramNode(ASTNode):
    statements: List[ASTNode]


@dataclass
class FieldDeclNode(ASTNode):
    name: str
    value: ASTNode
    line: int = 1


@dataclass
class FunctionDeclNode(ASTNode):
    name: str
    params: List[str]
    body: List[ASTNode]
    is_method: bool = False
    line: int = 1


@dataclass
class LambdaNode(ASTNode):
    params: List[str]
    body: List[ASTNode]
    line: int = 1


@dataclass
class EntityDeclNode(ASTNode):
    """
    Dichiarazione OOP Tri-Assiale:
    - `vertical_parents`: ereditarietà gerarchica profonda (superclassi/stack);
    - `horizontal_peers`: composizione paritaria di trait/mixin/cluster;
    - `diagonal_bridges`: ponti trasversali diretti (cross-layer + cross-domain).
    """
    name: str
    vertical_parents: List[str] = field(default_factory=list)
    horizontal_peers: List[str] = field(default_factory=list)
    diagonal_bridges: List[str] = field(default_factory=list)
    fields: List[FieldDeclNode] = field(default_factory=list)
    methods: List[FunctionDeclNode] = field(default_factory=list)
    line: int = 1


@dataclass
class LetStmtNode(ASTNode):
    name: str
    value: ASTNode
    is_wave: bool = False
    line: int = 1


@dataclass
class AssignStmtNode(ASTNode):
    target: ASTNode  # IdentifierNode, MemberAccessNode, IndexAccessNode
    value: ASTNode
    line: int = 1


@dataclass
class IfStmtNode(ASTNode):
    condition: ASTNode
    then_branch: List[ASTNode]
    else_branch: Optional[List[ASTNode]] = None
    line: int = 1


@dataclass
class WhileStmtNode(ASTNode):
    condition: ASTNode
    body: List[ASTNode]
    line: int = 1


@dataclass
class ForStmtNode(ASTNode):
    iterator_name: str
    iterable: ASTNode
    body: List[ASTNode]
    line: int = 1


@dataclass
class ReturnStmtNode(ASTNode):
    value: Optional[ASTNode] = None
    line: int = 1


@dataclass
class ExprStmtNode(ASTNode):
    expr: ASTNode
    line: int = 1


# --- Espressioni ---

@dataclass
class LiteralNode(ASTNode):
    value: Any
    line: int = 1


@dataclass
class IdentifierNode(ASTNode):
    name: str
    line: int = 1


@dataclass
class ListLiteralNode(ASTNode):
    elements: List[ASTNode]
    line: int = 1


@dataclass
class MapLiteralNode(ASTNode):
    entries: List[Tuple[ASTNode, ASTNode]]
    line: int = 1


@dataclass
class UnaryOpNode(ASTNode):
    op: str
    operand: ASTNode
    line: int = 1


@dataclass
class BinaryOpNode(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode
    line: int = 1


@dataclass
class TriAxialOpNode(ASTNode):
    """Operatore nativo Tri-Assiale: `|>` (VERTICAL), `<->` (HORIZONTAL), `</>` (DIAGONAL)."""
    axis: str  # "VERTICAL" | "HORIZONTAL" | "DIAGONAL"
    left: ASTNode
    right: ASTNode
    line: int = 1


@dataclass
class NewInstanceNode(ASTNode):
    entity_name: str
    args: List[ASTNode]
    line: int = 1


@dataclass
class CallNode(ASTNode):
    callee: ASTNode
    args: List[ASTNode]
    line: int = 1


@dataclass
class MemberAccessNode(ASTNode):
    obj: ASTNode
    member: str
    line: int = 1


@dataclass
class IndexAccessNode(ASTNode):
    obj: ASTNode
    index: ASTNode
    line: int = 1


def walk(node: Any):
    """Visita ricorsivamente un AST (o una sua parte) restituendo ogni `ASTNode` incontrato."""
    import dataclasses

    if isinstance(node, ASTNode):
        yield node
        for f in dataclasses.fields(node):
            yield from walk(getattr(node, f.name))
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from walk(item)
