"""
bytecode_vm.py — Ottimizzatore AST, Compilatore Bytecode a 64 bit, Disassemblatore e
Macchina Virtuale (`DhrivVM`) per il linguaggio Dhriv.

Ogni istruzione è codificata come `InstructionWord` a 64 bit (`binary_core.py`):
`[8b OPCODE | 2b AXIS (SELF/V/H/D) | 6b FLAGS | 16b ARG_A | 32b ARG_B]`.

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

from __future__ import annotations

import copy
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .ast_nodes import (
    ASTNode,
    AssignStmtNode,
    BinaryOpNode,
    CallNode,
    EntityDeclNode,
    ExprStmtNode,
    FieldDeclNode,
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
    AXIS_NAME_TO_CODE,
    BINARY_OP_TO_SUB,
    SUB_TO_BINARY_OP,
    SUB_TO_UNARY_OP,
    UNARY_OP_TO_SUB,
    AxisCode,
    InstructionWord,
    OpCode,
)
from .interpreter import DhrivCancelled, DhrivInterpreter, DhrivRuntimeError, Environment
from .parser import DhrivParser
from .triaxial_oop import TriAxialClass, TriAxialObject
from .wave_electromagnetic import synthesize_wave

BC_MAGIC = b"DHBC"
BC_VERSION = 1

# Tag binari per il Constant Pool
_TAG_NULL = 0x00
_TAG_FALSE = 0x01
_TAG_TRUE = 0x02
_TAG_INT = 0x03
_TAG_FLOAT = 0x04
_TAG_STR = 0x05
_TAG_CHUNK = 0x06
_TAG_ENTITY = 0x07


# ============================================================================
# 1. OTTIMIZZATORE COMPILE-TIME (CONSTANT FOLDING)
# ============================================================================

def optimize_ast(program: ProgramNode) -> ProgramNode:
    """Esegue il constant folding sicuro sull'AST preservando la semantica degli errori runtime."""
    return ProgramNode(statements=[_fold_node(s) for s in program.statements])


def _fold_node(node: ASTNode) -> ASTNode:
    t = type(node)
    if t is LetStmtNode:
        return LetStmtNode(node.name, _fold_node(node.value), node.is_wave, node.line)
    if t is AssignStmtNode:
        return AssignStmtNode(_fold_node(node.target), _fold_node(node.value), node.line)
    if t is ExprStmtNode:
        return ExprStmtNode(_fold_node(node.expr), node.line)
    if t is ReturnStmtNode:
        return ReturnStmtNode(_fold_node(node.value) if node.value is not None else None, node.line)
    if t is IfStmtNode:
        cond = _fold_node(node.condition)
        then_b = [_fold_node(s) for s in node.then_branch]
        else_b = [_fold_node(s) for s in node.else_branch] if node.else_branch is not None else None
        return IfStmtNode(cond, then_b, else_b, node.line)
    if t is WhileStmtNode:
        return WhileStmtNode(_fold_node(node.condition), [_fold_node(s) for s in node.body], node.line)
    if t is ForStmtNode:
        return ForStmtNode(node.iterator_name, _fold_node(node.iterable), [_fold_node(s) for s in node.body], node.line)
    if t is FunctionDeclNode:
        return FunctionDeclNode(node.name, list(node.params), [_fold_node(s) for s in node.body], node.is_method, node.line)
    if t is LambdaNode:
        return LambdaNode(list(node.params), [_fold_node(s) for s in node.body], node.line)
    if t is EntityDeclNode:
        fields = [FieldDeclNode(f.name, _fold_node(f.value), f.line) for f in node.fields]
        methods = [_fold_node(m) for m in node.methods]
        return EntityDeclNode(
            node.name,
            list(node.vertical_parents),
            list(node.horizontal_peers),
            list(node.diagonal_bridges),
            fields,
            methods,  # type: ignore[arg-type]
            node.line,
        )
    if t is ListLiteralNode:
        return ListLiteralNode([_fold_node(e) for e in node.elements], node.line)
    if t is MapLiteralNode:
        return MapLiteralNode([(_fold_node(k), _fold_node(v)) for k, v in node.entries], node.line)
    if t is CallNode:
        return CallNode(_fold_node(node.callee), [_fold_node(a) for a in node.args], node.line)
    if t is NewInstanceNode:
        return NewInstanceNode(node.entity_name, [_fold_node(a) for a in node.args], node.line)
    if t is MemberAccessNode:
        return MemberAccessNode(_fold_node(node.obj), node.member, node.line)
    if t is IndexAccessNode:
        return IndexAccessNode(_fold_node(node.obj), _fold_node(node.index), node.line)
    if t is TriAxialOpNode:
        return TriAxialOpNode(node.axis, _fold_node(node.left), _fold_node(node.right), node.line)
    if t is UnaryOpNode:
        opnd = _fold_node(node.operand)
        if isinstance(opnd, LiteralNode):
            v = opnd.value
            if node.op == "not":
                return LiteralNode(not v, node.line)
            if node.op == "-" and isinstance(v, (int, float)) and not isinstance(v, bool):
                return LiteralNode(-v, node.line)
            if node.op == "~" and isinstance(v, int) and not isinstance(v, bool):
                return LiteralNode(~v, node.line)
        return UnaryOpNode(node.op, opnd, node.line)
    if t is BinaryOpNode:
        left = _fold_node(node.left)
        right = _fold_node(node.right)
        if isinstance(left, LiteralNode) and isinstance(right, LiteralNode):
            folded = _try_fold_binary(node.op, left.value, right.value, node.line)
            if folded is not None:
                return folded
        return BinaryOpNode(node.op, left, right, node.line)
    return node


def _try_fold_binary(op: str, l: Any, r: Any, line: int) -> Optional[LiteralNode]:
    try:
        if op == "and":
            return LiteralNode(r if l else l, line)
        if op == "or":
            return LiteralNode(l if l else r, line)
        if isinstance(l, (int, float)) and isinstance(r, (int, float)) and not isinstance(l, bool) and not isinstance(r, bool):
            if op == "+":
                return LiteralNode(l + r, line)
            if op == "-":
                return LiteralNode(l - r, line)
            if op == "*":
                return LiteralNode(l * r, line)
            if op == "/" and r != 0:
                if isinstance(l, int) and isinstance(r, int) and l % r == 0:
                    return LiteralNode(l // r, line)
                return LiteralNode(l / r, line)
            if op == "%" and r != 0:
                return LiteralNode(l % r, line)
            if op == "**" and abs(r) <= 64:
                return LiteralNode(l ** r, line)
            if op in ("&", "|", "^", "<<", ">>") and isinstance(l, int) and isinstance(r, int):
                if op == "&":
                    return LiteralNode(l & r, line)
                if op == "|":
                    return LiteralNode(l | r, line)
                if op == "^":
                    return LiteralNode(l ^ r, line)
                if 0 <= r <= 64:
                    return LiteralNode((l << r) if op == "<<" else (l >> r), line)
            if op == "==":
                return LiteralNode(l == r, line)
            if op == "!=":
                return LiteralNode(l != r, line)
            if op == "<":
                return LiteralNode(l < r, line)
            if op == "<=":
                return LiteralNode(l <= r, line)
            if op == ">":
                return LiteralNode(l > r, line)
            if op == ">=":
                return LiteralNode(l >= r, line)
        if isinstance(l, str) and isinstance(r, str) and op == "+":
            return LiteralNode(l + r, line)
    except Exception:
        return None
    return None


# ============================================================================
# 2. RAPPRESENTAZIONE BINARIA DEL BYTECODE (`CodeChunk` & `EntitySpec`)
# ============================================================================

@dataclass
class EntitySpec:
    name: str
    vertical_parents: List[str]
    horizontal_peers: List[str]
    diagonal_bridges: List[str]
    field_names: List[str]
    field_chunks: List["CodeChunk"]
    methods: List["CodeChunk"]
    line: int = 1


@dataclass
class CodeChunk:
    name: str
    params: List[str] = field(default_factory=list)
    is_method: bool = False
    constants: List[Any] = field(default_factory=list)
    instructions: List[InstructionWord] = field(default_factory=list)
    lines: List[int] = field(default_factory=list)
    ast_node: Optional[FunctionDeclNode] = None  # opzionale per integrazione SelfLearningEngine

    def add_const(self, value: Any) -> int:
        for i, existing in enumerate(self.constants):
            if type(existing) is type(value) and isinstance(value, (int, float, str, bool, type(None))) and existing == value:
                return i
        self.constants.append(value)
        return len(self.constants) - 1

    def emit(
        self,
        opcode: OpCode,
        arg_a: int = 0,
        arg_b: int = 0,
        axis: AxisCode = AxisCode.SELF,
        flags: int = 0,
        line: int = 1,
    ) -> int:
        self.instructions.append(InstructionWord(opcode, axis, flags, arg_a, arg_b))
        self.lines.append(line)
        return len(self.instructions) - 1

    def patch_b(self, pc: int, target_b: int) -> None:
        old = self.instructions[pc]
        self.instructions[pc] = InstructionWord(old.opcode, old.axis, old.flags, old.arg_a, target_b)

    # ---------------------------------------------- serializzazione binaria

    def to_bytes(self) -> bytes:
        buf = bytearray()
        _write_str(buf, self.name)
        buf.extend(struct.pack(">HB", len(self.params), 1 if self.is_method else 0))
        for p in self.params:
            _write_str(buf, p)

        buf.extend(struct.pack(">I", len(self.constants)))
        for c in self.constants:
            _write_const(buf, c)

        buf.extend(struct.pack(">I", len(self.instructions)))
        for ins, ln in zip(self.instructions, self.lines):
            buf.extend(ins.pack())
            buf.extend(struct.pack(">I", max(0, int(ln))))
        return bytes(buf)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> Tuple["CodeChunk", int]:
        name, offset = _read_str(data, offset)
        param_count, is_meth = struct.unpack(">HB", data[offset : offset + 3])
        offset += 3
        params: List[str] = []
        for _ in range(param_count):
            p, offset = _read_str(data, offset)
            params.append(p)

        (const_count,) = struct.unpack(">I", data[offset : offset + 4])
        offset += 4
        constants: List[Any] = []
        for _ in range(const_count):
            val, offset = _read_const(data, offset)
            constants.append(val)

        (ins_count,) = struct.unpack(">I", data[offset : offset + 4])
        offset += 4
        instructions: List[InstructionWord] = []
        lines: List[int] = []
        for _ in range(ins_count):
            ins = InstructionWord.unpack(data[offset : offset + 8])
            (ln,) = struct.unpack(">I", data[offset + 8 : offset + 12])
            offset += 12
            instructions.append(ins)
            lines.append(ln)

        return cls(name=name, params=params, is_method=bool(is_meth), constants=constants, instructions=instructions, lines=lines), offset

    # ---------------------------------------------- disassemblatore leggibile

    def disassemble(self, show_bits: bool = False, indent: str = "") -> str:
        out: List[str] = []
        header = f"{indent}=== CHUNK '{self.name}' ({len(self.instructions)} words x 64-bit = {len(self.instructions)*8} bytes) ==="
        out.append(header)
        for pc, ins in enumerate(self.instructions):
            hint = self._hint_for(ins)
            out.append(indent + ins.format_asm(pc, hint=hint, show_bits=show_bits))
        for c in self.constants:
            if isinstance(c, CodeChunk):
                out.append("")
                out.append(c.disassemble(show_bits=show_bits, indent=indent + "  "))
            elif isinstance(c, EntitySpec):
                out.append("")
                out.append(f"{indent}  === ENTITY '{c.name}' (V={c.vertical_parents}, H={c.horizontal_peers}, D={c.diagonal_bridges}) ===")
                for m in c.methods:
                    out.append(m.disassemble(show_bits=show_bits, indent=indent + "    "))
        return "\n".join(out)

    def _hint_for(self, ins: InstructionWord) -> str:
        op = ins.opcode
        if op in (OpCode.LOAD_CONST, OpCode.DEF_VAR, OpCode.LOAD_VAR, OpCode.STORE_VAR, OpCode.GET_MEMBER, OpCode.SET_MEMBER, OpCode.NEW_ENTITY, OpCode.MAKE_FN, OpCode.DECL_ENTITY, OpCode.ITER_NEXT):
            if 0 <= ins.arg_a < len(self.constants):
                c = self.constants[ins.arg_a]
                if isinstance(c, CodeChunk):
                    return f"fn {c.name}({', '.join(c.params)})"
                if isinstance(c, EntitySpec):
                    return f"entity {c.name}"
                return repr(c)
        if op == OpCode.BINARY_OP:
            return SUB_TO_BINARY_OP.get(ins.arg_a, "?")  # type: ignore[arg-type]
        if op == OpCode.UNARY_OP:
            return SUB_TO_UNARY_OP.get(ins.arg_a, "?")  # type: ignore[arg-type]
        return ""


def _write_str(buf: bytearray, s: str) -> None:
    raw = s.encode("utf-8")
    buf.extend(struct.pack(">I", len(raw)))
    buf.extend(raw)


def _read_str(data: bytes, offset: int) -> Tuple[str, int]:
    (n,) = struct.unpack(">I", data[offset : offset + 4])
    offset += 4
    return data[offset : offset + n].decode("utf-8"), offset + n


def _write_str_list(buf: bytearray, items: List[str]) -> None:
    buf.extend(struct.pack(">H", len(items)))
    for s in items:
        _write_str(buf, s)


def _read_str_list(data: bytes, offset: int) -> Tuple[List[str], int]:
    (n,) = struct.unpack(">H", data[offset : offset + 2])
    offset += 2
    out: List[str] = []
    for _ in range(n):
        s, offset = _read_str(data, offset)
        out.append(s)
    return out, offset


def _write_const(buf: bytearray, c: Any) -> None:
    if c is None:
        buf.append(_TAG_NULL)
    elif c is False:
        buf.append(_TAG_FALSE)
    elif c is True:
        buf.append(_TAG_TRUE)
    elif isinstance(c, int):
        buf.append(_TAG_INT)
        # Codifica intero arbitrario come stringa ASCII compatta per supportare anche interi > 64 bit
        _write_str(buf, str(c))
    elif isinstance(c, float):
        buf.append(_TAG_FLOAT)
        buf.extend(struct.pack(">d", c))
    elif isinstance(c, str):
        buf.append(_TAG_STR)
        _write_str(buf, c)
    elif isinstance(c, CodeChunk):
        buf.append(_TAG_CHUNK)
        raw = c.to_bytes()
        buf.extend(struct.pack(">I", len(raw)))
        buf.extend(raw)
    elif isinstance(c, EntitySpec):
        buf.append(_TAG_ENTITY)
        _write_str(buf, c.name)
        _write_str_list(buf, c.vertical_parents)
        _write_str_list(buf, c.horizontal_peers)
        _write_str_list(buf, c.diagonal_bridges)
        _write_str_list(buf, c.field_names)
        buf.extend(struct.pack(">H", len(c.field_chunks)))
        for fc in c.field_chunks:
            raw_fc = fc.to_bytes()
            buf.extend(struct.pack(">I", len(raw_fc)))
            buf.extend(raw_fc)
        buf.extend(struct.pack(">H", len(c.methods)))
        for mc in c.methods:
            raw_mc = mc.to_bytes()
            buf.extend(struct.pack(">I", len(raw_mc)))
            buf.extend(raw_mc)
        buf.extend(struct.pack(">I", c.line))
    else:
        raise TypeError(f"Costante non serializzabile in bytecode: {type(c).__name__}")


def _read_const(data: bytes, offset: int) -> Tuple[Any, int]:
    tag = data[offset]
    offset += 1
    if tag == _TAG_NULL:
        return None, offset
    if tag == _TAG_FALSE:
        return False, offset
    if tag == _TAG_TRUE:
        return True, offset
    if tag == _TAG_INT:
        s, offset = _read_str(data, offset)
        return int(s), offset
    if tag == _TAG_FLOAT:
        (val,) = struct.unpack(">d", data[offset : offset + 8])
        return val, offset + 8
    if tag == _TAG_STR:
        return _read_str(data, offset)
    if tag == _TAG_CHUNK:
        (n,) = struct.unpack(">I", data[offset : offset + 4])
        offset += 4
        chunk, _ = CodeChunk.from_bytes(data, offset)
        return chunk, offset + n
    if tag == _TAG_ENTITY:
        name, offset = _read_str(data, offset)
        vp, offset = _read_str_list(data, offset)
        hp, offset = _read_str_list(data, offset)
        db, offset = _read_str_list(data, offset)
        fnames, offset = _read_str_list(data, offset)
        (fc_count,) = struct.unpack(">H", data[offset : offset + 2])
        offset += 2
        fchunks: List[CodeChunk] = []
        for _ in range(fc_count):
            (clen,) = struct.unpack(">I", data[offset : offset + 4])
            offset += 4
            ch, _ = CodeChunk.from_bytes(data, offset)
            fchunks.append(ch)
            offset += clen
        (m_count,) = struct.unpack(">H", data[offset : offset + 2])
        offset += 2
        mchunks: List[CodeChunk] = []
        for _ in range(m_count):
            (clen,) = struct.unpack(">I", data[offset : offset + 4])
            offset += 4
            ch, _ = CodeChunk.from_bytes(data, offset)
            mchunks.append(ch)
            offset += clen
        (line,) = struct.unpack(">I", data[offset : offset + 4])
        offset += 4
        return EntitySpec(name, vp, hp, db, fnames, fchunks, mchunks, line), offset
    raise ValueError(f"Tag costante bytecode sconosciuto: 0x{tag:02X}")


@dataclass
class BytecodeModule:
    root: CodeChunk

    def to_bytes(self) -> bytes:
        body = self.root.to_bytes()
        return BC_MAGIC + struct.pack(">HI", BC_VERSION, len(body)) + body

    @classmethod
    def from_bytes(cls, blob: bytes) -> "BytecodeModule":
        if len(blob) < 10 or blob[:4] != BC_MAGIC:
            raise ValueError("Magic 'DHBC' mancante nel modulo bytecode.")
        ver, length = struct.unpack(">HI", blob[4:10])
        if ver != BC_VERSION:
            raise ValueError(f"Versione bytecode non supportata: {ver}")
        chunk, _ = CodeChunk.from_bytes(blob[10 : 10 + length], 0)
        return cls(root=chunk)

    def disassemble(self, show_bits: bool = False) -> str:
        return self.root.disassemble(show_bits=show_bits)


# ============================================================================
# 3. COMPILATORE AST -> BYTECODE A 64 BIT
# ============================================================================

class BytecodeCompiler:
    """Compila un `ProgramNode` (ottimizzato) in un `BytecodeModule` a parole da 64 bit."""

    @classmethod
    def compile_program(cls, program: ProgramNode, optimize: bool = True) -> BytecodeModule:
        prog = optimize_ast(program) if optimize else program
        chunk = CodeChunk(name="<module>")
        compiler = cls(chunk)
        for stmt in prog.statements:
            compiler._compile_stmt(stmt)
        chunk.emit(OpCode.HALT, line=1)
        return BytecodeModule(root=chunk)

    @classmethod
    def compile_source(cls, source: str, optimize: bool = True) -> BytecodeModule:
        return cls.compile_program(DhrivParser.parse_source(source), optimize=optimize)

    def __init__(self, chunk: CodeChunk) -> None:
        self.chunk = chunk

    def _compile_function_body(self, name: str, params: List[str], body: List[ASTNode], is_method: bool, ast_ref: Optional[FunctionDeclNode] = None) -> CodeChunk:
        sub = CodeChunk(name=name, params=list(params), is_method=is_method, ast_node=ast_ref)
        sub_comp = BytecodeCompiler(sub)
        for s in body:
            sub_comp._compile_stmt(s)
        sub.emit(OpCode.RETURN, flags=0, line=getattr(ast_ref, "line", 1))
        return sub

    def _compile_expr_chunk(self, name: str, expr: ASTNode, line: int) -> CodeChunk:
        sub = CodeChunk(name=name)
        sub_comp = BytecodeCompiler(sub)
        sub_comp._compile_expr(expr)
        sub.emit(OpCode.RETURN, flags=1, line=line)
        return sub

    def _compile_stmt(self, node: ASTNode) -> None:
        t = type(node)
        line = getattr(node, "line", 1)

        if t is ExprStmtNode:
            self._compile_expr(node.expr)
            self.chunk.emit(OpCode.POP, line=line)
        elif t is LetStmtNode:
            self._compile_expr(node.value)
            idx = self.chunk.add_const(node.name)
            self.chunk.emit(OpCode.DEF_VAR, arg_a=idx, flags=1 if node.is_wave else 0, line=line)
        elif t is AssignStmtNode:
            target = node.target
            if isinstance(target, IdentifierNode):
                self._compile_expr(node.value)
                idx = self.chunk.add_const(target.name)
                self.chunk.emit(OpCode.STORE_VAR, arg_a=idx, line=line)
            elif isinstance(target, MemberAccessNode):
                self._compile_expr(target.obj)
                self._compile_expr(node.value)
                idx = self.chunk.add_const(target.member)
                self.chunk.emit(OpCode.SET_MEMBER, arg_a=idx, line=line)
            elif isinstance(target, IndexAccessNode):
                self._compile_expr(target.obj)
                self._compile_expr(target.index)
                self._compile_expr(node.value)
                self.chunk.emit(OpCode.SET_INDEX, line=line)
            else:
                raise ValueError("Bersaglio di assegnazione non supportato nel compilatore bytecode.")
        elif t is IfStmtNode:
            self._compile_expr(node.condition)
            jmp_false = self.chunk.emit(OpCode.JUMP_IF_FALSE, line=line)
            self.chunk.emit(OpCode.PUSH_SCOPE, line=line)
            for s in node.then_branch:
                self._compile_stmt(s)
            self.chunk.emit(OpCode.POP_SCOPE, line=line)
            if node.else_branch is not None:
                jmp_end = self.chunk.emit(OpCode.JUMP, line=line)
                self.chunk.patch_b(jmp_false, len(self.chunk.instructions))
                self.chunk.emit(OpCode.PUSH_SCOPE, line=line)
                for s in node.else_branch:
                    self._compile_stmt(s)
                self.chunk.emit(OpCode.POP_SCOPE, line=line)
                self.chunk.patch_b(jmp_end, len(self.chunk.instructions))
            else:
                self.chunk.patch_b(jmp_false, len(self.chunk.instructions))
        elif t is WhileStmtNode:
            loop_start = len(self.chunk.instructions)
            self._compile_expr(node.condition)
            jmp_exit = self.chunk.emit(OpCode.JUMP_IF_FALSE, line=line)
            self.chunk.emit(OpCode.PUSH_SCOPE, line=line)
            for s in node.body:
                self._compile_stmt(s)
            self.chunk.emit(OpCode.POP_SCOPE, line=line)
            self.chunk.emit(OpCode.JUMP, arg_b=loop_start, line=line)
            self.chunk.patch_b(jmp_exit, len(self.chunk.instructions))
        elif t is ForStmtNode:
            self._compile_expr(node.iterable)
            self.chunk.emit(OpCode.ITER_PREP, line=line)
            loop_step = len(self.chunk.instructions)
            name_idx = self.chunk.add_const(node.iterator_name)
            jmp_iter = self.chunk.emit(OpCode.ITER_NEXT, arg_a=name_idx, line=line)
            for s in node.body:
                self._compile_stmt(s)
            self.chunk.emit(OpCode.POP_SCOPE, line=line)
            self.chunk.emit(OpCode.JUMP, arg_b=loop_step, line=line)
            self.chunk.patch_b(jmp_iter, len(self.chunk.instructions))
        elif t is ReturnStmtNode:
            if node.value is not None:
                self._compile_expr(node.value)
                self.chunk.emit(OpCode.RETURN, flags=1, line=line)
            else:
                self.chunk.emit(OpCode.RETURN, flags=0, line=line)
        elif t is FunctionDeclNode:
            sub = self._compile_function_body(node.name, node.params, node.body, is_method=False, ast_ref=node)
            idx = self.chunk.add_const(sub)
            self.chunk.emit(OpCode.MAKE_FN, arg_a=idx, flags=1, line=line)
        elif t is EntityDeclNode:
            fnames = [f.name for f in node.fields]
            fchunks = [self._compile_expr_chunk(f"<field:{node.name}.{f.name}>", f.value, f.line) for f in node.fields]
            mchunks = [self._compile_function_body(m.name, m.params, m.body, is_method=True, ast_ref=m) for m in node.methods]
            spec = EntitySpec(
                name=node.name,
                vertical_parents=list(node.vertical_parents),
                horizontal_peers=list(node.horizontal_peers),
                diagonal_bridges=list(node.diagonal_bridges),
                field_names=fnames,
                field_chunks=fchunks,
                methods=mchunks,
                line=line,
            )
            idx = self.chunk.add_const(spec)
            self.chunk.emit(OpCode.DECL_ENTITY, arg_a=idx, line=line)
        else:
            raise ValueError(f"Nodo istruzione non supportato in BytecodeCompiler: {t.__name__}")

    def _compile_expr(self, node: ASTNode) -> None:
        t = type(node)
        line = getattr(node, "line", 1)

        if t is LiteralNode:
            idx = self.chunk.add_const(node.value)
            self.chunk.emit(OpCode.LOAD_CONST, arg_a=idx, line=line)
        elif t is IdentifierNode:
            idx = self.chunk.add_const(node.name)
            self.chunk.emit(OpCode.LOAD_VAR, arg_a=idx, line=line)
        elif t is LambdaNode:
            fnode = FunctionDeclNode("<lambda>", list(node.params), list(node.body), line=line)
            sub = self._compile_function_body("<lambda>", node.params, node.body, is_method=False, ast_ref=fnode)
            idx = self.chunk.add_const(sub)
            self.chunk.emit(OpCode.MAKE_FN, arg_a=idx, flags=0, line=line)
        elif t is UnaryOpNode:
            self._compile_expr(node.operand)
            self.chunk.emit(OpCode.UNARY_OP, arg_a=int(UNARY_OP_TO_SUB[node.op]), line=line)
        elif t is BinaryOpNode:
            if node.op == "and":
                self._compile_expr(node.left)
                jmp = self.chunk.emit(OpCode.JUMP_IF_FALSE_KEEP, line=line)
                self.chunk.emit(OpCode.POP, line=line)
                self._compile_expr(node.right)
                self.chunk.patch_b(jmp, len(self.chunk.instructions))
                return
            if node.op == "or":
                self._compile_expr(node.left)
                jmp = self.chunk.emit(OpCode.JUMP_IF_TRUE, line=line)
                self.chunk.emit(OpCode.POP, line=line)
                self._compile_expr(node.right)
                self.chunk.patch_b(jmp, len(self.chunk.instructions))
                return
            self._compile_expr(node.left)
            self._compile_expr(node.right)
            self.chunk.emit(OpCode.BINARY_OP, arg_a=int(BINARY_OP_TO_SUB[node.op]), line=line)
        elif t is ListLiteralNode:
            for e in node.elements:
                self._compile_expr(e)
            self.chunk.emit(OpCode.BUILD_LIST, arg_a=len(node.elements), line=line)
        elif t is MapLiteralNode:
            for k, v in node.entries:
                self._compile_expr(k)
                self._compile_expr(v)
            self.chunk.emit(OpCode.BUILD_MAP, arg_a=len(node.entries), line=line)
        elif t is IndexAccessNode:
            self._compile_expr(node.obj)
            self._compile_expr(node.index)
            self.chunk.emit(OpCode.GET_INDEX, line=line)
        elif t is MemberAccessNode:
            self._compile_expr(node.obj)
            idx = self.chunk.add_const(node.member)
            self.chunk.emit(OpCode.GET_MEMBER, arg_a=idx, line=line)
        elif t is NewInstanceNode:
            for a in node.args:
                self._compile_expr(a)
            idx = self.chunk.add_const(node.entity_name)
            self.chunk.emit(OpCode.NEW_ENTITY, arg_a=idx, arg_b=len(node.args), line=line)
        elif t is CallNode:
            self._compile_expr(node.callee)
            for a in node.args:
                self._compile_expr(a)
            self.chunk.emit(OpCode.CALL, arg_a=len(node.args), line=line)
        elif t is TriAxialOpNode:
            ax = AXIS_NAME_TO_CODE[node.axis]
            if node.axis == "VERTICAL":
                self._compile_expr(node.left)
                if isinstance(node.right, CallNode):
                    self._compile_expr(node.right.callee)
                    for a in node.right.args:
                        self._compile_expr(a)
                    self.chunk.emit(OpCode.TRIAXIAL_OP, axis=ax, arg_a=len(node.right.args), flags=1, line=line)
                else:
                    self._compile_expr(node.right)
                    self.chunk.emit(OpCode.TRIAXIAL_OP, axis=ax, arg_a=0, flags=0, line=line)
            else:
                self._compile_expr(node.left)
                self._compile_expr(node.right)
                self.chunk.emit(OpCode.TRIAXIAL_OP, axis=ax, line=line)
        else:
            raise ValueError(f"Espressione non supportata in BytecodeCompiler: {t.__name__}")


# ============================================================================
# 4. MACCHINA VIRTUALE A 64 BIT (`DhrivVM`)
# ============================================================================

class VMFunction:
    """Funzione utente compilata in bytecode a 64 bit, invocabile nativamente."""

    def __init__(self, chunk: CodeChunk, closure: Environment, vm: "DhrivVM") -> None:
        self.chunk = chunk
        self.closure = closure
        self.vm = vm

    @property
    def name(self) -> str:
        return self.chunk.name

    @property
    def node(self) -> Optional[FunctionDeclNode]:
        return self.chunk.ast_node

    def __call__(self, *args: Any) -> Any:
        return self.vm.call_chunk(self.chunk, list(args), self.closure, self_obj=None, line=None)

    def __repr__(self) -> str:
        return f"<fn {self.chunk.name}>"


class VMBoundMethod:
    def __init__(self, instance: TriAxialObject, chunk: CodeChunk, closure: Environment, vm: "DhrivVM") -> None:
        self.instance = instance
        self.chunk = chunk
        self.closure = closure
        self.vm = vm

    def __call__(self, *args: Any) -> Any:
        return self.vm.call_chunk(self.chunk, list(args), self.closure, self_obj=self.instance, line=None)

    def __repr__(self) -> str:
        return f"<fn {self.chunk.name}>"


class DhrivVM:
    """
    Macchina Virtuale di Dhriv: esegue direttamente stream di `InstructionWord` a 64 bit,
    condividendo il runtime Tri-Assiale, Onde e AI di `DhrivInterpreter`.
    """

    def __init__(
        self,
        echo: bool = False,
        max_steps: int = 20_000_000,
        allow_legacy: bool = False,
        learner: Any = None,
        cancel_event: Any = None,
    ) -> None:
        self.interp = DhrivInterpreter(
            echo=echo,
            max_steps=max_steps,
            allow_legacy=allow_legacy,
            learner=learner,
            cancel_event=cancel_event,
        )
        self.interp._vm = self

    @property
    def output_lines(self) -> List[str]:
        return self.interp.output_lines

    def run_source(self, source: str) -> str:
        mod = BytecodeCompiler.compile_source(source, optimize=True)
        return self.run_module(mod)

    def run_module(self, module: BytecodeModule) -> str:
        self.interp.heartbeat = time.monotonic()
        self._exec_chunk(module.root, self.interp.globals)
        return "\n".join(self.interp.output_lines)

    def call_chunk(
        self,
        chunk: CodeChunk,
        args: List[Any],
        closure: Environment,
        self_obj: Optional[TriAxialObject],
        line: Optional[int],
    ) -> Any:
        if len(args) != len(chunk.params):
            raise DhrivRuntimeError(
                f"'{chunk.name}' richiede {len(chunk.params)} argomenti, ricevuti {len(args)}.", line
            )
        self.interp._depth += 1
        try:
            if self.interp._depth > self.interp.MAX_CALL_DEPTH:
                raise DhrivRuntimeError("Profondità massima di chiamata superata.", line)
            use_learner = (
                self.interp.learner is not None
                and self_obj is None
                and chunk.ast_node is not None
                and chunk.name != "<lambda>"
            )
            if use_learner:
                hit, cached = self.interp.learner.lookup(chunk.ast_node, args)
                if hit:
                    return cached

            fenv = Environment(closure)
            if self_obj is not None:
                fenv.define("self", self_obj)
            for pname, val in zip(chunk.params, args):
                fenv.define(pname, val)
            res = self._exec_chunk(chunk, fenv)
            if use_learner:
                self.interp.learner.store(chunk.ast_node, args, res)
            return res
        finally:
            self.interp._depth -= 1

    def _resolve_vm_member(self, obj: Any, member: str, line: Optional[int]) -> Any:
        if isinstance(obj, TriAxialObject):
            res = self.interp.registry.resolve_instance_member(obj, member)
            if not res.found:
                raise DhrivRuntimeError(
                    f"L'entità '{obj.dhriv_class.name}' non ha il membro '{member}' su nessun asse (V/H/D).",
                    line,
                )
            val = res.value
            if isinstance(val, CodeChunk):
                return VMBoundMethod(obj, val, self.interp._closures.get(id(val), self.interp.globals), self)
            if isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], TriAxialObject) and isinstance(val[1], CodeChunk):
                peer, mchunk = val
                return VMBoundMethod(peer, mchunk, self.interp._closures.get(id(mchunk), self.interp.globals), self)
        return self.interp.resolve_member_value(obj, member, line)

    def _invoke_vm(self, callee: Any, args: List[Any], line: Optional[int]) -> Any:
        if isinstance(callee, VMBoundMethod):
            return self.call_chunk(callee.chunk, args, callee.closure, callee.instance, line)
        if isinstance(callee, VMFunction):
            return self.call_chunk(callee.chunk, args, callee.closure, None, line)
        if isinstance(callee, TriAxialObject):
            for hook in ("forward", "call", "esegui"):
                res = self.interp.registry.resolve_instance_member(callee, hook)
                if res.found and isinstance(res.value, CodeChunk):
                    return self.call_chunk(
                        res.value, args, self.interp._closures.get(id(res.value), self.interp.globals), callee, line
                    )
                if res.found and isinstance(res.value, tuple) and len(res.value) == 2 and isinstance(res.value[1], CodeChunk):
                    peer, mchunk = res.value
                    return self.call_chunk(
                        mchunk, args, self.interp._closures.get(id(mchunk), self.interp.globals), peer, line
                    )
        return self.interp._invoke(callee, args, line)

    def _exec_chunk(self, chunk: CodeChunk, initial_env: Environment) -> Any:
        stack: List[Any] = []
        env = initial_env
        ins_list = chunk.instructions
        consts = chunk.constants
        lines = chunk.lines
        pc = 0
        n = len(ins_list)

        while pc < n:
            self.interp._tick()
            ins = ins_list[pc]
            line = lines[pc]
            op = ins.opcode
            pc += 1

            if op == OpCode.LOAD_CONST:
                stack.append(consts[ins.arg_a])
            elif op == OpCode.POP:
                stack.pop()
            elif op == OpCode.DUP:
                stack.append(stack[-1])
            elif op == OpCode.DEF_VAR:
                val = stack.pop()
                if ins.flags == 1 and isinstance(val, (int, float)) and not isinstance(val, bool):
                    val = synthesize_wave(float(val))
                env.define(consts[ins.arg_a], val)
            elif op == OpCode.LOAD_VAR:
                stack.append(env.lookup(consts[ins.arg_a], line))
            elif op == OpCode.STORE_VAR:
                val = stack.pop()
                env.assign(consts[ins.arg_a], val, line)
            elif op == OpCode.PUSH_SCOPE:
                env = Environment(env)
            elif op == OpCode.POP_SCOPE:
                if env.parent is not None:
                    env = env.parent
            elif op == OpCode.BINARY_OP:
                r = stack.pop()
                l = stack.pop()
                stack.append(self.interp.eval_binary_values(SUB_TO_BINARY_OP[ins.arg_a], l, r, line))  # type: ignore[index]
            elif op == OpCode.UNARY_OP:
                v = stack.pop()
                stack.append(self.interp.eval_unary_value(SUB_TO_UNARY_OP[ins.arg_a], v, line))  # type: ignore[index]
            elif op == OpCode.JUMP:
                pc = ins.arg_b
            elif op == OpCode.JUMP_IF_FALSE:
                if not stack.pop():
                    pc = ins.arg_b
            elif op == OpCode.JUMP_IF_FALSE_KEEP:
                if not stack[-1]:
                    pc = ins.arg_b
            elif op == OpCode.JUMP_IF_TRUE:
                if stack[-1]:
                    pc = ins.arg_b
            elif op == OpCode.BUILD_LIST:
                cnt = ins.arg_a
                if cnt == 0:
                    stack.append([])
                else:
                    elems = stack[-cnt:]
                    del stack[-cnt:]
                    stack.append(elems)
            elif op == OpCode.BUILD_MAP:
                cnt = ins.arg_a
                d: Dict[Any, Any] = {}
                if cnt > 0:
                    raw = stack[-(2 * cnt) :]
                    del stack[-(2 * cnt) :]
                    for i in range(0, len(raw), 2):
                        d[raw[i]] = raw[i + 1]
                stack.append(d)
            elif op == OpCode.GET_INDEX:
                idx = stack.pop()
                obj = stack.pop()
                stack.append(self.interp.eval_index_value(obj, idx, line))
            elif op == OpCode.SET_INDEX:
                val = stack.pop()
                idx = stack.pop()
                obj = stack.pop()
                try:
                    obj[idx] = val
                except (IndexError, KeyError, TypeError) as exc:
                    raise DhrivRuntimeError(f"Assegnazione per indice non valida: {exc}", line)
            elif op == OpCode.ITER_PREP:
                iterable = stack.pop()
                if isinstance(iterable, dict):
                    items = list(iterable.keys())
                elif isinstance(iterable, (list, str)):
                    items = list(iterable)
                else:
                    raise DhrivRuntimeError("Il ciclo 'for' richiede lista, testo o mappa.", line)
                stack.append([items, 0])
            elif op == OpCode.ITER_NEXT:
                it_state = stack[-1]
                items, idx = it_state[0], it_state[1]
                if idx >= len(items):
                    stack.pop()
                    pc = ins.arg_b
                else:
                    item = items[idx]
                    it_state[1] = idx + 1
                    env = Environment(env)
                    env.define(consts[ins.arg_a], item)
            elif op == OpCode.MAKE_FN:
                sub: CodeChunk = consts[ins.arg_a]
                fn_obj = VMFunction(sub, env, self)
                if ins.flags == 1:
                    env.define(sub.name, fn_obj)
                else:
                    stack.append(fn_obj)
            elif op == OpCode.DECL_ENTITY:
                spec: EntitySpec = consts[ins.arg_a]
                for axis_list in (spec.vertical_parents, spec.horizontal_peers, spec.diagonal_bridges):
                    for parent in axis_list:
                        if parent == spec.name or parent not in self.interp.registry.classes:
                            raise DhrivRuntimeError(
                                f"Entità '{spec.name}': l'entità collegata '{parent}' non è ancora dichiarata.",
                                spec.line,
                            )
                default_fields = {
                    fname: self._exec_chunk(fchunk, env)
                    for fname, fchunk in zip(spec.field_names, spec.field_chunks)
                }
                methods = {m.name: m for m in spec.methods}
                for m in spec.methods:
                    self.interp._closures[id(m)] = env
                self.interp.registry.register_class(
                    TriAxialClass(
                        name=spec.name,
                        vertical_parents=list(spec.vertical_parents),
                        horizontal_peers=list(spec.horizontal_peers),
                        diagonal_bridges=list(spec.diagonal_bridges),
                        default_fields=default_fields,
                        methods=methods,
                    )
                )
            elif op == OpCode.NEW_ENTITY:
                ename = consts[ins.arg_a]
                argc = ins.arg_b
                args = stack[-argc:] if argc > 0 else []
                if argc > 0:
                    del stack[-argc:]
                if ename not in self.interp.registry.classes:
                    raise DhrivRuntimeError(f"Entità '{ename}' non dichiarata.", line)
                instance = self.interp.registry.instantiate(ename)
                instance.fields = copy.deepcopy(instance.fields)
                ctor = self.interp.registry.resolve_method(ename, "init")
                if ctor.found and isinstance(ctor.value, CodeChunk):
                    closure = self.interp._closures.get(id(ctor.value), self.interp.globals)
                    self.call_chunk(ctor.value, args, closure, instance, line)
                elif args:
                    raise DhrivRuntimeError(
                        f"L'entità '{ename}' non ha un metodo 'init' ma riceve {len(args)} argomenti.", line
                    )
                stack.append(instance)
            elif op == OpCode.GET_MEMBER:
                obj = stack.pop()
                stack.append(self._resolve_vm_member(obj, consts[ins.arg_a], line))
            elif op == OpCode.SET_MEMBER:
                val = stack.pop()
                obj = stack.pop()
                member = consts[ins.arg_a]
                if isinstance(obj, TriAxialObject):
                    obj.fields[member] = val
                elif isinstance(obj, dict):
                    obj[member] = val
                else:
                    raise DhrivRuntimeError("Impossibile assegnare un attributo a questo valore.", line)
            elif op == OpCode.TRIAXIAL_OP:
                if ins.axis == AxisCode.VERTICAL:
                    extra_argc = ins.arg_a
                    extra = stack[-extra_argc:] if extra_argc > 0 else []
                    if extra_argc > 0:
                        del stack[-extra_argc:]
                    callee = stack.pop()
                    left = stack.pop()
                    stack.append(self._invoke_vm(callee, [left] + extra, line))
                else:
                    r = stack.pop()
                    l = stack.pop()
                    stack.append(self.interp.apply_triaxial_values(ins.axis.name, l, r, line))
            elif op == OpCode.CALL:
                argc = ins.arg_a
                args = stack[-argc:] if argc > 0 else []
                if argc > 0:
                    del stack[-argc:]
                callee = stack.pop()
                stack.append(self._invoke_vm(callee, args, line))
            elif op == OpCode.RETURN:
                return stack.pop() if ins.flags == 1 else None
            elif op == OpCode.HALT:
                return None
            elif op == OpCode.NOP:
                pass
            else:
                raise DhrivRuntimeError(f"OpCode sconosciuto: {op}", line)
        return None
