"""
compiler.py — Contenitore binario crittografato `.dhrb` (v2 con Bytecode ISA a 64 bit + AST)
e auto-compilazione incrementale di Dhriv.

Formato `.dhrb` v2 (big-endian):

    offset  size  campo
    0       4     magic  b"DHRB"
    4       2     versione formato (2; retrocompatibile in lettura con 1)
    6       2     flag (0x0001 = contiene sezione Bytecode `DHBC` a 64 bit)
    8       4     lunghezza payload compresso
    12      32    SHA3-256 del payload compresso
    44      4     CRC32 del payload compresso
    48      ...   payload = zlib([u32 bc_len | BytecodeModule DHBC | JSON canonico {meta, ast}])

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import ast_nodes
from .ast_nodes import ASTNode, MapLiteralNode, ProgramNode
from .bytecode_vm import BytecodeCompiler, BytecodeModule
from .parser import DhrivParser
from .polyglot import detect_language

MAGIC = b"DHRB"
FORMAT_VERSION = 2
FLAG_HAS_BYTECODE = 0x0001
HEADER_SIZE = 48
COMPILER_ID = "dhriv-0.2.0-synapse"

_NODE_TYPES: Dict[str, type] = {
    name: cls
    for name, cls in vars(ast_nodes).items()
    if isinstance(cls, type) and issubclass(cls, ASTNode) and cls is not ASTNode
}


class DhrivBinaryError(Exception):
    """File `.dhrb` non valido, corrotto o manomesso."""


def _encode(node: Any) -> Any:
    if isinstance(node, ASTNode):
        out: Dict[str, Any] = {"_t": type(node).__name__}
        for f in dataclasses.fields(node):
            out[f.name] = _encode(getattr(node, f.name))
        return out
    if isinstance(node, (list, tuple)):
        return [_encode(x) for x in node]
    return node


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict) and "_t" in obj:
        cls = _NODE_TYPES.get(obj["_t"])
        if cls is None:
            raise DhrivBinaryError(f"Tipo di nodo AST sconosciuto: {obj['_t']!r}")
        kwargs = {k: _decode(v) for k, v in obj.items() if k != "_t"}
        if cls is MapLiteralNode:
            kwargs["entries"] = [tuple(e) for e in kwargs["entries"]]
        return cls(**kwargs)
    if isinstance(obj, list):
        return [_decode(x) for x in obj]
    return obj


def sha3_hex(data: bytes) -> str:
    return hashlib.sha3_256(data).hexdigest()


def program_to_binary(program: ProgramNode, meta: Optional[Dict[str, Any]] = None) -> bytes:
    bc_bytes = BytecodeCompiler.compile_program(program, optimize=True).to_bytes()
    doc = {"meta": dict(meta or {}), "ast": _encode(program)}
    json_raw = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    combined = struct.pack(">I", len(bc_bytes)) + bc_bytes + json_raw
    payload = zlib.compress(combined, 9)
    header = (
        MAGIC
        + struct.pack(">HHI", FORMAT_VERSION, FLAG_HAS_BYTECODE, len(payload))
        + hashlib.sha3_256(payload).digest()
        + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    )
    return header + payload


def verify_binary(blob: bytes) -> bool:
    try:
        _split_container(blob)
        return True
    except DhrivBinaryError:
        return False


def _split_container(blob: bytes) -> Tuple[int, int, bytes]:
    if len(blob) < HEADER_SIZE or blob[:4] != MAGIC:
        raise DhrivBinaryError("Magic 'DHRB' mancante: non è un binario Dhriv.")
    version, flags, length = struct.unpack(">HHI", blob[4:12])
    if version not in (1, 2):
        raise DhrivBinaryError(f"Versione formato non supportata: {version}")
    digest = blob[12:44]
    (crc,) = struct.unpack(">I", blob[44:48])
    payload = blob[HEADER_SIZE : HEADER_SIZE + length]
    if len(payload) != length:
        raise DhrivBinaryError("Payload troncato.")
    if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
        raise DhrivBinaryError("CRC32 non valido: binario corrotto.")
    if hashlib.sha3_256(payload).digest() != digest:
        raise DhrivBinaryError("SHA3-256 non valido: binario manomesso.")
    return version, flags, payload


def binary_to_program(blob: bytes) -> Tuple[ProgramNode, Dict[str, Any]]:
    version, flags, payload = _split_container(blob)
    try:
        raw = zlib.decompress(payload)
        if version == 2 and (flags & FLAG_HAS_BYTECODE):
            if len(raw) < 4:
                raise DhrivBinaryError("Intestazione sezione bytecode mancante.")
            (bc_len,) = struct.unpack(">I", raw[:4])
            json_bytes = raw[4 + bc_len :]
        else:
            json_bytes = raw
        doc = json.loads(json_bytes.decode("utf-8"))
    except (zlib.error, ValueError, UnicodeDecodeError) as exc:
        raise DhrivBinaryError(f"Payload illeggibile: {exc}")
    program = _decode(doc["ast"])
    if not isinstance(program, ProgramNode):
        raise DhrivBinaryError("La radice dell'AST non è un programma.")
    return program, doc.get("meta", {})


def binary_to_bytecode(blob: bytes) -> Tuple[BytecodeModule, Dict[str, Any]]:
    version, flags, payload = _split_container(blob)
    try:
        raw = zlib.decompress(payload)
        if version == 2 and (flags & FLAG_HAS_BYTECODE):
            (bc_len,) = struct.unpack(">I", raw[:4])
            bc_bytes = raw[4 : 4 + bc_len]
            json_bytes = raw[4 + bc_len :]
            doc = json.loads(json_bytes.decode("utf-8"))
            return BytecodeModule.from_bytes(bc_bytes), doc.get("meta", {})
    except (zlib.error, ValueError, UnicodeDecodeError) as exc:
        raise DhrivBinaryError(f"Bytecode binario illeggibile: {exc}")
    program, meta = binary_to_program(blob)
    return BytecodeCompiler.compile_program(program, optimize=True), meta


def disassemble_source(source: str, show_bits: bool = False) -> str:
    return BytecodeCompiler.compile_source(source, optimize=True).disassemble(show_bits=show_bits)


def disassemble_binary(blob: bytes, show_bits: bool = False) -> str:
    mod, _ = binary_to_bytecode(blob)
    return mod.disassemble(show_bits=show_bits)


def compile_source(source: str, source_name: str = "<memory>") -> bytes:
    program = DhrivParser.parse_source(source)
    meta = {
        "compiler": COMPILER_ID,
        "source_name": source_name,
        "source_sha3": sha3_hex(source.encode("utf-8")),
        "language": detect_language(source),
    }
    return program_to_binary(program, meta)


@dataclass
class CompileResult:
    dhrb_path: Path
    rebuilt: bool
    source_sha3: str
    language: str


class AutoCompiler:
    """Compila i sorgenti `.dhr` in `.dhrb` solo quando il contenuto cambia (cache per SHA3)."""

    def __init__(self, cache_dir: Path = Path(".dhriv_state") / "cache") -> None:
        self.cache_dir = Path(cache_dir)

    def compile_file(self, source_path: Path) -> CompileResult:
        source_path = Path(source_path)
        source = source_path.read_text(encoding="utf-8")
        digest = sha3_hex(source.encode("utf-8"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        out = self.cache_dir / (source_path.stem + ".dhrb")

        if out.exists():
            try:
                _, meta = binary_to_program(out.read_bytes())
                if meta.get("source_sha3") == digest and meta.get("compiler") == COMPILER_ID:
                    return CompileResult(out, False, digest, meta.get("language", "en"))
            except DhrivBinaryError:
                pass

        blob = compile_source(source, source_name=source_path.name)
        out.write_bytes(blob)
        _, meta = binary_to_program(blob)
        return CompileResult(out, True, digest, meta.get("language", "en"))
