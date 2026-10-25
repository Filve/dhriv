"""
compiler.py — Contenitore binario `.dhrb` e auto-compilazione incrementale di Dhriv.

Formato `.dhrb` (big-endian):

    offset  size  campo
    0       4     magic  b"DHRB"
    4       2     versione formato (1)
    6       2     flag riservati (0)
    8       4     lunghezza payload compresso
    12      32    SHA3-256 del payload compresso
    44      4     CRC32 del payload compresso
    48      ...   payload = zlib(JSON canonico {"meta": {...}, "ast": {...}})

Il payload contiene l'AST serializzato (NON codice eseguibile arbitrario): il caricamento
ricostruisce solo classi AST registrate, verifica SHA3 e CRC e rifiuta file manomessi.

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
from .parser import DhrivParser
from .polyglot import detect_language

MAGIC = b"DHRB"
FORMAT_VERSION = 1
HEADER_SIZE = 48
COMPILER_ID = "dhriv-0.1.0-genesis"

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
    doc = {"meta": dict(meta or {}), "ast": _encode(program)}
    raw = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload = zlib.compress(raw, 9)
    header = (
        MAGIC
        + struct.pack(">HHI", FORMAT_VERSION, 0, len(payload))
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


def _split_container(blob: bytes) -> bytes:
    if len(blob) < HEADER_SIZE or blob[:4] != MAGIC:
        raise DhrivBinaryError("Magic 'DHRB' mancante: non è un binario Dhriv.")
    version, _flags, length = struct.unpack(">HHI", blob[4:12])
    if version != FORMAT_VERSION:
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
    return payload


def binary_to_program(blob: bytes) -> Tuple[ProgramNode, Dict[str, Any]]:
    payload = _split_container(blob)
    try:
        doc = json.loads(zlib.decompress(payload).decode("utf-8"))
    except (zlib.error, ValueError) as exc:
        raise DhrivBinaryError(f"Payload illeggibile: {exc}")
    program = _decode(doc["ast"])
    if not isinstance(program, ProgramNode):
        raise DhrivBinaryError("La radice dell'AST non è un programma.")
    return program, doc.get("meta", {})


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
