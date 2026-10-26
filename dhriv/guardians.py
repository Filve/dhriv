"""
guardians.py — WatchGuardian (integrità) e WatchDog (vigilanza dell'esecuzione) di Dhriv.

- `WatchGuardian`: sigilla con SHA3-256 i file del runtime Dhriv, rileva modifiche/aggiunte/
  rimozioni, verifica i binari `.dhrb` e analizza l'AST alla ricerca di operazioni sensibili.
- `WatchDog`: esegue un programma sotto supervisione (timeout, stallo, riavvii controllati),
  lo auto-compila quando il sorgente cambia e può riscriverlo automaticamente nella lingua scelta.

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from .ast_nodes import CallNode, IdentifierNode, ProgramNode, walk
from .compiler import AutoCompiler, DhrivBinaryError, verify_binary
from .interpreter import DhrivCancelled, DhrivInterpreter, DhrivRuntimeError
from .lexer import DhrivSyntaxError
from .parser import DhrivParser
from .polyglot import detect_language, render_source

SENSITIVE_BUILTINS = {
    "legacy_run": "esegue software esterno",
    "em_transmit": "simula una trasmissione elettromagnetica",
    "em_transmit_ai": "simula trasmissione di pesi neurali su onda elettromagnetica",
}


@dataclass
class GuardianReport:
    ok: bool
    sealed: bool
    manifest_root: str
    modified: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)


class WatchGuardian:
    def __init__(self, package_dir: Optional[Path] = None, state_dir: Path = Path(".dhriv_state")) -> None:
        self.package_dir = Path(package_dir) if package_dir else Path(__file__).resolve().parent
        self.state_dir = Path(state_dir)
        self.manifest_file = self.state_dir / "guardian_manifest.json"

    def snapshot(self) -> Dict[str, str]:
        files: Dict[str, str] = {}
        for path in sorted(self.package_dir.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            files[path.relative_to(self.package_dir).as_posix()] = hashlib.sha3_256(path.read_bytes()).hexdigest()
        return files

    @staticmethod
    def manifest_root(files: Dict[str, str]) -> str:
        joined = "".join(f"{k}:{v}\n" for k, v in sorted(files.items()))
        return hashlib.sha3_256(joined.encode("utf-8")).hexdigest()

    def seal(self) -> GuardianReport:
        files = self.snapshot()
        root = self.manifest_root(files)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file.write_text(
            json.dumps({"root": root, "files": files}, indent=2, sort_keys=True), encoding="utf-8"
        )
        return GuardianReport(ok=True, sealed=True, manifest_root=root)

    def verify(self) -> GuardianReport:
        current = self.snapshot()
        root = self.manifest_root(current)
        if not self.manifest_file.exists():
            return GuardianReport(ok=False, sealed=False, manifest_root=root)
        sealed = json.loads(self.manifest_file.read_text(encoding="utf-8")).get("files", {})
        modified = sorted(k for k in current if k in sealed and current[k] != sealed[k])
        missing = sorted(k for k in sealed if k not in current)
        added = sorted(k for k in current if k not in sealed)
        return GuardianReport(
            ok=not (modified or missing or added),
            sealed=True,
            manifest_root=root,
            modified=modified,
            missing=missing,
            added=added,
        )

    @staticmethod
    def verify_binary(blob: bytes) -> bool:
        return verify_binary(blob)

    @staticmethod
    def scan_program(program: ProgramNode) -> List[str]:
        warnings: List[str] = []
        for node in walk(program):
            if isinstance(node, CallNode) and isinstance(node.callee, IdentifierNode):
                reason = SENSITIVE_BUILTINS.get(node.callee.name)
                if reason:
                    warnings.append(f"L{node.line}: '{node.callee.name}' {reason}")
        return warnings


@dataclass
class WatchDogResult:
    ok: bool
    attempts: int
    output: str
    error: Optional[str] = None
    timed_out: bool = False
    elapsed_s: float = 0.0
    language: str = "en"
    events: List[str] = field(default_factory=list)


class WatchDog:
    """Supervisore: esegue in un thread dedicato e interviene su timeout, stallo o crash inattesi."""

    def __init__(
        self,
        timeout_s: float = 30.0,
        stall_s: float = 10.0,
        max_restarts: int = 1,
        allow_legacy: bool = False,
        learner: Any = None,
        guardian: Optional[WatchGuardian] = None,
    ) -> None:
        self.timeout_s = timeout_s
        self.stall_s = stall_s
        self.max_restarts = max_restarts
        self.allow_legacy = allow_legacy
        self.learner = learner
        self.guardian = guardian

    def supervise(self, source_or_program: Union[str, ProgramNode]) -> WatchDogResult:
        events: List[str] = []
        language = "en"
        if isinstance(source_or_program, str):
            language = detect_language(source_or_program)
            try:
                program = DhrivParser.parse_source(source_or_program)
            except DhrivSyntaxError as exc:
                return WatchDogResult(False, 0, "", str(exc), language=language, events=["syntax-error"])
        else:
            program = source_or_program

        if self.guardian is not None:
            for warning in self.guardian.scan_program(program):
                events.append(f"guardian-warning: {warning}")

        started = time.monotonic()
        attempts = 0
        last_error: Optional[str] = None
        timed_out = False
        output = ""

        while attempts <= self.max_restarts:
            attempts += 1
            cancel = threading.Event()
            interp = DhrivInterpreter(
                allow_legacy=self.allow_legacy, learner=self.learner, cancel_event=cancel
            )
            holder: Dict[str, Any] = {}

            def target() -> None:
                try:
                    interp.run_program(program)
                    holder["ok"] = True
                except DhrivCancelled as exc:
                    holder["cancelled"] = str(exc)
                except DhrivRuntimeError as exc:
                    holder["runtime_error"] = str(exc)
                except Exception as exc:  # crash inatteso del runtime
                    holder["crash"] = f"{type(exc).__name__}: {exc}"

            previous_stack = threading.stack_size()
            try:
                threading.stack_size(64 * 1024 * 1024)
            except (ValueError, RuntimeError):
                pass
            worker = threading.Thread(target=target, name="dhriv-watchdog-worker", daemon=True)
            interp.heartbeat = time.monotonic()
            worker.start()
            try:
                threading.stack_size(previous_stack)
            except (ValueError, RuntimeError):
                pass

            attempt_started = time.monotonic()
            reason = ""
            while worker.is_alive():
                worker.join(0.02)
                now = time.monotonic()
                if now - attempt_started > self.timeout_s:
                    reason = "timeout"
                elif now - interp.heartbeat > self.stall_s:
                    reason = "stall"
                if reason:
                    cancel.set()
                    worker.join(2.0)
                    break

            output = "\n".join(interp.output_lines)
            if holder.get("ok"):
                events.append(f"attempt {attempts}: ok")
                return WatchDogResult(
                    True, attempts, output, None, False, time.monotonic() - started, language, events
                )
            if "runtime_error" in holder:
                events.append(f"attempt {attempts}: runtime-error (deterministico, nessun riavvio)")
                return WatchDogResult(
                    False, attempts, output, holder["runtime_error"], False, time.monotonic() - started, language, events
                )
            if reason or "cancelled" in holder:
                timed_out = True
                last_error = f"Esecuzione interrotta dal WatchDog ({reason or 'cancel'})."
                events.append(f"attempt {attempts}: {reason or 'cancelled'}")
            else:
                last_error = holder.get("crash", "errore sconosciuto")
                events.append(f"attempt {attempts}: crash ({last_error})")

        return WatchDogResult(
            False, attempts, output, last_error, timed_out, time.monotonic() - started, language, events
        )

    def render_in_language(self, source: str, target_language: str) -> str:
        """Riscrive automaticamente il sorgente nella lingua scelta (vedi `polyglot.render_source`)."""
        return render_source(source, target_language)

    def watch_events(
        self,
        source_path: Path,
        compiler: Optional[AutoCompiler] = None,
        interval_s: float = 0.5,
        max_cycles: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Controlla un file `.dhr` e lo ricompila automaticamente ad ogni modifica."""
        compiler = compiler or AutoCompiler()
        source_path = Path(source_path)
        last_digest: Optional[str] = None
        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            cycles += 1
            try:
                digest = hashlib.sha3_256(source_path.read_bytes()).hexdigest()
                if digest != last_digest:
                    result = compiler.compile_file(source_path)
                    last_digest = digest
                    yield {
                        "event": "recompiled" if result.rebuilt else "cache-hit",
                        "dhrb": str(result.dhrb_path),
                        "language": result.language,
                        "source_sha3": digest,
                    }
            except (OSError, DhrivSyntaxError, DhrivBinaryError) as exc:
                yield {"event": "error", "error": str(exc)}
            if max_cycles is None or cycles < max_cycles:
                time.sleep(interval_s)
