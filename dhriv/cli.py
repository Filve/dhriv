"""
cli.py — Interfaccia a riga di comando di Dhriv (`python3 -m dhriv ...`).

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .binary_core import BinaryALU
from .bytecode_vm import BytecodeCompiler, DhrivVM
from .compiler import (
    AutoCompiler,
    DhrivBinaryError,
    binary_to_bytecode,
    binary_to_program,
    compile_source,
    disassemble_binary,
    disassemble_source,
)
from .guardians import WatchDog, WatchGuardian
from .interpreter import DhrivInterpreter, DhrivRuntimeError, format_value
from .lexer import DhrivSyntaxError
from .packager import TARGETS, build_package
from .parser import DhrivParser
from .polyglot import SUPPORTED_LANGUAGES, detect_language, render_source
from .selflearn import SelfLearningEngine
from .wave_electromagnetic import ElectromagneticBinaryModem

STATE_DIR = Path(".dhriv_state")


def _load_program(path: Path):
    if path.suffix == ".dhrb":
        program, meta = binary_to_program(path.read_bytes())
        return program, meta.get("language", "en"), None
    source = path.read_text(encoding="utf-8")
    return DhrivParser.parse_source(source), detect_language(source), source


def cmd_run(args: argparse.Namespace) -> int:
    path = Path(args.file)
    program, language, _ = _load_program(path)
    learner = None if args.no_learn else SelfLearningEngine(STATE_DIR / "knowledge.json")
    if learner:
        learner.observe_language(language)
    if args.watchdog:
        result = WatchDog(
            timeout_s=args.timeout,
            allow_legacy=args.allow_legacy,
            learner=learner,
            guardian=WatchGuardian(),
        ).supervise(program)
        if result.output:
            print(result.output)
        for event in result.events:
            if event.startswith("guardian-warning"):
                print(f"[{event}]", file=sys.stderr)
        if learner:
            learner.finish_run()
        if not result.ok:
            print(result.error, file=sys.stderr)
            return 1
        return 0

    if args.engine == "vm":
        vm = DhrivVM(echo=True, allow_legacy=args.allow_legacy, learner=learner)
        try:
            if path.suffix == ".dhrb" and learner is None:
                bc_mod, _ = binary_to_bytecode(path.read_bytes())
            else:
                bc_mod = BytecodeCompiler.compile_program(program, optimize=True)
            vm.run_module(bc_mod)
        except DhrivRuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
        finally:
            if learner:
                learner.finish_run()
        if args.report and learner:
            print(json.dumps(learner.report(), indent=2, ensure_ascii=False), file=sys.stderr)
        return 0

    interp = DhrivInterpreter(echo=True, allow_legacy=args.allow_legacy, learner=learner)
    try:
        interp.run_program(program)
    except DhrivRuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        if learner:
            learner.finish_run()
    if args.report and learner:
        print(json.dumps(learner.report(), indent=2, ensure_ascii=False), file=sys.stderr)
    return 0


def cmd_disasm(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if path.suffix == ".dhrb":
        print(disassemble_binary(path.read_bytes(), show_bits=args.bits))
    else:
        print(disassemble_source(path.read_text(encoding="utf-8"), show_bits=args.bits))
    return 0


def cmd_alu(args: argparse.Namespace) -> int:
    res = BinaryALU(width=args.bits).execute(args.op, int(args.a, 0), int(args.b, 0))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    src = Path(args.file)
    if args.output:
        out = Path(args.output)
        out.write_bytes(compile_source(src.read_text(encoding="utf-8"), src.name))
        print(f"Compilato: {out}")
    else:
        result = AutoCompiler(STATE_DIR / "cache").compile_file(src)
        state = "ricompilato" if result.rebuilt else "invariato (cache)"
        print(f"{result.dhrb_path} [{state}, lingua: {result.language}]")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    source = Path(args.file).read_text(encoding="utf-8")
    target = args.to
    if target == "auto":
        target = SelfLearningEngine(STATE_DIR / "knowledge.json").preferred_language()
    sys.stdout.write(render_source(source, target))
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    print(detect_language(Path(args.file).read_text(encoding="utf-8")))
    return 0


def cmd_guard(args: argparse.Namespace) -> int:
    guardian = WatchGuardian(state_dir=STATE_DIR)
    if args.action == "seal":
        report = guardian.seal()
        print(f"Sigillato. Radice manifest: {report.manifest_root}")
        return 0
    report = guardian.verify()
    if not report.sealed:
        print("Nessun sigillo presente: esegui prima 'guard seal'.", file=sys.stderr)
        return 2
    print("INTEGRO" if report.ok else "ALTERATO")
    for label, items in (("modificati", report.modified), ("mancanti", report.missing), ("aggiunti", report.added)):
        for item in items:
            print(f"  {label}: {item}")
    return 0 if report.ok else 1


def cmd_watch(args: argparse.Namespace) -> int:
    dog = WatchDog()
    for event in dog.watch_events(Path(args.file), AutoCompiler(STATE_DIR / "cache"), args.interval, args.cycles or None):
        print(json.dumps(event, ensure_ascii=False))
    return 0


def cmd_package(args: argparse.Namespace) -> int:
    src = Path(args.file)
    dhrb = src if src.suffix == ".dhrb" else AutoCompiler(STATE_DIR / "cache").compile_file(src).dhrb_path
    out = build_package(dhrb, args.name or src.stem, args.target, Path(args.out))
    print(out)
    return 0


def cmd_transmit(args: argparse.Namespace) -> int:
    blob = Path(args.file).read_bytes()
    modem = ElectromagneticBinaryModem()
    signal = modem.modulate_binary(blob, scheme=args.scheme)
    modem.export_wav(signal, Path(args.wav))
    print(f"Trasmesso {len(blob)} byte come onda {args.scheme} -> {args.wav} ({len(signal.samples)} campioni)")
    return 0


def cmd_receive(args: argparse.Namespace) -> int:
    modem = ElectromagneticBinaryModem()
    signal = modem.import_wav(Path(args.wav), modulation_hint=args.scheme)
    payload, info = modem.demodulate_binary(signal, scheme=args.scheme)
    Path(args.output).write_bytes(payload)
    print(f"Ricevuti {len(payload)} byte (bit corretti dal FEC: {info['fec_corrected_bit_errors']}) -> {args.output}")
    return 0


def cmd_repl(_: argparse.Namespace) -> int:
    interp = DhrivInterpreter(echo=True)
    print(f"Dhriv {__version__} — REPL (Ctrl-D per uscire)")
    while True:
        try:
            line = input("dhriv> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line.strip():
            continue
        try:
            value = interp.evaluate_expression(line)
            if value is not None:
                print(format_value(value))
        except (DhrivSyntaxError, DhrivRuntimeError) as exc:
            print(exc)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dhriv", description="Linguaggio Dhriv — OOP Tri-Assiale (V/H/D), Binario & AI")
    p.add_argument("--version", action="version", version=f"dhriv {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("run", help="esegue un file .dhr o .dhrb")
    s.add_argument("file")
    s.add_argument("--engine", choices=["vm", "ast"], default="vm", help="motore di esecuzione (default: vm a 64 bit)")
    s.add_argument("--watchdog", action="store_true", help="esegue sotto supervisione WatchDog")
    s.add_argument("--timeout", type=float, default=30.0)
    s.add_argument("--allow-legacy", action="store_true", help="abilita legacy_run (software esterno)")
    s.add_argument("--no-learn", action="store_true", help="disabilita l'auto-apprendimento")
    s.add_argument("--report", action="store_true", help="stampa il report di apprendimento")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("disasm", help="disassembla un file .dhr o .dhrb in istruzioni ISA a 64 bit")
    s.add_argument("file")
    s.add_argument("--bits", action="store_true", help="mostra anche la scomposizione binaria completa a 64 bit")
    s.set_defaults(func=cmd_disasm)

    s = sub.add_parser("alu", help="esegue un'operazione sull'ALU binaria hardware a porte logiche")
    s.add_argument("op", help="ADD, SUB, MUL, AND, OR, XOR, NAND, NOR, NOT, SHL, SHR, ROL, ROR")
    s.add_argument("a", help="primo operando (es. 42, 0b101010, 0x2A)")
    s.add_argument("b", nargs="?", default="0", help="secondo operando")
    s.add_argument("--bits", type=int, default=8, help="ampiezza registro in bit (default: 8)")
    s.set_defaults(func=cmd_alu)

    s = sub.add_parser("compile", help="compila .dhr in .dhrb (v2 con bytecode + AST firmato)")
    s.add_argument("file")
    s.add_argument("-o", "--output")
    s.set_defaults(func=cmd_compile)

    s = sub.add_parser("render", help="riscrive le parole chiave nella lingua scelta")
    s.add_argument("file")
    s.add_argument("--to", required=True, choices=list(SUPPORTED_LANGUAGES) + ["auto"])
    s.set_defaults(func=cmd_render)

    s = sub.add_parser("detect", help="rileva la lingua delle parole chiave")
    s.add_argument("file")
    s.set_defaults(func=cmd_detect)

    s = sub.add_parser("guard", help="sigilla/verifica l'integrità del runtime")
    s.add_argument("action", choices=["seal", "verify"])
    s.set_defaults(func=cmd_guard)

    s = sub.add_parser("watch", help="ricompila automaticamente quando il sorgente cambia")
    s.add_argument("file")
    s.add_argument("--interval", type=float, default=0.5)
    s.add_argument("--cycles", type=int, default=0, help="0 = infinito")
    s.set_defaults(func=cmd_watch)

    s = sub.add_parser("package", help="crea pacchetto portabile macOS/Windows/Linux")
    s.add_argument("file")
    s.add_argument("--target", choices=list(TARGETS), default="universal")
    s.add_argument("--name")
    s.add_argument("--out", default="dist_packages")
    s.set_defaults(func=cmd_package)

    s = sub.add_parser("transmit", help="modula un file in un'onda (.wav)")
    s.add_argument("file")
    s.add_argument("--wav", required=True)
    s.add_argument("--scheme", choices=["BFSK", "BPSK"], default="BFSK")
    s.set_defaults(func=cmd_transmit)

    s = sub.add_parser("receive", help="demodula un'onda (.wav) in un file")
    s.add_argument("wav")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--scheme", choices=["BFSK", "BPSK"], default="BFSK")
    s.set_defaults(func=cmd_receive)

    s = sub.add_parser("repl", help="console interattiva")
    s.set_defaults(func=cmd_repl)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (DhrivSyntaxError, DhrivRuntimeError, DhrivBinaryError, ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
