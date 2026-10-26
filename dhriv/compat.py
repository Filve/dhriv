"""
compat.py — Ponte di retrocompatibilità con il software esistente.

Dhriv non riscrive i programmi legacy: li invoca. `LegacyBridge` esegue un comando
esterno (binari nativi, script Python/Node/shell, tool di qualsiasi linguaggio) senza shell,
con timeout e output catturato. La sintassi di Dhriv accetta inoltre le parole chiave classiche
di C/Java/JS/Python (`class`, `extends`, `function`, `def`, `var`, `//`, `#`) tramite il lexer.

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from typing import Any, Dict


class LegacyBridge:
    def run(self, command: str, timeout: float = 10.0) -> Dict[str, Any]:
        argv = shlex.split(command, posix=(os.name != "nt"))
        if not argv:
            raise ValueError("Comando vuoto.")
        if os.name == "nt":
            argv = [
                a[1:-1] if len(a) >= 2 and a[0] == a[-1] and a[0] in ('"', "'") else a
                for a in argv
            ]
        resolved = shutil.which(argv[0])
        if resolved is not None:
            argv[0] = resolved
        elif os.name == "nt" and argv[0].lower() == "echo":
            # Su Windows `echo` è un builtin di cmd.exe: lo emuliamo in modo sicuro senza shell.
            return {
                "code": 0,
                "stdout": " ".join(argv[1:]) + "\n",
                "stderr": "",
                "timed_out": False,
            }
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, shell=False)
        except subprocess.TimeoutExpired:
            return {"code": -1, "stdout": "", "stderr": "timeout", "timed_out": True}
        return {
            "code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
        }
