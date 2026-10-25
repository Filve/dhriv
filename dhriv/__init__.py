"""
Dhriv — linguaggio di programmazione orientato agli oggetti Tri-Assiale
(Verticale, Orizzontale, Diagonale) con onde, auto-compilazione e auto-apprendimento.

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

__version__ = "0.1.0-genesis"

from .compiler import AutoCompiler, binary_to_program, compile_source, verify_binary
from .guardians import WatchDog, WatchGuardian
from .interpreter import DhrivInterpreter, DhrivRuntimeError
from .lexer import DhrivSyntaxError
from .parser import DhrivParser
from .polyglot import detect_language, render_source
from .selflearn import SelfLearningEngine

__all__ = [
    "__version__",
    "AutoCompiler",
    "DhrivInterpreter",
    "DhrivParser",
    "DhrivRuntimeError",
    "DhrivSyntaxError",
    "SelfLearningEngine",
    "WatchDog",
    "WatchGuardian",
    "binary_to_program",
    "compile_source",
    "detect_language",
    "render_source",
    "verify_binary",
]
