"""
Dhriv — linguaggio di programmazione orientato agli oggetti Tri-Assiale
(Verticale, Orizzontale, Diagonale), costruito dal codice binario e dalle porte logiche
fino alla VM a 64 bit, alle onde elettromagnetiche e all'Intelligenza Artificiale nativa.

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

__version__ = "0.2.0-synapse"

from .ai_engine import (
    AdamOptimizer,
    LinearLayer,
    MindMemory,
    SGDOptimizer,
    SelfAttentionLayer,
    Tensor,
    TriAxialNetwork,
    pack_weights_binary,
    unpack_weights_binary,
)
from .binary_core import AxisCode, BinaryALU, InstructionWord, OpCode
from .bytecode_vm import BytecodeCompiler, BytecodeModule, DhrivVM, optimize_ast
from .compiler import (
    AutoCompiler,
    binary_to_bytecode,
    binary_to_program,
    compile_source,
    disassemble_binary,
    disassemble_source,
    verify_binary,
)
from .guardians import WatchDog, WatchGuardian
from .interpreter import DhrivInterpreter, DhrivRuntimeError
from .lexer import DhrivSyntaxError
from .parser import DhrivParser
from .polyglot import detect_language, render_source
from .selflearn import SelfLearningEngine

__all__ = [
    "__version__",
    "AdamOptimizer",
    "AutoCompiler",
    "AxisCode",
    "BinaryALU",
    "BytecodeCompiler",
    "BytecodeModule",
    "DhrivInterpreter",
    "DhrivParser",
    "DhrivRuntimeError",
    "DhrivSyntaxError",
    "DhrivVM",
    "InstructionWord",
    "LinearLayer",
    "MindMemory",
    "OpCode",
    "SGDOptimizer",
    "SelfAttentionLayer",
    "SelfLearningEngine",
    "Tensor",
    "TriAxialNetwork",
    "WatchDog",
    "WatchGuardian",
    "binary_to_bytecode",
    "binary_to_program",
    "compile_source",
    "detect_language",
    "disassemble_binary",
    "disassemble_source",
    "optimize_ast",
    "pack_weights_binary",
    "render_source",
    "unpack_weights_binary",
    "verify_binary",
]
