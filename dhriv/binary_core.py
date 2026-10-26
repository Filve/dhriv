"""
binary_core.py — Fondamenta Binarie, Porte Logiche, ALU Hardware e ISA a 64 bit di Dhriv.

Costruisce metodicamente il linguaggio dal singolo bit `0`/`1`:
1. Porte logiche fisiche tutte derivate dalla porta universale `NAND` (`NOT`, `AND`, `OR`, `NOR`, `XOR`, `XNOR`).
2. Circuiti combinatori (`half_adder`, `full_adder`, `mux2`) e Unità Aritmetico-Logica `BinaryALU`
   a N bit (Ripple-Carry Adder, Complemento a 2, Moltiplicatore Shift-and-Add, Flag Z/C/N/V).
3. Parola-istruzione macchina Tri-Assiale a 64 bit (`InstructionWord` / `OpCode` / `AxisCode`):
   layout `[8b OPCODE | 2b AXIS | 6b FLAGS | 16b ARG_A | 32b ARG_B]`.

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Sequence, Tuple


# ============================================================================
# 1. PORTE LOGICHE UNIVERSALI (TUTTE COSTRUITE A PARTIRE DA NAND)
# ============================================================================

def _bit(x: int) -> int:
    return 1 if int(x) & 1 else 0


def gate_nand(a: int, b: int) -> int:
    """Porta universale NAND: restituisce 0 solo quando entrambi gli ingressi valgono 1."""
    return 1 - (_bit(a) & _bit(b))


def gate_not(a: int) -> int:
    """Inverter NOT derivato da NAND(a, a)."""
    return gate_nand(a, a)


def gate_and(a: int, b: int) -> int:
    """Porta AND derivata da NOT(NAND(a, b))."""
    return gate_not(gate_nand(a, b))


def gate_or(a: int, b: int) -> int:
    """Porta OR per il teorema di De Morgan: NAND(NOT(a), NOT(b))."""
    return gate_nand(gate_not(a), gate_not(b))


def gate_nor(a: int, b: int) -> int:
    """Porta NOR: NOT(OR(a, b))."""
    return gate_not(gate_or(a, b))


def gate_xor(a: int, b: int) -> int:
    """Porta XOR classica a 4 porte NAND."""
    t1 = gate_nand(a, b)
    t2 = gate_nand(a, t1)
    t3 = gate_nand(b, t1)
    return gate_nand(t2, t3)


def gate_xnor(a: int, b: int) -> int:
    """Porta di equivalenza logica XNOR: NOT(XOR(a, b))."""
    return gate_not(gate_xor(a, b))


def mux2(sel: int, d0: int, d1: int) -> int:
    """Multiplexer 2:1 a porte logiche: se sel=0 sceglie d0, se sel=1 sceglie d1."""
    return gate_or(gate_and(gate_not(sel), d0), gate_and(sel, d1))


def half_adder(a: int, b: int) -> Tuple[int, int]:
    """Semi-sommatore binario: restituisce `(sum_bit, carry_bit)`."""
    return gate_xor(a, b), gate_and(a, b)


def full_adder(a: int, b: int, cin: int) -> Tuple[int, int]:
    """Sommatore completo a 1 bit: restituisce `(sum_bit, carry_out)`."""
    s1, c1 = half_adder(a, b)
    s2, c2 = half_adder(s1, cin)
    return s2, gate_or(c1, c2)


LOGIC_GATES = {
    "NAND": gate_nand,
    "NOT": lambda a, _b=0: gate_not(a),
    "AND": gate_and,
    "OR": gate_or,
    "NOR": gate_nor,
    "XOR": gate_xor,
    "XNOR": gate_xnor,
}


def eval_gate(gate_name: str, a: int, b: int = 0) -> int:
    name = str(gate_name).upper()
    fn = LOGIC_GATES.get(name)
    if fn is None:
        raise ValueError(f"Porta logica sconosciuta '{gate_name}'. Valide: {', '.join(LOGIC_GATES)}")
    return fn(a, b)


# ============================================================================
# 2. CONVERSIONI BIT <-> INTERI <-> BYTES E ALU HARDWARE A N BIT
# ============================================================================

def int_to_bits(value: int, width: int = 8) -> List[int]:
    """Converte un intero (anche negativo, in complemento a 2) in `width` bit MSB-first."""
    if width < 1 or width > 256:
        raise ValueError("Larghezza registro deve essere tra 1 e 256 bit.")
    mask = (1 << width) - 1
    masked = int(value) & mask
    return [(masked >> (width - 1 - i)) & 1 for i in range(width)]


def bits_to_int(bits: Sequence[int], signed: bool = False) -> int:
    """Converte una sequenza di bit MSB-first in intero (unsigned o complemento a 2)."""
    if not bits:
        return 0
    val = 0
    for b in bits:
        val = (val << 1) | _bit(b)
    if signed and _bit(bits[0]) == 1:
        val -= 1 << len(bits)
    return val


def bits_to_bitstring(bits: Sequence[int], group: int = 0) -> str:
    raw = "".join("1" if _bit(b) else "0" for b in bits)
    if group > 0 and len(raw) > group:
        chunks = [raw[max(0, i - group):i] for i in range(len(raw), 0, -group)]
        return "_".join(reversed(chunks))
    return raw


def bytes_to_bitstring(data: bytes) -> str:
    return "".join(f"{byte:08b}" for byte in data)


def bitstring_to_bytes(bstr: str) -> bytes:
    clean = bstr.replace("_", "").replace(" ", "")
    if any(c not in ("0", "1") for c in clean):
        raise ValueError("La stringa binaria può contenere solo '0', '1', '_' o spazi.")
    if len(clean) % 8 != 0:
        clean = clean.zfill(((len(clean) + 7) // 8) * 8)
    return bytes(int(clean[i : i + 8], 2) for i in range(0, len(clean), 8))


class BinaryALU:
    """
    Unità Aritmetico-Logica a `width` bit costruita su porte logiche e `full_adder`.
    Calcola anche i 4 flag classici di stato del processore:
    - `Z` (Zero): tutti i bit d'uscita valgono 0;
    - `C` (Carry): riporto uscente dal bit più significativo;
    - `N` (Negative): bit di segno MSB uguale a 1;
    - `V` (Overflow): traboccamento aritmetico in complemento a due (`carry_in_msb ^ carry_out`).
    """

    def __init__(self, width: int = 8) -> None:
        if width < 1 or width > 64:
            raise ValueError("BinaryALU supporta registri da 1 a 64 bit.")
        self.width = int(width)

    def _ripple_add(self, a_bits: List[int], b_bits: List[int], cin: int = 0) -> Tuple[List[int], int, int]:
        out = [0] * self.width
        carry = _bit(cin)
        carry_into_msb = 0
        for i in range(self.width - 1, -1, -1):
            if i == 0:
                carry_into_msb = carry
            s, carry = full_adder(a_bits[i], b_bits[i], carry)
            out[i] = s
        overflow = gate_xor(carry_into_msb, carry)
        return out, carry, overflow

    def execute(self, op: str, a: int, b: int = 0, shift: int = 1) -> Dict[str, Any]:
        op_u = str(op).upper()
        a_bits = int_to_bits(a, self.width)
        b_bits = int_to_bits(b, self.width)
        carry = 0
        overflow = 0

        if op_u == "ADD":
            res_bits, carry, overflow = self._ripple_add(a_bits, b_bits, cin=0)
        elif op_u in ("SUB", "CMP"):
            inv_b = [gate_not(bit) for bit in b_bits]
            res_bits, carry, overflow = self._ripple_add(a_bits, inv_b, cin=1)
        elif op_u == "MUL":
            acc = [0] * self.width
            multiplicand = list(a_bits)
            for i in range(self.width - 1, -1, -1):
                if b_bits[i] == 1:
                    acc, c_step, _ = self._ripple_add(acc, multiplicand, cin=0)
                    carry = gate_or(carry, c_step)
                multiplicand = multiplicand[1:] + [0]
            res_bits = acc
        elif op_u in ("AND", "OR", "XOR", "NAND", "NOR"):
            gfn = LOGIC_GATES[op_u]
            res_bits = [gfn(x, y) for x, y in zip(a_bits, b_bits)]
        elif op_u == "NOT":
            res_bits = [gate_not(x) for x in a_bits]
        elif op_u == "SHL":
            k = max(0, int(b if b else shift))
            if k >= self.width:
                res_bits = [0] * self.width
            elif k == 0:
                res_bits = list(a_bits)
            else:
                carry = a_bits[k - 1]
                res_bits = a_bits[k:] + [0] * k
        elif op_u == "SHR":
            k = max(0, int(b if b else shift))
            if k >= self.width:
                res_bits = [0] * self.width
            elif k == 0:
                res_bits = list(a_bits)
            else:
                carry = a_bits[self.width - k]
                res_bits = [0] * k + a_bits[: self.width - k]
        elif op_u == "ROL":
            k = int(b if b else shift) % self.width
            res_bits = a_bits[k:] + a_bits[:k]
        elif op_u == "ROR":
            k = int(b if b else shift) % self.width
            res_bits = a_bits[self.width - k :] + a_bits[: self.width - k]
        else:
            raise ValueError(f"Operazione ALU non supportata: '{op}'.")

        unsigned_val = bits_to_int(res_bits, signed=False)
        signed_val = bits_to_int(res_bits, signed=True)
        zero_flag = 1
        for bit in res_bits:
            if bit == 1:
                zero_flag = 0
                break
        neg_flag = res_bits[0]

        return {
            "op": op_u,
            "width": self.width,
            "value": unsigned_val,
            "signed": signed_val,
            "bits": res_bits,
            "bitstring": bits_to_bitstring(res_bits),
            "hex": f"0x{unsigned_val:0{(self.width + 3) // 4}X}",
            "flags": {
                "Z": zero_flag,
                "C": carry,
                "N": neg_flag,
                "V": overflow,
            },
        }


# ============================================================================
# 3. ISA TRI-ASSIALE A 64 BIT PER LA MACCHINA VIRTUALE DHRIV
# ============================================================================

class AxisCode(IntEnum):
    """Campo a 2 bit (bit 55..54) che codifica l'asse geometrico dell'istruzione."""
    SELF = 0b00        # Locale / Standard
    VERTICAL = 0b01    # Asse Verticale `|>` / Ereditarietà profonda
    HORIZONTAL = 0b10  # Asse Orizzontale `<->` / Risonanza paritaria
    DIAGONAL = 0b11    # Asse Diagonale `</>` / Ponte trasversale / MatMul


AXIS_NAME_TO_CODE = {
    "SELF": AxisCode.SELF,
    "VERTICAL": AxisCode.VERTICAL,
    "HORIZONTAL": AxisCode.HORIZONTAL,
    "DIAGONAL": AxisCode.DIAGONAL,
}

AXIS_CODE_TO_NAME = {v: k for k, v in AXIS_NAME_TO_CODE.items()}


class OpCode(IntEnum):
    """Set completo di codici operativi a 8 bit (bit 63..56) della DhrivVM."""
    NOP = 0x00
    HALT = 0x01

    # Stack & Costanti
    LOAD_CONST = 0x02      # arg_a = const_idx
    POP = 0x03
    DUP = 0x04

    # Ambiente & Variabili
    DEF_VAR = 0x10         # arg_a = name_const_idx, flags: 1=is_wave
    LOAD_VAR = 0x11        # arg_a = name_const_idx
    STORE_VAR = 0x12       # arg_a = name_const_idx
    PUSH_SCOPE = 0x13
    POP_SCOPE = 0x14

    # Aritmetica, Logica & Bitwise
    BINARY_OP = 0x20       # arg_a = BinarySubOp
    UNARY_OP = 0x21        # arg_a = UnarySubOp

    # Controllo di flusso
    JUMP = 0x30            # arg_b = target_pc
    JUMP_IF_FALSE = 0x31   # arg_b = target_pc (pop cond)
    JUMP_IF_TRUE = 0x32    # arg_b = target_pc (peek/short-circuit)
    JUMP_IF_FALSE_KEEP = 0x33

    # Collezioni & Indici
    BUILD_LIST = 0x40      # arg_a = count
    BUILD_MAP = 0x41       # arg_a = pair_count
    GET_INDEX = 0x42
    SET_INDEX = 0x43

    # Iterazione (cicli `for`)
    ITER_PREP = 0x48       # converte top-of-stack in iteratore interno `[items, idx]`
    ITER_NEXT = 0x49       # arg_a = iter_var_name_idx, arg_b = exit_pc

    # Funzioni & Chiamate
    MAKE_FN = 0x50         # arg_a = fn_const_idx, flags: 1=define_in_env
    CALL = 0x51            # arg_a = argc
    RETURN = 0x52          # flags: 1=has_value

    # OOP Tri-Assiale & Operatori Assiali
    DECL_ENTITY = 0x60     # arg_a = entity_spec_const_idx
    NEW_ENTITY = 0x61      # arg_a = entity_name_const_idx, arg_b = argc
    GET_MEMBER = 0x62      # arg_a = member_name_const_idx, axis = AxisCode
    SET_MEMBER = 0x63      # arg_a = member_name_const_idx
    TRIAXIAL_OP = 0x64     # axis = VERTICAL(pipe) | HORIZONTAL(<->) | DIAGONAL(</>), arg_a = pipe_argc_or_mode


class BinarySubOp(IntEnum):
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    MOD = 5
    POW = 6
    EQ = 7
    NEQ = 8
    LT = 9
    LTE = 10
    GT = 11
    GTE = 12
    BIT_AND = 13
    BIT_OR = 14
    BIT_XOR = 15
    SHL = 16
    SHR = 17


BINARY_OP_TO_SUB = {
    "+": BinarySubOp.ADD,
    "-": BinarySubOp.SUB,
    "*": BinarySubOp.MUL,
    "/": BinarySubOp.DIV,
    "%": BinarySubOp.MOD,
    "**": BinarySubOp.POW,
    "==": BinarySubOp.EQ,
    "!=": BinarySubOp.NEQ,
    "<": BinarySubOp.LT,
    "<=": BinarySubOp.LTE,
    ">": BinarySubOp.GT,
    ">=": BinarySubOp.GTE,
    "&": BinarySubOp.BIT_AND,
    "|": BinarySubOp.BIT_OR,
    "^": BinarySubOp.BIT_XOR,
    "<<": BinarySubOp.SHL,
    ">>": BinarySubOp.SHR,
}

SUB_TO_BINARY_OP = {v: k for k, v in BINARY_OP_TO_SUB.items()}


class UnarySubOp(IntEnum):
    NEG = 1
    NOT = 2
    BIT_NOT = 3


UNARY_OP_TO_SUB = {
    "-": UnarySubOp.NEG,
    "not": UnarySubOp.NOT,
    "~": UnarySubOp.BIT_NOT,
}

SUB_TO_UNARY_OP = {v: k for k, v in UNARY_OP_TO_SUB.items()}


@dataclass(frozen=True)
class InstructionWord:
    """
    Parola-istruzione binaria fissa a 64 bit di Dhriv:
    `[63..56 opcode | 55..54 axis | 53..48 flags | 47..32 arg_a | 31..0 arg_b]`
    """
    opcode: OpCode
    axis: AxisCode = AxisCode.SELF
    flags: int = 0
    arg_a: int = 0
    arg_b: int = 0

    def to_u64(self) -> int:
        op = int(self.opcode) & 0xFF
        ax = int(self.axis) & 0x03
        fl = int(self.flags) & 0x3F
        aa = int(self.arg_a) & 0xFFFF
        ab = int(self.arg_b) & 0xFFFFFFFF
        return (op << 56) | (ax << 54) | (fl << 48) | (aa << 32) | ab

    def pack(self) -> bytes:
        return struct.pack(">Q", self.to_u64())

    @classmethod
    def from_u64(cls, word: int) -> "InstructionWord":
        op = OpCode((word >> 56) & 0xFF)
        ax = AxisCode((word >> 54) & 0x03)
        fl = (word >> 48) & 0x3F
        aa = (word >> 32) & 0xFFFF
        ab_raw = word & 0xFFFFFFFF
        ab = ab_raw - 0x100000000 if (ab_raw & 0x80000000) else ab_raw
        return cls(opcode=op, axis=ax, flags=fl, arg_a=aa, arg_b=ab)

    @classmethod
    def unpack(cls, data: bytes) -> "InstructionWord":
        (word,) = struct.unpack(">Q", data[:8])
        return cls.from_u64(word)

    def format_bits(self) -> str:
        w = self.to_u64()
        op_b = f"{(w >> 56) & 0xFF:08b}"
        ax_b = f"{(w >> 54) & 0x03:02b}"
        fl_b = f"{(w >> 48) & 0x3F:06b}"
        aa_b = f"{(w >> 32) & 0xFFFF:016b}"
        ab_b = f"{w & 0xFFFFFFFF:032b}"
        return f"{op_b}_{ax_b}_{fl_b}_{aa_b}_{ab_b}"

    def format_asm(self, pc: int = 0, hint: str = "", show_bits: bool = False) -> str:
        hex_w = f"0x{self.to_u64():016X}"
        ax_tag = f".{self.axis.name}" if self.axis != AxisCode.SELF else ""
        mnemonic = f"{self.opcode.name}{ax_tag}"
        operands = f"a={self.arg_a}, b={self.arg_b}"
        if self.flags:
            operands += f", fl={self.flags}"
        suffix = f"  ; {hint}" if hint else ""
        if show_bits:
            return f"{pc:04d}  {hex_w}  [{self.format_bits()}]  {mnemonic:<24} {operands}{suffix}"
        return f"{pc:04d}  {hex_w}  {mnemonic:<24} {operands}{suffix}"
