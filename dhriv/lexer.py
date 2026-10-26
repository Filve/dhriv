"""
lexer.py — Analizzatore Lessicale Poliglotta e Tri-Assiale del linguaggio Dhriv (.dhr).

Progettato per:
1. Riconoscere la sintassi canonica di Dhriv orientata agli oggetti su 3 assi
   (Verticale `|>`, Orizzontale `<->`, Diagonale `</>`) e le primitive di Onde EM.
2. Supportare letterali binari (`0b1010_0110`), esadecimali (`0xFF`), ottali (`0o77`)
   e operatori bitwise nativi (`&`, `|`, `^`, `~`, `<<`, `>>`).
3. Essere nativamente retrocompatibile con ogni lingua umana (Italiano, Inglese,
   Spagnolo, Francese) e con le parole chiave classiche dei linguaggi legacy
   (C/Java/C#/JS/Python), normalizzandole verso i Token canonici di Dhriv.

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class TokenType(str, Enum):
    # Parole chiave strutturali OOP Tri-Assiali
    ENTITY = "ENTITY"             # entity / entita / classe / class
    VERTICAL = "VERTICAL"         # vertical / verticale / extends
    HORIZONTAL = "HORIZONTAL"     # horizontal / orizzontale / with / trait
    DIAGONAL = "DIAGONAL"         # diagonal / diagonale / bridges
    FN = "FN"                     # fn / funzione / def / function
    LET = "LET"                   # let / sia / var
    WAVE = "WAVE"                 # wave / onda
    NEW = "NEW"                   # new / nuovo
    RETURN = "RETURN"             # return / ritorna

    # Controllo di flusso
    IF = "IF"                     # if / se
    ELSE = "ELSE"                 # else / altrimenti
    WHILE = "WHILE"               # while / mentre
    FOR = "FOR"                   # for / per
    IN = "IN"                     # in / dentro

    # Letterali e logici
    TRUE = "TRUE"
    FALSE = "FALSE"
    NULL = "NULL"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

    # Valori
    IDENT = "IDENT"
    NUMBER = "NUMBER"
    STRING = "STRING"

    # Operatori Tri-Assiali esclusivi di Dhriv
    PIPE_VERTICAL = "PIPE_VERTICAL"           # |>
    RESONATE_HORIZONTAL = "RESONATE_HORIZONTAL"  # <->
    BRIDGE_DIAGONAL = "BRIDGE_DIAGONAL"       # </>

    # Operatori aritmetici, logici, bitwise e di confronto
    PLUS = "PLUS"                 # +
    MINUS = "MINUS"               # -
    STAR = "STAR"                 # *
    POW = "POW"                   # **
    SLASH = "SLASH"               # /
    PERCENT = "PERCENT"           # %
    BIT_AND = "BIT_AND"           # &
    BIT_OR = "BIT_OR"             # |
    BIT_XOR = "BIT_XOR"           # ^
    BIT_NOT = "BIT_NOT"           # ~
    SHL = "SHL"                   # <<
    SHR = "SHR"                   # >>
    ASSIGN = "ASSIGN"             # =
    FAT_ARROW = "FAT_ARROW"       # =>
    EQ = "EQ"                     # ==
    NEQ = "NEQ"                   # !=
    LT = "LT"                     # <
    LTE = "LTE"                   # <=
    GT = "GT"                     # >
    GTE = "GTE"                   # >=

    # Delimitatori
    LPAREN = "LPAREN"             # (
    RPAREN = "RPAREN"             # )
    LBRACE = "LBRACE"             # {
    RBRACE = "RBRACE"             # }
    LBRACKET = "LBRACKET"         # [
    RBRACKET = "RBRACKET"         # ]
    COMMA = "COMMA"               # ,
    DOT = "DOT"                   # .
    COLON = "COLON"               # :
    SEMICOLON = "SEMICOLON"       # ;

    EOF = "EOF"


# Dizionario Poliglotta Universale -> TokenType canonico di Dhriv
POLYGLOT_KEYWORDS: Dict[str, TokenType] = {
    # Inglese / Canonico Dhriv + Legacy OOP
    "entity": TokenType.ENTITY,
    "class": TokenType.ENTITY,
    "struct": TokenType.ENTITY,
    "vertical": TokenType.VERTICAL,
    "extends": TokenType.VERTICAL,
    "horizontal": TokenType.HORIZONTAL,
    "with": TokenType.HORIZONTAL,
    "trait": TokenType.HORIZONTAL,
    "diagonal": TokenType.DIAGONAL,
    "bridges": TokenType.DIAGONAL,
    "transversal": TokenType.DIAGONAL,
    "fn": TokenType.FN,
    "def": TokenType.FN,
    "function": TokenType.FN,
    "func": TokenType.FN,
    "let": TokenType.LET,
    "var": TokenType.LET,
    "const": TokenType.LET,
    "wave": TokenType.WAVE,
    "new": TokenType.NEW,
    "return": TokenType.RETURN,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "for": TokenType.FOR,
    "in": TokenType.IN,
    "true": TokenType.TRUE,
    "True": TokenType.TRUE,
    "false": TokenType.FALSE,
    "False": TokenType.FALSE,
    "null": TokenType.NULL,
    "None": TokenType.NULL,
    "nil": TokenType.NULL,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,

    # Italiano
    "entita": TokenType.ENTITY,
    "entità": TokenType.ENTITY,
    "classe": TokenType.ENTITY,
    "verticale": TokenType.VERTICAL,
    "estende": TokenType.VERTICAL,
    "orizzontale": TokenType.HORIZONTAL,
    "con": TokenType.HORIZONTAL,
    "diagonale": TokenType.DIAGONAL,
    "attraversa": TokenType.DIAGONAL,
    "funzione": TokenType.FN,
    "sia": TokenType.LET,
    "onda": TokenType.WAVE,
    "nuovo": TokenType.NEW,
    "nuova": TokenType.NEW,
    "ritorna": TokenType.RETURN,
    "restituisci": TokenType.RETURN,
    "se": TokenType.IF,
    "altrimenti": TokenType.ELSE,
    "mentre": TokenType.WHILE,
    "per": TokenType.FOR,
    "dentro": TokenType.IN,
    "vero": TokenType.TRUE,
    "falso": TokenType.FALSE,
    "nullo": TokenType.NULL,
    "non": TokenType.NOT,

    # Spagnolo
    "entidad": TokenType.ENTITY,
    "clase": TokenType.ENTITY,
    "funcion": TokenType.FN,
    "función": TokenType.FN,
    "sea": TokenType.LET,
    "nuevo": TokenType.NEW,
    "retornar": TokenType.RETURN,
    "sino": TokenType.ELSE,
    "mientras": TokenType.WHILE,
    "para": TokenType.FOR,
    "verdadero": TokenType.TRUE,
    "nulo": TokenType.NULL,

    # Francese
    "entite": TokenType.ENTITY,
    "entité": TokenType.ENTITY,
    "verticale_fr": TokenType.VERTICAL,
    "horizontale": TokenType.HORIZONTAL,
    "fonction": TokenType.FN,
    "soit": TokenType.LET,
    "onde": TokenType.WAVE,
    "nouveau": TokenType.NEW,
    "retourner": TokenType.RETURN,
    "si": TokenType.IF,
    "sinon": TokenType.ELSE,
    "tantque": TokenType.WHILE,
    "pour": TokenType.FOR,
    "dans": TokenType.IN,
    "vrai": TokenType.TRUE,
    "faux": TokenType.FALSE,
    "nul": TokenType.NULL,
}

# Alias poliglotta per il riferimento all'istanza corrente (`self`)
SELF_ALIASES = {"self", "this", "questo", "se_stesso", "este", "soi"}


@dataclass(frozen=True)
class Token:
    type: TokenType
    lexeme: str
    literal: object
    line: int
    column: int


class DhrivSyntaxError(Exception):
    """Errore lessicale o sintattico con coordinate precise nel sorgente .dhr."""

    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(f"[Dhriv SyntaxError L{line}:C{column}] {message}")
        self.message = message
        self.line = line
        self.column = column


class DhrivLexer:
    """Trasforma codice sorgente .dhr (in qualsiasi lingua supportata) in uno stream di Token."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.length = len(source)
        self.pos = 0
        self.line = 1
        self.col = 1

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        if idx >= self.length:
            return "\0"
        return self.source[idx]

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []

        while self.pos < self.length:
            ch = self._peek()

            # Spazi bianchi
            if ch in (" ", "\t", "\r", "\n"):
                self._advance()
                continue

            # Commenti su singola riga `//` oppure `#` (retrocompatibilità Python/Shell)
            if ch == "#" or (ch == "/" and self._peek(1) == "/"):
                while self.pos < self.length and self._peek() != "\n":
                    self._advance()
                continue

            # Commenti multi-riga `/* ... */`
            if ch == "/" and self._peek(1) == "*":
                start_line, start_col = self.line, self.col
                self._advance()
                self._advance()
                closed = False
                while self.pos < self.length:
                    if self._peek() == "*" and self._peek(1) == "/":
                        self._advance()
                        self._advance()
                        closed = True
                        break
                    self._advance()
                if not closed:
                    raise DhrivSyntaxError("Commento multi-riga '/*' non chiuso.", start_line, start_col)
                continue

            start_line, start_col = self.line, self.col

            # Operatori Tri-Assiali a 3 caratteri: `<->` (Orizzontale) e `</>` (Diagonale)
            if ch == "<" and self._peek(1) == "-" and self._peek(2) == ">":
                self._advance()
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.RESONATE_HORIZONTAL, "<->", None, start_line, start_col))
                continue

            if ch == "<" and self._peek(1) == "/" and self._peek(2) == ">":
                self._advance()
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.BRIDGE_DIAGONAL, "</>", None, start_line, start_col))
                continue

            # Operatore Verticale `|>`, OR logico `||` o OR bitwise `|`
            if ch == "|":
                self._advance()
                if self._peek() == ">":
                    self._advance()
                    tokens.append(Token(TokenType.PIPE_VERTICAL, "|>", None, start_line, start_col))
                elif self._peek() == "|":
                    self._advance()
                    tokens.append(Token(TokenType.OR, "||", None, start_line, start_col))
                else:
                    tokens.append(Token(TokenType.BIT_OR, "|", None, start_line, start_col))
                continue

            if ch == "&":
                self._advance()
                if self._peek() == "&":
                    self._advance()
                    tokens.append(Token(TokenType.AND, "&&", None, start_line, start_col))
                else:
                    tokens.append(Token(TokenType.BIT_AND, "&", None, start_line, start_col))
                continue

            # Stringhe `"..."` o `'...'`
            if ch in ('"', "'"):
                tokens.append(self._read_string(ch, start_line, start_col))
                continue

            # Numeri interi (decimali, binari 0b, esadecimali 0x, ottali 0o) e float
            if ch.isdigit():
                tokens.append(self._read_number(start_line, start_col))
                continue

            # Identificatori e Parole Chiave Poliglotta (supporto Unicode completo)
            if ch.isalpha() or ch == "_":
                tokens.append(self._read_identifier(start_line, start_col))
                continue

            # Operatori a 1 o 2 caratteri
            self._advance()
            if ch == "*":
                if self._peek() == "*":
                    self._advance()
                    tokens.append(Token(TokenType.POW, "**", None, start_line, start_col))
                else:
                    tokens.append(Token(TokenType.STAR, "*", None, start_line, start_col))
            elif ch == "=":
                if self._peek() == "=":
                    self._advance()
                    tokens.append(Token(TokenType.EQ, "==", None, start_line, start_col))
                elif self._peek() == ">":
                    self._advance()
                    tokens.append(Token(TokenType.FAT_ARROW, "=>", None, start_line, start_col))
                else:
                    tokens.append(Token(TokenType.ASSIGN, "=", None, start_line, start_col))
            elif ch == "!":
                if self._peek() == "=":
                    self._advance()
                    tokens.append(Token(TokenType.NEQ, "!=", None, start_line, start_col))
                else:
                    tokens.append(Token(TokenType.NOT, "!", None, start_line, start_col))
            elif ch == "<":
                if self._peek() == "=":
                    self._advance()
                    tokens.append(Token(TokenType.LTE, "<=", None, start_line, start_col))
                elif self._peek() == "<":
                    self._advance()
                    tokens.append(Token(TokenType.SHL, "<<", None, start_line, start_col))
                else:
                    tokens.append(Token(TokenType.LT, "<", None, start_line, start_col))
            elif ch == ">":
                if self._peek() == "=":
                    self._advance()
                    tokens.append(Token(TokenType.GTE, ">=", None, start_line, start_col))
                elif self._peek() == ">":
                    self._advance()
                    tokens.append(Token(TokenType.SHR, ">>", None, start_line, start_col))
                else:
                    tokens.append(Token(TokenType.GT, ">", None, start_line, start_col))
            elif ch == "^":
                tokens.append(Token(TokenType.BIT_XOR, "^", None, start_line, start_col))
            elif ch == "~":
                tokens.append(Token(TokenType.BIT_NOT, "~", None, start_line, start_col))
            elif ch == "+":
                tokens.append(Token(TokenType.PLUS, "+", None, start_line, start_col))
            elif ch == "-":
                tokens.append(Token(TokenType.MINUS, "-", None, start_line, start_col))
            elif ch == "/":
                tokens.append(Token(TokenType.SLASH, "/", None, start_line, start_col))
            elif ch == "%":
                tokens.append(Token(TokenType.PERCENT, "%", None, start_line, start_col))
            elif ch == "(":
                tokens.append(Token(TokenType.LPAREN, "(", None, start_line, start_col))
            elif ch == ")":
                tokens.append(Token(TokenType.RPAREN, ")", None, start_line, start_col))
            elif ch == "{":
                tokens.append(Token(TokenType.LBRACE, "{", None, start_line, start_col))
            elif ch == "}":
                tokens.append(Token(TokenType.RBRACE, "}", None, start_line, start_col))
            elif ch == "[":
                tokens.append(Token(TokenType.LBRACKET, "[", None, start_line, start_col))
            elif ch == "]":
                tokens.append(Token(TokenType.RBRACKET, "]", None, start_line, start_col))
            elif ch == ",":
                tokens.append(Token(TokenType.COMMA, ",", None, start_line, start_col))
            elif ch == ".":
                tokens.append(Token(TokenType.DOT, ".", None, start_line, start_col))
            elif ch == ":":
                tokens.append(Token(TokenType.COLON, ":", None, start_line, start_col))
            elif ch == ";":
                tokens.append(Token(TokenType.SEMICOLON, ";", None, start_line, start_col))
            else:
                raise DhrivSyntaxError(f"Simbolo non riconosciuto: {ch!r}", start_line, start_col)

        tokens.append(Token(TokenType.EOF, "", None, self.line, self.col))
        return tokens

    def _read_string(self, quote: str, start_line: int, start_col: int) -> Token:
        self._advance()  # salta quote iniziale
        chars: List[str] = []
        escapes = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "'": "'", "\\": "\\"}

        while self.pos < self.length:
            ch = self._peek()
            if ch == quote:
                self._advance()
                val = "".join(chars)
                return Token(TokenType.STRING, f"{quote}{val}{quote}", val, start_line, start_col)
            if ch == "\\":
                self._advance()
                esc = self._advance() if self.pos < self.length else ""
                chars.append(escapes.get(esc, esc))
            else:
                chars.append(self._advance())

        raise DhrivSyntaxError("Stringa non terminata.", start_line, start_col)

    def _read_number(self, start_line: int, start_col: int) -> Token:
        # Supporto nativo basi binaria (0b), esadecimale (0x), ottale (0o)
        if self._peek() == "0" and self._peek(1) in ("b", "B", "x", "X", "o", "O"):
            prefix_zero = self._advance()
            radix_ch = self._advance()
            radix_map = {
                "b": (2, "01"),
                "o": (8, "01234567"),
                "x": (16, "0123456789abcdefABCDEF"),
            }
            base, allowed = radix_map[radix_ch.lower()]
            digits: List[str] = []
            raw_lex = [prefix_zero, radix_ch]
            while self.pos < self.length and (self._peek() in allowed or self._peek() == "_"):
                c = self._advance()
                raw_lex.append(c)
                if c != "_":
                    digits.append(c)
            if not digits:
                raise DhrivSyntaxError(
                    f"Letterale numerico in base {base} incompleto dopo '0{radix_ch}'.",
                    start_line,
                    start_col,
                )
            return Token(TokenType.NUMBER, "".join(raw_lex), int("".join(digits), base), start_line, start_col)

        buf: List[str] = []
        is_float = False
        while self.pos < self.length and (self._peek().isdigit() or self._peek() == "_"):
            c = self._advance()
            if c != "_":
                buf.append(c)

        if self._peek() == "." and self._peek(1).isdigit():
            is_float = True
            buf.append(self._advance())
            while self.pos < self.length and (self._peek().isdigit() or self._peek() == "_"):
                c = self._advance()
                if c != "_":
                    buf.append(c)

        # Supporto notazione scientifica per frequenze/onde EM (es. `2.4e9`, `1e-6`)
        if self._peek() in ("e", "E"):
            nxt = self._peek(1)
            nxt2 = self._peek(2)
            if nxt.isdigit() or (nxt in ("+", "-") and nxt2.isdigit()):
                is_float = True
                buf.append(self._advance())
                if self._peek() in ("+", "-"):
                    buf.append(self._advance())
                while self.pos < self.length and self._peek().isdigit():
                    buf.append(self._advance())

        raw = "".join(buf)
        literal: Optional[object] = float(raw) if is_float else int(raw)
        return Token(TokenType.NUMBER, raw, literal, start_line, start_col)

    def _read_identifier(self, start_line: int, start_col: int) -> Token:
        buf: List[str] = []
        while self.pos < self.length and (self._peek().isalnum() or self._peek() == "_"):
            buf.append(self._advance())
        ident = "".join(buf)

        if ident in SELF_ALIASES:
            return Token(TokenType.IDENT, "self", "self", start_line, start_col)

        kw_type = POLYGLOT_KEYWORDS.get(ident)
        if kw_type is not None:
            if kw_type == TokenType.TRUE:
                return Token(kw_type, ident, True, start_line, start_col)
            if kw_type == TokenType.FALSE:
                return Token(kw_type, ident, False, start_line, start_col)
            return Token(kw_type, ident, None, start_line, start_col)

        return Token(TokenType.IDENT, ident, ident, start_line, start_col)
