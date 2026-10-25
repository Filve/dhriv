"""
polyglot.py — Rilevamento e rendering automatico del sorgente Dhriv nella lingua scelta.

Il lexer di Dhriv accetta parole chiave in italiano, inglese, spagnolo e francese (oltre
ai termini legacy C/Java/Python/JS). Questo modulo:
- `detect_language(source)`: stima la lingua prevalente delle parole chiave;
- `render_source(source, target)`: riscrive SOLO le parole chiave nella lingua scelta,
  preservando commenti, stringhe, spaziatura e identificatori (il significato non cambia).

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .lexer import DhrivLexer, TokenType

CANONICAL_KEYWORDS: Dict[str, Dict[TokenType, str]] = {
    "en": {
        TokenType.ENTITY: "entity", TokenType.VERTICAL: "vertical", TokenType.HORIZONTAL: "horizontal",
        TokenType.DIAGONAL: "diagonal", TokenType.FN: "fn", TokenType.LET: "let", TokenType.WAVE: "wave",
        TokenType.NEW: "new", TokenType.RETURN: "return", TokenType.IF: "if", TokenType.ELSE: "else",
        TokenType.WHILE: "while", TokenType.FOR: "for", TokenType.IN: "in", TokenType.TRUE: "true",
        TokenType.FALSE: "false", TokenType.NULL: "null", TokenType.AND: "and", TokenType.OR: "or",
        TokenType.NOT: "not",
    },
    "it": {
        TokenType.ENTITY: "entita", TokenType.VERTICAL: "verticale", TokenType.HORIZONTAL: "orizzontale",
        TokenType.DIAGONAL: "diagonale", TokenType.FN: "funzione", TokenType.LET: "sia", TokenType.WAVE: "onda",
        TokenType.NEW: "nuovo", TokenType.RETURN: "ritorna", TokenType.IF: "se", TokenType.ELSE: "altrimenti",
        TokenType.WHILE: "mentre", TokenType.FOR: "per", TokenType.IN: "dentro", TokenType.TRUE: "vero",
        TokenType.FALSE: "falso", TokenType.NULL: "nullo", TokenType.AND: "and", TokenType.OR: "or",
        TokenType.NOT: "non",
    },
    "es": {
        TokenType.ENTITY: "entidad", TokenType.VERTICAL: "vertical", TokenType.HORIZONTAL: "horizontal",
        TokenType.DIAGONAL: "diagonal", TokenType.FN: "funcion", TokenType.LET: "sea", TokenType.WAVE: "onda",
        TokenType.NEW: "nuevo", TokenType.RETURN: "retornar", TokenType.IF: "si", TokenType.ELSE: "sino",
        TokenType.WHILE: "mientras", TokenType.FOR: "para", TokenType.IN: "dentro", TokenType.TRUE: "verdadero",
        TokenType.FALSE: "falso", TokenType.NULL: "nulo", TokenType.AND: "and", TokenType.OR: "or",
        TokenType.NOT: "non",
    },
    "fr": {
        TokenType.ENTITY: "entite", TokenType.VERTICAL: "verticale", TokenType.HORIZONTAL: "horizontale",
        TokenType.DIAGONAL: "diagonale", TokenType.FN: "fonction", TokenType.LET: "soit", TokenType.WAVE: "onde",
        TokenType.NEW: "nouveau", TokenType.RETURN: "retourner", TokenType.IF: "si", TokenType.ELSE: "sinon",
        TokenType.WHILE: "tantque", TokenType.FOR: "pour", TokenType.IN: "dans", TokenType.TRUE: "vrai",
        TokenType.FALSE: "faux", TokenType.NULL: "nul", TokenType.AND: "and", TokenType.OR: "or",
        TokenType.NOT: "non",
    },
}

SUPPORTED_LANGUAGES = tuple(CANONICAL_KEYWORDS.keys())

_WORD_TYPES = set(CANONICAL_KEYWORDS["en"].keys())


def _keyword_tokens(source: str):
    for tok in DhrivLexer(source).tokenize():
        if tok.type in _WORD_TYPES and tok.lexeme and tok.lexeme[0].isalpha():
            yield tok


def detect_language(source: str) -> str:
    """Restituisce il codice lingua (`en`, `it`, `es`, `fr`) più coerente con le parole chiave usate."""
    scores: Dict[str, int] = {lang: 0 for lang in SUPPORTED_LANGUAGES}
    for tok in _keyword_tokens(source):
        for lang, table in CANONICAL_KEYWORDS.items():
            if table.get(tok.type) == tok.lexeme:
                scores[lang] += 1
    best = max(SUPPORTED_LANGUAGES, key=lambda lang: (scores[lang], lang == "en"))
    return best if scores[best] > 0 else "en"


def _offsets(source: str) -> List[int]:
    starts = [0]
    for idx, ch in enumerate(source):
        if ch == "\n":
            starts.append(idx + 1)
    return starts


def render_source(source: str, target_language: str) -> str:
    """Riscrive le parole chiave del sorgente nella lingua `target_language`."""
    if target_language not in CANONICAL_KEYWORDS:
        raise ValueError(
            f"Lingua '{target_language}' non supportata. Disponibili: {', '.join(SUPPORTED_LANGUAGES)}"
        )
    table = CANONICAL_KEYWORDS[target_language]
    starts = _offsets(source)
    edits: List[Tuple[int, int, str]] = []
    for tok in _keyword_tokens(source):
        start = starts[tok.line - 1] + (tok.column - 1)
        edits.append((start, start + len(tok.lexeme), table[tok.type]))
    out = source
    for start, end, repl in sorted(edits, reverse=True):
        out = out[:start] + repl + out[end:]
    return out
