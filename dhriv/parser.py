"""
parser.py — Parser a Discesa Ricorsiva completo per il linguaggio Dhriv (.dhr).

Converte lo stream di Token in un `ProgramNode` tipizzato, supportando:
- Entità OOP con clausole `vertical`, `horizontal` e `diagonal` in qualunque ordine;
- Dichiarazioni `wave` (onde di frequenza / elettromagnetiche), `let`, `fn` e lambda `fn(...) => ...`;
- Espressioni algebriche, logiche, bitwise (`&`, `|`, `^`, `~`, `<<`, `>>`) e
  operatori Tri-Assiali (`|>`, `<->`, `</>`);
- Blocchi `{ ... }`, cicli `while`/`for`, condizioni `if`/`else`, liste e dizionari.

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .ast_nodes import (
    ASTNode,
    AssignStmtNode,
    BinaryOpNode,
    CallNode,
    EntityDeclNode,
    ExprStmtNode,
    FieldDeclNode,
    ForStmtNode,
    FunctionDeclNode,
    IdentifierNode,
    IfStmtNode,
    IndexAccessNode,
    LambdaNode,
    LetStmtNode,
    ListLiteralNode,
    LiteralNode,
    MapLiteralNode,
    MemberAccessNode,
    NewInstanceNode,
    ProgramNode,
    ReturnStmtNode,
    TriAxialOpNode,
    UnaryOpNode,
    WhileStmtNode,
)
from .lexer import DhrivLexer, DhrivSyntaxError, Token, TokenType


class DhrivParser:
    """Costruisce l'AST formale di Dhriv da una lista di Token o direttamente dal testo sorgente."""

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.current = 0

    @classmethod
    def parse_source(cls, source: str) -> ProgramNode:
        tokens = DhrivLexer(source).tokenize()
        return cls(tokens).parse()

    def _peek(self, offset: int = 0) -> Token:
        idx = self.current + offset
        if idx >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[idx]

    def _is_at_end(self) -> bool:
        return self._peek().type == TokenType.EOF

    def _advance(self) -> Token:
        if not self._is_at_end():
            self.current += 1
        return self.tokens[self.current - 1]

    def _check(self, *types: TokenType) -> bool:
        if self._is_at_end():
            return False
        return self._peek().type in types

    def _match(self, *types: TokenType) -> bool:
        if self._check(*types):
            self._advance()
            return True
        return False

    def _consume(self, ttype: TokenType, message: str) -> Token:
        if self._check(ttype):
            return self._advance()
        tok = self._peek()
        raise DhrivSyntaxError(f"{message} (trovato {tok.lexeme!r})", tok.line, tok.column)

    def _skip_semicolons(self) -> None:
        while self._match(TokenType.SEMICOLON):
            pass

    def parse(self) -> ProgramNode:
        statements: List[ASTNode] = []
        self._skip_semicolons()
        while not self._is_at_end():
            statements.append(self._parse_declaration())
            self._skip_semicolons()
        return ProgramNode(statements=statements)

    def _parse_declaration(self) -> ASTNode:
        if self._match(TokenType.ENTITY):
            return self._parse_entity_decl()
        if self._check(TokenType.FN) and self._peek(1).type == TokenType.IDENT:
            self._advance()
            return self._parse_function_decl(is_method=False)
        if self._match(TokenType.LET):
            return self._parse_let_stmt(is_wave=False)
        if self._match(TokenType.WAVE):
            return self._parse_let_stmt(is_wave=True)
        return self._parse_statement()

    def _parse_entity_decl(self) -> EntityDeclNode:
        name_tok = self._consume(TokenType.IDENT, "Atteso nome dell'entità OOP")
        vertical_parents: List[str] = []
        horizontal_peers: List[str] = []
        diagonal_bridges: List[str] = []

        # Clausole Tri-Assiali opzionali e combinabili liberamente:
        # es. `entity HyperNode vertical BaseServer horizontal MeshPeer, Telemetry diagonal KernelCore { ... }`
        while self._check(TokenType.VERTICAL, TokenType.HORIZONTAL, TokenType.DIAGONAL):
            axis_tok = self._advance()
            target_list = (
                vertical_parents
                if axis_tok.type == TokenType.VERTICAL
                else horizontal_peers
                if axis_tok.type == TokenType.HORIZONTAL
                else diagonal_bridges
            )
            # Una clausola `horizontal` senza genitori seguita subito da `{` definisce un trait orizzontale puro
            if self._check(TokenType.LBRACE):
                break
            target_list.append(self._consume(TokenType.IDENT, "Atteso nome entità nella clausola assiale").lexeme)
            while self._match(TokenType.COMMA):
                target_list.append(self._consume(TokenType.IDENT, "Atteso nome entità dopo ','").lexeme)

        self._consume(TokenType.LBRACE, "Atteso '{' per aprire il corpo dell'entità")
        fields: List[FieldDeclNode] = []
        methods: List[FunctionDeclNode] = []

        self._skip_semicolons()
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            if self._match(TokenType.LET, TokenType.WAVE):
                f_name = self._consume(TokenType.IDENT, "Atteso nome del campo").lexeme
                if self._match(TokenType.ASSIGN):
                    f_val = self._parse_expression()
                else:
                    f_val = LiteralNode(None, line=name_tok.line)
                fields.append(FieldDeclNode(name=f_name, value=f_val, line=name_tok.line))
            elif self._match(TokenType.FN):
                methods.append(self._parse_function_decl(is_method=True))
            else:
                # Supporto campi abbreviati `power = 100`
                f_tok = self._consume(TokenType.IDENT, "Atteso 'let', 'fn' o nome campo nell'entità")
                self._consume(TokenType.ASSIGN, "Atteso '=' dopo il nome del campo")
                fields.append(FieldDeclNode(name=f_tok.lexeme, value=self._parse_expression(), line=f_tok.line))
            self._skip_semicolons()

        self._consume(TokenType.RBRACE, "Atteso '}' alla fine dell'entità")
        return EntityDeclNode(
            name=name_tok.lexeme,
            vertical_parents=vertical_parents,
            horizontal_peers=horizontal_peers,
            diagonal_bridges=diagonal_bridges,
            fields=fields,
            methods=methods,
            line=name_tok.line,
        )

    def _parse_function_decl(self, is_method: bool) -> FunctionDeclNode:
        name_tok = self._consume(TokenType.IDENT, "Atteso nome della funzione o metodo")
        self._consume(TokenType.LPAREN, "Atteso '(' dopo il nome della funzione")
        params: List[str] = []
        if not self._check(TokenType.RPAREN):
            first_param = self._consume(TokenType.IDENT, "Atteso nome parametro").lexeme
            # Se l'utente scrive esplicitamente `fn metodo(self, x)` (stile Python), rimuoviamo il `self` ridondante
            if not (is_method and first_param == "self"):
                params.append(first_param)
            while self._match(TokenType.COMMA):
                p = self._consume(TokenType.IDENT, "Atteso nome parametro dopo ','").lexeme
                if not (is_method and p == "self" and not params):
                    params.append(p)
        self._consume(TokenType.RPAREN, "Atteso ')' dopo i parametri")
        if self._match(TokenType.FAT_ARROW):
            expr = self._parse_expression()
            body: List[ASTNode] = [ReturnStmtNode(value=expr, line=name_tok.line)]
        else:
            body = self._parse_block()
        return FunctionDeclNode(
            name=name_tok.lexeme,
            params=params,
            body=body,
            is_method=is_method,
            line=name_tok.line,
        )

    def _parse_lambda_expr(self, fn_line: int) -> LambdaNode:
        self._consume(TokenType.LPAREN, "Atteso '(' per la funzione anonima")
        params: List[str] = []
        if not self._check(TokenType.RPAREN):
            params.append(self._consume(TokenType.IDENT, "Atteso nome parametro").lexeme)
            while self._match(TokenType.COMMA):
                params.append(self._consume(TokenType.IDENT, "Atteso nome parametro dopo ','").lexeme)
        self._consume(TokenType.RPAREN, "Atteso ')' dopo i parametri della lambda")
        if self._match(TokenType.FAT_ARROW):
            expr = self._parse_expression()
            body: List[ASTNode] = [ReturnStmtNode(value=expr, line=fn_line)]
        else:
            body = self._parse_block()
        return LambdaNode(params=params, body=body, line=fn_line)

    def _parse_let_stmt(self, is_wave: bool) -> LetStmtNode:
        name_tok = self._consume(TokenType.IDENT, "Atteso nome variabile")
        if self._match(TokenType.ASSIGN):
            val = self._parse_expression()
        else:
            val = LiteralNode(None, line=name_tok.line)
        return LetStmtNode(name=name_tok.lexeme, value=val, is_wave=is_wave, line=name_tok.line)

    def _parse_statement(self) -> ASTNode:
        if self._match(TokenType.IF):
            return self._parse_if_stmt()
        if self._match(TokenType.WHILE):
            return self._parse_while_stmt()
        if self._match(TokenType.FOR):
            return self._parse_for_stmt()
        if self._match(TokenType.RETURN):
            ret_line = self.tokens[self.current - 1].line
            if self._check(TokenType.RBRACE, TokenType.SEMICOLON) or self._is_at_end():
                return ReturnStmtNode(value=None, line=ret_line)
            return ReturnStmtNode(value=self._parse_expression(), line=ret_line)

        expr = self._parse_expression()
        if self._match(TokenType.ASSIGN):
            eq_tok = self.tokens[self.current - 1]
            if not isinstance(expr, (IdentifierNode, MemberAccessNode, IndexAccessNode)):
                raise DhrivSyntaxError("Bersaglio di assegnazione non valido.", eq_tok.line, eq_tok.column)
            val = self._parse_expression()
            return AssignStmtNode(target=expr, value=val, line=eq_tok.line)

        return ExprStmtNode(expr=expr, line=getattr(expr, "line", 1))

    def _parse_block(self) -> List[ASTNode]:
        self._consume(TokenType.LBRACE, "Atteso '{' per iniziare il blocco")
        stmts: List[ASTNode] = []
        self._skip_semicolons()
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            stmts.append(self._parse_declaration())
            self._skip_semicolons()
        self._consume(TokenType.RBRACE, "Atteso '}' per chiudere il blocco")
        return stmts

    def _parse_if_stmt(self) -> IfStmtNode:
        line = self.tokens[self.current - 1].line
        cond = self._parse_expression()
        then_branch = self._parse_block()
        else_branch: Optional[List[ASTNode]] = None
        if self._match(TokenType.ELSE):
            if self._match(TokenType.IF):
                else_branch = [self._parse_if_stmt()]
            else:
                else_branch = self._parse_block()
        return IfStmtNode(condition=cond, then_branch=then_branch, else_branch=else_branch, line=line)

    def _parse_while_stmt(self) -> WhileStmtNode:
        line = self.tokens[self.current - 1].line
        cond = self._parse_expression()
        body = self._parse_block()
        return WhileStmtNode(condition=cond, body=body, line=line)

    def _parse_for_stmt(self) -> ForStmtNode:
        line = self.tokens[self.current - 1].line
        iter_tok = self._consume(TokenType.IDENT, "Atteso nome iteratore dopo 'for'")
        self._consume(TokenType.IN, "Atteso 'in' / 'dentro' nel ciclo for")
        iterable = self._parse_expression()
        body = self._parse_block()
        return ForStmtNode(iterator_name=iter_tok.lexeme, iterable=iterable, body=body, line=line)

    # --- Gerarchia di Precedenza delle Espressioni ---

    def _parse_expression(self) -> ASTNode:
        return self._parse_triaxial()

    def _parse_triaxial(self) -> ASTNode:
        expr = self._parse_or()
        while self._check(
            TokenType.PIPE_VERTICAL,
            TokenType.RESONATE_HORIZONTAL,
            TokenType.BRIDGE_DIAGONAL,
        ):
            op_tok = self._advance()
            right = self._parse_or()
            axis = (
                "VERTICAL"
                if op_tok.type == TokenType.PIPE_VERTICAL
                else "HORIZONTAL"
                if op_tok.type == TokenType.RESONATE_HORIZONTAL
                else "DIAGONAL"
            )
            expr = TriAxialOpNode(axis=axis, left=expr, right=right, line=op_tok.line)
        return expr

    def _parse_or(self) -> ASTNode:
        expr = self._parse_and()
        while self._match(TokenType.OR):
            op_tok = self.tokens[self.current - 1]
            right = self._parse_and()
            expr = BinaryOpNode(op="or", left=expr, right=right, line=op_tok.line)
        return expr

    def _parse_and(self) -> ASTNode:
        expr = self._parse_bit_or()
        while self._match(TokenType.AND):
            op_tok = self.tokens[self.current - 1]
            right = self._parse_bit_or()
            expr = BinaryOpNode(op="and", left=expr, right=right, line=op_tok.line)
        return expr

    def _parse_bit_or(self) -> ASTNode:
        expr = self._parse_bit_xor()
        while self._match(TokenType.BIT_OR):
            op_tok = self.tokens[self.current - 1]
            right = self._parse_bit_xor()
            expr = BinaryOpNode(op="|", left=expr, right=right, line=op_tok.line)
        return expr

    def _parse_bit_xor(self) -> ASTNode:
        expr = self._parse_bit_and()
        while self._match(TokenType.BIT_XOR):
            op_tok = self.tokens[self.current - 1]
            right = self._parse_bit_and()
            expr = BinaryOpNode(op="^", left=expr, right=right, line=op_tok.line)
        return expr

    def _parse_bit_and(self) -> ASTNode:
        expr = self._parse_equality()
        while self._match(TokenType.BIT_AND):
            op_tok = self.tokens[self.current - 1]
            right = self._parse_equality()
            expr = BinaryOpNode(op="&", left=expr, right=right, line=op_tok.line)
        return expr

    def _parse_equality(self) -> ASTNode:
        expr = self._parse_comparison()
        while self._check(TokenType.EQ, TokenType.NEQ):
            op_tok = self._advance()
            right = self._parse_comparison()
            expr = BinaryOpNode(op=op_tok.lexeme, left=expr, right=right, line=op_tok.line)
        return expr

    def _parse_comparison(self) -> ASTNode:
        expr = self._parse_shift()
        while self._check(TokenType.LT, TokenType.LTE, TokenType.GT, TokenType.GTE):
            op_tok = self._advance()
            right = self._parse_shift()
            expr = BinaryOpNode(op=op_tok.lexeme, left=expr, right=right, line=op_tok.line)
        return expr

    def _parse_shift(self) -> ASTNode:
        expr = self._parse_term()
        while self._check(TokenType.SHL, TokenType.SHR):
            op_tok = self._advance()
            right = self._parse_term()
            expr = BinaryOpNode(op=op_tok.lexeme, left=expr, right=right, line=op_tok.line)
        return expr

    def _parse_term(self) -> ASTNode:
        expr = self._parse_factor()
        while self._check(TokenType.PLUS, TokenType.MINUS):
            op_tok = self._advance()
            right = self._parse_factor()
            expr = BinaryOpNode(op=op_tok.lexeme, left=expr, right=right, line=op_tok.line)
        return expr

    def _parse_factor(self) -> ASTNode:
        expr = self._parse_power()
        while self._check(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op_tok = self._advance()
            right = self._parse_power()
            expr = BinaryOpNode(op=op_tok.lexeme, left=expr, right=right, line=op_tok.line)
        return expr

    def _parse_power(self) -> ASTNode:
        expr = self._parse_unary()
        if self._match(TokenType.POW):
            op_tok = self.tokens[self.current - 1]
            right = self._parse_power()  # associativo a destra
            return BinaryOpNode(op="**", left=expr, right=right, line=op_tok.line)
        return expr

    def _parse_unary(self) -> ASTNode:
        if self._check(TokenType.MINUS, TokenType.NOT, TokenType.BIT_NOT):
            op_tok = self._advance()
            if op_tok.type == TokenType.NOT:
                op_str = "not"
            elif op_tok.type == TokenType.BIT_NOT:
                op_str = "~"
            else:
                op_str = "-"
            return UnaryOpNode(op=op_str, operand=self._parse_unary(), line=op_tok.line)
        return self._parse_postfix()

    def _parse_postfix(self) -> ASTNode:
        expr = self._parse_primary()
        while True:
            if self._match(TokenType.LPAREN):
                lp_line = self.tokens[self.current - 1].line
                args = self._parse_arg_list()
                expr = CallNode(callee=expr, args=args, line=lp_line)
            elif self._match(TokenType.DOT):
                dot_line = self.tokens[self.current - 1].line
                member_tok = self._consume(TokenType.IDENT, "Atteso nome attributo o metodo dopo '.'")
                expr = MemberAccessNode(obj=expr, member=member_tok.lexeme, line=dot_line)
            elif self._match(TokenType.LBRACKET):
                lb_line = self.tokens[self.current - 1].line
                idx_expr = self._parse_expression()
                self._consume(TokenType.RBRACKET, "Atteso ']' dopo l'indice")
                expr = IndexAccessNode(obj=expr, index=idx_expr, line=lb_line)
            else:
                break
        return expr

    def _parse_arg_list(self) -> List[ASTNode]:
        args: List[ASTNode] = []
        if not self._check(TokenType.RPAREN):
            args.append(self._parse_expression())
            while self._match(TokenType.COMMA):
                if self._check(TokenType.RPAREN):
                    break
                args.append(self._parse_expression())
        self._consume(TokenType.RPAREN, "Atteso ')' dopo la lista degli argomenti")
        return args

    def _parse_primary(self) -> ASTNode:
        tok = self._peek()

        if self._match(TokenType.NUMBER, TokenType.STRING, TokenType.TRUE, TokenType.FALSE, TokenType.NULL):
            return LiteralNode(value=tok.literal, line=tok.line)

        if self._match(TokenType.FN):
            return self._parse_lambda_expr(tok.line)

        if self._match(TokenType.NEW):
            ent_tok = self._consume(TokenType.IDENT, "Atteso nome entità dopo 'new'/'nuovo'")
            self._consume(TokenType.LPAREN, "Atteso '(' per istanziare l'entità")
            args = self._parse_arg_list()
            return NewInstanceNode(entity_name=ent_tok.lexeme, args=args, line=tok.line)

        if self._match(TokenType.IDENT):
            return IdentifierNode(name=tok.lexeme, line=tok.line)

        if self._match(TokenType.LPAREN):
            expr = self._parse_expression()
            self._consume(TokenType.RPAREN, "Atteso ')' dopo l'espressione tra parentesi")
            return expr

        if self._match(TokenType.LBRACKET):
            elems: List[ASTNode] = []
            if not self._check(TokenType.RBRACKET):
                elems.append(self._parse_expression())
                while self._match(TokenType.COMMA):
                    if self._check(TokenType.RBRACKET):
                        break
                    elems.append(self._parse_expression())
            self._consume(TokenType.RBRACKET, "Atteso ']' alla fine della lista")
            return ListLiteralNode(elements=elems, line=tok.line)

        if self._match(TokenType.LBRACE):
            entries: List[Tuple[ASTNode, ASTNode]] = []
            if not self._check(TokenType.RBRACE):
                k = self._parse_expression()
                self._consume(TokenType.COLON, "Atteso ':' tra chiave e valore nella mappa")
                v = self._parse_expression()
                entries.append((k, v))
                while self._match(TokenType.COMMA):
                    if self._check(TokenType.RBRACE):
                        break
                    k = self._parse_expression()
                    self._consume(TokenType.COLON, "Atteso ':' tra chiave e valore nella mappa")
                    v = self._parse_expression()
                    entries.append((k, v))
            self._consume(TokenType.RBRACE, "Atteso '}' alla fine della mappa")
            return MapLiteralNode(entries=entries, line=tok.line)

        raise DhrivSyntaxError(f"Espressione inattesa a {tok.lexeme!r}", tok.line, tok.column)
