"""
ai_engine.py — Motore Nativo di Intelligenza Artificiale, Tensori Differenziabili (Autograd),
Self-Attention, Reti Neurali Tri-Assiali e Memoria Neuro-Simbolica per il linguaggio Dhriv.

Zero dipendenze esterne (solo libreria standard Python), progettato per fondersi con i 3 assi:
- Asse VERTICALE (`|>`): propagazione forward profonda attraverso strati neurali ed entità;
- Asse ORIZZONTALE (`<->`): risonanza/interferenza simmetrica tra tensori e gate laterali intra-layer;
- Asse DIAGONALE (`</>`): proiezione matriciale cross-dimensionale (`matmul`) e skip-bridge diretti.

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

from __future__ import annotations

import math
import random
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union


def _to_2d(raw: Any) -> List[List[float]]:
    if isinstance(raw, Tensor):
        return [list(row) for row in raw.data]
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return [[float(raw)]]
    if isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            raise ValueError("Impossibile creare un Tensor vuoto.")
        if isinstance(raw[0], (list, tuple)):
            cols = len(raw[0])
            if cols == 0 or any(len(r) != cols for r in raw):
                raise ValueError("Tutte le righe di una matrice Tensor devono avere la stessa lunghezza.")
            return [[float(x) for x in r] for r in raw]
        return [[float(x) for x in raw]]
    raise TypeError(f"Tipo non convertibile in Tensor: {type(raw).__name__}")


def _zeros(rows: int, cols: int) -> List[List[float]]:
    return [[0.0] * cols for _ in range(rows)]


class Tensor:
    """
    Tensore differenziabile 2D `(rows, cols)` con motore Autograd Reverse-Mode integrato.
    Scalari `(1, 1)` e vettori riga `(1, D)` sono casi particolari uniformi e componibili.
    """

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        self.data: List[List[float]] = _to_2d(data)
        self.rows: int = len(self.data)
        self.cols: int = len(self.data[0])
        self.requires_grad: bool = bool(requires_grad)
        self.grad: List[List[float]] = _zeros(self.rows, self.cols)
        self._prev: Tuple[Tensor, ...] = _children
        self._op: str = _op
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self) -> List[int]:
        return [self.rows, self.cols]

    @property
    def values(self) -> List[List[float]]:
        return [list(r) for r in self.data]

    @property
    def grad_values(self) -> List[List[float]]:
        return [list(r) for r in self.grad]

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def item(self) -> float:
        if self.rows == 1 and self.cols == 1:
            return self.data[0][0]
        raise ValueError(f"item() richiede un Tensor (1, 1), trovato ({self.rows}, {self.cols}).")

    def zero_grad(self) -> None:
        self.grad = _zeros(self.rows, self.cols)

    def __repr__(self) -> str:
        if self.rows == 1 and self.cols == 1:
            return f"Tensor({self.data[0][0]:.6g})"
        formatted = "[" + ", ".join("[" + ", ".join(f"{v:.4g}" for v in r) + "]" for r in self.data) + "]"
        return f"Tensor(shape=({self.rows},{self.cols}), data={formatted})"

    # ------------------------------------------------- operazioni algebriche

    @staticmethod
    def _ensure(other: Any) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)

    def _accum_broadcast_grad(self, grad_out: List[List[float]]) -> None:
        """Accumula `grad_out` riducendo gli assi che erano stati espansi in broadcast."""
        if not self.requires_grad:
            return
        out_r, out_c = len(grad_out), len(grad_out[0])
        if self.rows == out_r and self.cols == out_c:
            for r in range(out_r):
                gr = self.grad[r]
                gor = grad_out[r]
                for c in range(out_c):
                    gr[c] += gor[c]
        elif self.rows == 1 and self.cols == out_c:
            g0 = self.grad[0]
            for r in range(out_r):
                gor = grad_out[r]
                for c in range(out_c):
                    g0[c] += gor[c]
        elif self.rows == out_r and self.cols == 1:
            for r in range(out_r):
                self.grad[r][0] += sum(grad_out[r])
        elif self.rows == 1 and self.cols == 1:
            total = 0.0
            for r in range(out_r):
                total += sum(grad_out[r])
            self.grad[0][0] += total

    @staticmethod
    def _broadcast_pair(a: "Tensor", b: "Tensor") -> Tuple[int, int]:
        if a.rows == b.rows or a.rows == 1 or b.rows == 1:
            out_r = max(a.rows, b.rows)
        else:
            raise ValueError(f"Dimensioni righe incompatibili per broadcast: ({a.rows},{a.cols}) vs ({b.rows},{b.cols}).")
        if a.cols == b.cols or a.cols == 1 or b.cols == 1:
            out_c = max(a.cols, b.cols)
        else:
            raise ValueError(f"Dimensioni colonne incompatibili per broadcast: ({a.rows},{a.cols}) vs ({b.rows},{b.cols}).")
        return out_r, out_c

    def add(self, other: Any) -> "Tensor":
        b = self._ensure(other)
        out_r, out_c = self._broadcast_pair(self, b)
        out_data = [
            [
                self.data[0 if self.rows == 1 else r][0 if self.cols == 1 else c]
                + b.data[0 if b.rows == 1 else r][0 if b.cols == 1 else c]
                for c in range(out_c)
            ]
            for r in range(out_r)
        ]
        req = self.requires_grad or b.requires_grad
        out = Tensor(out_data, requires_grad=req, _children=(self, b) if req else (), _op="+")

        if req:
            def _bw() -> None:
                self._accum_broadcast_grad(out.grad)
                b._accum_broadcast_grad(out.grad)

            out._backward = _bw
        return out

    def neg(self) -> "Tensor":
        out_data = [[-v for v in r] for r in self.data]
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,) if self.requires_grad else (), _op="neg")
        if self.requires_grad:
            def _bw() -> None:
                for r in range(self.rows):
                    for c in range(self.cols):
                        self.grad[r][c] -= out.grad[r][c]

            out._backward = _bw
        return out

    def sub(self, other: Any) -> "Tensor":
        return self.add(self._ensure(other).neg())

    def mul(self, other: Any) -> "Tensor":
        b = self._ensure(other)
        out_r, out_c = self._broadcast_pair(self, b)
        out_data = [
            [
                self.data[0 if self.rows == 1 else r][0 if self.cols == 1 else c]
                * b.data[0 if b.rows == 1 else r][0 if b.cols == 1 else c]
                for c in range(out_c)
            ]
            for r in range(out_r)
        ]
        req = self.requires_grad or b.requires_grad
        out = Tensor(out_data, requires_grad=req, _children=(self, b) if req else (), _op="*")

        if req:
            def _bw() -> None:
                if self.requires_grad:
                    ga = [
                        [
                            out.grad[r][c] * b.data[0 if b.rows == 1 else r][0 if b.cols == 1 else c]
                            for c in range(out_c)
                        ]
                        for r in range(out_r)
                    ]
                    self._accum_broadcast_grad(ga)
                if b.requires_grad:
                    gb = [
                        [
                            out.grad[r][c] * self.data[0 if self.rows == 1 else r][0 if self.cols == 1 else c]
                            for c in range(out_c)
                        ]
                        for r in range(out_r)
                    ]
                    b._accum_broadcast_grad(gb)

            out._backward = _bw
        return out

    def div(self, other: Any) -> "Tensor":
        b = self._ensure(other)
        out_r, out_c = self._broadcast_pair(self, b)
        out_data = []
        for r in range(out_r):
            row = []
            for c in range(out_c):
                denom = b.data[0 if b.rows == 1 else r][0 if b.cols == 1 else c]
                if denom == 0.0:
                    raise ZeroDivisionError("Divisione per zero nel Tensor.")
                num = self.data[0 if self.rows == 1 else r][0 if self.cols == 1 else c]
                row.append(num / denom)
            out_data.append(row)
        req = self.requires_grad or b.requires_grad
        out = Tensor(out_data, requires_grad=req, _children=(self, b) if req else (), _op="/")

        if req:
            def _bw() -> None:
                if self.requires_grad:
                    ga = [
                        [
                            out.grad[r][c] / b.data[0 if b.rows == 1 else r][0 if b.cols == 1 else c]
                            for c in range(out_c)
                        ]
                        for r in range(out_r)
                    ]
                    self._accum_broadcast_grad(ga)
                if b.requires_grad:
                    gb = [
                        [
                            -out.grad[r][c]
                            * self.data[0 if self.rows == 1 else r][0 if self.cols == 1 else c]
                            / (b.data[0 if b.rows == 1 else r][0 if b.cols == 1 else c] ** 2)
                            for c in range(out_c)
                        ]
                        for r in range(out_r)
                    ]
                    b._accum_broadcast_grad(gb)

            out._backward = _bw
        return out

    def matmul(self, other: "Tensor") -> "Tensor":
        """Prodotto matriciale differenziabile `(M, K) @ (K, N) -> (M, N)` (anche operatore `</>`)."""
        b = self._ensure(other)
        if self.cols != b.rows:
            raise ValueError(
                f"Dimensioni incompatibili per matmul/ponte diagonale: ({self.rows},{self.cols}) </> ({b.rows},{b.cols})."
            )
        m, k_dim, n = self.rows, self.cols, b.cols
        # Pre-trasponiamo `b.data` per località di accesso in Python
        b_t = [[b.data[p][j] for p in range(k_dim)] for j in range(n)]
        out_data = [
            [sum(self.data[i][p] * b_t[j][p] for p in range(k_dim)) for j in range(n)]
            for i in range(m)
        ]
        req = self.requires_grad or b.requires_grad
        out = Tensor(out_data, requires_grad=req, _children=(self, b) if req else (), _op="matmul")

        if req:
            def _bw() -> None:
                if self.requires_grad:
                    # dA += dOut @ B^T
                    for i in range(m):
                        gi = self.grad[i]
                        go_i = out.grad[i]
                        for p in range(k_dim):
                            bp = b.data[p]
                            s = 0.0
                            for j in range(n):
                                s += go_i[j] * bp[j]
                            gi[p] += s
                if b.requires_grad:
                    # dB += A^T @ dOut
                    for p in range(k_dim):
                        gp = b.grad[p]
                        for j in range(n):
                            s = 0.0
                            for i in range(m):
                                s += self.data[i][p] * out.grad[i][j]
                            gp[j] += s

            out._backward = _bw
        return out

    def transpose(self) -> "Tensor":
        out_data = [[self.data[r][c] for r in range(self.rows)] for c in range(self.cols)]
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,) if self.requires_grad else (), _op="T")
        if self.requires_grad:
            def _bw() -> None:
                for r in range(self.rows):
                    for c in range(self.cols):
                        self.grad[r][c] += out.grad[c][r]

            out._backward = _bw
        return out

    def sum(self) -> "Tensor":
        total = sum(sum(r) for r in self.data)
        out = Tensor([[total]], requires_grad=self.requires_grad, _children=(self,) if self.requires_grad else (), _op="sum")
        if self.requires_grad:
            def _bw() -> None:
                g = out.grad[0][0]
                for r in range(self.rows):
                    for c in range(self.cols):
                        self.grad[r][c] += g

            out._backward = _bw
        return out

    def mean(self) -> "Tensor":
        count = float(self.rows * self.cols)
        return self.sum().div(count)

    # ------------------------------------------------- attivazioni neurali

    def relu(self) -> "Tensor":
        out_data = [[v if v > 0.0 else 0.0 for v in r] for r in self.data]
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,) if self.requires_grad else (), _op="relu")
        if self.requires_grad:
            def _bw() -> None:
                for r in range(self.rows):
                    for c in range(self.cols):
                        if self.data[r][c] > 0.0:
                            self.grad[r][c] += out.grad[r][c]

            out._backward = _bw
        return out

    def sigmoid(self) -> "Tensor":
        def _sig(x: float) -> float:
            if x >= 0.0:
                z = math.exp(-x)
                return 1.0 / (1.0 + z)
            z = math.exp(x)
            return z / (1.0 + z)

        out_data = [[_sig(v) for v in r] for r in self.data]
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,) if self.requires_grad else (), _op="sigmoid")
        if self.requires_grad:
            def _bw() -> None:
                for r in range(self.rows):
                    for c in range(self.cols):
                        s = out.data[r][c]
                        self.grad[r][c] += out.grad[r][c] * s * (1.0 - s)

            out._backward = _bw
        return out

    def tanh(self) -> "Tensor":
        out_data = [[math.tanh(v) for v in r] for r in self.data]
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,) if self.requires_grad else (), _op="tanh")
        if self.requires_grad:
            def _bw() -> None:
                for r in range(self.rows):
                    for c in range(self.cols):
                        t = out.data[r][c]
                        self.grad[r][c] += out.grad[r][c] * (1.0 - t * t)

            out._backward = _bw
        return out

    def gelu(self) -> "Tensor":
        k = math.sqrt(2.0 / math.pi)
        c3 = 0.044715
        out_data = []
        tanh_cache: List[List[float]] = []
        for r in range(self.rows):
            row = []
            trow = []
            for c in range(self.cols):
                x = self.data[r][c]
                t = math.tanh(k * (x + c3 * x * x * x))
                trow.append(t)
                row.append(0.5 * x * (1.0 + t))
            out_data.append(row)
            tanh_cache.append(trow)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,) if self.requires_grad else (), _op="gelu")
        if self.requires_grad:
            def _bw() -> None:
                for r in range(self.rows):
                    for c in range(self.cols):
                        x = self.data[r][c]
                        t = tanh_cache[r][c]
                        sech2 = 1.0 - t * t
                        dgelu = 0.5 * (1.0 + t) + 0.5 * x * sech2 * k * (1.0 + 3.0 * c3 * x * x)
                        self.grad[r][c] += out.grad[r][c] * dgelu

            out._backward = _bw
        return out

    def softmax(self) -> "Tensor":
        """Softmax numericamente stabile riga per riga con Jacobiano analitico completo."""
        out_data = []
        for row in self.data:
            m = max(row)
            exps = [math.exp(v - m) for v in row]
            s = sum(exps)
            out_data.append([e / s for e in exps])
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,) if self.requires_grad else (), _op="softmax")
        if self.requires_grad:
            def _bw() -> None:
                for r in range(self.rows):
                    sr = out.data[r]
                    gr = out.grad[r]
                    dot = sum(gr[c] * sr[c] for c in range(self.cols))
                    for c in range(self.cols):
                        self.grad[r][c] += sr[c] * (gr[c] - dot)

            out._backward = _bw
        return out

    # ------------------------------------------------- backpropagation

    def backward(self) -> "Tensor":
        """Esegue la differenziazione automatica reverse-mode lungo l'intero grafo topologico."""
        topo: List[Tensor] = []
        visited: Set[int] = set()

        def _build(v: Tensor) -> None:
            vid = id(v)
            if vid not in visited:
                visited.add(vid)
                for child in v._prev:
                    _build(child)
                topo.append(v)

        _build(self)
        self.grad = [[1.0] * self.cols for _ in range(self.rows)]
        for node in reversed(topo):
            node._backward()
        return self

    # Operatori Python nativi
    __add__ = add
    __radd__ = lambda self, other: Tensor._ensure(other).add(self)
    __sub__ = sub
    __rsub__ = lambda self, other: Tensor._ensure(other).sub(self)
    __neg__ = neg
    __mul__ = mul
    __rmul__ = lambda self, other: Tensor._ensure(other).mul(self)
    __truediv__ = div


# ============================================================================
# FUNZIONI DI COSTO (LOSS) DIFFERENZIABILI
# ============================================================================

def mse_loss(pred: Tensor, target: Any) -> Tensor:
    t = Tensor._ensure(target)
    diff = pred.sub(t)
    return diff.mul(diff).mean()


def bce_loss(pred: Tensor, target: Any, eps: float = 1e-7) -> Tensor:
    t = Tensor._ensure(target)
    if pred.rows != t.rows or pred.cols != t.cols:
        raise ValueError("Dimensioni incompatibili in bce_loss.")
    n = float(pred.rows * pred.cols)
    total = 0.0
    for r in range(pred.rows):
        for c in range(pred.cols):
            p = min(max(pred.data[r][c], eps), 1.0 - eps)
            y = t.data[r][c]
            total += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    loss = Tensor([[total / n]], requires_grad=pred.requires_grad, _children=(pred,) if pred.requires_grad else (), _op="bce")
    if pred.requires_grad:
        def _bw() -> None:
            scale = loss.grad[0][0] / n
            for r in range(pred.rows):
                for c in range(pred.cols):
                    p = min(max(pred.data[r][c], eps), 1.0 - eps)
                    y = t.data[r][c]
                    pred.grad[r][c] += scale * ((p - y) / (p * (1.0 - p)))

        loss._backward = _bw
    return loss


def cross_entropy_loss(logits: Tensor, targets: Union[Sequence[int], Tensor]) -> Tensor:
    """Cross-entropy multi-classe stabile direttamente dai logit."""
    probs = logits.softmax()
    batch = logits.rows
    if isinstance(targets, Tensor):
        target_idx = [int(targets.data[r][0]) if targets.cols == 1 else max(range(targets.cols), key=lambda c: targets.data[r][c]) for r in range(batch)]
    else:
        target_idx = [int(x) for x in targets]
    if len(target_idx) != batch:
        raise ValueError("Il numero di etichette target non coincide con il batch.")
    eps = 1e-12
    total = sum(-math.log(max(probs.data[r][target_idx[r]], eps)) for r in range(batch)) / float(batch)
    req = logits.requires_grad
    loss = Tensor([[total]], requires_grad=req, _children=(logits,) if req else (), _op="cross_entropy")
    if req:
        def _bw() -> None:
            scale = loss.grad[0][0] / float(batch)
            for r in range(batch):
                for c in range(logits.cols):
                    y = 1.0 if c == target_idx[r] else 0.0
                    logits.grad[r][c] += scale * (probs.data[r][c] - y)

        loss._backward = _bw
    return loss


# ============================================================================
# MODULI NEURALI & RETE COGNITIVA TRI-ASSIALE (V / H / D)
# ============================================================================

def _xavier_matrix(rows: int, cols: int, rng: random.Random) -> Tensor:
    limit = math.sqrt(6.0 / float(rows + cols))
    data = [[rng.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]
    return Tensor(data, requires_grad=True)


class NeuralModule:
    """Classe radice per tutti i moduli neurali invocabili anche tramite pipe verticale `|>`."""

    def parameters(self) -> List[Tensor]:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Any) -> Tensor:
        return self.forward(Tensor._ensure(x))


class LinearLayer(NeuralModule):
    """Strato denso affine `y = x </> W + b`."""

    def __init__(self, in_features: int, out_features: int, seed: int = 42) -> None:
        rng = random.Random(int(seed))
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.W = _xavier_matrix(self.in_features, self.out_features, rng)
        self.b = Tensor([[0.0] * self.out_features], requires_grad=True)

    def parameters(self) -> List[Tensor]:
        return [self.W, self.b]

    def forward(self, x: Tensor) -> Tensor:
        xt = Tensor._ensure(x)
        return xt.matmul(self.W).add(self.b)

    def __repr__(self) -> str:
        return f"<LinearLayer {self.in_features}->{self.out_features}>"


class SelfAttentionLayer(NeuralModule):
    """
    Scaled Dot-Product Self-Attention (cuore dei Transformer):
    `Attention(Q, K, V) = softmax((Q </> K^T) / sqrt(d_model)) </> V </> W_O`
    """

    def __init__(self, d_model: int, seed: int = 42) -> None:
        rng = random.Random(int(seed))
        self.d_model = int(d_model)
        self.W_q = _xavier_matrix(self.d_model, self.d_model, rng)
        self.W_k = _xavier_matrix(self.d_model, self.d_model, rng)
        self.W_v = _xavier_matrix(self.d_model, self.d_model, rng)
        self.W_o = _xavier_matrix(self.d_model, self.d_model, rng)

    def parameters(self) -> List[Tensor]:
        return [self.W_q, self.W_k, self.W_v, self.W_o]

    def forward(self, x: Tensor) -> Tensor:
        xt = Tensor._ensure(x)
        q = xt.matmul(self.W_q)
        k = xt.matmul(self.W_k)
        v = xt.matmul(self.W_v)
        scores = q.matmul(k.transpose()).div(math.sqrt(float(self.d_model)))
        weights = scores.softmax()
        return weights.matmul(v).matmul(self.W_o)

    def __repr__(self) -> str:
        return f"<SelfAttentionLayer d_model={self.d_model}>"


class TriAxialNetwork(NeuralModule):
    """
    Architettura Neurale Tri-Assiale nativa ed esclusiva di Dhriv:
    - Asse VERTICALE (`V`): gerarchia profonda di strati non-lineari (`tanh`);
    - Asse ORIZZONTALE (`H`): gate sinaptico laterale intra-piano (`sigmoid` highway resonance);
    - Asse DIAGONALE (`D`): ponte proiettivo diretto dall'ingresso grezzo all'uscita (`X_0 </> W_D`),
        che taglia simultaneamente strati e dimensioni preservando il flusso del gradiente.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int = 2,
        output_activation: str = "sigmoid",
        seed: int = 42,
    ) -> None:
        rng = random.Random(int(seed))
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.depth = max(1, int(depth))
        self.output_activation = str(output_activation).lower()

        self.in_proj = LinearLayer(self.in_dim, self.hidden_dim, seed=rng.randint(1, 10_000_000))
        self.vertical_layers: List[LinearLayer] = [
            LinearLayer(self.hidden_dim, self.hidden_dim, seed=rng.randint(1, 10_000_000))
            for _ in range(self.depth)
        ]
        self.horizontal_gates: List[LinearLayer] = [
            LinearLayer(self.hidden_dim, self.hidden_dim, seed=rng.randint(1, 10_000_000))
            for _ in range(self.depth)
        ]
        self.diagonal_bridge = _xavier_matrix(self.in_dim, self.out_dim, rng)
        self.out_head = LinearLayer(self.hidden_dim, self.out_dim, seed=rng.randint(1, 10_000_000))

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        params.extend(self.in_proj.parameters())
        for vl, hg in zip(self.vertical_layers, self.horizontal_gates):
            params.extend(vl.parameters())
            params.extend(hg.parameters())
        params.append(self.diagonal_bridge)
        params.extend(self.out_head.parameters())
        return params

    def forward(self, x: Tensor) -> Tensor:
        x0 = Tensor._ensure(x)
        h = self.in_proj(x0).tanh()
        for vl, hg in zip(self.vertical_layers, self.horizontal_gates):
            v_feat = vl(h).tanh()          # Trasformazione Verticale profonda
            h_gate = hg(h).sigmoid()       # Risonanza Orizzontale intra-livello
            inv_gate = Tensor(1.0).sub(h_gate)
            h = v_feat.mul(h_gate).add(h.mul(inv_gate))
        raw_out = self.out_head(h).add(x0.matmul(self.diagonal_bridge))  # + Ponte Diagonale diretto
        if self.output_activation == "sigmoid":
            return raw_out.sigmoid()
        if self.output_activation == "tanh":
            return raw_out.tanh()
        if self.output_activation == "softmax":
            return raw_out.softmax()
        return raw_out

    def __repr__(self) -> str:
        return (
            f"<TriAxialNetwork V-depth={self.depth}, "
            f"{self.in_dim}->{self.hidden_dim}->{self.out_dim}, act={self.output_activation}>"
        )


# ============================================================================
# OTTIMIZZATORI (SGD & ADAM)
# ============================================================================

class SGDOptimizer:
    def __init__(self, params: Sequence[Tensor], lr: float = 0.05, momentum: float = 0.0) -> None:
        self.params = list(params)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self._vel = [_zeros(p.rows, p.cols) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        for p, v in zip(self.params, self._vel):
            for r in range(p.rows):
                for c in range(p.cols):
                    g = p.grad[r][c]
                    if self.momentum != 0.0:
                        v[r][c] = self.momentum * v[r][c] + g
                        p.data[r][c] -= self.lr * v[r][c]
                    else:
                        p.data[r][c] -= self.lr * g


class AdamOptimizer:
    def __init__(
        self,
        params: Sequence[Tensor],
        lr: float = 0.02,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params = list(params)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t = 0
        self._m = [_zeros(p.rows, p.cols) for p in self.params]
        self._v = [_zeros(p.rows, p.cols) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        self.t += 1
        b1_corr = 1.0 - (self.beta1 ** self.t)
        b2_corr = 1.0 - (self.beta2 ** self.t)
        for p, m, v in zip(self.params, self._m, self._v):
            for r in range(p.rows):
                for c in range(p.cols):
                    g = p.grad[r][c]
                    m[r][c] = self.beta1 * m[r][c] + (1.0 - self.beta1) * g
                    v[r][c] = self.beta2 * v[r][c] + (1.0 - self.beta2) * (g * g)
                    m_hat = m[r][c] / b1_corr
                    v_hat = v[r][c] / b2_corr
                    p.data[r][c] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


# ============================================================================
# MEMORIA ASSOCIATIVA NEURO-SIMBOLICA TRI-ASSIALE
# ============================================================================

@dataclass
class MemoryEntry:
    key: str
    embedding: List[float]
    payload: Any
    h: int = 0
    v: int = 0


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("I vettori per cosine_similarity devono avere uguale lunghezza non nulla.")
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) ** 2 for x in a))
    nb = math.sqrt(sum(float(y) ** 2 for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class MindMemory:
    """
    Memoria associativa neuro-simbolica: collega embedding vettoriali continui a
    simboli/entità OOP posizionati nel reticolo `(H, V)`.
    """

    def __init__(self) -> None:
        self.entries: List[MemoryEntry] = []

    @staticmethod
    def _flat_vec(v: Any) -> List[float]:
        if isinstance(v, Tensor):
            return [x for row in v.data for x in row]
        return [float(x) for x in v]

    def store(self, key: str, embedding: Any, payload: Any = None, h: int = 0, v: int = 0) -> int:
        vec = self._flat_vec(embedding)
        self.entries.append(MemoryEntry(str(key), vec, payload, int(h), int(v)))
        return len(self.entries)

    def recall(self, query: Any, top_k: int = 3) -> List[Dict[str, Any]]:
        qvec = self._flat_vec(query)
        scored = []
        for entry in self.entries:
            if len(entry.embedding) != len(qvec):
                continue
            sim = cosine_similarity(qvec, entry.embedding)
            scored.append({
                "key": entry.key,
                "score": round(sim, 6),
                "payload": entry.payload,
                "h": entry.h,
                "v": entry.v,
            })
        scored.sort(key=lambda item: (-item["score"], item["key"]))
        return scored[: max(1, int(top_k))]

    def __repr__(self) -> str:
        return f"<MindMemory entries={len(self.entries)}>"


# ============================================================================
# CODEC BINARIO IEEE-754 PER PESI NEURALI (TRASMISSIBILE ANCHE SU ONDE EM!)
# ============================================================================

AI_MAGIC = b"DHAI"


def pack_weights_binary(params: Sequence[Tensor]) -> bytes:
    """
    Impacchetta una lista di Tensor in binario puro IEEE-754 float32 (`>f`) + CRC32:
    `b"DHAI" | u16 count | [u16 rows, u16 cols, float32*rows*cols]... | u32 crc32`
    """
    chunks = [struct.pack(">H", len(params))]
    for p in params:
        chunks.append(struct.pack(">HH", p.rows, p.cols))
        flat = [x for r in p.data for x in r]
        chunks.append(struct.pack(f">{len(flat)}f", *flat))
    body = b"".join(chunks)
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return AI_MAGIC + body + struct.pack(">I", crc)


def unpack_weights_binary(blob: bytes, target_params: Optional[Sequence[Tensor]] = None) -> List[Tensor]:
    """Verifica CRC32 e ricostruisce (o carica in-place su `target_params`) i pesi neurali dal binario."""
    if len(blob) < 10 or blob[:4] != AI_MAGIC:
        raise ValueError("Magic 'DHAI' mancante: non è un pacchetto binario di pesi Dhriv.")
    body = blob[4:-4]
    (expected_crc,) = struct.unpack(">I", blob[-4:])
    if (zlib.crc32(body) & 0xFFFFFFFF) != expected_crc:
        raise ValueError("CRC32 pesi neurali non valido: dati corrotti.")
    (count,) = struct.unpack(">H", body[:2])
    offset = 2
    decoded: List[Tensor] = []
    for i in range(count):
        rows, cols = struct.unpack(">HH", body[offset : offset + 4])
        offset += 4
        n = rows * cols
        byte_len = n * 4
        vals = struct.unpack(f">{n}f", body[offset : offset + byte_len])
        offset += byte_len
        mat = [list(vals[r * cols : (r + 1) * cols]) for r in range(rows)]
        if target_params is not None and i < len(target_params):
            tp = target_params[i]
            if tp.rows != rows or tp.cols != cols:
                raise ValueError("Forma del tensore incompatibile durante il caricamento pesi.")
            tp.data = mat
            decoded.append(tp)
        else:
            decoded.append(Tensor(mat, requires_grad=True))
    return decoded
