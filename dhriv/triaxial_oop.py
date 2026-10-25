"""
triaxial_oop.py — Paradigma Orientato agli Oggetti Orizzontale, Verticale e Diagonale
e Motore di Ricerca su Larga Scala (Server, Oggetti e Cluster) per il linguaggio Dhriv.

I 3 Assi Fondamentali di Dhriv:
1. ASSE VERTICALE (`vertical` / `|>`): Ereditarietà profonda, gerarchia stack
   (dal registro binario bare-metal fino all'astrazione applicativa).
2. ASSE ORIZZONTALE (`horizontal` / `<->`): Simbiosi paritaria tra pari, trait,
   mixin e nodi server distribuiti sullo stesso livello ("Uno E Molti").
3. ASSE DIAGONALE (`diagonal` / `</>`): Morfismo trasversale inedito che taglia
   simultaneamente profondità verticale (ΔV) e dominio/cluster orizzontale (ΔH),
   collegando direttamente livelli e gerarchie distanti senza colli di bottiglia.

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class TriAxialClass:
    """Metaclasse Dhriv posizionata nel reticolo (Horizontal, Vertical, Diagonal)."""
    name: str
    vertical_parents: List[str] = field(default_factory=list)
    horizontal_peers: List[str] = field(default_factory=list)
    diagonal_bridges: List[str] = field(default_factory=list)
    default_fields: Dict[str, Any] = field(default_factory=dict)
    methods: Dict[str, Any] = field(default_factory=dict)
    h_coord: int = 0
    v_depth: int = 0
    d_vector: Tuple[int, int] = (0, 0)


class TriAxialObject:
    """Istanza OOP viva dotata di stato proprio e legami dinamici Orizzontali e Diagonali."""

    def __init__(self, dhriv_class: TriAxialClass, initial_fields: Optional[Dict[str, Any]] = None) -> None:
        self.dhriv_class = dhriv_class
        self.fields: Dict[str, Any] = dict(initial_fields or {})
        self.horizontal_links: List["TriAxialObject"] = []
        self.diagonal_links: List["TriAxialObject"] = []

    def link_horizontal(self, peer: "TriAxialObject") -> "TriAxialObject":
        """Operatore `<->`: risonanza simbiotica paritaria tra due istanze."""
        if peer is not self and peer not in self.horizontal_links:
            self.horizontal_links.append(peer)
        if self is not peer and self not in peer.horizontal_links:
            peer.horizontal_links.append(self)
        return self

    def link_diagonal(self, target: "TriAxialObject") -> "TriAxialObject":
        """Operatore `</>`: ponte diagonale diretto verso un'istanza di altro livello/cluster."""
        if target is not self and target not in self.diagonal_links:
            self.diagonal_links.append(target)
        return self

    def __repr__(self) -> str:
        return (
            f"<{self.dhriv_class.name} [H={self.dhriv_class.h_coord}, "
            f"V={self.dhriv_class.v_depth}, D={self.dhriv_class.d_vector}] {self.fields}>"
        )


@dataclass
class ResolutionResult:
    found: bool
    value: Any = None
    owner_class: str = ""
    axis_used: str = "SELF"  # SELF | VERTICAL | HORIZONTAL | DIAGONAL | DYNAMIC_LINK
    hops: int = 0


class TriAxialOOPRegistry:
    """
    Registro e Risolutore Tri-Assiale MRO (Method & Attribute Resolution Order) di Dhriv:
    Ordine deterministico a 3 dimensioni:
    1. `SELF` (istanza locale / classe propria)
    2. `VERTICAL` (risalita gerarchica in profondità sui `vertical_parents`)
    3. `HORIZONTAL` (scansione in ampiezza sui `horizontal_peers` e `horizontal_links` vivi)
    4. `DIAGONAL` (salto trasversale sui `diagonal_bridges` e `diagonal_links` vivi)
    Dotato di Inline Cache apprenditiva `O(1)` per azzerare il costo di lookup ripetuto.
    """

    def __init__(self) -> None:
        self.classes: Dict[str, TriAxialClass] = {}
        self._h_counter = 0
        self.inline_dispatch_cache: Dict[Tuple[str, str], ResolutionResult] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def register_class(self, cls_def: TriAxialClass) -> TriAxialClass:
        # Calcola automaticamente profondità Verticale V e coordinata Orizzontale H
        max_parent_v = -1
        for vp in cls_def.vertical_parents:
            if vp in self.classes:
                max_parent_v = max(max_parent_v, self.classes[vp].v_depth)
        cls_def.v_depth = max_parent_v + 1
        cls_def.h_coord = self._h_counter
        self._h_counter += 1
        diag_delta_h = len(cls_def.horizontal_peers) + len(cls_def.diagonal_bridges)
        cls_def.d_vector = (cls_def.h_coord + diag_delta_h, cls_def.v_depth)

        self.classes[cls_def.name] = cls_def
        self.inline_dispatch_cache.clear()
        return cls_def

    def instantiate(self, class_name: str) -> TriAxialObject:
        if class_name not in self.classes:
            raise KeyError(f"Entità OOP '{class_name}' non definita nel registro Tri-Assiale.")
        cls_def = self.classes[class_name]
        merged_fields = self._collect_inherited_fields(class_name)
        return TriAxialObject(dhriv_class=cls_def, initial_fields=merged_fields)

    def _collect_inherited_fields(self, class_name: str) -> Dict[str, Any]:
        visited: Set[str] = set()
        merged: Dict[str, Any] = {}

        def _walk(cname: str) -> None:
            if cname in visited or cname not in self.classes:
                return
            visited.add(cname)
            c = self.classes[cname]
            # Prima i ponti diagonali, poi i peer orizzontali, poi i genitori verticali, infine sé stessa
            for db in c.diagonal_bridges:
                _walk(db)
            for hp in c.horizontal_peers:
                _walk(hp)
            for vp in c.vertical_parents:
                _walk(vp)
            merged.update(c.default_fields)

        _walk(class_name)
        return merged

    def resolve_method(self, class_name: str, member: str) -> ResolutionResult:
        cache_key = (class_name, member)
        cached = self.inline_dispatch_cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self.cache_misses += 1
        cls_def = self.classes.get(class_name)
        if cls_def is None:
            return ResolutionResult(found=False)

        if member in cls_def.methods:
            res = ResolutionResult(True, cls_def.methods[member], class_name, "SELF", 0)
            self.inline_dispatch_cache[cache_key] = res
            return res

        # 1. Ricerca VERTICALE (DFS gerarchica)
        res = self._search_class_graph(cls_def.vertical_parents, member, "VERTICAL", set([class_name]))
        if res.found:
            self.inline_dispatch_cache[cache_key] = res
            return res

        # 2. Ricerca ORIZZONTALE (BFS paritaria)
        res = self._search_class_graph(cls_def.horizontal_peers, member, "HORIZONTAL", set([class_name]))
        if res.found:
            self.inline_dispatch_cache[cache_key] = res
            return res

        # 3. Ricerca DIAGONALE (Salto trasversale incrociato: esplora sia il target diagonale sia la sua catena verticale/orizzontale!)
        res = self._search_diagonal_graph(cls_def.diagonal_bridges, member, set([class_name]))
        if res.found:
            self.inline_dispatch_cache[cache_key] = res
            return res

        return ResolutionResult(found=False)

    def _search_class_graph(
        self,
        roots: List[str],
        member: str,
        axis_tag: str,
        visited: Set[str],
    ) -> ResolutionResult:
        queue: List[Tuple[str, int]] = [(r, 1) for r in roots]
        while queue:
            cname, hops = queue.pop(0)
            if cname in visited or cname not in self.classes:
                continue
            visited.add(cname)
            c = self.classes[cname]
            if member in c.methods:
                return ResolutionResult(True, c.methods[member], cname, axis_tag, hops)
            next_nodes = c.vertical_parents if axis_tag == "VERTICAL" else c.horizontal_peers
            for nxt in next_nodes:
                queue.append((nxt, hops + 1))
        return ResolutionResult(found=False)

    def _search_diagonal_graph(
        self,
        bridges: List[str],
        member: str,
        visited: Set[str],
    ) -> ResolutionResult:
        queue: List[Tuple[str, int]] = [(b, 1) for b in bridges]
        while queue:
            cname, hops = queue.pop(0)
            if cname in visited or cname not in self.classes:
                continue
            visited.add(cname)
            c = self.classes[cname]
            if member in c.methods:
                return ResolutionResult(True, c.methods[member], cname, "DIAGONAL", hops)
            for nxt in c.diagonal_bridges + c.vertical_parents + c.horizontal_peers:
                queue.append((nxt, hops + 1))
        return ResolutionResult(found=False)

    def resolve_instance_member(self, instance: TriAxialObject, member: str) -> ResolutionResult:
        """Risolve prima campi locali, poi metodi Tri-Assiali di classe, infine link dinamici `<->` e `</>`."""
        if member in instance.fields:
            return ResolutionResult(True, instance.fields[member], instance.dhriv_class.name, "SELF", 0)

        m_res = self.resolve_method(instance.dhriv_class.name, member)
        if m_res.found:
            return m_res

        # Risonanza dinamica Orizzontale `<->` tra istanze vive
        for peer in instance.horizontal_links:
            if member in peer.fields:
                return ResolutionResult(True, peer.fields[member], peer.dhriv_class.name, "HORIZONTAL_LINK", 1)
            pm = self.resolve_method(peer.dhriv_class.name, member)
            if pm.found:
                return ResolutionResult(True, (peer, pm.value), peer.dhriv_class.name, "HORIZONTAL_LINK", 1)

        # Ponte dinamico Diagonale `</>` tra istanze vive di livelli diversi
        for target in instance.diagonal_links:
            if member in target.fields:
                return ResolutionResult(True, target.fields[member], target.dhriv_class.name, "DIAGONAL_LINK", 1)
            dm = self.resolve_method(target.dhriv_class.name, member)
            if dm.found:
                return ResolutionResult(True, (target, dm.value), target.dhriv_class.name, "DIAGONAL_LINK", 1)

        return ResolutionResult(found=False)


# ============================================================================
# MOTORE DI RICERCA SU LARGA SCALA TRI-ASSIALE (SERVER, CLUSTER E OGGETTI)
# ============================================================================

@dataclass
class ScaleNodeRecord:
    """Nodo server o entità distribuita indicizzata su larga scala nello spazio (H, V, D)."""
    node_id: str
    h_shard: int               # Coordinata Orizzontale (Cluster / Regione / Shard)
    v_layer: int               # Coordinata Verticale (0=Hardware/Binario .. N=Cloud/OOP)
    capabilities: Set[str]
    frequency_hz: float = 0.0
    health_score: float = 1.0
    payload: Any = None


class TriAxialLargeScaleSearch:
    """
    Motore di Ricerca su Larga Scala (stile ricerca server distribuita Orizzontale, Verticale e Diagonale):
    Indicizza server, microservizi, nodi binari e oggetti OOP su 4 indici simultanei O(1):
    - `_by_v_layer[v]`: per Ricerca Orizzontale pura (spazzata su tutti i server dello stesso livello);
    - `_by_h_shard[h]`: per Ricerca Verticale pura (discesa/risalita profonda su uno shard);
    - `_by_main_diag[h - v]` e `_by_anti_diag[h + v]`: per Ricerca Diagonale Iper-Scalare!
      Come la Regina negli scacchi o il routing diagonale nei torus-cluster HPC, taglia
      contemporaneamente più cluster E più livelli architetturali in un unico vettore!
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, ScaleNodeRecord] = {}
        self._by_v_layer: Dict[int, List[ScaleNodeRecord]] = {}
        self._by_h_shard: Dict[int, List[ScaleNodeRecord]] = {}
        self._by_main_diag: Dict[int, List[ScaleNodeRecord]] = {}  # costante k = h - v
        self._by_anti_diag: Dict[int, List[ScaleNodeRecord]] = {}  # costante k = h + v

    def register_node(
        self,
        node_id: str,
        h_shard: int,
        v_layer: int,
        capabilities: List[str],
        frequency_hz: float = 432.0,
        health_score: float = 1.0,
        payload: Any = None,
    ) -> ScaleNodeRecord:
        rec = ScaleNodeRecord(
            node_id=node_id,
            h_shard=int(h_shard),
            v_layer=int(v_layer),
            capabilities=set(capabilities),
            frequency_hz=float(frequency_hz),
            health_score=float(health_score),
            payload=payload,
        )
        self.nodes[node_id] = rec
        self._by_v_layer.setdefault(rec.v_layer, []).append(rec)
        self._by_h_shard.setdefault(rec.h_shard, []).append(rec)
        self._by_main_diag.setdefault(rec.h_shard - rec.v_layer, []).append(rec)
        self._by_anti_diag.setdefault(rec.h_shard + rec.v_layer, []).append(rec)
        return rec

    def search_horizontal(self, v_layer: int, capability: Optional[str] = None) -> List[Dict[str, Any]]:
        """Ricerca ORIZZONTALE: scansiona tutti i server/oggetti sullo stesso piano `v_layer`."""
        candidates = self._by_v_layer.get(int(v_layer), [])
        return self._filter_and_rank(candidates, capability, axis="HORIZONTAL")

    def search_vertical(self, h_shard: int, capability: Optional[str] = None) -> List[Dict[str, Any]]:
        """Ricerca VERTICALE: attraversa l'intero stack (Hardware -> Kernel -> OOP -> Cloud) sullo shard `h_shard`."""
        candidates = self._by_h_shard.get(int(h_shard), [])
        return self._filter_and_rank(candidates, capability, axis="VERTICAL")

    def search_diagonal(
        self,
        origin_h: int,
        origin_v: int,
        capability: Optional[str] = None,
        include_cross_diagonals: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Ricerca DIAGONALE SU LARGA SCALA:
        Taglia trasversalmente sia gli shard orizzontali sia gli strati verticali lungo
        le rette iperplanari `h - v = origin_h - origin_v` e `h + v = origin_h + origin_v`.
        Se non trova match esatti sulla diagonale primaria, espande a raggio diagonale crescente.
        """
        seen: Set[str] = set()
        candidates: List[ScaleNodeRecord] = []

        primary = self._by_main_diag.get(int(origin_h) - int(origin_v), [])
        secondary = self._by_anti_diag.get(int(origin_h) + int(origin_v), []) if include_cross_diagonals else []

        for rec in primary + secondary:
            if rec.node_id not in seen:
                seen.add(rec.node_id)
                candidates.append(rec)

        filtered = self._filter_and_rank(candidates, capability, axis="DIAGONAL", origin=(origin_h, origin_v))
        if filtered:
            return filtered

        # Fallback a onda diagonale espansiva (Chebyshev distance `max(|Δh|, |Δv|)`)
        all_matches = self._filter_and_rank(list(self.nodes.values()), capability, axis="DIAGONAL_WAVE", origin=(origin_h, origin_v))
        return all_matches

    def _filter_and_rank(
        self,
        records: List[ScaleNodeRecord],
        capability: Optional[str],
        axis: str,
        origin: Tuple[int, int] = (0, 0),
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in records:
            if capability and capability not in r.capabilities:
                continue
            dist = math.hypot(r.h_shard - origin[0], r.v_layer - origin[1])
            out.append({
                "node_id": r.node_id,
                "axis": axis,
                "h_shard": r.h_shard,
                "v_layer": r.v_layer,
                "capabilities": sorted(r.capabilities),
                "frequency_hz": r.frequency_hz,
                "health_score": r.health_score,
                "vector_distance": round(dist, 4),
            })
        out.sort(key=lambda item: (-item["health_score"], item["vector_distance"], item["node_id"]))
        return out
