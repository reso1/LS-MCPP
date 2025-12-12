from __future__ import annotations

from abc import abstractmethod
from itertools import product
from bisect import bisect
from enum import Enum
from typing import Dict, List, Set, Tuple
from collections import defaultdict

import networkx as nx
from networkx import Graph
import numpy as np
from scipy.special import softmax

from lsmcpp.graph import DecGraph
from lsmcpp.operator import Operator, GrowOP, DedupOP, ExcOP
from lsmcpp.benchmark.solution import Solution
from lsmcpp.utils import DuplicationRec


class SampleType(Enum):
    Random = 0
    RouletteWheel = 1


class PrioType(Enum):
    Random = 0
    CompositeHeur = 1
    SeparateHeur = 2


class PoolType(Enum):
    Edgewise = 0
    VertexEdgewise = 1


class Pool:

    def __init__(self, name:str, D:nx.Graph, R:list, dg:DecGraph, decgraphs:List[DecGraph], dup_rec:DuplicationRec, prio_type:PrioType) -> None:
        self.name = name
        self.prio_type = prio_type
        self.D, self.R = D, R
        self.k = len(self.R)
        self.decgraphs = decgraphs
        self.dup_rec = dup_rec
        self.dg = dg

        self.op_uid_set: Set[tuple] = set()
        self.pool: List[List[Operator]] = [[] for _ in range(self.k)]
        self.heurs: List[List[float]] = [[] for _ in range(self.k)]
        self.Tv_rec: List[Dict[Tuple, Set[Operator]]] = [defaultdict(set) for _ in range(self.k)]
        self.sizes: List[float] = [0 for _ in range(self.k)]
        self.total_size = 0

        if prio_type == PrioType.CompositeHeur:
            self.heur_val = self.composite_heur
        elif prio_type == PrioType.SeparateHeur:
            self.heur_val = self.separate_heur

    def add(self, op: Operator) -> None:
        if op.uid not in self.op_uid_set:
            self.op_uid_set.add(op.uid)
            self.pool[op.idx].append(op)
            self.heurs[op.idx].append(op.heur)
            self.Tv_rec[op.idx][self.dg.undecomp(op.V[0])].add(op)
            self.sizes[op.idx] += 1
            self.total_size += 1

    def pop(self, op_idx: int, pi_idx:int) -> Operator:
        op = self.pool[pi_idx].pop(op_idx)
        self.op_uid_set.remove(op.uid)
        self.heurs[pi_idx].pop(op_idx)
        self.Tv_rec[pi_idx][self.dg.undecomp(op.V[0])].remove(op)
        self.sizes[pi_idx] -= 1
        self.total_size -= 1

        return op
    
    def remove(self, op:Operator) -> Operator:
        # used only for update func
        op_idx = self.pool[op.idx].index(op)
        op = self.pool[op.idx].pop(op_idx)
        self.op_uid_set.remove(op.uid)
        self.heurs[op.idx].pop(op_idx)
        self.sizes[op.idx] -= 1
        self.total_size -= 1

    def sample(self, rng:np.random.RandomState, idx_masks:list=None) -> Operator|None:        
        if idx_masks is None:
            idx_masks = list(range(self.k))

        selected_pi_inds = [i for i in idx_masks if self.sizes[i] != 0]
        if not selected_pi_inds:
            return None

        op_idx_cumsum = np.cumsum([self.sizes[i] for i in selected_pi_inds])
        self.selected_size = op_idx_cumsum[-1]

        if self.prio_type == PrioType.Random:
            prob = (1/self.selected_size) * np.ones(self.selected_size)
        elif self.prio_type == PrioType.CompositeHeur or self.prio_type == PrioType.SeparateHeur:
            prob = softmax(np.concatenate([self.heurs[i] for i in idx_masks]))
        
        idx = int(rng.choice(self.selected_size, size=1, p=prob))
        tmp_idx = bisect(op_idx_cumsum, idx)
        op_idx = idx - (op_idx_cumsum[tmp_idx-1] if tmp_idx > 0 else 0)
        return self.pop(op_idx, selected_pi_inds[tmp_idx])

    def clear(self) -> None:
        self.op_uid_set.clear()
        self.total_size = 0
        for idx in range(self.k):
            self.pool[idx].clear()
            self.heurs[idx].clear()
            self.Tv_rec[idx].clear()
            self.sizes[idx] = 0

    def is_empty(self, idx_masks:list) -> bool:
        for idx in idx_masks:
            if self.sizes[idx] != 0:
                return False
        return True
    
    def get_non_empty_pool_inds(self, idx_masks:list) -> List[int]:
        return [idx for idx in idx_masks if self.sizes[idx] != 0]

    @abstractmethod
    def init(self, sol:Solution, Og:Pool|None=None) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw(self, ax) -> None:
        raise NotImplementedError

    @abstractmethod
    def heur_val(self, op:Operator, sol:Solution) -> float:
        raise NotImplementedError

    @staticmethod
    def sorted_V(u:tuple, v:tuple) -> List[tuple, tuple]:
        if u[1] > v[1]:
            u, v = v, u
        if u[0] > v[0]:
            u, v = v, u
        return [u, v]


""" pools containing edge-wise and vertex-wise operators"""

class GrowPool(Pool):

    def __init__(self, D:nx.Graph, R:list, dg:DecGraph, decgraphs:List[DecGraph], dup_rec:DuplicationRec, prio_type:PrioType) -> None:
        super().__init__("GrowPool", D, R, dg, decgraphs, dup_rec, prio_type)

    def init(self, sol:Solution) -> None:
        self.clear()
        # initialize the set of boundary verts and all operators
        for idx in range(self.k):
            self.add_ops(idx, sol, self.decgraphs[idx].V)

    def update(self, op:Operator, sol:Solution, T:nx.Graph) -> list:
        Tv = self.dg.undecomp(op.V[0])
        ret = []
        for idx in range(self.k):
            # remove all involving operators
            V_mod = set()
            for Tu in list(self.dg.T.neighbors(Tv)) + [Tv]:
                if Tu in T.nodes:
                    V_mod = V_mod.union([dv for dv in self.dg.decomp(Tu) if dv is not None])
                for op_Tu in self.Tv_rec[idx][Tu]:
                    self.remove(op_Tu)
                self.Tv_rec[idx][Tu].clear()

            # add operators
            added = self.add_ops(idx, sol, V_mod)
            ret.append(added)

        return ret
    
    def add_ops(self, idx:int, sol:Solution, V_mod:set) -> list:
        ret = []
        V_mod_ngbs = set()
        for v in V_mod:
            V_mod_ngbs = V_mod_ngbs.union([v] + list(self.D.neighbors(v)))
            
        B = V_mod_ngbs.intersection(self.decgraphs[idx].B)

        while len(B) > 0:
            v, paired = B.pop(), False
            if v in self.decgraphs[idx].V:
                continue
            pairings = EdgewiseGrowPool.get_pairing(v)
            for u in pairings:
                op = None
                if u in self.dg.V and u not in self.decgraphs[idx].V:
                    if EdgewiseGrowPool.has_parallel_edge(v, u, self.decgraphs[idx].V):
                        op = GrowOP(idx, 0, Pool.sorted_V(v, u))
                    elif self.has_orthogonal_single_vert(v, u, self.decgraphs[idx].V):
                        op = GrowOP(idx, 0, Pool.sorted_V(v, u))
                if op:
                    op.heur = self.heur_val(op, sol)
                    self.add(op)
                    ret.append(op)
                    paired = True
            
            if not paired and (pairings[0] not in self.dg.V or pairings[1] not in self.dg.V):
                op = GrowOP(idx, 0, [v])
                op.heur = self.heur_val(op, sol)
                self.add(op)
                ret.append(op)
            
        return ret

    def draw(self, axs) -> None:
        for pool in self.pool:
            for op in pool:
                idx = op.idx
                if len(op.V) == 1:
                    axs[idx].plot(op.V[0][0], op.V[0][1], "+b")
                else:
                    u, v = op.V
                    axs[idx].plot([u[0], v[0]], [u[1], v[1]], "+b-")

    def composite_heur(self, op:Operator, sol:Solution) -> float:
        return - sol.costs[op.idx] * len(sol.costs) - sum([self.dup_rec.cnts[v] for v in op.V])/len(op.V)

    def separate_heur(self, op:Operator, sol:Solution) -> float:
        return - sum([self.dup_rec.cnts[v] for v in op.V])/len(op.V)

    def has_orthogonal_single_vert(self, u:tuple, v:tuple, V:set) -> bool:
        if u[0] == v[0]:
            return ( (u[0]+0.5, u[1]) in V and (v[0]+0.5, v[1]) not in self.dg.V ) or \
                   ( (u[0]-0.5, u[1]) in V and (v[0]-0.5, v[1]) not in self.dg.V ) or \
                   ( (v[0]+0.5, v[1]) in V and (u[0]+0.5, u[1]) not in self.dg.V ) or \
                   ( (v[0]-0.5, v[1]) in V and (u[0]-0.5, u[1]) not in self.dg.V )
        elif u[1] == v[1]:
            return ( (u[0], u[1]+0.5) in V and (v[0], v[1]+0.5) not in self.dg.V ) or \
                   ( (u[0], u[1]-0.5) in V and (v[0], v[1]-0.5) not in self.dg.V ) or \
                   ( (v[0], v[1]+0.5) in V and (u[0], u[1]+0.5) not in self.dg.V ) or \
                   ( (v[0], v[1]-0.5) in V and (u[0], u[1]-0.5) not in self.dg.V )
        else:
            assert True == False # should not reach here


class DeDupPool(Pool):

    def __init__(self, D:nx.Graph, R:list, dg:DecGraph, decgraphs:List[DecGraph], dup_rec:DuplicationRec, R_D:list, prio_type:PrioType) -> None:
        super().__init__("DedupPool", D, R, dg, decgraphs, dup_rec, prio_type)
        self.R_D = R_D

    def init(self, sol:Solution) -> None:
        self.clear()
        for idx, dg in enumerate(self.decgraphs):
            self.add_ops(idx, set.intersection(dg.V, self.dup_rec.dup_set), sol)

    def update(self, op:Operator, sol:Solution) -> None:
        Tv = self.dg.undecomp(op.V[0])
        for idx in range(self.k):
            # remove all involving operators
            V_mod = set()
            for Tu in list(self.dg.T.neighbors(Tv)) + [Tv]:
                if Tu in self.decgraphs[idx].T.nodes:
                    V_mod = V_mod.union([dv for dv in self.decgraphs[idx].decomp(Tu)])
                for op_Tu in self.Tv_rec[idx][Tu]:
                    self.remove(op_Tu)
                self.Tv_rec[idx][Tu].clear()
            # add operators
            self.add_ops(idx, set.intersection(V_mod, self.dup_rec.dup_set), sol)

    def add_ops(self, idx:int, duped:set, sol:Solution) -> None:

        while len(duped) > 0:
            v, paired = duped.pop(), False
            pairings = EdgewiseGrowPool.get_pairing(v)
            for u in pairings:
                if v != self.R_D[idx] and u and u != self.R_D[idx] and u in duped \
                   and DeDupPool.is_valid(self.decgraphs[idx], u, v):
                    op = DedupOP(idx, 0, Pool.sorted_V(v, u))
                    op.heur = self.heur_val(op, sol)
                    self.add(op)
                    paired = True
            if not paired and v != self.R_D[idx] \
               and (pairings[0] not in self.dg.V or pairings[1] not in self.dg.V):
                op = DedupOP(idx, 0, [v])
                op.heur = self.heur_val(op, sol)
                self.add(op)

    @staticmethod
    def is_valid(dg:DecGraph, u:tuple, v:tuple) -> bool:
        Tv = dg.undecomp(v)
        if Tv is None:
            return False
        
        if not dg.T.nodes[Tv]["complete"]:
            return True

        Tv_bot = dg.Tv_bot(u, v, Tv)
        if Tv_bot is None:
            return False
        
        if Tv_bot == dg.Tv_ngb(Tv, 0):                                                          #  x | x
            Tv_left, Tv_right, Tv_top = dg.Tv_ngb(Tv, 1), dg.Tv_ngb(Tv, 3), dg.Tv_ngb(Tv, 2)    #  o | o

        if Tv_bot == dg.Tv_ngb(Tv, 1):                                                          #  x | o
            Tv_left, Tv_right, Tv_top = dg.Tv_ngb(Tv, 0), dg.Tv_ngb(Tv, 2), dg.Tv_ngb(Tv, 3)    #  x | o

        if Tv_bot == dg.Tv_ngb(Tv, 2):                                                          #  o | o
            Tv_left, Tv_right, Tv_top = dg.Tv_ngb(Tv, 1), dg.Tv_ngb(Tv, 3), dg.Tv_ngb(Tv, 0)    #  x | x

        if Tv_bot == dg.Tv_ngb(Tv, 3):                                                          #  o | x
            Tv_left, Tv_right, Tv_top = dg.Tv_ngb(Tv, 0), dg.Tv_ngb(Tv, 2), dg.Tv_ngb(Tv, 1)    #  o | x
        
        valid = not Tv_top and Tv_bot and dg.T.nodes[Tv_bot]["complete"]
        if valid and Tv_left:
            common_ngb = dg.common_ngb(Tv_left, Tv_bot)
            valid = dg.T.nodes[Tv_left]["complete"] and common_ngb and dg.T.nodes[common_ngb]["complete"]
        if valid and Tv_right:
            common_ngb = dg.common_ngb(Tv_right, Tv_bot)
            valid = dg.T.nodes[Tv_right]["complete"] and common_ngb and dg.T.nodes[common_ngb]["complete"]

        return valid 

    def draw(self, axs) -> None:
        for pool in self.pool:
            for op in pool:
                idx = op.idx
                if len(op.V) == 1:
                    axs[idx].plot(op.V[0][0], op.V[0][1], "xr")
                else:
                    u, v = op.V
                    axs[idx].plot([u[0], v[0]], [u[1], v[1]], "xr-")

    def composite_heur(self, op: Operator, sol: Solution) -> float:
        return sol.costs[op.idx] * len(sol.costs) + sum([self.dup_rec.cnts[v] for v in op.V])/len(op.V)
    
    def separate_heur(self, op: Operator, sol: Solution) -> float:
        return sum([self.dup_rec.cnts[v] for v in op.V])/len(op.V)


class ExcPool(Pool):
    def __init__(self, D:Graph, R:List, dg:DecGraph, decgraphs:List[DecGraph], dup_rec:DuplicationRec, R_D:list, prio_type: PrioType) -> None:
        self.composite_heur = self.separate_heur = self.heur_val
        super().__init__("ExcPool", D, R, dg, decgraphs, dup_rec, prio_type)
        self.R_D = R_D
    
    def init(self, sol:Solution, Og:GrowPool) -> None:
        self.clear()
        for i in range(self.k):
            for og in Og.pool[i]:
                if len(og.V) == 1:
                    v = og.V[0]
                    for idx, dg in enumerate(self.decgraphs):
                        if idx != og.idx and v in dg.V and v != self.R_D[idx]:
                            op = ExcOP(og.idx, idx, 0, og.V)
                            op.heur = self.heur_val(op, sol)
                            self.add(op)
                else:
                    u, v = og.V[0], og.V[1]
                    for idx, dg in enumerate(self.decgraphs):
                        if idx != og.idx and u in dg.V and v in dg.V and u != self.R_D[idx] \
                        and v != self.R_D[idx] and EdgewiseDeDupPool.is_valid(dg, u, v):
                            op = ExcOP(og.idx, idx, 0, og.V)
                            op.heur = self.heur_val(op, sol)
                            self.add(op)
    
    def update(self, op:Operator, sol:Solution, added:List[GrowOP]) -> None:
        Tv = self.dg.undecomp(op.V[0])
        for idx in range(self.k):
            # remove all involving operators
            for Tu in list(self.dg.T.neighbors(Tv)) + [Tv]:
                for op_Tu in self.Tv_rec[idx][Tu]:
                    self.remove(op_Tu)
                self.Tv_rec[idx][Tu].clear()
            # add operators
            for og in added[idx]:
                if len(og.V) == 1:
                    v = og.V[0]
                    for idx, dg in enumerate(self.decgraphs):
                        if idx != og.idx and v in dg.V and v != self.R_D[idx]:
                            op = ExcOP(og.idx, idx, 0, og.V)
                            op.heur = self.heur_val(op, sol)
                            self.add(op)
                else:
                    u, v = og.V[0], og.V[1]
                    for idx, dg in enumerate(self.decgraphs):
                        if idx != og.idx and u in dg.V and v in dg.V and u != self.R_D[idx] \
                        and v != self.R_D[idx] and EdgewiseDeDupPool.is_valid(dg, u, v):
                            op = ExcOP(og.idx, idx, 0, og.V)
                            op.heur = self.heur_val(op, sol)
                            self.add(op)
    
    def draw(self, axs) -> None:
        grays = np.linspace(0, 0.8, self.size)
        for idx in np.argsort(self.heurs):
            op = self.pool[idx]
            if len(op.V) == 1:
                axs[op.idx].plot(op.V[0][0], op.V[0][1], "+-", color=str(grays[idx]))
                axs[op.op_rmv_idx].plot(op.V[0][0], op.V[0][1], "x-", color=str(grays[idx]))
            else:
                u, v = op.V
                axs[op.idx].plot([u[0], v[0]], [u[1], v[1]], "+-", color=str(grays[idx]))
                axs[op.op_rmv_idx].plot([u[0], v[0]], [u[1], v[1]], "x-", color=str(grays[idx]))

    def heur_val(self, op:Operator, sol:Solution) -> float:
        return sol.costs[op.idx] - sol.costs[op.op_rmv_idx]


""" the edgewise pools from AAAI'24 """

class EdgewiseGrowPool(GrowPool):
    
    def __init__(
        self, D: nx.Graph, R: List, dg: DecGraph, 
        decgraphs: List[DecGraph], dup_rec: DuplicationRec, 
        prio_type: PrioType
    ) -> None:
        super().__init__(D, R, dg, decgraphs, dup_rec, prio_type)
        self.name = "EdgewiseGrowPool"
    
    def add_ops(self, idx:int, sol:Solution, V_mod:set) -> list:
        ret = []
        V_mod_ngbs = set()
        for v in V_mod:
            V_mod_ngbs = V_mod_ngbs.union([v] + list(self.D.neighbors(v)))
        B = V_mod_ngbs.intersection(self.decgraphs[idx].B)

        while len(B) > 0:
            v = B.pop()
            for pairing in EdgewiseGrowPool.get_pairing(v):
                if pairing in B and EdgewiseGrowPool.has_parallel_edge(v, pairing, self.decgraphs[idx].V):
                    op = GrowOP(idx, 0, Pool.sorted_V(v, pairing))
                    op.heur = self.heur_val(op, sol)
                    self.add(op)
                    ret.append(op)

        return ret

    @staticmethod
    def get_pairing(v:tuple) -> List[tuple]:
        u0 = (v[0]+0.5, v[1]) if v[0] % 1 > 0.5 else (v[0]-0.5, v[1])
        u1 = (v[0], v[1]+0.5) if v[1] % 1 > 0.5 else (v[0], v[1]-0.5)
        return [u0, u1]

    @staticmethod
    def has_parallel_edge(u:tuple, v:tuple, V:set) -> bool:
        if u[0] == v[0]:
            return ( (u[0]+0.5, u[1]) in V and (u[0]+0.5, v[1]) in V ) or ( (u[0]-0.5, u[1]) in V and (u[0]-0.5, v[1]) in V )
        else:
            return ( (u[0], u[1]+0.5) in V and (v[0], u[1]+0.5) in V ) or ( (u[0], u[1]-0.5) in V and (v[0], u[1]-0.5) in V )


class EdgewiseDeDupPool(DeDupPool):
    
    def __init__(self, D:nx.Graph, R:list, dg:DecGraph, decgraphs:List[DecGraph], dup_rec:DuplicationRec, R_D:list, prio_type:PrioType) -> None:
        super().__init__(D, R, dg, decgraphs, dup_rec, R_D, prio_type)
        self.name = "EdgewiseDedupPool"

    def add_ops(self, idx:int, duped:set, sol:Solution) -> None:
        while len(duped) > 0:
            v = duped.pop()
            for u in EdgewiseGrowPool.get_pairing(v):
                if v != self.R_D[idx] and u and u != self.R_D[idx] and u in duped \
                   and DeDupPool.is_valid(self.decgraphs[idx], u, v):
                    op = DedupOP(idx, 0, Pool.sorted_V(v, u))
                    op.heur = self.heur_val(op, sol)
                    self.add(op)


class EdgewiseExcPool(ExcPool):

    def __init__(self, D:nx.Graph, R:list, dg:DecGraph, decgraphs:List[DecGraph], dup_rec:DuplicationRec, R_D:list, prio_type:PrioType) -> None:
        super().__init__(D, R, dg, decgraphs, dup_rec, R_D, prio_type)
        self.name = "EdgewiseExcPool"

    def init(self, sol:Solution, Og:GrowPool) -> None:
        self.clear()
        for i in range(self.k):
            for og in Og.pool[i]:
                u, v = og.V[0], og.V[1]
                for idx, dg in enumerate(self.decgraphs):
                    if idx != og.idx and u in dg.V and v in dg.V and u != self.R_D[idx] \
                       and v != self.R_D[idx] and EdgewiseDeDupPool.is_valid(dg, u, v):
                        op = ExcOP(og.idx, idx, 0, og.V)
                        op.heur = self.heur_val(op, sol)
                        self.add(op)

    def update(self, op:Operator, sol:Solution, added:list) -> None:
        Tv = self.dg.undecomp(op.V[0])

        for idx in range(self.k):
            # remove all involving operators
            for Tu in list(self.dg.T.neighbors(Tv)) + [Tv]:
                for op_Tu in self.Tv_rec[idx][Tu]:
                    self.remove(op_Tu)
                self.Tv_rec[idx][Tu].clear()
            # add operators
            for og in added[idx]:
                u, v = og.V[0], og.V[1]
                for idx, dg in enumerate(self.decgraphs):
                    if idx != og.idx and u in dg.V and v in dg.V and u != self.R_D[idx] \
                       and v != self.R_D[idx] and EdgewiseDeDupPool.is_valid(dg, u, v):
                        op = ExcOP(og.idx, idx, 0, og.V)
                        op.heur = self.heur_val(op, sol)
                        self.add(op)

