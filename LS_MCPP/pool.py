from __future__ import annotations

from abc import abstractmethod
from itertools import product
from bisect import bisect
from enum import Enum
from typing import Dict, List, Set, Tuple
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.special import softmax

from LS_MCPP.graph import DecGraph
from LS_MCPP.operator import Operator, GrowOP, DedupOP, ExcOP
from LS_MCPP.solution import Solution
from LS_MCPP.utils import DuplicationRec


class SampleType(Enum):
    Random = 0
    RouletteWheel = 1


class PrioType(Enum):
    Random = 0
    Heur = 1


class Pool:

    def __init__(self, name:str, D:nx.Graph, R:list, decgraphs:List[DecGraph], dup_rec:DuplicationRec, prio_type:PrioType) -> None:
        self.name = name
        self.prio_type = prio_type
        self.D, self.R = D, R
        self.k = len(self.R)
        self.decgraphs = decgraphs
        self.dup_rec = dup_rec

        self.op_uid_set: Set[tuple] = set()
        self.pool: List[List[Operator]] = [[] for _ in range(self.k)]
        self.heurs: List[List[float]] = [[] for _ in range(self.k)]
        self.Tv_rec: List[Dict[Tuple, Set[Operator]]] = [defaultdict(set) for _ in range(self.k)]
        self.sizes: List[float] = [0 for _ in range(self.k)]
        self.total_size = 0

    def add(self, op: Operator) -> None:
        if op.uid not in self.op_uid_set:
            self.op_uid_set.add(op.uid)
            self.pool[op.idx].append(op)
            self.heurs[op.idx].append(op.heur)
            self.Tv_rec[op.idx][DecGraph.undecomp(op.V[0])].add(op)
            self.sizes[op.idx] += 1
            self.total_size += 1

    def pop(self, op_idx: int, pi_idx:int) -> Operator:
        op = self.pool[pi_idx].pop(op_idx)
        self.op_uid_set.remove(op.uid)
        self.heurs[pi_idx].pop(op_idx)
        self.Tv_rec[pi_idx][DecGraph.undecomp(op.V[0])].remove(op)
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

    def sample(self, idx_masks:list=None) -> Operator:        
        if idx_masks is None:
            idx_masks = list(range(self.k))

        selected_pi_inds = [i for i in idx_masks if self.sizes[i] != 0]
        op_idx_cumsum = np.cumsum([self.sizes[i] for i in selected_pi_inds])
        self.selected_size = op_idx_cumsum[-1]

        if self.prio_type == PrioType.Random:
            prob = (1/self.selected_size) * np.ones(self.selected_size)
        elif self.prio_type == PrioType.Heur:
            prob = softmax(-np.concatenate([self.heurs[i] for i in idx_masks]))
        
        idx = int(np.random.choice(self.selected_size, size=1, p=prob))
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

    @abstractmethod
    def init(self, sol:Solution, Og:Pool|None=None) -> None:
        pass

    @abstractmethod
    def draw(self, ax) -> None:
        pass

    @staticmethod
    def sorted_V(u:tuple, v:tuple) -> List[tuple, tuple]:
        if u[1] > v[1]:
            u, v = v, u
        if u[0] > v[0]:
            u, v = v, u
        return [u, v]


class GrowPool(Pool):

    def __init__(self, D:nx.Graph, R:list, decgraphs:List[DecGraph], dup_rec:DuplicationRec, prio_type:PrioType=PrioType.Heur) -> None:
        super().__init__("GrowPool", D, R, decgraphs, dup_rec, prio_type)

    def init(self, sol:Solution) -> None:
        self.clear()
        for idx, dg in enumerate(self.decgraphs):
            B, heurs = set(), {}
            for v in dg.V:
                for ngb in self.D.neighbors(v):
                    if ngb not in self.decgraphs[idx].V:
                        B.add(ngb)
                        heurs[ngb] = sol.costs[idx] * self.k + self.dup_rec.cnts[ngb]

            self._add_ops(idx, B, heurs)
            
    def update(self, op:Operator, sol:Solution, T:nx.Graph) -> list:
        Tv = DecGraph.undecomp(op.V[0])
        offsets = list(product([0, 1, -1], [0, 1, -1]))

        ret = []
        for idx in range(self.k):
            # update operators
            V_mod = set()
            for inc in offsets:
                Tu = (Tv[0] + inc[0], Tv[1] + inc[1])
                if Tu in T.nodes:
                    V_mod = V_mod.union([dv for dv in DecGraph.decomp(Tu) if dv in self.D.nodes])
                for op_Tu in self.Tv_rec[idx][Tu]:
                    self.remove(op_Tu)
                self.Tv_rec[idx][Tu].clear()
            
            B, heurs = set(), {}
            for v in V_mod:
                for ngb in self.D.neighbors(v):
                    if ngb not in self.decgraphs[idx].V:
                        B.add(ngb)
                        heurs[ngb] = sol.costs[idx] * self.k + self.dup_rec.cnts[ngb]

            added = self._add_ops(idx, B, heurs)
            ret.append(added)
        
        return ret

    def draw(self, axs) -> None:
        grays = np.linspace(0, 0.8, self.size)
        for idx in np.argsort(self.heurs):
            op = self.pool[idx]
            u, v = op.V
            axs[op.idx].plot([u[0], v[0]], [u[1], v[1]], "+-", color=str(grays[idx]))

    def _add_ops(self, idx:int, B:set, heurs:dict) -> list:
        ret = []
        while len(B) > 0:
            v = B.pop()
            pairing = GrowPool.get_pairing(v, B, self.decgraphs[idx].V, heurs)
            if pairing:
                op = GrowOP(idx, (heurs[v]+heurs[pairing])/2, Pool.sorted_V(v, pairing))
                self.add(op)
                ret.append(op)
                B.remove(pairing)

        return ret

    @staticmethod
    def get_pairing(v:tuple, B:set, V:set, heurs:dict) -> tuple|None:
        lr = v[0] % 1 > 0.5
        tb = v[1] % 1 > 0.5 

        if lr and tb:
            u0 = (v[0]+0.5, v[1])
            u1 = (v[0], v[1]+0.5)
        elif not lr and tb:
            u0 = (v[0]-0.5, v[1])
            u1 = (v[0], v[1]+0.5)
        elif lr and not tb:
            u0 = (v[0]+0.5, v[1])
            u1 = (v[0], v[1]-0.5)
        elif not lr and not tb:
            u0 = (v[0]-0.5, v[1])
            u1 = (v[0], v[1]-0.5)

        u0_valid = (u0 in B) and GrowPool.has_parallel_edge(v, u0, V)
        u1_valid = (u1 in B) and GrowPool.has_parallel_edge(v, u1, V)
        
        if u0_valid and u1_valid:
            return u0 if heurs[u0] < heurs[u1] else u1
        elif u0_valid and not u1_valid:
            return u0
        elif not u0_valid and u1_valid:
            return u1
        else:
            return None

    @staticmethod
    def has_parallel_edge(u:tuple, v:tuple, V:set) -> bool:
        if u[0] == v[0]:
            return ( (u[0]+0.5, u[1]) in V and (u[0]+0.5, v[1]) in V ) or ( (u[0]-0.5, u[1]) in V and (u[0]-0.5, v[1]) in V )
        else:
            return ( (u[0], u[1]+0.5) in V and (v[0], u[1]+0.5) in V ) or ( (u[0], u[1]-0.5) in V and (v[0], u[1]-0.5) in V )


class DeDupPool(Pool):

    def __init__(self, D:nx.Graph, R:list, decgraphs:List[DecGraph], dup_rec:DuplicationRec, R_D:list, prio_type:PrioType=PrioType.Heur) -> None:
        super().__init__("DedupPool", D, R, decgraphs, dup_rec, prio_type)
        self.R_D = R_D

    def init(self, sol:Solution) -> None:
        self.clear()
        for idx, dg in enumerate(self.decgraphs):
            self._add_ops(idx, set.intersection(dg.V, self.dup_rec.dup_set), sol)

    def update(self, op:Operator, sol:Solution) -> None:
        Tv = DecGraph.undecomp(op.V[0])
        offsets = list(product([0, 1, -1], [0, 1, -1]))

        for idx in range(self.k):
            # update operators
            V_mod = set()
            for inc in offsets:
                Tu = (Tv[0] + inc[0], Tv[1] + inc[1])
                if Tu in self.decgraphs[idx].T.nodes:
                    V_mod = V_mod.union([dv for dv in DecGraph.decomp(Tu) if dv in self.decgraphs[idx].V])
                for op_Tu in self.Tv_rec[idx][Tu]:
                    self.remove(op_Tu)
                self.Tv_rec[idx][Tu].clear()

            self._add_ops(idx, set.intersection(V_mod, self.dup_rec.dup_set), sol)

    def _add_ops(self, idx:int, duped:set, sol:Solution) -> None:
        while len(duped) > 0:
            v = duped.pop()
            pairing = DeDupPool.get_pairing(v, duped, self.decgraphs[idx], self.dup_rec)
            if v != self.R_D[idx] and pairing and pairing != self.R_D[idx]:
                heur = - sol.costs[idx] * self.k - (self.dup_rec.cnts[v] + self.dup_rec.cnts[pairing]) / 2
                op = DedupOP(idx, heur, Pool.sorted_V(v, pairing))
                self.add(op)
                duped.remove(pairing)

    def draw(self, axs) -> None:
        grays = np.linspace(0, 0.8, self.size)
        for idx in np.argsort(self.heurs):
            op = self.pool[idx]
            u, v = op.V
            axs[op.idx].plot([u[0], v[0]], [u[1], v[1]], "x-", color=str(grays[idx]))

    @staticmethod
    def get_pairing(v:tuple, B:set, dg:DecGraph, dup_rec:DuplicationRec) -> tuple|None:
        lr = v[0] % 1 > 0.5
        tb = v[1] % 1 > 0.5
        if lr and tb:
            cond_list_a = DeDupPool.get_cond_list((v[0]+0.5, v[1]), v)
            cond_list_b = DeDupPool.get_cond_list((v[0], v[1]+0.5), v)
        elif not lr and tb:
            cond_list_a = DeDupPool.get_cond_list((v[0]-0.5, v[1]), v)
            cond_list_b = DeDupPool.get_cond_list((v[0], v[1]+0.5), v)
        elif lr and not tb:
            cond_list_a = DeDupPool.get_cond_list((v[0]+0.5, v[1]), v)
            cond_list_b = DeDupPool.get_cond_list((v[0], v[1]-0.5), v)
        elif not lr and not tb:
            cond_list_a = DeDupPool.get_cond_list((v[0]-0.5, v[1]), v)
            cond_list_b = DeDupPool.get_cond_list((v[0], v[1]-0.5), v)

        pairing_a_valid = DeDupPool.is_duped_pairing_valid(cond_list_a, B, dg)
        pairing_b_valid = DeDupPool.is_duped_pairing_valid(cond_list_b, B, dg)

        if pairing_a_valid and pairing_b_valid:
            return cond_list_a[0] if dup_rec.cnts[cond_list_a[0]] > dup_rec.cnts[cond_list_b[0]] else cond_list_b[0]
        elif pairing_a_valid and not pairing_b_valid:
            return cond_list_a[0]
        elif not pairing_a_valid and pairing_b_valid:
            return  cond_list_b[0]
        else:
            return None

    @staticmethod
    def get_cond_list(u:tuple, v:tuple) -> list:
        Tv = DecGraph.undecomp(v)
        Tv_perp = DecGraph.Tv_perp(u, v, Tv)

        if Tv_perp == DecGraph.Tv_ngb(Tv, 0):
            return [u, Tv, Tv_perp,                                     #  x | x
                    DecGraph.Tv_ngb(Tv, 1), (Tv[0] + 1, Tv[1] - 1),     #  o | o
                    DecGraph.Tv_ngb(Tv, 3), (Tv[0] - 1, Tv[1] - 1),
                    DecGraph.Tv_ngb(Tv, 2)]

        if Tv_perp == DecGraph.Tv_ngb(Tv, 1):
            return [u, Tv, Tv_perp,                                     #  x | o
                    DecGraph.Tv_ngb(Tv, 0), (Tv[0] + 1, Tv[1] - 1),     #  x | o
                    DecGraph.Tv_ngb(Tv, 2), (Tv[0] + 1, Tv[1] + 1),
                    DecGraph.Tv_ngb(Tv, 3)]

        if Tv_perp == DecGraph.Tv_ngb(Tv, 2):
            return [u, Tv, Tv_perp,                                     #  o | o
                    DecGraph.Tv_ngb(Tv, 1), (Tv[0] + 1, Tv[1] + 1),     #  x | x
                    DecGraph.Tv_ngb(Tv, 3), (Tv[0] - 1, Tv[1] + 1),
                    DecGraph.Tv_ngb(Tv, 0)]

        if Tv_perp == DecGraph.Tv_ngb(Tv, 3):
            return [u, Tv, Tv_perp,                                     #  o | x
                    DecGraph.Tv_ngb(Tv, 0), (Tv[0] - 1, Tv[1] - 1),     #  o | x 
                    DecGraph.Tv_ngb(Tv, 2), (Tv[0] - 1, Tv[1] + 1),
                    DecGraph.Tv_ngb(Tv, 1)]

    @staticmethod
    def is_duped_pairing_valid(cond_list:list, B:set, dg:DecGraph) -> bool:
        if not dg.T.nodes[cond_list[1]]["complete"]:
            pairing_valid = cond_list[0] in B
        else:
            pairing_valid = (cond_list[0] in B) and (cond_list[2] in dg.T.nodes and dg.T.nodes[cond_list[2]]["complete"]) and ( cond_list[7] not in dg.T.nodes)
            if pairing_valid and cond_list[3] in dg.T.nodes:
                pairing_valid = dg.T.nodes[cond_list[3]]["complete"] and cond_list[4] in dg.T.nodes and dg.T.nodes[cond_list[4]]["complete"]
            if pairing_valid and cond_list[5] in dg.T.nodes:
                pairing_valid = dg.T.nodes[cond_list[5]]["complete"] and cond_list[6] in dg.T.nodes and dg.T.nodes[cond_list[6]]["complete"]

        return pairing_valid


class ExcPool(Pool):

    def __init__(self, D:nx.Graph, R:list, decgraphs:List[DecGraph], dup_rec:DuplicationRec, R_D:list, prio_type:PrioType=PrioType.Heur) -> None:
        super().__init__("ExcPool", D, R, decgraphs, dup_rec, prio_type)
        self.R_D = R_D

    def init(self, sol:Solution, Og:GrowPool) -> None:
        self.clear()
        for i in range(self.k):
            for og in Og.pool[i]:
                u, v = og.V[0], og.V[1]
                cond_list = DeDupPool.get_cond_list(u, v)
                for idx, dg in enumerate(self.decgraphs):
                    if idx != og.idx and v in dg.V and u != self.R_D[idx] and v != self.R_D[idx] and \
                       DeDupPool.is_duped_pairing_valid(cond_list, set.intersection(dg.V, og.V), dg):
                        heur = sol.costs[og.idx] - sol.costs[idx]
                        op = ExcOP(og.idx, idx, heur, og.V)
                        self.add(op)

    def update(self, op:Operator, sol:Solution, added:list) -> None:
        Tv = DecGraph.undecomp(op.V[0])
        offsets = list(product([0, 1, -1], [0, 1, -1]))

        for idx in range(self.k):
            # update operators
            for inc in offsets:
                Tu = (Tv[0] + inc[0], Tv[1] + inc[1])
                for op_Tu in self.Tv_rec[idx][Tu]:
                    self.remove(op_Tu)
                self.Tv_rec[idx][Tu].clear()
            
            for og in added[idx]:
                u, v = og.V[0], og.V[1]
                cond_list = DeDupPool.get_cond_list(u, v)
                for idx, dg in enumerate(self.decgraphs):
                    if idx != og.idx and v in dg.V and u != self.R_D[idx] and v != self.R_D[idx] and \
                       DeDupPool.is_duped_pairing_valid(cond_list, set.intersection(dg.V, og.V), dg):
                        heur = sol.costs[og.idx] - sol.costs[idx]
                        op = ExcOP(og.idx, idx, heur, og.V)
                        self.add(op)

    def draw(self, axs) -> None:
        grays = np.linspace(0, 0.8, self.size)
        for idx in np.argsort(self.heurs):
            op = self.pool[idx]
            u, v = op.V
            axs[op.idx].plot([u[0], v[0]], [u[1], v[1]], "+-", color=str(grays[idx]))
            axs[op.op_rmv_idx].plot([u[0], v[0]], [u[1], v[1]], "x-", color=str(grays[idx]))
