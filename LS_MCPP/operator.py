from __future__ import annotations

from abc import abstractmethod
from typing import List, Tuple

import networkx as nx

from LS_MCPP.estc import ExtSTCPlanner
from LS_MCPP.graph import DecGraph
from LS_MCPP.solution import Solution
from LS_MCPP.utils import DuplicationRec


class Operator:

    def __init__(self, idx:int, heur:float, V:list) -> None:
        self.idx, self.heur, self.V = idx, heur, V
        self._eval_res = None
    
    @property
    def pid(self) -> tuple:
        return (self.idx, )
    
    @property
    def uid(self) -> int:
        return (self.pid, *self.V).__hash__()

    def __hash__(self) -> int:
        return self.uid
    
    @abstractmethod
    def eval(self, R:list, decgraphs:List[DecGraph], D:nx.Graph, sol:Solution) -> Tuple[float, float]:
        pass

    @abstractmethod
    def apply(self, R:list, decgraphs:List[DecGraph], D:nx.Graph, dup_rec:DuplicationRec, sol:Solution) -> str:
        pass


class GrowOP(Operator):

    def __init__(self, idx:int, heur:float, V_add:list) -> None:
        super().__init__(idx, heur, V_add)

    def eval(self, R:list, decgraphs:List[DecGraph], D:nx.Graph, sol:Solution) -> Tuple[float, float]:
        if self._eval_res:
            cost = self._eval_res["cost"]
            return max(sol.tau, cost), sol.sum_costs - sol.costs[self.idx] + cost

        decgraph, r = decgraphs[self.idx], R[self.idx]
        decgraph.add_pairing_verts(D, self.V)

        pi = ExtSTCPlanner.plan(r, decgraph)
        cost = DecGraph.path_cost(D, pi)
        new_tau = max(sol.tau, cost)
        new_sum_costs = sol.sum_costs - sol.costs[self.idx] + cost

        decgraph.del_pairing_verts(D, self.V)
        self._eval_res = {"pi": pi, "cost": cost}

        return new_tau, new_sum_costs

    def apply(self, R:list, decgraphs:List[DecGraph], D:nx.Graph, dup_rec:DuplicationRec, sol:Solution) -> str:
        self.eval(R, decgraphs, D, sol)
        sol.Pi[self.idx] = self._eval_res["pi"]
        sol.costs[self.idx] = self._eval_res["cost"]
        decgraphs[self.idx].add_pairing_verts(D, self.V)
        for v in self.V:
            dup_rec.dup(v, self.idx)
        
        return f"Add@{self.idx}:[{self.V}]"


class DedupOP(Operator):

    def __init__(self, idx:int, heur:float, V_rmv:list) -> None:
        super().__init__(idx, heur, V_rmv)

    def eval(self, R:list, decgraphs:List[DecGraph], D:nx.Graph, sol:Solution) -> Tuple[float, float]:
        if self._eval_res:
            cost = self._eval_res["cost"]
            return max(sol.tau, cost), sol.sum_costs - sol.costs[self.idx] + cost

        decgraph, r = decgraphs[self.idx], R[self.idx]
        decgraph.del_pairing_verts(D, self.V)

        pi = ExtSTCPlanner.plan(r, decgraph)
        cost = DecGraph.path_cost(D, pi)
        new_tau = max(sol.tau, cost)
        new_sum_costs = sol.sum_costs - sol.costs[self.idx] + cost

        decgraph.add_pairing_verts(D, self.V)
        self._eval_res = {"pi": pi, "cost": cost}

        return new_tau, new_sum_costs

    def apply(self, R:list, decgraphs:List[DecGraph], D:nx.Graph, dup_rec:DuplicationRec, sol:Solution) -> str:
        self.eval(R, decgraphs, D, sol)
        sol.Pi[self.idx] = self._eval_res["pi"]
        sol.costs[self.idx] = self._eval_res["cost"]
        decgraphs[self.idx].del_pairing_verts(D, self.V)
        for v in self.V:
            dup_rec.dedup(v, self.idx)
        
        return f"Rmv@{self.idx}:[{self.V}]"


class ExcOP(Operator):

    def __init__(self, op_add_idx:int, op_rmv_idx:int, heur:float, V:list) -> None:
        super().__init__(op_add_idx, heur, V)
        self.op_rmv_idx = op_rmv_idx

    @property
    def pid(self) -> tuple:
        return (self.idx, self.op_rmv_idx)
    
    def eval(self, R:list, decgraphs:List[DecGraph], D:nx.Graph, sol:Solution) -> Tuple[float, float]:
        decgraph, r = decgraphs[self.idx], R[self.idx]
        
        if self._eval_res:
            cost_add = self._eval_res["cost_add"]
            cost_rmv = self._eval_res["cost_rmv"]
            return max(max(sol.tau, cost_add), cost_rmv), \
                   sol.sum_costs - sol.costs[self.idx] - sol.costs[self.op_rmv_idx] + cost_add + cost_rmv

        decgraphs[self.idx].add_pairing_verts(D, self.V)
        decgraphs[self.op_rmv_idx].del_pairing_verts(D, self.V)

        pi_add = ExtSTCPlanner.plan(R[self.idx], decgraphs[self.idx])
        pi_rmv = ExtSTCPlanner.plan(R[self.op_rmv_idx], decgraphs[self.op_rmv_idx])
        cost_add = DecGraph.path_cost(D, pi_add)
        cost_rmv = DecGraph.path_cost(D, pi_rmv)
        new_tau = max(max(sol.tau, cost_add), cost_rmv)
        new_sum_costs = sol.sum_costs - sol.costs[self.idx] - sol.costs[self.op_rmv_idx] + cost_add + cost_rmv

        decgraphs[self.idx].del_pairing_verts(D, self.V)
        decgraphs[self.op_rmv_idx].add_pairing_verts(D, self.V)

        self._eval_res = {"pi_add": pi_add, "pi_rmv":pi_rmv, "cost_add": cost_add, "cost_rmv":cost_rmv}

        return new_tau, new_sum_costs

    def apply(self, R:list, decgraphs:List[DecGraph], D: nx.Graph, dup_rec: DuplicationRec, sol: Solution) -> str:
        self.eval(R, decgraphs, D, sol)
        sol.Pi[self.idx] = self._eval_res["pi_add"]
        sol.Pi[self.op_rmv_idx] = self._eval_res["pi_rmv"]
        sol.costs[self.idx] = self._eval_res["cost_add"]
        sol.costs[self.op_rmv_idx] = self._eval_res["cost_rmv"]
        decgraphs[self.idx].add_pairing_verts(D, self.V)
        decgraphs[self.op_rmv_idx].del_pairing_verts(D, self.V)
        for v in self.V:
            dup_rec.dup(v, self.idx)
            dup_rec.dedup(v, self.op_rmv_idx)

        return f"Exc@({self.op_rmv_idx}=>{self.idx}):[{self.V}]"
