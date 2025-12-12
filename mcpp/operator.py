from __future__ import annotations

from abc import abstractmethod
from typing import List, Tuple

import networkx as nx

from lsmcpp.estc import ExtSTCPlanner
from lsmcpp.graph import DecGraph
from lsmcpp.benchmark.solution import Solution
from lsmcpp.utils import DuplicationRec


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
    def eval(self, R:list, dg:DecGraph, decgraphs:List[DecGraph], D:nx.Graph, sol:Solution) -> Tuple[float, float]|None:
        raise NotImplementedError

    @abstractmethod
    def apply(self, R:list, dg:DecGraph, decgraphs:List[DecGraph], D:nx.Graph, dup_rec:DuplicationRec, sol:Solution) -> str:
        raise NotImplementedError


class GrowOP(Operator):

    def __init__(self, idx:int, heur:float, V_add:list) -> None:
        super().__init__(idx, heur, V_add)

    def eval(self, R:list, dg:DecGraph, decgraphs:List[DecGraph], D:nx.Graph, sol:Solution) -> Tuple[float, float]:
        if self._eval_res:
            cost = self._eval_res["cost"]
            return max(sol.tau, cost), sol.sum_costs - sol.costs[self.idx] + cost

        decgraph, r = decgraphs[self.idx], R[self.idx]
        decgraph.add_pairing_verts(dg, self.V)
        if r not in decgraph.T:
            r_top, r_bot = r + ("top",), r + ("bot",)
            r = r_top if r_top in decgraph.T else r_bot
        pi = ExtSTCPlanner.plan(r, decgraph)
        cost = Solution.path_cost(pi, D)
        new_tau = max(sol.tau, cost)
        new_sum_costs = sol.sum_costs - sol.costs[self.idx] + cost

        decgraph.del_pairing_verts(dg, self.V)
        self._eval_res = {"pi": pi, "cost": cost}

        return new_tau, new_sum_costs

    def apply(self, R:list, dg:DecGraph, decgraphs:List[DecGraph], D:nx.Graph, dup_rec:DuplicationRec, sol:Solution) -> str:
        self.eval(R, dg, decgraphs, D, sol)
        sol.Pi[self.idx] = self._eval_res["pi"]
        sol.costs[self.idx] = self._eval_res["cost"]
        decgraphs[self.idx].add_pairing_verts(dg, self.V)
        for v in self.V:
            dup_rec.dup(v, self.idx)
        
        return f"Add@{self.idx}:[{self.V}]"
    

class DedupOP(Operator):

    def __init__(self, idx:int, heur:float, V_rmv:list) -> None:
        super().__init__(idx, heur, V_rmv)

    def eval(self, R:list, dg:DecGraph, decgraphs:List[DecGraph], D:nx.Graph, sol:Solution) -> Tuple[float, float]|None:
        
        if self._eval_res:
            cost = self._eval_res["cost"]
            return max(sol.tau, cost), sol.sum_costs - sol.costs[self.idx] + cost

        decgraph, r = decgraphs[self.idx], R[self.idx]
        if decgraph.del_pairing_verts(dg, self.V):
            if r not in decgraph.T:
                r_top, r_bot = r + ("top",), r + ("bot",)
                r = r_top if r_top in decgraph.T else r_bot
            pi = ExtSTCPlanner.plan(r, decgraph)
            cost = Solution.path_cost(pi, D)
            new_tau = max(sol.tau, cost)
            new_sum_costs = sol.sum_costs - sol.costs[self.idx] + cost

            decgraph.add_pairing_verts(dg, self.V)
            self._eval_res = {"pi": pi, "cost": cost}
            return new_tau, new_sum_costs
        
        return None

    def apply(self, R:list, dg:DecGraph, decgraphs:List[DecGraph], D:nx.Graph, dup_rec:DuplicationRec, sol:Solution) -> str:
        if self._eval_res is None:
            return "NoOp"

        self.eval(R, dg, decgraphs, D, sol)
        sol.Pi[self.idx] = self._eval_res["pi"]
        sol.costs[self.idx] = self._eval_res["cost"]
        decgraphs[self.idx].del_pairing_verts(dg, self.V)
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
    
    def eval(self, R:list, dg:DecGraph, decgraphs:List[DecGraph], D:nx.Graph, sol:Solution) -> Tuple[float, float]|None:
        decgraph, r = decgraphs[self.idx], R[self.idx]
        
        if self._eval_res:
            cost_add = self._eval_res["cost_add"]
            cost_rmv = self._eval_res["cost_rmv"]
            return max(max(sol.tau, cost_add), cost_rmv), \
                   sol.sum_costs - sol.costs[self.idx] - sol.costs[self.op_rmv_idx] + cost_add + cost_rmv

        if decgraphs[self.op_rmv_idx].del_pairing_verts(dg, self.V):
            decgraphs[self.idx].add_pairing_verts(dg, self.V)
            r_grow, r_rmv = r, R[self.op_rmv_idx]
            if R[self.idx] not in decgraphs[self.idx].T:
                r_top, r_bot = R[self.idx] + ("top",), R[self.idx] + ("bot",)
                r_grow = r_top if r_top in decgraph.T else r_bot
            if R[self.op_rmv_idx] not in decgraphs[self.op_rmv_idx].T:
                r_top, r_bot = R[self.op_rmv_idx] + ("top",), R[self.op_rmv_idx] + ("bot",)
                r_rmv = r_top if r_top in decgraphs[self.op_rmv_idx].T else r_bot
            pi_add = ExtSTCPlanner.plan(r_grow, decgraphs[self.idx])
            pi_rmv = ExtSTCPlanner.plan(r_rmv, decgraphs[self.op_rmv_idx])
            cost_add = Solution.path_cost(pi_add, D)
            cost_rmv = Solution.path_cost(pi_rmv, D)
            new_tau = max(max(sol.tau, cost_add), cost_rmv)
            new_sum_costs = sol.sum_costs - sol.costs[self.idx] - sol.costs[self.op_rmv_idx] + cost_add + cost_rmv

            decgraphs[self.idx].del_pairing_verts(dg, self.V)
            decgraphs[self.op_rmv_idx].add_pairing_verts(dg, self.V)

            self._eval_res = {"pi_add": pi_add, "pi_rmv":pi_rmv, "cost_add": cost_add, "cost_rmv":cost_rmv}
            return new_tau, new_sum_costs
        
        return None

    def apply(self, R:list, dg:DecGraph, decgraphs:List[DecGraph], D: nx.Graph, dup_rec: DuplicationRec, sol: Solution) -> str:
        if self._eval_res is None:
            return "NoOp"
        
        self.eval(R, dg, decgraphs, D, sol)
        sol.Pi[self.idx] = self._eval_res["pi_add"]
        sol.Pi[self.op_rmv_idx] = self._eval_res["pi_rmv"]
        sol.costs[self.idx] = self._eval_res["cost_add"]
        sol.costs[self.op_rmv_idx] = self._eval_res["cost_rmv"]
        decgraphs[self.idx].add_pairing_verts(dg, self.V)
        decgraphs[self.op_rmv_idx].del_pairing_verts(dg, self.V)
        for v in self.V:
            dup_rec.dup(v, self.idx)
            dup_rec.dedup(v, self.op_rmv_idx)

        return f"Exc@({self.op_rmv_idx}=>{self.idx}):[{self.V}]"
