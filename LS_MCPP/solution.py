from __future__ import annotations
import pickle
import LS_MCPP

from typing import Dict, List, Set

import networkx as nx
import numpy as np

from MSTC_Star.mcpp.rtc_planner import RTCPlanner

from MIP_MCPP.instance import Instance
from MIP_MCPP.mcpp_planner import mfc_plan, mip_plan, mstcstar_plan

from LS_MCPP.estc import ExtSTCPlanner
from LS_MCPP.graph import DecGraph
from LS_MCPP.utils import Helper


class Solution:

    def __init__(self, Pi:list, costs:np.ndarray) -> None:
        self.Pi = Pi
        self.costs = costs

    def copy(self) -> Solution:
        return Solution(self.Pi.copy(), self.costs.copy())
    
    def save(self, filepath:str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def __lt__(self, other: Solution) -> bool:
        return self.tau < other.tau

    def __gt__(self, other: Solution) -> bool:
        return self.tau > other.tau

    def __eq__(self, other: Solution) -> bool:
        return self.tau == other.tau

    @property
    def tau(self) -> float:
        return self.costs.max()

    @property
    def avg_cost(self) -> float:
        return self.costs.mean()

    @property
    def sum_costs(self) -> float:
        return self.costs.sum()

    @property
    def cost_str(self) -> str:
        return f"costs={[round(self.costs[i], 2) for i in range(len(self.costs))]}"

    @staticmethod
    def load(filepath: str) -> Solution:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    

def Voronoi_sol(istc:Instance) -> Solution:
    T = Helper.to_position_graph(istc.G)
    D = DecGraph.get_decomposed_graph_complete(T)
    R = [istc.G.nodes[r]["pos"] for r in istc.R]

    paths, costs = [], []
    for r, V in nx.voronoi_cells(T, R).items():
        Di = DecGraph.get_decomposed_graph_complete(T.subgraph(V))
        G_prime, _dV = DecGraph.generate_G_prime(Di)
        paths.append(ExtSTCPlanner.plan(r, DecGraph(Di, G_prime, _dV)))
        costs.append(DecGraph.path_cost(D, paths[-1]))

    return Solution([pi[1:-1] for pi in paths], np.array(costs))


def MFC_sol(istc:Instance) -> Solution:
    _, paths, costs, _ = mfc_plan(istc)
    return Solution([pi[1:-1] for pi in paths], np.array(costs))


def incomplete_G_sol(dg:DecGraph, R:list) -> Solution:
    k = len(R)
    rtc_planner = RTCPlanner(dg.T, R, k)

    match_tuple, max_weights, opt_B = rtc_planner.k_tree_cover()

    nodes = {r:set([r]) for r in R}
    for r, val in match_tuple.items():
        L, S, P = val
        for idx in range(len(P)-1):
            nodes[r].add(P[idx])
            nodes[r].add(P[idx+1])
        for u, v in L.edges():
            nodes[r].add(u)
            nodes[r].add(v)
        for u, v in S.edges():
            nodes[r].add(u)
            nodes[r].add(v)
    
    paths, weights = [], np.zeros(k)
    for i, r in enumerate(R):
        Ti = dg.T.subgraph(nodes[r])
        Vdi = set()
        for Tv in nodes[r]:
            Vdi = Vdi.union([dv for dv in dg.dV[Tv] if dv is not None])
        Di = dg.D.subgraph(Vdi)
        pi = ExtSTCPlanner.plan(r, DecGraph(Di, Ti, dg.dV))
        paths.append(pi)
        weights[i] = DecGraph.path_cost(dg.D, pi)

    return Solution(paths, weights)


def MSTCStar_sol(istc:Instance) -> Solution:
    _, paths, costs, _ = mstcstar_plan(istc)
    return Solution([pi[1:-1] for pi in paths], np.array(costs))


def MIP_sol(istc:Instance, sol_edges) -> Solution:
    _, paths, costs = mip_plan(istc, sol_edges)
    return Solution([pi[1:-1] for pi in paths], np.array(costs))
