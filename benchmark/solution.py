from __future__ import annotations
import pickle

import networkx as nx
import numpy as np

from lsmcpp.benchmark.plan import Heading


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
    
    @staticmethod
    def path_cost_legacy(D:nx.Graph, pi:list) -> float:
        return sum([D[pi[i]][pi[i+1]]["weight"] for i in range(len(pi)-1)])
    
    @staticmethod
    def path_cost(pi:list, D:nx.Graph=None, default_heading:Heading=Heading.N) -> float:
        """ the path cost with turning costs """
        cost, heading_last = 0, default_heading
        if D is None:
            edge_cost = lambda u, v: np.hypot(u[0]-v[0], u[1]-v[1])
        else:
            edge_cost = lambda u, v: D[u][v]["weight"]

        for i in range(len(pi)-1):
            heading = Heading.get(pi[i], pi[i+1])
            cost += edge_cost(pi[i], pi[i+1]) + Heading.rot_cost(heading_last, heading)
            heading_last = heading

        return cost
