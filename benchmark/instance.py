from __future__ import annotations
from typing import List, Generator
from itertools import product

import os
from pathlib import Path
import yaml

import numpy as np
import networkx as nx

from matplotlib.colors import rgb2hex

from lsmcpp.graph import DecGraph, contract, decomp


xy_sorted = lambda e: (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])


class MCPP:

    def __init__(self, G:nx.Graph, R:list, name:str, width:int, height:int, Os:list, incomplete:bool=False, weighted=False) -> None:
        self.G, self.R, self.name, self.width, self.height, self.incomplete, self.weighted = G, R, name, width, height, incomplete, weighted
        self.k = len(R)
        self.n, self.m = G.number_of_nodes(), G.number_of_edges()
        self.I, self.V, self.E = list(range(self.k)), list(self.G.nodes), list(range(self.m))
        self._xy2e = {xy_sorted(e): i for i, e in enumerate(self.G.edges)}
        self.xy2e = lambda xy: self._xy2e[xy_sorted(xy)]
        self.e2xy = {i: xy for i, xy in enumerate(self.G.edges)}
        self.static_obstacles = Os
        self.VIDX = lambda pos: pos[0]*self.height + pos[1]
        self.pos2v = lambda v: self.VIDX((int(v[0]*2+0.5), int(v[1]*2+0.5)))

        # legacy graph for LS-MCPP code
        self._G_legacy = nx.Graph()
        self.legacy_vertex = lambda v: ((self.G.nodes[v]["pos"][0] - 0.5) / 2, (self.G.nodes[v]["pos"][1] - 0.5) / 2)
        for v in self.G.nodes:
            self._G_legacy.add_node(self.legacy_vertex(v), terrain_weight=0)
        for u, v in self.G.edges:
            self._G_legacy.add_edge(self.legacy_vertex(u), self.legacy_vertex(v), weight=self.G[u][v]["weight"])
    
    def __str__(self) -> str:
        return f"MCPP Instance ({self.n=}, {self.m=}, {self.k=})"

    def draw(self, ax, color:str='k', scale:float=1, alpha=1) -> None:
        for u, v in self.G.edges:
            u, v = self.legacy_vertex(u), self.legacy_vertex(v)
            ax.plot([u[0], v[0]], [u[1], v[1]], f"-{color}", lw=1.5*scale, alpha=alpha)

        for r in self.R:
            r = self.legacy_vertex(r)
            ax.plot(r[0], r[1], "*k", ms=8*scale, mfc=color, alpha=alpha)

        for v in self.static_obstacles:
            px, py = (v[0] - 0.5) / 2, (v[1] - 0.5) / 2
            ax.plot(px, py, "ks", ms=8*scale, alpha=alpha)
        
        ax.axis("equal")
        # ax.set_xlim(-1, self.width+1)
        # ax.set_ylim(-1, self.height+1)
        ax.set_xlabel("")
        ax.set_ylabel("")
        # ax.axis("off")

    def random_removal(self, percentage:float, seed:int) -> MCPP:
        rng = np.random.RandomState(seed)
        new_G = self.G.copy()
        Os = self.static_obstacles.copy()
        num_rmv = int(self.G.number_of_nodes() * percentage)
        
        # not allowing to remove root nodes and its neighbors
        V = set(new_G.nodes()) - set(self.R)
        for r in self.R:
            r_ngbs = [(r//self.height+ci, r%self.height+cj) for ci, cj in product([-1, 0, 1], [-1, 0, 1])]
            V = V - set([x[0]*self.height + x[1] for x in r_ngbs])

        V = list(V)
        while num_rmv > 0:
            to_rmv = rng.choice(V)
            _new_G_copy = new_G.copy()
            _new_G_copy.remove_node(to_rmv)
            if nx.is_connected(_new_G_copy):
                num_rmv -= 1
                V.remove(to_rmv)
                Os.append(new_G.nodes[to_rmv]["pos"])
                new_G = _new_G_copy
        
        return MCPP(new_G, self.R, f"{self.name}-rmv{percentage:.2f}-seed{seed}", self.width, self.height, Os, incomplete=True)

    def randomized_mutants(self, all_rmv_ratios, all_rand_seeds) -> Generator[MCPP]:
        terrain_graph = contract(self._G_legacy).T
        num_T_nodes, VT = terrain_graph.number_of_nodes(), list(terrain_graph.nodes)
        for rmv_ratio in all_rmv_ratios:
            for rand_seed in all_rand_seeds:
                rng = np.random.RandomState(rand_seed)
                fn = os.path.join("data", "instances", self.name, f"{rmv_ratio:.3f}-{rand_seed}.mcpp")
                if os.path.exists(fn):
                    with open(fn, "r") as f:
                        instance = yaml.load(f, Loader=yaml.FullLoader)
                    instance['map'] = os.path.join(instance['map'], f"{rmv_ratio:.3f}-{rand_seed}")
                    yield MCPP.build(f"{self.name}-rmv{rmv_ratio:.2f}-seed{rand_seed}.mcpp", instance)
                else:
                    R = []
                    for rand in rng.choice(num_T_nodes, size=self.k, replace=False):
                        subloc_list = list(range(0, 4))
                        subloc = rng.choice(subloc_list)
                        r = self.pos2v(decomp(VT[rand])[subloc])
                        while set(R).intersection(self.G.neighbors(r)):
                            subloc_list.remove(subloc)
                            subloc = rng.choice(subloc_list)
                            r = self.pos2v(decomp(VT[rand])[subloc])
                        R.append(r)

                    instance = {
                        "map": self.name,
                        "weighted": self.weighted,
                        "incomplete": rmv_ratio != 0,
                        "root": R,
                        "weight_seed": rand_seed
                    }

                    mcpp = self.build(self.name, instance)
                    yield mcpp.random_removal(rmv_ratio, rand_seed)

    @staticmethod
    def read_instance(filename:str) -> MCPP:
        with open(filename, "r") as f:
            instance = yaml.load(f, Loader=yaml.FullLoader)
            return MCPP.build(filename, instance)

    @staticmethod
    def build(filename:str, instance:dict) -> MCPP:
        weighted = bool(instance["weighted"])
        incomplete = bool(instance["incomplete"])
        G = nx.Graph()
        with open(os.path.join("data", "gridmaps", f"{instance['map']}.map"), "r") as f:
            lines = f.readlines()
            height = int(lines[1].strip().split(" ")[1])
            width = int(lines[2].strip().split(" ")[1])
            VIDX = lambda pos: pos[0]*height + pos[1]
            if type(instance["root"][0]) is int:
                R = instance["root"]
            else:
                R = [VIDX(r_pos) for r_pos in instance["root"]]
            Os = set([(-1, row)     for row in range(-1, height+1)] +
                    [(width, row)  for row in range(height+1)] +
                    [(col, -1)      for col in range(width+1)] +
                    [(col, height) for col in range(width+1)])
            for row in range(height):
                for col in range(width):
                    pos = (col, row)
                    if lines[4+row][col] == ".":
                        G.add_node(VIDX(pos), pos=pos)
                    else:
                        Os.add(pos)

            if weighted:
                rng = np.random.RandomState(instance["weight_seed"])
                weight_range = instance["weight_range"] if "weight_range" in instance else [1, 3]

            for u in G.nodes:
                u_pos = G.nodes[u]["pos"]
                for v_pos in [(u_pos[0]+1, u_pos[1]), (u_pos[0], u_pos[1]+1), 
                              (u_pos[0]-1, u_pos[1]), (u_pos[0], u_pos[1]-1)]:
                    v = VIDX(v_pos)
                    if 0 <= v_pos[0] < width and 0 <= v_pos[1] < height and v in G.nodes:
                        if weighted:
                            weight = rng.uniform(*weight_range)
                            weight = round(weight, instance["weight_precision"]) if "weight_precision" in instance else weight
                        else:
                            weight = 1
                        G.add_edge(u, v, weight=weight)

        # remove non-reachable nodes to root
        if not nx.is_connected(G):
            nrV = set()
            for C in nx.connected_components(G):
                if not any([r in C for r in R]):
                    nrV.update(C)
            G.remove_nodes_from(nrV)
        
        return MCPP(G, R, Path(filename).stem, width, height, list(Os), incomplete, weighted)
