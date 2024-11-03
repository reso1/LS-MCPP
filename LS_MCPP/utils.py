from functools import wraps
import os
import time
import pickle
from decimal import Decimal
from collections import defaultdict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def timeit(func):

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


class Helper:

    @staticmethod
    def to_position_graph(G: nx.Graph) -> nx.Graph:
        pos = nx.get_node_attributes(G, "pos")
        tw = nx.get_node_attributes(G, "terrain_weight")

        ret = nx.Graph()
        for u, v in G.edges():
            ret.add_edge(pos[u], pos[v], weight=G[u][v]["weight"])
        
        nx.set_node_attributes(ret, {pos[v]: w for v, w in tw.items()}, "terrain_weight")
        return ret

    @staticmethod
    def shift(idx:int, offset: int, len_pi:int) -> int:
        return (idx + offset) % len_pi

    @staticmethod
    def to_index_graph(G: nx.Graph) -> nx.Graph:
        pos = {v:p for v, p in enumerate(G.nodes)}
        pos2v = {p:v for v, p in enumerate(G.nodes)}

        ret = nx.Graph()
        for pu, pv in G.edges():
            u, v = pos2v[pu], pos2v[pv] 
            ret.add_edge(u, v, weight=G[pu][pv]["weight"])
        
        nx.set_node_attributes(ret, pos, "pos")
        nx.set_node_attributes(ret, {pos2v[p]: G.nodes[p]["terrain_weight"] for p in G.nodes}, "terrain_weight")
        return ret

    @staticmethod
    def verbose_print(msg:str, verbose:bool=False) -> None:
        if verbose:
            print(msg)

    @staticmethod
    def draw_terrain_graph(T: nx.Graph, ax, color='k') -> None:
        is_complete = nx.get_node_attributes(T, "complete")

        for u, v in T.edges:
            if len(u) == 3 and u[2] == "bot":
                u = (u[0]+0.25, u[1]+0.25)
            if len(u) == 3 and u[2] == "top":
                u = (u[0]-0.25, u[1]-0.25)
            if len(v) == 3 and v[2] == "bot":
                v = (v[0]+0.25, v[1]+0.25)
            if len(v) == 3 and v[2] == "top":
                v = (v[0]-0.25, v[1]-0.25)
            ax.plot([u[0], v[0]], [u[1], v[1]], f'-{color}')

        for v in T.nodes:
            c = f's-{color}' if is_complete[v] else f'^-{color}'
            if len(v) == 3 and v[2] == "bot":
                v = (v[0]+0.25, v[1]+0.25)
            if len(v) == 3 and v[2] == "top":
                v = (v[0]-0.25, v[1]-0.25)
            ax.plot(v[0], v[1], c)

    @staticmethod
    def draw_cov_path(pi, r, ax, color='k', alpha=1.0) -> None:
        ax.plot(r[0], r[1], 'or')
        for i in range(len(pi)):
            u, v = pi[i], pi[(i+1)%len(pi)]
            ax.plot([u[0], v[0]], [u[1], v[1]], f'-{color}', alpha=alpha)

    @staticmethod
    def draw_matrix(A, ax) -> None:
        n_rows, n_cols = A.shape
        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(j+0.5, i+0.5, f"{A[i][j]:.2f}", va='center', ha='center')

        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.grid()

    @staticmethod
    def is_path_valid(pi) -> bool:
        for i in range(len(pi) - 1):
            if abs(pi[i][0] - pi[i + 1][0]) + abs(pi[i][1] - pi[i + 1][1]) != 0.5:
                return False
        return True

    @staticmethod
    def pn_str(num:float) -> str:
        return f"{num:.2f}" if num < 0 else f"+{num:.2f}"


class DuplicationRec:

    def __init__(self) -> None:
        self._data = defaultdict(set)
        self.cnts = defaultdict(int)
        self.dup_set = set()
    
    def dup(self, v:tuple, idx:int) -> None:
        if idx not in self._data[v]:
            self._data[v].add(idx)
            self.cnts[v] += 1
            if self.cnts[v] > 1:
                self.dup_set.add(v)

    def dedup(self, v:tuple, idx:int) -> None:
        if idx in self._data[v]:
            self._data[v].remove(idx)
            self.cnts[v] -= 1
            if self.cnts[v] <= 1:
                self.dup_set.remove(v)


class Tracer:
    @staticmethod
    def load_tracer(s):
        with open(os.path.join("data", "runrecords", s), 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def plt_param(ticks):
        plt.yticks(fontsize=8)
        labels = []
        for v in ticks:
            if v == 0:
                labels.append('0')
            else:
                labels.append('%.1E' % Decimal(v))
        plt.xticks(ticks, labels, fontsize=8)

