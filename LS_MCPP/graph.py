from __future__ import annotations

import itertools
from typing import Tuple
from itertools import chain, combinations

import networkx as nx
import numpy as np

from MSTC_Star.mcpp.stc_planner import STCPlanner

from LS_MCPP.utils import Helper


class DecGraph:

    def __init__(self, D: nx.Graph, T:nx.Graph, dV:dict) -> None:
        self.D, self.T, self.dV = D, T, dV
        self.V = set(self.D.nodes)

    def copy(self) -> DecGraph:
        return DecGraph(self.D.copy(), self.T.copy(), self.dV.copy())

    def update(self, D:nx.Graph, V_new:set, dict_tw_T:dict, complete:dict) -> None:
        V_D = set.symmetric_difference(V_new, self.V)
        processed, V_added = set(), set()
        while V_D:
            D_v = V_D.pop()
            T_v = DecGraph.undecomp(D_v)
            if T_v not in processed:
                processed.add(T_v)
                for name in [T_v, T_v + ("bot",), T_v + ("top",)]:
                    if name in self.T:
                        self.T.remove_node(name)
                    if name in self.dV:
                        self.dV.pop(name)

                tw_Tv = dict_tw_T[T_v]
                dv, dv_cnt = DecGraph.decomp(T_v), 4
                for i, D_u in enumerate(dv):
                    if D_u not in V_new:
                        dv[i] = None
                        dv_cnt -= 1
                    elif D_u in V_D:
                        V_D.remove(D_u)

                if dv_cnt != 0:
                    if (dv[0] and not dv[1] and dv[2] and not dv[3]):
                        V_added.add(T_v + ("bot",))
                        V_added.add(T_v + ("top",))
                        self.dV[T_v + ("bot",)] = [dv[0], None, None, None]
                        self.dV[T_v + ("top",)] = [None, None, dv[2], None]
                        complete[T_v + ("bot",)] = False
                        complete[T_v + ("top",)] = False
                        dict_tw_T[T_v + ("bot",)] = tw_Tv
                        dict_tw_T[T_v + ("top",)] = tw_Tv
                    elif (not dv[0] and dv[1] and not dv[2] and dv[3]):
                        V_added.add(T_v + ("bot",))
                        V_added.add(T_v + ("top",))
                        self.dV[T_v + ("bot",)] = [None, dv[1], None, None]
                        self.dV[T_v + ("top",)] = [None, None, None, dv[3]]
                        complete[T_v + ("bot",)] = False
                        complete[T_v + ("top",)] = False
                        dict_tw_T[T_v + ("bot",)] = tw_Tv
                        dict_tw_T[T_v + ("top",)] = tw_Tv
                    else:
                        V_added.add(T_v)
                        self.dV[T_v] = dv
                        complete[T_v] = dv_cnt == 4
                        dict_tw_T[T_v] = tw_Tv

        for u in V_added:
            for xc, yc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                v, vt, vb = (u[0]+xc, u[1]+yc), (u[0]+xc, u[1]+yc, "top"), (u[0]+xc, u[1]+yc, "bot")
                v_l = []
                for name in [v, vt, vb]:
                    if name in self.dV:
                        v_l.append(name)
                
                for u, v in itertools.product([u], v_l):
                    if u[1] == v[1]:    # horizontal u -> v
                        _u, _v = (u, v) if u[0] < v[0] else (v, u)
                        if (self.dV[_u][0] and self.dV[_v][3]) or (self.dV[_u][1] and self.dV[_v][2]):
                            self.T.add_edge(u, v, weight=(dict_tw_T[u] + dict_tw_T[v]) / 2)
                    else:               # vertical u -> v
                        _u, _v = (u, v) if u[1] < v[1] else (v, u)
                        if (self.dV[_u][1] and self.dV[_v][0]) or (self.dV[_u][2] and self.dV[_v][3]):
                            self.T.add_edge(u, v, weight=(dict_tw_T[u] + dict_tw_T[v]) / 2)

        if len(self.dV) == 1:
            self.T.add_node(list(self.dV)[0])

        nx.set_node_attributes(self.T, complete, "complete")
        nx.set_node_attributes(self.T, dict_tw_T, "terrain_weight")

        self.D = D
        self.V = V_new

    def add_pairing_verts(self, D:nx.Graph, V_add:list) -> None:
        self.V = self.V.union(V_add)
        T_v = DecGraph.undecomp(V_add[0])
        D_v_terrain_weight = D.nodes[V_add[0]]["terrain_weight"]

        # update the subgraph D
        for D_v in V_add:
            for ngb in D.neighbors(D_v):
                if ngb in self.V:
                    self.D.add_edge(D_v, ngb, weight=D[D_v][ngb]["weight"])
                    self.D.nodes[D_v]["terrain_weight"] = D_v_terrain_weight

        # update dV
        dv, is_complete = DecGraph.decomp(T_v), True
        for i, v in enumerate(dv):
            if v not in self.V:
                dv[i] = None
                is_complete = False
        self.dV[T_v] = dv
        
        # update corresponding T
        if self.T.has_node(T_v):
            self.T.remove_node(T_v)
        self.T.add_node(T_v, complete=is_complete, terrain_weight=4*D_v_terrain_weight)
        for T_u in [DecGraph.Tv_ngb(T_v, d) for d in [0, 1, 2, 3]]:
            if self.T.has_node(T_u):
                if T_u[1] == T_v[1]:    # horizontal neighbor
                    _u, _v = (T_u, T_v) if T_u[0] < T_v[0] else (T_v, T_u)
                    if (self.dV[_u][0] and self.dV[_v][3]) or (self.dV[_u][1] and self.dV[_v][2]):
                        weight = (self.T.nodes[T_u]["terrain_weight"] + self.T.nodes[T_v]["terrain_weight"]) / 2
                        self.T.add_edge(T_u, T_v, weight=weight)
                else:                   # vertical neighbor
                    _u, _v = (T_u, T_v) if T_u[1] < T_v[1] else (T_v, T_u)
                    if (self.dV[_u][1] and self.dV[_v][0]) or (self.dV[_u][2] and self.dV[_v][3]):
                        weight = (self.T.nodes[T_u]["terrain_weight"] + self.T.nodes[T_v]["terrain_weight"]) / 2
                        self.T.add_edge(T_u, T_v, weight=weight)

    def del_pairing_verts(self, D:nx.Graph, V_del:list) -> bool:
        D_v_terrain_weight = self.D.nodes[V_del[0]]["terrain_weight"]
        self.D.remove_nodes_from(V_del)
        
        if not nx.is_connected(self.D):
            for D_v in V_del:
                for ngb in D.neighbors(D_v):
                    if ngb in self.V:
                        self.D.add_edge(D_v, ngb, weight=D[D_v][ngb]["weight"])
                        self.D.nodes[D_v]["terrain_weight"] = D_v_terrain_weight
            return False

        # update the subgraph D
        self.V -= set(V_del)
        T_v = DecGraph.undecomp(V_del[0])

        # update dV
        dv, cnts = DecGraph.decomp(T_v), 4
        for i, v in enumerate(dv):
            if v not in self.V:
                dv[i] = None
                cnts -= 1
                
        self.T.remove_node(T_v)
        if cnts == 0:
            self.dV.pop(T_v)
        else:
            self.dV[T_v] = dv
            # update corresponding T
            self.T.add_node(T_v, complete=False, terrain_weight=4*D_v_terrain_weight)
            for T_u in [DecGraph.Tv_ngb(T_v, d) for d in [0, 1, 2, 3]]:
                if self.T.has_node(T_u):
                    if T_u[1] == T_v[1]:    # horizontal neighbor
                        _u, _v = (T_u, T_v) if T_u[0] < T_v[0] else (T_v, T_u)
                        if (self.dV[_u][0] and self.dV[_v][3]) or (self.dV[_u][1] and self.dV[_v][2]):
                            weight = (self.T.nodes[T_u]["terrain_weight"] + self.T.nodes[T_v]["terrain_weight"]) / 2
                            self.T.add_edge(T_u, T_v, weight=weight)
                    else:                   # vertical neighbor
                        _u, _v = (T_u, T_v) if T_u[1] < T_v[1] else (T_v, T_u)
                        if (self.dV[_u][1] and self.dV[_v][0]) or (self.dV[_u][2] and self.dV[_v][3]):
                            weight = (self.T.nodes[T_u]["terrain_weight"] + self.T.nodes[T_v]["terrain_weight"]) / 2
                            self.T.add_edge(T_u, T_v, weight=weight)
        
        return True

    @staticmethod
    def path_cost(D:nx.Graph, pi:list) -> float:
        return sum([(D.nodes[pi[i]]["terrain_weight"] + D.nodes[pi[i+1]]["terrain_weight"])/2 for i in range(len(pi)-1)])
    
    def draw(self, ax, color='k') -> None:
        for v in self.D.nodes:
            ax.plot(v[0], v[1], f'.{color}')
        for u, v in self.D.edges:
            ax.plot([u[0], v[0]], [u[1], v[1]], f'.--{color}')
        for v in self.T.nodes:
            ax.plot(v[0], v[1], f's{color}')
        for u, v in self.T.edges:
            ax.plot([u[0], v[0]], [u[1], v[1]], f's-{color}')

    @staticmethod
    def get_decomposed_graph_complete(T:nx.Graph) -> nx.Graph:
        dummy_planner = STCPlanner(T)
        D = dummy_planner.generate_decomposed_graph(T, None)
        for v in D.nodes:
            D.nodes[v]["terrain_weight"] = T.nodes[DecGraph.undecomp(v)]["terrain_weight"] / 4

        return D
    
    @staticmethod
    def generate_G_prime(D: nx.Graph) -> Tuple[nx.Graph, dict]:
        V_D = set(D.nodes)
        dV, complete, terrain_weight = {}, {}, {}

        while V_D:
            D_v = V_D.pop()
            tw_Tv = 4 * D.nodes[D_v]["terrain_weight"]
            T_v = DecGraph.undecomp(D_v)
            is_complete = True
            dv = DecGraph.decomp(T_v)
            for i, D_u in enumerate(dv):
                if D_u in V_D:
                    V_D.remove(D_u)
                elif D_u != D_v:
                    dv[i] = None
                    is_complete = False
            if (dv[0] and not dv[1] and dv[2] and not dv[3]):
                dV[T_v + ("bot",)] = [dv[0], None, None, None]
                dV[T_v + ("top",)] = [None, None, dv[2], None]
                complete[T_v + ("bot",)] = False
                complete[T_v + ("top",)] = False
                terrain_weight[T_v + ("bot",)] = tw_Tv
                terrain_weight[T_v + ("top",)] = tw_Tv
            elif (not dv[0] and dv[1] and not dv[2] and dv[3]):
                dV[T_v + ("bot",)] = [None, dv[1], None, None]
                dV[T_v + ("top",)] = [None, None, None, dv[3]]
                complete[T_v + ("bot",)] = False
                complete[T_v + ("top",)] = False
                terrain_weight[T_v + ("bot",)] = tw_Tv
                terrain_weight[T_v + ("top",)] = tw_Tv
            else:
                dV[T_v] = dv
                complete[T_v] = is_complete
                terrain_weight[T_v] = tw_Tv
        
        G_prime = nx.Graph()
        for u in dV.keys():
            for xc, yc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                v, vt, vb = (u[0]+xc, u[1]+yc), (u[0]+xc, u[1]+yc, "top"), (u[0]+xc, u[1]+yc, "bot")
                if v not in dV and vt not in dV and vb not in dV:
                    continue

                u_l = [u] if u in dV else [u + ("bot", ), u + ("top", )]
                v_l = [v] if v in dV else [v + ("bot", ), v + ("top", )]
                for u, v in itertools.product(u_l, v_l):
                    if u[1] == v[1]:    # horizontal u -> v
                        u, v = (u, v) if u[0] < v[0] else (v, u)
                        if (dV[u][0] and dV[v][3]) or (dV[u][1] and dV[v][2]):
                            weight = (terrain_weight[u] + terrain_weight[v]) / 2
                            G_prime.add_edge(u, v, weight=weight)
                    else:               # vertical u -> v
                        u, v = (u, v) if u[1] < v[1] else (v, u)
                        if (dV[u][1] and dV[v][0]) or (dV[u][2] and dV[v][3]):
                            weight = (terrain_weight[u] + terrain_weight[v]) / 2
                            G_prime.add_edge(u, v, weight=weight)

        if len(dV) == 1:
            G_prime.add_node(list(dV)[0])

        nx.set_node_attributes(G_prime, complete, "complete")
        nx.set_node_attributes(G_prime, terrain_weight, "terrain_weight")

        assert(nx.is_connected(G_prime))
        return G_prime, dV

    @staticmethod
    def shortest_vert_set_path(dists:np.ndarray, Va:list, Vb:list) -> Tuple[float, tuple]:
        min_pair = (float('inf'), None)
        for u in Va:
            v_ind = np.argmin(dists[u, Vb])
            v = Vb[v_ind]
            if dists[u, v] < min_pair[0]:
                min_pair = (dists[u, v], (u, v))

        return min_pair

    @staticmethod
    def undecomp(D_v:tuple) -> tuple:
        return (round(D_v[0]), round(D_v[1]))
    
    @staticmethod
    def decomp(T_v:tuple) -> tuple:
        return [DecGraph.get_subnode_coords(T_v, d) for d in ["SE", "NE", "NW", "SW"]]

    @staticmethod
    def get_subnode_coords(node, direction):
        x, y = node
        if direction == 'SE':
            return (x+0.25, y-0.25)
        elif direction == 'SW':
            return (x-0.25, y-0.25)
        elif direction == 'NE':
            return (x+0.25, y+0.25)
        elif direction == 'NW':
            return (x-0.25, y+0.25)

    @staticmethod
    def Tv_ngb(T_v:tuple, dir:int) -> tuple:
        if dir == 0:    # S
            return (T_v[0], T_v[1] - 1)
        elif dir == 1:  # E
            return (T_v[0] + 1, T_v[1])
        elif dir == 2:  # N
            return (T_v[0], T_v[1] + 1)
        elif dir == 3:  # W
            return (T_v[0] - 1, T_v[1])

    @staticmethod
    def Tv_perp(u:tuple, v:tuple, Tv:tuple) -> tuple:
        lr = v[0] % 1 > 0.5
        tb = v[1] % 1 > 0.5
        if lr and tb:
            if u == (v[0]+0.5, v[1]):
                return DecGraph.Tv_ngb(Tv, 2)
            if u == (v[0], v[1]+0.5):
                return DecGraph.Tv_ngb(Tv, 1)
        elif not lr and tb:
            if u == (v[0]-0.5, v[1]):
                return DecGraph.Tv_ngb(Tv, 2)
            if u == (v[0], v[1]+0.5):
                return DecGraph.Tv_ngb(Tv, 3)
        elif lr and not tb:
            if u == (v[0]+0.5, v[1]):
                return DecGraph.Tv_ngb(Tv, 0)
            if u == (v[0], v[1]-0.5):
                return DecGraph.Tv_ngb(Tv, 1)
        elif not lr and not tb:
            if u == (v[0]-0.5, v[1]):
                return DecGraph.Tv_ngb(Tv, 0)
            if u == (v[0], v[1]-0.5):
                return DecGraph.Tv_ngb(Tv, 3)

    @staticmethod
    def randomly_remove_Vd(istc, perc:float, rand_seed:int, no_split=False) -> DecGraph:
        # create non-repeative random ints
        np.random.seed(rand_seed)
        G = Helper.to_position_graph(istc.G)
        Vg = list(G.nodes())
        D = DecGraph.get_decomposed_graph_complete(G) 
        cnts, num_rmv, S = 0, G.number_of_nodes() * perc, set()
        rm_list = list(chain.from_iterable(combinations(range(4), r) for r in range(1, 4)))
        if no_split:
            rm_list.remove((0, 2))
            rm_list.remove((1, 3))
        while cnts < num_rmv:
            delta_v = Vg[np.random.randint(0, G.number_of_nodes())]
            while delta_v in S:
                delta_v = Vg[np.random.randint(0, G.number_of_nodes())]
            S.add(delta_v)
            Dv = DecGraph.decomp(delta_v)
            rmv = [Dv[i] for i in rm_list[np.random.choice(len(rm_list))]]
            _D = D.copy()
            _D.remove_nodes_from(rmv)
            if nx.is_connected(_D):
                D = _D
                cnts += 1

        G_prime, dV = DecGraph.generate_G_prime(D)
        assert nx.is_connected(G_prime)
        return DecGraph(D, G_prime, dV)
