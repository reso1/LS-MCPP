from __future__ import annotations
from typing import List

import itertools
from typing import Tuple

import networkx as nx


class DecGraph:

    def __init__(self, D: nx.Graph, T:nx.Graph, dV:dict) -> None:
        self.D, self.T, self.dV = D, T, dV
        self.V = set(self.D.nodes)
        self.B = set()

    def copy(self) -> DecGraph:
        return DecGraph(self.D.copy(), self.T.copy(), self.dV.copy())

    def add_pairing_verts(self, dg_full:DecGraph, V_add:list) -> None:
        self.V = self.V.union(V_add)
        
        # update the subgraph D
        for D_v in V_add:
            for ngb in dg_full.D.neighbors(D_v):
                if ngb in self.V:
                    self.D.add_edge(D_v, ngb, weight=dg_full.D[D_v][ngb]["weight"])
    
        self._update_others(dg_full, V_add)

    def del_pairing_verts(self, dg_full:DecGraph, V_del:list) -> bool:
        self.D.remove_nodes_from(V_del)

        if not nx.is_connected(self.D):
            for D_v in V_del:
                for ngb in dg_full.D.neighbors(D_v):
                    if ngb in self.V:
                        self.D.add_edge(D_v, ngb, weight=dg_full.D[D_v][ngb]["weight"])
            return False
        
        self.V -= set(V_del)

        self._update_others(dg_full, V_del)
        return True
    
    def _update_others(self, dg_full:DecGraph, V_mod:list) -> None:
        D_v_terrain_weight = dg_full.D.nodes[V_mod[0]]["terrain_weight"]
        base = dg_full.undecomp(V_mod[0])
        old_Tvngb = get_all_possible_ngb(base)
        old_Tvs = [v for v in [base, base + ("top", ), base + ("bot", )]]
        dV = [None, None, None, None]
        for old_Tv in old_Tvs:
            for i, item in enumerate(self.decomp(old_Tv)):
                if item is not None and item in self.V:
                    dV[i] = item
        
        # remove old T, dV
        for old_Tv in old_Tvs:
            if old_Tv in self.T:
                self.T.remove_node(old_Tv)
                self.dV.pop(old_Tv)
        
        # add new T, dV
        for new_Tv, value in contract_single(dV, base[:2], 4*D_v_terrain_weight).items():
            self.T.add_node(new_Tv, complete=value["complete"])
            self.dV[new_Tv] = value["dV"]
            for Tu in [item for item in old_Tvngb if item in self.T.nodes]:
                if Tu[1] == new_Tv[1]:    # horizontal neighbor
                    _u, _v = (Tu, new_Tv) if Tu[0] < new_Tv[0] else (new_Tv, Tu)
                    if (self.dV[_u][0] and self.dV[_v][3]) or (self.dV[_u][1] and self.dV[_v][2]):
                        weight = terrain_edge_weight(Tu, new_Tv, self.D, self.dV)
                        self.T.add_edge(Tu, new_Tv, weight=weight)
                else:                   # vertical neighbor
                    _u, _v = (Tu, new_Tv) if Tu[1] < new_Tv[1] else (new_Tv, Tu)
                    if (self.dV[_u][1] and self.dV[_v][0]) or (self.dV[_u][2] and self.dV[_v][3]):
                        weight = terrain_edge_weight(Tu, new_Tv, self.D, self.dV)
                        self.T.add_edge(Tu, new_Tv, weight=weight)
        
        # update boundary vertices
        V_mod_ngbs = set()
        for v in V_mod:
            V_mod_ngbs = V_mod_ngbs.union([v] + list(dg_full.D.neighbors(v)))
        self._update_boundary_verts(dg_full, V_mod_ngbs)
    
    def _update_boundary_verts(self, dg_full:DecGraph, V_mod:list) -> None:
        self.B -= set(V_mod)
        for b in V_mod:
            if b not in self.V:
                for v in dg_full.D.neighbors(b):
                    if v in self.V:
                        self.B.add(b)


    def draw(self, ax, color='k') -> None:
        for v in self.D.nodes:
            ax.plot(v[0], v[1], f'.{color}')
        for u, v in self.D.edges:
            ax.plot([u[0], v[0]], [u[1], v[1]], f'.--{color}')
        for v in self.T.nodes:
            ax.plot(v[0], v[1], f's{color}')
        for u, v in self.T.edges:
            ax.plot([u[0], v[0]], [u[1], v[1]], f's-{color}')
    
    def undecomp(self, D_v:tuple) -> tuple:
        ret = None
        base = undecomp(D_v)
        for suffix in [None, "top", "bot"]:
            Tv = base + (suffix, ) if suffix else base
            if Tv in self.dV and D_v in self.dV[Tv]:
                assert ret is None
                ret = Tv
                
        return ret
    
    def decomp(self, Tv:tuple) -> List[tuple]:
        return [dv if dv in self.V else None for dv in decomp(Tv)]

    def Tv_ngb(self, Tv:tuple, dir:int) -> tuple:
        offset = [(0, -1), (1, 0), (0, 1), (-1, 0)] # S, E, N, W
        for ngb in self.T.neighbors(Tv):
            if ngb[0] == Tv[0] + offset[dir][0] and ngb[1] == Tv[1] + offset[dir][1]:
                return ngb

    def Tv_bot(self, u:tuple, v:tuple, Tv:tuple) -> tuple:
        lr = v[0] % 1 > 0.5
        tb = v[1] % 1 > 0.5
        if lr and tb:
            if u == (v[0]+0.5, v[1]):
                return self.Tv_ngb(Tv, 2)
            if u == (v[0], v[1]+0.5):
                return self.Tv_ngb(Tv, 1)
        elif not lr and tb:
            if u == (v[0]-0.5, v[1]):
                return self.Tv_ngb(Tv, 2)
            if u == (v[0], v[1]+0.5):
                return self.Tv_ngb(Tv, 3)
        elif lr and not tb:
            if u == (v[0]+0.5, v[1]):
                return self.Tv_ngb(Tv, 0)
            if u == (v[0], v[1]-0.5):
                return self.Tv_ngb(Tv, 1)
        elif not lr and not tb:
            if u == (v[0]-0.5, v[1]):
                return self.Tv_ngb(Tv, 0)
            if u == (v[0], v[1]-0.5):
                return self.Tv_ngb(Tv, 3)

    def common_ngb(self, Tu:tuple, Tv:tuple) -> tuple:
        for ngb in self.T.neighbors(Tu):
            if ngb in self.T.neighbors(Tv):
                return ngb


def get_subnode_coords(node:tuple, direction:str):
    if direction == 'SE':
        return (node[0]+0.25, node[1]-0.25)
    elif direction == 'SW':
        return (node[0]-0.25, node[1]-0.25)
    elif direction == 'NE':
        return (node[0]+0.25, node[1]+0.25)
    elif direction == 'NW':
        return (node[0]-0.25, node[1]+0.25)


def get_all_possible_ngb(Tv:tuple) -> List[tuple]:
    ret = []
    for cx, cy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        base = (Tv[0] + cx, Tv[1] + cy)
        ret.append(base)
        for suffix in ["top", "bot"]:
            ret.append(base + (suffix, ))
    return ret


def decomp(Tv:tuple) -> tuple:
    return [get_subnode_coords(Tv, d) for d in ["SE", "NE", "NW", "SW"]]


def undecomp(Dv:tuple) -> tuple:
    return (round(Dv[0]), round(Dv[1])) 


def generate_G_prime(D: nx.Graph) -> Tuple[nx.Graph, dict]:
    V_D = set(D.nodes)
    dV, complete, terrain_weight = {}, {}, {}

    while V_D:
        D_v = V_D.pop()
        tw_Tv = 4 * D.nodes[D_v]["terrain_weight"]
        T_v = undecomp(D_v)
        is_complete = True
        dv = decomp(T_v)
        for i, D_u in enumerate(dv):
            if D_u in V_D:
                V_D.remove(D_u)
            elif D_u != D_v:
                dv[i] = None
                is_complete = False

        for key, value in contract_single(dv, T_v, 4 * D.nodes[D_v]["terrain_weight"]).items():
            dV[key] = value["dV"]
            complete[key] = value["complete"]
            terrain_weight[key] = value["terrain_weight"]
    
    G_prime = nx.Graph()
    for u in dV.keys():
        for xc, yc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            v, vt, vb = (u[0]+xc, u[1]+yc), (u[0]+xc, u[1]+yc, "top"), (u[0]+xc, u[1]+yc, "bot")
            if v not in dV and vt not in dV and vb not in dV:
                continue

            u_l = [u] if u in dV else [x for x in [u + ("bot", ), u + ("top", )] if x in dV]
            v_l = [v] if v in dV else [x for x in [v + ("bot", ), v + ("top", )] if x in dV]
            for u, v in itertools.product(u_l, v_l):  
                if u[1] == v[1]:    # horizontal u -> v
                    u, v = (u, v) if u[0] < v[0] else (v, u)
                    if (dV[u][0] and dV[v][3]) or (dV[u][1] and dV[v][2]):
                        G_prime.add_edge(u, v, weight=terrain_edge_weight(u, v, D, dV))
                else:               # vertical u -> v
                    u, v = (u, v) if u[1] < v[1] else (v, u)
                    if (dV[u][1] and dV[v][0]) or (dV[u][2] and dV[v][3]):
                        G_prime.add_edge(u, v, weight=terrain_edge_weight(u, v, D, dV))

    if len(dV) == 1:
        G_prime.add_node(list(dV)[0])

    nx.set_node_attributes(G_prime, complete, "complete")
    nx.set_node_attributes(G_prime, terrain_weight, "terrain_weight")

    return G_prime, dV


def contract_single(dv:tuple, T_v:tuple, tw_Tv:float):
    ret = {}
    if (not dv[0] and not dv[1] and not dv[2] and not dv[3]):
        return ret
    elif (dv[0] and not dv[1] and dv[2] and not dv[3]):
        ret[T_v + ("bot",)] = {
            "dV": [dv[0], None, None, None],
            "complete": False,
            "terrain_weight": 4 * tw_Tv
        }
        ret[T_v + ("top",)] = {
            "dV": [None, None, dv[2], None],
            "complete": False,
            "terrain_weight": 4 * tw_Tv
        }
    elif (not dv[0] and dv[1] and not dv[2] and dv[3]):
        ret[T_v + ("bot",)] = {
            "dV": [None, dv[1], None, None],
            "complete": False,
            "terrain_weight": 4 * tw_Tv
        }
        ret[T_v + ("top",)] = {
            "dV": [None, None, None, dv[3]],
            "complete": False,
            "terrain_weight": 4 * tw_Tv
        }
    elif (dv[0] and not dv[1] and not dv[2] and not dv[3]) or (not dv[0] and dv[1] and not dv[2] and not dv[3]):
        ret[T_v + ("bot",)] = {
            "dV": dv,
            "complete": False,
            "terrain_weight": 4 * tw_Tv
        }
    elif (not dv[0] and not dv[1] and dv[2] and not dv[3]) or (not dv[0] and not dv[1] and not dv[2] and dv[3]):
        ret[T_v + ("top",)] = {
            "dV": dv,
            "complete": False,
            "terrain_weight": 4 * tw_Tv
        }
    else:
        ret[T_v] = {
            "dV": dv,
            "complete": sum([1 for item in dv if item]) == 4,
            "terrain_weight": 4 * tw_Tv
        }
    return ret


def contract(D:nx.Graph):
    H, dV = generate_G_prime(D)
    for v in H.nodes:
        H.nodes[v]["pos"] = v

    for u, v in H.edges:
        H[u][v]['weight'] = terrain_edge_weight(u, v, D, dV)
    
    if H.number_of_edges() != 0:
        min_edge_weight = min(H.edges(data='weight'), key=lambda x: x[2])[2]
        for u, v in H.edges:
            H[u][v]['weight'] -= min_edge_weight - 1

    return DecGraph(D, H, dV)


def terrain_edge_weight(u, v, D:nx.Graph, dV) -> float:
    if u[0] == v[0] and u[1] < v[1]:
        u1, u2 = get_subnode_coords(u, 'NW'), get_subnode_coords(u, 'NE')
        v1, v2 = get_subnode_coords(v, 'SW'), get_subnode_coords(v, 'SE')
    elif u[0] == v[0] and u[1] > v[1]:
        u1, u2 = get_subnode_coords(u, 'SW'), get_subnode_coords(u, 'SE')
        v1, v2 = get_subnode_coords(v, 'NW'), get_subnode_coords(v, 'NE')
    elif u[1] == v[1] and u[0] < v[0]:
        u1, u2 = get_subnode_coords(u, 'NE'), get_subnode_coords(u, 'SE')
        v1, v2 = get_subnode_coords(v, 'NW'), get_subnode_coords(v, 'SW')
    elif u[1] == v[1] and u[0] > v[0]:
        u1, u2 = get_subnode_coords(u, 'NW'), get_subnode_coords(u, 'SW')
        v1, v2 = get_subnode_coords(v, 'NE'), get_subnode_coords(v, 'SE')
    
    if (u1 in dV[u] and v1 in dV[v]) and (u2 in dV[u] and v2 in dV[v]):
        return D[u1][v1]['weight'] + D[u2][v2]['weight'] - D[u1][u2]['weight'] - D[v1][v2]['weight']
    elif (u1 in dV[u] and v1 in dV[v]) and not (u2 in dV[u] and v2 in dV[v]):
        return 2 * D[u1][v1]['weight']
    elif not (u1 in dV[u] and v1 in dV[v]) and (u2 in dV[u] and v2 in dV[v]):
        return 2 * D[u2][v2]['weight']
    else:
        assert True == False # should not reach here

