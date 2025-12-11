import sys
from collections import defaultdict, deque
from typing import List, Tuple
from itertools import combinations, product
import bisect

import networkx as nx

from lsmcpp.disjoint_set import DisjointSet

from lsmcpp.graph import DecGraph, undecomp
from lsmcpp.utils import Helper
from lsmcpp.benchmark.plan import TURN_COST_RATIO


class ExtSTCPlanner:
    """ Extended STC Planner """

    @staticmethod
    def plan(r: tuple, dec_graph: DecGraph) -> List[Tuple[float, float]]:
        tree = ExtSTCPlanner.modified_kruskal(dec_graph.T)
        pi = ExtSTCPlanner.full_stc(r, tree, dec_graph.dV)
        pi = ExtSTCPlanner.parallel_rewiring(pi, dec_graph.D)
        assert Helper.is_path_valid(pi)
        return pi

    @staticmethod
    def full_stc(st: tuple, tree: nx.Graph, dV:dict) -> List[Tuple[float, float]]:
        if tree.number_of_nodes() == 1:
            Tv = list(tree.nodes)[0]
            traj = [Dv for Dv in dV[Tv] if Dv is not None]
            if len(traj) == 1:
                pass
            elif len(traj) == 3:
                traj = traj + [traj[1], traj[0]]
            else:
                traj = traj + [traj[0]]
        else:
            # select a neighbor of st as dummy pre-node to generate spiral cover
            dummy_parent = next(tree.neighbors(st))
            route = ExtSTCPlanner.spiral_route(st, dummy_parent, tree)
            traj, L, last = [], len(route), dummy_parent
            for i, cur in enumerate(route):
                motion = ExtSTCPlanner.get_motion_coords(last, cur, dV)
                # interpolate a round trip when travsersing into leafnode
                if i <= L-2 and last == route[i+1]:
                    motion += ExtSTCPlanner.get_round_trip_coords(last, cur, dV)

                traj.extend(motion)
                last = cur
        
        ### POSTPROCESSING artifacts
        i = 0
        while i < len(traj)-1:
            if traj[i+1][0] == traj[i][0] and traj[i+1][1] == traj[i][1]:
                traj.pop(i)
            else:
                i += 1

        i = 0
        while i < len(traj)-1:
            dx = traj[i+1][0] - traj[i][0]
            dy = traj[i+1][1] - traj[i][1]

            if abs(dx) + abs(dy) == 0:
                traj.pop(i)
            elif abs(dx) + abs(dy) == 1:
                if dx * dy > 0:  # SW/NE
                    t = (traj[i+1][0], traj[i][1])
                    if t not in dV[undecomp(t)]:
                        t = (traj[i][0], traj[i+1][1])
                else:
                    t = (traj[i][0], traj[i+1][1])
                    if t not in dV[undecomp(t)]:
                        t = (traj[i+1][0], traj[i][1])
                traj.insert(i+1, t)
            elif i < len(traj)-3 and traj[i] == traj[i+2] and traj[i+1] == traj[i+3]:
                for _ in range(2):
                    traj.pop(i)
            elif i < len(traj)-4 and traj[i] == traj[i+2] == traj[i+4] and traj[i+1] == traj[i+3]:
                for _ in range(4):
                    traj.pop(i)
            elif i < len(traj)-5 and traj[i] == traj[i+2] == traj[i+4] and traj[i+1] == traj[i+5]:
                traj.pop(i+1)
                traj.pop(i+2)
            else:
                i += 1
        
        st = 0
        while traj[st] != traj[-1]:
            st += 1

        if st != len(traj)-1:
            return traj[st:]
        else:
            return traj
    
    @staticmethod
    def modified_kruskal(G: nx.Graph):
        if G.number_of_nodes() == 1:
            return G

        E, node_map = [], {}
        costs = nx.get_edge_attributes(G, 'weight')
        order_direction = lambda e: 0 if abs(e[0][0] - e[1][0]) == 0 else 1
        order_num_ngbs = lambda e: len(list(G.neighbors(e[0]))) + len(list(G.neighbors(e[1])))
        sorted_edges = sorted(costs.keys(), key=lambda x: (costs[x], order_direction(x), order_num_ngbs(x)))

        disjoint_set = DisjointSet()
        for n in G.nodes:
            node_map[n] = disjoint_set.make(n)

        for edge in sorted_edges:
            s, t = node_map[edge[0]], node_map[edge[1]]
            root_s, root_t = disjoint_set.find(s), disjoint_set.find(t)
            
            if root_s != root_t:
                E.append((s.data, t.data))
                disjoint_set.union(root_s, root_t)

        return G.edge_subgraph(E)
    
    @staticmethod
    def modified_kruskal_no_turn_reduction(G: nx.Graph):
        if G.number_of_nodes() == 1:
            return G

        E, node_map = [], {}
        costs = nx.get_edge_attributes(G, 'weight')
        sorted_edges = sorted(costs.keys(), key=lambda x:costs[x])
        disjoint_set = DisjointSet()
        for n in G.nodes:
            node_map[n] = disjoint_set.make(n)

        for edge in sorted_edges:
            s, t = node_map[edge[0]], node_map[edge[1]]
            root_s, root_t = disjoint_set.find(s), disjoint_set.find(t)
            
            if root_s != root_t:
                E.append((s.data, t.data))
                disjoint_set.union(root_s, root_t)

        return G.edge_subgraph(E)

    @staticmethod
    def kruskal_unweighted(G: nx.Graph):
        if G.number_of_nodes() == 1:
            return G

        E, node_map = [], {}
        disjoint_set = DisjointSet()
        for n in G.nodes:
            node_map[n] = disjoint_set.make(n)

        for edge in G.edges:
            s, t = node_map[edge[0]], node_map[edge[1]]
            root_s, root_t = disjoint_set.find(s), disjoint_set.find(t)
            
            if root_s != root_t:
                E.append((s.data, t.data))
                disjoint_set.union(root_s, root_t)

        return G.edge_subgraph(E)

    @staticmethod
    def spiral_route(start: tuple, dummy_parent: tuple, G: nx.Graph):
        """ using DFS with step-wise backtracing """
        # to avoid reach max recursion depth of 980 in python
        sys.setrecursionlimit(10**6)

        route = []
        visited_nodes = set([start])
        visited_edges = defaultdict(list)

        def ccw_traverse(node: tuple, parent: tuple, is_backtracing: bool):
            route.append(node)
            visited_nodes.add(node)

            if not is_backtracing and (parent, node) != (dummy_parent, start):
                visited_edges[parent].append(node)

            # get original counter-clockwise ordered neighbors
            ccw_ordered_ngbs = deque(ExtSTCPlanner.get_ccw_neighbors(G, node))
            ccw_ordered_ngbs.rotate(1 - ExtSTCPlanner.get_motion_dir(parent, node))

            for ngb in ccw_ordered_ngbs:
                if ngb and ngb not in visited_nodes:
                    ccw_traverse(ngb, node, False)

            # backtracing
            for node in visited_edges[parent]:
                visited_edges[parent].remove(node)
                ccw_traverse(parent, node, True)

        ccw_traverse(start, dummy_parent, False)

        return route

    @staticmethod
    def get_motion_coords(p: tuple, q: tuple, dV:dict):
        motion_direction = ExtSTCPlanner.get_motion_dir(p, q)
        dv_p = ExtSTCPlanner.rotate(dV[p], -motion_direction)
        dv_q = ExtSTCPlanner.rotate(dV[q], -motion_direction)
        if dv_p[3] and dv_q[2]:                                         #     (p)         (p)         (p)         (p)
            m = [dv_p[3], dv_q[2]]                                      #   3     0     3     0     x     0     x     0
        elif dv_p[3] and dv_q[2] is None:                               #   |                 |           |           |
            m = [dv_p[3], dv_p[0], dv_q[1], dv_q[0], dv_q[3], dv_q[0]]  #  \|/               \|/         \|/         \|/
        elif dv_p[3] is None and dv_q[2]:                               #   2     1     x     1     2     1     x     1
            m = [dv_p[2], dv_p[1], dv_p[0], dv_q[1], dv_q[2]]           #     (q)         (q)         (q)         (q)
        else:                                                           #   3     0     3     0     3     0     3     0
            m = [dv_p[2], dv_p[1], dv_p[0], dv_q[1], dv_q[0], dv_q[3], dv_q[0]]

        return [val for val in m if val is not None]

    @staticmethod
    def get_motion_dir(p: tuple, q: tuple):
        # direction from node p to node q
        if q[1] < p[1]:    # S
            return 0
        elif q[0] > p[0]:  # E
            return 1
        elif q[1] > p[1]:  # N
            return 2
        elif q[0] < p[0]:  # W
            return 3

    @staticmethod
    def get_round_trip_coords(last: tuple, pivot: tuple, dV: dict):
        motion_direction = ExtSTCPlanner.get_motion_dir(last, pivot)
        dv = ExtSTCPlanner.rotate(dV[pivot], -motion_direction)
        if dv[3] and dv[0]:                   #      |  
            m = [dv[3], dv[0]]                #   2  |  1
        elif dv[3] and dv[0] is None:         #   |  S /|\ 
            m = [dv[2], dv[3], dv[2]]         #  \|/    |
        elif dv[3] is None and dv[0]:         #   3 --> 0
            m = [dv[1], dv[0], dv[1]]
        else:
            m = []

        return [val for val in m if val is not None]
    
    @staticmethod
    def get_ccw_neighbors(G: nx.Graph, node):
        ccw_ordered_nodes = [None] * 4
        for ngb in G.neighbors(node):
            prio = ExtSTCPlanner.get_motion_dir(node, ngb)
            ccw_ordered_nodes[prio] = ngb
        return ccw_ordered_nodes

    @staticmethod
    def rotate(arr:list, rot:int) -> list:
        return [arr[(i-rot)%4] for i in range(4)]

    @staticmethod
    def parallel_rewiring(pi: List[Tuple], D:nx.Graph):
        is_ngb = lambda p, q: abs(p[0] - q[0]) + abs(p[1] - q[1]) == 0.5
        e_sorted = lambda p, q: (p, q) if p < q else (q, p)
        def _group_edges(_pi):
            _E, _dup, _ngbs = defaultdict(list), set(), defaultdict(set)
            for i in range(len(_pi)-1):
                e = e_sorted(_pi[i], _pi[i+1])
                _ngbs[e[0]].add(e[1])
                _ngbs[e[1]].add(e[0])
                _E[e].append((i,i+1))
                # assert len(E[e]) <= 2
                if len(_E[e]) >= 2:
                    _dup.add(e)
            return _E, _dup, _ngbs

        # Type A     
        E, dup, ngb_in_pi = _group_edges(pi)
        candidate = []
        for e1, e2 in combinations(dup, 2):
            if e1[0] != e2[1] and e1[1] != e2[0] and len(ngb_in_pi[e1[0]]) > 2 and len(ngb_in_pi[e1[1]]) > 2:
                if e_sorted(e1[0], e2[0]) in E and is_ngb(e1[1], e2[1]):
                    candidate.append((1, e1, e2))
                    dup.remove(e1)
                    dup.remove(e2)
                elif e_sorted(e1[1], e2[1]) in E and is_ngb(e1[0], e2[0]):
                    candidate.append((0, e1, e2))
                    dup.remove(e1)
                    dup.remove(e2)

        rm_list = set()
        for i, e1, e2 in candidate:
            ed_e1, ed_e2 = e1[i], e2[i]
            e1_matched, e2_matched = [e1], [e2]
            nonmatching = 0
            while nonmatching < len(dup):
                e = dup.pop()
                if e[1-i] == ed_e1:
                    ed_e1 = e[i]
                    bisect.insort(e1_matched, e)
                    nonmatching = 0
                elif e[1-i] == ed_e2:
                    ed_e2 = e[i]
                    bisect.insort(e2_matched, e)
                    nonmatching = 0
                else:
                    nonmatching += 1
                    dup.add(e)
            
            parallel_matched = 0
            if e1 == e1_matched[0] and e2 == e2_matched[0]:
                check_list = list(range(min(len(e1_matched), len(e2_matched))))
            else:
                check_list = range(-1, -min(len(e1_matched), len(e2_matched)), -1)
                
            for j in check_list:
                if is_ngb(e1_matched[j][0], e2_matched[j][0]) and is_ngb(e1_matched[j][1], e2_matched[j][1]):
                    parallel_matched += 1
                else:
                    break

            if parallel_matched != 0:
                order = 1 if e1_matched[0] == e1 else -1
                E1 = [E[e] for e in e1_matched[::order][:parallel_matched]][0]
                E2 = [E[e] for e in e2_matched[::order][:parallel_matched]][0]
                for a, b in product(E1, E2):
                    if a[0] == b[1] + 1:
                        rm_list = rm_list.union(
                            [a[0] + i for i in range(parallel_matched)] + \
                            [b[1] - i for i in range(parallel_matched)]
                        )
                    elif a[1] + 1 == b[0]:
                        rm_list = rm_list.union(
                            [a[1] - i for i in range(parallel_matched)] + \
                            [b[0] + i for i in range(parallel_matched)]
                        )

        pi = [pi[i] for i in range(len(pi)) if i not in rm_list]
        
        E, dup, _ = _group_edges(pi)
        rm_list = set()
        for e in dup:
            for i, ip1 in E[e]:
                if i > 0 and ip1 < len(pi)-1 and is_ngb(pi[i-1], pi[ip1+1]) and pi[i-1]!=pi[ip1] and pi[i]!=pi[ip1+1]:
                    rm_list = rm_list.union([i, ip1])
                    break
        
        pi = [pi[i] for i in range(len(pi)) if i not in rm_list]

        # Type B
        E = set([e_sorted(pi[i], pi[i+1]) for i in range(len(pi)-1)])
        in_E = lambda p, q: e_sorted(p, q) in E
        mdf_list = []
        for e1 in E:
            u1, u2 = e1
            if abs(u1[0] - u2[0]) == 0.5:   # vertical case
                cand = [[(u1[0], u1[1] + 0.5), (u2[0], u2[1] + 0.5), # v1, v2
                         (u1[0], u1[1] - 0.5), (u2[0], u2[1] - 0.5), # u3, u4
                         (u1[0], u1[1] - 1),   (u2[0], u2[1] - 1)],  # s1, s2
                        [(u1[0], u1[1] - 0.5), (u2[0], u2[1] - 0.5), # v1, v2
                         (u1[0], u1[1] + 0.5), (u2[0], u2[1] + 0.5), # u3, u4
                         (u1[0], u1[1] + 1),   (u2[0], u2[1] + 1)]]  # s1, s2
            else:   # horizontal case
                cand = [[(u1[0] + 0.5, u1[1]), (u2[0] + 0.5, u2[1]),
                         (u1[0] - 0.5, u1[1]), (u2[0] - 0.5, u2[1]),
                         (u1[0] - 1,   u1[1]), (u2[0] - 1,   u2[1])],
                        [(u1[0] - 0.5, u1[1]), (u2[0] - 0.5, u2[1]),
                         (u1[0] + 0.5, u1[1]), (u2[0] + 0.5, u2[1]),
                         (u1[0] + 1,   u1[1]), (u2[0] + 1,   u2[1])]]

            for v1, v2, u3, u4, s1, s2 in cand:
                e2 = (v1, v2)
                turning_cost = 2 * TURN_COST_RATIO if in_E(v1, v2) else 0
                if in_E(*e2) and in_E(u1, u3) and in_E(u2, u4) and in_E(u3, s1) and in_E(u4, s2) and \
                   D[u1][u3]['weight'] + D[u2][u4]['weight'] + D[v1][v2]['weight'] > \
                   D[u3][u4]['weight'] + D[u1][v1]['weight'] + D[u2][v2]['weight'] + turning_cost:
                    mdf_list.append((e1, e2, u3, u4, s1, s2))

        for e1, e2, u3, u4, s1, s2 in mdf_list:
            u1, u2 = e1
            v1, v2 = e2
            if not (pi.count(u1) == 1 and pi.count(u2) == 1 and pi.count(v1) == 1 and pi.count(v2) == 1) or \
               not (in_E(*e1) and in_E(*e2) and in_E(u1, u3) and in_E(u2, u4) and in_E(u3, s1) and in_E(u4, s2)):
                continue

            iu1 = pi.index(u1)
            pi = (pi[:iu1] + pi[iu1+2:]) if pi[iu1+1] == u2 else (pi[:iu1-1] + pi[iu1+1:])

            iv1 = pi.index(v1)
            if pi[iv1+1] == v2:
                iv1, iv2 = iv1, iv1 + 1
            else:
                iv2 = iv1
                iv1 = iv1 - 1
                v1, v2 = v2, v1

            if is_ngb(v1, u1) and is_ngb(u2, v2):
                pi = pi[:iv1+1] + [u1, u2] + pi[iv2:]
            elif is_ngb(v1, u2) and is_ngb(u1, v2):
                pi = pi[:iv1+1] + [u2, u1] + pi[iv2:]
            else:
                assert True == False
            

            E.remove(e_sorted(v1, v2))
            E.remove(e_sorted(u1, u3))
            E.remove(e_sorted(u2, u4))
        
        return pi
    
    @staticmethod
    def root_align(pi:List[tuple], rd:tuple) -> List[tuple]:
        for i in range(len(pi)):
            if pi[i] == rd:
                return pi[i:-1] + pi[:i+1]
            