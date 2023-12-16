import sys
from collections import defaultdict, deque
from typing import List, Tuple

import networkx as nx

from MSTC_Star.utils.disjoint_set import DisjointSet

from LS_MCPP.graph import DecGraph
from LS_MCPP.utils import Helper


class ExtSTCPlanner:
    """ Extended STC Planner """

    @staticmethod
    def plan(r: tuple, dec_graph: DecGraph) -> List[Tuple[float, float]]:
        tree = ExtSTCPlanner.modified_kruskal(dec_graph.T)
        tree = ExtSTCPlanner.leaf_incomplete_vertex_rewiring(tree, dec_graph)
        pi = ExtSTCPlanner.full_stc(r, tree, dec_graph.dV)
        assert set(list(dec_graph.D.nodes)) == set(pi)
        return pi

    @staticmethod
    def full_stc(st: tuple, tree: nx.Graph, dV:dict) -> List[Tuple[float, float]]:
        if tree.number_of_nodes() == 1:
            Tv = list(tree.nodes)[0]
            path = [Dv for Dv in dV[Tv] if Dv is not None]
            if len(path) == 1:
                return path
            elif len(path) == 3:
                return path + [path[1], path[0]]
            else:
                return path + [path[0]]

        # select a neighbor of st as dummy pre-node to generate spiral cover
        dummy_parent = next(tree.neighbors(st))
        route = ExtSTCPlanner.spiral_route(st, dummy_parent, tree)

        traj = []
        L = len(route)

        last = dummy_parent
        for i, cur in enumerate(route):
            motion = ExtSTCPlanner.get_motion_coords(last, cur, dV)
            # interpolate a round trip when travsersing into leafnode
            if i <= L-2 and last == route[i+1]:
                motion += ExtSTCPlanner.get_round_trip_coords(last, cur, dV)

            traj.extend(motion)
            last = cur
        
        # remove consecutive repetition
        i = 0
        while i < len(traj)-1:
            if abs(traj[i+1][0] - traj[i][0]) + abs(traj[i+1][1] - traj[i][1]) == 0:
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
                    if t not in dV[DecGraph.undecomp(t)]:
                        t = (traj[i][0], traj[i+1][1])
                else:
                    t = (traj[i][0], traj[i+1][1])
                    if t not in dV[DecGraph.undecomp(t)]:
                        t = (traj[i+1][0], traj[i][1])
                traj.insert(i+1, t)
            elif i < len(traj)-3 and traj[i] == traj[i+2] and traj[i+1] == traj[i+3]:
                for _ in range(2):
                    traj.pop(i)
            elif i < len(traj)-4 and traj[i] == traj[i+2] == traj[i+4] and traj[i+1] == traj[i+3]:
                for _ in range(4):
                    traj.pop(i)
            else:
                i += 1         

        pi = traj[1:-1]
        assert Helper.is_path_valid(pi)
        return pi
    
    @staticmethod
    def modified_kruskal(G: nx.Graph):
        if G.number_of_nodes() == 1:
            return G

        E, node_map = [], {}
        costs = nx.get_edge_attributes(G, 'weight')
        type_order = lambda e: - G.nodes[e[0]]["complete"] - G.nodes[e[1]]["complete"]
        sorted_edges = sorted(costs.keys(), key=lambda x: (type_order(x), costs[x]))

        disjoint_set = DisjointSet()
        for n in G.nodes:
            node_map[n] = disjoint_set.make(n)

        # TODO: add turning-minimized tie-breaking for multiple equal-costly edges
        for edge in sorted_edges:
            s, t = node_map[edge[0]], node_map[edge[1]]
            root_s, root_t = disjoint_set.find(s), disjoint_set.find(t)

            if root_s != root_t:
                E.append((s.data, t.data))
                disjoint_set.union(root_s, root_t)

        return G.edge_subgraph(E)

    @staticmethod
    def leaf_incomplete_vertex_rewiring(tree: nx.Graph, dec_graph:DecGraph) -> None:
        T, dV = dec_graph.T, dec_graph.dV
        m_pairs = {}
        for ic in tree.nodes:
            if tree.degree[ic] == 1 and not tree.nodes[ic]["complete"]:
                u = next(tree.neighbors(ic))
                dir_u_ic = ExtSTCPlanner.get_motion_dir(u, ic)
                dv_ic = ExtSTCPlanner.rotate(dV[ic], -dir_u_ic)
  
                if dv_ic[0] and dv_ic[1] and not dv_ic[2]:                          #   |x   o|____E1___|o   ?|
                    v = DecGraph.Tv_ngb(ic, (1+dir_u_ic)%4)                         #   |?   o|         |o   ?|
                    if T.has_edge(ic, v):                                           #              /\         
                        dv_v = ExtSTCPlanner.rotate(dV[v], -dir_u_ic)               # S0 __|__     ||    _____    
                        if (not tree.nodes[v]["complete"] and dv_v[2] and dv_v[3]) or \
                                tree.nodes[v]["complete"]:                          #   |x   o|         |o   ?|
                            m_pairs[(u, ic)] = v                                    #   |?   o|         |o   ?|

                if dv_ic[2] and dv_ic[3] and not dv_ic[1]:                          #   |?   o|____W3___|o   x| 
                    v = DecGraph.Tv_ngb(ic, (3+dir_u_ic)%4)                         #   |?   o|         |o   ?|
                    if T.has_edge(ic, v):                                           #              /\      
                        dv_v = ExtSTCPlanner.rotate(dV[v], -dir_u_ic)               #    _____     ||    __|__ S0
                        if (not tree.nodes[v]["complete"] and dv_v[0] and dv_v[3]) or \
                                tree.nodes[v]["complete"]:                          #   |?   o|         |o   x|
                            m_pairs[(u, ic)] = v                                    #   |?   o|         |o   ?|
                
                if dv_ic[0] and dv_ic[3] and \
                    ((dv_ic[1] and not dv_ic[2]) or (dv_ic[2] and not dv_ic[1])):   # S0 __|__           _____        S0 __|__           _____     
                    v = DecGraph.Tv_ngb(ic, (dir_u_ic)%4)                           #   |x   o|         |x   o|         |o   x|         |x   o|
                    if T.has_edge(ic, v):                                           #   |o   o|         |o   o|         |o   o|         |o   o|
                        dv_v = ExtSTCPlanner.rotate(dV[v], -dir_u_ic)               #    -----     =>    --|--           -----     =>    --|--
                        if (not tree.nodes[v]["complete"] and dv_v[1] and dv_v[2]) or \
                                tree.nodes[v]["complete"]:                          #   |o   o|         |o   o|         |o   o|         |o   o|
                            m_pairs[(u, ic)] = v                                    #   |?   ?|         |?   ?|         |?   ?|         |?   ?|
                
                
        tree, added = tree.copy(), set()
        for u_ic, v in m_pairs.items():
            u, ic = u_ic
            # avoid duplicate edge adding
            if (ic, v) not in added and (v, ic) not in added:
                tree.remove_edge(u, ic)
                tree.add_edge(ic, v)
                added.add((ic, v))
        
        assert nx.is_connected(tree)
        return tree

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
