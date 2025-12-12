from __future__ import annotations
from typing import List, Tuple, Set, Dict
from itertools import combinations, product
from collections import defaultdict

import numpy as np
import networkx as nx

from lsmcpp.benchmark.solution import Solution
from lsmcpp.estc import ExtSTCPlanner
from lsmcpp.planners import MSTCStarPlanner
from lsmcpp.benchmark.instance import MCPP
from lsmcpp.graph import DecGraph, contract
from lsmcpp.disjoint_set import DisjointSet
from lsmcpp.turn_minimization.rectangle import Rectangle, HORIZONTAL, VERTICAL


edge_key = lambda ra, rb: (ra, rb) if ra.lowerleft < rb.lowerleft else (rb, ra)


def compute_bipartite_graph(dg:DecGraph) -> Tuple[nx.DiGraph, nx.Graph]:

    rects:List[Rectangle] = []
    for xc, yc in dg.T.nodes:
        xn, yn = xc + 1, yc + 1
        rects.append(Rectangle((xc, yc), (xn, yn)))

    I, E, V, H = nx.DiGraph(), set(), set(), set()
    for ra, rb in combinations(rects, 2):
        if ra.lowerleft[0] == rb.lowerleft[0]:
            if ra.lowerleft[1] + ra.height == rb.lowerleft[1]:
                key = edge_key(ra, rb)
                E.add(key)
                v = (ra.lowerleft, rb.lowerleft)
                V.add(v)
                I.add_node(v, pair=key, bipartite=VERTICAL)
            if ra.lowerleft[1] == rb.lowerleft[1] + rb.height:
                key = edge_key(ra, rb)
                E.add(key)
                v = (ra.lowerleft, rb.lowerleft)
                V.add(v)
                I.add_node(v, pair=key, bipartite=VERTICAL)
           
        if ra.lowerleft[1] == rb.lowerleft[1]:
            if ra.lowerleft[0] + ra.width == rb.lowerleft[0]:
                key = edge_key(ra, rb)
                E.add(key)
                h = (ra.lowerleft, rb.lowerleft)
                H.add(h)
                I.add_node(h, pair=key, bipartite=HORIZONTAL)
            if ra.lowerleft[0] == rb.lowerleft[0] + rb.width:
                key = edge_key(ra, rb)
                E.add(key)
                h = (ra.lowerleft, rb.lowerleft)
                H.add(h)
                I.add_node(h, pair=key, bipartite=HORIZONTAL)

    I.graph["V"], I.graph["H"] = V, H
    for h, v in product(H, V):
        if v[0] == h[0] or v[0] == h[1] or v[1] == h[0] or v[1] == h[1]:
            I.add_edge(h, v, capacity=1)
    
    return I, nx.Graph(E)


def bipartite_min_vertex_cover(I:nx.DiGraph, G:nx.Graph) -> List[Rectangle]:
    # find maximum independent set
    src, dst = "s", "t"
    I.add_edges_from([(src, h) for h in I.graph["H"]], capacity=1)
    I.add_edges_from([(v, dst) for v in I.graph["V"]], capacity=1)

    residual, H, V = nx.Graph(), set(), set()
    _, flow_dict = nx.maximum_flow(I, src, dst)
    for u, neighbors in flow_dict.items():
        for v, flow in neighbors.items():
            if flow == 0:
                residual.add_edge(u, v)
                if u != src and u != dst:
                    if I.nodes[u]["bipartite"] == HORIZONTAL:
                        H.add(u)
                    else:
                        V.add(u)
                if v != src and v != dst:
                    if I.nodes[v]["bipartite"] == HORIZONTAL:
                        H.add(v)
                    else:
                        V.add(v)
    
    C, D = set(), set()
    for h in H:
        if not nx.has_path(residual, src, h):
            C.add(h)
    for v in V:
        if nx.has_path(residual, src, v):
            D.add(v)
    
    C, D = set(I.graph["H"]) - C, set(I.graph["V"]) - D 
    
    # merge
    ds, ds_nodes = DisjointSet(), {}
    for grid in G.nodes:
        ds_nodes[grid] = ds.make(grid)

    for v in set.union(C, D):
        grid_a, grid_b = I.nodes[v]["pair"]
        ds.union(ds_nodes[grid_a], ds_nodes[grid_b])

    roots = defaultdict(list)
    for grid in G.nodes:
        roots[ds.find(ds_nodes[grid])].append(grid)

    B:List[Rectangle] = []
    for _, val in roots.items():
        val.sort(key=lambda x:x.lowerleft[0])
        val.sort(key=lambda x:x.lowerleft[1])
        Rectangle: Rectangle = val[0].copy()
        for grid in val[1:]:
            if Rectangle.merge(grid):
                Rectangle = Rectangle.merge(grid)
            else:
                B.append(Rectangle)
                Rectangle = grid
        B.append(Rectangle)

    return B


def generate_spanning_tree(B:List[Rectangle]) -> nx.Graph:
    grids, T = {}, nx.Graph()
    for Rectangle in B:
        grids[Rectangle] = Rectangle.grids()
        if len(grids[Rectangle]) == 1:
            T.add_node(grids[Rectangle][0])

        for grid_a, grid_b in combinations(grids[Rectangle], 2):
            if grid_a.is_adjacent(grid_b):
                T.add_edge(grid_a, grid_b)

    E = defaultdict(set)
    for Rectangle_a, Rectangle_b in combinations(B, 2):
        for grid_a, grid_b in product(grids[Rectangle_a], grids[Rectangle_b]):
            if grid_a.is_adjacent(grid_b):
                E[edge_key(Rectangle_a, Rectangle_b)].add(edge_key(grid_a, grid_b))

    def f(grid:Rectangle) -> int:
        deg = T.degree(grid)
        assert 0 <= deg <= 4
        if deg == 1 or deg == 3:
            return 2
        if deg == 0 or deg == 4:
            return 4
        if deg == 2:
            return 0
            # ngb_a, ngb_b = list(T.neighbors(grid))
            # if grid.left_rect == ngb_a and grid.right_rect == ngb_b or \
            #    grid.right_rect == ngb_a and grid.left_rect == ngb_b or \
            #    grid.top_rect == ngb_a and grid.bot_rect == ngb_b or \
            #    grid.bot_rect == ngb_a and grid.top_rect == ngb_b:
            #     return 0
            # else:
            #     return 2
    
    def g(grid_a:Rectangle, grid_b:Rectangle) -> int:
        ret = - f(grid_a) - f(grid_b)
        T.add_edge(grid_a, grid_b)
        ret += f(grid_a) + f(grid_b)
        T.remove_edge(grid_a, grid_b)
        return ret

    S = set()
    while not nx.is_connected(T):
        min_g_pair = (float('inf'), None, None)
        for key, val in E.items():
            sub_E = list(val)
            gs = [g(grid_a, grid_b) for grid_a, grid_b in sub_E]
            idx = np.argmin(gs)
            if gs[idx] < min_g_pair[0]:
                min_g_pair = (gs[idx], key, sub_E[idx])
        
        if not nx.has_path(T, min_g_pair[2][0], min_g_pair[2][1]):
            T.add_edge(*min_g_pair[2])
            S = S.union(min_g_pair[1])

        E.pop(min_g_pair[1])

    tree = nx.Graph()
    for node in T.nodes:
        tree.add_node(node.lowerleft)
    for u, v in T.edges:
        tree.add_edge(u.lowerleft, v.lowerleft)

    return tree


def TMSTCStar_planner(mcpp:MCPP) -> Solution:
    dg = contract(mcpp._G_legacy)
    I, G = compute_bipartite_graph(dg)
 
    merged_rects = bipartite_min_vertex_cover(I, G)
    T = generate_spanning_tree(merged_rects)
    
    R = [dg.undecomp(mcpp.legacy_vertex(r)) for r in mcpp.R]
    R_D = [mcpp.legacy_vertex(r) for r in mcpp.R]

    # replace the initial STC path with by the one using the generated spanning tree
    planner = MSTCStarPlanner(dg.T, mcpp.k, R, R_D, float('inf'), dg, True)
    pi = ExtSTCPlanner.full_stc(R[0], T, dg.dV)
    planner.rho = ExtSTCPlanner.parallel_rewiring(pi, dg.D)

    plans = planner.allocate()
    paths, _ = planner.simulate(plans, False)

    costs = [Solution.path_cost(pi[1:-1], dg.D) for pi in paths]
    return Solution([pi[1:-1] for pi in paths], np.array(costs))

