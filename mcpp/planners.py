import heapq
import numpy as np
import networkx as nx

from lsmcpp.benchmark.solution import Solution, Heading
from lsmcpp.rooted_tree_cover import RootedTreeCover

from lsmcpp.estc import ExtSTCPlanner
from lsmcpp.graph import DecGraph, contract

from lsmcpp.benchmark.instance import MCPP

# MCPP baseline algorithms


def Voronoi_planner(mcpp:MCPP) -> Solution:
    dg = contract(mcpp._G_legacy)
    R = [dg.undecomp(mcpp.legacy_vertex(r)) for r in mcpp.R]
    R_D = [mcpp.legacy_vertex(r) for r in mcpp.R]

    paths = []
    for idx, val in enumerate(nx.voronoi_cells(dg.D, R_D).items()):
        ri, Vi = val
        dgi = contract(dg.D.subgraph(Vi))
        paths.append(ExtSTCPlanner.plan(R[idx], dgi))

    costs = [Solution.path_cost(pi, dg.D) for pi in paths]
    return Solution([pi for pi in paths], np.array(costs))


def MFC_planner(mcpp:MCPP) -> Solution:
    dg = contract(mcpp._G_legacy)
    R = [dg.undecomp(mcpp.legacy_vertex(r)) for r in mcpp.R]

    rtc_planner = RootedTreeCover(dg.T, R, mcpp.k)
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
    paths = []
    for i, r in enumerate(R):
        Ti = dg.T.subgraph(nodes[r])
        Vdi = set()
        for Tv in nodes[r]:
            Vdi = Vdi.union([dv for dv in dg.dV[Tv] if dv is not None])
        Di = dg.D.subgraph(Vdi)
        pi = ExtSTCPlanner.plan(r, DecGraph(Di, Ti, dg.dV))
        paths.append(pi)
    costs = [Solution.path_cost(pi, dg.D) for pi in paths]
    return Solution(paths, np.array(costs))


class MSTCStarPlanner:

    """ rewrite to support incomplete terrain verts """

    def __init__(self, G:nx.Graph, k, R, R_D, cap, dg:DecGraph, cut_off_opt=True):
        self.G = G
        self.k = k
        self.R = R
        self.R_D_map = {R[i]:R_D[i] for i in range(k)}
        self.capacity = cap
        self.H = dg.D
        self.rho = ExtSTCPlanner.plan(R[0], dg)
        self.cut_off_opt = cut_off_opt

    def allocate(self, alloc_filename=None):
        num_of_nodes = self.__split(len(self.rho)-1, self.k, {})

        start, plans = 0, {}
        for i, n in enumerate(num_of_nodes):
            end = start + n
            plans[self.R[i]] = self.rho[start:end]
            start = end

        if self.cut_off_opt:
            _, weights = self.simulate(plans)
            self.__optimal_cut_opt(weights, plans, debug=True)

        self.__write_alloc_file(plans, alloc_filename)

        return plans

    def simulate(self, plans, is_print=True):
        paths, weights = [[] for _ in range(self.k)], [0] * self.k
        for idx, val in enumerate(plans.items()):
            depot, serv_pts = val
            path, weight = self.__sim(depot, serv_pts)
            paths[idx], weights[idx] = path, weight

            if is_print:
                print(f'#{idx} Total Weights: {weights[idx]}')
        if is_print:
            print(f'---\nFinal Max Weights: {max(weights)}')

        return paths, weights

    def __write_alloc_file(self, plans, alloc_filename=None):
        if not alloc_filename:
            return

        f = open(alloc_filename, 'w')
        for idx, val in enumerate(plans.items()):
            depot, serv_pts = val
            xs, ys = zip(*serv_pts)
            ns = len(serv_pts)
            f.writelines(
                ' '.join([str(xs[i])+','+str(ys[i]) for i in range(ns)])+'\n')
        f.close()

    def __split(self, N, K, res):
        if (N, K) in res:
            return res[(N, K)]

        if K == 1:
            return [N]

        left = K // 2
        left_N = round(N * left / K)
        left_res = self.__split(left_N, left, res)
        res[(left_N, left)] = left_res

        right = K - left
        right_N = N - left_N
        right_res = self.__split(right_N, right, res)
        res[(right_N, right)] = right_res

        return left_res + right_res

    def __optimal_cut_opt(self, weights: list, plans: dict, debug=False):
        opt = max(weights)
        cur_iter, num_of_iters = 0, 1e3
        while cur_iter < num_of_iters:
            r_min = min(list(range(self.k)), key=lambda x: weights[x])
            r_max = max(list(range(self.k)), key=lambda x: weights[x])
            print(f'iter #{cur_iter}: rmin={r_min}, rmax={r_max}, max weight={opt: .3f}', end=' ')
            # clockwise cutoff opt
            clw = self.__get_intermediate_r_index(r_min, r_max, -1)
            # counter-clockwise cutoff opt
            ccw = self.__get_intermediate_r_index(r_min, r_max, 1)
            # select smaller loop
            r_index = clw if len(clw) < len(ccw) else ccw
            self.__find_optimial_cut(r_index, weights, plans, debug)

            for i in sorted(list(range(self.k)), key=lambda x: weights[x]):
                print(f', {i}: {weights[i]: .3f}', end=' ')
            print(',')

            if max(weights) >= opt:
                print('MSTC-Star Cutoff OPT Finished')
                break
            else:
                opt = max(weights)
                cur_iter += 1

    def __sim(self, depot, serv_pts):
        if serv_pts == []:
            return [depot], 0

        path = []
        depot_small = self.R_D_map[depot]
        path.extend([depot] + shortest_path_with_turn_cost(self.H, depot_small, serv_pts[0]))
        L, num_of_served = len(serv_pts), 1

        for i in range(L-1):
            if num_of_served == self.capacity:
                num_of_served = 0
                beta = shortest_path_with_turn_cost(self.H, path[-1], depot_small)
                alpha = shortest_path_with_turn_cost(self.H, depot_small, serv_pts[i])
                path.extend(beta[1:-1] + [depot] + alpha)

            l1 = abs(serv_pts[i+1][0] - serv_pts[i][0]) + \
                abs(serv_pts[i+1][1] - serv_pts[i][1])

            if l1 != 0.5:
                gamma = shortest_path_with_turn_cost(self.H, serv_pts[i], serv_pts[i+1])
                path.extend(gamma[1:-1])

            path.append(serv_pts[i+1])
            num_of_served += 1

        if path[-1] != depot:
            path.extend(shortest_path_with_turn_cost(self.H, path[-1], depot_small)[1:] + [depot])

        return path, Solution.path_cost(path[1:-1], self.H)

    def __find_optimial_cut(self, r_index, weights, plans, debug=True):
        """ find optimal-cut point of U{P_cutoff_index} using binary search """

        plan, N = [], {}
        r_first, r_last = r_index[0], r_index[-1]
        for ri in r_index:
            plan += plans[self.R[ri]]
            N[ri] = len(plans[self.R[ri]])

        old_weight_max = max(weights)
        old_weight_sum = sum([weights[ri] for ri in r_index])
        opt = (-1, old_weight_max, old_weight_sum, {}, weights)
        first, last = 0, N[r_last] + N[r_first] - 1

        if debug:
            print(f'--- Cutoff point={N[r_first]}', end='\t')
            for ri in r_index:
                print(f'{ri}: {weights[ri]: .3f}', end='\t')
            print(f'Weight Max: {old_weight_max: .3f}, Weight Sum: {old_weight_sum: .3f}')

        old_N_r_first, old_N_r_last = N[r_first], N[r_last]
        while first < last:
            c = (first + last) // 2
            N[r_first] = c
            N[r_last] = old_N_r_first + old_N_r_last - c

            plan_moved, weight_moved = {}, weights.copy()
            start, max_weight, sum_weight = 0, 0, 0

            if debug:
                print(f'--- Cutoff point={c}', end='\t')

            for ri, ni in N.items():
                end = start + ni
                _, weight = self.__sim(self.R[ri], plan[start:end])
                plan_moved[self.R[ri]] = plan[start:end]
                weight_moved[ri] = weight
                sum_weight += weight
                max_weight = max(max_weight, weight)
                start = end
                if debug:
                    print(f'{ri}: {weight: .3f}', end='\t')
            if debug:
                print(f'Weight Max: {max_weight: .3f}, Weight Sum: {sum_weight: .3f}')

            if max_weight < opt[1]:
                opt = (c, max_weight, sum_weight, plan_moved, weight_moved)
            elif max_weight == opt[1] and sum_weight < opt[2]:
                opt = (c, max_weight, sum_weight, plan_moved, weight_moved)

            if weight_moved[r_first] < weight_moved[r_last]:
                first = c + 1
            elif weight_moved[r_first] > weight_moved[r_last]:
                last = c - 1
            else:
                break

        if opt[0] != -1:
            for ri in r_index:
                weights[ri] = opt[4][ri]
                plans[self.R[ri]] = opt[3][self.R[ri]]
            output_str = f'--- Found OPT-CUT: c={opt[0]}, max weight={opt[1]}({old_weight_max}), weight sum={opt[2]}({old_weight_sum})'
        else:
            output_str = '--- Did not found OPT-CUT'

        if debug:
            print(output_str)

    def __get_intermediate_r_index(self, r_min, r_max, d_ri):
        r_mid, ri = [r_min], r_min
        while ri != r_max:
            ri = (ri + d_ri) % self.k
            r_mid.append(ri)

        return r_mid if d_ri == 1 else list(reversed(r_mid))

    
def shortest_path_with_turn_cost(H:nx.Graph, start, goal):
    # A* search on H with Solution.path_cost
    heur = lambda p, q: np.hypot(p[0]-q[0], p[1]-q[1])

    OPEN = []
    CLOSED = set()
    g = {start: 0}
    f = {start: heur(start, goal)}
    parent = {start: None}
    OPEN.append((f[start], None, start))
    while OPEN:
        _, predecessor, current = heapq.heappop(OPEN)
        if current == goal:
            break
        CLOSED.add(current)

        for neighbor in H.neighbors(current):
            if neighbor in CLOSED:
                continue
            if predecessor is None:
                tentative_g = g[current] + H[current][neighbor]['weight']
            else:
                heading_last = Heading.get(predecessor, current)
                heading = Heading.get(current, neighbor)
                tentative_g = g[current] + H[current][neighbor]['weight'] + Heading.rot_cost(heading_last, heading)
            if neighbor not in g or tentative_g < g[neighbor]:
                parent[neighbor] = current
                g[neighbor] = tentative_g
                f[neighbor] = tentative_g + heur(neighbor, goal)
                heapq.heappush(OPEN, (f[neighbor], current, neighbor))

    path = []
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def MSTCStar_planner(mcpp:MCPP) -> Solution:

    dg = contract(mcpp._G_legacy)
    R = [dg.undecomp(mcpp.legacy_vertex(r)) for r in mcpp.R]
    R_D = [mcpp.legacy_vertex(r) for r in mcpp.R]
    planner = MSTCStarPlanner(dg.T, mcpp.k, R, R_D, float('inf'), dg, True)
    plans = planner.allocate()
    paths, _ = planner.simulate(plans, False)

    costs = [Solution.path_cost(pi[1:-1], dg.D) for pi in paths]
    return Solution([pi[1:-1] for pi in paths], np.array(costs))
