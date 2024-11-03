from __future__ import annotations
import os
from networkx import Graph
import yaml
import time
import pickle
from pathlib import Path

import LS_MCPP

from typing import Dict, List, Set

import networkx as nx
import numpy as np

from MSTC_Star.mcpp.stc_planner import STCPlanner
from MSTC_Star.mcpp.rtc_planner import RTCPlanner
from MSTC_Star.utils.nx_graph import navigate

from MIP_MCPP.instance import Instance
from MIP_MCPP.mcpp_planner import mfc_plan, mip_plan, STC_on_MMRTC_sol
from MIP_MCPP.model import Model
from MIP_MCPP.warmstarter import WarmStarter

from LS_MCPP.estc import ExtSTCPlanner
from LS_MCPP.graph import DecGraph, contract

from benchmark.instance import MCPP
from benchmark.plan import Heading


### ---------------- FOR non-deconflicted solutions ---------------- ###

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
    def path_cost(D:nx.Graph, pi:list, default_heading:Heading=Heading.N) -> float:
        """ the path cost with turning costs """
        cost, heading_last = 0, default_heading
        for i in range(len(pi)-1):
            heading = Heading.get(pi[i], pi[i+1])
            cost += (D[pi[i]][pi[i+1]]["weight"] + Heading.rot_cost(heading_last, heading))
            heading_last = heading

        return cost


def Voronoi_sol(mcpp:MCPP) -> Solution:
    dg = contract(mcpp._G_legacy)
    R = [dg.undecomp(mcpp.legacy_vertex(r)) for r in mcpp.R]
    R_D = [mcpp.legacy_vertex(r) for r in mcpp.R]

    paths = []
    for idx, val in enumerate(nx.voronoi_cells(dg.D, R_D).items()):
        ri, Vi = val
        dgi = contract(dg.D.subgraph(Vi))
        paths.append(ExtSTCPlanner.plan(R[idx], dgi))

    costs = [Solution.path_cost(dg.D, pi) for pi in paths]
    return Solution([pi for pi in paths], np.array(costs))


def MFC_sol(mcpp:MCPP) -> Solution:
    dg = contract(mcpp._G_legacy)
    R = [dg.undecomp(mcpp.legacy_vertex(r)) for r in mcpp.R]

    rtc_planner = RTCPlanner(dg.T, R, mcpp.k)
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
    costs = [Solution.path_cost(dg.D, pi) for pi in paths]
    return Solution(paths, np.array(costs))


def MSTCStar_sol(mcpp:MCPP) -> Solution:
    class ModifiedMSTCStarPlanner(STCPlanner):
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
            path.extend([depot] + navigate(self.H, depot_small, serv_pts[0]))
            L, num_of_served = len(serv_pts), 1

            for i in range(L-1):
                if num_of_served == self.capacity:
                    num_of_served = 0
                    beta = navigate(self.H, path[-1], depot_small)
                    alpha = navigate(self.H, depot_small, serv_pts[i])
                    path.extend(beta[1:-1] + [depot] + alpha)

                l1 = abs(serv_pts[i+1][0] - serv_pts[i][0]) + \
                    abs(serv_pts[i+1][1] - serv_pts[i][1])

                if l1 != 0.5:
                    gamma = navigate(self.H, serv_pts[i], serv_pts[i+1])
                    path.extend(gamma[1:-1])

                path.append(serv_pts[i+1])
                num_of_served += 1

            if path[-1] != depot:
                path.extend(navigate(self.H, path[-1], depot_small)[1:] + [depot])

            return path, self.__get_travel_weights__(path)

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


    dg = contract(mcpp._G_legacy)
    R = [dg.undecomp(mcpp.legacy_vertex(r)) for r in mcpp.R]
    R_D = [mcpp.legacy_vertex(r) for r in mcpp.R]
    planner = ModifiedMSTCStarPlanner(dg.T, mcpp.k, R, R_D, float('inf'), dg, True)
    plans = planner.allocate()
    paths, _ = planner.simulate(plans, False)

    costs = [Solution.path_cost(dg.D, pi[1:-1]) for pi in paths]
    return Solution([pi[1:-1] for pi in paths], np.array(costs))


def MIP_sol(mcpp:MCPP, sol_edges) -> Solution:
    dg = contract(mcpp._G_legacy)
    v2pos = {}
    for vid, v in enumerate(dg.T):
        v2pos[vid] = v

    paths = []
    for i, E in enumerate(sol_edges):
        VTi = set()
        for u, v in E:
            VTi.add(v2pos[u])
            VTi.add(v2pos[v])
        Ti = dg.T.subgraph(VTi)
        Vdi = set()
        for Tv in VTi:
            Vdi = Vdi.union([dv for dv in dg.dV[Tv] if dv is not None])
        Di = dg.D.subgraph(Vdi)
        if VTi != set():
            pi = ExtSTCPlanner.plan(list(VTi)[0], DecGraph(Di, Ti, dg.dV))
        else:
            pi = []
        paths.append(pi)

    costs = [Solution.path_cost(dg.D, pi) for pi in paths]
    return Solution([pi for pi in paths], np.array(costs))


def MIP_solve(mcpp:MCPP, rmv_ratios, seeds):
    with open(os.path.join("data", "mcpp", "gurobi_cfg.yaml")) as f:
        solver_args = yaml.load(f, yaml.Loader)
        solver_args["OptimalityTol"] = float(solver_args["OptimalityTol"])

    save_dir = os.path.join("data", "mcpp", "solutions", mcpp.name, "MIP")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for rmv_ratio in rmv_ratios:
        for i, mutant in enumerate(mcpp.randomized_mutants([rmv_ratio], seeds)):
            if not os.path.exists(os.path.join(save_dir, f"{rmv_ratio:.3f}-{seeds[i]}.solu")):
                dg = contract(mutant._G_legacy)
                R = set([dg.undecomp(mcpp.legacy_vertex(r)) for r in mcpp.R])
                istc_graph, istc_R, pos2v = nx.Graph(), [], {}
                for vid, v in enumerate(dg.T):
                    istc_graph.add_node(vid, pos=v)
                    pos2v[v] = vid
                    if v in R:
                        istc_R.append(vid)
                for u, v in dg.T.edges:
                    istc_graph.add_edge(pos2v[u], pos2v[v], weight=dg.T[u][v]["weight"])

                istc = Instance(istc_graph, istc_R, f"{mutant.width}x{mutant.height}-{mutant.name}")
                model = Model(istc)
                ts = time.perf_counter()
                H, perc_vars_removed = model.apply_heur(None, None)
                model.wrapup(solver_args, H)
                model = WarmStarter.apply(model, "RTC", H)
                sol_edges, sol_verts = model.solve()
                runtime = time.perf_counter() - ts

                res_str = ",\t".join(str(s) for s in [
                    mutant.name,
                    model.num_vars * (1 - round(perc_vars_removed, 3))
                ])

                if sol_edges == []:
                    obj_val, mip_gap = "/", "/"
                else:
                    obj_val = round(model.model.objVal, 3)
                    mip_gap = round(model.model.MIPGap, 3)
                    with open(os.path.join(save_dir, f"{rmv_ratio:.3f}-{seeds[i]}.solu"), "wb") as f:
                        pickle.dump((sol_edges, runtime), f, pickle.HIGHEST_PROTOCOL)

                res_str += ",\t".join(str(s) for s in [
                    "",
                    obj_val,
                    mip_gap,
                    round(model.model.objBound, 3),
                    round(model.model.RunTime, 3)
                ])

                print(res_str)
