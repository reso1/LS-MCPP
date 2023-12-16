from __future__ import annotations

import time
from typing import List, Tuple

import LS_MCPP

import numpy as np
from scipy.special import softmax

from MIP_MCPP.instance import Instance

from LS_MCPP.graph import DecGraph
from LS_MCPP.operator import Operator
from LS_MCPP.pool import PrioType, SampleType, Pool, GrowPool, DeDupPool, ExcPool
from LS_MCPP.solution import Solution
from LS_MCPP.utils import DuplicationRec, Helper


class LocalSearchMCPP:

    def __init__(
        self, 
        obj,
        sol:Solution,
        prio_type: PrioType,
        verbose=True,
        R=None
    ) -> None:
        if type(obj) is DecGraph:
            self.init_with_decgraph(obj, R, sol, prio_type, verbose)
        elif type(obj) is Instance:
            self.init_with_instance(obj, sol, prio_type, verbose)
    
    def init_with_instance(
        self,
        istc:Instance,
        sol:Solution, 
        prio_type: PrioType,
        verbose=True
    ) -> None:
        self.vbs = verbose
            
        self.T = Helper.to_position_graph(istc.G)
        self.D = DecGraph.get_decomposed_graph_complete(self.T)
        self.R = [istc.G.nodes[r]["pos"] for r in istc.R]

        self.k = len(self.R)
        self.I = list(range(self.k))
        self.R_D = [DecGraph.get_subnode_coords(r, "SE") for r in self.R]
        self.dup_rec = DuplicationRec()
        self.decgraphs: List[DecGraph] = []
        self.v2pos = {v:p for v, p in enumerate(self.D.nodes)}
        self.pos2v = {p:v for v, p in enumerate(self.D.nodes)}
        
        for i in self.I:
            Di = self.D.subgraph(set(sol.Pi[i])).copy()
            G_prime, dV = DecGraph.generate_G_prime(Di)
            self.decgraphs.append(DecGraph(Di, G_prime, dV))
            for v in sol.Pi[i]:
                self.dup_rec.dup(v, i)

        self.sol_0 = sol
        self.pools: List[Pool] = [
            GrowPool(self.D, self.R, self.decgraphs, self.dup_rec, prio_type),
            DeDupPool(self.D, self.R, self.decgraphs, self.dup_rec, self.R_D, prio_type),
            ExcPool(self.D, self.R, self.decgraphs, self.dup_rec, self.R_D, prio_type),
        ]
        self.num_of_pools = len(self.pools)
        self.rhos = np.array([1.0 for _ in range(self.num_of_pools)])

    def init_with_decgraph(
        self,
        dg:DecGraph,
        R:list,
        sol:Solution, 
        prio_type: PrioType,
        verbose=True
    ) -> None:
        self.vbs = verbose
            
        self.T, self.D, self.R = dg.T, dg.D, R

        self.k = len(self.R)
        self.I = list(range(self.k))
        self.R_D = []
        self.dup_rec = DuplicationRec()
        self.decgraphs: List[DecGraph] = []
        self.v2pos = {v:p for v, p in enumerate(self.D.nodes)}
        self.pos2v = {p:v for v, p in enumerate(self.D.nodes)}
        
        for i in self.I:
            Di = self.D.subgraph(set(sol.Pi[i])).copy()
            G_prime, dV = DecGraph.generate_G_prime(Di)
            self.decgraphs.append(DecGraph(Di, G_prime, dV))
            for v in sol.Pi[i]:
                self.dup_rec.dup(v, i)
            self.R_D.append([r for r in dg.dV[R[i]] if r in self.decgraphs[i].V][0])

        self.sol_0 = sol
        self.pools: List[Pool] = [
            GrowPool(self.D, self.R, self.decgraphs, self.dup_rec, prio_type),
            DeDupPool(self.D, self.R, self.decgraphs, self.dup_rec, self.R_D, prio_type),
            ExcPool(self.D, self.R, self.decgraphs, self.dup_rec, self.R_D, prio_type),
        ]
        self.num_of_pools = len(self.pools)
        self.rhos = np.array([1.0 for _ in range(self.num_of_pools)])

    def run(
        self,
        M: int,
        S:int,
        alpha: float,
        gamma: float,
        sample_type: SampleType,
        T0: float = 1,
        record:dict|None = None,
        seed:int = 0,
    ) -> Tuple[Solution, float]:
        np.random.seed(seed)
        
        sol, sol_opt = self.sol_0.copy(), self.sol_0.copy()
        T, iter, ts = T0, 0, time.time()
        self._update_pools(sol)

        while iter < M:
            if self.vbs:
                print(f"\nLS-MCPP iter {iter} (t={T:.3f}): {self._sol_summary(sol)}")
            else:
                print(f"LS-MCPP iter {iter} (t={T:.3f}): makespan={sol.tau:.2f}")

            op, new_tau, new_sum_costs = self._sample(sol, sample_type, gamma)
            if op is None:
                break
            
            delta_tau = new_tau - sol.tau
            iter = iter + 1

            # Accept Criteria: 1) found a better solution; 2) worse solution with probablity
            prob = np.exp(-delta_tau / T) if delta_tau > 0 else 1
            if delta_tau < 0 or np.random.random() < prob:
                _log = op.apply(self.R, self.decgraphs, self.D, self.dup_rec, sol)                
                T = alpha * T
                
                Helper.verbose_print(
                    f" -> ACCEPT makespan {new_tau:.2f}({Helper.pn_str(delta_tau)}) w/ prob={prob:.2f}: {_log}",
                    self.vbs,
                )

                self._update_pools(sol, op)

                # force deduplicate
                if iter % S == 0 or new_tau < sol_opt.tau:
                    self.forced_deduplication(sol)
                    self._update_pools(sol)

                # update the optimal solution
                if new_tau < sol_opt.tau:
                    sol_opt = sol.copy()
                    Helper.verbose_print(f" -> UPDATE Optimal Solution", self.vbs)

            if record is not None:
                record["uids"].append(op.uid)
                record["costs"].append(sol.costs.copy())

        runtime = time.time() - ts
        print(f"\nLS-MCPP TERMINATE at iter {iter} ({runtime:.3f} secs): {self._sol_summary(sol_opt)}")

        if record is not None:
            record["sol_opt"] = sol_opt

        return sol_opt, runtime

    def _sample(
        self, 
        sol: Solution,
        sample_type: SampleType,
        gamma:float = 1e-2,
    ) -> Tuple[Operator, float, float]:
        
        beq_avg = np.argwhere(sol.costs <= sol.avg_cost).T.tolist()[0]
        gt_avg = np.argwhere(sol.costs > sol.avg_cost).T.tolist()[0]
        masks = [beq_avg, gt_avg, beq_avg]
        nonempty = [i for i in range(self.num_of_pools) if not self.pools[i].is_empty(masks[i])]

        if nonempty == []:
            return None, None, None

        if sample_type == SampleType.RouletteWheel:
            prob = softmax(self.rhos[nonempty])
        elif sample_type == SampleType.Random:
            prob = (1/len(nonempty)) * np.ones(len(nonempty))

        pool_idx = int(np.random.choice(nonempty, p=prob, size=1))
        pool = self.pools[pool_idx]
        op = pool.sample(masks[pool_idx])
        new_tau, new_sum_costs = op.eval(self.R, self.decgraphs, self.D, sol)

        if sample_type == SampleType.RouletteWheel:
            # update roulette wheel weights
            psy = sol.tau - new_tau
            self.rhos[pool_idx] = (1 - gamma) * self.rhos[pool_idx] + gamma * max(psy, 0)

        Helper.verbose_print(
            f" -> SAMPLE from {pool.name} out of {pool.selected_size-1} ops (selected from {pool.total_size})",
            self.vbs,
        )

        return op, new_tau, new_sum_costs

    def _sol_summary(self, sol: Solution):
        return ", ".join(
            [
                f"makespan={sol.tau:.2f} ({Helper.pn_str(sol.tau - self.sol_0.tau)})",
                f"pool-sizes={[self.pools[i].total_size for i in range(self.num_of_pools) if self.pools[i] is not None]}",
                f"sum-costs={sol.sum_costs:.2f} ({Helper.pn_str(sol.sum_costs - self.sol_0.sum_costs)})",
                sol.cost_str,
            ]
        )

    def _update_pools(self, sol:Solution, op:Operator=None) -> None:
        if op is None:
            self.pools[0].init(sol)
            self.pools[1].init(sol)
            self.pools[2].init(sol, self.pools[0])
        else:
            added = self.pools[0].update(op, sol, self.T)
            self.pools[1].update(op, sol)
            self.pools[2].update(op, sol, added)

            for i in range(self.k):

                for op_idx in range(self.pools[0].sizes[i]):
                    u, v = self.pools[0].pool[i][op_idx].V
                    self.pools[0].heurs[i][op_idx] = sol.costs[i] * self.k + (self.dup_rec.cnts[u] + self.dup_rec.cnts[v]) / 2

                for op_idx in range(self.pools[1].sizes[i]):
                    u, v = self.pools[1].pool[i][op_idx].V
                    self.pools[1].heurs[i][op_idx] = - sol.costs[i] * self.k - (self.dup_rec.cnts[u] + self.dup_rec.cnts[v]) / 2
                    
                for op_idx in range(self.pools[2].sizes[i]):
                    add_idx = self.pools[2].pool[i][op_idx].idx
                    rmv_idx = self.pools[2].pool[i][op_idx].op_rmv_idx
                    self.pools[2].heurs[i][op_idx] = sol.costs[add_idx] - sol.costs[rmv_idx]

    def forced_deduplication(self, sol:Solution) -> None:
        cnts_dedup_op, cnts_U_turn = 0, 0

        for idx in np.argsort(sol.costs)[::-1]:
            pi = sol.Pi[idx]
            len_pi = len(pi)
            u_idx, fwd, d_fwd = 0, 1, 1
            while len_pi > 2:
                v_idx, p_idx, q_idx = (
                    Helper.shift(u_idx, 1, len_pi),
                    Helper.shift(u_idx, -1, len_pi),
                    Helper.shift(u_idx, 2, len_pi),
                )
                u, v, p, q = pi[u_idx], pi[v_idx], pi[p_idx], pi[q_idx]
                
                if u in self.dup_rec.dup_set and u != self.R_D[idx] and u in self.decgraphs[idx].V and \
                   v in self.dup_rec.dup_set and v != self.R_D[idx] and v in self.decgraphs[idx].V and \
                   self.is_U_turn(p, u, v, q):
                    pi = self.remove_seg(pi, u_idx, v_idx)
                    self.decgraphs[idx].del_pairing_verts(self.D, [u, v])
                    self.dup_rec.dedup(u, idx)
                    self.dup_rec.dedup(v, idx)
                    cnts_U_turn += 1
                    len_pi -= 2
                    u_idx = Helper.shift(p_idx, -1, len_pi)
                    d_fwd = fwd - 1
                    fwd = -1
                else:
                    d_fwd = fwd + 1
                    fwd = 1
                    if u_idx + 1 == len_pi and d_fwd != 0:
                        break
                    else:
                        u_idx = Helper.shift(u_idx, 1, len_pi)

            sol.Pi[idx] = pi
            sol.costs[idx] = DecGraph.path_cost(self.D, pi)

        self.pools[1].init(sol)
        for pi_idx in np.argsort(sol.costs)[::-1]:
            updated = set()
            while self.pools[1].sizes[pi_idx] != 0: 
                op_idx = np.argmin(self.pools[1].heurs[pi_idx])
                op = self.pools[1].pool[pi_idx][op_idx]
                if op.uid in updated:
                    break
                else:
                    updated.add(op.uid)
                    op.apply(self.R, self.decgraphs, self.D, self.dup_rec, sol)
                    self.pools[1].update(op, sol)
                    for op_idx in range(self.pools[1].sizes[pi_idx]):
                        u, v = self.pools[1].pool[pi_idx][op_idx].V
                        self.pools[1].heurs[pi_idx][op_idx] = - sol.costs[pi_idx] * self.k - (self.dup_rec.cnts[u] + self.dup_rec.cnts[v]) / 2
                    cnts_dedup_op += 1
  
    @staticmethod
    def is_U_turn(p, u, v, q) -> bool:
        # p -> . -> . -> q
        return abs(p[0] - q[0]) + abs(p[1] - q[1]) == 0.5 and DecGraph.undecomp(u) == DecGraph.undecomp(v)

    @staticmethod
    def remove_seg(pi: List[Tuple], rm_st_idx:int, rm_ed_idx:int) -> List[Tuple]:
        if rm_st_idx <= rm_ed_idx:
            return pi[:rm_st_idx] + pi[rm_ed_idx+1:]
        else:
            return pi[rm_ed_idx+1:rm_st_idx]
