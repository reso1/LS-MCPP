from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
from scipy.special import softmax

from lsmcpp.graph import DecGraph, contract
from lsmcpp.operator import Operator
from lsmcpp.pool import *
from lsmcpp.utils import DuplicationRec, Helper
from lsmcpp.estc import ExtSTCPlanner
from lsmcpp.benchmark.solution import Solution

from lsmcpp.benchmark.instance import MCPP


class LocalSearchMCPP:

    def __init__(
        self, 
        mcpp:MCPP,
        sol:Solution,
        prio_type: PrioType,
        pool_type: PoolType,
        verbose=True,
    ) -> None:
        
        self.vbs = verbose
        self.dg = contract(mcpp._G_legacy)
        self.T = Helper.to_position_graph(self.dg.T)
        self.D = mcpp._G_legacy
        self.mcpp = mcpp
        self.prio_type = prio_type
        self.R = [self.dg.undecomp(mcpp.legacy_vertex(r)) for r in mcpp.R]
        self.R_D = [mcpp.legacy_vertex(r) for r in mcpp.R]

        self.k = len(self.R)
        self.I = list(range(self.k))
        self.dup_rec = DuplicationRec()
        self.decgraphs: List[DecGraph] = []
        self.v2pos = {v:p for v, p in enumerate(self.D.nodes)}
        self.pos2v = {p:v for v, p in enumerate(self.D.nodes)}
        
        for i in self.I:
            Di = self.D.subgraph(set(sol.Pi[i])).copy()
            dgi = contract(Di)
            dgi._update_boundary_verts(self.dg, self.dg.V)
            self.decgraphs.append(dgi)
            for v in sol.Pi[i]:
                self.dup_rec.dup(v, i)

        self.sol_0 = sol
        if pool_type == PoolType.Edgewise:
            self.pools: List[Pool] = [
                EdgewiseGrowPool(self.D, self.R, self.dg, self.decgraphs, self.dup_rec, prio_type),
                EdgewiseDeDupPool(self.D, self.R, self.dg, self.decgraphs, self.dup_rec, self.R_D, prio_type),
                EdgewiseExcPool(self.D, self.R, self.dg, self.decgraphs, self.dup_rec, self.R_D, prio_type),
            ]
        else:
            self.pools: List[Pool] = [
                GrowPool(self.D, self.R, self.dg, self.decgraphs, self.dup_rec, prio_type),
                DeDupPool(self.D, self.R, self.dg, self.decgraphs, self.dup_rec, self.R_D, prio_type),
                ExcPool(self.D, self.R, self.dg, self.decgraphs, self.dup_rec, self.R_D, prio_type),
            ]
        self.num_of_pools = len(self.pools)
        if prio_type == PrioType.SeparateHeur:
            # [rho_grow_1, rho_grow_2, rho_dedup_1, rho_dedup_2, rho_exc_1, rho_exc_2, ...]
            self.rhos = np.array([1.0 for _ in range(self.num_of_pools * self.k)])
        if prio_type == PrioType.CompositeHeur:
            # [rho_grow, rho_dedup, rho_exc]
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
        rng = np.random.RandomState(seed)
        
        sol, sol_opt = self.sol_0.copy(), self.sol_0.copy()
        T, iter, ts = T0, 0, time.perf_counter()
        self._update_pools(sol)
        
        while iter < M:
            if self.prio_type == PrioType.SeparateHeur:
                op, new_tau, new_sum_costs = self._sample_separate(sol, sample_type, rng, gamma)
            elif self.prio_type == PrioType.CompositeHeur:
                op, new_tau, new_sum_costs = self._sample_composite(sol, sample_type, rng, gamma)
            
            if op is None:
                break
            
            delta_tau = new_tau - sol.tau
            iter = iter + 1

            # Accept Criteria: 1) found a better solution; 2) worse solution with probablity
            prob = np.exp(-delta_tau / T) if delta_tau > 0 else 1
            if delta_tau < 0 or rng.random() < prob:
                _log = op.apply(self.R, self.dg, self.decgraphs, self.D, self.dup_rec, sol)                
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
                    new_tau = sol.tau
                    Helper.verbose_print(f" -> Forced Deduplication", self.vbs)

                # update the optimal solution
                if new_tau < sol_opt.tau:
                    sol_opt = sol.copy()
                    self.forced_deduplication(sol_opt)
                    Helper.verbose_print(f" -> UPDATE Optimal Solution: makespan={sol_opt.tau}", self.vbs)
            
            Helper.verbose_print(f"\nLS-MCPP iter {iter} (t={T:.3f}): {self._sol_summary(sol)}", self.vbs)
            
            if not self.vbs and iter % 1e3 == 0:
                print(f"\nLS-MCPP iter {iter} (t={T:.3f}): {self._sol_summary(sol)}")

            if record is not None:
                record["uids"].append(op.uid)
                record["costs"].append(sol.costs.copy())
            
        runtime = time.perf_counter() - ts
        print(f"\nLS-MCPP TERMINATE at iter {iter} ({runtime:.3f} secs): {self._sol_summary(sol_opt)}")

        if record is not None:
            record["sol_opt"] = sol_opt

        Pi = [ExtSTCPlanner.root_align(sol_opt.Pi[i], self.R_D[i]) for i in self.I]
        sol_ret = Solution(Pi, np.array([Solution.path_cost(pi, self.D) for pi in Pi]))
        return sol_ret, runtime

    def _sample_separate(
        self, 
        sol: Solution,
        sample_type: SampleType,
        rng: np.random.RandomState,
        gamma:float = 1e-2,
    ) -> Tuple[Operator, float, float]:
        
        # get valid non-empty pools
        beq_avg = np.argwhere(sol.costs <= sol.avg_cost).T.tolist()[0]
        gt_avg = np.argwhere(sol.costs > sol.avg_cost).T.tolist()[0]
        masks = [beq_avg, gt_avg, beq_avg]
        nonempty = [i for i in range(self.num_of_pools) if not self.pools[i].is_empty(masks[i])]
        if nonempty == []:
            return None, None, None
    
        # calc sampling probability and sample which pool to sample
        if sample_type == SampleType.RouletteWheel:
            prob = softmax(self.rhos[nonempty])
        elif sample_type == SampleType.Random:
            prob = (1/len(nonempty)) * np.ones(len(nonempty))

        sampled = int(rng.choice(nonempty, p=prob, size=1))
        pool_idx, pi_idx = divmod(sampled, self.k)
        pool = self.pools[pool_idx]

        # sample operator from the selected pool
        while True:
            op = pool.sample(rng, masks[pool_idx])
            if op:
                res = op.eval(self.R, self.dg, self.decgraphs, self.D, sol)
                if res:
                    new_tau, new_sum_costs = res
                    break
            else:
                return self._sample_separate(sol, sample_type, rng, gamma)

        if sample_type == SampleType.RouletteWheel:
            # update roulette wheel weights
            psy = sol.tau - new_tau
            self.rhos[pool_idx*self.k+pi_idx] = (1 - gamma) * self.rhos[pool_idx*self.k+pi_idx] + gamma * max(psy, 0)

        Helper.verbose_print(f" -> SAMPLE from {pool.name}[path {pi_idx}] out of {pool.sizes[pi_idx]} ops: [{op.V}]", self.vbs)

        return op, new_tau, new_sum_costs

    def _sample_composite(
        self, 
        sol: Solution,
        sample_type: SampleType,
        rng: np.random.RandomState,
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

        pool_idx = int(rng.choice(nonempty, p=prob, size=1))
        pool = self.pools[pool_idx]
        
        while True:
            op = pool.sample(rng, masks[pool_idx])
            if op:
                res = op.eval(self.R, self.dg, self.decgraphs, self.D, sol)
                if res:
                    new_tau, new_sum_costs = res
                    break
            else:
                return self._sample_composite(sol, sample_type, rng, gamma)

        if sample_type == SampleType.RouletteWheel:
            # update roulette wheel weights
            psy = sol.tau - new_tau
            self.rhos[pool_idx] = (1 - gamma) * self.rhos[pool_idx] + gamma * max(psy, 0)

        Helper.verbose_print(f" -> SAMPLE from {pool.name}[path {op.idx}] out of {pool.sizes[op.idx]} ops", self.vbs)

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
            for pool in self.pools:
                if type(pool) is ExcPool or type(pool) is EdgewiseExcPool:
                    pool.init(sol, self.pools[0])
                if type(pool) is GrowPool or type(pool) is EdgewiseGrowPool:
                    pool.init(sol)
                if type(pool) is DeDupPool or type(pool) is EdgewiseDeDupPool:
                    pool.init(sol)
        else:
            for pool in self.pools:
                if type(pool) is GrowPool or type(pool) is EdgewiseGrowPool:
                    added = pool.update(op, sol, self.T)
                if type(pool) is DeDupPool or type(pool) is EdgewiseDeDupPool:
                    pool.update(op, sol)
                if type(pool) is ExcPool or type(pool) is EdgewiseExcPool:
                    pool.update(op, sol, added)

            for i in range(self.k):
                for pool in self.pools:
                    if pool is not None:
                        for op_idx, op in enumerate(pool.pool[i]):
                            pool.heurs[i][op_idx] = pool.heur_val(op, sol)

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
                if self.remove_U_turn(p, u, v, q, idx):
                    pi = self.remove_seg(pi, u_idx, v_idx)
                    if p == v:
                        self.dup_rec.dedup(u, idx)
                        self.decgraphs[idx].del_pairing_verts(self.dg, [u])
                    else:
                        self.decgraphs[idx].del_pairing_verts(self.dg, [u, v])
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
            sol.costs[idx] = Solution.path_cost(pi, self.D)

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
                    op.apply(self.R, self.dg, self.decgraphs, self.D, self.dup_rec, sol)
                    self.pools[1].update(op, sol)
                    for op_idx in range(self.pools[1].sizes[pi_idx]):
                        self.pools[1].heurs[pi_idx][op_idx] = self.pools[1].heur_val(self.pools[1].pool[pi_idx][op_idx], sol)
                    cnts_dedup_op += 1
        
    def remove_U_turn(self, p, u, v, q, idx) -> bool:
        # collapsed U-turn: (p -> u -> v(=p) -> q) => (p -> q)
        if u in self.dup_rec.dup_set and u != self.R_D[idx] and u in self.decgraphs[idx].V and p == v:
            return True

        # normal U-turn: (p -> u -> v -> q) => (p -> q)
        if u in self.dup_rec.dup_set and u != self.R_D[idx] and u in self.decgraphs[idx].V and \
           v in self.dup_rec.dup_set and v != self.R_D[idx] and v in self.decgraphs[idx].V and \
           abs(p[0] - q[0]) + abs(p[1] - q[1]) == 0.5 and self.dg.undecomp(u) == self.dg.undecomp(v) \
           and p != v and q != u:
            return True
        
        return False
        
    @staticmethod
    def remove_seg(pi: List[Tuple], rm_st_idx:int, rm_ed_idx:int) -> List[Tuple]:
        if rm_st_idx <= rm_ed_idx:
            return pi[:rm_st_idx] + pi[rm_ed_idx+1:]
        else:
            return pi[rm_ed_idx+1:rm_st_idx]
