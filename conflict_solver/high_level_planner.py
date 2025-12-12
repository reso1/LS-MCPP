from __future__ import annotations
from typing import List, Dict, Set, Tuple, Generator
import psutil

import numpy as np
import networkx as nx

from lsmcpp.utils import Helper
from lsmcpp.benchmark.solution import Solution
from lsmcpp.conflict_solver.low_level_planner import *


class Node:

    def __init__(self, diG:nx.DiGraph, plans:List[Plan], init_rt_combined=False) -> None:
        self.diG, self.plans = diG, plans
        self.rt_combined = ReservationTable()
        self.reserved = set()
        if init_rt_combined:
            for i, P in enumerate(plans):
                self.rt_combined.reserve_plan(P, i)
                self.reserved.add(i)

    def __str__(self) -> str:
        return super().__str__()

    def __hash__(self) -> int:
        return hash(tuple(self.diG.edges))

    def __str__(self) -> str:
        return f"({', '.join([f'{u}<{v}' for u, v in self.diG.edges()])})"

    def __lt__(self, other:Node) -> bool:
        return self.heur < other.heur

    @property
    def num_of_conflicts(self) -> int:
        ret = set()
        for i in range(len(self.plans)):
            ret.update(self.get_plan_all_conflicts(i))
        return len(ret)
    
    def cost(self, i:int) -> float:
        return self.plans[i][-1].time
    
    @property
    def makespan(self) -> float:
        return max([p[-1].time for p in self.plans])

    def get_valid_children(
        self, planner:LowlevelPlanner, CLOSED:set, sol:Solution, 
        ci:int, cj:int, num_max_nodes:int, time_limit:float, verbose:bool
    ) -> Tuple[Status, List[Node], List[Tuple[Node, Node, int]]]:
        status, ret, postponed = Status.SUCCESS, [], []
        for i, j in [(ci, cj), (cj, ci)]:
            G = self.diG.copy()
            G.add_edge(i, j)
            if nx.is_directed_acyclic_graph(G):
                child = Node(G, [None for _ in self.plans])
                if child.__hash__() in CLOSED:
                    continue

                Helper.verbose_print(f"\t+ Child node {str(child)}:", verbose)
                status = Node.update_plans(planner, sol, self, child, j, num_max_nodes, time_limit, verbose)
                if status == Status.SUCCESS:
                    ret.append(child)
                elif status == Status.TIMEOUT:
                    return status, ret, postponed
                elif status == Status.POSTPONED:
                    postponed.append((self, Node(G, [None for _ in self.plans]), j)) # (parent, child, pi_idx)

        return status, ret, postponed

    @staticmethod
    def update_plans(planner:LowlevelPlanner, sol:Solution, parent:Node, child:Node, pi_idx:int, num_max_nodes:int, time_limit:float, verbose=False) -> Status:
        # replan for pi_idx and all other paths with lower priority than pi_idx
        replan_list = set([pi_idx] + [j for j in range(len(parent.plans)) if nx.has_path(child.diG, pi_idx, j)])
        for i in range(len(parent.plans)):
            if i not in replan_list:
                child.plans[i] = parent.plans[i].copy()
                child.rt_combined.reserve_plan(child.plans[i], i)

        for j in nx.topological_sort(child.diG.subgraph(replan_list)):
            higher_prio_inds = list(child.diG.predecessors(j))
            rt_higher_prio = ReservationTable()
            for k in higher_prio_inds:
                rt_higher_prio.reserve_plan(child.plans[k], k)

            if j == pi_idx or rt_higher_prio.is_conflicted_with(parent.plans[j], j, set(higher_prio_inds)):
                if len(sol.Pi[j]) == 0:
                    r = planner.mcpp.legacy_vertex(planner.mcpp.R[j])
                    status, P, num_exlored = planner.plan([r, r], rt_higher_prio, num_max_nodes, time_limit)
                elif len(sol.Pi[j]) == 1:
                    status, P, num_exlored = planner.plan([sol.Pi[j][0], sol.Pi[j][0]], rt_higher_prio, num_max_nodes, time_limit)
                else:
                    status, P, num_exlored = planner.plan(sol.Pi[j], rt_higher_prio, num_max_nodes, time_limit)

                if status == Status.FAILURE:
                    Helper.verbose_print(f"\t\t\u2717 Replan for robot {j}: Failed w/ # of search nodes={num_exlored} (max per goal={num_max_nodes})", verbose)
                    return Status.FAILURE
                elif status == Status.TIMEOUT:
                    Helper.verbose_print(f"\t\t\u23F3 Replan for robot {j}: Timeout w/ # of search nodes={num_exlored} (max per goal={num_max_nodes})", verbose)
                    return Status.TIMEOUT
                elif status == Status.POSTPONED:
                    Helper.verbose_print(f"\t\t\u23F3 Replan for robot {j}: Postponed w/ # of search nodes={num_exlored} (max per goal={num_max_nodes})", verbose)
                    return Status.POSTPONED
                
                Helper.verbose_print(f"\t\t\u2713 Replan for robot {j}: Success w/ # of search nodes={num_exlored} (max per goal={num_max_nodes})", verbose)
                child.plans[j] = P
            else:
                child.plans[j] = parent.plans[j].copy()

            child.rt_combined.reserve_plan(child.plans[j], j)
            # assert not child.rt_combined.is_conflicted_with(child.plans[j], j, set(higher_prio_inds))

        Helper.verbose_print(f"\t\t\u2713 Updated plan successfully", verbose)
        return Status.SUCCESS

    def get_plan_first_conflict(self, i:int) -> Tuple[float|None, Set[int]|None]:
        for v, itvl in ReservationTable.occupying_itvls(self.plans[i], i):
            conflicted = self.rt_combined.get_all_conflicted_inds(v, itvl, i)
            if conflicted:
                return itvl.start, conflicted

        return None, None
    
    def get_plan_all_conflicts(self, i:int) -> Set[Tuple[int, int, Interval]]:
        sorted_pair = lambda x, y: (x, y) if x < y else (y, x)
        ret = set()
        for v, itvl in ReservationTable.occupying_itvls(self.plans[i], i):
            for j in self.rt_combined.get_all_conflicted_inds(v, itvl, i):
                ret.add(sorted_pair(i, j) + (itvl, ))

        return ret
    
    def get_first_conflict(self) -> Tuple[float|None, int|None, int|None]:
        earliest_pair = (float('inf'), None, None)
        for i in range(len(self.plans)):
            time, conflicted = self.get_plan_first_conflict(i)
            if time is not None and time < earliest_pair[0]:
                earliest_pair = (time, i, list(conflicted)[0])

        return earliest_pair


class PBS:

    def __init__(self, mcpp, low_level_planner:LowlevelPlanner, 
                 runtime_limit=float('inf'), num_max_nodes_explored_per_goal_limit=float('inf')) -> None:
        self.mcpp = mcpp
        self.planner = low_level_planner
        self.holistic_planner = HolisticApproach(self.mcpp, HeurType.TrueDist)  # for adaptive low-level planner only
        self.runtime_limit = float(runtime_limit)
        self.num_max_nodes_explored_per_goal_limit = num_max_nodes_explored_per_goal_limit

    def run(self, sol:Solution, verbose=False) -> List[Plan]|None:
        G = nx.DiGraph()
        G.add_nodes_from(self.mcpp.I)
        init_plans = []
        init_planner = ChainingApproach(self.mcpp, HeurType.TrueDist)
        for i in self.mcpp.I:
            sol.Pi[i] = [v for v in sol.Pi[i] if self.mcpp.pos2v(v) == self.mcpp.R[i] or self.mcpp.pos2v(v) not in self.mcpp.R]
            if sol.Pi[i] == []:
                sol.Pi[i] = [self.mcpp.legacy_vertex(self.mcpp.R[i])]
            status, P, num_explored = init_planner.plan(sol.Pi[i], ReservationTable(), float('inf'))
            if len(P) == 0:
                P.states = [State(self.mcpp.R[i], 0), State(self.mcpp.R[i], float('inf'))]
            init_plans.append(P)
            Helper.verbose_print(f"PBS: Initialized Plan for robot {i}, # of explored nodes={num_explored}", verbose)
        
        if self.num_max_nodes_explored_per_goal_limit == "default":
            self.num_max_nodes_explored_per_goal_limit = int(np.sqrt(self.planner.mcpp.k) * max([len(pi) for pi in sol.Pi]))
        else:
            self.num_max_nodes_explored_per_goal_limit = float(self.num_max_nodes_explored_per_goal_limit)
        
        print(f"PBS: Starting PBS with # of robots={self.mcpp.k}, runtime limit={self.runtime_limit}, # of max nodes explored per goal={self.num_max_nodes_explored_per_goal_limit}")

        root = Node(G, init_plans, True)
        OPEN, CLOSED, POSTPONED = [root], set(), []
        start_time = time.perf_counter()
        process = psutil.Process()
        status = Status.SUCCESS

        while True:
            if len(OPEN) != 0:
                node = OPEN.pop()
            elif len(POSTPONED) != 0:
                _parent, _child, _pi_idx = POSTPONED.pop()
                Helper.verbose_print(f"PBS: Update plans for postponed node {str(_child)}", verbose)
                status = Node.update_plans(self.holistic_planner, sol, _parent, _child, _pi_idx, self.num_max_nodes_explored_per_goal_limit, self.runtime_limit, verbose)
                if status == Status.SUCCESS:
                    node = _child
                else:
                    continue
            else:
                break

            self.time_usage = time.perf_counter() - start_time
            Helper.verbose_print(f"PBS: Current node = {str(node)}, # of conflicts={node.num_of_conflicts}, time usage={self.time_usage:.0f}secs, RAM usage={process.memory_info().rss/1024/1024:.0f}MB", verbose)
            
            c_time, ci, cj = node.get_first_conflict()
            if ci is None:  # return if there is no conflict
                print(f"PBS: Successfully found a valid set of plans, # of evaluated node={len(CLOSED)}, makespan={max([p[-1].time for p in node.plans]):.3f}")
                return [Plan([State(self.mcpp.legacy_vertex(X.pos), X.time, X.heading) for X in P.states]) for P in node.plans]
            
            if status == Status.TIMEOUT or self.time_usage > self.runtime_limit:
                print(f"PBS: Time limit of {self.runtime_limit} secs exceeded.")
                return None

            Helper.verbose_print(f"Found conflict between {ci} and {cj} @ time {c_time}", verbose)
            # assert (ci, cj) not in node.diG.edges and (cj, ci) not in node.diG.edges

            status, succeeded, postponed = node.get_valid_children(
                self.planner, CLOSED, sol, ci, cj, self.num_max_nodes_explored_per_goal_limit, self.runtime_limit-self.time_usage, verbose)

            OPEN.extend([child for child in sorted(succeeded, key=lambda x: x.makespan, reverse=True)]) # push the one child with lower first
            CLOSED.update([child.__hash__() for child in succeeded])
            POSTPONED.extend(postponed)

        print(f"PBS: Failed to find a path for the robot, # of evaluated node={len(CLOSED)}.")
        return None
