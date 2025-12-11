from __future__ import annotations
from abc import abstractmethod
from typing import Callable, Tuple, Dict, List, Iterator
from collections import defaultdict
from enum import IntEnum
import heapq, time

from lsmcpp.benchmark.instance import MCPP
from lsmcpp.benchmark.plan import State, Plan, Heading

from lsmcpp.conflict_solver.reservation_table import ReservationTable, Interval
from lsmcpp.conflict_solver.states import SearchState, LabeledState


class Status(IntEnum):
    SUCCESS = 0
    TIMEOUT = 1
    FAILURE = 2
    POSTPONED = 3 # for adaptive approach


class HeurType(IntEnum):
    Manhatten = 0
    TrueDist = 1
    
    def __str__(self) -> str:
            return super().__str__()


class SIPP:

    def __init__(self, mcpp:MCPP) -> None:
        self.mcpp = mcpp
        self.G = mcpp.G
        self.W = lambda p, q: self.G[p][q]["weight"]

    def plan(
        self, X_start:SearchState, goal:int, rt:ReservationTable, 
        h:Callable[[SearchState, int], float], 
        max_num_nodes_explored:int=2e3, time_limit:float=float('inf'),
        stay=False
    ) -> Tuple[Status, Plan|None, int]:
        parent: Dict[SearchState, Tuple[SearchState, float]] = {}
        g = defaultdict(lambda: float('inf'))
        g[X_start] = 0
        OPEN, CLOSED = [(h(X_start, goal), X_start)], set()
        candidate = (float('inf'), None)
        start_time = time.perf_counter()

        # doomed to fail if the goal is not reachable
        goal_col_itvl = rt.get(goal).intervals
        if goal_col_itvl and goal_col_itvl[-1].end == float('inf') and goal_col_itvl[-1].start <= X_start.time:
            print(f"\t\t\u2717 Goal {goal} not reacheable, occupied as root starting at {goal_col_itvl[-1].start:.3f}(<= {X_start.time:.3f})")
            return Status.FAILURE, None, len(CLOSED)
        
        while OPEN:
            f_Xc, Xc = heapq.heappop(OPEN)
            CLOSED.add(Xc)
            if time.perf_counter() - start_time > time_limit:
                print(f"\t\t\u2717ML-SIPP: Time limit of {time_limit} secs exceeded.")
                return Status.TIMEOUT, None, len(CLOSED)

            if self.goal_test(Xc, goal, stay):
                return Status.SUCCESS, Plan(self.reconstruct_path(X_start, Xc, parent)), len(CLOSED)

            if len(CLOSED) > max_num_nodes_explored:
                if candidate[1] is not None:
                    print(f"\t\t* Max # of nodes (={max_num_nodes_explored:.0f}) explored reached, but found a valid path in OPEN")
                    return Status.SUCCESS, Plan(self.reconstruct_path(X_start, candidate[1], parent)), len(CLOSED)
                print(f"\t\t\u2717 Max # of nodes (={max_num_nodes_explored:.0f}) explored reached: {X_start.pos}->{goal}")
                return Status.FAILURE, None, len(CLOSED)

            for Xn, wait_time in self.get_successors(Xc, rt):
                tentative_g = round(g[Xc] + Xn.time - Xc.time, 9)
                # tie-breaking on same h-value: the action with more wait time is preferred
                if tentative_g < g[Xn] or (tentative_g == g[Xn] and parent[Xn][1] < wait_time):
                    parent[Xn] = (Xc, wait_time)
                    g[Xn] = tentative_g
                    f_Xn = g[Xn] + h(Xn, goal)
                    if (f_Xn, Xn) not in OPEN and Xn not in CLOSED:
                        heapq.heappush(OPEN, (f_Xn, Xn))
                        # record valid candidate solutions
                        if Xn.pos == goal and (not stay or Xn.safe_itvl.end == float('inf')) and f_Xn < candidate[0]:
                            candidate = (f_Xn, Xn)

        print(f"\t\t\u2717 Failed with an empty OPEN list: {X_start.pos}->{goal}")  
        return Status.FAILURE, None, len(CLOSED)

    def get_successors(self, Xc:SearchState, rt:ReservationTable) -> Iterator[Tuple[SearchState, float]]:
        for ngb in self.G.neighbors(Xc.pos):
            for si in rt.get_safe_intervals(ngb):
                # turn -> wait -> move
                Xn_heading = Xc.get_heading(ngb, self.mcpp)
                t_rot = Heading.rot_cost(Xc.heading, Xn_heading)
                t_start = max(Xc.time + t_rot, si.start)
                t_arrival = t_start + self.W(Xc.pos, ngb)
                t_wait = t_start - Xc.time - t_rot
                Xn = SearchState(State(ngb, t_arrival, Xn_heading), si)
                if si.contains(Interval(t_start, t_arrival)) and \
                   not rt.get(Xc.pos).intersects(Interval(Xc.time, t_arrival)):
                    yield (Xn, t_wait)
    
    def goal_test(self, X:SearchState, goal:int, stay:bool) -> bool:
        return (X.pos == goal) and (not stay or X.safe_itvl.end == float('inf'))

    def reconstruct_path(self, X_start:SearchState, Xc:SearchState, parent:dict) -> List[SearchState]:
        states = []
        while Xc in parent:
            states.append(Xc)
            Xc = parent[Xc][0]

        states = [X_start] + states[::-1]

        # turn -> wait -> move to transit X to Xn
        ret = [states[0]]
        for i in range(1, len(states)):
            X, Xn = states[i-1], states[i]
            if round(Xn.time - X.time, 9) != 0:
                t_rot, t_wait = Heading.rot_cost(X.heading, Xn.heading), parent[Xn][1]
                if round(t_rot, 9) > 0:
                    ret.append(State(X.pos, X.time + t_rot, Xn.heading))
                if round(t_wait, 9) > 0:
                    ret.append(State(X.pos, X.time + t_rot + t_wait, Xn.heading))
                ret.append(Xn)

        return ret
   

class MultiLabelSIPP(SIPP):
    
    def __init__(self, mcpp: MCPP) -> None:
        super().__init__(mcpp)
    
    def plan(
            self, X_start: SearchState, goal_seq: List[int], 
            rt: ReservationTable, h: Callable[[SearchState, int], float], 
            max_num_nodes_explored: int = 2e3, time_limit: float = float('inf'),
            stay=False
        ) -> Tuple[Status, Plan | None, int]:
            """
            ### params:
            - start: int, the starting position
            - goal_seq: List[int], a sequence of goal positions
            - rt: ReservationTable, the reservation table for collision checking
            - h: Calable[[int, int], float], the heuristic function for estimating the cost from a position to a goal
            - max_num_nodes_explored: int, the maximum number of nodes to explore before terminating the search (default: 1000)
            ### returns:
            - Plan | int: the optimal plan from the starting position to the goal sequence, 
                        or the state of last goal successfully reached  
            """

            def _h_multilabel(X: LabeledState) -> float:
                h_comp = h(X, goal_seq[X.label])
                for i in range(X.label, len(goal_seq)-1):
                    h_comp += h(State(goal_seq[i], 0, X.heading), goal_seq[i+1])
                return h_comp
            
            def _on_intermediate_goal_reached(_X:LabeledState) -> int:
                Xprime = LabeledState(_X, _X.safe_itvl, _X.label+1)
                # only update the parent if the new local path is shorter than existing
                if Xprime not in parent or parent[Xprime][0].time > _X.time:
                    parent[Xprime] = (_X, 0)
                    g[Xprime] = g[Xc]
                    f_Xprime = g[Xprime] + _h_multilabel(Xprime)
                    heapq.heappush(OPEN, (f_Xprime, Xprime))
                return Xprime.label

            parent: Dict[LabeledState, Tuple[LabeledState, float]] = {}
            g = defaultdict(lambda: float('inf'))

            furthest_goal_label, final_goal_label = 0, len(goal_seq) - 1
            X_start = LabeledState(X_start, X_start.safe_itvl, label=0)
            g[X_start] = X_start.time # no necessarily 0 since the start state may not be at time 0 in the adpative approach
            
            start_time = time.perf_counter()
            candidates = defaultdict(lambda:(float('inf'), None))
            OPEN, CLOSED, num_explored_per_goal = [(h(X_start, goal_seq[0]), X_start)], set(), 0
            while OPEN:
                h_Xc, Xc = heapq.heappop(OPEN)
                CLOSED.add(Xc)
                num_explored_per_goal += 1

                if time.perf_counter() - start_time > time_limit:
                    print(f"\t\t\u2717ML-SIPP: Time limit of {time_limit} secs exceeded.")
                    return Status.TIMEOUT, None, len(CLOSED)

                # somehow dominated nodes get added into OPEN, so we need to skip them
                if Xc.time > g[Xc]:
                    continue

                if Xc.pos == goal_seq[Xc.label]:
                    if Xc.label == final_goal_label:
                        if (not stay or Xc.safe_itvl.end == float('inf')):
                            return Status.SUCCESS, Plan(self.reconstruct_path(X_start, Xc, parent)), len(CLOSED)
                    else:
                        cur_goal_label = _on_intermediate_goal_reached(Xc)
                        furthest_goal_label = max(furthest_goal_label, cur_goal_label)
                        num_explored_per_goal = 0
                        continue

                # if reached max number of nodes to explore, then return the best subpath from candidates
                if num_explored_per_goal > max_num_nodes_explored:
                    Xn:LabeledState = candidates[Xc.label][1]
                    if Xn is not None:
                        if Xn.label == final_goal_label:
                            print(f"\t\t* Max # of nodes (={max_num_nodes_explored:.0f}) explored per node reached, but found a valid path in OPEN")
                            return Status.SUCCESS, Plan(self.reconstruct_path(X_start, Xn, parent)), len(CLOSED)
                        else:
                            cur_goal_label = _on_intermediate_goal_reached(Xn)
                            furthest_goal_label = max(furthest_goal_label, cur_goal_label)
                            num_explored_per_goal = 0
                            continue

                    print(f"\t\t\u2717 Max # of nodes (={max_num_nodes_explored:.0f}) explored per node reached: {X_start.pos}")
                    return Status.FAILURE, None, len(CLOSED)

                # doomed to fail if the goal is not reachable
                goal_col_itvl = rt.get(goal_seq[Xc.label]).intervals
                if goal_col_itvl and goal_col_itvl[-1].end == float('inf') and goal_col_itvl[-1].start <= X_start.time:
                    print(f"\t\t\u2717 Goal {goal_seq[Xc.label]} not reacheable, occupied as root starting at {goal_col_itvl[-1].start:.3f}(<= {X_start.time:.3f})")
                    return Status.FAILURE, None, len(CLOSED)

                t_max_goal = rt.get_safe_intervals(goal_seq[Xc.label])[-1].end
                if Xc.label != final_goal_label and g[Xc] > t_max_goal:
                    continue

                for Xn, wait_time in self.get_successors(Xc, rt):
                    tentative_g = round(g[Xc] + Xn.time - Xc.time, 9)
                    # tie-breaking on same h-value: the action with more wait time is preferred
                    if tentative_g < g[Xn] or (tentative_g == g[Xn] and parent[Xn][1] < wait_time):
                        parent[Xn] = (Xc, wait_time)
                        g[Xn] = tentative_g
                        f_Xn = g[Xn] + _h_multilabel(Xn)
                        if (f_Xn, Xn) not in OPEN and Xn not in CLOSED:
                            heapq.heappush(OPEN, (f_Xn, Xn))
                            # record valid candidate solutions
                            if Xn.pos == goal_seq[Xn.label] and f_Xn < candidates[Xn.label][0]:
                                if Xn.label != final_goal_label:
                                    candidates[Xn.label] = (f_Xn, Xn)
                                elif not stay or Xn.safe_itvl.end == float('inf'):
                                    candidates[Xn.label] = (f_Xn, Xn)

            print(f"\t\t\u2717 Failed with an empty OPEN list: {X_start.pos}")  
            return Status.FAILURE, None, len(CLOSED)

    def get_successors(self, Xc:LabeledState, rt:ReservationTable) -> Iterator[Tuple[LabeledState, float]]:
        for Xn, t_wait in super().get_successors(Xc, rt):
            yield LabeledState(Xn, Xn.safe_itvl, Xc.label), t_wait


""" --------------------- Low-Level Planners --------------------- """

class LowlevelPlanner:

    def __init__(self, mcpp:MCPP, heuristic:HeurType) -> None:
        self.mcpp = mcpp
        self.G = mcpp.G
        self.W = lambda p, q: self.G[p][q]["weight"]
        self.h_manhatten = lambda X, goal: abs(self.G.nodes[X.pos]["pos"][0] - self.G.nodes[goal]["pos"][0]) + abs(self.G.nodes[X.pos]["pos"][1] - self.G.nodes[goal]["pos"][1])
        if heuristic == HeurType.Manhatten:
            self.h = self.h_manhatten
        elif heuristic == HeurType.TrueDist:
            # use true shortest distances from space A* ignoring all obstacles
            self.h = self.h_true_dist
            self.OPEN = defaultdict(list)
            self.CLOSED = defaultdict(set)
            self.g_values = defaultdict(lambda: defaultdict(lambda: float('inf')))
            self.reset_h_true_dist()

    @abstractmethod
    def plan(self, pi:List[int], rt:ReservationTable, max_num_nodes_explored:int, time_limit:float = float('inf')) -> Tuple[Status, Plan, int]:
        raise NotImplementedError
    
    def h_true_dist(self, X:State, goal:int) -> float:
        X = State(X.pos, 0, X.heading)
        if X not in self.CLOSED[goal]:
            # continue the backward search from goal until reaching X
            X_goal = State(goal, 0)
            self.OPEN[goal].append((0, X_goal))
            self.g_values[goal][X_goal] = 0
            self._continue_search(X, self.OPEN[goal], self.CLOSED[goal], self.g_values[goal])

        return self.g_values[goal][X]

    def _continue_search(self, X_goal:State, OPEN:List[Tuple[float, State]], CLOSED:set, g:Dict[State, float]) -> bool:
        # recompute the f-values of the OPEN list
        for i in range(len(OPEN)):
            new_f = g[OPEN[i][1]] + self.h_manhatten(OPEN[i][1], X_goal.pos)
            OPEN[i] = (new_f, OPEN[i][1])
        
        heapq.heapify(OPEN)
        while OPEN:
            f_Xc, Xc = heapq.heappop(OPEN)
            CLOSED.add(Xc)
            
            successors = []
            # move actions
            for ngb in self.G.neighbors(Xc.pos):
                Xn_heading = Xc.get_heading(ngb, self.mcpp)
                Xn = State(ngb, 0, Xn_heading)
                t_rot = Heading.rot_cost(Xc.heading, Xn_heading)
                tentative_g = g[Xc] + self.W(Xc.pos, ngb) + t_rot
                if Xn not in CLOSED:
                    successors.append((Xn, tentative_g))
            
            # turn actions
            for heading in [Heading.N, Heading.E, Heading.S, Heading.W]:
                if Xc.heading != heading:
                    Xn = State(Xc.pos, 0, heading)
                    tentative_g = g[Xc] + Heading.rot_cost(Xc.heading, heading)
                    if Xn not in CLOSED:
                        successors.append((Xn, tentative_g))

            for Xn, tentative_g in successors:
                if tentative_g < g[Xn]:
                    g[Xn] = tentative_g
                    f_Xn = g[Xn] + self.h_manhatten(Xn, X_goal.pos)
                    if (f_Xn, Xn) not in OPEN:
                        heapq.heappush(OPEN, (f_Xn, Xn))
            
            if Xc.pos == X_goal.pos and Xc.heading == X_goal.heading:
                return
            
    def reset_h_true_dist(self) -> None:
        del self.OPEN, self.CLOSED, self.g_values
        self.OPEN = defaultdict(list)
        self.CLOSED = defaultdict(set)
        self.g_values = defaultdict(lambda: defaultdict(lambda: float('inf')))


class ChainingApproach(LowlevelPlanner):

    def __init__(self, mcpp:MCPP, heuristic:HeurType) -> None:
        super().__init__(mcpp, heuristic)
        self.planner = SIPP(self.mcpp)

    def plan(self, pi:List[int], rt:ReservationTable, max_num_nodes_explored:int, time_limit:float = float('inf')) -> Tuple[Status, Plan, int]:
        r = self.mcpp.pos2v(pi[0])
        X_last = SearchState(State(r, 0), rt.get_safe_intervals(r)[0])
        states: List[SearchState] = [X_last]
        N, ts = 0, time.perf_counter()
        for i in range(1, len(pi)):
            self.reset_h_true_dist()
            status, P, n = self.planner.plan(
                X_start = X_last,
                goal = self.mcpp.pos2v(pi[i]),
                rt = rt,
                h = self.h,
                max_num_nodes_explored = max_num_nodes_explored,
                time_limit = time_limit - (time.perf_counter() - ts),
                stay = i == len(pi)-1
            )
            N += n
            if status == Status.SUCCESS:
                states.extend(P.states[1:])
                X_last = states[-1]
            else:
                return status, None, N

        return Status.SUCCESS, Plan(states), N


class HolisticApproach(LowlevelPlanner):
    
    def __init__(self, mcpp:MCPP, heuristic:HeurType) -> None:
        super().__init__(mcpp, heuristic)
        self.planner = MultiLabelSIPP(self.mcpp)

    def plan(self, pi:List[int], rt:ReservationTable, max_num_nodes_explored:int, time_limit:float = float('inf')) -> Tuple[Status, Plan, int]:
        self.reset_h_true_dist()
        r = self.mcpp.pos2v(pi[0])
        return self.planner.plan(
            X_start     = SearchState(State(r, 0), rt.get_safe_intervals(r)[0]),
            goal_seq    = [self.mcpp.pos2v(v) for v in pi[1:]],
            rt          = rt,
            h           = self.h,
            max_num_nodes_explored = max_num_nodes_explored,
            time_limit = time_limit,
            stay = True
        )


class AdaptiveApproach(LowlevelPlanner):
    
    def __init__(self, mcpp:MCPP, heuristic:HeurType, max_backtrack_step:int=5) -> None:
        super().__init__(mcpp, heuristic)
        self.planner = MultiLabelSIPP(self.mcpp)
        self.max_backtrack_step = max_backtrack_step

    def plan(self, pi:List[int], rt:ReservationTable, max_num_nodes_explored:int, time_limit:float = float('inf')) -> Tuple[Status, Plan, int]:
        r = self.mcpp.pos2v(pi[0])
        X_last = SearchState(State(r, 0), rt.get_safe_intervals(r)[0])
        states: List[SearchState] = [X_last]
        partition_inds = []
        goal_seq = [1]
        N, backtrack_steps, ts = 0, 0, time.perf_counter()
        while goal_seq[-1] < len(pi):
            self.reset_h_true_dist()
            status, sub_P, n = self.planner.plan(
                X_start = X_last,
                goal_seq = [self.mcpp.pos2v(pi[g]) for g in goal_seq],
                rt = rt,
                h = self.h,
                max_num_nodes_explored = max_num_nodes_explored,
                time_limit = time_limit - (time.perf_counter() - ts),
                stay = goal_seq[-1] == len(pi)-1
            )
            N += n
            if status == Status.SUCCESS:
                partition_inds.append([goal_seq, (len(states)-1, len(states) + len(sub_P.states) - 2)])
                states.extend(sub_P.states[1:])
                X_last = states[-1]
                goal_seq = [goal_seq[-1] + 1]
                backtrack_steps = 0
            elif partition_inds != [] and backtrack_steps < self.max_backtrack_step:
                backtrack_steps += 1
                last_goal_seq, to_rmv = partition_inds.pop()
                goal_seq = last_goal_seq + goal_seq
                states = states[:to_rmv[0]+1]
                X_last = states[-1]
                print(f"\t\t-> Backtrack to solve {goal_seq} using ML-SIPP")
            else:
                print(f"\t\t-> Failed to solve {goal_seq} with adaptive approach, postponded to solve with ML-SIPP on the full goal sequence")
                return Status.POSTPONED, None, N

        return Status.SUCCESS, Plan(states), N
