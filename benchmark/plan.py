from __future__ import annotations
from typing import List, Tuple, Dict, Set

import math
from enum import IntEnum
from lsmcpp.benchmark.instance import MCPP


TURN_COST_RATIO = 0.5

class Heading(IntEnum):
    E = 0
    N = 1
    W = 2
    S = 3

    def __str__(self) -> str:
        return self.name
    
    @staticmethod
    def rot_cost(this, other:Heading) -> float:
        if this == other:
            return 0
        elif abs(this - other) % 2 == 1:
            return TURN_COST_RATIO
        else:
            return 2 * TURN_COST_RATIO
    
    @property
    def radian(self) -> float:
        return self * 0.5 * math.pi
    
    @staticmethod
    def get(p:tuple, q:tuple) -> Heading:
        dx, dy = q[0] - p[0], q[1] - p[1]
        if dx > 0:
            return Heading.E
        elif dx < 0:
            return Heading.W
        elif dy > 0:
            return Heading.N
        elif dy < 0:
            return Heading.S
        else:
            raise ValueError(f"Invalid heading from {p} to {q}")


class State:

    def __init__(self, pos:int, time:float, heading=Heading.N) -> None:
        # floating point precision issue for time in weighted graphs
        self.pos, self.heading, self.time = pos, heading, round(time, 9)
    
    @property
    def __index__(self) -> tuple:
        return (self.time, self.heading, self.pos)
    
    def __eq__(self, other:State) -> bool:
        return self.__index__ == other.__index__
    
    def __lt__(self, other:State) -> bool:
        return self.__index__ < other.__index__
    
    def __hash__(self) -> int:
        return hash(self.__index__)
    
    def __str__(self) -> str:
        return f"(pos={self.pos}, t={self.time}, heading={self.heading})"
    
    def get_heading(self, goal:int, mcpp:MCPP) -> Heading:
        p, q = mcpp.legacy_vertex(self.pos), mcpp.legacy_vertex(goal)
        return Heading.get(p, q)
    

class Plan:
    
    def __init__(self, states:List[State]) -> None:
        self.states = states
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx:int) -> State:
        return self.states[idx]

    def copy(self) -> Plan:
        return Plan([State(X.pos, X.time, X.heading) for X in self.states])
    
    @staticmethod
    def construct_from_path(pi:list, mcpp:MCPP) -> Plan:
        S = [State(mcpp.pos2v(pi[0]), 0)]
        for idx in range(1, len(pi)):
            t0, p = S[-1].time, mcpp.pos2v(pi[idx])
            w_e = mcpp.G[S[-1].pos][p]["weight"]
            heading = S[-1].get_heading(p, mcpp)
            rot_time = Heading.rot_cost(S[-1].heading, heading) # * w_e
            if rot_time == 0:
                S.append(State(p, t0 + w_e, heading))
            else:
                S.append(State(S[-1].pos, t0 + rot_time, heading))
                S.append(State(p, t0 + w_e + rot_time, heading))

        return Plan(S)

