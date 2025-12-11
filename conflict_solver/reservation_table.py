from __future__ import annotations
from typing import List, Tuple, Dict, Set
from collections import defaultdict

from lsmcpp.benchmark.plan import Plan


class Interval:

    def __init__(self, start:float, end:float, pi_idx:int=-1) -> None:
        # floating point precision issue
        self.start, self.end, self.pi_idx = round(start, 9), round(end, 9), pi_idx
        self.merged_from: Set[Interval] = set()
    
    @property
    def duration(self) -> float:
        return self.end - self.start
    
    def __hash__(self) -> int:
        return hash((self.start, self.end, self.pi_idx))
    
    def __eq__(self, other: Interval) -> bool:
        return self.start == other.start and self.end == other.end and self.pi_idx == other.pi_idx
    
    def __str__(self) -> str:
        return f"{self.pi_idx}:[{self.start}, {self.end})"

    def __lt__(self, other:Interval) -> bool:
        return self.end <= other.start
    
    def __gt__(self, other:Interval) -> bool:
        return self.start >= other.end
    
    def contains(self, other:Interval) -> bool:
        return self.start <= other.start and other.end <= self.end

    def intersects(self, other:Interval) -> bool:
        return not (self < other or self > other)
    
    def mergeable(self, other:Interval) -> bool:
        return self.start == other.end or self.end == other.start or self.intersects(other)

    @staticmethod
    def merge(this:Interval, other:Interval) -> Interval:
        merged = Interval(min(this.start, other.start), max(this.end, other.end))
        merged.merged_from.update(this.merged_from if this.merged_from else [this])
        merged.merged_from.update(other.merged_from if other.merged_from else [other])
        return merged


class TimeLine:

    def __init__(self) -> None:
        self.intervals: List[Interval] = []

    def copy(self) -> TimeLine:
        new_tl = TimeLine()
        new_tl.intervals = [Interval(itvl.start, itvl.end, itvl.pi_idx) for itvl in self.intervals]
        return new_tl

    def add(self, interval:Interval) -> None:
        idx = self.binary_search(interval)
        self.intervals.insert(idx, interval)
        self.check_merge(idx)

    def binary_search(self, val:Interval, start:int=None, end:int=None) -> int:
        start = 0 if start is None else start
        end = len(self.intervals)-1 if end is None else end
        
        if start > end:
            return start
        
        if start == end:
            if self.intervals[start] > val:
                return start
            else:
                return start+1
    
        mid = (start+end)//2
        if self.intervals[mid] < val:
            return self.binary_search(val, mid+1, end)
        elif self.intervals[mid] > val:
            return self.binary_search(val, start, mid-1)
        else:
            return mid
    
    def check_merge(self, check_idx:int) -> None:
        if check_idx > 0 and self.intervals[check_idx].mergeable(self.intervals[check_idx-1]):
            left, right, merged = check_idx-1, check_idx, True
        elif check_idx < len(self.intervals)-1 and self.intervals[check_idx].mergeable(self.intervals[check_idx+1]):
            left, right, merged = check_idx, check_idx+1, True
        else:
            merged = False

        while merged:
            self.intervals[left] = Interval.merge(self.intervals[left], self.intervals[right])
            self.intervals.pop(right)
            right = left + 1
            merged = right < len(self.intervals) and self.intervals[left].mergeable(self.intervals[right])

    def contains(self, itvl:Interval) -> bool:
        idx = self.binary_search(itvl)
        return (0 <= idx-1 < len(self.intervals) and self.intervals[idx-1].contains(itvl)) or \
               (0 <= idx   < len(self.intervals) and self.intervals[idx].contains(itvl))

    def intersects(self, itvl:Interval) -> bool:
        idx = self.binary_search(itvl)
        return (0 <= idx-1 < len(self.intervals) and self.intervals[idx-1].intersects(itvl)) or \
               (0 <= idx   < len(self.intervals) and self.intervals[idx].intersects(itvl))
    
    def get_intersected_itvl(self, itvl:Interval) -> Interval|None:
        idx = self.binary_search(itvl)
        if 0 <= idx-1 < len(self.intervals) and self.intervals[idx-1].intersects(itvl):
            return self.intervals[idx-1]
        if 0 <= idx   < len(self.intervals) and self.intervals[idx].intersects(itvl):
            return self.intervals[idx]
        return None


class ReservationTable:

    def __init__(self) -> None:
        self.P: Set[int] = set()
        # Timeline indexed by position instead of state
        self.collision_itvls: Dict[int, TimeLine] = defaultdict(TimeLine)

    def copy(self) -> ReservationTable:
        new_rt = ReservationTable()
        new_rt.P = self.P.copy()

        new_rt.collision_itvls = defaultdict(TimeLine)
        for pos, timeline in self.collision_itvls.items():
            new_rt.collision_itvls[pos] = timeline.copy()

        return new_rt

    def get(self, pos:int) -> TimeLine:
        return self.collision_itvls[pos]
    
    def get_safe_intervals(self, pos:int) -> List[Interval]:
        ret = []
        safe_interval = Interval(0, float('inf'))
        for coll_itvl in self.collision_itvls[pos].intervals:
            if safe_interval.start != coll_itvl.start:
                ret.append(Interval(safe_interval.start, coll_itvl.start))
            if coll_itvl.end < safe_interval.end:
                safe_interval = Interval(coll_itvl.end, safe_interval.end)
            else:
                return ret
        
        return ret + [safe_interval]

    @staticmethod
    def occupying_itvls(P:Plan, pi_idx:int) -> List[Tuple[int, Interval]]:
        if len(P) == 1:
            return [(P[0].pos, Interval(0, float('inf'), pi_idx))]
        if len(P) == 2:
            return [(P[0].pos, Interval(0, P[1].time, pi_idx)), (P[1].pos, Interval(P[0].time, float('inf'), pi_idx))]
        
        return [(P[idx].pos, Interval(P[idx-1].time, P[idx+1].time, pi_idx)) for idx in range(1, len(P)-1)] + \
               [(P[0].pos, Interval(0, P[1].time, pi_idx)), (P[-1].pos, Interval(P[-2].time, float('inf'), pi_idx))]

    def reserve_plan(self, P:Plan, i:int) -> None:
        """ reserve the whole path """
        assert P[-1].time != float('inf') 
        for v, itvl in ReservationTable.occupying_itvls(P, i):
            self.P.add(v)
            self.collision_itvls[v].add(itvl)

    def get_all_conflicted_inds(self, pos:int, itvl:Interval, pi_idx:int) -> Set[int]:
        ret = set()
        col_itvls = self.get(pos).get_intersected_itvl(itvl)
        if col_itvls:
            # when the collision interval is not merged from other intervals
            if col_itvls.merged_from == set() and col_itvls.pi_idx != pi_idx:
                ret.add(col_itvls.pi_idx)
            
            for col_itvl in col_itvls.merged_from:
                if col_itvl.pi_idx != pi_idx and itvl.intersects(col_itvl):
                    ret.add(col_itvl.pi_idx)

        return ret

    def is_conflicted_with(self, P:Plan, idx:int, target:Set[int]) -> bool:
        for v, itvl in ReservationTable.occupying_itvls(P, idx):
            if target.intersection(self.get_all_conflicted_inds(v, itvl, idx)):
                return True
        return False
