from __future__ import annotations
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Interval:
    start: float
    end: float
    
    def copy(self) -> Interval:
        return Interval(self.start, self.end)
    
    @property
    def duration(self) -> float:
        return self.end - self.start
    
    def __eq__(self, other: Interval) -> bool:
        return self.start == other.start and self.end == other.end

    def __lt__(self, other:Interval) -> bool:
        return self.end < other.start
    
    def __gt__(self, other:Interval) -> bool:
        return self.start > other.end
    
    def contains(self, other:Interval) -> bool:
        return self.start <= other.start and other.end <= self.end

    def intersects(self, other:Interval) -> bool:
        return not (self < other or self > other)

    def intersection(self, other:Interval) -> Optional[Interval]:
        if self.intersects(other):
            return Interval(max(self.start, other.start), min(self.end, other.end))
        return None
    

def AABB(box1:List[Interval], box2:List[Interval]) -> bool:
    """ check if two axis-aligned bounding boxes overlap """
    for i in range(len(box1)):
        if not box1[i].intersects(box2[i]):
            return False
    return True
