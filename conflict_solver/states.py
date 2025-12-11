from lsmcpp.benchmark.plan import State
from lsmcpp.conflict_solver.reservation_table import Interval


class SearchState(State):
    def __init__(self, X:State, safe_interval: Interval) -> None:
        super().__init__(X.pos, X.time, X.heading)
        self.safe_itvl = safe_interval
    
    def __str__(self) -> str:
        return f"(v={self.pos}, t={self.time}, heading={self.heading}, safe_itvl={self.safe_itvl})"

    @property
    def __index__(self) -> tuple:
        return (self.safe_itvl, self.heading, self.pos)


class LabeledState(SearchState):
    def __init__(self, X:State, safe_interval: Interval, label:int) -> None:
        super().__init__(X, safe_interval)
        self.label = label
    
    @property
    def __index__(self) -> tuple:
        return super().__index__ + (self.label,)
    
    def __str__(self) -> str:
        return f"(v={self.pos}, t={self.time}, heading={self.heading}, label={self.label}, safe_itvl={self.safe_itvl})"

    @property
    def search_state(self) -> SearchState:
        return SearchState(self, self.safe_itvl)
