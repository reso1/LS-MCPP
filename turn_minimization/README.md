# Notes 
- This folder implements the turn-minimization spanning tree generation and the subsequent MSTC* with a unit turn cost per 90deg encoded (TURN_COST_RATIO = 0.5 in benchmark/plan.py) from the paper "TMSTC*: A Path Planning Algorithm for Minimizing Turns in Multi-Robot Coverage". The simplified cost model is not identical to the original paper. Pplease refer to official implementation in https://github.com/CourierLo/TMSTC-Star.


- The Turn-minimization MCPP instances are assumed to be grid-complete, i.e., each quadrant coarsened grid H can subdivided into 4 small cells in graph G, where the coverage task is performed. Therefore, you only load the instances from data/instances, but do not call "randomized_mutants" as in demo.py to remove some cells in G and resample the roots.