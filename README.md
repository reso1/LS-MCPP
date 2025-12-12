# LS-MCPP
This repository is the benchmark and implementation of the algorithms for the graph-based multi-robot coverage path planning problem from the following two papers:

- Branch master: *Jingtao Tang, Zining Mao, and Hang Ma. "Large-Scale Multi-Robot Coverage Path Planning on Grids with Path Deconfliction." T-RO (to appear). [[paper]](https://arxiv.org/abs/2411.01707), [[project]](https://sites.google.com/view/lsmcpp)

- Branch aaai24: *Jingtao Tang and Hang Ma. "Large-Scale Multi-Robot Coverage Path Planning via Local Search." AAAI 2024. [[paper]](https://arxiv.org/pdf/2312.10797.pdf), [[simulation]](https://vimeo.com/894744842)*, [[project]](https://reso1.github.io/blog/posts/grid_mcpp)

Please cite us if you use this code for the multi-robot coverage path planning problem.

## Installation

`pip install .`

## Usage
Please refer to `demo.ipynb` and `demo_turn_min_mcpp.ipynb`.

## File Structure
- benchmark/
  - instance.py: the class of MCPP instance
  - plan.py: the class of plan (trajectories) for the robots
  - simulation.py: a simple visualizer for MCPP execution animation
  - solution.py: the class of the MCPP solution
- conflict_solver/ 
  - high_level_planner.py: the high-level planner of priority-based search
  - low_level_planner.py: the chaining, holistic (multi-label), and adaptive approaches for the low-level planner
  - reservation_table.py: the reservation table of time intervals (for safe-interval path planning)
  - states.py: state representations for the low-level planner
- data[optinal]: 
  - the accompanying simulation exp results for the paper ([download link](https://drive.google.com/file/d/1P4infbS0uEnRhemXQyvgKJvTucT-lO92/view?usp=drive_link)).
  - gridmaps: the 2d grid maps (partly from https://movingai.com/benchmarks/grids.html)
  - instances: the MCPP instances with roots and weights specified
- mcpp/
  - disjoint_set.py: disjoint set data structure
  - estc.py: the Extended STC algorithm
  - graph.py: class of the decomposed graph
  - local_search.py: the proposed local search framework for MCPP
  - operator.py: the three boundary editing operators
  - planners.py: MCPP planner wrappers 
  - pool.py: class of operator pool
  - rooted_tree_cover.py: implmentation of Even, Guy, et al. "Min–max tree covers of graphs." OR-L'04
  - utils.py: utility functions
- turn_minimization/
  - interval.py: linear interval class
  - rectangle.py: 2d rectangle class
  - TMSTCStar.py: implementaion of Lu, Junjie, et al. "TMSTC*: A path planning algorithm for minimizing turns in multi-robot coverage." RA-L'23
- demo.ipynb: a demo code for a small MCPP instance
- demo_turn_min_mcpp.ipynb: a demo code for turn-minimizing MCPP algorithms

## Benchmark dataset
Please upzip data.zip to access the MCPP benchmark instances

## License
LS-MCPP is released under the GPL version 3. See LICENSE.txt for further details.
