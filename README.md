# LS-MCPP
This repository is the benchmark and implementation of the algorithms for the graph-based multi-robot coverage path planning problem from the following two papers:

- Branch master: *Jingtao Tang, Zining Mao, and Hang Ma. "Large-Scale Multi-Robot Coverage Path Planning on Grids with Path Deconfliction." T-RO (to appear). [[paper]](https://arxiv.org/abs/2411.01707), [[project]](https://sites.google.com/view/lsmcpp)

- Branch aaai24: *Jingtao Tang and Hang Ma. "Large-Scale Multi-Robot Coverage Path Planning via Local Search." AAAI 2024. [[paper]](https://arxiv.org/pdf/2312.10797.pdf), [[simulation]](https://vimeo.com/894744842)*, [[project]](https://reso1.github.io/blog/posts/grid_mcpp)

Please cite us if you use this code for the multi-robot coverage path planning problem.

## Installation

`pip install -e .`

## Usage
```bash
python main.py [-h] [--init_sol_type INIT_SOL_TYPE] [--prio_type PRIO_TYPE] [--M M] [--S S] [--gamma GAMMA] [--tf TF] [--scale SCALE] [--write WRITE] [--verbose VERBOSE] istc
```
- Required:
  - `istc`: the instance name stored in directory 'data/instances' or 'MIP-MCPP/data/instances'.
- Optional:
  - `--init_sol_type INIT_SOL_TYPE`: Initial solution type. Choose from {VOR, MFC, MSTCStar, MIP} (default=MFC)
  - `-prio_type PRIO_TYPE`: Operator sampling type. Choose from {Heur, Rand} (default=Heur)
  - `--M M`: Max iteration (default=3e3)
  - `--S S`: Forced deduplication step size (default=100)
  - `--gamma GAMMA`: Pool weight decaying factor (default=1e-2)
  - `--tf TF`: The final temperature used to calculate the temperature decaying factor
  - `--scale SCALE`: Plot scaling factor
  - `--verbose VERBOSE`: Is verbose printing
  - `--write WRITE`: Is writing the solution
  - `--record RECORD`: Is recording the path costs of each iteration
  - `--draw DRAW`: Is drawing the final solution
  - `--random_remove RANDOM_REMOVE`: Is randomly making 20 percentage of terrain vertices incomplete

## File Structure
- benchmark/
  - gridmaps: the 2d grid maps (partly from https://movingai.com/benchmarks/grids.html)
  - instances: the MCPP instances with roots and weights specified
  - instance.py: the class of MCPP instance
  - plans.py: the class of plan (trajectories) for the robots
  - simulation.py: a simple visualizer for MCPP execution animation
- conflict_solver/ 
  - high_level_planner.py: the high-level planner of priority-based search
  - low_level_planner.py: the chaining, holistic (multi-label), and adaptive approaches for the low-level planner
  - reservation_table.py: the reservation table of time intervals (for safe-interval path planning)
  - states.py: state representations for the low-level planner
- data[optinal]: the accompanying simulation exp results for the paper ([download link](https://drive.google.com/file/d/1P4infbS0uEnRhemXQyvgKJvTucT-lO92/view?usp=drive_link)).
- mcpp/
  - estc.py: the Extended STC algorithm
  - graph.py: class of the decomposed graph
  - local_search.py: the proposed local search framework for MCPP
  - operator.py: the three boundary editing operators
  - pool.py: class of operator pool
  - solution.py: class of the MCPP solution
  - utils.py: other ultility functions
- MIP-MCPP: repo of the work "*[Mixed Integer Programming for Time-Optimal Multi-Robot Coverage Path Planning With Efficient Heuristics](https://arxiv.org/pdf/2306.17609)*"
- demo.ipynb: a demo code for a small MCPP instance
- exp_runner.py: the experiment runner
- plot.py: plot functions for the experiments

## Benchmark dataset
Please upzip data.zip to access the MCPP benchmark instances

## License
LS-MCPP is released under the GPL version 3. See LICENSE.txt for further details.
