# LS-MCPP
This repository is the implementation of the boundary editing operators and the local search framework for the graph-based multi-robot coverage path planning problem from the following paper:

*Jingtao Tang and Hang Ma. "Large-Scale Multi-Robot Coverage Path Planning via Local Search." AAAI 2024. [[paper]](), [[simulation]](https://vimeo.com/894744842)*

Please cite this paper if you use this code for the multi-robot coverage path planning problem.

## Installation
`pip install -r requirements.txt`

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
- main.py: LS-MCPP main function
- exp_plot.py: functions related to experiments in the paper
- record_simulation.py: recorder for MCPP simulation
- MIP-MCPP: repo of the work "*Mixed Integer Programming for Time-Optimal Multi-Robot Coverage Path Planning With Efficient Heuristics*"
- data/
  - instances: the three very large-scale MCPP instances adopted from MAPF benchmark
  - MIP_solutions: MMRTC MIP solutions for the MCPP instances
  - runrecords: running results of the experiements in the paper
- LS_MCPP/
  - estc.py: the Extended STC algorithm
  - graph.py: class of the decomposed graph
  - local_search.py: the proposed local search framework for MCPP
  - operator.py: the three boundary editing operators
  - pool.py: class of operator pool
  - solution.py: class of the MCPP solution
  - utils.py: other ultility functions


## License
LS-MCPP is released under the GPL version 3. See LICENSE.txt for further details.
