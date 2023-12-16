import os
import sys
import argparse
from collections import defaultdict

import LS_MCPP

from MIP_MCPP.instance import Instance
from MIP_MCPP.misc import colormap

import matplotlib.pyplot as plt

from LS_MCPP.local_search import *
from LS_MCPP.solution import *


def runner(
    istc_name:str,
    M:int=3e3,
    S:int=1e2,
    init_sol_type:str="MFC",
    prio_type=PrioType.Heur,
    gamma:float=0.01,
    T_final:float=0.2,
    scale = 1.0,
    verbose:bool=False,
    is_write_sol=False,
    is_record=False,
    is_draw_sol=False,
    is_random_remove=False,
    seed=0,
) -> Tuple[Solution, float]:
    
    fn = f"{istc_name}.istc"
    try:
        dir = os.path.join("data", "instances")
        if fn in os.listdir(dir):
            istc = Instance.read(fn, dir)
        else:
            dir = os.path.join("MIP-MCPP", "data", "instances")
            istc = Instance.read(fn, dir)
    except:
        print(f"Does not found instance [{istc_name}].")
        sys.exit()
    
    if is_random_remove:
        dg = DecGraph.randomly_remove_Vd(istc, 0.2, seed, True)
        R = [istc.G.nodes[r]["pos"] for r in istc.R]
        ts = time.time()
        init_sol = incomplete_G_sol(dg, R)
        runtime = time.time() - ts
        planner = LocalSearchMCPP(dg, init_sol, PrioType.Heur, verbose=verbose, R=R)
    else:
        if init_sol_type == "VOR":
            ts = time.time()
            init_sol = Voronoi_sol(istc)
            runtime = time.time() - ts
        elif init_sol_type == "MFC":
            ts = time.time()
            init_sol = MFC_sol(istc)
            runtime = time.time() - ts
        elif init_sol_type == "MSTCStar":
            ts = time.time()
            init_sol = MSTCStar_sol(istc)
            runtime = time.time() - ts
        elif init_sol_type == "MIP":
            try:
                sols, dirs = [], []
                dir = os.path.join("MIP-MCPP", "data", "solutions")
                for file in os.listdir(dir):
                    if file.startswith(istc_name):
                        sols.append(file)
                        dirs.append(dir)
                dir = os.path.join("data", "MIP_solutions")
                for file in os.listdir(dir):
                    if file.startswith(istc_name):
                        sols.append(file)
                        dirs.append(dir)
                i = 2 # input(f"Found {len(sols)} solutions: [{', '.join([f'({i}):{n}' for i, n in enumerate(sols)])}].\nPlease enter the index of the solution to be used:\n")
                ts = time.time()
                init_sol = MIP_sol(istc, Instance.read_solution(os.path.splitext(sols[int(i)])[0], dirs[int(i)]))
                runtime = time.time() - ts
            except:
                print("MIP Solution does not exist.")
                sys.exit()

        planner = LocalSearchMCPP(istc, init_sol, prio_type, verbose=verbose)

    recorder = defaultdict(list) if is_record else None

    sol_opt, rt = planner.run(
        M=M,
        S=S,
        alpha=np.exp(np.log(T_final) / M),
        gamma=gamma,
        sample_type=SampleType.RouletteWheel,
        record=recorder, 
        seed=seed
    )

    if is_write_sol:
        sol_opt.save(os.path.join("data", "solutions", f"{istc_name}.sol"))
    
    if recorder:
        with open(os.path.join("data", "runrecords", f"{istc_name}-{init_sol_type}-{seed}.rec"), 'wb') as f:
            pickle.dump(recorder, f, pickle.HIGHEST_PROTOCOL)
    
    if is_draw_sol:
        fig, ax = plt.subplots(1, 2)
        c = colormap("spring")
        
        if is_random_remove:
            Vd = np.array(list(dg.V)).T
            ax[0].plot(Vd[0], Vd[1], '.k', ms=2*scale)
        else:
            istc.draw_covering_nodes(ax[0], 2 * scale)

        for i, t in enumerate(init_sol.Pi):
            tx, ty = zip(*t)
            ax[0].set_title(r"Initial Solution: $\tau$=" + f"{init_sol.tau:.3f}")
            ax[0].plot(tx, ty, lw=3 * scale, alpha=0.4, color=c(i / len(init_sol.Pi)))
            ax[0].axis("equal")

        if is_random_remove:
            Vd = np.array(list(dg.V)).T
            ax[1].plot(Vd[0], Vd[1], '.', ms=2*scale)
        else:
            istc.draw_covering_nodes(ax[1], 2 * scale)

        for i, t in enumerate(sol_opt.Pi):
            tx, ty = zip(*t)
            ax[1].set_title(r"After LS-MCPP: $\tau$=" + f"{sol_opt.tau:.3f}")
            ax[1].plot(tx, ty, lw=3 * scale, alpha=0.4, color=c(i / len(sol_opt.Pi)))
            ax[1].axis("equal")

        plt.tight_layout()
        plt.show()

    return sol_opt, runtime + rt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("istc", help="Instance name")
    parser.add_argument("--init_sol_type", default="MFC", help="Initial solution type. Choose from {VOR, MFC, MSTCStar, MIP}")
    parser.add_argument("--prio_type", default="Heur", help="Operator sampling type. Choose from {Heur, Rand}")
    parser.add_argument("--M", default=3e3, help="Max iteration")
    parser.add_argument("--S", default=1e2, help="Forced deduplication step size")
    parser.add_argument("--gamma", default=1e-2, help="Pool weight decaying factor")
    parser.add_argument("--tf", default=0.2, help="The final temperature used to calculate the temperature decaying factor")
    parser.add_argument("--scale", default=1.0, help="Plot scaling factor")
    parser.add_argument("--verbose", default=False, help="Is verbose printing")
    parser.add_argument("--write", default=False, help="Is writing the solution")
    parser.add_argument("--record", default=False, help="Is recording the path costs of each iteration")
    parser.add_argument("--draw", default=False, help="Is drawing the final solution")
    parser.add_argument("--random_remove", default=False, help="Is randomly making 20 percentage of terrain vertices incomplete")

    args = parser.parse_args()

    runner(
        istc_name       = args.istc,
        M               = int(args.M),
        S               = int(args.S),
        init_sol_type   = args.init_sol_type,
        prio_type       = PrioType.Heur if args.prio_type == "Heur" else PrioType.Random,
        gamma           = float(args.gamma),
        T_final         = float(args.tf),
        scale           = float(args.scale),
        verbose         = bool(args.verbose),
        is_write_sol    = bool(args.write),
        is_record       = bool(args.record),
        is_draw_sol     = bool(args.draw),
        is_random_remove= bool(args.random_remove)
    )
