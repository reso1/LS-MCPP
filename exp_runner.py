import os
import argparse
from pathlib import Path

from lsmcpp.solution import *
from lsmcpp.local_search import *
from lsmcpp.conflict_solver.low_level_planner import *
from lsmcpp.conflict_solver.high_level_planner import PBS, Node
from lsmcpp.benchmark.instance import MCPP
from lsmcpp.benchmark.plan import *


def run(name, num_rmv_ratios, num_seeds, runner, decmcpp_limit=None, methods=None, save=False):
    mcpp = MCPP.read_instance(os.path.join("benchmark", "instances", f"{name}.mcpp"))
    rmv_ratios = np.linspace(0, 0.2, num_rmv_ratios, endpoint=True)
    seeds = np.arange(num_seeds)
    
    if runner is CPP_exp_runner:
        baseline_name = "uw"
        base_dir = os.path.join("data", "cpp")
    elif runner is MCPP_exp_runner or runner is MCPP_MIP_exp_runner:
        baseline_name = "VOR"
        base_dir = os.path.join("data", "mcpp")
    elif runner is DecMCPP_exp_runner:
        base_dir = os.path.join("data", "mcpp_dec")

    if runner is DecMCPP_exp_runner:
        ret = runner(mcpp, rmv_ratios, seeds, decmcpp_limit, methods, base_dir=base_dir if save else None)
    elif runner is MCPP_exp_runner:
        ret = runner(mcpp, rmv_ratios, seeds, base_dir=base_dir if save else None)
    else:
        ret = runner(mcpp, rmv_ratios, seeds, base_dir=base_dir if save else None)
    
    if save and runner is not MCPP_MIP_exp_runner:
        save_dir = os.path.join(base_dir, "ablations")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(save_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(ret, f, pickle.HIGHEST_PROTOCOL)

    if runner is CPP_exp_runner or runner is MCPP_exp_runner:
        print("\nRUN SUMMARY:")
        for rmv_ratio, value in calc_stats(ret, baseline_name).items():
            print(f"rmv ratio={rmv_ratio}:")
            for method, vals in value.items():
                if method == baseline_name:
                    mean, std = np.mean(ret[rmv_ratio][0][method]), np.std(ret[rmv_ratio][0][method])
                    print(f"\t{method}:\t metric(mean)={mean:.2f},\t metric(std)={std:.2f},\t runtime(mean)={vals[2]:.2E}\t, runtime(mean)={vals[3]:.2E}")
                else:
                    print(f"\t{method}:\t metric(mean)={vals[0]:.2%},\t metric(std)={vals[1]:.2%},\t runtime(mean)={vals[2]:.2E}\t, runtime(mean)={vals[3]:.2E}")
    
    if runner is DecMCPP_exp_runner:
        print("\nRUN SUMMARY:")
        for rmv_ratio, value in ret.items():
            print(f"rmv ratio={rmv_ratio}:")
            num_success, taus = 0, []
            for method in ret[0][0].keys():
                taus, runtime = np.array(ret[rmv_ratio][0][method]), np.array(ret[rmv_ratio][1][method])
                taus = [tau for tau in taus if tau != float("inf")]
                print(f"\t{method}:\t tau(mean)={np.mean(taus):.2f},\t tau(std)={np.std(taus):.2f},\t success-ratio={len(taus)/num_seeds:.2%},\t runtime(mean)={np.mean(runtime):.2E}\t, runtime(mean)={np.std(runtime):.2E}")


def calc_stats(data, baseline_name):
    ratios = {}
    for x in data.keys():
        ratios[x] = {}
        baseline_cost, baseline_runtime = data[x][0][baseline_name], data[x][1][baseline_name]
        for method in data[0][0].keys():
            cost, runtime = np.array(data[x][0][method]), np.array(data[x][1][method])
            _reduction = (baseline_cost - cost) / baseline_cost
            ratios[x][method] = (np.mean(_reduction), np.std(_reduction), np.mean(runtime), np.std(runtime))
            
    return ratios


def report_mcpp_runtime(name):
    with open(os.path.join("data", "mcpp", "ablations", f"{name}.pkl"), "rb") as f:
        data = pickle.load(f)

    runtimes = defaultdict(list)
    for x in data.keys():
        for method in data[0][0].keys():
            cost, runtime = np.array(data[x][0][method]), np.array(data[x][1][method])
            runtimes[method].append(runtime)
    for method, t in runtimes.items():
        print(f"{method}: {np.mean(t):.4f}s, {np.std(t):.4f}s, {np.mean(t)/60:.4f}m, {np.std(t)/60:.3f}m")


def report_dec_mcpp_runtime(name):
    with open(os.path.join("data", "mcpp_dec", "ablations", f"{name}.pkl"), "rb") as f:
        data = pickle.load(f)

    success_inds = defaultdict(lambda: defaultdict(list))
    methods = ["chaining", "holistic", "adaptive"]
    for rmv_ratio, value in data.items():
        for method in methods:
            taus, runtime = np.array(value[0][method]), np.array(value[1][method])
            succeeded = [i for i, tau in enumerate(taus) if tau != float("inf")]
            if succeeded:
                success_inds[method][rmv_ratio] = succeeded


    runtime_total_chaining, runtime_total_holistic, runtime_total_adaptive = [], [], []
    runtime_chaining_all, runtime_holistic_all, runtime_adaptive_all = [], [], []
    for rmv_ratio in success_inds["chaining"].keys():
        inds_chaining = success_inds["chaining"][rmv_ratio]
        inds_holistic = success_inds["holistic"][rmv_ratio]
        inds_adaptive = success_inds["adaptive"][rmv_ratio]
        if inds_chaining != [] and inds_holistic == [] and inds_adaptive != []:
            inds_all_success = set(inds_chaining) & set(inds_adaptive)
            runtime_total_chaining.extend([data[rmv_ratio][1]["chaining"][idx] for idx in inds_all_success])
            runtime_total_adaptive.extend([data[rmv_ratio][1]["adaptive"][idx] for idx in inds_all_success])
            runtime_chaining_all.extend([data[rmv_ratio][1]["chaining"][idx] for idx in inds_chaining])
            runtime_adaptive_all.extend([data[rmv_ratio][1]["adaptive"][idx] for idx in inds_adaptive])
        elif inds_chaining == [] and inds_holistic != [] and inds_adaptive != []:
            inds_all_success = set(inds_chaining) & set(inds_adaptive)
            runtime_total_holistic.extend([data[rmv_ratio][1]["holistic"][idx] for idx in inds_all_success])
            runtime_total_adaptive.extend([data[rmv_ratio][1]["adaptive"][idx] for idx in inds_all_success])
            runtime_holistic_all.extend([data[rmv_ratio][1]["holistic"][idx] for idx in inds_holistic])
            runtime_adaptive_all.extend([data[rmv_ratio][1]["adaptive"][idx] for idx in inds_adaptive])
        else:
            inds_all_success = set(inds_chaining) & set(inds_holistic) & set(inds_adaptive)
            runtime_total_chaining.extend([data[rmv_ratio][1]["chaining"][idx] for idx in inds_all_success])
            runtime_total_holistic.extend([data[rmv_ratio][1]["holistic"][idx] for idx in inds_all_success])
            runtime_total_adaptive.extend([data[rmv_ratio][1]["adaptive"][idx] for idx in inds_all_success])
            runtime_chaining_all.extend([data[rmv_ratio][1]["chaining"][idx] for idx in inds_chaining])
            runtime_holistic_all.extend([data[rmv_ratio][1]["holistic"][idx] for idx in inds_holistic])
            runtime_adaptive_all.extend([data[rmv_ratio][1]["adaptive"][idx] for idx in inds_adaptive])

    if runtime_total_chaining != []:
        print(f"chaining: {np.mean(runtime_total_chaining):.3f}s, {np.std(runtime_total_chaining):.3f}s, {np.mean(runtime_total_chaining)/60:.3f}m, {np.std(runtime_total_chaining)/60:.3f}m")
        print(f"chaining (all): {np.mean(runtime_chaining_all):.3f}s, {np.std(runtime_chaining_all):.3f}s, {np.mean(runtime_chaining_all)/60:.3f}m, {np.std(runtime_chaining_all)/60:.3f}m")
    if runtime_total_holistic != []:
        print(f"holistic: {np.mean(runtime_total_holistic):.3f}s, {np.std(runtime_total_holistic):.3f}s, {np.mean(runtime_total_holistic)/60:.3f}m, {np.std(runtime_total_holistic)/60:.3f}m")
        print(f"holistic (all): {np.mean(runtime_holistic_all):.3f}s, {np.std(runtime_holistic_all):.3f}s, {np.mean(runtime_holistic_all)/60:.3f}m, {np.std(runtime_holistic_all)/60:.3f}m")
    print(f"adaptive: {np.mean(runtime_total_adaptive):.3f}s, {np.std(runtime_total_adaptive):.3f}s, {np.mean(runtime_total_adaptive)/60:.3f}m, {np.std(runtime_total_adaptive)/60:.3f}m")
    print(f"adaptive (all): {np.mean(runtime_adaptive_all):.3f}s, {np.std(runtime_adaptive_all):.3f}s, {np.mean(runtime_adaptive_all)/60:.3f}m, {np.std(runtime_adaptive_all)/60:.3f}m")
    

def CPP_exp_runner(mcpp:MCPP, rmv_ratios, seeds, base_dir=None):

    def full_stc_unweighted(dec_graph:DecGraph, r:tuple):
        tree = ExtSTCPlanner.kruskal_unweighted(dec_graph.T)
        pi = ExtSTCPlanner.full_stc(r, tree, dec_graph.dV)
        assert Helper.is_path_valid(pi)
        assert set(list(dec_graph.D.nodes)) == set(pi)
        return pi


    def full_stc(dec_graph:DecGraph, r:tuple):
        tree = ExtSTCPlanner.modified_kruskal_no_turn_reduction(dec_graph.T)
        pi = ExtSTCPlanner.full_stc(r, tree, dec_graph.dV)
        assert Helper.is_path_valid(pi)
        assert set(list(dec_graph.D.nodes)) == set(pi)
        return pi


    def estc_no_parallel_rewiring(dec_graph:DecGraph, r:tuple):
        tree = ExtSTCPlanner.modified_kruskal(dec_graph.T)
        pi = ExtSTCPlanner.full_stc(r, tree, dec_graph.dV)
        assert Helper.is_path_valid(pi)
        assert set(list(dec_graph.D.nodes)) == set(pi)
        return pi


    def estc_no_turn_reduction(dec_graph:DecGraph, r:tuple):
        tree = ExtSTCPlanner.modified_kruskal_no_turn_reduction(dec_graph.T)
        pi = ExtSTCPlanner.full_stc(r, tree, dec_graph.dV)
        pi = ExtSTCPlanner.parallel_rewiring(pi, dec_graph.D)
        assert Helper.is_path_valid(pi)
        assert set(list(dec_graph.D.nodes)) == set(pi)
        return pi

    ret = {}
    for rmv_ratio in rmv_ratios:
        costs, runtime = defaultdict(list), defaultdict(list)
        for i, mutant in enumerate(mcpp.randomized_mutants([rmv_ratio], seeds)):
            _pi_res = {}
            print(f"\n{mutant.name}: rmv ratio={rmv_ratio}, seed={seeds[i]}")
            dg = contract(mutant._G_legacy)
            rd = mutant.legacy_vertex(mutant.R[0])
            r = dg.undecomp(rd)
            if r not in dg.T:
                for part in ['top', 'bot']:
                    new_r = (r[0], r[1], part)
                    if new_r in dg.dV and rd in dg.dV[new_r]:
                        r = new_r
                        break
            
            ts = time.perf_counter()
            pi_uw = full_stc_unweighted(dg, r)
            pi_uw = ExtSTCPlanner.root_align(pi_uw, rd)
            runtime["uw"].append(time.perf_counter()-ts)
            costs["uw"].append(Solution.path_cost(mutant._G_legacy, pi_uw))
            _pi_res["uw"] = pi_uw

            ts = time.perf_counter()
            pi_full = full_stc(dg, r)
            pi_full = ExtSTCPlanner.root_align(pi_full, rd)
            runtime["full"].append(time.perf_counter()-ts)
            costs["full"].append(Solution.path_cost(mutant._G_legacy, pi_full))
            _pi_res["full"] = pi_full

            ts = time.perf_counter()
            pi_no_tr = estc_no_turn_reduction(dg, r)
            pi_no_tr = ExtSTCPlanner.root_align(pi_no_tr, rd)
            runtime["pr"].append(time.perf_counter()-ts)
            costs["pr"].append(Solution.path_cost(mutant._G_legacy, pi_no_tr))
            _pi_res["pr"] = pi_no_tr

            ts = time.perf_counter()
            pi_no_pr = estc_no_parallel_rewiring(dg, r)
            pi_no_pr = ExtSTCPlanner.root_align(pi_no_pr, rd)
            runtime["tr"].append(time.perf_counter()-ts)
            costs["tr"].append(Solution.path_cost(mutant._G_legacy, pi_no_pr))
            _pi_res["tr"] = pi_no_pr

            ts = time.perf_counter()
            pi = ExtSTCPlanner.plan(r, dg)
            pi = ExtSTCPlanner.root_align(pi, rd)
            runtime["estc"].append(time.perf_counter()-ts)
            costs["estc"].append(Solution.path_cost(mutant._G_legacy, pi))
            _pi_res["estc"] = pi

            if base_dir:
                for method in ["uw", "full", "pr", "tr", "estc"]:
                    save_dir = os.path.join(base_dir, "solutions", mcpp.name, method)
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    with open(os.path.join(save_dir, f"{rmv_ratio:.3f}-{seeds[i]}.pkl"), "wb") as f:
                        pickle.dump(_pi_res[method], f, pickle.HIGHEST_PROTOCOL)

        ret[rmv_ratio] = (costs, runtime)

    return ret


def MCPP_exp_runner(mcpp:MCPP, rmv_ratios, seeds, base_dir=None):
    ret = {}
    M = 1e3 * np.sqrt(mcpp.G.number_of_nodes()/mcpp.k)
    S = int(M / 20)
    
    for rmv_ratio in rmv_ratios:
        makespan, runtime = defaultdict(list), defaultdict(list)
        for i, mutant in enumerate(mcpp.randomized_mutants([rmv_ratio], seeds)):            
            _pi_res = {}
            print(f"\n{mutant.name}: rmv ratio={rmv_ratio}, seed={seeds[i]}")
            ts = time.perf_counter()
            if os.path.exists(os.path.join("data", "mcpp", "solutions", mcpp.name, "VOR", f"{rmv_ratio:.3f}-{seeds[i]}.pkl")):
                with open(os.path.join("data", "mcpp", "solutions", mcpp.name, "VOR", f"{rmv_ratio:.3f}-{seeds[i]}.pkl"), "rb") as f:
                    _VOR_sol = pickle.load(f)
            else:
                _VOR_sol = Voronoi_sol(mutant)
            _VOR_rt = time.perf_counter() - ts
            makespan["VOR"].append(_VOR_sol.tau)
            runtime["VOR"].append(_VOR_rt)
            _pi_res["VOR"] = _VOR_sol

            ts = time.perf_counter()
            if os.path.exists(os.path.join("data", "mcpp", "solutions", mcpp.name, "MFC", f"{rmv_ratio:.3f}-{seeds[i]}.pkl")):
                with open(os.path.join("data", "mcpp", "solutions", mcpp.name, "MFC", f"{rmv_ratio:.3f}-{seeds[i]}.pkl"), "rb") as f:
                    _MFC_sol = pickle.load(f)
            else:
                _MFC_sol = MFC_sol(mutant)
            _MFC_rt = time.perf_counter() - ts
            makespan["MFC"].append(_MFC_sol.tau)
            runtime["MFC"].append(_MFC_rt)
            _pi_res["MFC"] = _MFC_sol

            ts = time.perf_counter()
            if os.path.exists(os.path.join("data", "mcpp", "solutions", mcpp.name, "MSTC*", f"{rmv_ratio:.3f}-{seeds[i]}.pkl")):
                with open(os.path.join("data", "mcpp", "solutions", mcpp.name, "MSTC*", f"{rmv_ratio:.3f}-{seeds[i]}.pkl"), "rb") as f:
                    _MSTCStar_sol = pickle.load(f)
            else:
                _MSTCStar_sol = MSTCStar_sol(mutant)
            _MSTCStar_rt = time.perf_counter() - ts
            makespan["MSTC*"].append(_MSTCStar_sol.tau)
            runtime["MSTC*"].append(_MSTCStar_rt)
            _pi_res["MSTC*"] = _MSTCStar_sol

            init_sol = _MFC_sol if _MFC_sol.tau < _VOR_sol.tau else _VOR_sol

            if os.path.exists(os.path.join("data", "mcpp", "solutions", mcpp.name, "LS", f"{rmv_ratio:.3f}-{seeds[i]}.pkl")):
                with open(os.path.join("data", "mcpp", "solutions", mcpp.name, "LS", f"{rmv_ratio:.3f}-{seeds[i]}.pkl"), "rb") as f:
                    sol_opt = pickle.load(f)
                    rt = 0
            else:
                planner = LocalSearchMCPP(mutant, init_sol, PrioType.CompositeHeur, PoolType.Edgewise, verbose=False)
                recorder = defaultdict(list)
                sol_opt, rt = planner.run(
                    M=M,
                    S=S,
                    alpha=np.exp(np.log(0.2) / M),
                    gamma=0.01,
                    sample_type=SampleType.RouletteWheel,
                    record=recorder,
                    seed=seeds[i]
                )
            makespan["LS"].append(sol_opt.tau)
            runtime["LS"].append(rt)
            _pi_res["LS"] = sol_opt

            read = False
            if os.path.exists(os.path.join("data", "mcpp", "solutions", mcpp.name, "+VO", f"{rmv_ratio:.3f}-{seeds[i]}.pkl")):
                with open(os.path.join("data", "mcpp", "solutions", mcpp.name, "+VO", f"{rmv_ratio:.3f}-{seeds[i]}.pkl"), "rb") as f:
                    sol_opt = pickle.load(f)
                read = True
            else:
                planner = LocalSearchMCPP(mutant, init_sol, PrioType.CompositeHeur, PoolType.VertexEdgewise, verbose=False)
                recorder = defaultdict(list)
                sol_opt, rt = planner.run(
                    M=M,
                    S=S,
                    alpha=np.exp(np.log(0.2) / M),
                    gamma=0.01,
                    sample_type=SampleType.RouletteWheel,
                    record=recorder,
                    seed=seeds[i]
                )
            makespan["+VO"].append(sol_opt.tau)
            runtime["+VO"].append(rt)
            _pi_res["+VO"] = sol_opt

            if base_dir and not read:
                for method in ["VOR", "MFC", "MSTC*", "LS", "+VO"]:
                    save_dir = os.path.join(base_dir, "solutions", mcpp.name, method)
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    with open(os.path.join(save_dir, f"{rmv_ratio:.3f}-{seeds[i]}.pkl"), "wb") as f:
                        pickle.dump(_pi_res[method], f, pickle.HIGHEST_PROTOCOL)

        ret[rmv_ratio] = (makespan, runtime)
        
    return ret


def DecMCPP_exp_runner(mcpp:MCPP, rmv_ratios, seeds, limit, methods=['chaining', 'holistic', 'adaptive'], base_dir=None):
    ret = {}
    for rmv_ratio in rmv_ratios:
        costs, runtime = defaultdict(list), defaultdict(list)
        for i, mutant in enumerate(mcpp.randomized_mutants([rmv_ratio], seeds)):
            _pi_res = {}
            print(f"\n{mutant.name}: rmv ratio={rmv_ratio}, seed={seeds[i]}")
            with open(os.path.join("data", "mcpp", "solutions", mcpp.name, "+VO", f"{rmv_ratio:.3f}-{seeds[i]}.pkl"), "rb") as f:
                sol = pickle.load(f)

            for method in ['chaining', 'adaptive', 'holistic']:
                ts = time.perf_counter()            
                if method in methods:
                    if os.path.exists(os.path.join("data", "mcpp_dec", "solutions", mcpp.name, method, f"{rmv_ratio:.3f}-{seeds[i]}.pkl")):
                        with open(os.path.join("data", "mcpp_dec", "solutions", mcpp.name, method, f"{rmv_ratio:.3f}-{seeds[i]}.pkl"), "rb") as f:
                            plans = pickle.load(f)
                    else:
                        if method == "chaining":
                            low_level_planner = ChainingApproach(mutant, HeurType.TrueDist)
                        elif method == "adaptive":
                            low_level_planner = AdaptiveApproach(mutant, HeurType.TrueDist)
                        elif method == "holistic":
                            low_level_planner = HolisticApproach(mutant, HeurType.TrueDist)
                        plans = PBS(mutant, low_level_planner, *limit).run(sol, verbose=True)
                else:
                    plans = None

                runtime[method].append(time.perf_counter()-ts)
                if plans is None:
                    costs[method].append(float("inf"))
                    _pi_res[method] = None
                else:
                    costs[method].append(max([P[-1].time for P in plans]))
                    _pi_res[method] = plans

            if base_dir:
                for method in ['chaining', 'adaptive', 'holistic']:
                    save_dir = os.path.join(base_dir, "solutions", mcpp.name, method)
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    with open(os.path.join(save_dir, f"{rmv_ratio:.3f}-{seeds[i]}.pkl"), "wb") as f:
                        pickle.dump(_pi_res[method], f, pickle.HIGHEST_PROTOCOL)

        ret[rmv_ratio] = (costs, runtime)

    return ret


def MCPP_MIP_exp_runner(mcpp:MCPP, rmv_ratios, seeds, base_dir=None):
    MIP_solve(mcpp, rmv_ratios, seeds)
    save_dir = os.path.join(base_dir, "solutions", mcpp.name, "MIP")
    for rmv_ratio in rmv_ratios:
        for i, mutant in enumerate(mcpp.randomized_mutants([rmv_ratio], seeds)):
            print(f"\n{mutant.name}: rmv ratio={rmv_ratio}, seed={seeds[i]}")
            with open(os.path.join(save_dir, f"{rmv_ratio:.3f}-{seeds[i]}.solu"), "rb") as f:
                sol_edges, runtime = pickle.load(f)
            ts = time.perf_counter()
            _MIP_sol = MIP_sol(mutant, sol_edges)
            runtime += time.perf_counter() - ts            
            if base_dir:
                with open(os.path.join(save_dir, f"{rmv_ratio:.3f}-{seeds[i]}.pkl"), "wb") as f:
                    pickle.dump(_MIP_sol, f, pickle.HIGHEST_PROTOCOL)


def save_random_mutant_map(name, rmv_ratios, seeds):
    mcpp = MCPP.read_instance(os.path.join("benchmark", "instances", f"{name}.mcpp"))
    height, width = mcpp.height, mcpp.width
    for rmv_ratio in rmv_ratios:
        for i, mutant in enumerate(mcpp.randomized_mutants([rmv_ratio], seeds)):
            print(f"saving for {mutant.name}: rmv ratio={rmv_ratio}, seed={seeds[i]}")
            base_dir = os.path.join("benchmark", "gridmaps", name)
            Path(base_dir).mkdir(parents=True, exist_ok=True)
            f = open(os.path.join(base_dir, f"{rmv_ratio:.3f}-{seeds[i]}.map"), "w")
            f.writelines(f"type octile\n")
            f.writelines(f"height {height}\n")
            f.writelines(f"width {width}\n")
            f.writelines(f"map\n")

            M = np.zeros((height, width), dtype=int)
            for v in mutant.G.nodes:
                px, py = mutant.G.nodes[v]['pos']
                if 0 <= px < height and 0 <= py < width:
                    M[py][px] = 1
            for row in range(height):
                for col in range(width):
                    f.writelines("." if M[row][col] == 1 else "@")
                f.writelines("\n")
            f.close()
            
            base_dir = os.path.join("benchmark", "instances", name)
            Path(base_dir).mkdir(parents=True, exist_ok=True)
            f = open(os.path.join(base_dir, f"{rmv_ratio:.3f}-{seeds[i]}.mcpp"), "w")
            f.writelines(f"map: {name}\n")
            f.writelines(f"root: {[[r//mcpp.height, r%mcpp.height] for r in mutant.R]}\n")
            f.writelines(f"weighted: {mcpp.weighted}\n")
            f.writelines(f"weight_seed: 0\n")
            f.writelines(f"incomplete: {rmv_ratio != 0}\n")


def diff_sol_adaptive_approach():
    name = "floor_large" # "terrain_large"

    mcpp = MCPP.read_instance(os.path.join("benchmark", "instances", f"{name}.mcpp"))
    rmv_ratios = np.linspace(0, 0.2, 12, endpoint=True)
    seeds = np.arange(12)

    ret = {}
    save_dir = os.path.join("data", "mcpp_dec", "diff_init_sols", "solutions", mcpp.name)
    for rmv_ratio in rmv_ratios:
            makespan, runtime = defaultdict(list), defaultdict(list)
            for i, mutant in enumerate(mcpp.randomized_mutants([rmv_ratio], seeds)):
                for method in ["VOR", "MFC", "MSTC*", "LS", "+VO"]:
                    print(f"{name}, {rmv_ratio:.3f}, {i}, {method}")
                    ts = time.perf_counter()
                    if os.path.exists(os.path.join(save_dir, method, f"{rmv_ratio:.3f}-{seeds[i]}.pkl")):
                        with open(os.path.join(save_dir, method, f"{rmv_ratio:.3f}-{seeds[i]}.pkl"), "rb") as f:
                            plans = pickle.load(f)
                            runtime[method].append(time.perf_counter()-ts)
                    else:   
                        with open(os.path.join("data", "mcpp", "solutions", mcpp.name, method, f"{rmv_ratio:.3f}-{i}.pkl"), 'rb') as f:
                            sol = pickle.load(f)
                        plans = PBS(mutant, AdaptiveApproach(mutant, HeurType.TrueDist), runtime_limit=3600).run(sol, verbose=True)
                        runtime[method].append(time.perf_counter()-ts)
                    if plans is None:
                        makespan[method].append(float("inf"))
                    else:
                        makespan[method].append(max([P[-1].time for P in plans]))

                    Path(os.path.join(save_dir, method)).mkdir(parents=True, exist_ok=True)
                    with open(os.path.join(save_dir, method, f"{rmv_ratio:.3f}-{i}.pkl"), 'wb') as f:
                        pickle.dump(plans, f)
            
            ret[rmv_ratio] = (makespan, runtime)


    base_dir = os.path.join("data", "mcpp_dec", "diff_init_sols", "ablations")
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(base_dir, f"{name}.pkl"), "wb") as f:
        pickle.dump(ret, f, pickle.HIGHEST_PROTOCOL)


def diff_num_iters(name, num_rmv_ratios, num_seeds, scalers, save):
    mcpp = MCPP.read_instance(os.path.join("benchmark", "instances", f"{name}.mcpp"))
    rmv_ratios = np.linspace(0, 0.2, num_rmv_ratios, endpoint=True)
    seeds = np.arange(num_seeds)

    ret = {}
    for rmv_ratio in rmv_ratios:
        makespan, runtime = defaultdict(list), defaultdict(list)
        for i, mutant in enumerate(mcpp.randomized_mutants([rmv_ratio], seeds)):
            with open(os.path.join("data", "mcpp", "solutions", mcpp.name, "MFC", f"{rmv_ratio:.3f}-{i}.pkl"), 'rb') as f:
                MFC_sol = pickle.load(f)
            with open(os.path.join("data", "mcpp", "solutions", mcpp.name, "VOR", f"{rmv_ratio:.3f}-{i}.pkl"), 'rb') as f:
                VOR_sol = pickle.load(f)
            init_sol = MFC_sol if MFC_sol.tau < VOR_sol.tau else VOR_sol

            for n_iters_scaler in scalers:
                M = n_iters_scaler * 1e3 * np.sqrt(mcpp.G.number_of_nodes()/mcpp.k)
                S = int(M / 20)
                print(f"\n{name}, {rmv_ratio:.3f}, {i}: {n_iters_scaler}, # of iters={M:.0f}")
                ts = time.perf_counter()
                if os.path.exists(os.path.join("data", "mcpp", "diff_iters", mcpp.name, f"{n_iters_scaler}", f"{rmv_ratio:.3f}-{i}.pkl")):
                    with open(os.path.join("data", "mcpp", "diff_iters", mcpp.name, f"{n_iters_scaler}", f"{rmv_ratio:.3f}-{i}.pkl"), 'rb') as f:
                        sol = pickle.load(f)
                    runtime[n_iters_scaler].append(time.perf_counter()-ts)
                    makespan[n_iters_scaler].append(sol.tau)
                else:
                    planner = LocalSearchMCPP(mutant, init_sol, PrioType.CompositeHeur, PoolType.VertexEdgewise, verbose=False)
                    sol_opt, rt = planner.run(
                        M = M,
                        S = S,
                        alpha = np.exp(np.log(0.2) / M),
                        gamma = 0.01,
                        sample_type = SampleType.RouletteWheel,
                        record = None,
                        seed = seeds[i]
                    )
                    runtime[n_iters_scaler].append(time.perf_counter()-ts)
                    makespan[n_iters_scaler].append(sol_opt.tau)

                if save:
                    Path(os.path.join("data", "mcpp", "diff_iters", mcpp.name, f"{n_iters_scaler}")).mkdir(parents=True, exist_ok=True)
                    with open(os.path.join("data", "mcpp", "diff_iters", mcpp.name, f"{n_iters_scaler}", f"{rmv_ratio:.3f}-{i}.pkl"), 'wb') as f:
                        pickle.dump(sol_opt, f)

        ret[rmv_ratio] = (makespan, runtime)
    
    if save:
        with open(os.path.join("data", "mcpp", "diff_iters", f"{mcpp.name}.pkl"), "wb") as f:
            pickle.dump(ret, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Instance name")
    parser.add_argument("problem", help="problem type: [CPP, RelaxedMCPP, DecMCPP, SaveMutant, MIPSolving, DiffSols, DiffIters]")
    parser.add_argument("--num_rmv_ratios", default=12, help="# of removal ratios ranging from 0 to 0.2")
    parser.add_argument("--num_seeds", default=12, help="# of seeds ranging from 0 to # of seeds-1")
    parser.add_argument("--DecMCPP_runtime_limit", default="3600", help="running timeout in seconds for DecMCPP")
    parser.add_argument("--DecMCPP_node_limit", default="default", help="the maximum number of nodes to be explored per goal in the low-level planner for DecMCPP")
    parser.add_argument("--DecMCPP_methods", nargs='+', default=['chaining', 'holistic', 'adaptive'], help="methods to be used in DecMCPP")
    parser.add_argument("--save", help="Save the result")
    parser.add_argument("--num_iters_scalers_low", default=1, help="lower scaler for # of iterations for RelaxedMCPP exp with different iterations")
    parser.add_argument("--num_iters_scalers_high", default=2, help="high scaler for # of iterations for RelaxedMCPP exp with different iterations")
    args = parser.parse_args()

    name = args.name
    if args.problem == "CPP":
        ret = run(name, int(args.num_rmv_ratios), int(args.num_seeds), CPP_exp_runner, save=bool(args.save))
    elif args.problem == "RelaxedMCPP":
        ret = run(name, int(args.num_rmv_ratios), int(args.num_seeds), MCPP_exp_runner, save=bool(args.save))
    elif args.problem == "DecMCPP":
        ret = run(name, int(args.num_rmv_ratios), int(args.num_seeds), DecMCPP_exp_runner, (args.DecMCPP_runtime_limit, args.DecMCPP_node_limit), args.DecMCPP_methods, save=bool(args.save))
    elif args.problem == "SaveMutant":
        rmv_ratios = np.linspace(0, 0.2, int(args.num_rmv_ratios), endpoint=True)
        seeds = np.arange(int(args.num_seeds))
        save_random_mutant_map(name, rmv_ratios, seeds)
    elif args.problem == "MIPSolving":
        run(name, int(args.num_rmv_ratios), int(args.num_seeds), MCPP_MIP_exp_runner, save=True)
    elif args.problem == "DiffSols":
        diff_sol_adaptive_approach(name, int(args.num_rmv_ratios), int(args.num_seeds), save=bool(args.save))
    elif args.problem == "DiffIters":
        scalers = np.arange(float(args.num_iters_scalers_low), float(args.num_iters_scalers_high)+0.1, 0.5)
        diff_num_iters(name, int(args.num_rmv_ratios), int(args.num_seeds), scalers, save=bool(args.save))
