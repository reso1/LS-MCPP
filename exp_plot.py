import os
import pickle
import LS_MCPP

import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

from MIP_MCPP.instance import Instance
from MIP_MCPP.misc import colormap

from LS_MCPP.utils import *
from LS_MCPP.local_search import *
from LS_MCPP.solution import *
from LS_MCPP.estc import ExtSTCPlanner
from LS_MCPP.graph import DecGraph

from main import runner


def load_tracer(s):
    with open(os.path.join("data", "runrecords", s), 'rb') as f:
        return pickle.load(f)


def plt_param():
    plt.yticks(fontsize=8)
    plt.xticks([0, 1500, 3000], ["0", "1.5e3", "3e3"], fontsize=8)


def plot(istc_name, ax, xy=(0.98, 0.2)):
    taus = np.zeros(12)
    data = np.zeros((12, 3000, 4))
    for i in range(12):
        tracer = load_tracer(f"{istc_name}-{i}.rec")
        data[i, :len(tracer["costs"])] = np.array(tracer["costs"])
        taus[i] = tracer["sol_opt"].tau

    c = ['r', 'g', 'b', 'k']
    iters = np.arange(3000)
    for i in range(4):
        mean = data[:, :, i].mean(axis=0)
        ax.plot(iters, mean, f"{c[i]}-")
        ax.fill_between(iters, mean-data[:, :, i].std(axis=0), mean+data[:, :, i].std(axis=0), color=f"{c[i]}", alpha=0.2)

    ax.annotate(r'$\tau^*$='+f"{np.mean(taus):.1f}"+r"$\pm$"+f"{np.std(taus):.1f}", xy=xy, xycoords='axes fraction',  horizontalalignment='right', verticalalignment='top')


def _plot(istc_name, ax, xy=(0.98, 0.2)):
    taus = np.zeros(12)
    data = np.zeros((12, 3000, 4))
    n_iters = []
    for i in range(12):
        tracer = load_tracer(f"{istc_name}-{i}.rec")
        data[i, :len(tracer["costs"])] = np.array(tracer["costs"])
        taus[i] = tracer["sol_opt"].tau
        n_iters.append(np.array(tracer["costs"]).shape[0])

    max_iters = max(n_iters)
    inds = sorted(range(12), key=lambda x:n_iters[x])
    selected = []
    for i in range(12):
        selected.append(inds[i:])

    c = ['r', 'g', 'b', 'k']
    data = data[:, :max_iters, :]
    for j in range(12):
        iters = np.arange(n_iters[inds[j]])
        _dat = np.ndarray((12-j, n_iters[inds[j]], 4))
        cnt = 0
        for k in selected[j]:
            _dat[cnt, :, :] = data[k, :n_iters[inds[j]], :]
            cnt += 1
        for i in range(4):
            mean = _dat[:, :, i].mean(axis=0)
            ax.plot(iters, mean, f"{c[i]}-")
            ax.fill_between(iters, mean-_dat[:, :, i].std(axis=0), mean+_dat[:, :, i].std(axis=0), color=f"{c[i]}", alpha=0.2)


    ax.annotate(r'$\tau^*$='+f"{np.mean(taus):.1f}"+r"$\pm$"+f"{np.std(taus):.1f}", xy=xy, xycoords='axes fraction',  horizontalalignment='right', verticalalignment='top')


def ablation_ESTC():

    def full_stc(dec_graph:DecGraph, r:tuple):
        tree = nx.minimum_spanning_tree(dec_graph.T)
        pi = ExtSTCPlanner.full_stc(r, tree, dec_graph.dV)
        assert set(list(dec_graph.D.nodes)) == set(pi)
        return pi

    def exp(istc):
        n_seeds, n_percs = 12, 32
        percs = np.linspace(0, 0.5, n_percs, endpoint=True)
        rand_seed = np.arange(0, n_seeds)
        tau0, tau1 = np.zeros((n_seeds, n_percs)), np.zeros((n_seeds, n_percs))

        for seed in rand_seed:
            for i in range(n_percs):
                dg = DecGraph.randomly_remove_Vd(istc, percs[i], seed)
                r = list(dg.T.nodes())[0]
                pi = ExtSTCPlanner.plan(r, dg)
                tau = DecGraph.path_cost(dg.D, pi)
                _pi = full_stc(dg, r)
                _tau = DecGraph.path_cost(dg.D, _pi)
                tau0[seed, i] = tau
                tau1[seed, i] = _tau
                print(seed, i, tau, _tau)
        
        ret = {"percs":percs, "tau0": tau0, "tau1":tau1}
        with open(istc.name, 'wb') as f:
            pickle.dump(ret, f, pickle.HIGHEST_PROTOCOL)

        return ret
    
    def plot(rec, ax):
        percs, tau0, tau1 = rec["percs"], rec["tau0"], rec["tau1"]
        mean0, mean1 = tau0.mean(axis=0), tau1.mean(axis=0)
        ax.plot(percs, mean0, 'k--', label='ESTC')
        ax.plot(percs, mean1, 'k-', label='Full-STC')
        ax.fill_between(percs, mean0-tau0.std(axis=0), mean0+tau0.std(axis=0), color='r', alpha=0.2)
        ax.fill_between(percs, mean1-tau1.std(axis=0), mean1+tau1.std(axis=0), color='b', alpha=0.2)
        plt.xticks([0, 0.125, 0.25, 0.375, 0.5], ["0%", "12.5%", "25%", "37.5%", "50%"])
        plt.legend()

    fig = plt.figure(figsize=(8, 1.5))
    plt.subplots_adjust(bottom=0.2)  # Adjust the value as needed
    gs = GridSpec(nrows=1, ncols=2, wspace=0.5)

    istc = Instance.read("floor_large-30x30-k4.istc", os.path.join("MIP-MCPP", "data", "instances"))
    ax = fig.add_subplot(gs[0, 0])
    # rec = exp(istc)
    with open(os.path.join("data", "runrecords", f"floor_large-30x30-k4-CPP.rec"), 'rb') as f:
        rec = pickle.load(f)
    plot(rec, ax)

    istc = Instance.read("terrain_large_1-32x32-k4.istc", os.path.join("MIP-MCPP", "data", "instances"))
    ax = fig.add_subplot(gs[0, 1])
    # rec = exp(istc)
    with open(os.path.join("data", "runrecords", f"terrain_large_1-32x32-k4-CPP.rec"), 'rb') as f:
        rec = pickle.load(f)
    plot(rec, ax)

    plt.tight_layout()
    plt.savefig("out.pdf", dpi=200)


def ablation_init_sol():

    fig = plt.figure(figsize=(8, 2.5))
    gs = GridSpec(nrows=2, ncols=4, wspace=0.3, hspace=0.3)

    # istc_1 - Voronoi
    ax = fig.add_subplot(gs[0, 0])
    plot("floor_large-30x30-k4-VOR", ax)
    plt_param()

    # istc_1 - MFC
    ax = fig.add_subplot(gs[0, 1])
    plot("floor_large-30x30-k4-MFC", ax)
    plt_param()

    # istc_1 - MSTC*
    ax = fig.add_subplot(gs[0, 2])
    plot("floor_large-30x30-k4-MSTCStar", ax)
    plt_param()
    
    # istc_1 - MIP
    ax = fig.add_subplot(gs[0, 3])
    plot("floor_large-30x30-k4-MIP", ax)
    plt_param()

    # istc_2 - Voronoi
    ax = fig.add_subplot(gs[1, 0])
    plot("terrain_large_1-32x32-k4-VOR", ax, (0.98, 0.5))
    plt_param()

    # istc_2 - MFC
    ax = fig.add_subplot(gs[1, 1])
    plot("terrain_large_1-32x32-k4-MFC", ax)
    plt_param()

    # istc_2 - MSTC*
    ax = fig.add_subplot(gs[1, 2])
    plot("terrain_large_1-32x32-k4-MSTCStar", ax)
    plt_param()

    # istc_2 - MIP
    ax = fig.add_subplot(gs[1, 3])
    plot("terrain_large_1-32x32-k4-MIP", ax)
    plt_param()
    plt.yticks([416, 418, 420], fontsize=8)


    plt.savefig("out.pdf", dpi=300)


def ablation_operators():

    fig = plt.figure(figsize=(8, 2.5))
    gs = GridSpec(nrows=2, ncols=4, wspace=0.3, hspace=0.3)

    # istc_1 - nG
    ax = fig.add_subplot(gs[0, 0])
    _plot("floor_large-30x30-k4-MFC-nG", ax, (0.98, 0.5))
    plt_param()
    plt.xlim(0, 100)
    plt.xticks([0, 50, 100], ["0", "50", "100"], fontsize=8)

    # istc_1 - nD
    ax = fig.add_subplot(gs[0, 1])
    plot("floor_large-30x30-k4-MFC-nD", ax)
    plt_param()

    # istc_1 - nE
    ax = fig.add_subplot(gs[0, 2])
    plot("floor_large-30x30-k4-MFC-nE", ax)
    plt_param()

    # istc_1
    ax = fig.add_subplot(gs[0, 3])
    plot("floor_large-30x30-k4-MFC", ax)
    plt_param()

    # istc_2 - nG
    ax = fig.add_subplot(gs[1, 0])
    _plot("terrain_large_1-32x32-k4-MFC-nG", ax)
    plt_param()

    # istc_2 - nD
    ax = fig.add_subplot(gs[1, 1])
    plot("terrain_large_1-32x32-k4-MFC-nD", ax)
    plt_param()

    # istc_2 - nE
    ax = fig.add_subplot(gs[1, 2])
    plot("terrain_large_1-32x32-k4-MFC-nE", ax)
    plt_param()
    
    # istc_2 - nE
    ax = fig.add_subplot(gs[1, 3])
    plot("terrain_large_1-32x32-k4-MFC", ax)
    plt_param()

    plt.savefig("out.pdf", dpi=300)


def ablation_sampling_forced_deduplication():
    fig = plt.figure(figsize=(8, 2.5))
    gs = GridSpec(nrows=2, ncols=4, wspace=0.3, hspace=0.3)

    # istc_1 - Random + no forced dedup
    ax = fig.add_subplot(gs[0, 0])
    plot("floor_large-30x30-k4-MFC-Rand-nFD", ax)
    plt_param()

    # istc_1 - Random
    ax = fig.add_subplot(gs[0, 1])
    plot("floor_large-30x30-k4-MFC-Rand-FD", ax)
    plt_param()

    # istc_1 - no forced dedup
    ax = fig.add_subplot(gs[0, 2])
    plot("floor_large-30x30-k4-MFC-Heur-nFD", ax)
    plt_param()

    # istc_1
    ax = fig.add_subplot(gs[0, 3])
    plot("floor_large-30x30-k4-MFC", ax)
    plt_param()

    # istc_2 - Random + no forced dedup
    ax = fig.add_subplot(gs[1, 0])
    plot("terrain_large_1-32x32-k4-MFC-Rand-nFD", ax)
    plt_param()

    # istc_2 - Random
    ax = fig.add_subplot(gs[1, 1])
    plot("terrain_large_1-32x32-k4-MFC-Rand-FD", ax)
    plt_param()

    # istc_2 - no forced dedup
    ax = fig.add_subplot(gs[1, 2])
    plot("terrain_large_1-32x32-k4-MFC-Heur-nFD", ax)
    plt_param()

    # istc_2
    ax = fig.add_subplot(gs[1, 3])
    plot("terrain_large_1-32x32-k4-MFC", ax)
    plt_param()

    plt.savefig("out.pdf", dpi=300)


def exp_run_all():

    istc_names = ["floor_small-5x10-k4", "maze_medium-20x20-k6", "terrain_medium-20x20-k4", 
                  "maze_large-30x30-k8", "floor_large-30x30-k4", "terrain_large_1-32x32-k4"]
    for istc_name in istc_names:
        for i in range(12):
            sol_opt, runtime = runner(istc_name, seed=i)
            with open("res.txt", "a") as f:
                f.writelines(f"{runtime:.3f}, {sol_opt.tau:.2f}\n")

    istc_names = ["AR0701SR-107x117-k20", "Shanghai2-128x128-k25", "NewYork1-128x128-k32"]
    for istc_name in istc_names:
        for i in range(12):
            sol_opt, runtime = runner(istc_name, M=1.5e4, S=5e2, seed=i)
            with open("res.txt", "a") as f:
                f.writelines(f"{runtime:.3f}, {sol_opt.tau:.2f}\n")


def exp_run_all_incomplete_G():

    istc_names = ["floor_small-5x10-k4", "maze_medium-20x20-k6", "terrain_medium-20x20-k4", 
                  "maze_large-30x30-k8", "floor_large-30x30-k4", "terrain_large_1-32x32-k4"]
    for istc_name in istc_names:
        for i in range(12):
            sol_opt, runtime = runner(istc_name, seed=i, is_random_remove=True)
            with open("res.txt", "a") as f:
                f.writelines(f"{runtime:.3f}, {sol_opt.tau:.2f}\n")

    istc_names = ["AR0701SR-107x117-k20", "Shanghai2-128x128-k25", "NewYork1-128x128-k32"]
    for istc_name in istc_names:
        for i in range(12):
            sol_opt, runtime = runner(istc_name, M=1.5e4, S=5e2, seed=i, is_random_remove=True)
            with open("res.txt", "a") as f:
                f.writelines(f"{runtime:.3f}, {sol_opt.tau:.2f}\n")


if __name__ == "__main__":
    # ablation_ESTC()

    # ablation_init_sol()

    # ablation_operators()

    ablation_sampling_forced_deduplication()

    # exp_run_all()

    # exp_run_all_incomplete_G()  
    