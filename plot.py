import os
import pickle
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

from lsmcpp.solution import *
from lsmcpp.local_search import *
from lsmcpp.conflict_solver.low_level_planner import *
from lsmcpp.conflict_solver.high_level_planner import PBS, Node
from lsmcpp.benchmark.instance import MCPP
from lsmcpp.benchmark.plan import *
from lsmcpp.benchmark.simulation import simulate

SEP_LITERAL = "="*50 + "\n"
PLT_SHAPES = ['o', 'p', "X", "s", 'p', 'P',
              '*', 'v', '^', '<', '>', '+', "x", "h"]

colormap = lambda name='Accent': matplotlib.cm.get_cmap(name)


def load_tracer(s):
    with open(os.path.join("data", "runrecords", s), 'rb') as f:
        return pickle.load(f)


def plot_runrecord(istc_name, n_agents, seeds, ax, xy=(0.98, 0.2)):
    n_seeds = len(seeds)
    taus = np.zeros(n_seeds)
    n_iters = len(tracer["costs"])
    data = np.zeros((n_seeds, n_iters, n_agents))
    for i, seed in enumerate(seeds):
        tracer = load_tracer(f"{istc_name}-seed{seed}.rec")
        data[i, :len(tracer["costs"])] = np.array(tracer["costs"])
        taus[i] = tracer["sol_opt"].tau

    c = ['r', 'g', 'b', 'k']
    iters = np.arange(n_iters)
    for i in range(n_agents):
        mean = data[:, :, i].mean(axis=0)
        ax.plot(iters, mean, f"{c[i%len(c)]}-")
        ax.fill_between(iters, mean-data[:, :, i].std(axis=0), mean+data[:, :, i].std(axis=0), color=f"{c[i]}", alpha=0.2)

    ax.annotate(r'$\tau^*$='+f"{np.mean(taus):.1f}"+r"$\pm$"+f"{np.std(taus):.1f}", xy=xy, xycoords='axes fraction',  horizontalalignment='right', verticalalignment='top')


def plot_single_runrecord(istc_name, n_agents, ax, xy=(0.98, 0.2)):
    tracer = load_tracer(f"{istc_name}.rec")
    data = np.array(tracer["costs"])
    tau = tracer["sol_opt"].tau
    n_iters = data.shape[0]

    c = ['r', 'g', 'b', 'k']
    iters = np.arange(n_iters)
    for i in range(n_agents):
        ax.plot(iters, data[:, i], f"{c[i%len(c)]}-")

    ax.annotate(r'$\tau$='+f"{np.mean(tau):.1f}", xy=xy, xycoords='axes fraction', horizontalalignment='right', verticalalignment='top')


def plot_ablations():

    def _set_xlabels(ax):
        x_labels = np.linspace(0, 0.2, 3)
        ax.set_xticks(x_labels)
        ax.set_xticklabels([f"{_val:.1%}" for _val in x_labels], fontsize=axis_fontsize)
    
    def _set_ylabels(ax, y_lim=None):
        MIN, MAX = ax.get_ylim()
        y_labels = np.linspace(MIN + (MAX-MIN)*0.15, MIN + (MAX-MIN)*0.85, 3)
        ax.set_yticks(y_labels)
        ax.set_yticklabels([f"{_val:.1%}" for _val in y_labels], fontsize=axis_fontsize, rotation=50)
        ax.tick_params(axis='y', which='major', pad=-4)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')
        if y_lim:
            ax.set_ylim(y_lim)
    
    def _set_ylabels_dec_mcpp(ax):
        _, MAX = ax.get_ylim()
        y_labels = np.linspace(0, MAX*0.8, 3)
        ax.set_yticks(y_labels)
        ax.set_yticklabels([f"{_val:.1%}" for _val in y_labels], fontsize=axis_fontsize, rotation=50)
        ax.tick_params(axis='y', which='major', pad=-4)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')

        ax.set_ylim((-0.5*MAX, MAX))

    num_cols = 6
    widths = [2] * num_cols
    heights = [2, 1.1, 1.8, 1.3] * 2
    title_fontsize = 11
    legend_fontsize = 8
    axis_fontsize = 8
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, axes = plt.subplots(ncols=num_cols, nrows=8, constrained_layout=True, gridspec_kw=gs_kw, figsize=(2.2*num_cols, sum(heights)))
    red_ratios = defaultdict(list)

    def _grouped_plot(instance_names, gid):
        for i, name in enumerate(instance_names):
            mcpp = MCPP.read_instance(os.path.join("benchmark", "instances", f"{name}.mcpp"))
            
            # 1st row: draw instance as image
            I = np.zeros((mcpp.height+2, mcpp.width+2))
            for v in mcpp.G.nodes:
                x, y = mcpp.G.nodes[v]["pos"]
                I[mcpp.height-y, x+1] = 1
            axes[gid, i].imshow(I, cmap="gray")
            axes[gid, i].set_title(f"{name} (k={mcpp.k})", fontsize=title_fontsize)
            x_labels, y_labels = np.linspace(0, mcpp.width, 3, dtype=int), np.linspace(0, mcpp.height, 3, dtype=int)
            axes[gid, i].set_xticks(x_labels + [0.5]*3)
            axes[gid, i].set_xticklabels(x_labels, fontsize=axis_fontsize)
            axes[gid, i].set_yticks(y_labels + [0.5]*3)
            axes[gid, i].set_yticklabels(y_labels, fontsize=axis_fontsize)

        for i, name in enumerate(instance_names):
            with open(os.path.join("data", "cpp", "ablations", f"{name}.pkl"), "rb") as f:
                cpp_data = pickle.load(f)

            # 2nd row: ESTC ablation
            cpp_methods = {'full': 'ESTC', 'pr':"+PR", 'tr':"+TR", 'estc':"+Both"}
            colors = {'full': 'r', 'pr':'c', 'tr':'b', 'estc':'k'}
            lines = {'full': '-', 'pr':'--', 'tr':'-.', 'estc':':'}
            X = np.array(list(cpp_data.keys()))
            for method in cpp_methods.keys():
                _mean, _std = np.zeros_like(X), np.zeros_like(X)
                for _j, x in enumerate(cpp_data.keys()):
                    baseline_cost = np.array(cpp_data[x][0]['uw'])
                    _reduction = (baseline_cost - np.array(cpp_data[x][0][method])) / baseline_cost 
                    _mean[_j], _std[_j] = np.mean(_reduction), np.std(_reduction)

                axes[gid+1, i].plot(X, _mean, f'{lines[method]}{colors[method]}', label=cpp_methods[method])
                axes[gid+1, i].fill_between(X, _mean-_std, _mean+_std, alpha=0.2)
                axes[gid+1, i].legend(fancybox=True, framealpha=0.5, prop={'size': legend_fontsize}, labelspacing=0.0, borderpad=0)

            _set_xlabels(axes[gid+1, i])
            ymin, ymax = axes[gid+1, i].get_ylim()
            _set_ylabels(axes[gid+1, i], y_lim=(ymin - 0.1 * (ymax-ymin), ymax))
        
        for i, name in enumerate(instance_names):
            with open(os.path.join("data", "mcpp", "ablations", f"{name}.pkl"), "rb") as f:
                mcpp_data = pickle.load(f)

            mcpp_methods = {'MFC':"MFC", 'MIP':'MIP', 'MSTC*':"MSTC*", 'LS':"LS(-VO)", '+VO':"LS(+VO)"}
            colors = {'MFC':'r', 'MIP':'b', 'MSTC*':'c', 'LS':'g', '+VO':'k'}
            lines = {'MFC':'-', 'MIP':'-.', 'MSTC*':'--', 'LS':'-', '+VO':':'}

            mip_data, mip_dir = defaultdict(list), os.path.join("data", "mcpp", "solutions", name, "MIP")
            try:
                for fn in os.listdir(mip_dir):
                    if fn[-3:] == 'pkl':
                        with open(os.path.join(mip_dir, fn), "rb") as f:
                            mip_data[fn[:5]].append(pickle.load(f))
            except:
                mcpp_methods.pop('MIP')
                
            # 3rd row: LS-MCPP ablation
            for method in mcpp_methods:
                _mean, _std = np.zeros_like(X), np.zeros_like(X)
                for _j, x in enumerate(mcpp_data.keys()):
                    baseline_cost = np.array(mcpp_data[x][0]['VOR'])
                    if method == 'MIP':
                        if mip_data[f"{x:.3f}"] == []:
                            _reduction = np.nan
                        else:
                            _reduction = (baseline_cost - np.array([sol.tau for sol in mip_data[f"{x:.3f}"]])) / baseline_cost
                    else:
                        _reduction = (baseline_cost - np.array(mcpp_data[x][0][method])) / baseline_cost 
                    _mean[_j], _std[_j] = np.mean(_reduction), np.std(_reduction)

                axes[gid+2, i].plot(X, _mean, f'{lines[method]}{colors[method]}', label=mcpp_methods[method])
                axes[gid+2, i].fill_between(X, _mean-_std, _mean+_std, alpha=0.2)
                axes[gid+2, i].legend(fancybox=True, framealpha=0.5, prop={'size': legend_fontsize}, labelspacing=0.0, borderpad=0)

                red_ratios[method].append(_mean[_j])
            
            _set_xlabels(axes[gid+2, i])
            _set_ylabels(axes[gid+2, i])
        
        for i, name in enumerate(instance_names):
            with open(os.path.join("data", "mcpp_dec", "ablations", f"{name}.pkl"), "rb") as f:
                dec_mcpp_data = pickle.load(f)

            with open(os.path.join("data", "mcpp", "ablations", f"{name}.pkl"), "rb") as f:
                mcpp_data = pickle.load(f)
                
            # 4th row: Dec-MCPP
            dec_mcpp_methods = {'chaining':"CHA", 'holistic':"MLA", 'adaptive':"ADA"}
            colors = {'chaining':'r', 'holistic':'b', 'adaptive':'k'}
            lines = {'chaining':'-', 'holistic':'--', 'adaptive':':'}
            bar_width = 0.2/45
            bar_X_offset = {'chaining':-bar_width, 'holistic':0, 'adaptive':bar_width}

            success_inds = defaultdict(lambda: defaultdict(list))
            for method in dec_mcpp_methods:
                _suc_ratio = np.zeros_like(X)
                for _j, x in enumerate(mcpp_data.keys()):
                    taus = np.array(dec_mcpp_data[x][0][method])
                    succeeded = [_i for _i, tau in enumerate(taus) if tau != float("inf")]
                    _suc_ratio[_j] = len(succeeded)/len(X)
                    if succeeded:
                        success_inds[method][x] = succeeded
                
                # plot success ratio bars
                _ax = axes[gid+3, i].twinx()
                _ax.bar(X+bar_X_offset[method], height=_suc_ratio, width=bar_width, color=f'{colors[method]}')
                _ax.set_ylim(0, 5)
                _ax.set_yticks([1.0])
                _ax.set_yticklabels([f"{_val:.0%}" for _val in [1.0]], fontsize=7, rotation=-90)
                _ax.tick_params(axis='y', which='major', pad=0)
                for label in _ax.get_yticklabels():
                    label.set_verticalalignment('center')
            
            _improve_ratio = lambda data, x, _i: (np.array(data) - mcpp_data[x][0]["+VO"][_i] ) / mcpp_data[x][0]["+VO"][_i]
            _mean, _std = defaultdict(lambda: np.zeros_like(X)), defaultdict(lambda: np.zeros_like(X))
            for _j, x in enumerate(mcpp_data.keys()):
                if success_inds["chaining"][x] == [] and success_inds["holistic"][x] == []:
                    _mean["chaining"][_j] = _std["chaining"][_j] = np.nan
                    _mean["holistic"][_j] = _std["holistic"][_j] = np.nan
                    _mean["adaptive"][_j] = np.mean([_improve_ratio(dec_mcpp_data[x][0]["adaptive"][_i], x, _i) for _i in success_inds["adaptive"][x]])
                    _std["adaptive"][_j]  =  np.std([_improve_ratio(dec_mcpp_data[x][0]["adaptive"][_i], x, _i) for _i in success_inds["adaptive"][x]])
                if success_inds["chaining"][x] != [] and success_inds["holistic"][x] == []:
                    _mean["holistic"][_j] = _std["holistic"][_j] = np.nan
                    in_common_inds = list(set(success_inds["chaining"][x]) & set(success_inds["adaptive"][x]))
                    _mean["chaining"][_j] = np.mean([_improve_ratio(dec_mcpp_data[x][0]["chaining"][_i], x, _i) for _i in in_common_inds])
                    _std["chaining"][_j]  =  np.std([_improve_ratio(dec_mcpp_data[x][0]["chaining"][_i], x, _i) for _i in in_common_inds])
                    _mean["adaptive"][_j] = np.mean([_improve_ratio(dec_mcpp_data[x][0]["adaptive"][_i], x, _i) for _i in in_common_inds])
                    _std["adaptive"][_j]  =  np.std([_improve_ratio(dec_mcpp_data[x][0]["adaptive"][_i], x, _i) for _i in in_common_inds])
                if success_inds["chaining"][x] == [] and success_inds["holistic"][x] != []:
                    _mean["chaining"][_j] = _std["chaining"][_j] = np.nan
                    in_common_inds = list(set(success_inds["holistic"][x]) & set(success_inds["adaptive"][x]))
                    _mean["holistic"][_j] = np.mean([_improve_ratio(dec_mcpp_data[x][0]["holistic"][_i], x, _i) for _i in in_common_inds])
                    _std["holistic"][_j]  =  np.std([_improve_ratio(dec_mcpp_data[x][0]["holistic"][_i], x, _i) for _i in in_common_inds])
                    _mean["adaptive"][_j] = np.mean([_improve_ratio(dec_mcpp_data[x][0]["adaptive"][_i], x, _i) for _i in in_common_inds])
                    _std["adaptive"][_j]  =  np.std([_improve_ratio(dec_mcpp_data[x][0]["adaptive"][_i], x, _i) for _i in in_common_inds])
                if success_inds["chaining"][x] != [] and success_inds["holistic"][x] != []:
                    in_common_inds = list(set(success_inds["chaining"][x]) & set(success_inds["holistic"][x]) & set(success_inds["adaptive"][x]))
                    if in_common_inds:
                        _mean["chaining"][_j] = np.mean([_improve_ratio(dec_mcpp_data[x][0]["chaining"][_i], x, _i) for _i in in_common_inds])
                        _std["chaining"][_j]  =  np.std([_improve_ratio(dec_mcpp_data[x][0]["chaining"][_i], x, _i) for _i in in_common_inds])
                        _mean["holistic"][_j] = np.mean([_improve_ratio(dec_mcpp_data[x][0]["holistic"][_i], x, _i) for _i in in_common_inds])
                        _std["holistic"][_j]  =  np.std([_improve_ratio(dec_mcpp_data[x][0]["holistic"][_i], x, _i) for _i in in_common_inds])
                        _mean["adaptive"][_j] = np.mean([_improve_ratio(dec_mcpp_data[x][0]["adaptive"][_i], x, _i) for _i in in_common_inds])
                        _std["adaptive"][_j]  =  np.std([_improve_ratio(dec_mcpp_data[x][0]["adaptive"][_i], x, _i) for _i in in_common_inds])

            for method in dec_mcpp_methods:
                axes[gid+3, i].plot(X, _mean[method], f'{lines[method]}{colors[method]}', label=dec_mcpp_methods[method])
                axes[gid+3, i].fill_between(X, _mean[method]-_std[method], _mean[method]+_std[method], alpha=0.2)
                axes[gid+3, i].legend(fancybox=True, framealpha=0.5, loc='upper left', prop={'size': legend_fontsize}, labelspacing=0.0, borderpad=0)
                
            _set_xlabels(axes[gid+3, i])
            ymin, ymax = axes[gid+3, i].get_ylim()
            _set_ylabels_dec_mcpp(axes[gid+3, i])

    _grouped_plot(["floor_small", "floor_medium", "floor_large", "ht_chantry", "AR0205SR", "Shanghai2"], 0)
    _grouped_plot(["terrain_small", "terrain_medium", "terrain_large", "ost002d", "AR0701SR", "NewYork1"], 4)
    # axes[3, 2].legend(fancybox=True, framealpha=0.5, loc='upper center', prop={'size': legend_fontsize}, labelspacing=0.0, borderpad=0)
    # axes[3, 3].legend(fancybox=True, framealpha=0.5, loc='lower left', prop={'size': legend_fontsize}, labelspacing=0.0, borderpad=0)
    axes[7, 0].legend(fancybox=True, framealpha=0.5, loc='upper right', prop={'size': legend_fontsize}, labelspacing=0.0, borderpad=0)
    
    plt.savefig("ablations.pdf", bbox_inches='tight', dpi=300)
    # plt.show()
    red_ratios["MIP"] = np.array(red_ratios["MIP"])[~np.isnan(red_ratios["MIP"])]
    # print({k: np.mean(v) for k, v in red_ratios.items()})


def plot_diff_sol():

    def _plot_single(name, i, axes):
        ax = axes[2*i]
        init_sols = {"VOR":"VOR", "MFC":"MFC", "MSTC*":"MSTC*", "LS":"LS(-VO)", "+VO":"LS(+VO)"}
        colors = {'VOR':'r', 'MFC':'b', "MSTC*":'g', "LS":'c', '+VO':'k'}
        lines = {'VOR':'-', 'MFC':'--', "MSTC*":'-.', "LS":'-', '+VO':':'}
        bar_width = 0.12/45
        bar_X_offset = {'VOR':-2*bar_width, 'MFC':-bar_width, 'MSTC*':0, "LS":bar_width, '+VO':2*bar_width}
        success_inds = defaultdict(lambda: defaultdict(list))
        X = np.linspace(0, 0.2, 12, endpoint=True)
        with open(os.path.join("data", "mcpp_dec", "diff_init_sols", "ablations", f"{name}.pkl"), "rb") as f:
            data = pickle.load(f)
        
        for method in init_sols:
            _suc_ratio = np.zeros_like(X)
            for _j, x in enumerate(data.keys()):
                taus = np.array(data[x][0][method])
                succeeded = [_i for _i, tau in enumerate(taus) if tau != float("inf")]
                _suc_ratio[_j] = len(succeeded)/len(X)
                if succeeded:
                    success_inds[method][x] = succeeded
            
            # plot success ratio bars
            _ax = ax.twinx()
            _ax.bar(X+bar_X_offset[method], height=_suc_ratio, width=bar_width, color=f'{colors[method]}')
            _ax.set_ylim(0, 5)
            _ax.set_yticks([1.0])
            _ax.set_yticklabels([f"{_val:.0%}" for _val in [1.0]], fontsize=axis_fontsize, rotation=-90)
            _ax.tick_params(axis='y', which='major', pad=0)
            for label in _ax.get_yticklabels():
                label.set_verticalalignment('center')

        _mean, _std = defaultdict(lambda: np.zeros_like(X)), defaultdict(lambda: np.zeros_like(X))
        for _j, x in enumerate(data.keys()):
            in_common_inds = list(set(success_inds["VOR"][x]) & set(success_inds["MFC"][x]) & \
                set(success_inds["MSTC*"][x]) & set(success_inds["LS"][x]) & set(success_inds["+VO"][x]))
            for method in init_sols:
                _mean[method][_j] = np.mean([data[x][0][method][_i] for _i in in_common_inds])
                _std[method][_j]  =  np.std([data[x][0][method][_i] for _i in in_common_inds])

        for method in init_sols:
            ax.plot(X, _mean[method], f'{lines[method]}{colors[method]}', label=init_sols[method])
            ax.fill_between(X, _mean[method]-_std[method], _mean[method]+_std[method], alpha=0.2)
            ax.legend(fancybox=True, framealpha=0.5, loc='upper left', prop={'size': legend_fontsize}, labelspacing=0.0, borderpad=0)

        # time
        _mean, _std = defaultdict(lambda: np.zeros_like(X)), defaultdict(lambda: np.zeros_like(X))
        for _j, x in enumerate(data.keys()):
            for method in init_sols:
                runtime = [data[x][1][method][_i] for _i in success_inds[method][x]]
                _mean[method][_j] = np.mean(runtime)
                _std[method][_j]  = np.std(runtime)

        _ax = axes[2*i+1]
        for method in init_sols:
            _ax.plot(X, _mean[method], f'{lines[method]}{colors[method]}', label=init_sols[method])
            _ax.fill_between(X, _mean[method]-_std[method], _mean[method]+_std[method], alpha=0.2)
            _ax.legend(fancybox=True, framealpha=0.5, loc='upper left', prop={'size': legend_fontsize}, labelspacing=0.0, borderpad=0, columnspacing=0.8)

        # set x-axis
        x_labels = np.linspace(0, 0.2, 3)
        ax.set_xticks(x_labels)
        ax.set_xticklabels([f"{_val:.0%}" for _val in x_labels], fontsize=axis_fontsize)
        
        _ax.set_xticks(x_labels)
        _ax.set_xticklabels([f"{_val:.0%}" for _val in x_labels], fontsize=axis_fontsize)

        # set y-axis
        MIN, MAX = ax.get_ylim()
        y_labels = np.linspace(1.4*MIN, MAX*0.85, 3)
        ax.set_ylim(MIN-(MAX-MIN)*0.2, MAX)
        ax.set_yticks(y_labels)
        ax.set_yticklabels([f"{_val:.1f}" for _val in y_labels], fontsize=axis_fontsize, rotation=90)
        ax.tick_params(axis='y', which='major', pad=0)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')
        
        MIN, MAX = _ax.get_ylim()
        y_labels = np.linspace(0, MAX-0.3*(MAX-MIN), 3)
        _ax.set_ylim(0, 0.8*MAX)
        _ax.set_yticks(y_labels)
        _ax.set_yticklabels([f"{_val:.1f}s" for _val in y_labels], fontsize=axis_fontsize, rotation=90)
        _ax.tick_params(axis='y', which='major', pad=0)
        for label in _ax.get_yticklabels():
            label.set_verticalalignment('center')

        ax.set_title(f"{name} (makespan)", fontsize=title_fontsize)
        _ax.set_title(f"{name} (runtime)", fontsize=title_fontsize)


    widths = [3.5, 3.5, 3.5, 3.5]
    heights = [2]
    title_fontsize = 10.5
    legend_fontsize = 8
    axis_fontsize = 8
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, axes = plt.subplots(ncols=4, nrows=1, constrained_layout=True, gridspec_kw=gs_kw, figsize=(7.5, 2))
    _plot_single("floor_large", 0, axes)
    _plot_single("terrain_large", 1, axes)
    plt.savefig("diff_sols.pdf", dpi=300)


def plot_diff_iters():

    def _plot_single(name, i, axes):
        ax = axes[2*i]
        num_iters = {1.0: "1.0e3", 1.5:"1.5e3", 2.0:"2.0e3", 2.5:"2.5e3", 3.0:"3.0e3"}
        colors = {1.0: "r", 1.5:"g", 2.0:"b", 2.5:"k", 3.0:"c"}
        lines = {1.0:"-", 1.5:'--', 2.0:'-.', 2.5:'-', 3.0:':'}
        X = np.linspace(0, 0.2, 12, endpoint=True)
        with open(os.path.join("data", "mcpp", "diff_iters", f"{name}.pkl"), "rb") as f:
            data = pickle.load(f)
        
        # makespan
        _mean, _std = defaultdict(lambda: np.zeros_like(X)), defaultdict(lambda: np.zeros_like(X))
        for _j, x in enumerate(data.keys()):
            for method in num_iters:
                _mean[method][_j] = np.mean(data[x][0][method])
                _std[method][_j]  =  np.std(data[x][0][method])

        for method in num_iters:
            ax.plot(X, _mean[method], f'{lines[method]}{colors[method]}', label=num_iters[method])
            ax.fill_between(X, _mean[method]-_std[method], _mean[method]+_std[method], alpha=0.2)
            ax.legend(fancybox=True, framealpha=0.5, loc='upper left', prop={'size': legend_fontsize}, labelspacing=0.0, borderpad=0, ncol=2, columnspacing=0.8)
                
        # time
        _mean, _std = defaultdict(lambda: np.zeros_like(X)), defaultdict(lambda: np.zeros_like(X))
        for _j, x in enumerate(data.keys()):
            for method in num_iters:
                _mean[method][_j] = np.mean(data[x][1][method]) /60
                _std[method][_j]  =  np.std(data[x][1][method])/60

        _ax = axes[2*i+1]
        for method in num_iters:
            _ax.plot(X, _mean[method], f'{lines[method]}{colors[method]}', label=num_iters[method])
            _ax.fill_between(X, _mean[method]-_std[method], _mean[method]+_std[method], alpha=0.2)
            _ax.legend(fancybox=True, framealpha=0.5, loc='upper left', prop={'size': legend_fontsize}, labelspacing=0.0, borderpad=0, ncol=2, columnspacing=0.8)

        # set x-axis
        x_labels = np.linspace(0, 0.2, 3)
        ax.set_xticks(x_labels)
        ax.set_xticklabels([f"{_val:.0%}" for _val in x_labels], fontsize=axis_fontsize)
        
        _ax.set_xticks(x_labels)
        _ax.set_xticklabels([f"{_val:.0%}" for _val in x_labels], fontsize=axis_fontsize)

        # set y-axis
        MIN, MAX = ax.get_ylim()
        y_labels = np.linspace(MIN+0.1*(MAX-MIN), MAX-0.1*(MAX-MIN), 3)
        ax.set_ylim(MIN, MAX)
        ax.set_yticks(y_labels)
        ax.set_yticklabels([f"{_val:.1f}" for _val in y_labels], fontsize=axis_fontsize, rotation=90)
        ax.tick_params(axis='y', which='major', pad=0)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')
        
        MIN, MAX = _ax.get_ylim()
        y_labels = np.linspace(MIN+0.1*(MAX-MIN), MAX-0.1*(MAX-MIN), 3)
        _ax.set_ylim(MIN, 1.15*MAX)
        _ax.set_yticks(y_labels)
        _ax.set_yticklabels([f"{_val:.1f}m" for _val in y_labels], fontsize=axis_fontsize, rotation=90)
        _ax.tick_params(axis='y', which='major', pad=0)
        for label in _ax.get_yticklabels():
            label.set_verticalalignment('center')

        ax.set_title(f"{name} (makespan)", fontsize=title_fontsize)
        _ax.set_title(f"{name} (runtime)", fontsize=title_fontsize)

    widths = [3.5, 3.5, 3.5, 3.5]
    heights = [2]
    title_fontsize = 11
    legend_fontsize = 8
    axis_fontsize = 8
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, axes = plt.subplots(ncols=4, nrows=1, constrained_layout=True, gridspec_kw=gs_kw, figsize=(7.5, 2))
    _plot_single("ht_chantry", 0, axes)
    _plot_single("ost002d", 1, axes)
    plt.savefig("diff_iters.pdf", dpi=300)


def plot_NewYork1():
    name, rmv_ratio, seed = "NewYork1", 0.000, 0
    mcpp = MCPP.read_instance(os.path.join("benchmark", "instances", f"{name}.mcpp"))
    mutant = list(mcpp.randomized_mutants([rmv_ratio], [seed]))[0]

    with open(os.path.join("data", "mcpp", "solutions", mcpp.name, "VOR", f"{rmv_ratio:.3f}-{seed}.pkl"), 'rb') as f:
        VOR_pi = pickle.load(f)

    with open(os.path.join("data", "mcpp", "solutions", mcpp.name, "+VO", f"{rmv_ratio:.3f}-{seed}.pkl"), 'rb') as f:
        pi = pickle.load(f)
        G = nx.DiGraph()
        G.add_nodes_from(mutant.I)
        init_plans = []
        init_planner = ChainingApproach(mutant, HeurType.TrueDist)
        for i in mutant.I:
            pi.Pi[i] = [v for v in pi.Pi[i] if mutant.pos2v(v) == mutant.R[i] or mutant.pos2v(v) not in mutant.R]
            status, P, num_explored = init_planner.plan(pi.Pi[i], ReservationTable(), float('inf'))
            if len(P) == 0:
                P.states = [State(mutant.R[i], 0), State(mutant.R[i], float('inf'))]
            init_plans.append(P)
            Helper.verbose_print(f"PBS: Initialized Plan for robot {i}, # of explored nodes={num_explored}")

        root = Node(G, init_plans, True)
        # print(root.num_of_conflicts)
        # print((VOR_pi.tau - pi.tau)/VOR_pi.tau)

    with open(os.path.join("data", "mcpp_dec", "solutions", mcpp.name, "adaptive", f"{rmv_ratio:.3f}-{seed}.pkl"), 'rb') as f:
        sol = pickle.load(f)
    
    coord_x = lambda x: (np.array(x)+0.5)*2 + 0.5
    coord_y = lambda x: 257 - (np.array(x)+0.5)*2 - 0.5
    
    def subplot(ax, idx):
        cmap = colormap("tab20")
        I = np.zeros((mutant.width+2, mutant.height+2))
        for v in mutant.G.nodes:
            x, y = mutant.G.nodes[v]["pos"]
            I[x+1, mutant.height-y] = 1
        r = {}
        for i, P in enumerate(sol):
            pi = [X.pos for X in P[:idx]]
            xs, ys = zip(*pi)
            r[i] = (xs[-1], ys[-1])
            color = rgb2hex(cmap(i/mutant.k))
            ax.plot(coord_y(ys), coord_x(xs), f"-", color=color, alpha=0.5, lw=1.4)
        
        for i, P in enumerate(sol):
            color = rgb2hex(cmap(i/mutant.k))
            rx, ry = coord_x(r[i][0]), coord_y(r[i][1])
            ax.plot(ry, rx, f"o", color=color, ms=8)
            ax.text(ry, rx, i, va='center', ha='center', c='w', font={'size': 8})

        ax.imshow(I, cmap="gray")
        ax.axis("off")

    widths = [4, 4]
    heights = [4]
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True, gridspec_kw=gs_kw, figsize=(8, 4))
    L = max([len(P) for P in sol])
    subplot(axes[0], int(L//3))
    subplot(axes[1], int(L))
    plt.savefig("example.png", bbox_inches="tight", dpi=600)


def record_sim():
    name,rmv_ratio, seed = "Shanghai2", 0.036, 8
    scale, dt, size = 0.15, 0.6, 256

    # name, rmv_ratio, seed = "AR0205SR", 0.05454, 0
    # scale, dt, size = 0.18, 0.5, 220

    # name, rmv_ratio, seed = "terrain_small", 0.12727, 0
    # scale, dt, size = 3.0, 0.12, 20

    # name, rmv_ratio, seed = "ost002d", 0.12727, 0 # ht_chantry
    # scale, dt, size = 0.3, 0.6, 150

    # name, rmv_ratio, seed = "terrain_large", 0.091, 7 # ht_chantry
    # scale, dt, size = 0.85, 0.3, 64
    
    # name, rmv_ratio, seed = "terrain_medium", 0, 3 # ht_chantry
    # scale, dt, size = 1.5, 0.2, 40    # floor_medium

    mcpp = MCPP.read_instance(os.path.join("benchmark", "instances", f"{name}.mcpp"))
    mutant = list(mcpp.randomized_mutants([rmv_ratio], [seed]))[0]

    def conflict_sim(method):
        with open(os.path.join("data", "mcpp", "solutions", mcpp.name, method, f"{rmv_ratio:.3f}-{seed}.pkl"), 'rb') as f:
            sol = pickle.load(f)
            
        G = nx.DiGraph()
        G.add_nodes_from(mutant.I)
        plans = []
        init_planner = ChainingApproach(mutant, HeurType.TrueDist)
        for i in mutant.I:
            sol.Pi[i] = [v for v in sol.Pi[i] if mutant.pos2v(v) == mutant.R[i] or mutant.pos2v(v) not in mutant.R]
            status, P, num_explored = init_planner.plan(sol.Pi[i], ReservationTable(), float('inf'))
            if len(P) == 0:
                P.states = [State(mutant.R[i], 0), State(mutant.R[i], float('inf'))]
            plans.append(P)
            Helper.verbose_print(f"PBS: Initialized Plan for robot {i}, # of explored nodes={num_explored}")
        root = Node(G, plans, True)
        simulate(mutant, plans, scale, dt, method)
        # print(sol.tau)
        # print(root.num_of_conflicts)

    def conflict_sim_fix(method):
        with open(os.path.join("data", "mcpp", "solutions", mcpp.name, method, f"{rmv_ratio:.3f}-{seed}.pkl"), 'rb') as f:
            sol = pickle.load(f)
        
        G = nx.DiGraph()
        G.add_nodes_from(mutant.I)
        plans = []
        init_planner = ChainingApproach(mutant, HeurType.TrueDist)
        for i in mutant.I:
            sol.Pi[i] = [v for v in sol.Pi[i] if mutant.pos2v(v) == mutant.R[i] or mutant.pos2v(v) not in mutant.R]
            sol.Pi[i] = ExtSTCPlanner.root_align(sol.Pi[i], mutant.legacy_vertex(mutant.R[i]))
            status, P, num_explored = init_planner.plan(sol.Pi[i], ReservationTable(), float('inf'))
            if len(P) == 0:
                P.states = [State(mutant.R[i], 0), State(mutant.R[i], float('inf'))]
            for X in P:
                X.pos = (X.pos//size, X.pos%size)
                X.pos = (X.pos[0]/2 - 0.25, X.pos[1]/2 - 0.25)
            plans.append(P)
            Helper.verbose_print(f"PBS: Initialized Plan for robot {i}, # of explored nodes={num_explored}")
        root = Node(G, plans, True)
        simulate(mutant, plans, scale, dt, method)
        # print(sol.tau)
        # print(root.num_of_conflicts)

    def deconf_sim():
        with open(os.path.join("data", "mcpp_dec", "solutions", mcpp.name, "adaptive", f"{rmv_ratio:.3f}-{seed}.pkl"), "rb") as f:
            plans = pickle.load(f)
            # print(max([P[-1].time for P in plans]))
        simulate(mutant, plans, scale, dt, "Adaptive")

    # conflict_sim_fix("MIP")
    deconf_sim()


if __name__ == "__main__":
    # plot_ablations()
    # plot_diff_sol()
    # plot_diff_iters()
    # plot_NewYork1()
    record_sim()
