import os
from itertools import product

import LS_MCPP

import matplotlib
from matplotlib import animation
from MIP_MCPP.instance import Instance
from MIP_MCPP.misc import colormap
from MSTC_Star.utils.robot import Robot

import matplotlib.pyplot as plt

from LS_MCPP.local_search import *
from LS_MCPP.solution import *
from LS_MCPP.utils import *


def simulate(
    name: str,
    G: nx.Graph,
    D: nx.Graph,
    R: list,
    width: int,
    height:int,
    paths: list,
    weights: list,
    scale: float,
    dt: float,
    is_write: bool = False,
    is_show: bool = False
) -> None:

    k = len(R)
    color = ['r', 'm', 'b', 'k', 'c', 'g']
    fig = plt.figure()
    fig.set_size_inches(8*width/height, 8)
    fig.tight_layout()
    ax = plt.axes()
    ax.set_xlim(-1, width)
    ax.set_ylim(-1, height)
    plt.grid(True, linestyle='--')
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    plt.waitforbuttonpress()
    plt.pause(5)
    robots = [Robot(paths[i]+[R[i]], D) for i in range(k)]
    t_finish = [robots[i].T[-1] for i in range(k)]
    t_max = max(t_finish)

    if not is_write and not is_show:
        print(f'Final Max Weights: {max(weights)}')
        return

    lines, markers, texts = [None]*k, [None]*k, [None]*(k+1)
    xs_vec, ys_vec = [None]*k, [None]*k

    def init():
        plt.title(f'makespan={max(weights): .2f}',
                  fontdict={'size': 20*scale})
        texts[-1] = ax.text(
            1, 1, '', va='top', ha='right', transform=ax.transAxes,
            font={'size': 8*scale})
        
        # obstacle graph
        for p in product(range(width), range(height)):
            for dv in DecGraph.decomp(p):
                if dv not in D.nodes:
                    ax.scatter(dv[0], dv[1], marker='s', s=120*scale, color='k')

        for i in range(k):
            c = color[i % len(color)]
            line, = ax.plot([], [], '-'+c, alpha=0.3, lw=16*scale)
            marker, = ax.plot([], [], 'o'+c, ms=8)
            # changable texts
            texts[i] = ax.text(
                1, 0.975-i*0.025, '', va='top', ha='right',
                transform=ax.transAxes, font={'size': 8*scale})
            # trajectories and robots
            lines[i], markers[i] = line, marker
            xs_vec[i], ys_vec[i] = zip(*paths[i])
            # depots
            ax.plot(R[i][0], R[i][1], '*k', mfc=c, ms=20*scale)
            # ax.text(R[i][0]+0.1, R[i][1]+0.1, f'R{i}')

        return lines + markers + texts

    # # record remaining uncovered nodes
    # uncovered = set()
    # direction = ['SE', 'NE', 'NW', 'SW']
    # for node in planner.G.nodes:
    #     for sn in [planner.__get_subnode_coords__(node, d) for d in direction]:
    #         uncovered.add(sn)

    def animate(ti):
        ts = ti * dt
        for i in range(k):
            last_coord_idx, cur_state = robots[i].get_cur_state(ts)
            xs = xs_vec[i][:last_coord_idx+1] + (cur_state.x, )
            ys = ys_vec[i][:last_coord_idx+1] + (cur_state.y, )
            # texts[i].set_text(f'R{i}: ')
            lines[i].set_data(xs, ys)
            markers[i].set_data(cur_state.x, cur_state.y)
            # node = (xs_vec[i][last_coord_idx], ys_vec[i][last_coord_idx])
        #     if node in uncovered:
        #         uncovered.remove(node)
        # texts[-1].set_text(f'T[s]={ts: .2f}, # of uncovered={len(uncovered)}')

        return lines + markers + texts

    anim = matplotlib.animation.FuncAnimation(
        fig, animate, int(t_max/dt), init, interval=1,
        blit=True, repeat=False, cache_frame_data=False)

    if is_write:
        FFwriter=animation.FFMpegWriter(fps=900, extra_args=['-vcodec', 'libx264'])
        anim.save(f'data/simrecords/{name}.mp4', writer=FFwriter, dpi=200,
                  progress_callback=lambda i, n: print(f'{name}: saving frame {i}/{n}'))

    if is_show:
        plt.show()
        plt.close()


def rec_sim(istc_name, method, width, height, scale, incomplete=False, write=True):
    # istc = Instance.read(istc_name+".istc", os.path.join("data", "instances"))
    istc = Instance.read(istc_name+".istc", os.path.join("MIP-MCPP", "data", "instances"))

    if incomplete:
        dg = DecGraph.randomly_remove_Vd(istc, 0.2, 0, True)
        R = [istc.G.nodes[r]["pos"] for r in istc.R]
        init_sol = incomplete_G_sol(dg, R)
        planner = LocalSearchMCPP(dg, init_sol, PrioType.Heur, verbose=False, R=R)
        name=f"{istc_name}-incomplete-{method}"
    else:
        init_sol = MFC_sol(istc)
        planner = LocalSearchMCPP(istc, init_sol, PrioType.Heur, verbose=False)
        name=f"{istc_name}-{method}"
    
    if method == "LS-MCPP":
        M, S, T_final, gamma = 3e3, 1e2, 0.2, 1e-2
        sol_opt, rt = planner.run(
            M=M,
            S=S,
            alpha=np.exp(np.log(T_final) / M),
            gamma=gamma,
            sample_type=SampleType.RouletteWheel
        )
        simulate(
            name=name,
            G=planner.T,
            D=planner.D,
            R=planner.R_D,
            width=width,
            height=height,
            paths=sol_opt.Pi,
            weights=sol_opt.costs,
            scale=scale,
            dt=0.02,
            is_write=write,
            is_show=not write,
        )
    else:
        simulate(
            name=name,
            G=planner.T,
            D=planner.D,
            R=planner.R_D,
            width=width,
            height=height,
            paths=init_sol.Pi,
            weights=init_sol.costs,
            scale=scale,
            dt=0.02,
            is_write=write,
            is_show=not write,
        )


if __name__ == "__main__":
    rec_sim("terrain_large_1-32x32-k4", "LS-MCPP", 32, 32, 0.35, incomplete=True, write=False)
