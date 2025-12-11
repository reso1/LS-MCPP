from __future__ import annotations
from typing import List, Tuple

import os
import math
import bisect

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

from lsmcpp.benchmark.instance import MCPP
from lsmcpp.benchmark.plan import State, Plan, Heading


class Robot:
    # assume 1) 1st-order dynamics and 2) the time equals the cost
    def __init__(self, path:Plan):
        self.state = path.states[0]
        self.pi = path
        self.T = np.array([X.time for X in path.states])

    def get_cur_state(self, ts) -> Tuple[int, State]:
        if ts >= self.T[-1]:
            return len(self.pi)-1, self.state
        
        ti = bisect.bisect(self.T, ts) - 1
        dx = self.pi.states[ti+1].pos[0] - self.pi.states[ti].pos[0]
        dy = self.pi.states[ti+1].pos[1] - self.pi.states[ti].pos[1]
        if self.pi.states[ti+1].heading == Heading.E and self.pi.states[ti].heading == Heading.S:
            dheading = math.pi/2
        elif self.pi.states[ti+1].heading == Heading.S and self.pi.states[ti].heading == Heading.E:
            dheading = -math.pi/2
        else:
            dheading = self.pi.states[ti+1].heading.radian - self.pi.states[ti].heading.radian
        k = (ts - self.T[ti]) / (self.T[ti+1] - self.T[ti])

        self.state = State(
            pos = (self.pi.states[ti].pos[0] + dx*k, self.pi.states[ti].pos[1] + dy*k),
            time = ts,
            heading = (self.pi.states[ti].heading.radian + dheading*k) % (2*math.pi)
        )
        
        return ti, self.state


def simulate(mcpp: MCPP, Pi: List[Plan], scale:float, dt:float, suffix:str=None):
    # padding the last coordinate to make the robots stop at the end
    for i in mcpp.I:
        Pi[i].states += [State(Pi[i].states[-1].pos, Pi[i].states[-1].time + 3, Pi[i].states[-1].heading)]

    k = mcpp.k
    color = ['r', 'm', 'b', 'k', 'c', 'g']
    fig, ax = plt.subplots()
    fig.tight_layout()
    fig.set_size_inches(8*mcpp.width/mcpp.height, 8)
    ax.set_xlim(-1, mcpp.width)
    ax.set_ylim(-1, mcpp.height)
    ax.grid(True, linestyle='--')
    fig.canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    
    plt.pause(3)

    robots = [Robot(Pi[i]) for i in range(k)]
    t_finish = [robots[i].T[-1] for i in range(k)]
    t_max = max(t_finish)

    lines, markers, arrows, texts = [None]*k, [None]*k, [None]*k, [None]*(k+1)
    xs_vec, ys_vec = [None]*k, [None]*k

    def init():
        # plt.title(f'makespan={solution.tau: .2f}', fontdict={'size': 20*scale})
        texts[-1] = ax.text(1, 1, '', va='top', ha='right', transform=ax.transAxes, font={'size': 8*scale})
        mcpp.draw(ax, scale=scale)
        for i in range(k):
            c = color[i % len(color)]
            line, = ax.plot([], [], '-'+c, alpha=0.3, lw=8*scale)
            marker, = ax.plot([], [], 'o'+c, ms=5) #6*scale)
            lines[i], markers[i] = line, marker
            arrows[i] = ax.arrow(0, 0, 0, 0, head_width=0, head_length=0, fc=c, ec=c)
            xs_vec[i] = [X.pos[0] for X in Pi[i].states]
            ys_vec[i] = [X.pos[1] for X in Pi[i].states]
            r = Pi[i].states[0].pos
            texts[i] = ax.text(r[0], r[1], f'', va='center', ha='center', c='w', font={'size': 5}) #6*scale})
        
        return lines + markers + arrows + texts
    
    def animate(ti):
        for i in range(k):
            last_coord_idx, cur_state = robots[i].get_cur_state(ti * dt)
            xs = xs_vec[i][:last_coord_idx+1] + [cur_state.pos[0]]
            ys = ys_vec[i][:last_coord_idx+1] + [cur_state.pos[1]]
            lines[i].set_data(xs, ys)
            markers[i].set_data(cur_state.pos[0], cur_state.pos[1])
            arrows[i].set_data(x = cur_state.pos[0], 
                               y = cur_state.pos[1], 
                               dx = 0.25*math.cos(cur_state.heading),
                               dy = 0.25*math.sin(cur_state.heading),
                               head_width = 0.15, head_length = 0.15)
            texts[i].set_position((cur_state.pos[0], cur_state.pos[1]))
            texts[i].set_text(f'{i}')
        
        return lines + markers + arrows + texts
    
    anim = matplotlib.animation.FuncAnimation(
        fig, animate, int(t_max/dt)+1, init, interval=1,
        blit=True, repeat=False, cache_frame_data=False)
    
    FFwriter=animation.FFMpegWriter(fps=60, extra_args=['-vcodec', 'libx264'])
    fp = os.path.join(os.getcwd(), 'data', 'videos', f'{mcpp.name}-{suffix}.mp4')
    anim.save(fp, writer=FFwriter, dpi=200,
                progress_callback=lambda i, n: print(f'{mcpp.name}: saving frame {i}/{n}'))

