import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import time

import networkx as nx
import matplotlib.pyplot as plt

from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

from src.data_generater.spring import SpringSim


if __name__ == "__main__":
    model = SpringSim(n_balls=10)
    loc, vel, adj = model.sample_trajectory()
    fig_mpl, ax = plt.subplots(1,figsize=(10,6), facecolor='white')

    FPS = 120

    def draw(G, one_sec_loc):
        count = one_sec_loc.shape[0]
        delta_alpha = 0.5/count
        for cnt, one_loc in enumerate(one_sec_loc[:-1]):
            pos = {node:position for node, position in zip(G.nodes(), one_loc)}
            nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_size=30, alpha=delta_alpha*cnt)

        final_loc = one_sec_loc[-1]
        pos = {node:position for node, position in zip(G.nodes(), final_loc)}
        nx.draw(G, pos=pos, ax=ax, node_size=30, width=0.5, alpha=0.5)
        ax.set_axis_on()
        ax.set_xlim([-5,5])
        ax.set_ylim([-5,5])
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        
    def make_frame_mpl(t):
        ax.clear()
        t = int(t*FPS)+1
        one_sec_loc = loc[(t-FPS):(t+1)].reshape(-1, 10, 2) if t>FPS else loc[:t].reshape(-1, 10, 2)
        draw(G, one_sec_loc)
        return mplfig_to_npimage(fig_mpl) # 图形的RGB图像

    G = nx.Graph()
    G = nx.from_numpy_array(adj)
    animation =mpy.VideoClip(make_frame_mpl, duration=5)
    animation.write_gif("../gifs/spring.gif", fps=FPS)
