import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize


class CloneAnimation:

    def __init__(self, pop):
        self.pop = pop

        self.cmap = plt.cm.viridis
        self.norm = Normalize(0, 2)

    def initialize_elements(self, frame, ax):
        colors = self.cmap(self.norm(frame['genotypes']))
        ax.scatter(*frame['xy'], s=50, lw=0, c=colors)

    def update_elements(self, frame, ax):
        ax.collections.pop()
        colors = self.cmap(self.norm(frame['genotypes']))
        ax.scatter(*frame['xy'], s=50, lw=0, c=colors)

    def animate(self,
                framerate=10,
                figsize=(5, 5),
                **kwargs):
        """
        Generate animation by sequentially updating plot elements with voronoi region history.

        Args:
        framerate (float) - defines animation speed in Hz
        cmap (matplotlib colormap) - colormap used to color patches
        figsize (tuple) - figure size

        kwargs: element formatting arguments

        Returns:
        anim (matplotlib.FuncAnimation)
        """

        # create and format figure
        fig = plt.figure(figsize=figsize)
        xlim, ylim = self.pop.get_boundary_limits()
        ax = plt.axes(xlim=xlim, ylim=ylim)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_aspect(1)

        # initialize plot elements
        self.initialize_elements(self.pop.history[0], ax)

        # generate animation
        anim = FuncAnimation(fig,
                           func=self.update_elements,
                           frames=self.pop.history,
                           interval=1e3/framerate,
                           blit=False,
                           fargs=(ax,))

        return anim

    def get_video(self, **kwargs):
        return self.animate(**kwargs).to_html5_video()
