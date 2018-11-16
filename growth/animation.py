import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize



class CloneAnimation:

    def __init__(self, population):
        self.population = population

        self.cmap = plt.cm.viridis
        self.norm = Normalize(0, 2)

    def scatter_points(self, xy, z, lw=0, s=30, **kwargs):
        c = self.cmap(self.norm(z))
        self.ax.scatter(*xy.T, c=c, s=s, lw=lw, **kwargs)

    def update(self, frame, **kwargs):
        if len(self.ax.collections) > 0:
            self.ax.collections.pop()
        self.scatter_points(frame['xy'], frame['genotypes'], **kwargs)

    def animate(self,
                framerate=10,
                figsize=(5, 5),
                xlim=(-1.2, 1.2),
                ylim=(-1.2, 1.2),
                endframe_repeats=0,
                **kwargs):
        """
        Generate animation by sequentially updating plot elements with voronoi region history.

        Args:

            framerate (float) - defines animation speed in Hz

            figsize (tuple) - figure size

            xlim, ylim (tuple) - axis range

            endframe_repeats (int) - number of times to repeat final frame

            kwargs: element formatting arguments

        Returns:

            animation (matplotlib.FuncAnimation)

        """

        # create and format figure
        fig = plt.figure(figsize=figsize)
        self.ax = plt.axes(xlim=xlim, ylim=ylim)
        self.ax.set_aspect(1)
        self.ax.axis('off')

        # initialize plot elements
        self.update(self.population.history[0], **kwargs)

        # get frames
        frames = self.population.history
        frames += ([frames[-1]] * endframe_repeats)

        # generate animation
        animation = FuncAnimation(fig,
                           func=self.update,
                           frames=frames,
                           interval=1e3/framerate,
                           blit=False,
                           fargs=(kwargs))

        return animation

    def get_video(self, **kwargs):
        return self.animate(**kwargs).to_html5_video()
