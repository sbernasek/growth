import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize


class Animation:

    def __init__(self, frames):
        self.frames = frames
        self.cmap = plt.cm.viridis
        self.norm = Normalize(0, 2)

    # def scatter_points(self, xy, z, lw=0, s=30, **kwargs):
    #     c = self.cmap(self.norm(z))
    #     self.ax.scatter(*xy.T, c=c, s=s, lw=lw, **kwargs)

    def update(self, culture, kwargs={}):
        if len(self.ax.collections) > 0:
            self.ax.collections.pop()
        _ = culture.plot(ax=self.ax, **kwargs)

    def animate(self,
                figsize=(5, 5),
                xlim=(-1.2, 1.2),
                ylim=(-1.2, 1.2),
                interval=500,
                repeat_delay=2500,
                **kwargs):
        """
        Generate animation by sequentially updating plot elements with voronoi region history.

        Args:

            figsize (tuple) - figure size

            xlim, ylim (tuple) - axis range

            interval (float) - interval between frames (milliseconds)

            repeat_delay (float) - interval before repeat (milliseconds)

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
        self.update(self.frames[0], kwargs)

        # generate animation
        animation = FuncAnimation(fig,
                           func=self.update,
                           frames=self.frames,
                           interval=interval,
                           blit=False,
                           fargs=(kwargs,),
                           repeat_delay=repeat_delay)

        return animation

    def get_video(self, **kwargs):
        return self.animate(**kwargs).to_html5_video()
