import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from .animation import Animation


class CultureVisualization:

    def animate(self, interval=500, **kwargs):
        """ Returns animation of culture growth. """
        freeze = np.vectorize(self.freeze)
        frames = freeze(np.arange(self.generation+1))
        animation = Animation(frames)
        video = animation.get_video(interval=interval, **kwargs)
        return video

    def plot(self,
             ax=None,
             colorby='genotype',
             tri=False,
             s=2,
             cmap=plt.cm.viridis):
        """
        Scatter cells in space.

        """

        # evaluate marker colors
        if colorby == 'genotype':
            norm = Normalize(0, 2)
            c = cmap(norm(self.genotypes))
        elif colorby == 'lineage':
            norm = Normalize(-1, 1)
            c = cmap(norm(self.diversification))

        # create and format figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.axis('off')

        # add triangulation
        if tri:
            ax.triplot(self.triangulation, 'r-', lw=1, alpha=1, zorder=0)

        # scatter points
        ax.scatter(*self.xy.T, s=s, lw=0, c=c)
        ax.set_aspect(1)
