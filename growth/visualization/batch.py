from math import floor
import numpy as np
import matplotlib.pyplot as plt


class BatchVisualization:
    """
    Visualization methods for Sweep object.
    """

    @staticmethod
    def ordinal(n):
        """ Returns ordinal representation of <n>. """
        return "%d%s" % (n,"tsnrhtdd"[(floor(n/10)%10!=1)*(n%10<4)*n%10::4])

    def plot_culture_grid(self, ncols=5, size=3, title=False, **kwargs):
        """
        Plots grid of cell cultures.

        Args:

            ncols (int) - number of columns

            size (int) - figure panel width

            title (bool) - if True, add title

        Returns:

            fig (matplotlib.figures.Figure)

        """

        # determine figure shape
        nrows = self.size // ncols
        if self.size % ncols != 0:
            nrows += 1

        # create figure
        figsize = (nrows*size, ncols*size)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        for replicate_id, ax in enumerate(axes.ravel()):

            # load simulation
            sim = self[replicate_id]

            # plot culture
            sim.plot(ax=ax, **kwargs)

            # format axis
            if title:
                title = '{:s} replicate'.format(self.ordinal(replicate_id))
                ax.set_title(title, fontsize=8)
            ax.axis('off')

        return fig
