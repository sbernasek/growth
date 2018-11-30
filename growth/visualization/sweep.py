from math import floor
import numpy as np
import matplotlib.pyplot as plt


class SweepVisualization:
    """
    Visualization methods for Sweep object.
    """

    @staticmethod
    def ordinal(n):
        """ Returns ordinal representation of <n>. """
        return "%d%s" % (n,"tsnrhtdd"[(floor(n/10)%10!=1)*(n%10<4)*n%10::4])

    def plot_culture_grid(self,
        replicate_id=0,
        figsize=(25, 25),
        title=False,
        **kwargs):
        """
        Plots grid of cell cultures.

        Args:

            replicate_id (int) - replicate shown

            figsize (tuple) - figure size

            title (bool) - if True, include title

        Returns:

            fig (matplotlib.figures.Figure)

        """

        # create figure
        nrows, ncols = self.shape
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        for index, batch in enumerate(self.batches.ravel()):

            # load simulation
            sim = batch[replicate_id]

            # get row/column indices
            row_id, column_id = index // ncols, index % ncols

            # get axis
            ax = axes.flatten()[index]

            # plot culture
            sim.plot(ax=ax, **kwargs)

            # format axis
            if title:
                gen = self.ordinal(int(sim.recombination_start))
                rate = sim.recombination
                title = 'Started {:s} generation\n'.format(gen)
                title += 'Recombination rate: {:0.2f}'.format(rate)
                ax.set_title(title, fontsize=8)
            ax.axis('off')

        return fig
