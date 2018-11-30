import numpy as np
import matplotlib.pyplot as plt


class SweepResults:
    """
    Container for managing aggregated results of a parameter sweep.
    """

    def __init__(self, data, shape):

        # evaluate mean across all replicates
        gb = data.groupby(['row_id', 'column_id'])
        self.df = gb.agg(lambda x: np.mean(x[x!=np.inf])).reset_index()
        self.shape = shape

    @property
    def xx(self):
        """ Grid of X values (recombination start generations). """
        return self.to_grid('start_time')

    @property
    def yy(self):
        """ Grid of Y values (recombination rates). """
        return self.to_grid('recombination_rate')

    def to_grid(self, attribute):
        """ Returns 2D grid of <attribute> values. """
        return self.df[attribute].values.reshape(self.shape)

    def plot(self, attribute, log=False, complement=False, **kwargs):
        """
        Plot 2D grid of <attribute>.
        """

        # get attribute grid
        zz = self.to_grid(attribute)

        # apply transformations
        if complement:
            zz = 1 - zz
        if log:
            zz = np.log10(zz)

        fig, ax = self._plot(zz, **kwargs)
        ax.set_title(attribute)
        return fig

    def _plot(self, zz, figsize=(2, 2), cmap=plt.cm.viridis):
        """
        Plot 2D grid.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(zz, cmap=cmap)
        self.format_axis(ax)
        return fig, ax

    def format_axis(self, ax):
        _ = ax.set_xticks(np.arange(self.shape[0]))
        _ = ax.set_yticks(np.arange(self.shape[1]))
        ax.invert_yaxis()
        ax.set_xlabel('Start Generation')
        ax.set_ylabel('Recombination Rate')
        xticklabels = ['{:d}'.format(int(l)) for l in self.xx[0]]+['']
        _ = ax.set_xticklabels(xticklabels, rotation=0)
        yticklabels = ['{:.2f}'.format(l) for l in self.yy[:,0]]
        _ = ax.set_yticklabels(yticklabels)
        ax.tick_params(labelsize=8, length=3, pad=3)

    @property
    def recombination_extent(self):
        """ Extent of recombination. """
        return self.plot('percent_heterozygous', complement=True)

    @property
    def num_clones(self):
        """ Number of clones. """
        return self.plot('num_clones')

    @property
    def clone_size(self):
        """ Mean clone size (log-transformed). """
        return self.plot('mean_clone_size', log=True)

    @property
    def transclone_edges(self):
        """ Number of transclone edges. """
        return self.plot('transclone_edges')

    @property
    def size_variation(self):
        """ Coefficient of variation for clone size. """
        return self.plot('clone_size_variation')
