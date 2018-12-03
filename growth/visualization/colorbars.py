from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize


class ColorBar:
    def __init__(self,
                 figsize=(2, 0.1),
                 cmap=plt.cm.viridis,
                 vlim=(0, 1),
                 orient='horizontal',
                 label=None):
        self.fig, self.ax = self.create_figure(figsize)
        self.cmap = cmap
        self.norm = Normalize(*vlim)
        self.orient = orient
        self.label = label

        # render colorbar
        self.cbar = self.render()

    @staticmethod
    def create_figure(figsize):
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    def render(self):
        cbar = ColorbarBase(self.ax,
                            cmap=self.cmap,
                            norm=self.norm,
                            orientation=self.orient)
        self.fmt(cbar)
        return cbar

    def fmt(self, cbar):

        if self.label is not None:
            cbar.set_label(self.label)

        self.fmt_ticks(cbar)

    def fmt_ticks(self, cbar):
        pass


class ErrorColorbar(ColorBar):

    def __init__(self, **kwargs):
        super().__init__(cmap=plt.cm.seismic, **kwargs)
