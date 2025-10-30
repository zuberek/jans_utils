import plotly.express as px
import matplotlib.pyplot as plt
from py.oct.config import BORDER_COLORS, POSTERIOR_BORDER_ORDER
import numpy as np


def plotHW(tensor, interactive=False, legend=False, multichannel=False):
    if interactive:
        fig = px.imshow(
            tensor, color_continuous_scale="viridis", origin="upper")
        fig.show()
    else:
        ax = plt.gca()  # get current axis
        arr = tensor.numpy()
        if not multichannel:
            im = ax.imshow(arr, vmin=0, vmax=1, cmap="viridis")
        else:
            # plot first channel fully
            im = ax.imshow(arr[..., 0], vmin=0, vmax=1, cmap="Greys")
            # overlay others only where >0
            cmaps = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges']
            for c in range(1, arr.shape[-1]):
                masked = np.ma.masked_where(arr[..., c] < 1e-6, arr[..., c])
                ax.imshow(masked, cmap=plt.get_cmap(cmaps[c % len(cmaps)]), vmin=0, vmax=1, alpha=0.7)

        if legend: plt.colorbar(im, ax=ax)
        return ax   # return Axes, not AxesImage


def plotCW(ax, cw_tensor, step: int = 1, legend=False):
    """
    Overlay border curves on an existing matplotlib axis.

    Args:
        ax: matplotlib axis object (from plotHW).
        cw_tensor: tensor with shape (C, W), one row per border.
        step: downsampling step along width (plot every nth point).
    """
    borders_np = cw_tensor.numpy()
    n_borders, width = borders_np.shape

    # x = np.arange(width)[::step]
    for i, name in enumerate(POSTERIOR_BORDER_ORDER[:n_borders]):
        x = np.arange(width)[::step]
        y = borders_np[i, ::step]
        if x[-1] != width - 1:  # make sure we include the very last point
            x = np.append(x, width - 1)
            y = np.append(y, borders_np[i, -1])

        color = BORDER_COLORS.get(name, None)
        ax.plot(x, y, color=color, linewidth=1.0, label=name)

    if legend: ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    return ax