# %%
import plotly.express as px
import matplotlib.pyplot as plt
from jan.py.oct.config import BORDER_COLORS, POSTERIOR_BORDER_ORDER
import numpy as np

# %% 

def plotHW(tensor, interactive=False, legend=False):
    if interactive:
        fig = px.imshow(
            tensor, color_continuous_scale="viridis", origin="upper")
        fig.show()
    else:
        ax = plt.gca()  # get current axis
        arr = tensor.numpy()
        im = ax.imshow(arr, vmin=0, vmax=1, cmap="viridis")

        if legend: plt.colorbar(im, ax=ax)
        return ax   # return Axes, not AxesImage
    
def plotHWC(tensor, ax=None, threshold=1e-6, cmaps=None, **imshow_kwargs):
    ax = ax if ax else plt.gca() 
    cmaps = cmaps if cmaps else ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges']
    arr = tensor.numpy()
    for c in range(arr.shape[-1]):
        masked = np.ma.masked_where(arr[..., c] < threshold, arr[..., c])
        ax.imshow(masked, cmap=plt.get_cmap(cmaps[c % len(cmaps)]), **imshow_kwargs)


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