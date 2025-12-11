# %%
import plotly.express as px
import matplotlib.pyplot as plt
from jan.py.oct.config import BORDER_COLORS, POSTERIOR_BORDER_ORDER
import numpy as np

# %% 

def plotHW(tensor, ax=None, interactive=False, legend=False):
    if interactive:
        fig = px.imshow(
            tensor, color_continuous_scale="viridis", origin="upper")
        fig.show()
    else:
        ax = ax if ax else plt.gca() 
        arr = tensor.numpy()
        im = ax.imshow(arr, vmin=0, vmax=1, cmap="Greys_r")
        ax.set_xticks([])
        ax.set_yticks([])
        # im = ax.imshow(arr, vmin=0, vmax=1, cmap="Greys")

        if legend: plt.colorbar(im, ax=ax)
        return ax   # return Axes, not AxesImage

from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, to_rgba

def alpha_cmap(color, n=256):
    """Colormap where RGB stays constant, alpha increases with value."""
    rgba = np.ones((n, 4))
    rgb = np.array(to_rgba(color))[:3]
    rgba[:, :3] = rgb
    rgba[:, 3] = np.linspace(0, 0.8, n)  # transparency gradient
    return ListedColormap(rgba)

def plotHWC(tensor, ax=None, threshold=1e-6, cmaps=None, labels=None, legend=True, **imshow_kwargs):
    ax = ax if ax else plt.gca() 
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    cmaps = [alpha_cmap(c) for c in (cmaps or colors)]
    arr = tensor.numpy()
    for c in range(arr.shape[-1]):
        masked = np.ma.masked_where(arr[..., c] < threshold, arr[..., c])
        ax.imshow(masked, cmap=plt.get_cmap(cmaps[c % len(cmaps)]), **imshow_kwargs)
        
    if labels and legend:
        handles = [
            Patch(color=to_rgba(colors[c % len(colors)], 0.8), label=labels[c])
            for c in range(len(labels))
        ]
        ax.add_artist(
            ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), borderaxespad=0,  loc="upper left"))
        
    return ax

def plotCW(cw_tensor, ax=None, c_names=None, step: int = 1, legend=False):
    """
    Overlay border curves on an existing matplotlib axis.

    Args:
        ax: matplotlib axis object (from plotHW).
        cw_tensor: tensor with shape (C, W), one row per border.
        step: downsampling step along width (plot every nth point).
    """
    ax = ax if ax else plt.gca() 
    
    borders_np = cw_tensor.numpy()
    n_borders, width = borders_np.shape
    
    border_names = c_names if c_names else POSTERIOR_BORDER_ORDER[:n_borders]

    # x = np.arange(width)[::step]
    for i, name in enumerate(border_names):
        x = np.arange(width)[::step]
        y = borders_np[i, ::step]
        if x[-1] != width - 1:  # make sure we include the very last point
            x = np.append(x, width - 1)
            y = np.append(y, borders_np[i, -1])

        y = np.ma.masked_where(y == 0, y)  # skip zero regions
        color = BORDER_COLORS.get(name, None)
        ax.plot(x, y, color=color, linewidth=1.0, label=name)

    if legend:
        leg = ax.legend(bbox_to_anchor=(1.02, 0), ncol=3, borderaxespad=0,  loc="lower left")
        for line in leg.get_lines():
            line.set_linewidth(3)
        ax.add_artist(leg)

    return ax