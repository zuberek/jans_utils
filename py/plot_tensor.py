import plotly.express as px
import matplotlib.pyplot as plt


def plotHW(tensor, interactive=False):
    if interactive:
        fig = px.imshow(
            tensor, color_continuous_scale="viridis", origin="upper")
        fig.show()
    else:
        plt.imshow(tensor.numpy(), vmin=0, vmax=1, cmap="viridis")
        plt.colorbar()
        plt.show()
