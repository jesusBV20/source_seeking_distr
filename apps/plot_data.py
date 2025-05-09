import matplotlib.pyplot as plt

from ssl_simulator import add_src_to_path

add_src_to_path(__file__)
from apps import AppPlotData
from sim_core.visualization import PlotterSimDataSI

# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    app = AppPlotData(PlotterSimDataSI(
        None, None, dpi=100, figsize=(18,7), xlims=[-50,80], ylims=[-60,70],
        num_patches=3
    ))
    plt.show()
