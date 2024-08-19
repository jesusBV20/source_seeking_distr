import matplotlib.pyplot as plt
import numpy as np


class Animation:
    def __init__(self):
        self.ax = None
        self.title = None
        self.traces = None
        self.icons = None

        self.tail_frames = None
        self.size = None

        self.N = None
        self.p = None
        self.status = None
        self.dt = None

    def set_data(self, N: int, p: np.ndarray, status: np.ndarray, dt: float):
        self.N = N
        self.p = p
        self.status = status
        self.dt = dt

    def set_graphics(
        self,
        ax: plt.Axes,
        title: str,
        traces: list,
        icons: list,
        tail_frames: int,
        size: float,
    ):
        self.ax = ax
        self.title = title
        self.traces = traces
        self.icons = icons
        self.tail_frames = tail_frames
        self.size = size

    # Function to update the animation
    def animate(self, i: int):
        agents_colors = [
            "royalblue" if self.status[i, n] else "red" for n in range(self.N)
        ]

        # Update agent's icon and trace
        for n in range(self.N):
            self.icons[n].remove()
            self.icons[n] = plt.Circle(
                (self.p[i, n, 0], self.p[i, n, 1]),
                self.size,
                color=agents_colors[n],
            )

            self.ax.add_patch(self.icons[n])

            if i > self.tail_frames:
                self.traces[n].set_data(
                    self.p[i - self.tail_frames : i, n, 0],
                    self.p[i - self.tail_frames : i, n, 1],
                )
            else:
                self.traces[n].set_data(self.p[0:i, n, 0], self.p[0:i, n, 1])
            self.traces[n].set_color(color=agents_colors[n])

            self.title.set_text(
                "Frame = {0:>4} | Tf = {1:>5.2f} [T] | N = {2:>3} robots".format(
                    i, i * self.dt, self.N
                )
            )
