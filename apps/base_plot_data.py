import matplotlib
matplotlib.use("TkAgg")

import os
import glob

import matplotlib.pyplot as plt
import tkinter as tk

# Tell matplotlib to use latex (way more laggy)
# from ssl_simulator.visualization import set_paper_parameters
# set_paper_parameters(fontsize=12)

from ssl_simulator import load_sim, parse_kwargs

#######################################################################################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class AppPlotData:
    def __init__(self, plotter, **kwargs):
        self.plotter = plotter
        self.fig = self.plotter.fig

        self.plotter.settings = None
        self.plotter.data = None

        self.csv_file_paths = None
        self.csv_filenames = None
        self.actual_filename = None

        # Filename label
        self.file_label = tk.Label(
        self.fig.canvas.manager.window, text=f"", 
        fg="black", font=("arial", 20))
        self.file_label.pack(anchor="s")

        # Set options
        default_data_path = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
        opts = dict(data_path=default_data_path, max_size_mb=100, ascending=True)
        self.opts = parse_kwargs(kwargs, opts)

        # Timers

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # Track mouse state
        self.is_pressed = [False]

        # ----------------------------------------------------

        # Load the data directory path and look for data .csv files
        if not os.path.exists(self.opts["data_path"]):
            raise FileNotFoundError(f"Data path does not exist: {self.opts['data_path']}")

        self.csv_file_paths = glob.glob(os.path.join(self.opts["data_path"], "*.csv"))
        if self.csv_file_paths:
            self.csv_file_paths.sort()
        else:
            raise FileNotFoundError(f"No CSV files found in '{self.opts['data_path']}'")
        
        self.csv_filenames = [os.path.basename(path) for path in self.csv_file_paths]
        print(self.csv_filenames)

        # Draw the initial plot

        self.actual_filename = self.csv_filenames[0]
        self.plotter.data, self.plotter.settings = load_sim(self.csv_file_paths[0], debug=False, max_size_mb=self.opts["max_size_mb"]) # TODO: fix when no settings provided
        self.file_label.config(text=f"{self.actual_filename}")
        self.plotter.draw()

    # ---------------------------------------------------------------------------------
    # Main Methods

    def update_figure(self, idx):
        filepath = self.csv_file_paths[idx]
        self.actual_filename = self.csv_filenames[idx]

        self.file_label.config(text="LOADING . . .")
        self.file_label.update_idletasks()

        self.plotter.data, self.plotter.settings = load_sim(filepath, debug=False, max_size_mb=self.opts["max_size_mb"])
        self.plotter.title = self.actual_filename
        self.plotter.update()
        self.fig.canvas.draw_idle()

        self.file_label.config(text=f"{self.actual_filename}")

    def next_file(self):
        actual_idx = self.csv_filenames.index(self.actual_filename) # TODO: catch error
        next_idx = actual_idx + 1 if actual_idx + 1 < len(self.csv_filenames) else 0
        self.update_figure(next_idx)

    def prev_file(self):
        actual_idx = self.csv_filenames.index(self.actual_filename) # TODO: catch error
        next_idx = actual_idx - 1 if actual_idx - 1 >= 0 else -1
        self.update_figure(next_idx)

    # ---------------------------------------------------------------------------------
    # Event Handlers

    def on_press(self, event):
        self.is_pressed[0] = True
        pass

    def on_release(self, event):
        self.is_pressed[0] = False

    def on_motion(self, event):
        if self.is_pressed[0]:
            pass

    def on_scroll(self, event):
        if event.button == "up":
            pass
        elif event.button == "down":
            pass

    def on_key(self, event):
        if event.key in ["left", "a"]:
            self.prev_file()
        elif event.key in ["right", "d"]:
            self.next_file()       

    # ---------------------------------------------------------------------------------
    #  Timer Methods

    def _init_draw_timer(self):
        self._draw_timer = self.fig.canvas.new_timer(interval=self._draw_delay)
        self._draw_timer.add_callback(self._draw_if_pending)

#######################################################################################