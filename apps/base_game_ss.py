import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import tkinter as tk

# # Tell matplotlib to use latex (way more laggy)
# from ssl_simulator.visualization import set_paper_parameters
# set_paper_parameters(fontsize=12)

from ssl_simulator import SimulationEngine, parse_kwargs
from ssl_simulator.components.scalar_fields import SigmaNonconvex, PlotterScalarField
from ssl_simulator.visualization import Plotter, config_axis

#######################################################################################

class AppGameSS:
    def __init__(self, fig, ax, scalar_field, 
                 simulator_engine: SimulationEngine, simulator_plotter: Plotter, 
                 **kwargs):
        self.fig, self.ax = fig, ax
        self.scalar_field = scalar_field
        self.simulator_engine = simulator_engine
        self.simulator_plotter = simulator_plotter

        self.field_plotter = PlotterScalarField(self.scalar_field)
        
        kw_field = dict(contour_levels=8, n=200, cbar_minor_ticks=False, cbar_color="white")
        self.kw_field = parse_kwargs(kwargs, kw_field)

        # Configure the given axis
        self.ax.set_title("Click to move the source • Scroll to rotate • Ctrl + scroll to scale", color="white")
        self.ax.set_xlabel(r"$X$", color="white")
        self.ax.set_ylabel(r"$Y$", color="white")
        config_axis(self.ax, xlims=[-130,130], ylims=[-90,90])

        self.fig.set_facecolor("black")
        self.ax.set_facecolor("black")
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_color('white')

        self.status_label = tk.Label(
            self.fig.canvas.manager.window, text=f"tf = {simulator_engine.time:.2f}", 
            fg="black", font=("arial", 12))
        self.status_label.pack(anchor="s")

        # Set options
        opts = dict(step_scale = 0.2, step_rot = np.pi/2 * 0.2, fps=20)
        self.opts = parse_kwargs(kwargs, opts)

        # Timers
        self.interval = 1/opts["fps"] # s
        self._sim_timer = self.fig.canvas.new_timer(interval=int(1000*self.interval))
        self._init_sim_timer()

        self._draw_pending = False
        self._draw_delay = 10
        self._draw_timer_running = False
        self._init_draw_timer()

        self._delay_timer = None
        self._delay_ms = 10
        self._delay_timer_running = False
        self._init_delay_timer()

        self._contours_timer = None
        self._contours_ms = 500
        self._contours_timer_running = False
        self._init_contours_timer()

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # Track mouse state
        self.is_pressed = [False]

        # Draw the initial plot
        self.field_plotter.draw(fig=self.fig, ax=self.ax, **self.kw_field)
        self.field_plotter.draw_contours = False
        self.simulator_plotter.draw()
        
    # ---------------------------------------------------------------------------------
    # Event Handlers

    def on_press(self, event):
        if event.inaxes == self.ax:
            self.is_pressed[0] = True
            self._update_scalar_field([event.xdata, event.ydata])

    def on_release(self, event):
        self.is_pressed[0] = False

    def on_motion(self, event):
        if self.is_pressed[0] and event.inaxes == self.ax:
            self._update_scalar_field([event.xdata, event.ydata])

    def on_scroll(self, event):
        if event.inaxes == self.ax:
            if event.key == 'control':
                step_scale = self.opts["step_scale"]
                if event.button == "up":
                    self.scalar_field.transf_a = max(step_scale, self.scalar_field.transf_a - step_scale)
                    self._update_scalar_field()
                elif event.button == "down":
                    self.scalar_field.transf_a = self.scalar_field.transf_a + step_scale
                    self._update_scalar_field()
            else:
                step_rot = self.opts["step_rot"]
                if event.button == "up":
                    self.scalar_field.transf_w += step_rot
                    self._update_scalar_field()
                elif event.button == "down":
                    self.scalar_field.transf_w -= step_rot
                    self._update_scalar_field()

    def on_key(self, event):
        pass     

    # ---------------------------------------------------------------------------------
    # Main Methods

    def _step_simulation(self):
        self.simulator_engine.run(self.interval, eta=False)
        self.simulator_plotter.update()
        self.status_label.config(text=f"tf = {self.simulator_engine.time:.2f}")

        self._draw_pending = True
        if not self._draw_timer_running:
            self._draw_timer.start()

    def _update_scalar_field(self, source=None):
        if source:
            self.scalar_field.mu = source

        self.field_plotter.update()
        self._contours_timer.start()
        
        self._draw_pending = True
        if not self._draw_timer_running:
            self._draw_timer.start()

    # ---------------------------------------------------------------------------------
    #  Timer Methods

    def _init_sim_timer(self):
        self._sim_timer.add_callback(self._schedule_canvas_task, self._step_simulation)
        self._sim_timer.start()

    def _init_draw_timer(self):
        self._draw_timer = self.fig.canvas.new_timer(interval=self._draw_delay)
        self._draw_timer.add_callback(self._draw_if_pending)

    def _init_delay_timer(self):
        self._delay_timer = self.fig.canvas.new_timer(interval=self._delay_ms)
        self._delay_timer.add_callback(self._stop_delay_timer)

    def _init_contours_timer(self):
        self._contours_timer = self.fig.canvas.new_timer(interval=self._contours_ms)
        self._contours_timer.add_callback(self._stop_contours_timer)

    def _stop_delay_timer(self):
        self._delay_timer.stop()
        self._delay_timer_running = False

    def _stop_contours_timer(self):
        self.field_plotter.draw_contours = True
        self.field_plotter.update()
        self.field_plotter.draw_contours = False
        self.fig.canvas.draw_idle()

        self._contours_timer.stop()
        self._contours_timer_running = False
    
    def _schedule_canvas_task(self, task):
        # QTimer.singleShot(0, task)
        self.fig.canvas.manager.window.after(0, task)

    def _draw_if_pending(self):
        if self._draw_pending:
            self.fig.canvas.draw_idle()
            self._draw_pending = False

#######################################################################################