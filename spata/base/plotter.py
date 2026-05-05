"""spata.base.plotter"""

import time
import numpy as np


class Plotter:

    CONFIG_RADAR_AXES = "SPATA"

    CONFIG_SAVE_DIR_DEFAULT = ""
    CONFIG_SAVE_EXT_DEFAULT = "png"
    CONFIG_SAVE_DPI_DEFAULT = 100

    CONFIG_FIG_SIZE_DEFAULT = (9, 9)
    CONFIG_FIG_STYLE_DEFAULT = "seaborn-v0_8-colorblind"

    CONFIG_AXIS_LIMITS_DEFAULT = (0, 1)
    CONFIG_BUTTON_COLORS_DEFAULT = (
        "#0173b2",
        "#d55e00",
        "#de8f05",
        "#029e73",
        "#cc78bc",
        "#ca9161",
        "#fbafe4",
        "#949494",
        "#ece133",
        "#56b4e9",
    )

    def __init__(
        self,
        show=True,
        show_animated=True,
        save=False,
        save_dir=None,
        save_ext=None,
        save_dpi=None,
        fig_size=None,
        fig_style=None,
        **fig_kwargs,
    ):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.projections as projections
        except Exception:
            raise ImportError(
                "The Plotter object requires the 'matplotlib' package. "
                + "Please install optional dependencies with 'pip install spata[plots]'"
            )

        if self.CONFIG_RADAR_AXES not in projections.get_projection_names():
            self.__init_radar_axes()

        # ------------------------------

        try:
            self.show = bool(show)
        except Exception as e:
            raise TypeError("The Plotter argument 'show' must be a boolean") from e

        if self.show:
            try:
                self.show_animated = bool(show_animated)
            except Exception as e:
                raise TypeError(
                    "The Plotter argument 'show_animated' must be a boolean"
                ) from e
        else:
            self.show_animated = False

        try:
            self.save = bool(save)
        except Exception as e:
            raise TypeError("The Plotter argument 'save' must be a boolean") from e

        # ------------------------------

        if save_dir is None:
            self.save_dir = self.CONFIG_SAVE_DIR_DEFAULT

        else:
            try:
                self.save_dir = str(save_dir)
            except Exception as e:
                raise TypeError(
                    "The Plotter argument 'save_dir' must be a string filepath"
                ) from e

        if save_ext is None:
            self.save_ext = self.CONFIG_SAVE_EXT_DEFAULT

        else:
            try:
                self.save_ext = str(save_ext)
            except Exception as e:
                raise TypeError(
                    "The Plotter argument 'save_ext' must be a string extension"
                ) from e

        if save_dpi is None:
            self.save_dpi = int(self.CONFIG_SAVE_DPI_DEFAULT)

        else:
            try:
                self.save_dpi = int(save_dpi)
            except Exception as e:
                raise TypeError(
                    "The Plotter argument 'save_dpi' must be an integer value"
                ) from e

        # ------------------------------

        if fig_size is None:
            self.fig_size = self.CONFIG_FIG_SIZE_DEFAULT

        else:
            try:
                self.fig_size = [float(item) for item in fig_size]
            except Exception as e:
                raise TypeError(
                    "The Plotter argument 'fig_size' must be a 1D array-like with two values"
                    + ", limiting the vertical and horizontal size of the figure"
                ) from e

            if len(fig_size) != 2 or fig_size[0] <= 0 or fig_size[1] <= 0:
                raise TypeError(
                    "The Plotter argument 'fig_size' must have two values, for the vertical and horizontal size"
                )

        if fig_style is None:
            self.fig_style = self.CONFIG_FIG_STYLE_DEFAULT

        else:
            try:
                self.fig_style = str(fig_style)
                plt.style.use(self.fig_style)

            except Exception:
                raise ValueError(
                    "The Plotter argument 'fig_style' must be a string supported by matplotlib"
                )

        self.fig_kwargs = fig_kwargs

    def __init_radar_axes(self):
        import matplotlib.projections as projections
        from matplotlib.projections.polar import PolarAxes
        from matplotlib.patches import Circle
        from matplotlib.path import Path

        class RadarTransform(PolarAxes.PolarTransform):

            num_vertices = None

            def transform_path_non_affine(self, path):
                # Paths with non-unit interpolation steps correspond to gridlines,
                # where interpolation must be enforced
                if path._interpolation_steps > 1:
                    if self.num_vertices is None:
                        raise RuntimeError("Something went wrong with matplotlib")

                    path = path.interpolated(self.num_vertices)

                return Path(self.transform(path.vertices), path.codes)

        class RadarAxes(PolarAxes):

            name = self.CONFIG_RADAR_AXES
            PolarTransform = RadarTransform

            def _gen_axes_patch(self):
                return Circle((0.5, 0.5), 0.5)

            def set_theta_vertices(self, vertices):
                nvert = len(vertices)

                # Update RadarTransform
                self.PolarTransform.num_vertices = nvert

                # Calculate evenly-spaced axis angles
                theta = np.linspace(0, 2 * np.pi, nvert, endpoint=False)

                # Rotate plot such that the first axis is at the top
                self.set_theta_zero_location("N")

                # Place the labels of the vertices
                self.set_thetagrids(np.degrees(theta), vertices)

                return theta

        projections.register_projection(RadarAxes)

    def __call__(
        self,
        lines,
        alphas=None,
        vertice_labels=None,
        button_labels=None,
        button_colors=None,
        tick_labels=None,
        tick_values=None,
        axis_limits=None,
        verify=True,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        try:
            verify = bool(verify)
        except Exception as e:
            raise TypeError("The Plotter argument 'verify' must be a boolean") from e

        # ------------------------------

        try:
            n_buttons = len(lines)
            n_vertices = len(lines[0][0])
        except Exception as e:
            raise TypeError(
                "The Plotter argument 'lines' must be a 3D array-like"
                + ", in the (n_buttons, n_lines, n_vertices) shape"
            ) from e

        if n_buttons == 0:
            raise ValueError(
                "The Plotter argument 'lines' must have a positive n_buttons"
            )

        if n_vertices == 0:
            raise ValueError(
                "The Plotter argument 'lines' must have a positive n_vertices"
            )

        if verify:
            n_lines = []

            for sublst in lines:
                if len(sublst) == 0:
                    raise ValueError(
                        "The Plotter argument 'lines' must have a positive n_lines"
                    )

                n_lines.append(len(sublst))

                for line in sublst:
                    if len(line) != n_vertices:
                        raise ValueError(
                            "The Plotter argument 'lines' must have"
                            + " the same n_vertices in every line"
                        )

        # ------------------------------

        if alphas is not None:
            try:
                temp_a = alphas[0]
                temp_a = len(alphas)
            except Exception as e:
                raise TypeError(
                    "The Plotter argument 'alphas' must be a 2D array-like"
                    + ", in the (n_buttons, n_lines) shape"
                ) from e

            if temp_a != n_buttons:
                raise ValueError(
                    "The Plotter argument 'alphas' must have"
                    + " the same n_buttons as argument 'lines'"
                )

            if verify:
                for k, sublst in enumerate(alphas):
                    if len(sublst) != n_lines[k]:
                        raise ValueError(
                            "The Plotter argument 'alphas' must have"
                            + " the same n_lines as argument 'lines'"
                        )

                    # for line in sublst:
                    #     if len(line) != n_vertices:
                    #         raise ValueError(
                    #             "The Plotter argument 'alphas' must have"
                    #             + " the same n_vertices as argument 'lines'"
                    #         )

        # ------------------------------

        if vertice_labels is None:
            vertice_labels = [str(item) for item in range(n_vertices)]

        else:
            try:
                vertice_labels = [str(item) for item in vertice_labels]
            except Exception as e:
                raise TypeError(
                    "The Plotter argument 'vertices' must be a 1D array-like of labels"
                    + ", in the (n_vertices, ) shape"
                ) from e

            if len(vertice_labels) != n_vertices:
                raise ValueError(
                    "The Plotter argument 'vertices' must match"
                    + " the n_vertices of argument 'lines'"
                )

        # ------------------------------

        if button_labels is None:
            button_labels = [str(item) for item in range(n_buttons)]

        else:
            try:
                button_labels = [str(item) for item in button_labels]
            except Exception as e:
                raise TypeError(
                    "The Plotter argument 'button_labels' must be a 1D array-like of labels"
                    + ", in the (n_buttons, ) shape"
                ) from e

            if len(button_labels) != n_buttons:
                raise ValueError(
                    "The Plotter argument 'button_labels' must match"
                    + " the n_buttons of argument 'lines'"
                )

        # ------------------------------

        if button_colors is None:
            temp_q, temp_r = divmod(n_buttons, len(self.CONFIG_BUTTON_COLORS_DEFAULT))

            button_colors = temp_q * list(self.CONFIG_BUTTON_COLORS_DEFAULT) + list(
                self.CONFIG_BUTTON_COLORS_DEFAULT[:temp_r]
            )

        else:
            try:
                button_colors = [str(item) for item in button_colors]
            except Exception as e:
                raise TypeError(
                    "The Plotter argument 'button_colors' must be a 1D array-like of color codes"
                    + ", in the (n_buttons, ) shape"
                ) from e

            if len(button_colors) != n_buttons:
                raise ValueError(
                    "The Plotter argument 'button_colors' must match"
                    + " the n_buttons of argument 'lines'"
                )

        # ------------------------------

        if axis_limits is None:
            axis_limits = self.CONFIG_AXIS_LIMITS_DEFAULT

        else:
            try:
                axis_limits = [float(item) for item in axis_limits]
            except Exception as e:
                raise TypeError(
                    "The Plotter argument 'axis_limits' must be a 1D array-like with two values"
                    + ", limiting the minimum and maximum values of the axis"
                ) from e

            if len(axis_limits) != 2 or axis_limits[0] >= axis_limits[1]:
                raise TypeError(
                    "The Plotter argument 'axis_limits' must have two values, for the minimum and maximum values"
                )

        # ------------------------------

        if tick_values is None:
            if tick_labels is not None:
                raise ValueError(
                    "The Plotter argument 'tick_labels' requires the use of the argument 'tick_values'"
                )

        else:
            try:
                tick_values = [float(item) for item in tick_values]
            except Exception as e:
                raise TypeError(
                    "The Plotter argument 'tick_values' must be a 1D array-like of axis tick values"
                ) from e

            for item in tick_values:
                if item < axis_limits[0] or item > axis_limits[1]:
                    raise ValueError(
                        "The Plotter argument 'tick_values' must be within the minimum and maximum of argument 'axis_limits'"
                    )

            if tick_labels is None:
                tick_labels = [str(item) for item in tick_values]

            else:
                try:
                    tick_labels = [str(item) for item in tick_labels]
                except Exception as e:
                    raise TypeError(
                        "The Plotter argument 'tick_labels' must be a 1D array-like of axis tick labels"
                    ) from e

                if len(tick_labels) != len(tick_values):
                    raise ValueError(
                        "The Plotter argument 'tick_labels' must match the number of ticks of argument 'tick_values'"
                    )

        # ------------------------------

        if n_vertices == 1:
            lines = [[sublst[0], sublst[0], sublst[0]] for sublst in lines]

        elif n_vertices == 2:
            lines = [[sublst[0], sublst[1], sublst[1]] for sublst in lines]

        # ------------------------------

        # Start preparing a plot
        figtype = self.CONFIG_RADAR_AXES

        plt.style.use(self.fig_style)

        fig, ax0 = plt.subplots(
            subplot_kw=dict(projection=self.CONFIG_RADAR_AXES),
            figsize=self.fig_size,
            **self.fig_kwargs,
        )

        theta = ax0.set_theta_vertices(vertice_labels)
        theta = [float(item) for item in theta]
        theta = theta + [theta[0]]

        ax0.set_rlim(axis_limits)

        if tick_values is not None:
            ax0.set_rticks(ticks=tick_values, labels=tick_labels, minor=False)
            # zorder=10

        # ------------------------------

        if self.show_animated:
            blit_mapping = {}
            blit_order = [ax0]
        else:
            blit_mapping = None
            blit_order = None

        for k, sublst in enumerate(lines):
            axk = fig.add_axes(
                ax0.get_position(),
                projection=self.CONFIG_RADAR_AXES,
                frameon=False,
                theta_direction=ax0.get_theta_direction(),
                theta_offset=ax0.get_theta_offset(),
            )

            axk.set_axis_off()
            axk.set_rlim(axis_limits)
            axk.set_visible(False)

            if self.show_animated:
                axk.set_animated(True)
                blit_mapping[button_labels[k]] = axk
                blit_order.append(axk)

            # stacked = [np.column_stack((theta, line + line[0])) for line in sublst]
            stacked = [list(zip(theta, list(line) + [line[0]])) for line in sublst]

            if alphas is None:
                lcollection = LineCollection(
                    stacked, color=button_colors[k], linewidth=3
                )
            else:
                lcollection = LineCollection(
                    stacked, alpha=alphas[k], color=button_colors[k], linewidth=3
                )

            axk.add_collection(lcollection)

        # ------------------------------

        # Finish preparing a plot
        if figtype != self.CONFIG_RADAR_AXES:
            fig.tight_layout()

        # Prepare the animations
        if self.show_animated:
            from matplotlib.widgets import CheckButtons

            fig.blit_mapping = blit_mapping
            fig.blit_order = blit_order

            for a in fig.blit_order:
                a.set_visible(True)

            nlabels = len(fig.blit_mapping)
            nchars = max(len(label) for label in fig.blit_mapping)

            if figtype == self.CONFIG_RADAR_AXES:
                bax = fig.add_axes(
                    [0.001, 0.001, nchars * 0.012 + 0.02, nlabels * 0.02 + 0.015]
                )
            else:
                bax = fig.blit_order[0].inset_axes(
                    [0, 0, nchars * 0.007 + 0.02, nlabels * 0.02 + 0.015]
                )
            props = {"color": button_colors}

            buttons = CheckButtons(
                bax,
                labels=list(fig.blit_mapping.keys()),
                actives=[True for _ in range(nlabels)],
                label_props=props,
                frame_props=props,
                check_props=props,
                useblit=True,
            )

        self.last_fig = fig

        # Show figure with or without animations
        if self.show:
            if self.show_animated:

                def button_callback(label):
                    if label is not None and label not in self.last_fig.blit_mapping:
                        raise RuntimeError("Something went wrong with matplotlib")

                    self.last_fig.canvas.restore_region(
                        self.last_fig.blit_ax_background
                    )

                    a = self.last_fig.blit_mapping[label]
                    if a.get_visible():
                        a.set_visible(False)
                        self.last_fig.blit_order.remove(a)
                    else:
                        a.set_visible(True)
                        # if a in fig.blit_order:
                        #     fig.blit_order.remove(a)
                        self.last_fig.blit_order.append(a)

                    for a in self.last_fig.blit_order:
                        a.draw(self.last_fig.canvas.renderer)

                    self.last_fig.canvas.blit(self.last_fig.bbox)
                    self.last_fig.canvas.flush_events()

                buttons.on_clicked(button_callback)

                def draw_callback(event):
                    if event is not None and event.canvas != self.last_fig.canvas:
                        raise RuntimeError("Something went wrong with matplotlib")

                    self.last_fig.blit_ax_background = (
                        self.last_fig.canvas.copy_from_bbox(self.last_fig.bbox)
                    )

                    for a in self.last_fig.blit_order:
                        a.draw(self.last_fig.canvas.renderer)

                fig.canvas.mpl_connect("draw_event", draw_callback)

            try:
                plt.show()
            except Exception:
                raise ValueError(
                    "The current figure could not be showed."
                    + " Check the matplotlib backend alternatives"
                )

            if self.show_animated:
                plt.pause(0.1)
                fig.blit_ax_background = fig.canvas.copy_from_bbox(fig.bbox)

                for a in fig.blit_order:
                    a.draw(fig.canvas.renderer)

                fig.canvas.blit(fig.bbox)
                fig.canvas.flush_events()

        # Save figure with or without animations
        if self.save:
            if self.show_animated:

                if hasattr(fig, "blit_ax_background"):
                    fig.canvas.restore_region(fig.blit_ax_background)

                for a in fig.blit_order:
                    # a.set_visible(True)
                    a.draw(fig.canvas.renderer)

                fig.canvas.blit(fig.bbox)
                fig.canvas.flush_events()

            try:
                plt.savefig(
                    f"{self.save_dir}figure_{int(time.time())}.{self.save_ext}",
                    dpi=self.save_dpi,
                )
            except Exception:
                raise ValueError(
                    "The current figure could not be saved."
                    + " Check the 'save' related arguments of the Plotter object"
                )

        return fig
