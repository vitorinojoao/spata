"""spata.base.plotter"""

import time
import numpy as np


class Plotter:

    CONFIG_PLT_RADAR_AXES = "SPATA"

    CONFIG_PLT_VERTICE_LABELS_LENGTH = 12
    CONFIG_PLT_BUTTON_LABELS_LENGTH = 12

    CONFIG_PLT_FIG_SIZE = (9, 9)
    CONFIG_PLT_FIG_STYLE = "seaborn-v0_8-colorblind"

    CONFIG_PLT_AXES_LIMITS = (0.01, 1.05)
    CONFIG_PLT_AXES_COLORS = (
        "#0173b2",
        "#d55e00",
        "#de8f05",
        "#cc78bc",
        "#ca9161",
        "#56b4e9",
        "#ece133",
        "#029e73",
        "#fbafe4",
    )

    CONFIG_PLT_TICK_VALUES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    CONFIG_PLT_TICK_LABELS = (
        "\u03bc-3.5\u03c3",
        "\u03bc-2.5\u03c3",
        "\u03bc-1.5\u03c3",
        "\u03bc-0.5\u03c3",
        "\u03bc0.5\u03c3",
        "\u03bc1.5\u03c3",
        "\u03bc2.5\u03c3",
        "\u03bc3.5\u03c3",
    )

    def __init__(
        self,
        show=True,
        show_animated=True,
        save=False,
        save_directory="",
        save_prefix="figure_",
        save_extension="png",
        save_dpi=300,
        fig_size=None,
        fig_style=None,
        axes_colors=None,
        axes_limits=None,
        tick_values=None,
        tick_labels=None,
        **kwargs,
    ):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.projections as projections
        except Exception:
            raise ImportError(
                "The Plotter object requires the 'matplotlib' package. "
                + "Please install optional dependencies with 'pip install spata[plots]'"
            )

        if self.CONFIG_PLT_RADAR_AXES not in projections.get_projection_names():
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

        # ------------------------------

        try:
            self.save = bool(save)
        except Exception as e:
            raise TypeError("The Plotter argument 'save' must be a boolean") from e

        try:
            self.save_directory = str(save_directory)
        except Exception as e:
            raise TypeError(
                "The Plotter argument 'save_directory' must be a string directory path"
            ) from e

        try:
            self.save_prefix = str(save_prefix)
        except Exception as e:
            raise TypeError(
                "The Plotter argument 'save_prefix' must be a string file name prefix"
            ) from e

        try:
            self.save_extension = str(save_extension)
        except Exception as e:
            raise TypeError(
                "The Plotter argument 'save_extension' must be a string file extension"
            ) from e

        try:
            self.save_dpi = int(save_dpi)
        except Exception as e:
            raise TypeError("The Plotter argument 'save_dpi' must be an integer") from e

        if self.save_dpi <= 0:
            raise TypeError(
                "The Plotter argument 'save_dpi' must be a positive integer"
            )

        # ------------------------------

        if fig_size is None:
            self.fig_size = self.CONFIG_PLT_FIG_SIZE
        else:
            try:
                self.fig_size = [float(item) for item in fig_size]
            except Exception as e:
                raise TypeError(
                    "The Plotter argument 'fig_size' must be a 1D array-like with two values"
                    + ", limiting the vertical and horizontal size of the figure"
                ) from e

            if (
                len(self.fig_size) != 2
                or self.fig_size[0] <= 0
                or self.fig_size[1] <= 0
            ):
                raise ValueError(
                    "The Plotter argument 'fig_size' must have two values, for the vertical and horizontal size"
                )

        # ------------------------------

        if fig_style is None:
            self.fig_style = self.CONFIG_PLT_FIG_STYLE
        else:
            try:
                self.fig_style = str(fig_style)
                plt.style.use(self.fig_style)

            except Exception:
                raise ValueError(
                    "The Plotter argument 'fig_style' must be a string supported by matplotlib"
                )

        # ------------------------------

        if axes_colors is None:
            self.axes_colors = self.CONFIG_PLT_AXES_COLORS

        else:
            try:
                self.axes_colors = [str(item) for item in axes_colors]
            except Exception as e:
                raise TypeError(
                    "The Plotter argument 'axes_colors' must be a 1D array-like"
                    + " with strings representing color codes"
                ) from e

            if len(self.axes_colors) < 1:
                raise ValueError(
                    "The Plotter argument 'axes_colors' must have at least one string representing a color code"
                )

        # ------------------------------

        if axes_limits is None:
            self.axes_limits = self.CONFIG_PLT_AXES_LIMITS

        else:
            try:
                self.axes_limits = [float(item) for item in axes_limits]
            except Exception as e:
                raise TypeError(
                    "The Plotter argument 'axes_limits' must be a 1D array-like with two values"
                    + ", limiting the minimum and maximum values of the axes"
                ) from e

            if len(self.axes_limits) != 2 or self.axes_limits[0] >= self.axes_limits[1]:
                raise ValueError(
                    "The Plotter argument 'axes_limits' must have two values, for the minimum and maximum values"
                )

        # ------------------------------

        if tick_values is None:
            if tick_labels is None:
                self.tick_values = self.CONFIG_PLT_TICK_VALUES
                self.tick_labels = self.CONFIG_PLT_TICK_LABELS

            else:
                raise ValueError(
                    "The Plotter argument 'tick_labels' requires the use of the argument 'tick_values'"
                )

        else:
            try:
                self.tick_values = [float(item) for item in tick_values]
            except Exception as e:
                raise TypeError(
                    "The Plotter argument 'tick_values' must be a 1D array-like of axis tick values"
                ) from e

            for item in self.tick_values:
                if item < self.axes_limits[0] or item > self.axes_limits[1]:
                    raise ValueError(
                        "The Plotter argument 'tick_values' must be within the minimum and maximum of argument 'axes_limits'"
                    )

            if tick_labels is None:
                self.tick_labels = [str(item) for item in tick_values]

            else:
                try:
                    self.tick_labels = [str(item) for item in tick_labels]
                except Exception as e:
                    raise TypeError(
                        "The Plotter argument 'tick_labels' must be a 1D array-like of axis tick labels"
                    ) from e

                if len(self.tick_labels) != len(self.tick_values):
                    raise ValueError(
                        "The Plotter argument 'tick_labels' must match the number of ticks of argument 'tick_values'"
                    )

        # ------------------------------

        if kwargs:
            kwargs.pop("subplot_kw", None)
            kwargs.pop("figsize", None)
            kwargs.pop("nrows", None)
            kwargs.pop("ncols", None)
            kwargs.pop("sharex", None)
            kwargs.pop("sharey", None)
            kwargs.pop("squeeze", None)

        self.kwargs = kwargs

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

            name = self.CONFIG_PLT_RADAR_AXES
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
        verify=None,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.widgets import CheckButtons

        if verify is None:
            verify = True
        else:
            try:
                verify = bool(verify)
            except Exception as e:
                raise TypeError(
                    "The Plotter argument 'verify' must be a boolean"
                ) from e

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

        if alphas is None:
            alphas = [1 for _ in range(n_buttons)]
        else:
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
            vertice_labels = [
                str(item)[: self.CONFIG_PLT_VERTICE_LABELS_LENGTH]
                for item in range(n_vertices)
            ]

        else:
            try:
                vertice_labels = [
                    str(item)[: self.CONFIG_PLT_VERTICE_LABELS_LENGTH]
                    for item in vertice_labels
                ]
            except Exception as e:
                raise TypeError(
                    "The Plotter argument 'vertice_labels' must be a 1D array-like of labels"
                    + ", in the (n_vertices, ) shape"
                ) from e

            if len(vertice_labels) != n_vertices:
                raise ValueError(
                    "The Plotter argument 'vertice_labels' must match"
                    + " the n_vertices of argument 'lines'"
                )

        # ------------------------------

        if button_labels is None:
            button_labels = [
                str(item)[: self.CONFIG_PLT_BUTTON_LABELS_LENGTH]
                for item in range(n_buttons)
            ]

        else:
            try:
                button_labels = [
                    str(item)[: self.CONFIG_PLT_BUTTON_LABELS_LENGTH]
                    for item in button_labels
                ]
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
        # Adjust to n_vertices and n_buttons

        if n_vertices == 1:
            lines = [[sublst[0], sublst[0], sublst[0]] for sublst in lines]

        elif n_vertices == 2:
            lines = [[sublst[0], sublst[1], sublst[1]] for sublst in lines]

        temp_q, temp_r = divmod(n_buttons, len(self.axes_colors))

        axes_colors = temp_q * list(self.axes_colors) + list(self.axes_colors[:temp_r])

        # ------------------------------
        # Prepare the base: ax0

        plt.style.use(self.fig_style)

        fig, ax0 = plt.subplots(
            subplot_kw={"projection": self.CONFIG_PLT_RADAR_AXES},
            figsize=self.fig_size,
            nrows=1,
            ncols=1,
            sharex=False,
            sharey=False,
            squeeze=True,
            **self.kwargs,
        )

        theta = ax0.set_theta_vertices(vertice_labels)
        theta = [float(item) for item in theta]
        theta = theta + [theta[0]]

        ax0.set_rlim(self.axes_limits)
        ax0.set_rticks(ticks=self.tick_values, labels=self.tick_labels, minor=False)

        # ------------------------------
        # Prepare axes: ax1, ..., axk

        if self.show_animated:
            fig.blit_mapping = {}
            fig.blit_order = [ax0]

        for k, sublst in enumerate(lines):
            axk = fig.add_axes(
                ax0.get_position(),
                projection=self.CONFIG_PLT_RADAR_AXES,
                frameon=False,
                theta_direction=ax0.get_theta_direction(),
                theta_offset=ax0.get_theta_offset(),
            )

            axk.set_axis_off()
            axk.set_rlim(self.axes_limits)

            if self.show_animated:
                axk.set_animated(True)
                fig.blit_mapping[button_labels[k]] = axk
                fig.blit_order.append(axk)

            # stacked = [np.column_stack((theta, line + line[0])) for line in sublst]
            stacked = [list(zip(theta, list(line) + [line[0]])) for line in sublst]

            collection = LineCollection(
                stacked, alpha=alphas[k], color=axes_colors[k], linewidth=3
            )

            axk.add_collection(collection)

        # ------------------------------
        # Prepare check buttons

        n_chars = max(len(label) for label in button_labels)

        bax = fig.add_axes(
            [0.001, 0.001, n_chars * 0.02 + 0.01, n_buttons * 0.02 + 0.01]
        )

        props = {"color": axes_colors}

        buttons = CheckButtons(
            bax,
            labels=button_labels,
            actives=[True for _ in range(n_buttons)],
            label_props=props,
            frame_props=props,
            check_props=props,
            useblit=self.show_animated,
        )

        # ------------------------------
        # Prepare animations

        self.fig = fig

        if self.show_animated:
            # Animated check buttons for axes with blitting

            def button_callback(label):
                if label not in self.fig.blit_mapping:
                    raise RuntimeError("Something went wrong with matplotlib")

                self.fig.canvas.restore_region(self.fig.blit_background)

                a = self.fig.blit_mapping[label]

                if a.get_visible():
                    a.set_visible(False)
                    self.fig.blit_order.remove(a)
                else:
                    a.set_visible(True)
                    self.fig.blit_order.append(a)

                for a in self.fig.blit_order:
                    a.draw(self.fig.canvas.renderer)

                self.fig.canvas.blit(self.fig.bbox)
                self.fig.canvas.flush_events()

            buttons.on_clicked(button_callback)

        else:
            # Static check buttons used just as a legend
            buttons.on_clicked(lambda label: None)

        if self.show_animated:
            # Animated drawing of axes with blitting

            def draw_callback(event):
                if event is not None and event.canvas != self.fig.canvas:
                    raise RuntimeError("Something went wrong with matplotlib")

                self.fig.blit_background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

                for a in self.fig.blit_order:
                    a.draw(self.fig.canvas.renderer)

            self.fig.canvas.mpl_connect("draw_event", draw_callback)
            self.fig.blit_background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

            for a in self.fig.blit_order:
                a.draw(self.fig.canvas.renderer)

            self.fig.canvas.blit(self.fig.bbox)
            self.fig.canvas.flush_events()

        # ------------------------------
        # Save and/or show figure

        if self.save:
            try:
                plt.savefig(
                    f"{self.save_directory}{self.save_prefix}{int(time.time())}.{self.save_extension}",
                    dpi=self.save_dpi,
                )
            except Exception:
                raise ValueError(
                    "The current figure could not be saved."
                    + " Check Plotter arguments related to 'save'"
                )

        if self.show:
            try:
                plt.show()
            except Exception:
                raise ValueError(
                    "The current figure could not be showed."
                    + " Check matplotlib backend alternatives"
                )

        return self.fig
