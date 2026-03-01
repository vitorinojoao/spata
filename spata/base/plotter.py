"""spata.base.plotter"""

import time
import numpy as np


class Plotter:

    PAIRWISE_FIGURE = "spata_pairwise"
    RADAR_FIGURE = "spata_radar"

    DEFAULT_FIGSIZE = (9, 9)
    DEFAULT_FIGCOLORS = (
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
    DEFAULT_FIGMARKERS = (
        "s",
        "o",
        "v",
        "^",
        "<",
        ">",
        "P",
        "X",
        "p",
        "*",
    )
    DEFAULT_FIGSTYLE = "seaborn-v0_8-colorblind"

    def __init__(
        self,
        show=True,
        save=False,
        figsize=None,
        figcolors=None,
        figmarkers=None,
        figstyle=None,
        savedir="",
        saveext="png",
        savedpi=100,
        **kwargs,
    ):
        try:
            import matplotlib.pyplot
        except Exception:
            raise ImportError(
                "The matplotlib package is required for showing and saving figures"
            )

        if figsize is None:
            self.figsize = Plotter.DEFAULT_FIGSIZE
        elif not isinstance(figsize, (list, tuple)) or len(figsize) != 2:
            raise TypeError(
                "The 'figsize' argument must be a valid '(width, height)' tuple for matplotlib"
                + f". Found: {figsize}"
            )
        else:
            self.figsize = tuple(figsize)

        if figcolors is None:
            self.figcolors = Plotter.DEFAULT_FIGCOLORS
        elif not isinstance(figcolors, (list, tuple)) or len(figcolors) < 1:
            raise TypeError(
                "The 'figcolors' argument must be a list of valid colors for matplotlib"
                + f". Found: {figcolors}"
            )
        else:
            self.figcolors = tuple(figcolors)

        if figmarkers is None:
            self.figmarkers = Plotter.DEFAULT_FIGMARKERS
        elif not isinstance(figmarkers, (list, tuple)) or len(figcolors) < 1:
            raise TypeError(
                "The 'figmarkers' argument must be a list of valid markers for matplotlib"
                + f". Found: {figmarkers}"
            )
        else:
            self.figmarkers = tuple(figmarkers)

        if figstyle is None:
            self.figstyle = Plotter.DEFAULT_FIGSTYLE
        else:
            self.figstyle = figstyle

        kwargs["figsize"] = self.figsize
        self.figparams = kwargs

        self.show = bool(show)
        self.save = bool(save)

        self.savedir = str(savedir)
        self.saveext = str(saveext)
        self.savedpi = int(savedpi)

    def start_plot(
        self,
        figtype,
        features=None,
    ):
        import matplotlib.pyplot as plt

        try:
            plt.style.use(self.figstyle)
        except Exception:
            raise ValueError(
                "The 'figstyle' argument of the Plotter object is not supported by matplotlib"
                + f". Found: {self.figstyle}"
            )

        if figtype == Plotter.PAIRWISE_FIGURE:
            fig, axbase = plt.subplots(**self.figparams)

            res = fig, axbase, self.figmarkers, self.figcolors

        elif figtype == Plotter.RADAR_FIGURE:
            import matplotlib.projections as projections

            if Plotter.RADAR_FIGURE not in projections.get_projection_names():
                _setup_radar_axes()

            fig, axbase = plt.subplots(
                subplot_kw=dict(projection=Plotter.RADAR_FIGURE), **self.figparams
            )

            theta = axbase.set_theta_vertices(features)

            theta = np.append(theta, theta[0])

            res = fig, axbase, theta, self.figcolors

        else:
            raise NotImplementedError(
                f"The requested type of figure is not yet supported: {figtype}"
            )

        self.last_figure_ = fig
        self.last_figtype_ = figtype
        return res

    def finish_plot(
        self,
        animated,
        ax_ordered=None,
        ax_labels=None,
    ):
        import matplotlib.pyplot as plt

        if self.last_figtype_ != Plotter.RADAR_FIGURE:
            self.last_figure_.tight_layout()

        # Prepare the animations
        if animated:
            from matplotlib.widgets import CheckButtons

            self.last_figure_.blit_ax_ordered = ax_ordered
            self.last_figure_.blit_ax_labels = ax_labels

            for a in self.last_figure_.blit_ax_ordered:
                a.set_visible(True)

            nlabels = len(self.last_figure_.blit_ax_labels)
            nchars = max(len(label) for label in self.last_figure_.blit_ax_labels)

            if self.last_figtype_ == Plotter.RADAR_FIGURE:
                bax = self.last_figure_.add_axes(
                    [0.001, 0.001, nchars * 0.012 + 0.02, nlabels * 0.02 + 0.015]
                )
            else:
                bax = self.last_figure_.blit_ax_ordered[0].inset_axes(
                    [0, 0, nchars * 0.007 + 0.02, nlabels * 0.02 + 0.015]
                )
            props = {"color": self.figcolors[:nlabels]}

            buttons = CheckButtons(
                bax,
                labels=list(self.last_figure_.blit_ax_labels.keys()),
                actives=[True for _ in range(nlabels)],
                label_props=props,
                frame_props=props,
                check_props=props,
                useblit=True,
            )

        # Show figure with or without animations
        if self.show:
            if animated:

                def button_callback(label):
                    self.last_figure_.canvas.restore_region(
                        self.last_figure_.blit_ax_background
                    )

                    a = self.last_figure_.blit_ax_labels[label]
                    if a.get_visible():
                        a.set_visible(False)
                        self.last_figure_.blit_ax_ordered.remove(a)
                    else:
                        a.set_visible(True)
                        # if a in fig.blit_ax_ordered:
                        #     fig.blit_ax_ordered.remove(a)
                        self.last_figure_.blit_ax_ordered.append(a)

                    for a in self.last_figure_.blit_ax_ordered:
                        a.draw(self.last_figure_.canvas.renderer)

                    self.last_figure_.canvas.blit(self.last_figure_.bbox)
                    self.last_figure_.canvas.flush_events()

                buttons.on_clicked(button_callback)

                def draw_callback(event):
                    if event is not None and event.canvas != self.last_figure_.canvas:
                        raise RuntimeError("Something went wrong with matplotlib")

                    self.last_figure_.blit_ax_background = (
                        self.last_figure_.canvas.copy_from_bbox(self.last_figure_.bbox)
                    )

                    for a in self.last_figure_.blit_ax_ordered:
                        a.draw(self.last_figure_.canvas.renderer)

                self.last_figure_.canvas.mpl_connect("draw_event", draw_callback)

            try:
                plt.show()
            except Exception:
                raise ValueError(
                    "The current figure could not be showed."
                    + " Check the matplotlib backend alternatives"
                )

            if animated:
                plt.pause(0.1)
                self.last_figure_.blit_ax_background = (
                    self.last_figure_.canvas.copy_from_bbox(self.last_figure_.bbox)
                )

                for a in self.last_figure_.blit_ax_ordered:
                    a.draw(self.last_figure_.canvas.renderer)

                self.last_figure_.canvas.blit(self.last_figure_.bbox)
                self.last_figure_.canvas.flush_events()

        # Save figure with or without animations
        if self.save:
            if animated:

                if hasattr(self.last_figure_, "blit_ax_background"):
                    self.last_figure_.canvas.restore_region(
                        self.last_figure_.blit_ax_background
                    )

                for a in self.last_figure_.blit_ax_ordered:
                    # a.set_visible(True)
                    a.draw(self.last_figure_.canvas.renderer)

                self.last_figure_.canvas.blit(self.last_figure_.bbox)
                self.last_figure_.canvas.flush_events()

            try:
                plt.savefig(
                    f"{self.savedir}figure_{int(time.time())}.{self.saveext}",
                    dpi=self.savedpi,
                )
            except Exception:
                raise ValueError(
                    "The current figure could not be saved."
                    + " Check the 'save' related arguments of the Plotter object"
                )


def _setup_radar_axes():
    import matplotlib.projections as projections
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.patches import Circle
    from matplotlib.path import Path

    class RadarTransform(PolarAxes.PolarTransform):

        num_vertices = None

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                if self.num_vertices is None:
                    raise RuntimeError("Something went wrong with matplotlib")

                path = path.interpolated(self.num_vertices)

            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = Plotter.RADAR_FIGURE
        PolarTransform = RadarTransform

        def _gen_axes_patch(self):
            return Circle((0.5, 0.5), 0.5)

        def set_theta_vertices(self, vertices):
            nvert = len(vertices)

            # update RadarTransform
            self.PolarTransform.num_vertices = nvert

            # calculate evenly-spaced axis angles
            theta = np.linspace(0, 2 * np.pi, nvert, endpoint=False)

            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

            # place the labels of the vertices
            self.set_thetagrids(np.degrees(theta), vertices)

            return theta

    projections.register_projection(RadarAxes)
