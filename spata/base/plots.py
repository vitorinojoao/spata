"""spata.base.plots"""

from spata.base.card import Card
from spata.base.plotter import Plotter

import math
import numpy as np


def prepare_features(card, features):

    if not isinstance(card, Card):
        raise TypeError(
            "The 'card' argument must be a valid Card object" + f". Found: {type(card)}"
        )

    if features is None:
        features = [j for j in range(len(card.features))]

    elif isinstance(features, int):
        if features < 0 or features >= (ndist := len(card.features)):
            raise ValueError(
                f"The 'features' optional argument must contain feature indexes between 0 and {ndist - 1}"
                + f". Found the value: {features}"
            )
        features = [features]

    else:
        try:
            features = list(features)
        except Exception:
            raise TypeError(
                "The 'features' optional argument must be a feature index (int) or an array-like of feature indexes (list of ints)"
                + f". Found: {type(features)}"
            )

        if (nfeat := len(features)) == 0 or nfeat > (ndist := len(card.features)):
            raise ValueError(
                f"The 'features' optional argument must contain at least 1 and at most {ndist} feature indexes"
                + f". Found an array-like of size: {nfeat}"
            )

        for j in features:
            if not isinstance(j, int):
                raise TypeError(
                    f"The 'features' optional argument must be a feature index (int) or an array-like of feature indexes (list of ints)"
                    + f". Found an array-like that contains: {type(j)}"
                )
            if j < 0 or j >= ndist:
                raise ValueError(
                    f"The 'features' optional argument must contain feature indexes between 0 and {ndist - 1}"
                    + f". Found an array-like that contains the value: {j}"
                )

    return features


def summary(card, features=None, plotter=None):

    features = prepare_features(card, features)

    if plotter is not None and not isinstance(plotter, Plotter):
        raise TypeError(
            "The 'plotter' argument must be a valid Plotter object"
            + f". Found: {type(plotter)}"
        )

    nfeat = len(features)

    if nfeat == 1:
        features = features * 3
        nfeat = 3

    elif nfeat == 2:
        features.extend(features)
        nfeat = 4

    res = []

    if plotter is not None:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        # Start preparing a plot
        fig, ax0, theta, figcolors = plotter.start_plot(
            figtype=Plotter.RADAR_FIGURE,
            features=["f" + str(num) for num in features],
        )

        rlim = (0.01, 1.05)
        ax0.set_rlim(rlim)

        # gran = float("0." + "5" * card.granularity)

        ax0.set_rticks(
            # ticks=[(val / 10) + gran for val in Card.CLIST],
            ticks=[(val / 10) for val in Card.CLIST[1:]],
            labels=[
                "\u03bc-3.5\u03c3",
                "\u03bc-2.5\u03c3",
                "\u03bc-1.5\u03c3",
                "\u03bc-0.5\u03c3",
                "\u03bc0.5\u03c3",
                "\u03bc1.5\u03c3",
                "\u03bc2.5\u03c3",
                "\u03bc3.5\u03c3",
            ],
            # ticks=(
            #     ticks := [code - Card.C5 for code in Card.CLIST]
            # ),
            # labels=ticks,
            minor=False,
            # zorder=10,
        )

        axs = [
            fig.add_axes(
                ax0.get_position(),
                projection=Plotter.RADAR_FIGURE,
                frameon=False,
                theta_direction=ax0.get_theta_direction(),
                theta_offset=ax0.get_theta_offset(),
            )
            for i in range(len(card.classes))
        ]

        for a in axs:
            a.set_axis_off()
            a.set_rlim(rlim)
            a.set_visible(False)
            a.set_animated(True)

        labels = {}
        order = [ax0]

        # difference = int("5" * card.granularity)
        # scaling = 1 / card.scale

        vectorized = np.vectorize(lambda x: 10 ** int(math.log10(x) + 1), otypes=[int])

        for c, combos in card.combinations.items():
            label = card.classes[c][0]

            color = figcolors[c]
            alpha_scale = 0.85 / (card.classes[c][2] ** 0.3)

            comboarray = combos[:, features]  # - difference

            scaling = vectorized(comboarray)
            comboarray = comboarray / scaling

            for j in range(nfeat):
                if j != 0:
                    prevf = j - 1
                    temparray = comboarray[:, prevf : j + 1]

                else:
                    prevf = nfeat - 1
                    temparray = comboarray[:, [prevf, 0]]

                minicombos, counts = np.unique(temparray, return_counts=True, axis=0)

                final = np.zeros(shape=(minicombos.shape[0], nfeat + 1), dtype=float)

                final[:, prevf] = minicombos[:, 0]
                final[:, j] = minicombos[:, 1]

                # if scaling == 1:
                #     final[:, prevf] = minicombos[:, 0]
                #     final[:, j] = minicombos[:, 1]
                # else:
                #     final[:, prevf] = minicombos[:, 0] * scaling
                #     final[:, j] = minicombos[:, 1] * scaling

                alphas = (counts**0.3) * alpha_scale + 0.1

                # Close the lines
                final[:, -1] = final[:, 0]

                lines = [np.column_stack((theta, combo)) for combo in final]
                lc = LineCollection(lines, alpha=alphas, color=color, linewidth=3)
                axs[c].add_collection(lc)

            labels[label] = axs[c]
            order.append(axs[c])

        # Finish preparing a plot
        plotter.finish_plot(
            animated=True,
            ax_ordered=order,
            ax_labels=labels,
        )

    return res


# def feature_selection_summary(card, features=None, plotter=None):

#     features = prepare_features(card, features)

#     if plotter is not None and not isinstance(plotter, Plotter):
#         raise TypeError(
#             "The 'plotter' argument must be a valid Plotter object"
#             + f". Found: {type(plotter)}"
#         )

#     nfeat = len(features)

#     if nfeat == 1:
#         features = features * 3
#         nfeat = 3

#     elif nfeat == 2:
#         features.extend(features)
#         nfeat = 4

#     # TODO: create summary based on "minicombos", which are currently being used below
#     res = []

#     if plotter is not None:
#         import matplotlib.pyplot as plt
#         from matplotlib.collections import LineCollection

#         # Start preparing a plot
#         fig, ax0, theta, figcolors = plotter.start_plot(
#             figtype=Plotter.RADAR_FIGURE, features=features
#         )

#         rlim = (-0.5, 10.5)
#         ax0.set_rlim(rlim)

#         ax0.set_rticks(
#             ticks=Card.CLIST,
#             labels=Card.CLIST,
#             # ticks=(
#             #     ticks := [code - Card.C5 for code in Card.CLIST]
#             # ),
#             # labels=ticks,
#             minor=False,
#             # zorder=10,
#         )

#         axs = [
#             fig.add_axes(
#                 ax0.get_position(),
#                 projection=Plotter.RADAR_FIGURE,
#                 frameon=False,
#                 theta_direction=ax0.get_theta_direction(),
#                 theta_offset=ax0.get_theta_offset(),
#             )
#             for i in range(len(card.classes))
#         ]

#         for a in axs:
#             a.set_axis_off()
#             a.set_rlim(rlim)
#             a.set_visible(False)
#             a.set_animated(True)

#         labels = {}
#         order = [ax0]

#         # difference = int("5" * card.granularity)
#         scaling = 10 ** (card.granularity - 1)

#         for c, combos in card.combinations.items():
#             label = card.classes[c][0]
#             color = figcolors[c]
#             alpha_scale = 0.85 / (card.classes[c][2] ** 0.3)

#             comboarray = combos[:, features]  # - difference

#             for j in range(nfeat):
#                 if j != 0:
#                     prevf = j - 1
#                     temparray = comboarray[:, prevf : j + 1]

#                 else:
#                     prevf = nfeat - 1
#                     temparray = comboarray[:, [prevf, 0]]

#                 minicombos, counts = np.unique(temparray, return_counts=True, axis=0)

#                 final = np.zeros(shape=(minicombos.shape[0], nfeat + 1), dtype=float)

#                 if scaling == 1:
#                     final[:, prevf] = minicombos[:, 0]
#                     final[:, j] = minicombos[:, 1]
#                 else:
#                     final[:, prevf] = minicombos[:, 0] / scaling
#                     final[:, j] = minicombos[:, 1] / scaling

#                 alphas = (counts**0.3) * alpha_scale + 0.1

#                 # Close the lines
#                 final[:, -1] = final[:, 0]

#                 lines = [np.column_stack((theta, combo)) for combo in final]
#                 lc = LineCollection(lines, alpha=alphas, color=color, linewidth=3)
#                 axs[c].add_collection(lc)

#             labels[label] = axs[c]
#             order.append(axs[c])

#         # Finish preparing a plot
#         plotter.finish_plot(
#             animated=True,
#             ax_ordered=order,
#             ax_labels=labels,
#         )

#     return res
