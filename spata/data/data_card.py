"""spata.data.data_card"""

import json
import numpy as np
from spata.base.card import Card


class DataCard(Card):
    """DataCard"""

    CONFIG_Y_INFERRED_GRANULARITY_DEFAULT = 1

    CONFIG_C_NAME_DEFAULT = "Class"
    CONFIG_C_NAME_LENGTH = 20

    KEY_I_COUNT_C = "n_classes"

    KEY_C_INIT = "init"
    KEY_C_NAME = "name"
    KEY_C_COUNT_X = "n_inverses"
    KEY_C_COUNT_R = "n_regions"
    KEY_C_MIN_R = "min_region"
    KEY_C_MAX_R = "max_region"

    KEY_CR_COUNT_X = "n_inverses"

    KEY_R_COUNT_CR_X = "class_regions"

    VAL_C_INIT_CONFIGURED = "configured"
    VAL_C_INIT_INFERRED = "inferred"

    VALS_C_INIT = (
        VAL_C_INIT_CONFIGURED,
        VAL_C_INIT_INFERRED,
    )

    def __init__(
        self,
        X,
        y=None,
        cnames=None,
        classification=None,
        **kwargs,
    ):
        if cnames is not None:
            try:
                cnames = [str(item) for item in cnames]
            except Exception as e:
                raise TypeError(
                    "The 'cnames' parameter must be an array-like of feature names"
                ) from e

        if classification is None:
            is_classification = True

        else:
            try:
                is_classification = bool(classification)
            except Exception as e:
                raise TypeError(
                    "The 'classification' parameter must be a boolean"
                ) from e

        # ------------------------------

        try:
            keep_inverses = bool(kwargs.get("inverses", False))
        except Exception as e:
            raise TypeError("The 'inverses' parameter must be a boolean") from e

        # Ensure that base Card is keeping inverses of original input
        kwargs["inverses"] = True
        super().__init__(X, **kwargs)

        # ------------------------------

        if self.info[self.KEY_I_INIT] == self.VAL_I_INIT_BYCLONE:
            if not isinstance(X, DataCard):
                raise TypeError("Provided Card is not a DataCard")

            self.__init_from_dict(
                X.info,
                X.classes,
                X.class_regions,
                True,
                cnames,
            )

        elif self.info[self.KEY_I_INIT] == self.VAL_I_INIT_BYDICT:
            # if "classes" not in X:
            #     raise ValueError(
            #         "Card cannot be loaded from dict because it has an incompatible format"
            #     )

            X["classes"] = {int(k): kdict for k, kdict in X["classes"].items()}

            # if "class_regions" in X:
            #     self.__init_from_dict(
            #         X["info"],
            #         X["classes"],
            #         X["class_regions"],
            #         True,
            #         cnames,
            #     )
            # else:

            self.__init_from_dict(
                X["info"],
                X["classes"],
                X["regions"],
                False,
                cnames,
            )

        elif self.info[self.KEY_I_INIT] == self.VAL_I_INIT_BYX:

            self.__init_from_input(y, cnames, is_classification)

        else:
            raise TypeError("Provided Card is not supported")

        # ------------------------------

        # Ensure compliance with DataCard argument for keeping inverses
        if not keep_inverses and hasattr(self, "inverses"):
            delattr(self, "inverses")

    def __hexdigest(self):
        if not hasattr(self, "__hxdgst"):
            from hashlib import sha3_512

            str_regions = {str(r): rdict for r, rdict in self.regions.items()}

            str_class_regions = {
                k: {str(r): rdict for r, rdict in krdict.items()}
                for k, krdict in self.class_regions.items()
            }

            a = sha3_512()
            a.update(json.dumps(self.projections, sort_keys=True).encode("utf-8"))
            a.update(json.dumps(str_regions, sort_keys=True).encode("utf-8"))
            a.update(json.dumps(str_class_regions, sort_keys=True).encode("utf-8"))

            self.__hxdgst = a.hexdigest()

        return self.__hxdgst

    def save(self, filepath=None):
        res = super().save(filepath=None)

        res["hexdigest"]["data_card"] = self.__hexdigest()

        res["info"][self.KEY_I_COUNT_C] = self.info[self.KEY_I_COUNT_C]

        res["classes"] = {
            k: {
                self.KEY_C_INIT: kdict[self.KEY_C_INIT],
                self.KEY_C_NAME: kdict[self.KEY_C_NAME],
                self.KEY_C_COUNT_X: kdict[self.KEY_C_COUNT_X],
                self.KEY_C_COUNT_R: kdict[self.KEY_C_COUNT_R],
                self.KEY_C_MIN_R: str(kdict[self.KEY_C_MIN_R]),
                self.KEY_C_MAX_R: str(kdict[self.KEY_C_MAX_R]),
            }
            for k, kdict in self.classes.items()
        }

        for k, krdict in self.class_regions.items():
            for r, rdict in krdict.items():
                r = str(r)

                if self.KEY_R_COUNT_CR_X in res["regions"][r]:
                    res["regions"][r][self.KEY_R_COUNT_CR_X][k] = {
                        self.KEY_CR_COUNT_X: rdict[self.KEY_CR_COUNT_X]
                    }

                else:
                    res["regions"][r][self.KEY_R_COUNT_CR_X] = {
                        k: {self.KEY_CR_COUNT_X: rdict[self.KEY_CR_COUNT_X]}
                    }

        if filepath is not None:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(res, f, sort_keys=True)

            except Exception as e:
                raise ValueError(
                    "This Card could not be saved to the 'filepath' parameter"
                ) from e

        return res

    def __init_from_dict(
        self,
        X_info,
        X_classes,
        X_class_regions,
        is_class_regions_standalone,
        cnames,
    ):

        if cnames is None:
            self.classes = {
                k: {
                    self.KEY_C_NAME: kdict[self.KEY_C_NAME][
                        : self.CONFIG_C_NAME_LENGTH
                    ],
                    self.KEY_C_INIT: kdict[self.KEY_C_INIT],
                    self.KEY_C_COUNT_X: kdict[self.KEY_C_COUNT_X],
                }
                for k, kdict in X_classes.items()
            }

        else:
            if len(cnames) != len(X_classes):
                raise ValueError(
                    "The 'cnames' parameter must be an array-like"
                    + " of class names in the (n_classes, ) shape,"
                    + " where n_classes matches the provided Card"
                )

            self.classes = {
                k: {
                    self.KEY_C_NAME: cnames[: self.CONFIG_C_NAME_LENGTH],
                    self.KEY_C_INIT: kdict[self.KEY_C_INIT],
                    self.KEY_C_COUNT_X: kdict[self.KEY_C_COUNT_X],
                }
                for k, kdict in X_classes.items()
            }

        self.info[self.KEY_I_COUNT_C] = X_info[self.KEY_I_COUNT_C]

        # ------------------------------

        self.class_regions = {k: {} for k in self.classes}

        if self._downscaling == 1:
            if is_class_regions_standalone:
                for k, krdict in X_class_regions.items():

                    self.class_regions[k] = {
                        r: {self.KEY_CR_COUNT_X: rdict[self.KEY_CR_COUNT_X]}
                        for r, rdict in krdict.items()
                    }

                for k, kdict in self.classes.items():

                    kdict[self.KEY_C_MIN_R] = X_classes[k][self.KEY_C_MIN_R]
                    kdict[self.KEY_C_MAX_R] = X_classes[k][self.KEY_C_MAX_R]
                    kdict[self.KEY_C_COUNT_R] = X_classes[k][self.KEY_C_COUNT_R]

            else:
                for r, rdict in X_class_regions.items():
                    for k, krdict in rdict[self.KEY_R_COUNT_CR_X].items():

                        self.class_regions[int(k)][r] = {
                            self.KEY_CR_COUNT_X: krdict[self.KEY_CR_COUNT_X]
                        }

                for k, kdict in self.classes.items():

                    kdict[self.KEY_C_MIN_R] = tuple(
                        int(val)
                        for val in str(X_classes[k][self.KEY_C_MIN_R])[1:-1].split(",")
                    )
                    kdict[self.KEY_C_MAX_R] = tuple(
                        int(val)
                        for val in str(X_classes[k][self.KEY_C_MAX_R])[1:-1].split(",")
                    )
                    kdict[self.KEY_C_COUNT_R] = X_classes[k][self.KEY_C_COUNT_R]

        else:
            if is_class_regions_standalone:
                for k, krdict in X_class_regions.items():
                    for r, rdict in krdict.items():
                        rnew = tuple([int(p / self._downscaling) for p in r])

                        if rnew in self.class_regions[k]:
                            self.class_regions[k][rnew][self.KEY_CR_COUNT_X] += rdict[
                                self.KEY_CR_COUNT_X
                            ]
                        else:
                            self.class_regions[k][rnew] = {
                                self.KEY_CR_COUNT_X: rdict[self.KEY_CR_COUNT_X]
                            }

                for k, kdict in self.classes.items():

                    kdict[self.KEY_C_MIN_R] = tuple(
                        [
                            int(p / self._downscaling)
                            for p in X_classes[k][self.KEY_C_MIN_R]
                        ]
                    )
                    kdict[self.KEY_C_MAX_R] = tuple(
                        [
                            int(p / self._downscaling)
                            for p in X_classes[k][self.KEY_C_MAX_R]
                        ]
                    )
                    kdict[self.KEY_C_COUNT_R] = len(self.class_regions[k])

            else:
                for r, rdict in X_class_regions.items():
                    rnew = tuple([int(p / self._downscaling) for p in r])

                    for k, krdict in rdict[self.KEY_R_COUNT_CR_X].items():
                        k = int(k)

                        if rnew in self.class_regions[k]:
                            self.class_regions[k][rnew][self.KEY_CR_COUNT_X] += krdict[
                                self.KEY_CR_COUNT_X
                            ]
                        else:
                            self.class_regions[k][rnew] = {
                                self.KEY_CR_COUNT_X: krdict[self.KEY_CR_COUNT_X]
                            }

                for k, kdict in self.classes.items():

                    kdict[self.KEY_C_MIN_R] = tuple(
                        [
                            int(int(p) / self._downscaling)
                            for p in str(X_classes[k][self.KEY_C_MIN_R])[1:-1].split(
                                ","
                            )
                        ]
                    )
                    kdict[self.KEY_C_MAX_R] = tuple(
                        [
                            int(int(p) / self._downscaling)
                            for p in str(X_classes[k][self.KEY_C_MAX_R])[1:-1].split(
                                ","
                            )
                        ]
                    )
                    kdict[self.KEY_C_COUNT_R] = len(self.class_regions[k])

    def __init_from_input(self, y, cnames, is_classification):

        ## Setup y and 'classes' dictionary
        if y is None:
            # Without y
            y = np.zeros(self.info[self.KEY_I_COUNT_X], dtype=int)

            if cnames is None:
                # Without y, with default class name
                cnames = [f"{self.CONFIG_C_NAME_DEFAULT} 0"]

            else:
                # Without y, with provided one-class name
                if len(cnames) != 1:
                    raise ValueError(
                        "When the 'y' parameter is not provided,"
                        + " the 'cnames' parameter must be an array-like"
                        + " containing just one class name"
                    )

            self.classes = {
                0: {
                    self.KEY_C_NAME: cnames[0][: self.CONFIG_C_NAME_LENGTH],
                    self.KEY_C_INIT: self.VAL_C_INIT_INFERRED,
                    self.KEY_C_COUNT_X: self.info[self.KEY_I_COUNT_X],
                }
            }

        else:
            # With y
            try:
                y = np.array(y, copy=None, ndmin=1)
            except Exception as e:
                raise TypeError(
                    "The 'y' parameter must be a 1D array-like in the (n_rows, ) shape"
                ) from e

            if len(y.shape) != 1 or y.shape[0] != self.info[self.KEY_I_COUNT_X]:
                raise TypeError(
                    "The 'y' parameter must be a 1D array-like in the (n_rows, ) shape,"
                    + " where n_rows matches the 'X' parameter"
                )

            if is_classification:
                # For classification tasks
                try:
                    ynames, y, ycounts = np.unique(
                        y, return_inverse=True, return_counts=True
                    )
                except Exception as e:
                    raise TypeError(
                        "The 'y' parameter must be a 1D array-like in the (n_rows, ) shape"
                    ) from e

                y = y.astype(int)

                if cnames is None:
                    # With y, using its values as class names
                    try:
                        cnames = [str(yname.item()) for yname in ynames]
                    except Exception as e:
                        raise TypeError(
                            "The 'y' parameter must be a 1D array-like with adequate class labels"
                        ) from e

                else:
                    # With y, using provided class names
                    if len(cnames) != len(ynames):
                        raise ValueError(
                            "The 'cnames' parameter must be an array-like"
                            + " of class names in the (n_classes, ) shape,"
                            + " where n_classes matches the 'y' parameter"
                        )

                self.classes = {
                    k: {
                        self.KEY_C_NAME: cname[: self.CONFIG_C_NAME_LENGTH],
                        self.KEY_C_INIT: self.VAL_C_INIT_CONFIGURED,
                        self.KEY_C_COUNT_X: ycounts[k].item(),
                    }
                    for k, cname in enumerate(cnames)
                }

            else:
                # For regression tasks
                ydict = {}

                try:
                    minmin = y.min().item()

                    y = self.__analyze_recursive(
                        np.zeros(y.shape, dtype=self.info[self.KEY_I_STORE_DTYPE]),
                        y,
                        ydict,
                        minmin,
                        minmin,
                        y.max().item(),
                        self.CONFIG_Y_INFERRED_GRANULARITY_DEFAULT,
                        0,
                    )

                except Exception as e:
                    raise TypeError(
                        "The 'y' parameter must contain continuous values when the 'continuous' parameter is used"
                    ) from e

                self.classes = {
                    k: {
                        self.KEY_C_NAME: f"{self.CONFIG_C_NAME_DEFAULT} {str(k)}"[
                            : self.CONFIG_C_NAME_LENGTH
                        ]
                        + f" [{str(kdict[self.KEY_P_BIN_MIN])}, {str(kdict[self.KEY_P_BIN_MAX])}]",
                        self.KEY_C_INIT: self.VAL_C_INIT_INFERRED,
                        self.KEY_C_COUNT_X: kdict[self.KEY_P_COUNT_X],
                    }
                    for k, kdict in ydict.items()
                }

        self.info[self.KEY_I_COUNT_C] = len(self.classes)

        # ------------------------------

        self.class_regions = {k: {} for k in self.classes}

        for k, kdict in self.classes.items():
            kdict[self.KEY_C_MIN_R] = self._max_region
            kdict[self.KEY_C_MAX_R] = self._min_region

        for i, r in enumerate(self.inverses):
            k = y[i]

            if r in self.class_regions[k]:
                self.class_regions[k][r][self.KEY_CR_COUNT_X] += 1

            else:
                self.class_regions[k][r] = {self.KEY_CR_COUNT_X: 1}

                if r < self.classes[k][self.KEY_C_MIN_R]:
                    self.classes[k][self.KEY_C_MIN_R] = r

                elif r > self.classes[k][self.KEY_C_MAX_R]:
                    self.classes[k][self.KEY_C_MAX_R] = r

        # for i, r in enumerate(self.inverses):
        #     k = y[i]

        #     if r in self.class_regions[k]:
        #         self.regions[r][self.KEY_CR_COUNT_X][k] += 1

        #     else:
        #         self.class_regions[k].append(r)

        #         if self.KEY_CR_COUNT_X in self.regions[r]:
        #             if k in self.regions[r][self.KEY_CR_COUNT_X]:
        #                 self.regions[r][self.KEY_CR_COUNT_X][k] += 1
        #             else:
        #                 self.regions[r][self.KEY_CR_COUNT_X][k] = 1
        #         else:
        #             self.regions[r][self.KEY_CR_COUNT_X] = {k: 1}

        #         if r < self.classes[k][self.KEY_C_MIN_R]:
        #             self.classes[k][self.KEY_C_MIN_R] = r

        #         elif r > self.classes[k][self.KEY_C_MAX_R]:
        #             self.classes[k][self.KEY_C_MAX_R] = r

        for k, kdict in self.classes.items():
            kdict[self.KEY_C_COUNT_R] = len(self.class_regions[k])

            # freq_r = None
            # freq_max = 0

            # for r in self.class_regions[k]:
            #     freq_val = self.regions[r][self.KEY_R_COUNT_X_BY_C][k]

            #     if freq_val > freq_max:
            #         freq_r = r

            # self.classes[k][self.KEY_C_FREQ_R] = freq_r

    def summary(self, features=None, classes=None, plotter=None):

        if plotter is not None:
            from spata.base.plotter import Plotter

            if not isinstance(plotter, Plotter):
                try:
                    plotter = bool(plotter)
                except Exception as e:
                    raise TypeError(
                        "The 'plotter' argument must be a Plotter object or a boolean"
                    ) from e

                if plotter:
                    plotter = Plotter()

        if classes is None:
            classes = [k for k in range(len(self.classes))]
        else:
            try:
                classes = [int(item) for item in classes]
            except Exception as e:
                raise TypeError(
                    "The 'classes' parameter must be an array-like of class indexes"
                ) from e

            if len(classes) == 0:
                raise ValueError("The 'classes' parameter must not be empty") from e

            for k in classes:
                if k not in self.classes:
                    raise ValueError(
                        "The 'classes' parameter contains invalid class indexes"
                    ) from e

        if features is None:
            features = [j for j in range(len(self.features))]

            miniregions = {
                k: {
                    r: rdict[self.KEY_CR_COUNT_X]
                    for r, rdict in self.class_regions[k].items()
                }
                for k in classes
            }

        else:
            try:
                features = [int(item) for item in features]
            except Exception as e:
                raise TypeError(
                    "The 'features' parameter must be an array-like of feature indexes"
                ) from e

            if len(features) == 0:
                raise ValueError("The 'features' parameter must not be empty") from e

            for j in features:
                if j not in self.features:
                    raise ValueError(
                        "The 'features' parameter contains invalid feature indexes"
                    ) from e

            miniregions = {k: {} for k in classes}

            for k in classes:
                for r, rdict in self.class_regions[k].items():
                    tup = tuple(r[j] for j in features)

                    if tup in miniregions[k]:
                        miniregions[k][tup] += rdict[self.KEY_CR_COUNT_X]
                    else:
                        miniregions[k][tup] = rdict[self.KEY_CR_COUNT_X]

        if plotter is not None:
            alpha_scaling = 0.8 / self.info[self.KEY_I_COUNT_X]

            minilines = {k: {} for k in classes}

            for k in classes:
                for r, val in miniregions[k].items():
                    for minij in range(len(features)):
                        minitup = [0 for _ in range(len(features))]

                        if minij == 0:
                            minitup[0] = r[0]
                            minitup[-1] = r[-1]
                        else:
                            minitup[minij - 1] = r[minij - 1]
                            minitup[minij] = r[minij]

                        minitup = tuple(minitup)

                        if minitup in minilines[k]:
                            minilines[k][minitup] += val
                        else:
                            minilines[k][minitup] = val

            try:
                fig = plotter(
                    lines=[
                        [[item / self._scaling for item in tup] for tup in krdict]
                        for krdict in minilines.values()
                    ],
                    alphas=[
                        [item * alpha_scaling + 0.2 for item in krdict.values()]
                        for krdict in minilines.values()
                    ],
                    # alphas=[
                    #     [
                    #         (item * 0.8 / self.classes[k][self.KEY_C_COUNT_X]) + 0.2
                    #         for item in krdict.values()
                    #     ]
                    #     for k, krdict in minilines.items()
                    # ],
                    vertice_labels=[
                        self.features[j][self.KEY_F_NAME] for j in features
                    ],
                    button_labels=[self.classes[k][self.KEY_C_NAME] for k in classes],
                    button_colors=None,
                    tick_labels=self.CONFIG_PLT_TICK_LABELS,
                    tick_values=self.CONFIG_PLT_TICK_VALUES,
                    axis_limits=self.CONFIG_PLT_AXIS_LIMITS,
                    verify=False,
                )
            except Exception as e:
                raise ValueError("The call to the 'plotter' argument failed") from e

            return miniregions, fig

        return miniregions
