"""spata.model.model_card"""

import json
import numpy as np
from spata.base.card import Card


class ModelCard(Card):
    """ModelCard"""

    CONFIG_Y_QUANTITY_DEFAULT = 9
    CONFIG_Y_INFERRED_GRANULARITY_DEFAULT = 1

    CONFIG_C_NAME_DEFAULT = "C"
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

    KEY_TIFS_T = "t"
    KEY_TIFS_I = "i"
    KEY_TIFS_F = "f"

    VAL_C_INIT_CONFIGURED = "configured"
    VAL_C_INIT_INFERRED = "inferred"

    VALS_C_INIT = (
        VAL_C_INIT_CONFIGURED,
        VAL_C_INIT_INFERRED,
    )

    def __init__(
        self,
        X,
        model=None,
        cnames=None,
        classification=None,
        quantity=None,
        **kwargs,
    ):
        if cnames is not None:
            try:
                cnames = [str(item) for item in cnames]
            except Exception as e:
                raise TypeError(
                    "The 'cnames' argument must be an array-like of feature names"
                ) from e

        if classification is None:
            is_classification = True

        else:
            try:
                is_classification = bool(classification)
            except Exception as e:
                raise TypeError(
                    "The 'classification' argument must be a boolean"
                ) from e

        if quantity is None:
            quantity = self.CONFIG_Y_QUANTITY_DEFAULT

        else:
            try:
                quantity = int(quantity)
            except Exception as e:
                raise TypeError(
                    "The 'quantity' argument must be an integer value"
                ) from e

        # ------------------------------

        try:
            keep_base = bool(kwargs.get("base", False))
        except Exception as e:
            raise TypeError("The 'base' argument must be a boolean") from e

        # Ensure that base Card is keeping inverses of original input
        kwargs["base"] = True
        super().__init__(X, **kwargs)

        # ------------------------------

        if isinstance(X, ModelCard):
            self.__init_from_dict(
                X.info,
                X.classes,
                X.class_regions,
                True,
                cnames,
            )

        elif isinstance(X, dict) and "type" in X and X["type"] == "modelcard":
            X["classes"] = {int(k): kdict for k, kdict in X["classes"].items()}

            self.__init_from_dict(
                X["info"],
                X["classes"],
                X["regions"],
                False,
                cnames,
            )

        else:
            if isinstance(X, dict) and "type" in X and X["type"] != "card":
                raise TypeError(
                    "The dictionary provided in the 'X' argument must be a base Card object"
                )
            self.__init_from_input(model, cnames, is_classification, quantity)

        # ------------------------------

        # Ensure compliance with ModelCard argument for keeping inverses
        if not keep_base and hasattr(self, "base"):
            delattr(self, "base")

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
                    "The 'cnames' argument must be an array-like"
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

    def __init_from_input(
        self,
        model,
        cnames,
        is_classification,
        quantity,
    ):

        if not callable(model):
            raise ValueError(
                "The 'model' argument must be a callable function"
                + " that returns predictions for a 2D array-like of data instances"
            )

        try:
            base_y = model(self.generate(self.base, quantity=quantity))
        except Exception as e:
            raise ValueError(
                "The 'model' function had an error when called"
                + " with a 2D array-like of the original data instances"
            ) from e

        try:
            base_y = np.array(base_y, copy=None, ndmin=1)
        except Exception as e:
            raise TypeError(
                "The 'model' function must return a 1D array-like of predictions in the (n_rows, ) shape"
            ) from e

        if len(base_y.shape) != 1 or base_y.shape[0] != self.info[self.KEY_I_COUNT_X]:
            raise TypeError(
                "The 'model' function must return a 1D array-like of predictions in the (n_rows, ) shape,"
                + " where n_rows matches the function's input"
            )

        perm_X = self.permutate()

        try:
            perm_y = model(self.generate(perm_X, quantity=quantity))
        except Exception as e:
            raise ValueError(
                "The 'model' function had an error when called"
                + " with a 2D array-like of permutated data instances"
            ) from e

        try:
            perm_y = np.array(perm_y, copy=None, ndmin=1)
        except Exception as e:
            raise TypeError(
                "The 'model' function must return a 1D array-like of predictions in the (n_rows, ) shape"
            ) from e

        if len(perm_y.shape) != 1 or perm_y.shape[0] != len(perm_X):
            raise TypeError(
                "The 'model' function must return a 1D array-like of predictions in the (n_rows, ) shape,"
                + " where n_rows matches the function's input"
            )

        final_X = self.base + perm_X
        final_y = np.concatenate(base_y, perm_y)

        # ------------------------------

        if is_classification:
            # For classification tasks
            try:
                ynames, final_y, ycounts = np.unique(
                    final_y, return_inverse=True, return_counts=True
                )
            except Exception as e:
                raise TypeError(
                    "The 'y' argument must be a 1D array-like in the (n_rows, ) shape"
                ) from e

            final_y = final_y.astype(int)

            if cnames is None:
                # With y, using its values as class names
                try:
                    cnames = [str(yname.item()) for yname in ynames]
                except Exception as e:
                    raise TypeError(
                        "The 'y' argument must be a 1D array-like with adequate class labels"
                    ) from e

            else:
                # With y, using provided class names
                if len(cnames) != len(ynames):
                    raise ValueError(
                        "The 'cnames' argument must be an array-like"
                        + " of class names in the (n_classes, ) shape,"
                        + " where n_classes matches the 'y' argument"
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
                minmin = final_y.min().item()

                final_y = self.__analyze_recursive(
                    np.zeros(final_y.shape, dtype=self.info[self.KEY_I_STORE_DTYPE]),
                    final_y,
                    ydict,
                    minmin,
                    minmin,
                    final_y.max().item(),
                    self.CONFIG_Y_INFERRED_GRANULARITY_DEFAULT,
                    0,
                )

            except Exception as e:
                raise TypeError(
                    "The 'y' argument must contain continuous values when the 'classification' argument is False"
                ) from e

            self.classes = {
                k: {
                    self.KEY_C_NAME: f"{self.CONFIG_C_NAME_DEFAULT}{str(k)}"[
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

        for i, r in enumerate(final_X):
            k = final_y[i]

            if r in self.class_regions[k]:
                self.class_regions[k][r][self.KEY_CR_COUNT_X] += 1

            else:
                self.class_regions[k][r] = {self.KEY_CR_COUNT_X: 1}

                if r < self.classes[k][self.KEY_C_MIN_R]:
                    self.classes[k][self.KEY_C_MIN_R] = r

                elif r > self.classes[k][self.KEY_C_MAX_R]:
                    self.classes[k][self.KEY_C_MAX_R] = r

        for k, kdict in self.classes.items():
            kdict[self.KEY_C_COUNT_R] = len(self.class_regions[k])

    def __call__(self, X, out_by_tuple=False):
        try:
            out_by_tuple = bool(out_by_tuple)
        except Exception as e:
            raise TypeError("The 'out_by_tuple' argument must be a boolean") from e

        # The out_by_tuple argument to the convert function must always be True here
        X = self.convert(X, out_by_tuple=True)

        if out_by_tuple:
            res = []
            for r in X:
                for k, kdict in self.class_regions.items():
                    if r in kdict:
                        res.append(k)
                        break

        else:
            res = np.empty(shape=(len(X),), dtype=int)
            for i, r in enumerate(X):
                for k, kdict in self.class_regions.items():
                    if r in kdict:
                        res[i] = k
                        break

        return res

    def predict(self, X, out_by_tuple=False):
        return self(X, out_by_tuple)

    def save(self, filepath=None):
        res = super().save(filepath=None)

        res["type"] = "datacard"
        res["hexdigest"]["datacard"] = self.__hexdigest()

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
                    "This Card could not be saved to the 'filepath' argument"
                ) from e

        return res

    def summary(self, plotter=None, features=None, classes=None):

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
                    "The 'classes' argument must be an array-like of class indexes"
                ) from e

            if len(classes) == 0:
                raise ValueError("The 'classes' argument must not be empty")

            for k in classes:
                if k not in self.classes:
                    raise ValueError(
                        "The 'classes' argument contains invalid class indexes"
                    )

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
                    "The 'features' argument must be an array-like of feature indexes"
                ) from e

            if len(features) == 0:
                raise ValueError("The 'features' argument must not be empty")

            for j in features:
                if j not in self.features:
                    raise ValueError(
                        "The 'features' argument contains invalid feature indexes"
                    )

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
                vertice_labels=[self.features[j][self.KEY_F_NAME] for j in features],
                button_labels=[self.classes[k][self.KEY_C_NAME] for k in classes],
                verify=False,
            )

            return miniregions, fig

        return miniregions
