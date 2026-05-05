"""spata.base.card"""

import time
import json
import numpy as np


class Card:
    """Card"""

    CONFIG_I_GRANULARITY_DEFAULT = 2

    CONFIG_I_NAME_DEFAULT = "Card"
    CONFIG_I_NAME_LENGTH = 100

    CONFIG_F_NAME_DEFAULT = "Feature"
    CONFIG_F_NAME_LENGTH = 20

    CONFIG_PLT_AXIS_LIMITS = (0.01, 1.05)
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

    KEY_I_TIMESTAMP = "timestamp"
    KEY_I_INIT = "init"
    KEY_I_NAME = "name"
    KEY_I_GRANULARITY = "granularity"
    KEY_I_STORE_DTYPE = "optimized_dtype"
    KEY_I_SAFE_DTYPE = "safe_dtype"
    KEY_I_COUNT_F = "n_features"
    KEY_I_COUNT_P = "n_projections"
    KEY_I_COUNT_X = "n_inverses"
    KEY_I_COUNT_R = "n_regions"

    KEY_F_INIT = "init"
    KEY_F_NAME = "name"
    KEY_F_TYPE = "type"
    KEY_F_DTYPE = "dtype"
    KEY_F_COUNT_P = "n_projections"
    KEY_F_MIN_P = "min_projection"
    KEY_F_MAX_P = "max_projection"

    KEY_P_BIN_MIN = "bin"
    KEY_P_BIN_MAX = "bin"
    KEY_P_COUNT_X = "n_inverses"
    KEY_P_COUNT_R = "n_regions"

    KEY_R_COUNT_X = "n_inverses"

    VAL_I_INIT_BYCLONE = "clone"
    VAL_I_INIT_BYDICT = "dict"
    VAL_I_INIT_BYX = "input"

    VALS_I_INIT = (
        VAL_I_INIT_BYCLONE,
        VAL_I_INIT_BYDICT,
        VAL_I_INIT_BYX,
    )

    VAL_F_INIT_CONFIGURED = "configured"
    VAL_F_INIT_INFERRED = "inferred"

    VALS_F_INIT = (
        VAL_F_INIT_CONFIGURED,
        VAL_F_INIT_INFERRED,
    )

    VAL_F_TYPE_CONTINUOUS = "continuous"
    VAL_F_TYPE_DISCRETE = "discrete"
    VAL_F_TYPE_BOOLEAN = "boolean"
    VAL_F_TYPE_CATEGORICAL = "categorical"

    VALS_F_TYPE = (
        VAL_F_TYPE_CONTINUOUS,
        VAL_F_TYPE_DISCRETE,
        VAL_F_TYPE_BOOLEAN,
        VAL_F_TYPE_CATEGORICAL,
    )

    VAL_F_DTYPE_CONTINUOUS = float
    VAL_F_DTYPE_DISCRETE = int
    VAL_F_DTYPE_BOOLEAN = int
    VAL_F_DTYPE_CATEGORICAL = int

    VALS_F_DTYPE = (
        VAL_F_DTYPE_CONTINUOUS,
        VAL_F_DTYPE_DISCRETE,
        VAL_F_DTYPE_BOOLEAN,
        VAL_F_DTYPE_CATEGORICAL,
    )

    BIN_1 = 1
    BIN_2 = 2
    BIN_3 = 3
    BIN_4 = 4
    BIN_5 = 5
    BIN_6 = 6
    BIN_7 = 7
    BIN_8 = 8
    BIN_9 = 9

    BINS_ASC = (BIN_1, BIN_2, BIN_3, BIN_4, BIN_5, BIN_6, BIN_7, BIN_8, BIN_9)
    BINS_DESC = (BIN_9, BIN_8, BIN_7, BIN_6, BIN_5, BIN_4, BIN_3, BIN_2, BIN_1)

    def __init__(
        self,
        X,
        granularity=None,
        name=None,
        fnames=None,
        fdtypes=None,
        inverses=None,
        random_state=None,
        n_jobs=None,
    ):
        self.info = {self.KEY_I_TIMESTAMP: int(time.time())}

        # ------------------------------

        if granularity is not None:
            try:
                granularity = int(granularity)
            except Exception as e:
                raise TypeError(
                    "The 'granularity' argument must be an integer value"
                ) from e

        if name is not None:
            try:
                name = str(name)
            except Exception as e:
                raise TypeError(
                    "The 'name' parameter must be a short name for this Card"
                ) from e

        if fnames is not None:
            try:
                fnames = [str(item) for item in fnames]
            except Exception as e:
                raise TypeError(
                    "The 'fnames' parameter must be an array-like of feature names"
                ) from e

        if fdtypes is not None:
            try:
                fdtypes = [np.dtype(item) for item in fdtypes]
            except Exception as e:
                raise TypeError(
                    "The 'fnames' parameter must be an array-like of feature names"
                ) from e

            if fnames is not None and len(fdtypes) != len(fnames):
                raise TypeError(
                    "The 'fdtypes' parameter must be an array-like of feature dtypes"
                    + "  in the (n_features, ) shape, where n_features matches the 'fnames' parameter"
                )

        if inverses is None:
            keep_inverses = False

        else:
            try:
                keep_inverses = bool(inverses)
            except Exception as e:
                raise TypeError("The 'inverses' parameter must be a boolean") from e

        try:
            self.random_state = np.random.default_rng(random_state)
        except Exception as e:
            raise TypeError(
                "The 'random_state' argument must be an integer value to enable reproducibility,"
                + " a numpy Generator object to use it unaltered,"
                + " or None to use pseudo-random numbers"
            ) from e

        if n_jobs is None:
            self.n_jobs = None

        else:
            try:
                self.n_jobs = int(n_jobs)
            except Exception as e:
                raise TypeError(
                    "The 'n_jobs' argument must be an integer value to enable parallel processing,"
                    + " or 1 to disable paralellism"
                ) from e

        # ------------------------------

        if isinstance(X, Card):
            self.info[self.KEY_I_INIT] = self.VAL_I_INIT_BYCLONE

            try:
                self.__init_from_dict(
                    X.info,
                    X.features,
                    X.projections,
                    X.regions,
                    granularity,
                    name,
                    fnames,
                    fdtypes,
                )
                # X.inverses if hasattr(X, "inverses") else None,
            except Exception as e:
                raise ValueError("Error while cloning a Card") from e

        elif isinstance(X, dict):
            self.info[self.KEY_I_INIT] = self.VAL_I_INIT_BYDICT

            X["features"] = {int(j): jdict for j, jdict in X["features"].items()}

            X["projections"] = {
                int(j): {int(p): pdict for p, pdict in jpdict.items()}
                for j, jpdict in X["projections"].items()
            }

            X["regions"] = {
                tuple(int(val) for val in str(r)[1:-1].split(",")): rdict
                for r, rdict in X["regions"].items()
            }

            try:
                self.__init_from_dict(
                    X["info"],
                    X["features"],
                    X["projections"],
                    X["regions"],
                    granularity,
                    name,
                    fnames,
                    fdtypes,
                )
                # X["inverses"] if "inverses" in X else None,
            except Exception as e:
                raise ValueError("Error while loading a Card from a dict") from e

        else:
            self.info[self.KEY_I_INIT] = self.VAL_I_INIT_BYX

            try:
                self.__init_from_input(
                    X,
                    granularity,
                    name,
                    fnames,
                    fdtypes,
                    keep_inverses,
                )
            except Exception as e:
                raise ValueError("Error while creating a Card from input") from e

        # ------------------------------

        self._min_region = tuple(
            self._min_projection for _ in range(len(self.features))
        )
        self._max_region = tuple(
            self._max_projection for _ in range(len(self.features))
        )

    def __str__(self):
        return f"{self.__class__.__name__}({str(self.info)[1:-1]})"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if self.__hexdigest() != other.__hexdigest():
            return False

        return True

    def __hexdigest(self):
        if not hasattr(self, "__hxdgst"):
            from hashlib import sha3_512

            str_regions = {str(r): rdict for r, rdict in self.regions.items()}

            a = sha3_512()
            a.update(json.dumps(self.projections, sort_keys=True).encode("utf-8"))
            a.update(json.dumps(str_regions, sort_keys=True).encode("utf-8"))

            self.__hxdgst = a.hexdigest()

        return self.__hxdgst

    def save(self, filepath=None):
        res = {
            "hexdigest": {"card": self.__hexdigest()},
            "info": {
                self.KEY_I_TIMESTAMP: self.info[self.KEY_I_TIMESTAMP],
                self.KEY_I_INIT: self.info[self.KEY_I_INIT],
                self.KEY_I_NAME: self.info[self.KEY_I_NAME],
                self.KEY_I_GRANULARITY: self.info[self.KEY_I_GRANULARITY],
                self.KEY_I_STORE_DTYPE: self.info[self.KEY_I_STORE_DTYPE],
                self.KEY_I_SAFE_DTYPE: self.info[self.KEY_I_SAFE_DTYPE],
                self.KEY_I_COUNT_F: self.info[self.KEY_I_COUNT_F],
                self.KEY_I_COUNT_P: self.info[self.KEY_I_COUNT_P],
                self.KEY_I_COUNT_X: self.info[self.KEY_I_COUNT_X],
                self.KEY_I_COUNT_R: self.info[self.KEY_I_COUNT_R],
            },
            "features": {
                j: {
                    self.KEY_F_INIT: jdict[self.KEY_F_INIT],
                    self.KEY_F_NAME: jdict[self.KEY_F_NAME],
                    self.KEY_F_TYPE: jdict[self.KEY_F_TYPE],
                    self.KEY_F_DTYPE: jdict[self.KEY_F_DTYPE],
                    self.KEY_F_COUNT_P: jdict[self.KEY_F_COUNT_P],
                    self.KEY_F_MIN_P: jdict[self.KEY_F_MIN_P],
                    self.KEY_F_MAX_P: jdict[self.KEY_F_MAX_P],
                }
                for j, jdict in self.features.items()
            },
            "projections": {
                j: {
                    p: {
                        self.KEY_P_BIN_MIN: pdict[self.KEY_P_BIN_MIN],
                        self.KEY_P_BIN_MAX: pdict[self.KEY_P_BIN_MAX],
                        self.KEY_P_COUNT_X: pdict[self.KEY_P_COUNT_X],
                        self.KEY_P_COUNT_R: pdict[self.KEY_P_COUNT_R],
                    }
                    for p, pdict in jpdict.items()
                }
                for j, jpdict in self.projections.items()
            },
            "regions": {
                str(r): {self.KEY_R_COUNT_X: rdict[self.KEY_R_COUNT_X]}
                for r, rdict in self.regions.items()
            },
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

    def convert(
        self,
        X,
        out_by_feature=False,
        out_by_tuple=False,
        out_normalized=False,
        out_standardized=False,
    ):
        try:
            out_by_feature = bool(out_by_feature)
        except Exception as e:
            raise TypeError("The 'out_by_feature' parameter must be a boolean") from e

        try:
            out_by_tuple = bool(out_by_tuple)
        except Exception as e:
            raise TypeError("The 'out_by_tuple' parameter must be a boolean") from e

        if out_by_feature and out_by_tuple:
            raise ValueError(
                "The 'out_normalized' and 'out_standardized' parameters cannot be used at the same time"
            )

        try:
            out_normalized = bool(out_normalized)
        except Exception as e:
            raise TypeError("The 'out_normalized' parameter must be a boolean") from e

        try:
            out_standardized = bool(out_standardized)
        except Exception as e:
            raise TypeError("The 'out_standardized' parameter must be a boolean") from e

        if out_normalized and out_standardized:
            raise ValueError(
                "The 'out_normalized' and 'out_standardized' parameters cannot be used at the same time"
            )

        Xbycols, nrows, ncols = self.__prepare_input(X)

        if ncols < len(self.features):
            raise ValueError(
                "The array-like provided in 'X' contains less"
                + " features than this Card"
            )
        elif ncols > len(self.features):
            raise ValueError(
                "The array-like provided in 'X' contains more"
                + " features than this Card"
            )

        if out_by_feature or out_by_tuple:
            res = []
        else:
            res = np.empty((nrows, ncols), dtype=self.info[self.KEY_I_SAFE_DTYPE])

        for j in range(ncols):
            if Xbycols[j].dtype != self.features[j][self.KEY_F_DTYPE]:
                try:
                    Xbycols[j] = Xbycols[j].astype(
                        dtype=self.features[j][self.KEY_F_DTYPE]
                    )
                except Exception as e:
                    raise ValueError(
                        f"The 'X' parameter contains numpy dtypes incompatible with this Card in feature {j}"
                    ) from e

            if out_by_feature or out_by_tuple:
                res.append(self.__convert_feature(Xbycols[j], j))
            else:
                res[:, j] = self.__convert_feature(Xbycols[j], j)

        if out_by_feature or out_by_tuple:
            if out_normalized:
                res = [farray / self._scaling for farray in res]

            elif out_standardized:
                res = [
                    (farray - self._center_projection) / self._scaling for farray in res
                ]
        else:
            if out_normalized:
                res = res / self._scaling

            elif out_standardized:
                res = (res - self._center_projection) / self._scaling

        if out_by_tuple:
            res = [tuple(p.item() for p in tup) for tup in zip(res)]

        return res

    def __prepare_info(self, granularity, name, prevgranularity=None, prevname=None):
        if name is None:
            if prevname is None:
                name = self.CONFIG_I_NAME_DEFAULT
            else:
                name = prevname

        self.info[self.KEY_I_NAME] = name[: self.CONFIG_I_NAME_LENGTH]

        # self.info[self.KEY_I_TYPE] = self.__class__.__name__

        if granularity is None:
            if prevgranularity is None:
                granularity = self.CONFIG_I_GRANULARITY_DEFAULT
            else:
                granularity = prevgranularity

        if granularity < 1 or granularity > 8:
            raise ValueError(
                "The 'granularity' argument must be a value from 1 up to 8"
                + ", representing codes from 1 digit [1, 9] up to 8 digits [11111111, 99999999]"
            )

        if prevgranularity is None or prevgranularity == granularity:
            self._downscaling = 1

        else:
            if prevgranularity < 1 or prevgranularity > 8:
                raise ValueError(
                    "The previous 'granularity' is not accepted"
                    + ", as it must be a value from 1 up to 8"
                )
            if prevgranularity < granularity:
                raise ValueError(
                    "The new 'granularity' cannot be higher than the previous granularity"
                    + ", as it must be a value from 1 up to the previous"
                )

            self._downscaling = 10 ** (prevgranularity - granularity)

        self._scaling = 10**granularity

        self.info[self.KEY_I_GRANULARITY] = granularity

        self.info[self.KEY_I_STORE_DTYPE] = (
            "int8" if granularity <= 2 else "int16" if granularity <= 4 else "int32"
        )

        self.info[self.KEY_I_SAFE_DTYPE] = (
            "int8"
            if granularity == 1
            else (
                "int16"
                if granularity == 2
                else "int32" if granularity <= 4 else "int64"
            )
        )

        self._min_projection = int(str(self.BIN_1) * self.info[self.KEY_I_GRANULARITY])
        self._max_projection = int(str(self.BIN_9) * self.info[self.KEY_I_GRANULARITY])

        self._center_projection = int(
            str(self.BIN_5) * self.info[self.KEY_I_GRANULARITY]
        )

    def __prepare_input(self, X):
        ## Identify how to access the columns of 'X'
        if isinstance(X, np.ndarray):
            # By index with known shape
            byindex = True
            shape = X.shape

        else:
            try:
                farray = X[:, 0]
                byindex = True
                try:
                    # By index with known shape
                    shape = X.shape
                except Exception:
                    # By index without known shape
                    shape = None

            except Exception:
                try:
                    # By function call
                    farray = X(0)
                    byindex = False
                    shape = None

                except Exception:
                    raise TypeError(
                        "The array-like provided in 'X' cannot be accessed by slicing"
                        + ". Please provide a 2D array-like in the (n_rows, n_columns) shape"
                        + ". Alternatively, you can replace 'X' with a function to access columns"
                        + " like 'X[:,0], X[:,1], ..., X[:,n_columns]' or 'X(0), X(1), ..., X(n_columns)'"
                    )

        ## Prepare list of feature arrays according to the columns of 'X'
        if byindex and isinstance(shape, (tuple, list)):
            # By index with known shape
            if len(shape) != 2:
                raise ValueError(
                    "The array-like provided in 'X' must have 2 dimensions"
                    + " in the (n_rows, n_columns) shape"
                )

            nrows = int(shape[0])
            ncols = int(shape[1])

            try:
                Xbycols = [
                    np.nan_to_num(np.array(X[:, j], copy=False, ndmin=1), copy=False)
                    for j in range(ncols)
                ]
            except Exception:
                raise TypeError(
                    "The array-like provided in 'X' cannot be accessed by slicing"
                    + ". Please provide a 2D array-like in the (n_rows, n_columns) shape"
                    + ". Alternatively, you can replace 'X' with a function to access columns"
                    + " like 'X[:,0], X[:,1], ..., X[:,n_columns]' or 'X(0), X(1), ..., X(n_columns)'"
                )

        else:
            # By index without known shape, or by function call
            j = 0
            more = True
            nrows = None
            Xbycols = []

            while more:
                try:
                    farray = np.nan_to_num(
                        np.array(farray, copy=False, ndmin=1), copy=False
                    )
                    new_nrows = farray.shape[0]
                except Exception:
                    raise TypeError("The 'X' parameter provides invalid data")

                if nrows is None:
                    nrows = new_nrows

                elif new_nrows != nrows:
                    raise ValueError(
                        "The 'X' parameter must always provide the same number of rows for every column"
                    )

                Xbycols.append(farray)
                j += 1

                if byindex:
                    try:
                        farray = X[:, j]
                    except Exception:
                        more = False

                else:
                    try:
                        farray = X(j)
                    except Exception:
                        more = False

            ncols = len(Xbycols)

        ## Validate size of X
        if nrows == 0:
            raise TypeError(
                "The 'X' parameter must be a 2D array-like in the (n_rows, n_columns) shape,"
                + " where n_rows is at least 1"
            )

        if ncols == 0:
            raise TypeError(
                "The 'X' parameter must be a 2D array-like in the (n_rows, n_columns) shape,"
                + " where n_columns is at least 1"
            )

        return Xbycols, nrows, ncols

    def __init_from_dict(
        self,
        X_info,
        X_features,
        X_projections,
        X_regions,
        granularity,
        name,
        fnames,
        fdtypes,
    ):

        self.__prepare_info(
            granularity,
            name,
            X_info[self.KEY_I_GRANULARITY],
            X_info[self.KEY_I_NAME],
        )

        # ------------------------------

        if fnames is None:
            self.features = {
                j: {
                    self.KEY_F_NAME: X_features[j][self.KEY_F_NAME][
                        : self.CONFIG_F_NAME_LENGTH
                    ]
                }
                for j in range(len(X_features))
            }

        else:
            if len(fnames) != len(X_features):
                raise ValueError(
                    "The 'fnames' parameter must be an array-like"
                    + " of feature names in the (n_features, ) shape,"
                    + " where n_features matches the provided Card"
                )

            self.features = {
                j: {self.KEY_F_NAME: item[: self.CONFIG_F_NAME_LENGTH]}
                for j, item in enumerate(fnames)
            }

        if fdtypes is None:
            for j, jdict in X_features.items():
                self.features[j][self.KEY_F_TYPE] = jdict[self.KEY_F_TYPE]
                self.features[j][self.KEY_F_DTYPE] = np.dtype(
                    jdict[self.KEY_F_DTYPE]
                ).name
                self.features[j][self.KEY_F_INIT] = jdict[self.KEY_F_INIT]

        else:
            if len(fdtypes) != len(X_features):
                raise ValueError(
                    "The 'fdtypes' parameter must be an array-like of feature dtypes"
                    + "  in the (n_features, ) shape, where n_features matches the provided Card"
                )

            for j, item in enumerate(fdtypes):
                item = np.dtype(item)

                if np.issubdtype(item, np.inexact):
                    # Provided dtype was floating point
                    self.features[j][self.KEY_F_TYPE] = self.VAL_F_TYPE_CONTINUOUS

                elif np.issubdtype(item, np.integer):
                    # Provided dtype was integer
                    self.features[j][self.KEY_F_TYPE] = self.VAL_F_TYPE_DISCRETE

                else:
                    raise ValueError(
                        "The 'fdtypes' parameter contains invalid numpy dtypes for a Card"
                    )

                self.features[j][self.KEY_F_DTYPE] = item.name
                self.features[j][self.KEY_F_INIT] = self.VAL_F_INIT_CONFIGURED

        self.info[self.KEY_I_COUNT_F] = X_info[self.KEY_I_COUNT_F]

        # ------------------------------

        if self._downscaling == 1:

            self.projections = {
                j: {
                    p: {
                        self.KEY_P_BIN_MIN: pdict[self.KEY_P_BIN_MIN],
                        self.KEY_P_BIN_MAX: pdict[self.KEY_P_BIN_MAX],
                        self.KEY_P_COUNT_X: pdict[self.KEY_P_COUNT_X],
                        self.KEY_P_COUNT_R: pdict[self.KEY_P_COUNT_R],
                    }
                    for p, pdict in jpdict.items()
                }
                for j, jpdict in X_projections.items()
            }

            self.info[self.KEY_I_COUNT_P] = X_info[self.KEY_I_COUNT_P]

            for j in self.features:

                self.features[j][self.KEY_F_COUNT_P] = X_features[j][self.KEY_F_COUNT_P]
                self.features[j][self.KEY_F_MIN_P] = X_features[j][self.KEY_F_MIN_P]
                self.features[j][self.KEY_F_MAX_P] = X_features[j][self.KEY_F_MAX_P]

            self.regions = {
                r: {
                    self.KEY_R_COUNT_X: rdict[self.KEY_R_COUNT_X],
                }
                for r, rdict in X_regions.items()
            }

            self.info[self.KEY_I_COUNT_R] = X_info[self.KEY_I_COUNT_R]

            # if X_inverses is not None:
            #     self.inverses = [r for r in X_inverses]

        else:
            self.projections = {
                j: {
                    p: {
                        self.KEY_P_BIN_MIN: pdict[self.KEY_P_BIN_MIN],
                        self.KEY_P_BIN_MAX: pdict[self.KEY_P_BIN_MAX],
                        self.KEY_P_COUNT_X: pdict[self.KEY_P_COUNT_X],
                        self.KEY_P_COUNT_R: 0,
                    }
                    for p, pdict in jpdict.items()
                    if p <= self._max_projection
                }
                for j, jpdict in X_projections.items()
            }

            self.info[self.KEY_I_COUNT_P] = 0

            for j, jdict in self.features.items():

                lenjp = len(self.projections[j])
                jdict[self.KEY_F_COUNT_P] = lenjp
                self.info[self.KEY_I_COUNT_P] += lenjp

                jdict[self.KEY_F_MIN_P] = int(
                    X_features[j][self.KEY_F_MIN_P] / self._downscaling
                )
                jdict[self.KEY_F_MAX_P] = int(
                    X_features[j][self.KEY_F_MAX_P] / self._downscaling
                )

            self.regions = {}

            for r, rdict in X_regions.items():
                rnew = tuple(int(p / self._downscaling) for p in r)

                if rnew in self.regions:
                    self.regions[rnew][self.KEY_R_COUNT_X] += rdict[self.KEY_R_COUNT_X]

                else:
                    self.regions[rnew] = {
                        self.KEY_R_COUNT_X: rdict[self.KEY_R_COUNT_X],
                    }

                    for j, p in enumerate(rnew):
                        while p != 0:
                            self.projections[j][p][self.KEY_P_COUNT_R] += 1
                            p = p // 10

            self.info[self.KEY_I_COUNT_R] = len(self.regions)

        # ------------------------------

        self.info[self.KEY_I_COUNT_X] = X_info[self.KEY_I_COUNT_X]

        # if X_inverses is not None:
        #     self.inverses = [
        #         tuple(int(p / scaling) for p in r) for r in X_inverses
        #     ]

    def __init_from_input(self, X, granularity, name, fnames, fdtypes, keep_inverses):

        self.__prepare_info(granularity, name)

        Xbycols, nrows, ncols = self.__prepare_input(X)

        # ------------------------------

        ## Setup 'features' dictionary
        if fnames is None:
            # With default feature name
            self.features = {
                j: {
                    self.KEY_F_NAME: f"{self.CONFIG_F_NAME_DEFAULT} {str(j)}"[
                        : self.CONFIG_F_NAME_LENGTH
                    ]
                }
                for j in range(ncols)
            }

        else:
            # With provided feature names
            if len(fnames) != ncols:
                raise ValueError(
                    "The 'fnames' parameter must be an array-like"
                    + " of feature names in the (n_features, ) shape,"
                    + " where n_features matches the 'X' parameter"
                )

            self.features = {
                j: {self.KEY_F_NAME: item[: self.CONFIG_F_NAME_LENGTH]}
                for j, item in enumerate(fnames)
            }

        # ------------------------------

        ## Add dtypes to 'features' dictionary
        if fdtypes is None:
            # With inferred numpy dtypes
            for j in range(ncols):
                dtype = Xbycols[j].dtype

                if np.issubdtype(dtype, np.inexact):
                    # Inferred dtype was floating point
                    if np.all(np.mod(Xbycols[j], 1) == 0):
                        # But data is integer
                        Xbycols[j] = Xbycols[j].astype(int)
                        self.features[j][self.KEY_F_TYPE] = self.VAL_F_TYPE_DISCRETE

                    else:
                        # And data is floating point
                        self.features[j][self.KEY_F_TYPE] = self.VAL_F_TYPE_CONTINUOUS

                elif np.issubdtype(dtype, np.integer):
                    # Inferred dtype was integer
                    self.features[j][self.KEY_F_TYPE] = self.VAL_F_TYPE_DISCRETE

                else:
                    raise ValueError(
                        "The 'X' parameter contains invalid data for a Card"
                    )

                min_dtype = np.min_scalar_type(Xbycols[j])

                if min_dtype != dtype:
                    # Technical dtype can be optimized in X
                    Xbycols[j] = Xbycols[j].astype(min_dtype)
                    dtype = min_dtype

                self.features[j][self.KEY_F_DTYPE] = dtype.name
                self.features[j][self.KEY_F_INIT] = self.VAL_F_INIT_INFERRED

        else:
            # With provided numpy dtypes
            if len(fdtypes) != ncols:
                raise ValueError(
                    "The 'fdtypes' parameter must be an array-like of feature dtypes"
                    + "  in the (n_features, ) shape, where n_features matches the 'X' parameter"
                )

            for j, item in enumerate(fdtypes):
                if np.issubdtype(item, np.inexact):
                    # Provided dtype was floating point
                    self.features[j][self.KEY_F_TYPE] = self.VAL_F_TYPE_CONTINUOUS

                elif np.issubdtype(item, np.integer):
                    # Provided dtype was integer
                    self.features[j][self.KEY_F_TYPE] = self.VAL_F_TYPE_DISCRETE

                else:
                    raise ValueError(
                        "The 'fdtypes' parameter contains invalid numpy dtypes for a Card"
                    )

                item = np.dtype(item)

                if item != Xbycols[j].dtype:
                    # Provided dtype was different from X
                    try:
                        Xbycols[j] = Xbycols[j].astype(item)
                    except Exception:
                        raise ValueError(
                            "The 'fdtypes' parameter contains numpy dtypes incompatible with the data of the 'X' parameter"
                        )

                self.features[j][self.KEY_F_DTYPE] = item.name
                self.features[j][self.KEY_F_INIT] = self.VAL_F_INIT_CONFIGURED

        # ------------------------------

        self.info[self.KEY_I_COUNT_F] = len(self.features)
        self.info[self.KEY_I_COUNT_P] = 0

        # ------------------------------

        self.projections = {j: {} for j in range(self.info[self.KEY_I_COUNT_F])}

        for j, farray in enumerate(Xbycols):
            Xbycols[j] = self.__analyze_feature(farray, self.projections[j])

            ptup = tuple(self.projections[j].keys())

            pmin = -1
            for p in ptup:
                if p // 10 > pmin // 10:
                    pmin = p
                else:
                    break

            lenjp = len(ptup)
            self.features[j][self.KEY_F_COUNT_P] = lenjp
            self.info[self.KEY_I_COUNT_P] += lenjp

            self.features[j][self.KEY_F_MIN_P] = pmin
            self.features[j][self.KEY_F_MAX_P] = ptup[-1]

        # ------------------------------

        self.regions = {}

        if keep_inverses:
            self.inverses = []

        Xiter = np.nditer(tuple(Xbycols))

        for tup in Xiter:
            tup = tuple(p.item() for p in tup)

            if keep_inverses:
                self.inverses.append(tup)

            if tup in self.regions:
                self.regions[tup][self.KEY_R_COUNT_X] += 1

            else:
                self.regions[tup] = {self.KEY_R_COUNT_X: 1}

                for j, p in enumerate(tup):
                    while p != 0:
                        self.projections[j][p][self.KEY_P_COUNT_R] += 1
                        p = p // 10

        self.info[self.KEY_I_COUNT_R] = len(self.regions)

        # ------------------------------

        self.info[self.KEY_I_COUNT_X] = (
            len(self.inverses)
            if keep_inverses
            else sum(
                [len(rdict[self.KEY_R_COUNT_X]) for rdict in self.regions.values()]
            )
        )

    def __analyze_feature(self, farray, jpdict):
        # Prepare recursive function
        minmin = farray.min().item()

        return self.__analyze_recursive(
            np.zeros(farray.shape, dtype=self.info[self.KEY_I_STORE_DTYPE]),
            farray,
            jpdict,
            minmin,
            minmin,
            farray.max().item(),
            self.info[self.KEY_I_GRANULARITY],
            0,
        )

    def __analyze_recursive(
        self,
        new_farray,
        farray,
        jpdict,
        minmin,
        fmin,
        fmax,
        level,
        prevcode,
    ):
        # Prepare current granularity level
        level -= 1
        prevcode *= 10

        # Compute mean and population standard deviation of current subarray
        fmean = farray.mean().item()
        fstd = farray.std(mean=fmean, ddof=0).item()
        f0half = fstd / 2
        f1half = fstd + f0half
        f2half = fstd + f1half
        f3half = fstd + f2half

        minus0half = fmean - f0half
        # if discrete:
        #     minus0half = math.floor(minus0half)

        if minus0half > fmin and minus0half != fmax:
            minus1half = fmean - f1half
            # if discrete:
            #     minus1half = math.floor(minus1half)

            if minus1half > fmin and minus1half != minus0half:
                minus2half = fmean - f2half
                # if discrete:
                #     minus2half = math.floor(minus2half)

                if minus2half > fmin and minus2half != minus1half:
                    minus3half = fmean - f3half
                    # if discrete:
                    #     minus3half = math.floor(minus3half)

                    if minus3half > fmin and minus3half != minus2half:

                        code = prevcode + self.BIN_1
                        mask = (
                            (farray > fmin) & (farray <= minus3half)
                            if fmin != minmin
                            else (farray >= fmin) & (farray <= minus3half)
                        )

                        # Analyze branch of code C1
                        if mask.any():
                            mmfarray = farray[mask]

                            jpdict[code] = {
                                self.KEY_P_BIN_MIN: fmin,
                                self.KEY_P_BIN_MAX: minus3half,
                                self.KEY_P_COUNT_X: np.count_nonzero(mask).item(),
                                self.KEY_P_COUNT_R: 0,
                            }

                            # Secondary stopping condition:
                            # Reached maximum granularity level
                            if level == 0:
                                new_farray[mask] = code

                            else:
                                new_farray[mask] = self.__analyze_recursive(
                                    new_farray[mask],
                                    mmfarray,
                                    jpdict,
                                    minmin,
                                    fmin,
                                    minus3half,
                                    level,
                                    code,
                                )

                    else:
                        minus3half = fmin

                    code = prevcode + self.BIN_2
                    mask = (
                        (farray > minus3half) & (farray <= minus2half)
                        if minus3half != minmin
                        else (farray >= minus3half) & (farray <= minus2half)
                    )

                    # Analyze branch of code C2
                    if mask.any():
                        mmfarray = farray[mask]

                        jpdict[code] = {
                            self.KEY_P_BIN_MIN: minus3half,
                            self.KEY_P_BIN_MAX: minus2half,
                            self.KEY_P_COUNT_X: np.count_nonzero(mask).item(),
                            self.KEY_P_COUNT_R: 0,
                        }

                        # Secondary stopping condition:
                        # Reached maximum granularity level
                        if level == 0:
                            new_farray[mask] = code

                        else:
                            new_farray[mask] = self.__analyze_recursive(
                                new_farray[mask],
                                mmfarray,
                                jpdict,
                                minmin,
                                minus3half,
                                minus2half,
                                level,
                                code,
                            )

                else:
                    minus2half = fmin

                code = prevcode + self.BIN_3
                mask = (
                    (farray > minus2half) & (farray <= minus1half)
                    if minus2half != minmin
                    else (farray >= minus2half) & (farray <= minus1half)
                )

                # Analyze branch of code C3
                if mask.any():
                    mmfarray = farray[mask]

                    jpdict[code] = {
                        self.KEY_P_BIN_MIN: minus2half,
                        self.KEY_P_BIN_MAX: minus1half,
                        self.KEY_P_COUNT_X: np.count_nonzero(mask).item(),
                        self.KEY_P_COUNT_R: 0,
                    }

                    # Secondary stopping condition:
                    # Reached maximum granularity level
                    if level == 0:
                        new_farray[mask] = code

                    else:
                        new_farray[mask] = self.__analyze_recursive(
                            new_farray[mask],
                            mmfarray,
                            jpdict,
                            minmin,
                            minus2half,
                            minus1half,
                            level,
                            code,
                        )

            else:
                minus1half = fmin

            code = prevcode + self.BIN_4
            mask = (
                (farray > minus1half) & (farray <= minus0half)
                if minus1half != minmin
                else (farray >= minus1half) & (farray <= minus0half)
            )

            # Analyze branch of code C4
            if mask.any():
                mmfarray = farray[mask]

                jpdict[code] = {
                    self.KEY_P_BIN_MIN: minus1half,
                    self.KEY_P_BIN_MAX: minus0half,
                    self.KEY_P_COUNT_X: np.count_nonzero(mask).item(),
                    self.KEY_P_COUNT_R: 0,
                }

                # Secondary stopping condition:
                # Reached maximum granularity level
                if level == 0:
                    new_farray[mask] = code

                else:
                    new_farray[mask] = self.__analyze_recursive(
                        new_farray[mask],
                        mmfarray,
                        jpdict,
                        minmin,
                        minus1half,
                        minus0half,
                        level,
                        code,
                    )

        else:
            minus0half = fmin

        plus0half = fmean + f0half
        # if discrete:
        #     plus0half = math.ceil(plus0half)

        if plus0half > fmax or plus0half == minus0half:
            plus0half = fmax

        code = prevcode + self.BIN_5
        mask = (
            (farray > minus0half) & (farray <= plus0half)
            if minus0half != minmin
            else (farray >= minus0half) & (farray <= plus0half)
        )

        # Analyze branch of code C5 (center code)
        if mask.any():
            mmfarray = farray[mask]

            jpdict[code] = {
                self.KEY_P_BIN_MIN: minus0half,
                self.KEY_P_BIN_MAX: plus0half,
                self.KEY_P_COUNT_X: np.count_nonzero(mask).item(),
                self.KEY_P_COUNT_R: 0,
            }

            # Primary stopping condition:
            # Current subarray cannot be split any further
            # Remaining granularity levels will only have code C5 (center code)
            if mask.all():
                if level != 0:
                    prevcode = code
                    for _ in range(level):
                        code = code * 10 + self.BIN_5
                        jpdict[code] = jpdict[prevcode]

                new_farray[:] = code
                return new_farray

            # Secondary stopping condition:
            # Reached maximum granularity level
            if level == 0:
                new_farray[mask] = code

            else:
                new_farray[mask] = self.__analyze_recursive(
                    new_farray[mask],
                    mmfarray,
                    jpdict,
                    minmin,
                    minus0half,
                    plus0half,
                    level,
                    code,
                )

        if plus0half < fmax:
            plus1half = fmean + f1half
            # if discrete:
            #     plus1half = math.ceil(plus1half)

            if plus1half > fmax or plus1half == plus0half:
                plus1half = fmax

            code = prevcode + self.BIN_6
            mask = (farray > plus0half) & (farray <= plus1half)

            # Analyze branch of code C6
            if mask.any():
                mmfarray = farray[mask]

                jpdict[code] = {
                    self.KEY_P_BIN_MIN: plus0half,
                    self.KEY_P_BIN_MAX: plus1half,
                    self.KEY_P_COUNT_X: np.count_nonzero(mask).item(),
                    self.KEY_P_COUNT_R: 0,
                }

                # Secondary stopping condition:
                # Reached maximum granularity level
                if level == 0:
                    new_farray[mask] = code

                else:
                    new_farray[mask] = self.__analyze_recursive(
                        new_farray[mask],
                        mmfarray,
                        jpdict,
                        minmin,
                        plus0half,
                        plus1half,
                        level,
                        code,
                    )

            if plus1half < fmax:
                plus2half = fmean + f2half
                # if discrete:
                #     plus2half = math.ceil(plus2half)

                if plus2half > fmax or plus2half == plus1half:
                    plus2half = fmax

                code = prevcode + self.BIN_7
                mask = (farray > plus1half) & (farray <= plus2half)

                # Analyze branch of code C7
                if mask.any():
                    mmfarray = farray[mask]

                    jpdict[code] = {
                        self.KEY_P_BIN_MIN: plus1half,
                        self.KEY_P_BIN_MAX: plus2half,
                        self.KEY_P_COUNT_X: np.count_nonzero(mask).item(),
                        self.KEY_P_COUNT_R: 0,
                    }

                    # Secondary stopping condition:
                    # Reached maximum granularity level
                    if level == 0:
                        new_farray[mask] = code

                    else:
                        new_farray[mask] = self.__analyze_recursive(
                            new_farray[mask],
                            mmfarray,
                            jpdict,
                            minmin,
                            plus1half,
                            plus2half,
                            level,
                            code,
                        )

                if plus2half < fmax:
                    plus3half = fmean + f3half
                    # if discrete:
                    #     plus3half = math.ceil(plus3half)

                    if plus3half > fmax or plus3half == plus2half:
                        plus3half = fmax

                    code = prevcode + self.BIN_8
                    mask = (farray > plus2half) & (farray <= plus3half)

                    # Analyze branch of code C8
                    if mask.any():
                        mmfarray = farray[mask]

                        jpdict[code] = {
                            self.KEY_P_BIN_MIN: plus2half,
                            self.KEY_P_BIN_MAX: plus3half,
                            self.KEY_P_COUNT_X: np.count_nonzero(mask).item(),
                            self.KEY_P_COUNT_R: 0,
                        }

                        # Secondary stopping condition:
                        # Reached maximum granularity level
                        if level == 0:
                            new_farray[mask] = code

                        else:
                            new_farray[mask] = self.__analyze_recursive(
                                new_farray[mask],
                                mmfarray,
                                jpdict,
                                minmin,
                                plus2half,
                                plus3half,
                                level,
                                code,
                            )

                    if plus3half < fmax:

                        code = prevcode + self.BIN_9
                        mask = (farray > plus3half) & (farray <= fmax)

                        # Analyze branch of code C9
                        if mask.any():
                            mmfarray = farray[mask]

                            jpdict[code] = {
                                self.KEY_P_BIN_MIN: plus3half,
                                self.KEY_P_BIN_MAX: fmax,
                                self.KEY_P_COUNT_X: np.count_nonzero(mask).item(),
                                self.KEY_P_COUNT_R: 0,
                            }

                            # Secondary stopping condition:
                            # Reached maximum granularity level
                            if level == 0:
                                new_farray[mask] = code

                            else:
                                new_farray[mask] = self.__analyze_recursive(
                                    new_farray[mask],
                                    mmfarray,
                                    jpdict,
                                    minmin,
                                    plus3half,
                                    fmax,
                                    level,
                                    code,
                                )

        return new_farray

    def __convert_feature(self, farray, j):
        # Prepare recursive function
        new_farray = np.zeros(farray.shape, dtype=self.info[self.KEY_I_STORE_DTYPE])
        use_mask = False

        mask_lower = (
            farray
            < self.projections[j][self.features[j][self.KEY_F_MIN_P]][
                self.KEY_P_BIN_MIN
            ]
        )

        mask_higher = (
            farray
            > self.projections[j][self.features[j][self.KEY_F_MAX_P]][
                self.KEY_P_BIN_MAX
            ]
        )

        # Adjust values lower than known feature minimum
        if mask_lower.any():
            new_farray[mask_lower] = self._min_projection
            use_mask = True

        # Adjust values higher than known feature maximum
        if mask_higher.any():
            new_farray[mask_higher] = self._max_projection
            use_mask = True

        if use_mask:
            mask_final = ~(mask_lower | mask_higher)

            new_farray[mask_final] = self.__convert_recursive(
                new_farray[mask_final],
                farray[mask_final],
                self.projections[j],
                self.info[self.KEY_I_GRANULARITY],
                0,
            )
            return new_farray

        else:
            return self.__convert_recursive(
                new_farray,
                farray,
                self.projections[j],
                self.info[self.KEY_I_GRANULARITY],
                0,
            )

    def __convert_recursive(
        self,
        new_farray,
        farray,
        jpdict,
        level,
        prevcode,
    ):
        # Prepare current granularity level
        # totalmask = np.ones(farray.shape[0], dtype=bool)
        level -= 1
        prevcode *= 10
        # tempcode = prevcode * 10

        for c in self.BINS_DESC:
            code = prevcode + c
            # code = tempcode + c

            if code in jpdict:
                mask = (farray >= jpdict[code][self.KEY_P_BIN_MIN]) & (
                    farray <= jpdict[code][self.KEY_P_BIN_MAX]
                )

                if mask.any():
                    # totalmask = totalmask & ~mask

                    # Primary stopping condition:
                    # Found complete code at maximum granularity level
                    if level == 0:
                        new_farray[mask] = code

                    else:
                        new_farray[mask] = self.__convert_recursive(
                            new_farray[mask],
                            farray[mask],
                            jpdict,
                            level,
                            code,
                        )

        # # Secondary stopping condition:
        # # Current subarray cannot be split any further
        # if totalmask.any():
        #     new_farray[totalmask] = prevcode

        return new_farray

    def summary(self, features=None, plotter=None):

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

        if features is None:
            features = [j for j in range(len(self.features))]

            miniregions = {
                r: rdict[self.KEY_R_COUNT_X] for r, rdict in self.regions.items()
            }

        else:
            try:
                features = [int(item) for item in features]
            except Exception as e:
                raise TypeError(
                    "The 'features' parameter must be an array-like of feature indexes"
                ) from e

            if len(features) == 0:
                raise ValueError(
                    "The 'features' parameter contains must not be empty"
                ) from e

            for j in features:
                if j not in self.features:
                    raise ValueError(
                        "The 'features' parameter contains invalid feature indexes"
                    ) from e

            miniregions = {}

            for r, rdict in self.regions.items():
                tup = tuple(r[j] for j in features)

                if tup in miniregions:
                    miniregions[tup] += rdict[self.KEY_R_COUNT_X]
                else:
                    miniregions[tup] = rdict[self.KEY_R_COUNT_X]

        if plotter is not None:
            alpha_scaling = 0.8 / self.info[self.KEY_I_COUNT_X]

            minilines = {}

            for r, val in miniregions.items():
                for minij in range(len(features)):
                    minitup = [0 for _ in range(len(features))]

                    if minij == 0:
                        minitup[0] = r[0]
                        minitup[-1] = r[-1]
                    else:
                        minitup[minij - 1] = r[minij - 1]
                        minitup[minij] = r[minij]

                    minitup = tuple(minitup)

                    if minitup in minilines:
                        minilines[minitup] += val
                    else:
                        minilines[minitup] = val

            try:
                fig = plotter(
                    lines=[
                        [[item / self._scaling for item in tup] for tup in minilines]
                    ],
                    alphas=[
                        [item * alpha_scaling + 0.2 for item in minilines.values()]
                    ],
                    vertice_labels=[
                        self.features[j][self.KEY_F_NAME] for j in features
                    ],
                    button_labels=None,
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
