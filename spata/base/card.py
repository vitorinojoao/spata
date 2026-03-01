"""spata.base.card"""

import copy
import math
import time
import numpy as np

# from collections.abc import Sequence


class Card:
    """Card"""

    # Positions of 'Classes' information
    CNAME = 0
    CSAMPLES = 1
    CCOMBOS = 2

    # Positions of 'Features' information
    FNAME = 0
    FTYPE = 1
    FDTYPE = 2
    FCODES = 3

    # The status of each feature
    # STATUS_CONFIGURED = "configured"
    # STATUS_INFERRED = "inferred"
    # STATUS_MERGED = "merged"

    # The keywords for each feature type
    TYPE_CONTINUOUS = "continuous"
    TYPE_DISCRETE = "discrete"
    TYPE_BOOLEAN = "boolean"
    TYPE_CATEGORICAL = "categorical"

    # The corresponding dtypes for each feature type
    TYPETABLE = {
        TYPE_CONTINUOUS: float,
        TYPE_DISCRETE: int,
        TYPE_BOOLEAN: int,
        TYPE_CATEGORICAL: str,
    }

    # The 7 digits to be used in the encodings from fmin to fmax
    C1 = 1
    C2 = 2
    C3 = 3
    C4 = 4
    C5 = 5
    C6 = 6
    C7 = 7
    C8 = 8
    C9 = 9

    # The list of codes belonging to the Card
    CLIST = (C1, C2, C3, C4, C5, C6, C7, C8, C9)
    CLISTHL = (C9, C8, C7, C6, C5, C4, C3, C2, C1)

    DEFAULT_CLASS_LABEL = "No Class"
    DEFAULT_CLASS_PREFIX = "Class "

    DEFAULT_FEATURE_NAME = "Feature "
    DEFAULT_FEATURE_PREFIX = ""

    def __init__(
        self,
        X,
        y=None,
        granularity=None,
        fnames=None,
        fdtypes=None,
        seed=None,
    ):
        if isinstance(X, Card):
            self.granularity = X.granularity
            self.scale = X.scale
            self.dtype = X.dtype
            self.classes = X.classes
            self.features = X.features
            self.encodings = X.encodings
            self.combinations = X.combinations

        elif isinstance(X, dict):
            self.__init_from_dict(X)

        else:
            self.__init_from_data(X, y, granularity, fnames, fdtypes, seed)

    def __str__(self):
        sumcodes = 0
        for tup in self.features.values():
            sumcodes += tup[3]

        return (
            "Card("
            + str(
                {
                    "granularity": self.granularity,
                    "classes": len(self.classes),
                    "features": len(self.features),
                    "encodings": sumcodes,
                    "combinations": len(self),
                },
            )[1:-1]
            + ")"
        )

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False

        if (
            self.granularity != other.granularity
            or self.classes != other.classes
            or self.features != other.features
        ):
            return False

        if self.encodings.keys() != other.encodings.keys():
            return False

        if self.combinations.keys() != other.combinations.keys():
            return False

        for j, codedict in self.encodings.items():
            if codedict != other.encodings[j]:
                return False

            for code, tup in codedict.items():
                if tup[1] != other.encodings[j][code][1]:
                    return False

        for c, combodict in self.combinations.items():
            if combodict != other.combinations[c]:
                return False

        return True

    def __hash__(self):
        return hash(
            (
                self.granularity,
                frozenset(
                    (j, frozenset(codedict)) for j, codedict in self.encodings.items()
                ),
            )
        )

    def save(self, filepath=None):
        res = {
            "timestamp": int(time.time()),
            "analysis_time": self.atime,
            "granularity": self.granularity,
            "classes": {
                k: {
                    "description": tup[0],
                    "total_samples": tup[1],
                    "total_combinations": tup[2],
                }
                for k, tup in self.classes.items()
            },
            "features": {
                j: {
                    "description": tup[0],
                    "type": tup[1],
                    "dtype": tup[2],
                    "total_encodings": tup[3],
                }
                for j, tup in self.features.items()
            },
            "subfeatures": {
                j: {
                    code: {
                        f"[{str(intvl)[1:-1]}]": str(self.encoding_overlaps[j][code])
                    }
                    for code, intvl in dct.items()
                }
                for j, dct in self.encodings.items()
            },
            "instances": {
                f"[{str(combo)[1:-1]}]": str(ovlps)
                for combo, ovlps in self.combination_overlaps.items()
            },
        }

        if filepath is None:
            return res

        else:
            import json

            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=4)

            except Exception as e:
                raise TypeError(
                    "This Card could not be saved to the 'filepath' parameter"
                ) from e

    def __init_from_dict(self, X):
        if (
            "granularity" not in X
            or "classes" not in X
            or "features" not in X
            or "encodings" not in X
            or "combinations" not in X
        ):
            raise ValueError(
                "The provided dictionary does not contain the required 'granularity',"
                + " 'classes', 'features', 'encodings', and 'combinations' keys"
                + f". Found: {tuple(X.keys())}"
            )

        try:
            self.granularity = int(X["granularity"])

        except Exception:
            raise ValueError(
                "The provided dictionary contains an invalid entry in 'granularity'"
                # + f". Found: {X["granularity"]}"
            )

        self.classes = X["classes"]
        self.features = X["features"]
        self.encodings = X["encodings"]
        self.combinations = X["combinations"]

    @property
    def encoded(self):
        vectorized = np.vectorize(lambda x: 10 ** int(math.log10(x) + 1), otypes=[int])

        res = []
        for farray in self.__encoded:
            scaling = vectorized(farray)
            res.append(farray / scaling)

        return res

    def convert(self, X, return_numpy=True):
        Xbycols, nrows, ncols = self.__validate_X(X)

        if ncols < len(self.features):
            raise ValueError(
                "The array-like provided in 'X' contains less"
                + " features than this Card."
            )
        elif ncols > len(self.features):
            raise ValueError(
                "The array-like provided in 'X' contains more"
                + " features than this Card."
            )

        res = np.empty((nrows, ncols), dtype=self.dtype)

        for j in range(ncols):
            if Xbycols[j].dtype != self.features[j][2]:
                try:
                    Xbycols[j] = np.array(Xbycols[j], dtype=self.features[j][2])
                except Exception:
                    raise ValueError(
                        "The 'fdtypes' parameter contains numpy dtypes incompatible with the data of the 'X' parameter"
                    )

            res[:, j] = self.__convert_feature(Xbycols[j], j)

            # print(res[:, j])

        return res

    def __init_from_data(self, X, y, granularity, fnames, fdtypes, seed):
        Xbycols, y, ncols, nclasses = self.__validate_init_data(
            X, y, granularity, fnames, fdtypes, seed
        )

        atime = time.time()

        ## Analyze the distribution of each feature
        for j in range(ncols):
            farray = self.__analyze_feature(Xbycols[j], y, j)

            fkeys = tuple(self.encodings[j].keys())
            lower = fkeys[0]
            for fk in fkeys[1:]:
                if fk // 10 > lower // 10:
                    lower = fk
                else:
                    break

            self.features[j].extend((len(self.encodings[j]), lower, fkeys[-1]))

            self.features[j] = tuple(self.features[j])

            Xbycols[j] = farray

        self.__encoded = Xbycols

        ## Analyze the combinations of each class
        for k in range(nclasses):
            mask = y == k
            filtered = np.empty((np.count_nonzero(mask), ncols), dtype=self.dtype)

            for j in range(ncols):
                filtered[:, j] = Xbycols[j][mask]

            self.combinations[k], ccounts = np.unique(
                filtered, return_counts=True, axis=0
            )

            for combo, ct in np.nditer(
                (
                    self.combinations[k].copy(order="F"),
                    ccounts.reshape((-1, 1)),
                ),
                flags=["external_loop"],
                order="C",
            ):
                tup = tuple(combo.tolist())

                if tup in self.combination_overlaps:
                    self.combination_overlaps[tup][k] = ct[0].item()

                else:
                    self.combination_overlaps[tup] = {k: ct[0].item()}

            self.classes[k].append(self.combinations[k].shape[0])
            self.classes[k] = tuple(self.classes[k])

        self.atime = time.time() - atime

    def __validate_init_data(self, X, y, granularity, fnames, fdtypes, seed):
        Xbycols, nrows, ncols = self.__validate_X(X)

        ## Convert 'y' array, and setup 'classes' dictionary
        if y is None:
            # With default class label
            # y = np.zeros(nrows, dtype=self.dtype)
            y = np.zeros(nrows, dtype=int)
            self.classes = {0: [Card.DEFAULT_CLASS_LABEL, nrows]}

        else:
            # With provided class labels
            try:
                cnames, y, ccounts = np.unique(
                    y, return_inverse=True, return_counts=True
                )
                # y = y.astype(self.dtype)
                y = y.astype(int)
            except Exception:
                raise TypeError(
                    "The 'y' parameter must be a 1D array-like in the (n_rows, ) shape"
                )

            if y.shape[0] != nrows:
                raise TypeError(
                    "The 'y' parameter must be a 1D array-like in the (n_rows, ) shape,"
                    + " where n_rows matches the 'X' parameter"
                    + f". Expected: {nrows}. Found: {y.shape[0]}"
                )

            self.classes = {
                k: [
                    Card.DEFAULT_CLASS_PREFIX + str(cnames.item(k)),
                    ccounts.item(k),
                ]
                for k in range(len(cnames))
            }

        nclasses = len(self.classes)

        ## Setup 'granularity', 'scale', and 'dtype'
        if granularity is None:
            self.granularity = 4

        else:
            try:
                granularity = int(granularity)
            except Exception as e:
                raise TypeError(
                    "The 'granularity' argument must be an integer value"
                ) from e

            if granularity < 1 or granularity > 8:
                raise ValueError(
                    "The 'granularity' argument must be a value from 1 up to 8"
                    + ", representing codes from 1 digit [1, 9] up to 8 digits [11111111, 99999999]"
                )

            self.granularity = granularity

        self.code_min = int(str(Card.C1) * self.granularity)
        self.code_max = int(str(Card.C9) * self.granularity)

        self.scale = 10**self.granularity

        self.dtype = (
            np.int8
            if granularity <= 2
            else (np.int16 if granularity <= 4 else np.int32)
        )

        self.dtype_safe = (
            np.int8
            if granularity == 1
            else (
                np.int16
                if granularity == 2
                else (np.int32 if granularity <= 4 else np.int64)
            )
        )

        ## Setup 'features' dictionary
        if fnames is None:
            # With default feature name
            self.features = {
                0: [Card.DEFAULT_FEATURE_NAME + str(j)] for j in range(ncols)
            }

        else:
            # With provided feature names
            try:
                fnames = [str(name) for name in fnames]
            except Exception:
                raise ValueError(
                    "The 'fnames' parameter must be an array-like of feature names"
                )

            if len(fnames) != ncols:
                raise ValueError(
                    "The 'fnames' parameter must be an array-like"
                    + " of feature names in the (n_features, ) shape,"
                    + " where n_features matches the 'X' parameter"
                )

            self.features = {
                j: [Card.DEFAULT_FEATURE_PREFIX + fnames[j]] for j in range(len(fnames))
            }

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
                        self.features[j].append(Card.TYPE_DISCRETE)

                    else:
                        # And data is floating point
                        self.features[j].append(Card.TYPE_CONTINUOUS)

                elif np.issubdtype(dtype, np.integer):
                    # Inferred dtype was integer
                    self.features[j].append(Card.TYPE_DISCRETE)

                else:
                    raise ValueError(
                        "The 'X' parameter contains invalid data for a Card"
                    )

                new_dtype = np.min_scalar_type(Xbycols[j])

                if new_dtype != dtype:
                    Xbycols[j] = Xbycols[j].astype(new_dtype)
                    self.features[j].append(str(new_dtype))

                else:
                    self.features[j].append(str(dtype))

        else:
            # With provided numpy dtypes
            try:
                fdtypes = list(fdtypes)
            except Exception:
                raise ValueError(
                    "The 'fdtypes' parameter must be an array-like of feature dtypes"
                )

            if len(fdtypes) != ncols:
                raise ValueError(
                    "The 'fdtypes' parameter must be an array-like of feature dtypes"
                    + "  in the (n_features, ) shape, where n_features matches the 'X' parameter"
                )

            for j in range(ncols):
                if np.issubdtype(fdtypes[j], np.inexact):
                    # Provided dtype was floating point
                    self.features[j].append(Card.TYPE_CONTINUOUS)

                elif np.issubdtype(fdtypes[j], np.integer):
                    # Provided dtype was integer
                    self.features[j].append(Card.TYPE_DISCRETE)

                else:
                    raise ValueError(
                        "The 'fdtypes' parameter contains invalid numpy dtypes for a Card"
                    )

                if fdtypes[j] != Xbycols[j].dtype:
                    try:
                        Xbycols[j] = np.array(Xbycols[j], dtype=fdtypes[j])
                    except Exception:
                        raise ValueError(
                            "The 'fdtypes' parameter contains numpy dtypes incompatible with the data of the 'X' parameter"
                        )

                self.features[j].append(str(fdtypes[j]))

        ## Setup 'seed' and 'generator'
        try:
            self._seed = copy.deepcopy(seed)
            self._generator = np.random.default_rng(seed)
        except Exception as e:
            raise TypeError(
                "The 'seed' argument must be an integer value to enable reproducibility,"
                + " a numpy Generator object to use it unaltered,"
                + " or None to use pseudo-random numbers"
            ) from e

        ## Setup 'encodings' and 'encoding_overlaps' dictionaries
        self.encodings = {j: {} for j in range(ncols)}
        self.encoding_overlaps = {j: {} for j in range(ncols)}

        ## Setup 'combinations' and 'combination_overlaps' dictionaries
        self.combinations = {k: None for k in range(nclasses)}
        self.combination_overlaps = {}

        return Xbycols, y, ncols, nclasses

    def __validate_X(self, X):
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
                    new_nrows = int(farray.shape[0])
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
                + f". Found: {nrows}"
            )

        if ncols == 0:
            raise TypeError(
                "The 'X' parameter must be a 2D array-like in the (n_rows, n_columns) shape,"
                + " where n_columns is at least 1"
                + f". Found: {ncols}"
            )

        return Xbycols, nrows, ncols

    def __analyze_feature(self, farray, y, j):
        # Prepare recursive function
        return self.__analyze_recursive(
            np.zeros(farray.shape, dtype=self.dtype),
            farray,
            y,
            fmin := farray.min().item(),
            fmin,
            farray.max().item(),
            j,
            self.granularity,
            0,
        )

    def __analyze_recursive(
        self,
        new_farray,
        farray,
        y,
        minmin,
        fmin,
        fmax,
        j,
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

                        code = prevcode + Card.C1
                        mask = (
                            (farray > fmin) & (farray <= minus3half)
                            if fmin != minmin
                            else (farray >= fmin) & (farray <= minus3half)
                        )

                        # Analyze branch of code C1
                        if mask.any():
                            mmy = y[mask]
                            mmfarray = farray[mask]
                            fl, fc = np.unique(mmfarray, return_counts=True)

                            self.encodings[j][code] = (
                                fmin,
                                fl[np.argmax(fc)].item(),
                                minus3half,
                            )

                            self.encoding_overlaps[j][code] = {
                                cl.item(): ct.item()
                                for cl, ct in np.nditer(
                                    np.unique(mmy, return_counts=True)
                                )
                            }

                            # Secondary stopping condition:
                            # Reached maximum granularity level
                            if level == 0:
                                new_farray[mask] = code

                            else:
                                new_farray[mask] = self.__analyze_recursive(
                                    new_farray[mask],
                                    mmfarray,
                                    mmy,
                                    minmin,
                                    fmin,
                                    minus3half,
                                    j,
                                    level,
                                    code,
                                )

                    else:
                        minus3half = fmin

                    code = prevcode + Card.C2
                    mask = (
                        (farray > minus3half) & (farray <= minus2half)
                        if minus3half != minmin
                        else (farray >= minus3half) & (farray <= minus2half)
                    )

                    # Analyze branch of code C2
                    if mask.any():
                        mmy = y[mask]
                        mmfarray = farray[mask]
                        fl, fc = np.unique(mmfarray, return_counts=True)

                        self.encodings[j][code] = (
                            minus3half,
                            fl[np.argmax(fc)].item(),
                            minus2half,
                        )

                        self.encoding_overlaps[j][code] = {
                            cl.item(): ct.item()
                            for cl, ct in np.nditer(np.unique(mmy, return_counts=True))
                        }

                        # Secondary stopping condition:
                        # Reached maximum granularity level
                        if level == 0:
                            new_farray[mask] = code

                        else:
                            new_farray[mask] = self.__analyze_recursive(
                                new_farray[mask],
                                mmfarray,
                                mmy,
                                minmin,
                                minus3half,
                                minus2half,
                                j,
                                level,
                                code,
                            )

                else:
                    minus2half = fmin

                code = prevcode + Card.C3
                mask = (
                    (farray > minus2half) & (farray <= minus1half)
                    if minus2half != minmin
                    else (farray >= minus2half) & (farray <= minus1half)
                )

                # Analyze branch of code C3
                if mask.any():
                    mmy = y[mask]
                    mmfarray = farray[mask]
                    fl, fc = np.unique(mmfarray, return_counts=True)

                    self.encodings[j][code] = (
                        minus2half,
                        fl[np.argmax(fc)].item(),
                        minus1half,
                    )

                    self.encoding_overlaps[j][code] = {
                        cl.item(): ct.item()
                        for cl, ct in np.nditer(np.unique(mmy, return_counts=True))
                    }

                    # Secondary stopping condition:
                    # Reached maximum granularity level
                    if level == 0:
                        new_farray[mask] = code

                    else:
                        new_farray[mask] = self.__analyze_recursive(
                            new_farray[mask],
                            mmfarray,
                            mmy,
                            minmin,
                            minus2half,
                            minus1half,
                            j,
                            level,
                            code,
                        )

            else:
                minus1half = fmin

            code = prevcode + Card.C4
            mask = (
                (farray > minus1half) & (farray <= minus0half)
                if minus1half != minmin
                else (farray >= minus1half) & (farray <= minus0half)
            )

            # Analyze branch of code C4
            if mask.any():
                mmy = y[mask]
                mmfarray = farray[mask]
                fl, fc = np.unique(mmfarray, return_counts=True)

                self.encodings[j][code] = (
                    minus1half,
                    fl[np.argmax(fc)].item(),
                    minus0half,
                )

                self.encoding_overlaps[j][code] = {
                    cl.item(): ct.item()
                    for cl, ct in np.nditer(np.unique(mmy, return_counts=True))
                }

                # Secondary stopping condition:
                # Reached maximum granularity level
                if level == 0:
                    new_farray[mask] = code

                else:
                    new_farray[mask] = self.__analyze_recursive(
                        new_farray[mask],
                        mmfarray,
                        mmy,
                        minmin,
                        minus1half,
                        minus0half,
                        j,
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

        code = prevcode + Card.C5
        mask = (
            (farray > minus0half) & (farray <= plus0half)
            if minus0half != minmin
            else (farray >= minus0half) & (farray <= plus0half)
        )

        # Analyze branch of code C5 (center code)
        if mask.any():
            mmy = y[mask]
            mmfarray = farray[mask]
            fl, fc = np.unique(mmfarray, return_counts=True)

            self.encodings[j][code] = (
                minus0half,
                fl[np.argmax(fc)].item(),
                plus0half,
            )

            self.encoding_overlaps[j][code] = {
                cl.item(): ct.item()
                for cl, ct in np.nditer(np.unique(mmy, return_counts=True))
            }

            # Primary stopping condition:
            # Current subarray cannot be split any further
            # Remaining granularity levels will only have code C5 (center code)
            if mask.all():
                if level != 0:
                    prevcode = code
                    for _ in range(level):
                        code = code * 10 + Card.C5
                        self.encodings[j][code] = self.encodings[j][prevcode]
                        self.encoding_overlaps[j][code] = self.encoding_overlaps[j][
                            prevcode
                        ]

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
                    mmy,
                    minmin,
                    minus0half,
                    plus0half,
                    j,
                    level,
                    code,
                )

        if plus0half < fmax:
            plus1half = fmean + f1half
            # if discrete:
            #     plus1half = math.ceil(plus1half)

            if plus1half > fmax or plus1half == plus0half:
                plus1half = fmax

            code = prevcode + Card.C6
            mask = (farray > plus0half) & (farray <= plus1half)

            # Analyze branch of code C6
            if mask.any():
                mmy = y[mask]
                mmfarray = farray[mask]
                fl, fc = np.unique(mmfarray, return_counts=True)

                self.encodings[j][code] = (
                    plus0half,
                    fl[np.argmax(fc)].item(),
                    plus1half,
                )

                self.encoding_overlaps[j][code] = {
                    cl.item(): ct.item()
                    for cl, ct in np.nditer(np.unique(mmy, return_counts=True))
                }

                # Secondary stopping condition:
                # Reached maximum granularity level
                if level == 0:
                    new_farray[mask] = code

                else:
                    new_farray[mask] = self.__analyze_recursive(
                        new_farray[mask],
                        mmfarray,
                        mmy,
                        minmin,
                        plus0half,
                        plus1half,
                        j,
                        level,
                        code,
                    )

            if plus1half < fmax:
                plus2half = fmean + f2half
                # if discrete:
                #     plus2half = math.ceil(plus2half)

                if plus2half > fmax or plus2half == plus1half:
                    plus2half = fmax

                code = prevcode + Card.C7
                mask = (farray > plus1half) & (farray <= plus2half)

                # Analyze branch of code C7
                if mask.any():
                    mmy = y[mask]
                    mmfarray = farray[mask]
                    fl, fc = np.unique(mmfarray, return_counts=True)

                    self.encodings[j][code] = (
                        plus1half,
                        fl[np.argmax(fc)].item(),
                        plus2half,
                    )

                    self.encoding_overlaps[j][code] = {
                        cl.item(): ct.item()
                        for cl, ct in np.nditer(np.unique(mmy, return_counts=True))
                    }

                    # Secondary stopping condition:
                    # Reached maximum granularity level
                    if level == 0:
                        new_farray[mask] = code

                    else:
                        new_farray[mask] = self.__analyze_recursive(
                            new_farray[mask],
                            mmfarray,
                            mmy,
                            minmin,
                            plus1half,
                            plus2half,
                            j,
                            level,
                            code,
                        )

                if plus2half < fmax:
                    plus3half = fmean + f3half
                    # if discrete:
                    #     plus3half = math.ceil(plus3half)

                    if plus3half > fmax or plus3half == plus2half:
                        plus3half = fmax

                    code = prevcode + Card.C8
                    mask = (farray > plus2half) & (farray <= plus3half)

                    # Analyze branch of code C8
                    if mask.any():
                        mmy = y[mask]
                        mmfarray = farray[mask]
                        fl, fc = np.unique(mmfarray, return_counts=True)

                        self.encodings[j][code] = (
                            plus2half,
                            fl[np.argmax(fc)].item(),
                            plus3half,
                        )

                        self.encoding_overlaps[j][code] = {
                            cl.item(): ct.item()
                            for cl, ct in np.nditer(np.unique(mmy, return_counts=True))
                        }

                        # Secondary stopping condition:
                        # Reached maximum granularity level
                        if level == 0:
                            new_farray[mask] = code

                        else:
                            new_farray[mask] = self.__analyze_recursive(
                                new_farray[mask],
                                mmfarray,
                                mmy,
                                minmin,
                                plus2half,
                                plus3half,
                                j,
                                level,
                                code,
                            )

                    if plus3half < fmax:

                        code = prevcode + Card.C9
                        mask = (farray > plus3half) & (farray <= fmax)

                        # Analyze branch of code C9
                        if mask.any():
                            mmy = y[mask]
                            mmfarray = farray[mask]
                            fl, fc = np.unique(mmfarray, return_counts=True)

                            self.encodings[j][code] = (
                                plus3half,
                                fl[np.argmax(fc)].item(),
                                fmax,
                            )

                            self.encoding_overlaps[j][code] = {
                                cl.item(): ct.item()
                                for cl, ct in np.nditer(
                                    np.unique(mmy, return_counts=True)
                                )
                            }

                            # Secondary stopping condition:
                            # Reached maximum granularity level
                            if level == 0:
                                new_farray[mask] = code

                            else:
                                new_farray[mask] = self.__analyze_recursive(
                                    new_farray[mask],
                                    mmfarray,
                                    mmy,
                                    minmin,
                                    plus3half,
                                    fmax,
                                    j,
                                    level,
                                    code,
                                )

        return new_farray

    def __convert_feature(self, farray, j):
        # Prepare recursive function
        new_farray = np.zeros(farray.shape, dtype=self.dtype)
        use_mask = False

        # self.features[j][-2] is used instead of trying Card.C1, C2, C3, C4, C5
        mask_lower = farray < self.encodings[j][self.features[j][-2]][0]
        # self.features[j][-1] is used instead of trying Card.C9, C8, C7, C6, C5
        mask_higher = farray > self.encodings[j][self.features[j][-1]][-1]

        # Adjust values lower than known feature minimum
        if mask_lower.any():
            new_farray[mask_lower] = self.code_min
            use_mask = True

        # Adjust values higher than known feature maximum
        if mask_higher.any():
            new_farray[mask_higher] = self.code_max
            use_mask = True

        # print()
        # print(new_farray)

        if use_mask:
            # print()
            # print(mask_lower)
            # print(mask_higher)
            mask_final = ~(mask_lower | mask_higher)
            # print(mask_final)
            # print()
            new_farray[mask_final] = self.__convert_recursive(
                new_farray[mask_final],
                farray[mask_final],
                self.encodings[j],
                self.granularity,
                0,
            )
            # print()
            # print(new_farray)
            # print()
            return new_farray

        else:
            return self.__convert_recursive(
                new_farray,
                farray,
                self.encodings[j],
                self.granularity,
                0,
            )

    def __convert_recursive(
        self,
        new_farray,
        farray,
        jdict,
        level,
        prevcode,
    ):
        # Prepare current granularity level
        # totalmask = np.ones(farray.shape[0], dtype=bool)
        level -= 1
        prevcode *= 10
        # tempcode = prevcode * 10

        for c in Card.CLISTHL:
            code = prevcode + c
            # code = tempcode + c

            if code in jdict:
                mask = (farray >= jdict[code][0]) & (farray <= jdict[code][-1])

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
                            jdict,
                            level,
                            code,
                        )

        # # Secondary stopping condition:
        # # Current subarray cannot be split any further
        # if totalmask.any():
        #     new_farray[totalmask] = prevcode

        return new_farray
