"""spata.model.model_card"""

import json
import numpy as np
from spata.base.card import Card


class ModelCard(Card):
    """ModelCard"""

    def __init__(
        self,
        X,
        model=None,
        cnames=None,
        classification=None,
        **kwargs,
    ):
        raise NotImplementedError("Model Cards are under development")
