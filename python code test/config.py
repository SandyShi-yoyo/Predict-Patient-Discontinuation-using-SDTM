###############################################################################
# config.py
# Purpose: Single source of truth for the random seed and reproducibility
#          settings shared by load_data.py, feature_engineering.py, and
#          modeling.py.
#
# Python version used to build/validate this pipeline: 3.12.3
# Package versions: see requirements.txt (pin these exactly for byte-for-byte
# reproducibility across machines).
###############################################################################
import os
import random

import numpy as np

# Single global seed reused everywhere (numpy, random, and every
# random_state= argument passed to pandas/sklearn objects downstream).
RANDOM_SEED = 123


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Fix every source of randomness we control at the process level.

    Call this once at the top of every script before any random operation
    (sampling, model fitting, etc.) so results are identical run-to-run and
    machine-to-machine given the same environment (see requirements.txt).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# Seed immediately on import so any script that does `from config import *`
# or `import config` gets a deterministic process state before doing anything
# else.
set_global_seed(RANDOM_SEED)
