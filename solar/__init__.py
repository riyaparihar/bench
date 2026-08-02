# !!! PLACEHOLDER -- REPLACE WITH YOUR REAL solar/__init__.py !!!
#
# You did not upload this file, so I've reconstructed only the ONE line we
# know it must contain (referenced by gpu_kernels.py's own docstring): the
# NUMBA_FORCE_CUDA_CC workaround for the RTX 5060 (Blackwell/sm_120), set
# BEFORE gpu_kernels.py's `from numba import cuda` import runs.
#
# Your real __init__.py may do more than this (other package-level setup,
# imports, etc.) -- copy your actual file over this one before running
# anything. This stub is only enough to make the import chain not crash.
#
# Reminder: os.environ.setdefault() only sets the value if it's not already
# set -- so exporting NUMBA_FORCE_CUDA_CC=6.1 yourself (for the GTX 1070)
# before running Python still wins over whatever this file sets.

import os

# os.environ.setdefault('NUMBA_FORCE_CUDA_CC', '8.7')
