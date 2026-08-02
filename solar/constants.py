# !!! PLACEHOLDER -- REPLACE WITH YOUR REAL solar/constants.py !!!
#
# You did not upload this file either. house_bbox_accumulation.py imports:
#   from constants import HOUSE_BBOX_EXPAND_PCT, FACET_BUFFER_PX
#
# NOTE the import above is `from constants import ...` (absolute, no dot) --
# NOT `from .constants import ...` (relative). That means in YOUR real repo,
# constants.py might actually live at the top level (sibling to solar/ and
# energy/), the same way pvlib_core.py lives in energy/ and is imported
# absolutely as `from energy.pvlib_core import ...`.
#
# Check your real repo's layout before running:
#   - If constants.py is top-level (bench/constants.py) -- move this file
#     there instead of inside solar/, and make sure bench/ is on the path
#     (it will be, since you run everything from bench/).
#   - If it's actually inside solar/ despite the absolute-style import --
#     that only works if solar/ itself is added directly to sys.path
#     somewhere (e.g. in solar/__init__.py) rather than relying on package
#     resolution. Check your real __init__.py for that.
#
# Values below are placeholders -- copy your real constants over these.
import numpy as np


HOUSE_BBOX_EXPAND_PCT = 0.20   # PLACEHOLDER -- confirm real value
FACET_BUFFER_PX = 3.0
NODATA_SENTINEL_FOR_IRRADIANCE = np.float32(-9999.0)       # PLACEHOLDER -- confirm real value
