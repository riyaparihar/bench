import os
import tempfile
from typing import Any, Dict, List, TypedDict, Union

import rasterio
import requests
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS


class Tiff_Meta_Data(TypedDict, total=False):
    driver: str
    dtype: str
    width: int
    height: int
    count: int
    crs: CRS
    transform: Affine
    bounds: BoundingBox
    dtypes: List[str]
    nodata: Any
    # Add any other fields that might be present in dataset.meta

class Tiff_Data_Result(TypedDict):
    data: Union[List[List[float]],List[float]]
    metadata: Tiff_Meta_Data


def download_file(url, temp_dir):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    filename = os.path.join(temp_dir, os.path.basename(url))
    with open(filename, 'wb') as file:
        file.write(response.content)
    return filename

def read_tiff_file(url: str) -> Tiff_Data_Result:
    print('Reading tiff file', flush=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        with rasterio.open(download_file(url, temp_dir)) as dataset:
          count = dataset.meta['count']
          
          if count == 1:
              data = dataset.read(1).tolist()  # If there's only one band, read it into a list
          else:
              data = [dataset.read(i+1).tolist() for i in range(count)]  # Read each band into a list

          # profile is a superset of meta — includes compress, predictor, tiling, etc.
          meta = dataset.profile.copy()

          additional_meta = {
              'crs': dataset.crs,
              'bounds': dataset.bounds,
              'width': dataset.width,
              'height': dataset.height,
              'transform': dataset.transform,
              'dtype': dataset.dtypes[0],
              'overviews': dataset.overviews(1),
              'nodata': dataset.nodata
          }

          meta.update(additional_meta)

          return {'data': data, 'metadata': meta}
