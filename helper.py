import os
import rasterio
import numpy as np

class Helper: 
    def __init__(self):
        pass

    def load_raster(filename) -> np.ndarray:
        """
        load_raster loads and normalizes raster data
        :filename: .tif data filename in current folder
        :return: normalized raster data as np.ndarray
        """
        # Get current working directory
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        # Load test file
        raster= rasterio.open(os.path.join(__location__, filename))

        # Extract test channels
        red = raster.read(1).flatten()
        green = raster.read(2).flatten()
        blue = raster.read(3).flatten()

        # Join and normalize data
        data = np.array(list(zip(red, green, blue)))
        data_min, data_max = data.min(), data.max()
        data_normalized = (data-data_min)/(data_max-data_min)
        return data_normalized