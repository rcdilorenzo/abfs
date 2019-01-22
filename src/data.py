import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../vendor'))

import numpy as np
import shapely.wkt
import matplotlib.pyplot as plt
from skimage import color
from toolz import memoize
from osgeo import ogr, gdal, osr
from spacenetutilities import geoTools

from src.path import *
from src.constants import *
from src.conversions import area_in_square_feet

BLACK = 0
BINARY_WHITE = 1
ALWAYS_TRUE = lambda df: df.index != -1

class Data():
    def __init__(self, config):
        self.config = config
        self._df = None
        self._image_ids = []
        self.data_filter = ALWAYS_TRUE

    @property
    def df(self):
        if self._df is None:
            region_upper = self.config.region.upper()

            self._df = geoTools.readwktcsv(file_path(
                self.config, SUMMARY,
                f'{region_upper}_polygons_solution_{self.config.band}.csv'))

            self._df['sq_ft'] = self._df.geometry.apply(area_in_square_feet)

        return self._df[self.data_filter(self._df)]

    @property
    def image_ids(self):
        if self._image_ids == []:
            self._image_ids = self.df.ImageId.unique()

        return self._image_ids

    def reset_filter(self):
        self.data_filter = ALWAYS_TRUE

    def grouped_df(self):
        return self.df.groupby('ImageId')

    def image_for(self, image_id):
        return plt.imread(image_id_to_path(self.config, image_id))

    def mask_for(self, image_id):
        # Note: If certain parts of this method are extracted, a segmentation
        # fault occurs. See https://trac.osgeo.org/gdal/ticket/1936 and
        # https://trac.osgeo.org/gdal/wiki/PythonGotchas for why.

        # Get all rows with the given image id
        rows = self.grouped_df().get_group(image_id)

        # Determine output mask width and height
        srcRas_ds = gdal.Open(image_id_to_path(self.config, image_id))
        x_size = srcRas_ds.RasterXSize
        y_size = srcRas_ds.RasterYSize
        transform = srcRas_ds.GetGeoTransform()
        projection = srcRas_ds.GetProjection()

        # Create polygon layer
        polygon_ds = ogr.GetDriverByName('Memory').CreateDataSource('polygon')
        polygon_layer = polygon_ds.CreateLayer('poly', srs=None)

        # Create feature with all polygons
        feat = ogr.Feature(polygon_layer.GetLayerDefn())

        # Add all row polygons to multi-polygon
        multi_polygon = ogr.Geometry(ogr.wkbMultiPolygon)
        for _, row in rows.iterrows():
            geometry = ogr.CreateGeometryFromWkt(row.PolygonWKT_Geo)
            multi_polygon.AddGeometry(geometry)

        # Set multi-polygon geometry back on feature
        feat.SetGeometry(multi_polygon)

        # Set feature on polygon layer
        polygon_layer.SetFeature(feat)

        # Create raster layer of image size
        destination_layer = (gdal
                             .GetDriverByName('MEM')
                             .Create('', x_size, y_size, 1, gdal.GDT_Byte))

        # Match image transform and projection so that lat/long polygon
        # coordinates map to the proper location
        destination_layer.SetGeoTransform(transform)
        destination_layer.SetProjection(projection)

        # Set empty value of output mask to be black
        band = destination_layer.GetRasterBand(1)
        band.SetNoDataValue(BLACK)

        # Rasterize image with white polygon areas
        gdal.RasterizeLayer(destination_layer, [1], polygon_layer,
                            burn_values=[BINARY_WHITE])

        # Return image mask result as np.array
        return np.array(destination_layer.ReadAsArray())

    def green_mask_for(self, image_id):
        mask = self.mask_for(image_id)
        blank = np.zeros(mask.shape)

        return np.dstack([blank, mask, blank])

    def mask_overlay_for(self, image_id):
        color_mask_hsv = color.rgb2hsv(self.green_mask_for(image_id))

        image_hsv = color.rgb2hsv(self.image_for(image_id))
        image_hsv[..., 0] = color_mask_hsv[..., 0]
        image_hsv[..., 1] = color_mask_hsv[..., 1] * 0.8

        return color.hsv2rgb(image_hsv)

