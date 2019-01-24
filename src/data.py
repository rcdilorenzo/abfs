import numpy as np
import shapely.wkt
import matplotlib.pyplot as plt
from math import ceil
from PIL import Image
from skimage import color
from funcy import iffy, constantly, tap
from toolz import memoize, curry, compose, pipe
from toolz.curried import map, juxt, mapcat, concatv
from toolz.sandbox.core import unzip
from osgeo import ogr, gdal, osr
from spacenetutilities import geoTools

from src.path import *
from src.constants import *
from src.group_data_split import GroupDataSplit, DEFAULT_SPLIT_CONFIG
from src.conversions import area_in_square_feet
from src.segmentation_augmentation import SegmentationAugmentation, MOVE_SCALE_ROTATE

list_unzip = compose(map(list), unzip)
list_concatv = compose(list, concatv)

BLACK = 0
BINARY_WHITE = 1
ALWAYS_TRUE = lambda df: df.index != -1

class Data():
    def __init__(self, config,
                 split_config=DEFAULT_SPLIT_CONFIG,
                 seg_aug_config=MOVE_SCALE_ROTATE,
                 batch_size=16,
                 override_df=None,
                 aug_random_seed=None,
                 augment=False):

        self.config = config
        self.split_config = split_config
        self._df = override_df
        self._image_ids = []
        self._data_filter = ALWAYS_TRUE
        self._split_data = None

        # Require batch size to be divisible by two if augmentation enabled
        assert augment == False or batch_size % 2 == 0

        self.batch_size = batch_size
        self.augment = augment
        self.augmentation = SegmentationAugmentation(seg_aug_config,
                                                     seed=aug_random_seed)

    # ============
    # General
    # ============

    @property
    def image_ids(self):
        if self._image_ids == []:
            self._image_ids = self.df.ImageId.unique()

        return self._image_ids

    def grouped_df(self):
        return self.df.groupby('ImageId')

    @property
    def df(self):
        if self._df is None:
            region_upper = self.config.region.upper()

            self._df = geoTools.readwktcsv(file_path(
                self.config, SUMMARY,
                f'{region_upper}_polygons_solution_{self.config.band}.csv'))

            self._df['sq_ft'] = self._df.geometry.apply(area_in_square_feet)

        return self._df[self.data_filter(self._df)]

    # =============================
    # Neural Network Input/Output
    # =============================

    def to_nn(self, shape):
        """Convert data to neural network inputs/outputs

        NOTE: Image pixels have an output range of 0-255. They should be
        fractionalized before being sent to a neural network.
        """

        return pipe(
            self.df.ImageId.unique(),
            map(self._to_single_nn(shape)),
            list_unzip,
            iffy(constantly(self.augment), self._augment_nn),
            map(np.array)
        )

    # ====================
    # Train/Val/Test Data
    # ====================

    def train_data(self, batch_id):
        df = self.split_data.train_df()
        return self._batch_data(df, self.augment, batch_id)

    def train_batch_count(self):
        return self._batch_count(self.split_data.train_df())

    def val_data(self, batch_id):
        df = self.split_data.val_df
        return self._batch_data(df, False, batch_id)

    def val_batch_count(self):
        return self._batch_count(self.split_data.val_df)

    def test_data(self, batch_id):
        df = self.split_data.test_df
        return self._batch_data(df, False, batch_id)

    def test_batch_count(self):
        return self._batch_count(self.split_data.test_df)

    @property
    def split_data(self):
        if self._split_data is None:
            self._split_data = GroupDataSplit(
                self.df, 'ImageId', self.split_config
            )

        return self._split_data

    # ==================
    # Data Filters
    # ==================

    @property
    def data_filter(self):
        return self._data_filter

    @data_filter.setter
    def data_filter(self, data_filter):
        self._data_filter = data_filter
        self._split_data = None

    def reset_filter(self):
        self.data_filter = ALWAYS_TRUE

    # ==================
    # Images / Masks
    # ==================

    def image_for(self, image_id):
        return plt.imread(image_id_to_path(self.config, image_id))

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

    # ==================
    # Private Methods
    # ==================

    def _batch_data(self, df, augment, batch_id):
        batch_size = int(self.batch_size / (int(augment) + 1))

        start_index = batch_id * batch_size
        end_index = start_index + batch_size
        image_ids = df.ImageId.unique()[start_index:end_index]

        return Data(self.config,
                    override_df=df[df.ImageId.isin(image_ids)],
                    augment=augment)

    def _batch_count(self, df):
        return ceil(df.ImageId.nunique() / self.batch_size)

    @curry
    def _to_single_nn(self, shape, image_id):
        return pipe(
            image_id,
            juxt(self.image_for, self.mask_for),
            map(self._resize_image(shape)),
            list
        )

    def _augment_nn(self, inputs_and_outputs):
        images, masks = inputs_and_outputs
        aug_images, aug_masks = self.augmentation.run(images, masks)

        return list_concatv(images, aug_images), list_concatv(masks, aug_masks)

    @curry
    def _resize_image(self, shape, image):
        return np.array(Image.fromarray(image).resize(shape))
