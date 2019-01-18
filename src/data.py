import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../vendor'))

import numpy as np
import shapely.wkt
import matplotlib.pyplot as plt
from skimage import color
from toolz import memoize
from spacenetutilities import geoTools
from spacenet_lib.create_poly_mask import create_poly_mask

from path import *
from constants import *

class Data():
    def __init__(self, config):
        self.config = config
        self._df = None
        self._image_ids = []

    @property
    @memoize
    def df(self):
        if self._df is None:
            region_upper = self.config.region.upper()

            self._df = geoTools.readwktcsv(file_path(
                self.config, SUMMARY,
                f'{region_upper}_polygons_solution_{self.config.band}.csv'))

        return self._df

    @property
    def image_ids(self):
        if self._image_ids == []:
            self._image_ids = self.df.ImageId.unique()

        return self._image_ids

    def image_for(self, image_id):
        return plt.imread(image_id_to_path(self.config, image_id))

    def mask_for(self, image_id):
        return self.build_truth_mask(image_id, output_dir=False)

    def mask_overlay_for(self, image_id):
        single_mask = self.mask_for(image_id) / 255
        blank = np.zeros(single_mask.shape)

        colored_mask = np.dstack([blank, single_mask, blank])
        color_mask_hsv = color.rgb2hsv(colored_mask)

        image_hsv = color.rgb2hsv(self.image_for(image_id))
        image_hsv[..., 0] = color_mask_hsv[..., 0]
        image_hsv[..., 1] = color_mask_hsv[..., 1] * 0.8
        return color.hsv2rgb(image_hsv)

    def build_truth_mask(self, image_id, output_dir=True):
        if output_dir:
            output_filename = os.path.join(self.config.output_dir,
                                           f'{image_id}.tif')
        else:
            output_filename = '/tmp/out.tif'

        return create_poly_mask(
            image_id_to_path(self.config, image_id),
            image_id_to_geo_json_path(self.config, image_id),
            npDistFileName=output_filename,
            noDataValue=0, burn_values=255)



