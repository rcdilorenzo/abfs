import cv2
import mercantile
from shapely.geometry import Polygon, MultiPolygon, mapping
from toolz import curry, pipe
from toolz.curried import *
import numpy as np

from abfs.api.prediction.image import ImagePrediction

BASE_URL = 'https://api.mapbox.com/v4/mapbox.satellite'

class LatLongPrediction():
    def __init__(self, keras_model, latitude, longitude, zoom=17,
                 tolerance=0.3, api_key=None, image_path_only=False):
        self.keras_model = keras_model
        self.tolerance = tolerance
        self.tile = mercantile.tile(longitude, latitude, zoom)
        self.image_path_only = image_path_only

        t = self.tile
        self.url = f'{BASE_URL}/{t.z}/{t.x}/{t.y}.png?access_token={api_key}'

    def run(self):
        if self.image_path_only:
            return ImagePrediction(self.url,
                                   self.keras_model,
                                   temp_file=True).run()

        prediction_image = ImagePrediction(self.url, self.keras_model).run()
        multi_polygon = self._multi_polygon(prediction_image, self.tile)
        return mapping(multi_polygon), prediction_image

    def _multi_polygon(self, prediction_image, tile):
        return pipe(
            self._find_contours(prediction_image),
            map(self._contour_to_lat_long(prediction_image.shape[0:2], tile)),
            filter(lambda p: p is not None),
            list,
            MultiPolygon
        )

    def _find_contours(self, prediction_image):
        _, threshold = cv2.threshold(prediction_image, 1 - self.tolerance, 1,
                                     cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(np.array(threshold, np.uint8),
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @curry
    def _contour_to_lat_long(self, image_shape, tile, contour):
        def xy_to_lng_lat(xy):
            lat_long = mercantile.lnglat(xy[0], xy[1])
            return (lat_long.lng, lat_long.lat)

        xy_bounds = mercantile.xy_bounds(tile)
        height, width = image_shape
        width_scale = (xy_bounds.right - xy_bounds.left) / width
        height_scale = (xy_bounds.bottom - xy_bounds.top) / height
        xy_points = (contour[:, :] *
                     (width_scale, height_scale) +
                     (xy_bounds.left, xy_bounds.top))

        if xy_points.shape[0] < 3:
            return None

        return Polygon(list(map(xy_to_lng_lat, xy_points[:, 0])))

