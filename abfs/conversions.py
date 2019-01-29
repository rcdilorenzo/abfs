import pyproj
from shapely.ops import transform
from toolz import partial

DEGREES_PROJECTION = pyproj.Proj(init='epsg:4326')
FEET_PROJECTION = pyproj.Proj(init='epsg:3735')

projection = partial(pyproj.transform, DEGREES_PROJECTION, FEET_PROJECTION)


def area_in_square_feet(shape):
    return transform(projection, shape).area
