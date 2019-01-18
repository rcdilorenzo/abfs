import os
from constants import *

def file_path(config, subfolder, filename):
    return os.path.join(config.path, config.region,
                        BUILDING_LABELS_PATH, subfolder, filename)

def image_id_to_path(config, image_id):
    return file_path(config, config.band, f'{config.band}_{image_id}.tif')

def image_id_to_geo_json_path(config, image_id):
    return file_path(config, VECTOR, f'Geo_{image_id}.geojson')
