import os
from collections import namedtuple as Struct

DataConfig = Struct('DataConfig', ['path', 'band', 'region'])

DEFAULT_DATA_DIR =  os.path.expanduser('~/workspaces/data/spacenet/')

RIO_REGION = 'AOI_1_Rio';

BUILDING_LABELS_PATH = 'processedData/processedBuildingLabels'

BAND3 = '3band'
BAND8 = '8band'
VECTOR = 'vectordata/geojson'
SUMMARY = 'vectordata/summarydata'
