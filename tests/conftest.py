import os
import pytest
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from pytest import fail
from scipy.misc import imsave, imread

from abfs.data import Data
from abfs.constants import DataConfig, BAND3, RIO_REGION

TOP_LEVEL_DATA_DIR = os.path.join(os.path.dirname(__file__),
                                  './fixtures/spacenet')

@pytest.fixture
def config():
    return DataConfig(TOP_LEVEL_DATA_DIR, BAND3, RIO_REGION)

@pytest.fixture
def data(config):
    return Data(config)

@pytest.fixture
def sample_df():
    return pd.DataFrame(data={
        'group_id': [ 1,  1,  1,  2,  2,  3,  3,  4,  5,  5],
        'value':    [56, 18, 19, 51, 15, 96, 99, 95, 66, 41]
    })

@pytest.fixture
def sample_image():
    return plt.imread(os.path.join(
        TOP_LEVEL_DATA_DIR,
        'AOI_1_Rio/processedData/processedBuildingLabels/3band',
        '3band_AOI_1_RIO_img5792.tif'
    ))

@pytest.fixture
def box_mask():
    def _box_mask(shape):
        box_mask_height, box_mask_width = shape
        x1 = int(box_mask_width / 4)
        y1 = int(box_mask_height / 4)
        x2 = x1 + int(box_mask_width / 2)
        y2 = y1 + int(box_mask_height / 2)

        box_mask = np.zeros(shape, dtype='uint8')
        box_mask[y1:y2, x1:x2] = 1
        return box_mask

    return _box_mask

def compare_image(received, filename):
    approved_path = f'tests/fixtures/{filename}.approved.png'
    received_path = f'tests/fixtures/{filename}.received.png'
    imsave(received_path, received)

    if os.path.isfile(approved_path) is False:
        fail(f'No image explicitly approved. Please verify {received_path}.')
        return

    approved_image = imread(approved_path)
    received_image = imread(received_path)

    if (approved_image != received_image).any():
        fail(f'{received_path} != {approved_path}. Please verify.')
    else:
        os.remove(received_path)
