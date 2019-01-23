import os
import pytest
import pandas as pd
from src.data import Data
from src.constants import DataConfig, BAND3, RIO_REGION

TOP_LEVEL_DATA_DIR = os.path.join(os.path.dirname(__file__),
                                  './fixtures/spacenet')

@pytest.fixture
def config():
    return DataConfig(TOP_LEVEL_DATA_DIR, BAND3, RIO_REGION)

@pytest.fixture
def data():
    return Data(config())

@pytest.fixture
def sample_df():
    return pd.DataFrame(data={
        'group_id': [ 1,  1,  1,  2,  2,  3,  3,  4,  5,  5],
        'value':    [56, 18, 19, 51, 15, 96, 99, 95, 66, 41]
    })
