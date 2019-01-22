import os
import pytest
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
