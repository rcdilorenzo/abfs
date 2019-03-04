from waitress import serve
from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.view import view_config
from pyramid.response import Response
from wsgiref.simple_server import make_server
import json

from abfs.api.keras_model import KerasModel
from abfs.api.mask_prediction import MaskPrediction
from abfs.api.polygon_prediction import PolygonPrediction

WEIGHTS_PATH = None
MODEL_PATH = None
MAPBOX_API_KEY = None
DEFAULT_TOLERANCE = None

@view_config(route_name='predict-mask')
def predict_mask(request):
    return MaskPrediction(
        request,
        KerasModel(MODEL_PATH, WEIGHTS_PATH),
        MAPBOX_API_KEY
    ).respond()

@view_config(route_name='predict-polygon')
def predict_polygon(request):
    return PolygonPrediction(
        request,
        KerasModel(MODEL_PATH, WEIGHTS_PATH),
        MAPBOX_API_KEY,
        DEFAULT_TOLERANCE
    ).respond()

def main(global_config, **settings):
    config = Configurator(settings=settings)
    config.add_route('predict-mask', '/predict/mask', request_method='POST')
    config.add_route('predict-polygon', '/predict/polygon', request_method='POST')
    config.scan('.')
    return config.make_wsgi_app()

def serve(address='0.0.0.0', port=1337, model_path=None,
          weights_path=None, mapbox_api_key=None, default_tolerance='0.5'):

    global MODEL_PATH, WEIGHTS_PATH, MAPBOX_API_KEY, DEFAULT_TOLERANCE
    MODEL_PATH = model_path
    WEIGHTS_PATH = weights_path
    MAPBOX_API_KEY = mapbox_api_key
    DEFAULT_TOLERANCE = default_tolerance

    server = make_server(address, port, main(None))
    print(f'Serving on {address}:{port}')
    server.serve_forever()
