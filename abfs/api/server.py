from waitress import serve
from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.view import view_config
from pyramid.response import Response
from wsgiref.simple_server import make_server
import json

from abfs.api.keras_model import KerasModel
from abfs.api.mask_prediction import MaskPrediction

WEIGHTS_PATH = None
MODEL_PATH = None

@view_config(route_name='predict-mask')
def predict_mask(request):
    return MaskPrediction(
        request,
        KerasModel(MODEL_PATH, WEIGHTS_PATH)
    ).respond()

def main(global_config, **settings):
    config = Configurator(settings=settings)
    config.add_route('predict-mask', '/predict/mask', request_method='POST')
    config.scan('.')
    return config.make_wsgi_app()

def serve(address='0.0.0.0', port=1337, model_path=None, weights_path=None):
    global MODEL_PATH, WEIGHTS_PATH
    MODEL_PATH = model_path
    WEIGHTS_PATH = weights_path

    server = make_server(address, port, main(None))
    print(f'Serving on {address}:{port}')
    server.serve_forever()
