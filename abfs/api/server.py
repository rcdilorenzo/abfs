from waitress import serve
from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.view import view_config
from pyramid.response import Response
from wsgiref.simple_server import make_server

@view_config(route_name='predict-mask')
def predict_mask(request):
    print(request)
    return Response(status=200)

def main(global_config, **settings):
    config = Configurator(settings=settings)
    config.add_route('predict-mask', '/predict/mask', request_method='GET')
    config.scan('.')
    return config.make_wsgi_app()

def serve(address='0.0.0.0', port=1337):
    server = make_server(address, port, main(None))
    print(f'Serving on {address}:{port}')
    server.serve_forever()
