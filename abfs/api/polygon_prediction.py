from pyramid.response import Response
from abfs.api.prediction.lat_long import LatLongPrediction
from abfs.api.params import Params
from toolz import juxt, partial, curry, compose
from lenses import lens
import json

class PolygonPrediction():
    def __init__(self, request, keras_model, api_key, default_tolerance):
        self.request = request
        self.keras_model = keras_model
        self.json_body = json.loads(request.body)
        self.api_key = api_key
        self.defaults = { 'tolerance': default_tolerance }

    def respond(self):
        lat, lng, tolerance, zoom = Params.coordinates({
            **self.defaults,
            **self.json_body
        })
        geo_json, _ = LatLongPrediction(self.keras_model, lat, lng,
                                        zoom, tolerance,
                                        api_key=self.api_key).run()

        return Response(json_body=geo_json)
