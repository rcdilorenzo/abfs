from pyramid.response import FileResponse
from abfs.api.prediction.image import ImagePrediction
from abfs.api.prediction.lat_long import LatLongPrediction
from abfs.api.params import Params
import json

class MaskPrediction():
    def __init__(self, request, keras_model, api_key):
        self.request = request
        self.keras_model = keras_model
        self.json_body = json.loads(request.body)
        self.api_key = api_key

    def respond(self):
        prediction = None
        if 'coordinate' in self.json_body:
            lat, lng, _, zoom = Params.coordinates(self.json_body)
            prediction = LatLongPrediction(self.keras_model, lat, lng, zoom,
                                           image_path_only=True,
                                           api_key=self.api_key)
        elif 'image_url' in self.json_body:
            url = Params.image_url(self.json_body)
            prediction = ImagePrediction(url, self.keras_model, temp_file=True)

        return FileResponse(prediction.run(), self.request)
