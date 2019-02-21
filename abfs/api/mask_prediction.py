from pyramid.response import FileResponse
from abfs.api.prediction.image import ImagePrediction
from abfs.api.prediction.lat_long import LatLongPrediction
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
            params = self.json_body['coordinate']
            lat, lng = float(params['latitude']), float(params['longitude'])
            zoom = int(self.json_body.get('zoom', '16'))
            prediction = LatLongPrediction(self.keras_model, lat, lng, zoom,
                                           image_path_only=True,
                                           api_key=self.api_key)
        elif 'image_url' in self.json_body:
            prediction = ImagePrediction(self.json_body['image_url'],
                                        self.keras_model,
                                        temp_file=True)

        return FileResponse(prediction.run(), self.request)
